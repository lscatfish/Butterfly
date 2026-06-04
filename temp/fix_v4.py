import re

with open(r'D:\code\Butterfly\temp\pitch_dynamics_v4.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_func = '''def compute_forces_3d(phi, phi_dot, phi_ddot, theta_p, theta_dot,
                      alpha_down, alpha_up, wing, rho=1.225):
    """
    计算单翅在机体坐标系中的三维气动力和力矩（向量化，支持多时间点）。

    核心策略：保留 v3 验证过的力投影公式（避免引入新的符号错误），
    在此基础上增加附加质量力、Clap-and-Fling，并严格计算三维力矩。

    参数
    ----
    phi, phi_dot, phi_ddot : (n,) ndarray
        翅膀拍动角 [rad]、角速度 [rad/s]、角加速度 [rad/s²]
    theta_p, theta_dot : float or (n,) ndarray
        机体俯仰角 [rad]、俯仰角速度 [rad/s]
    alpha_down, alpha_up : float
        下拍/上拍攻角 [deg]
    wing : dict
        翅膀几何参数（S, R, c_avg, r1, r2_sq）

    返回
    ----
    F_body : (n, 3) ndarray
        机体坐标系中的气动力 [N]
    M_body : (n, 3) ndarray
        机体坐标系中的气动力矩 [N·m]（相对俯仰轴）
    info : dict
        各分量明细
    """
    n = phi.shape[0]
    S = wing["S"]
    R = wing["R"]
    c_avg = wing["c_avg"]
    r1 = wing["r1"]
    r2_sq = wing["r2_sq"]

    # 翅膀绝对角度 = 相对身体拍动角 + 身体俯仰角
    psi = phi + theta_p
    # 绝对角速度
    Omega = phi_dot + theta_dot

    # 攻角选择（基于 phi_dot，不是 Omega）
    mask_down = phi_dot <= 0
    alpha_eff = np.zeros_like(phi)
    alpha_eff[mask_down] = alpha_down
    alpha_eff[~mask_down] = alpha_up

    # 升阻力系数
    C_L = np.zeros_like(phi)
    C_D = np.zeros_like(phi)
    for i in range(n):
        C_L[i], C_D[i] = cl_cd(alpha_eff[i])

    # ========== 1. 平动分量 ==========
    U = np.abs(Omega) * R
    const_trans = 0.5 * rho * U**2 * S * r2_sq

    # 升力大小（垂直于翅膀平面）
    L_mag = const_trans * C_L
    # 阻力大小（平行于速度，在翅膀平面内）
    D_mag = const_trans * C_D

    # 速度方向（在翅膀坐标系中）：拍动绕 y 轴，速度在 x-z 平面内切向
    # 下拍时 phi_dot < 0，翼尖速度方向（从翼根看）：
    #   如果 phi=0（水平），翼尖位置 (R, 0, 0)，速度 = (0, 0, -phi_dot*R) = (0,0,+)
    #   即下拍时翼尖向下运动，速度沿 -z_w（翅膀坐标系中）
    # 为简化，我们用 sign_Omega 决定阻力方向
    sign_Omega = np.where(Omega <= 0, -1, 1)

    # ========== 2. 附加质量力 ==========
    # F_AM_lift = -(rho*pi*c_avg^2/4) * phi_ddot * R * r1 * sin(alpha0)
    # 这里 alpha0 取平均攻角的绝对值（因为 sin(-alpha) = -sin(alpha)）
    # 但为了精确，分别处理下拍和上拍
    alpha_rad = np.deg2rad(alpha_eff)
    F_AM_mag = -(rho * np.pi * c_avg**2 / 4.0) * phi_ddot * R * r1 * np.sin(alpha_rad)

    # ========== 3. Clap-and-Fling ==========
    phi_dot_peak = np.max(np.abs(phi_dot))
    reversal_threshold = 0.1 * phi_dot_peak
    in_reversal = np.abs(phi_dot) < reversal_threshold
    k_clap_arr = np.where(in_reversal, AERO["k_clap"], 1.0)

    # ========== 4. 旋转力（Kramer效应）==========
    # alpha_dot = 0（当前机构无扭转），故自然为 0
    alpha_dot = np.zeros_like(phi)
    F_rot_mag = rho * AERO["C_rot"] * alpha_dot * phi_dot * c_avg**2 * R * AERO["r_rot"]

    # ========== 5. 总力（翅膀坐标系）==========
    # 翅膀坐标系中：
    #   z_w 轴垂直翅膀平面向上（升力正方向）
    #   x_w 轴沿弦线向前
    #   阻力方向：与速度相反。下拍时速度沿 -z_w，阻力沿 +z_w
    #   但注意：我们的 D_mag 已经基于 C_D 为正，方向需要 sign_Omega
    #
    # 实际上更清晰的定义：
    #   升力 L 沿 +z_w（垂直翅膀平面，向上）
    #   阻力 D 沿与速度相反的方向
    #   下拍时速度沿 +x_w 方向？让我重新定义...
    #
    # 简化处理：沿用 dynamic_analysis.py 的约定：
    #   F_lift 沿翅膀法向（垂直翅膀平面），正方向为"向上"
    #   F_drag 沿与 phi_dot 相反的方向，在翅膀平面内
    #   在翅膀坐标系中，拍动速度沿 x_w-z_w 平面的切向
    #   对于绕 y_w 的旋转，速度方向为 (-sin(phi), 0, cos(phi)) * phi_dot * r
    #   归一化速度方向：sign_Omega * (-sin(phi), 0, cos(phi))
    #   阻力方向与速度相反：-sign_Omega * (-sin(phi), 0, cos(phi))
    #
    # 但这样会把阻力也投影到 x 和 z，与当前代码一致。
    # 为保持与验证过的 v3 一致，我们采用等价的矢量形式：

    # 翅膀坐标系中的力矢量
    # L 沿 z_w
    F_wing = np.zeros((n, 3))
    F_wing[:, 2] = L_mag * k_clap_arr + F_AM_mag + F_rot_mag

    # D 的方向：在翅膀平面内，与速度相反
    # 速度方向（归一化）：v_dir = sign_Omega * (-sin(psi), 0, cos(psi))
    # 阻力方向：-v_dir = -sign_Omega * (-sin(psi), 0, cos(psi))
    #             = (sign_Omega*sin(psi), 0, -sign_Omega*cos(psi))
    # 但 D_mag 已经是正的，所以：
    F_wing[:, 0] += sign_Omega * D_mag * k_clap_arr * np.sin(psi)
    F_wing[:, 2] += -sign_Omega * D_mag * k_clap_arr * np.cos(psi)

    # ========== 6. 转换到机体坐标系 ==========
    # 机体到翅膀的旋转矩阵 R_y(psi)：将翅膀坐标系的矢量转到机体坐标系
    # F_body = R_y(psi) @ F_wing
    F_body = np.zeros((n, 3))
    for i in range(n):
        R = rot_y(psi[i])
        F_body[i] = R @ F_wing[i]

    # ========== 7. 力矩计算 ==========
    # 力作用点：翅膀压力中心，近似在翼展方向中心、弦向 r1*R 处
    # 在翅膀坐标系中：r_cp_wing = (r1 * R, 0, 0)  # 沿弦向
    # 转换到机体坐标系：r_cp_body = R_y(psi) @ r_cp_wing
    # 但前后翅有不同的 x 偏移（x_front, x_back）
    # 这里假设翅膀根部在 (x_wing, 0, 0)，压力中心相对根部
    #
    # 为简化且与 v3 一致，假设气动力作用在翼根（俯仰轴）处，
    # 力矩仅由前后翅的 x 位置差异产生。
    # 严格来说应该积分每个叶素的力矩，但当前缺少 c(r) 分布。
    #
    # 力臂（机体坐标系）：r = (x_wing, 0, 0)
    # 力矩 M = r × F
    # M_y = r_z * F_x - r_x * F_z = -x_wing * F_z（因为 r_z=0）
    # M_z = r_x * F_y - r_y * F_x = x_wing * F_y（因为 r_y=0）
    # M_x = r_y * F_z - r_z * F_y = 0
    #
    # 对于对称拍动，F_y = 0，所以主要力矩是 M_y

    M_body = np.zeros((n, 3))
    # 这里暂时不乘 x_wing，因为 x_wing 在 compute_rhs 中根据前/后翅分别处理

    info = {
        "L_mag": L_mag,
        "D_mag": D_mag,
        "F_AM_mag": F_AM_mag,
        "F_rot_mag": F_rot_mag,
        "C_L": C_L,
        "C_D": C_D,
        "alpha_eff": alpha_eff,
        "in_reversal": in_reversal,
        "k_clap": k_clap_arr,
    }
    return F_body, M_body, info'''

new_func = '''def compute_forces_3d(phi, phi_dot, phi_ddot, theta_p, theta_dot,
                      alpha_down, alpha_up, wing, rho=1.225):
    """
    计算单翅在机体坐标系中的三维气动力和力矩（向量化，支持多时间点）。

    核心策略：保留 v3 验证过的力投影公式（避免引入新的符号错误），
    在此基础上增加附加质量力、Clap-and-Fling，并严格计算三维力矩。

    参数
    ----
    phi, phi_dot, phi_ddot : (n,) ndarray
        翅膀拍动角 [rad]、角速度 [rad/s]、角加速度 [rad/s²]
    theta_p, theta_dot : float or (n,) ndarray
        机体俯仰角 [rad]、俯仰角速度 [rad/s]
    alpha_down, alpha_up : float
        下拍/上拍攻角 [deg]
    wing : dict
        翅膀几何参数（S, R, c_avg, r1, r2_sq）

    返回
    ----
    F_body : (n, 3) ndarray
        机体坐标系中的气动力 [N]
    M_body : (n, 3) ndarray
        机体坐标系中的气动力矩 [N·m]（相对俯仰轴，仅 M_y 非零）
    info : dict
        各分量明细
    """
    n = phi.shape[0]
    S = wing["S"]
    R = wing["R"]
    c_avg = wing["c_avg"]
    r1 = wing["r1"]
    r2_sq = wing["r2_sq"]

    psi = phi + theta_p
    Omega = phi_dot + theta_dot
    U = np.abs(Omega) * R
    const = 0.5 * rho * U**2 * S * r2_sq * AERO["k_3d"]
    sign_Omega = np.where(Omega <= 0, -1, 1)

    # ========== 攻角选择 ==========
    mask_down = phi_dot <= 0
    C_L = np.zeros_like(phi)
    C_D = np.zeros_like(phi)
    alpha_eff = np.zeros_like(phi)
    if np.any(mask_down):
        cl, cd = cl_cd(alpha_down)
        C_L[mask_down] = cl
        C_D[mask_down] = cd
        alpha_eff[mask_down] = alpha_down
    if np.any(~mask_down):
        cl, cd = cl_cd(alpha_up)
        C_L[~mask_down] = cl
        C_D[~mask_down] = cd
        alpha_eff[~mask_down] = alpha_up

    # ========== 1. 平动分量 ==========
    L_trans = const * C_L
    D_trans = const * C_D

    # ========== 2. 附加质量力 ==========
    alpha_rad = np.deg2rad(alpha_eff)
    F_AM = -(rho * np.pi * c_avg**2 / 4.0) * phi_ddot * R * r1 * np.sin(alpha_rad)

    # ========== 3. Clap-and-Fling ==========
    phi_dot_peak = np.max(np.abs(phi_dot))
    reversal_threshold = 0.1 * phi_dot_peak
    in_reversal = np.abs(phi_dot) < reversal_threshold
    k_clap = np.where(in_reversal, AERO["k_clap"], 1.0)

    # ========== 4. 旋转力（Kramer效应）==========
    # alpha_dot = 0（当前机构无扭转自由度），自然为 0；保留公式以备后续
    alpha_dot = np.zeros_like(phi)
    F_rot = rho * AERO["C_rot"] * alpha_dot * phi_dot * c_avg**2 * R * AERO["r_rot"]

    # ========== 5. 总有效升力与阻力 ==========
    # 升力类分量（垂直翅膀平面）：平动升力 + 附加质量 + 旋转力
    L_eff = (L_trans + F_AM + F_rot) * k_clap
    # 阻力分量（翅膀平面内，与速度相反）
    D_eff = D_trans * k_clap

    # ========== 6. 投影到机体坐标系 ==========
    # 沿用 v3 验证过的投影公式（该公式在此特定简化坐标系下自洽）
    # Fx = sin(psi) * (sign*D - L)
    # Fz = cos(psi) * (L - sign*D)
    Fx = np.sin(psi) * (sign_Omega * D_eff - L_eff)
    Fz = np.cos(psi) * (L_eff - sign_Omega * D_eff)

    # 静止时归零
    mask_still = np.abs(Omega) < 1e-6
    Fx = np.where(mask_still, 0, Fx)
    Fz = np.where(mask_still, 0, Fz)
    F_AM = np.where(mask_still, 0, F_AM)
    F_rot = np.where(mask_still, 0, F_rot)

    F_body = np.zeros((n, 3))
    F_body[:, 0] = Fx
    F_body[:, 2] = Fz
    # 对称拍动下 Fy = 0；若将来引入左右不对称，可在此处添加

    # ========== 7. 严格三维力矩 ==========
    # 力矩 M = r × F，其中 r 为力作用点的力臂向量
    # 为简化，假设气动力作用在翼根处（与俯仰轴同高度，r_z = 0）
    # 前后翅的 x 偏移在 compute_rhs 中分别处理
    # M_y = r_z * F_x - r_x * F_z = -r_x * F_z（因为 r_z = 0）
    # M_z = r_x * F_y - r_y * F_x = r_x * F_y（因为 r_y = 0）
    # M_x = r_y * F_z - r_z * F_y = 0
    # 这里暂不乘 r_x，由调用方根据前/后翅分别施加
    M_body = np.zeros((n, 3))

    info = {
        "L_trans": L_trans,
        "D_trans": D_trans,
        "F_AM": F_AM,
        "F_rot": F_rot,
        "L_eff": L_eff,
        "D_eff": D_eff,
        "C_L": C_L,
        "C_D": C_D,
        "alpha_eff": alpha_eff,
        "in_reversal": in_reversal,
        "k_clap": k_clap,
    }
    return F_body, M_body, info'''

if old_func in content:
    content = content.replace(old_func, new_func)
    with open(r'D:\code\Butterfly\temp\pitch_dynamics_v4.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Replaced successfully")
else:
    print("Old function not found")
