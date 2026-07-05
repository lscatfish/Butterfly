#!/usr/bin/env python3
"""
前置机构运动学模块（曲柄摇杆四连杆机构 → 翅膀拍动角）

机构参数的权威来源是 config/design_v69.yaml（通过 src.config 读取）。
"""

import numpy as np
from pathlib import Path

from src.config import get_mech_params

PROJECT_ROOT = Path(__file__).parent.parent

DEFAULT_PARAMS = get_mech_params()


def solve_phi(theta: float, params: dict) -> float:
    """
    给定曲柄角 θ，求解翅膀转角 φ。

    几何关系
    --------
      P1 = B + R*(cosθ, sinθ)        # 主点（曲柄端点）
      P2 满足 |P2 - A| = l 且 |P2 - P1| = c

    求解方法：三角形余弦定理
      在 ΔA-P1-P2 中，已知三边：
        AP2 = l,   P1P2 = c,   AP1 = d(θ) = |P1 - A|
      由余弦定理：
        cos(∠P2AP1) = (l² + d² - c²) / (2*l*d)
      向量 A→P1 的方向角：θ_AP1 = atan2(P1y - a, P1x)
      翅膀角 φ = θ_AP1 + ∠P2AP1          （取上方/外侧交点，使机构连续）

    参数
    ----
    theta : float
        曲柄角 [rad]。顺时针旋转对应 θ 从 0 向 -2π 变化。
    params : dict
        机构参数 dict，需包含 a, b, R, c, l。

    返回
    ----
    phi : float
        翅膀转角 [rad]，向上为正。若机构无解返回 np.nan。
    """
    a = params['a']
    b = params['b']
    R = params['R']
    c_len = params['c']   # 避免与数学常数 e 混淆，本地变量用 c_len
    l_len = params['l']

    # 固定点坐标（按坐标系定义：A 在 y 轴上，B 在 x 轴上）
    A = np.array([0.0, a])
    B = np.array([b, 0.0])

    # 主点 P1（曲柄端点）
    P1 = B + R * np.array([np.cos(theta), np.sin(theta)])

    # 向量 A → P1
    AP1 = P1 - A
    d = np.linalg.norm(AP1)

    # 机构可达性检查（Grashof 条件已满足时通常不会触发）
    min_reach = abs(c_len - l_len)
    max_reach = c_len + l_len
    if d < min_reach - 1e-6 or d > max_reach + 1e-6:
        return np.nan

    # 向量 A→P1 的方向角
    theta_AP1 = np.arctan2(AP1[1], AP1[0])

    # 余弦定理求夹角 ∠P2AP1
    cos_angle = (l_len**2 + d**2 - c_len**2) / (2.0 * l_len * d)
    # 数值截断，防止浮点误差导致 |cos| > 1
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # 选择上方（外侧）交点，使得 P2 位于 AP1 的逆时针方向。
    # 这样在整个顺时针曲柄运动过程中，φ 连续变化，无需分支切换。
    phi = theta_AP1 + angle

    return phi


def wing_kinematics(
    f: float,
    a: float = None,
    rotation: str = 'cw',
    phi_offset_deg: float = None,
    params: dict = None,
    n_points: int = 2000,
):
    """
    生成翅膀运动学。

    曲柄匀速转动，转一圈 = 翅膀一个完整拍动周期。
    运动学结果受控于频率 f、参数 a 的大小以及主点旋转方向。

    参数
    ----
    f : float
        振翅频率 [Hz]。曲柄转一圈对应翅膀一拍。
    a : float, optional
        翅膀转轴 A 的 y 坐标 [mm]。若指定则覆盖 params / DEFAULT_PARAMS 中的值。
        这是最重要的可调参数，直接影响摆幅和急回特性。
    rotation : {'cw', 'ccw'}, default 'cw'
        主点（曲柄）旋转方向。'cw' = 顺时针，'ccw' = 逆时针。
        旋转方向会改变翅膀下拍/上拍的先后顺序和角速度分布。
    phi_offset_deg : float, optional
        翅膀安装基准固定偏移 [deg]。用于补偿翅膀折弯角度，
        使机构输出关于水平位置对称。默认使用 DEFAULT_PARAMS['phi_offset_deg']。
        若显式传 0 则关闭偏移。偏移在微分前施加，不改变角速度/角加速度。
    params : dict, optional
        其他机构参数（b, R, c, l）。可部分覆盖 DEFAULT_PARAMS。
    n_points : int, default 2000
        每周期时间采样点数。

    返回
    ----
    t : (n,) ndarray
        时间 [s]，0 → T = 1/f
    phi : (n,) ndarray
        翅膀拍动角 [rad]，向上为正
    phi_dot : (n,) ndarray
        角速度 [rad/s]
    phi_ddot : (n,) ndarray
        角加速度 [rad/s²]
    info : dict
        机构信息（含频率、参数 a、旋转方向、角度范围、峰值角速度/加速度等）
    """
    full_params = DEFAULT_PARAMS.copy()
    if params is not None:
        full_params.update(params)
    # a 作为独立参数，优先级最高
    if a is not None:
        full_params['a'] = float(a)
    p = full_params

    T = 1.0 / f
    t = np.linspace(0, T, n_points)
    dt = t[1] - t[0]

    # 曲柄角 θ：根据旋转方向决定变化趋势
    if rotation == 'cw':
        # 顺时针：θ 从 0 线性减小到 -2π
        theta = np.linspace(0, -2.0 * np.pi, n_points, endpoint=False)
    elif rotation == 'ccw':
        # 逆时针：θ 从 0 线性增大到 +2π
        theta = np.linspace(0, 2.0 * np.pi, n_points, endpoint=False)
    else:
        raise ValueError("rotation 必须为 'cw'（顺时针）或 'ccw'（逆时针）")

    # 逐点求解每个曲柄角对应的翅膀角
    phi = np.array([solve_phi(th, p) for th in theta])

    # 处理可能的无效解（机构死点或参数越界）
    valid = ~np.isnan(phi)
    if not np.all(valid):
        n_invalid = np.sum(~valid)
        print(f"Warning: {n_invalid}/{n_points} 个位置机构无解（参数可能需要调整）")
        if np.any(valid):
            valid_idx = np.where(valid)[0]
            phi = np.interp(
                np.arange(n_points), valid_idx, phi[valid_idx], period=n_points
            )

    # 解缠绕：消除 ±2π 跳变，保证连续微分正确
    phi = np.unwrap(phi)

    # 固定角度偏移（翅膀折弯/安装基准补偿）
    # 若未显式传入，使用 DEFAULT_PARAMS / params 中的默认值
    if phi_offset_deg is None:
        phi_offset_deg = full_params.get('phi_offset_deg', 0.0)
    phi = phi + np.deg2rad(phi_offset_deg)

    # 时间微分得到角速度和角加速度
    phi_dot = np.gradient(phi, dt)
    phi_ddot = np.gradient(phi_dot, dt)

    info = {
        'f_Hz': f,
        'T_s': T,
        'params': p,
        'a': p['a'],
        'rotation': rotation,
        'phi_offset_deg': phi_offset_deg,
        'phi_range_rad': (float(np.min(phi)), float(np.max(phi))),
        'phi_range_deg': (float(np.rad2deg(np.min(phi))), float(np.rad2deg(np.max(phi)))),
        'phi_span_deg': float(np.rad2deg(np.max(phi) - np.min(phi))),
        'phi_dot_max_rad_s': float(np.max(np.abs(phi_dot))),
        'phi_ddot_max_rad_s2': float(np.max(np.abs(phi_ddot))),
        'n_points': n_points,
    }

    return t, phi, phi_dot, phi_ddot, info


# ==================== 自测 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("前置曲柄摇杆机构运动学自测")
    print("=" * 60)

    f = 15.0  # Hz

    # ---- 测试 1：默认参数（a=7.92，顺时针） ----
    print("\n【测试 1】默认参数：a=7.92, rotation='cw'")
    t, phi, phi_dot, phi_ddot, info = wing_kinematics(f=f)
    print(f"  φ range  = [{info['phi_range_deg'][0]:.2f}, {info['phi_range_deg'][1]:.2f}]°")
    print(f"  span     = {info['phi_span_deg']:.2f}°")
    print(f"  |φ̇|_max  = {info['phi_dot_max_rad_s']:.2f} rad/s")

    # ---- 测试 2：不同 a 值扫描 ----
    print("\n【测试 2】a 参数扫描（f=15Hz, cw）:")
    print(f"{'a':>6} {'φ_min':>10} {'φ_max':>10} {'span':>10} {'|φ̇|max':>10}")
    print("-" * 50)
    for a_val in [6.0, 7.0, 7.92, 9.0, 10.0, 11.0]:
        _, _, _, _, info_i = wing_kinematics(f=f, a=a_val)
        print(f"{a_val:6.2f} {info_i['phi_range_deg'][0]:10.2f} "
              f"{info_i['phi_range_deg'][1]:10.2f} {info_i['phi_span_deg']:10.2f} "
              f"{info_i['phi_dot_max_rad_s']:10.2f}")

    # ---- 测试 3：旋转方向对比（a=7.92） ----
    print("\n【测试 3】旋转方向对比（a=7.92, f=15Hz）:")
    for rot in ['cw', 'ccw']:
        _, phi_i, phi_dot_i, _, info_i = wing_kinematics(f=f, a=7.92, rotation=rot)
        print(f"  rotation='{rot}': span={info_i['phi_span_deg']:.2f}°, "
              f"|φ̇|max={info_i['phi_dot_max_rad_s']:.2f} rad/s, "
              f"φ̇>0占比={np.mean(phi_dot_i > 0)*100:.1f}%")

    print("\nDone!")
