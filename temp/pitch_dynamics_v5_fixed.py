#!/usr/bin/env python3
"""
俯仰动力学验证 v5-fixed：固定点 + 俯仰自由度 + 四分量准定常模型

坐标系定义（所有力均在惯性系/机体坐标系中计算）：
- 惯性系 I：原点在地面，X_I 水平向前（蝴蝶头方向），Z_I 垂直向上
- 机体坐标系 B：原点在俯仰轴（重心附近），x_B 沿机身向前，z_B 垂直向上，
  y_B 沿翼展向右（右手系：x × y = z）
- 由于蝴蝶固定在空间某点（不移动），机体坐标系与惯性系对齐（无平移）

翅膀姿态：绕 y 轴旋转 psi = phi + theta_p
- psi = 0：翅膀水平，x_W 沿 x_B，z_W 沿 z_B
- psi > 0：翅膀前缘上抬

力的方向（简化拍动翼模型）：
- 升力 L：垂直于翅膀平面，"向上"方向 n_L = (-sin(psi), 0, cos(psi))
- 阻力 D：平行于速度方向，与运动相反
- 投影公式（v3 验证过）：
    Fx = sin(psi) * (sign*D - L)
    Fz = cos(psi) * (L - sign*D)
- 物理含义：
    - psi = 0（水平）：Fx = 0，Fz = L - sign*D（纯垂直力）
    - psi = 90°（垂直）：Fx = sign*D - L，Fz = 0（纯水平力）
    - 下拍（sign = -1）：Fz = L + D > 0（向上升力）
    - 上拍（sign = +1）：Fx = D - L > 0（当 L < D 时，向前推力）

攻角约定：
- 下拍（phi_dot <= 0）：有效攻角 = +alpha（如 40°）
- 上拍（phi_dot > 0）：有效攻角 = -alpha（如 -10° 到 -40°）
- |alpha| <= 60°（经验公式有效范围）

四分量力模型：
- F_t：平动升力/阻力（基于瞬时速度）
- F_AM：附加质量力（基于 phi_ddot，垂直翅膀平面）
- F_clap：Clap-and-Fling（反转点附近增强 1.3 倍）
- F_rot：旋转力（Kramer 效应，当前 alpha_dot = 0，自然为 0）
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mechanism import wing_kinematics

# ==================== 物理参数 ====================
PHYS = {
    "rho": 1.225,
    "g": 9.81,
    "m_total": 0.020,
    "I_yy": 3e-5,
    "x_front": 0.025,
    "x_back": -0.025,
    "d_cg": 0.015,
    "c_damp": 5e-4,
}

AERO = {
    "k_3d": 0.7,
    "C_rot": 1.5,
    "r_rot": 0.5,
    "k_clap": 1.3,
    "alpha_max_deg": 60.0,  # 经验公式上限
}


def load_geometry():
    """读取 wing geometry（含面积矩等）"""
    json_path = Path(__file__).parent.parent / "data" / "wing_analysis_results.json"
    with open(json_path) as f:
        data = json.load(f)
    geo = {}
    for g in data["geometry"]:
        geo[g["name"]] = {
            "S": g["S"],
            "R": g["R"],
            "c_avg": g["c_avg"],
            "r1": g["r1"],
            "r2_sq": g["r2_sq"],
            "AR": g["AR"],
        }
    return geo


GEO = load_geometry()
WING_FRONT = GEO["Front"]
WING_BACK = GEO["Back"]


def cl_cd(alpha_deg):
    """升阻力系数（alpha 单位：度），经验公式有效范围约 |alpha| <= 60°"""
    C_L = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
    C_D = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))
    return C_L, C_D


def clamp_alpha(alpha):
    """限制攻角在经验公式有效范围内"""
    return np.clip(alpha, -AERO["alpha_max_deg"], AERO["alpha_max_deg"])


def precompute_kinematics(f, a, phi_offset_deg, n_points=2000):
    """预计算一个周期的运动学"""
    t, phi, phi_dot, phi_ddot, info = wing_kinematics(
        f=f, a=a, phi_offset_deg=phi_offset_deg, n_points=n_points
    )
    T = 1.0 / f
    dt = t[1] - t[0]
    t_ext = np.concatenate([[t[-1] - T], t, [t[0] + T]])
    phi_ext = np.concatenate([[phi[-1]], phi, [phi[0]]])
    phi_dot_ext = np.concatenate([[phi_dot[-1]], phi_dot, [phi[0]]])
    phi_ddot_ext = np.concatenate([[phi_ddot[-1]], phi_ddot, [phi[0]]])
    return t_ext, phi_ext, phi_dot_ext, phi_ddot_ext, T, dt


def get_states(t_arr, t_ext, phi_ext, phi_dot_ext, phi_ddot_ext, phase, T):
    """向量化插值获取状态"""
    t_eff = np.mod(t_arr + phase / (2 * np.pi * T), T)
    phi = np.interp(t_eff, t_ext, phi_ext)
    phi_dot = np.interp(t_eff, t_ext, phi_dot_ext)
    phi_ddot = np.interp(t_eff, t_ext, phi_ddot_ext)
    return phi, phi_dot, phi_ddot


def compute_forces(phi, phi_dot, phi_ddot, theta_p, theta_dot,
                   alpha_down, alpha_up, wing, rho=1.225):
    """
    计算单翅在机体坐标系中的三维气动力和力矩（向量化）。
    保留 v3 验证过的投影公式，增加附加质量力和 Clap-and-Fling。
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

    # ========== 攻角选择（基于 phi_dot，不是 Omega）==========
    mask_down = phi_dot <= 0
    alpha_eff = np.zeros_like(phi)
    C_L = np.zeros_like(phi)
    C_D = np.zeros_like(phi)
    if np.any(mask_down):
        a = clamp_alpha(alpha_down)
        cl, cd = cl_cd(a)
        C_L[mask_down] = cl
        C_D[mask_down] = cd
        alpha_eff[mask_down] = a
    if np.any(~mask_down):
        a = clamp_alpha(alpha_up)
        cl, cd = cl_cd(a)
        C_L[~mask_down] = cl
        C_D[~mask_down] = cd
        alpha_eff[~mask_down] = a

    # ========== 1. 平动分量 ==========
    L_trans = const * C_L
    D_trans = const * C_D

    # ========== 2. 附加质量力 ==========
    # F_AM = -(rho * pi * c_avg^2 / 4) * phi_ddot * R * r1 * sin(alpha)
    # 方向：垂直于翅膀平面，与升力同向（当 phi_ddot > 0 且 sin(alpha) > 0 时为负）
    alpha_rad = np.deg2rad(alpha_eff)
    F_AM = -(rho * np.pi * c_avg**2 / 4.0) * phi_ddot * R * r1 * np.sin(alpha_rad)

    # ========== 3. 旋转力（Kramer效应）==========
    # alpha_dot = 0（当前机构无扭转自由度），自然为 0
    alpha_dot = np.zeros_like(phi)
    F_rot = rho * AERO["C_rot"] * alpha_dot * phi_dot * c_avg**2 * R * AERO["r_rot"]

    # ========== 4. Clap-and-Fling ==========
    phi_dot_peak = np.max(np.abs(phi_dot))
    reversal_threshold = 0.1 * phi_dot_peak
    in_reversal = np.abs(phi_dot) < reversal_threshold
    k_clap = np.where(in_reversal, AERO["k_clap"], 1.0)

    # ========== 5. 总有效升力与阻力 ==========
    # 升力类分量（垂直翅膀平面）：平动升力 + 附加质量 + 旋转力
    L_eff = (L_trans + F_AM + F_rot) * k_clap
    # 阻力分量（翅膀平面内，与速度相反）
    D_eff = D_trans * k_clap

    # ========== 6. 投影到机体坐标系 ==========
    # 使用 v3 验证过的投影公式
    # Fx = sin(psi) * (sign*D - L)
    # Fz = cos(psi) * (L - sign*D)
    Fx = np.sin(psi) * (sign_Omega * D_eff - L_eff)
    Fz = np.cos(psi) * (L_eff - sign_Omega * D_eff)

    # 静止时归零
    mask_still = np.abs(Omega) < 1e-6
    Fx = np.where(mask_still, 0, Fx)
    Fz = np.where(mask_still, 0, Fz)

    F_body = np.zeros((n, 3))
    F_body[:, 0] = Fx
    F_body[:, 2] = Fz

    # ========== 7. 严格三维力矩 ==========
    # 力矩 M = r × F，力臂 r = (x_wing, 0, 0)
    # M_y = r_z * F_x - r_x * F_z = -x_wing * F_z（因为 r_z = 0）
    # 这里暂不乘 x_wing，由调用方根据前/后翅分别施加
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
    return F_body, M_body, info


def compute_rhs(theta_p, theta_dot, phi_f, phi_dot_f, phi_ddot_f,
                phi_b, phi_dot_b, phi_ddot_b, fp, bp):
    """计算 RHS: [theta_dot, theta_ddot]"""
    F_f, _, info_f = compute_forces(
        phi_f, phi_dot_f, phi_ddot_f, theta_p, theta_dot,
        fp["alpha_down"], fp["alpha_up"], WING_FRONT
    )
    F_b, _, info_b = compute_forces(
        phi_b, phi_dot_b, phi_ddot_b, theta_p, theta_dot,
        bp["alpha_down"], bp["alpha_up"], WING_BACK
    )

    # 四翅合力（左右对称 ×2）
    Fx_total = 2 * (F_f[:, 0] + F_b[:, 0])
    Fy_total = 2 * (F_f[:, 1] + F_b[:, 1])
    Fz_total = 2 * (F_f[:, 2] + F_b[:, 2])

    # 严格三维力矩：r × F
    # 前翅力臂 r_f = (x_front, 0, 0)，后翅力臂 r_b = (x_back, 0, 0)
    # M_y = -x_front * Fz_f - x_back * Fz_b
    M_aero = 2 * (-PHYS["x_front"] * F_f[:, 2] - PHYS["x_back"] * F_b[:, 2])

    # 重力恢复力矩
    M_gravity = -PHYS["m_total"] * PHYS["g"] * PHYS["d_cg"] * np.sin(theta_p)
    # 阻尼力矩
    M_damp = -PHYS["c_damp"] * theta_dot

    theta_ddot = (M_aero + M_gravity + M_damp) / PHYS["I_yy"]

    # 标量化（兼容单步 RK4）
    if np.ndim(theta_ddot) > 0:
        theta_ddot = float(theta_ddot[0])
    if np.ndim(Fx_total) > 0:
        Fx_total = float(Fx_total[0])
        Fy_total = float(Fy_total[0])
        Fz_total = float(Fz_total[0])
        M_aero = float(M_aero[0])

    return float(theta_dot), float(theta_ddot), Fx_total, Fy_total, Fz_total, M_aero


def rk4_step(y, dt, *args):
    """单步 RK4"""
    theta_p, theta_dot = y

    k1_td, k1_tdd, _, _, _, _ = compute_rhs(theta_p, theta_dot, *args)
    k2_td, k2_tdd, _, _, _, _ = compute_rhs(
        theta_p + 0.5*dt*k1_td, theta_dot + 0.5*dt*k1_tdd, *args
    )
    k3_td, k3_tdd, _, _, _, _ = compute_rhs(
        theta_p + 0.5*dt*k2_td, theta_dot + 0.5*dt*k2_tdd, *args
    )
    k4_td, k4_tdd, _, _, _, _ = compute_rhs(
        theta_p + dt*k3_td, theta_dot + dt*k3_tdd, *args
    )

    theta_p_new = theta_p + dt * (k1_td + 2*k2_td + 2*k3_td + k4_td) / 6
    theta_dot_new = theta_dot + dt * (k1_tdd + 2*k2_tdd + 2*k3_tdd + k4_tdd) / 6

    return theta_p_new, theta_dot_new


def run_case(title, fp, bp, filename, t_end=2.0, n_steps=40000):
    """固定步长 RK4 积分"""
    print(f"\nPrecomputing: {title} ...")

    t_ext_f, phi_ext_f, phi_dot_ext_f, phi_ddot_ext_f, T_f, _ = precompute_kinematics(
        fp["f"], fp["a"], fp["phi_offset_deg"]
    )
    t_ext_b, phi_ext_b, phi_dot_ext_b, phi_ddot_ext_b, T_b, _ = precompute_kinematics(
        bp["f"], bp["a"], bp["phi_offset_deg"]
    )

    dt = t_end / n_steps
    print(f"  dt = {dt*1e6:.1f} us, steps = {n_steps}")

    t = np.linspace(0, t_end, n_steps)
    phi_f, phi_dot_f, phi_ddot_f = get_states(
        t, t_ext_f, phi_ext_f, phi_dot_ext_f, phi_ddot_ext_f, fp["phase"], T_f
    )
    phi_b, phi_dot_b, phi_ddot_b = get_states(
        t, t_ext_b, phi_ext_b, phi_dot_ext_b, phi_ddot_ext_b, bp["phase"], T_b
    )

    theta_p = np.zeros(n_steps)
    theta_dot = np.zeros(n_steps)
    Fx_total = np.zeros(n_steps)
    Fy_total = np.zeros(n_steps)
    Fz_total = np.zeros(n_steps)
    M_aero = np.zeros(n_steps)

    for i in range(n_steps - 1):
        _, _, Fx_t, Fy_t, Fz_t, M_t = compute_rhs(
            theta_p[i], theta_dot[i],
            phi_f[i:i+1], phi_dot_f[i:i+1], phi_ddot_f[i:i+1],
            phi_b[i:i+1], phi_dot_b[i:i+1], phi_ddot_b[i:i+1],
            fp, bp
        )
        Fx_total[i] = Fx_t
        Fy_total[i] = Fy_t
        Fz_total[i] = Fz_t
        M_aero[i] = M_t

        theta_p[i+1], theta_dot[i+1] = rk4_step(
            [theta_p[i], theta_dot[i]], dt,
            phi_f[i:i+1], phi_dot_f[i:i+1], phi_ddot_f[i:i+1],
            phi_b[i:i+1], phi_dot_b[i:i+1], phi_ddot_b[i:i+1],
            fp, bp
        )

    _, _, Fx_t, Fy_t, Fz_t, M_t = compute_rhs(
        theta_p[-1], theta_dot[-1],
        phi_f[-1:], phi_dot_f[-1:], phi_ddot_f[-1:],
        phi_b[-1:], phi_dot_b[-1:], phi_ddot_b[-1:],
        fp, bp
    )
    Fx_total[-1] = Fx_t
    Fy_total[-1] = Fy_t
    Fz_total[-1] = Fz_t
    M_aero[-1] = M_t

    steady = slice(n_steps // 2, n_steps)
    weight_mN = PHYS["m_total"] * PHYS["g"] * 1000

    res = {
        "title": title,
        "t": t, "t_ms": t * 1000,
        "theta_p_deg": np.degrees(theta_p),
        "Fx_total_mN": Fx_total * 1000,
        "Fy_total_mN": Fy_total * 1000,
        "Fz_total_mN": Fz_total * 1000,
        "M_aero_uNm": M_aero * 1e6,
        "avg_Fx": np.mean(Fx_total[steady]) * 1000,
        "avg_Fy": np.mean(Fy_total[steady]) * 1000,
        "avg_Fz": np.mean(Fz_total[steady]) * 1000,
        "avg_M": np.mean(M_aero[steady]) * 1e6,
        "max_theta": np.max(np.abs(np.degrees(theta_p[steady]))),
        "weight": weight_mN,
        "ratio": np.mean(Fz_total[steady]) * 1000 / weight_mN,
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(res["t_ms"], res["theta_p_deg"], "b-", lw=1.2)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Pitch (deg)")
    ax.set_title(f"Body Pitch | max={res['max_theta']:.1f} deg")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(res["t_ms"], theta_dot, "r-", lw=1.2)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Pitch rate (rad/s)")
    ax.set_title("Pitch Rate")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(res["t_ms"], res["Fz_total_mN"], "g-", lw=1.2)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axhline(res["avg_Fz"], color="g", ls="--", alpha=0.7)
    ax.axhline(res["weight"], color="r", ls=":", alpha=0.7, label="Weight")
    ax.set_ylabel("Lift (mN)")
    ax.set_title(f"Lift | avg={res['avg_Fz']:+.2f} mN")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(res["t_ms"], res["Fx_total_mN"], "m-", lw=1.2)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axhline(res["avg_Fx"], color="m", ls="--", alpha=0.7)
    ax.set_ylabel("Thrust (mN)")
    ax.set_title(f"Thrust | avg={res['avg_Fx']:+.2f} mN")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(res["t_ms"], res["M_aero_uNm"], "orange", lw=1.2)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axhline(res["avg_M"], color="orange", ls="--", alpha=0.7)
    ax.set_ylabel("Moment (uN m)")
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Pitch Moment | avg={res['avg_M']:+.1f}")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(res["theta_p_deg"], res["Fz_total_mN"], ".", ms=1.5, alpha=0.4, color="purple")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Pitch angle (deg)")
    ax.set_ylabel("Lift (mN)")
    ax.set_title("Lift vs Pitch")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")

    return res


def alpha_scan():
    """扫描最佳安装角组合"""
    print("\n" + "=" * 70)
    print("攻角扫描：寻找最佳安装角")
    print("=" * 70)

    base = {"f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84}

    # 扫描参数：下拍攻角 10°-60°，上拍攻角 -5° 到 -60°
    alpha_down_range = [20, 30, 40, 45, 50, 55, 60]
    alpha_up_range = [-5, -10, -15, -20, -30, -40, -50, -60]

    results = []
    for ad in alpha_down_range:
        for au in alpha_up_range:
            fp = base.copy()
            bp = base.copy()
            fp["alpha_down"] = ad
            fp["alpha_up"] = au
            bp["alpha_down"] = ad
            bp["alpha_up"] = au

            title = f"ad={ad}, au={au}"
            res = run_case(title, fp, bp, f"temp/scan_{ad}_{au}.png", t_end=1.5, n_steps=30000)
            if res:
                results.append({
                    "ad": ad, "au": au,
                    "lift": res["avg_Fz"],
                    "thrust": res["avg_Fx"],
                    "ratio": res["ratio"],
                    "max_pitch": res["max_theta"],
                })
                print(f"  ad={ad:2d} au={au:3d} | L={res['avg_Fz']:+.1f} mN | T={res['avg_Fx']:+.1f} mN | L/W={res['ratio']:.3f} | |Pitch|={res['max_theta']:.1f}°")

    # 找最佳组合
    print("\n" + "=" * 70)
    print("最佳组合（按升重比排序）")
    print("=" * 70)
    results.sort(key=lambda x: x["ratio"], reverse=True)
    print(f"{'ad':>4} {'au':>4} {'Lift':>10} {'Thrust':>10} {'L/W':>8} {'|Pitch|':>8}")
    for r in results[:10]:
        print(f"{r['ad']:4d} {r['au']:4d} {r['lift']:>+10.1f} {r['thrust']:>+10.1f} {r['ratio']:>8.3f} {r['max_pitch']:>7.1f}°")

    return results


def main():
    print("=" * 70)
    print("俯仰动力学验证 v5-fixed (固定点 + 俯仰 + 四分量模型)")
    print("=" * 70)

    # 先做攻角扫描找最佳值
    scan_results = alpha_scan()

    # 用最佳参数跑详细 case
    if scan_results:
        best = scan_results[0]
        print(f"\n最佳参数: alpha_down={best['ad']}°, alpha_up={best['au']}°")

        base = {
            "f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84,
            "alpha_down": best["ad"], "alpha_up": best["au"]
        }

        cases = []

        f1, b1 = base.copy(), base.copy()
        cases.append(("Case 1: Baseline", f1, b1, "temp/case1_v5_fixed.png"))

        f2, b2 = base.copy(), base.copy()
        b2["phase"] = np.pi
        cases.append(("Case 2: Anti-phase 180", f2, b2, "temp/case2_v5_fixed.png"))

        f3, b3 = base.copy(), base.copy()
        b3["phase"] = np.pi
        f3["f"] = 25.0
        b3["f"] = 25.0
        cases.append(("Case 3: Anti-phase f=25", f3, b3, "temp/case3_v5_fixed.png"))

        f4, b4 = base.copy(), base.copy()
        b4["phase"] = np.pi / 2
        cases.append(("Case 4: 90° phase diff", f4, b4, "temp/case4_v5_fixed.png"))

        results = []
        for title, fp, bp, fn in cases:
            res = run_case(title, fp, bp, fn, t_end=2.0, n_steps=40000)
            if res:
                results.append(res)
                print(f"  Lift: {res['avg_Fz']:+.2f} mN | Thrust: {res['avg_Fx']:+.2f} mN | L/W: {res['ratio']:.3f} | |Pitch|: {res['max_theta']:.1f}°")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Case':<35} {'Lift(mN)':>10} {'Thrust(mN)':>12} {'L/W':>8} {'|Pitch|':>10}")
        print("-" * 80)
        for r in results:
            short = r['title'].split(":")[0]
            print(f"{short:<35} {r['avg_Fz']:>+10.2f} {r['avg_Fx']:>+12.2f} {r['ratio']:>8.3f} {r['max_theta']:>9.1f} deg")

    print("\nDone!")


if __name__ == "__main__":
    main()
