#!/usr/bin/env python3
"""
俯仰动力学验证 v3：固定步长 RK4，避免 stiff 问题
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
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

WING_FRONT = {"S": 16166.6e-6, "R": 154.3e-3, "r2_sq": 0.2382}
WING_BACK = {"S": 15537.6e-6, "R": 147.4e-3, "r2_sq": 0.2876}

AERO = {"k_3d": 0.7}


def cl_cd(alpha_deg):
    C_L = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
    C_D = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))
    return C_L, C_D


def precompute_kinematics(f, a, phi_offset_deg, n_points=2000):
    """预计算一个周期的运动学"""
    t, phi, phi_dot, phi_ddot, info = wing_kinematics(
        f=f, a=a, phi_offset_deg=phi_offset_deg, n_points=n_points
    )
    T = 1.0 / f
    # 扩展以支持安全插值
    t_ext = np.concatenate([[t[-1] - T], t, [t[0] + T]])
    phi_ext = np.concatenate([[phi[-1]], phi, [phi[0]]])
    phi_dot_ext = np.concatenate([[phi_dot[-1]], phi_dot, [phi_dot[0]]])
    return t_ext, phi_ext, phi_dot_ext, T


def get_states(t_arr, t_ext, phi_ext, phi_dot_ext, phase, T):
    """向量化插值"""
    t_eff = np.mod(t_arr + phase / (2 * np.pi * T), T)
    phi = np.interp(t_eff, t_ext, phi_ext)
    phi_dot = np.interp(t_eff, t_ext, phi_dot_ext)
    return phi, phi_dot


def compute_forces(phi, phi_dot, theta_p, theta_dot, alpha_down, alpha_up, wing):
    """计算单翅气动力（全部向量化），支持非对称攻角"""
    psi = phi + theta_p
    Omega = phi_dot + theta_dot

    U = np.abs(Omega) * wing["R"]
    const = 0.5 * PHYS["rho"] * U**2 * wing["S"] * wing["r2_sq"] * AERO["k_3d"]

    sign_Omega = np.where(Omega <= 0, -1, 1)

    # 攻角选择（基于 phi_dot，不是 Omega）
    C_L = np.zeros_like(Omega)
    C_D = np.zeros_like(Omega)
    mask_down = phi_dot <= 0  # 下拍
    if np.any(mask_down):
        cl, cd = cl_cd(alpha_down)
        C_L[mask_down] = cl
        C_D[mask_down] = cd
    if np.any(~mask_down):
        cl, cd = cl_cd(alpha_up)
        C_L[~mask_down] = cl
        C_D[~mask_down] = cd

    # 力的方向投影用绝对运动 sign_Omega，大小用 |Omega|
    Fx = const * np.sin(psi) * (sign_Omega * C_D - C_L)
    Fz = const * np.cos(psi) * (C_L - sign_Omega * C_D)

    # 静止时归零
    Fx = np.where(np.abs(Omega) < 1e-6, 0, Fx)
    Fz = np.where(np.abs(Omega) < 1e-6, 0, Fz)

    return Fx, Fz


def compute_rhs(theta_p, theta_dot, phi_f, phi_dot_f, phi_b, phi_dot_b,
                fp, bp):
    """计算 RHS: [theta_dot, theta_ddot]"""
    Fx_f, Fz_f = compute_forces(phi_f, phi_dot_f, theta_p, theta_dot,
                                fp["alpha_down"], fp["alpha_up"], WING_FRONT)
    Fx_b, Fz_b = compute_forces(phi_b, phi_dot_b, theta_p, theta_dot,
                                bp["alpha_down"], bp["alpha_up"], WING_BACK)

    Fx_total = 2 * (Fx_f + Fx_b)
    Fz_total = 2 * (Fz_f + Fz_b)

    M_aero = 2 * (Fz_f * PHYS["x_front"] + Fz_b * PHYS["x_back"])
    M_gravity = -PHYS["m_total"] * PHYS["g"] * PHYS["d_cg"] * np.sin(theta_p)
    M_damp = -PHYS["c_damp"] * theta_dot

    theta_ddot = (M_aero + M_gravity + M_damp) / PHYS["I_yy"]

    return theta_dot, theta_ddot, Fx_total, Fz_total, M_aero


def rk4_step(y, dt, *args):
    """单步 RK4"""
    theta_p, theta_dot = y

    k1_td, k1_tdd, _, _, _ = compute_rhs(theta_p, theta_dot, *args)
    k2_td, k2_tdd, _, _, _ = compute_rhs(
        theta_p + 0.5*dt*k1_td, theta_dot + 0.5*dt*k1_tdd, *args
    )
    k3_td, k3_tdd, _, _, _ = compute_rhs(
        theta_p + 0.5*dt*k2_td, theta_dot + 0.5*dt*k2_tdd, *args
    )
    k4_td, k4_tdd, _, _, _ = compute_rhs(
        theta_p + dt*k3_td, theta_dot + dt*k3_tdd, *args
    )

    theta_p_new = theta_p + dt * (k1_td + 2*k2_td + 2*k3_td + k4_td) / 6
    theta_dot_new = theta_dot + dt * (k1_tdd + 2*k2_tdd + 2*k3_tdd + k4_tdd) / 6

    return theta_p_new, theta_dot_new


def run_case(title, fp, bp, filename, t_end=2.0, n_steps=40000):
    """固定步长 RK4 积分"""
    print(f"\nPrecomputing: {title} ...")

    t_ext_f, phi_ext_f, phi_dot_ext_f, T_f = precompute_kinematics(
        fp["f"], fp["a"], fp["phi_offset_deg"]
    )
    t_ext_b, phi_ext_b, phi_dot_ext_b, T_b = precompute_kinematics(
        bp["f"], bp["a"], bp["phi_offset_deg"]
    )

    dt = t_end / n_steps
    print(f"  dt = {dt*1e6:.1f} us, steps = {n_steps}")

    # 预计算所有时刻的翅膀状态
    t = np.linspace(0, t_end, n_steps)
    phi_f, phi_dot_f = get_states(t, t_ext_f, phi_ext_f, phi_dot_ext_f, fp["phase"], T_f)
    phi_b, phi_dot_b = get_states(t, t_ext_b, phi_ext_b, phi_dot_ext_b, bp["phase"], T_b)

    # 积分
    theta_p = np.zeros(n_steps)
    theta_dot = np.zeros(n_steps)
    Fx_total = np.zeros(n_steps)
    Fz_total = np.zeros(n_steps)
    M_aero = np.zeros(n_steps)

    for i in range(n_steps - 1):
        _, _, Fx_t, Fz_t, M_t = compute_rhs(
            theta_p[i], theta_dot[i],
            phi_f[i], phi_dot_f[i], phi_b[i], phi_dot_b[i],
            fp, bp
        )
        Fx_total[i] = Fx_t
        Fz_total[i] = Fz_t
        M_aero[i] = M_t

        theta_p[i+1], theta_dot[i+1] = rk4_step(
            [theta_p[i], theta_dot[i]], dt,
            phi_f[i], phi_dot_f[i], phi_b[i], phi_dot_b[i],
            fp, bp
        )

    # 最后一时刻的力
    _, _, Fx_t, Fz_t, M_t = compute_rhs(
        theta_p[-1], theta_dot[-1],
        phi_f[-1], phi_dot_f[-1], phi_b[-1], phi_dot_b[-1],
        fp, bp
    )
    Fx_total[-1] = Fx_t
    Fz_total[-1] = Fz_t
    M_aero[-1] = M_t

    # 统计
    steady = slice(n_steps // 2, n_steps)
    weight_mN = PHYS["m_total"] * PHYS["g"] * 1000

    res = {
        "title": title,
        "t": t, "t_ms": t * 1000,
        "theta_p_deg": np.degrees(theta_p),
        "Fx_total_mN": Fx_total * 1000,
        "Fz_total_mN": Fz_total * 1000,
        "M_aero_uNm": M_aero * 1e6,
        "avg_Fx": np.mean(Fx_total[steady]) * 1000,
        "avg_Fz": np.mean(Fz_total[steady]) * 1000,
        "avg_M": np.mean(M_aero[steady]) * 1e6,
        "max_theta": np.max(np.abs(np.degrees(theta_p[steady]))),
        "weight": weight_mN,
        "ratio": np.mean(Fz_total[steady]) * 1000 / weight_mN,
    }

    # 绘图
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


def main():
    print("=" * 70)
    print("俯仰动力学验证 v3 (RK4 fixed step)")
    print("=" * 70)

    # 非对称攻角：下拍 45°，上拍 -10°（模拟翅膀扭转/supination）
    base = {
        "f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84,
        "alpha_down": 45.0, "alpha_up": -10.0
    }

    cases = []

    # Case 1: 无相位差
    f1, b1 = base.copy(), base.copy()
    cases.append(("Case 1: Baseline (asym alpha)", f1, b1, "temp/case1_v3.png"))

    # Case 2: 反相 180°
    f2, b2 = base.copy(), base.copy()
    b2["phase"] = np.pi
    cases.append(("Case 2: Anti-phase 180", f2, b2, "temp/case2_v3.png"))

    # Case 3: 反相 + 大下拍攻角
    f3, b3 = base.copy(), base.copy()
    b3["phase"] = np.pi
    f3["alpha_down"] = 60.0
    b3["alpha_down"] = 60.0
    cases.append(("Case 3: Anti-phase + down=60", f3, b3, "temp/case3_v3.png"))

    # Case 4: 反相 + 高频率
    f4, b4 = base.copy(), base.copy()
    b4["phase"] = np.pi
    f4["f"] = 25.0
    b4["f"] = 25.0
    cases.append(("Case 4: Anti-phase + f=25", f4, b4, "temp/case4_v3.png"))

    # Case 5: 小相位差 90°
    f5, b5 = base.copy(), base.copy()
    b5["phase"] = np.pi / 2
    cases.append(("Case 5: 90 deg phase diff", f5, b5, "temp/case5_v3.png"))

    results = []
    for title, fp, bp, fn in cases:
        res = run_case(title, fp, bp, fn, t_end=2.0, n_steps=40000)
        if res:
            results.append(res)
            print(f"  Lift: {res['avg_Fz']:+.2f} mN | Thrust: {res['avg_Fx']:+.2f} mN | L/W: {res['ratio']:.3f}")

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
