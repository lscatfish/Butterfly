#!/usr/bin/env python3
"""
俯仰动力学验证 v2：预计算运动学 + 向量化 + 稳定积分
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
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
WING_BACK = {"S": 15537.6e-3, "R": 147.4e-3, "r2_sq": 0.2876}

AERO = {"k_3d": 0.7}


def cl_cd(alpha_deg):
    C_L = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
    C_D = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))
    return C_L, C_D


def precompute_kinematics(f, a, phi_offset_deg, n_points=2000):
    """预计算一个周期的运动学，返回插值函数"""
    t, phi, phi_dot, phi_ddot, info = wing_kinematics(
        f=f, a=a, phi_offset_deg=phi_offset_deg, n_points=n_points
    )
    T = 1.0 / f
    # 用线性插值（2000点足够密），避免CubicSpline周期边界问题
    # 扩展数组确保模运算后的插值安全
    dt = t[1] - t[0]
    t_ext = np.concatenate([[t[-1] - T], t, [t[0] + T]])
    phi_ext = np.concatenate([[phi[-1]], phi, [phi[0]]])
    phi_dot_ext = np.concatenate([[phi_dot[-1]], phi_dot, [phi_dot[0]]])

    def interp_phi(t_eff):
        return np.interp(t_eff, t_ext, phi_ext)

    def interp_phi_dot(t_eff):
        return np.interp(t_eff, t_ext, phi_dot_ext)

    return interp_phi, interp_phi_dot, T


def get_state_vectorized(t_arr, interp_phi, interp_phi_dot, phase, T):
    """向量化获取多个时刻的状态"""
    t_eff = np.mod(t_arr + phase / (2 * np.pi * T), T)
    return interp_phi(t_eff), interp_phi_dot(t_eff)


def compute_forces_vectorized(phi, phi_dot, theta_p, theta_dot, alpha_deg, wing):
    """向量化计算单翅气动力"""
    psi = phi + theta_p
    Omega = phi_dot + theta_dot

    U = np.abs(Omega) * wing["R"]

    # 攻角选择
    mask_down = Omega <= 0
    C_L = np.zeros_like(Omega)
    C_D = np.zeros_like(Omega)

    if np.any(mask_down):
        cl, cd = cl_cd(alpha_deg)
        C_L[mask_down] = cl
        C_D[mask_down] = cd
    if np.any(~mask_down):
        cl, cd = cl_cd(-alpha_deg)
        C_L[~mask_down] = cl
        C_D[~mask_down] = cd

    const = 0.5 * PHYS["rho"] * U**2 * wing["S"] * wing["r2_sq"] * AERO["k_3d"]
    sign_Omega = np.where(Omega <= 0, -1, 1)

    Fx = const * np.sin(psi) * (sign_Omega * C_D - C_L)
    Fz = const * np.cos(psi) * (C_L - sign_Omega * C_D)

    # 避免静止时奇异值
    Fx = np.where(np.abs(Omega) < 1e-6, 0, Fx)
    Fz = np.where(np.abs(Omega) < 1e-6, 0, Fz)

    return Fx, Fz


def make_rhs(sp_phi_f, sp_phi_dot_f, T_f, sp_phi_b, sp_phi_dot_b, T_b,
             fp, bp):
    """创建 RHS 函数（闭包）"""

    def rhs(t, y):
        theta_p, theta_dot = y

        # 获取前后翅状态
        phi_f, phi_dot_f = get_state_vectorized(
            np.array([t]), sp_phi_f, sp_phi_dot_f, fp["phase"], T_f
        )
        phi_b, phi_dot_b = get_state_vectorized(
            np.array([t]), sp_phi_b, sp_phi_dot_b, bp["phase"], T_b
        )

        phi_f, phi_dot_f = phi_f[0], phi_dot_f[0]
        phi_b, phi_dot_b = phi_b[0], phi_dot_b[0]

        # 气动力
        Fx_f, Fz_f = compute_forces_vectorized(
            np.array([phi_f]), np.array([phi_dot_f]),
            np.array([theta_p]), np.array([theta_dot]),
            fp["alpha_deg"], WING_FRONT
        )
        Fx_b, Fz_b = compute_forces_vectorized(
            np.array([phi_b]), np.array([phi_dot_b]),
            np.array([theta_p]), np.array([theta_dot]),
            bp["alpha_deg"], WING_BACK
        )

        Fx_f, Fz_f = Fx_f[0], Fz_f[0]
        Fx_b, Fz_b = Fx_b[0], Fz_b[0]

        # 四翅合力
        Fx_total = 2 * (Fx_f + Fx_b)
        Fz_total = 2 * (Fz_f + Fz_b)

        # 俯仰力矩
        M_aero = 2 * (Fz_f * PHYS["x_front"] + Fz_b * PHYS["x_back"])
        M_gravity = -PHYS["m_total"] * PHYS["g"] * PHYS["d_cg"] * np.sin(theta_p)
        M_damp = -PHYS["c_damp"] * theta_dot

        theta_ddot = (M_aero + M_gravity + M_damp) / PHYS["I_yy"]

        return [theta_dot, theta_ddot]

    return rhs


def run_case(title, fp, bp, filename, t_span=(0, 2.0)):
    """运行单个 case"""
    print(f"\nPrecomputing: {title} ...")

    # 预计算运动学
    sp_phi_f, sp_phi_dot_f, T_f = precompute_kinematics(
        fp["f"], fp["a"], fp["phi_offset_deg"]
    )
    sp_phi_b, sp_phi_dot_b, T_b = precompute_kinematics(
        bp["f"], bp["a"], bp["phi_offset_deg"]
    )

    rhs = make_rhs(sp_phi_f, sp_phi_dot_f, T_f,
                   sp_phi_b, sp_phi_dot_b, T_b, fp, bp)

    print(f"  Simulating {t_span[1]}s ...")
    sol = solve_ivp(rhs, t_span, [0.0, 0.0], method="RK45",
                    max_step=1e-4, dense_output=True)

    if not sol.success:
        print(f"  FAILED: {sol.message}")
        return None

    # 后处理
    n = 4000
    t = np.linspace(t_span[0], t_span[1], n)
    y = sol.sol(t)
    theta_p = y[0]
    theta_dot = y[1]

    # 重新计算力
    phi_f, phi_dot_f = get_state_vectorized(t, sp_phi_f, sp_phi_dot_f, fp["phase"], T_f)
    phi_b, phi_dot_b = get_state_vectorized(t, sp_phi_b, sp_phi_dot_b, bp["phase"], T_b)

    Fx_f, Fz_f = compute_forces_vectorized(phi_f, phi_dot_f, theta_p, theta_dot, fp["alpha_deg"], WING_FRONT)
    Fx_b, Fz_b = compute_forces_vectorized(phi_b, phi_dot_b, theta_p, theta_dot, bp["alpha_deg"], WING_BACK)

    Fx_total = 2 * (Fx_f + Fx_b)
    Fz_total = 2 * (Fz_f + Fz_b)
    M_aero = 2 * (Fz_f * PHYS["x_front"] + Fz_b * PHYS["x_back"])

    # 稳态统计
    steady = slice(n // 2, n)
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
    print("俯仰动力学验证 v2")
    print("=" * 70)

    base = {"f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84, "alpha_deg": 45.0}

    cases = []

    # Case 1: 无相位差
    f1, b1 = base.copy(), base.copy()
    cases.append(("Case 1: Baseline (no phase diff)", f1, b1, "temp/case1_v2.png"))

    # Case 2: 反相 180°
    f2, b2 = base.copy(), base.copy()
    b2["phase"] = np.pi
    cases.append(("Case 2: Anti-phase 180 deg", f2, b2, "temp/case2_v2.png"))

    # Case 3: 反相 + 85°
    f3, b3 = base.copy(), base.copy()
    b3["phase"] = np.pi
    f3["alpha_deg"] = 85.0
    b3["alpha_deg"] = 85.0
    cases.append(("Case 3: Anti-phase + alpha=85", f3, b3, "temp/case3_v2.png"))

    # Case 4: 反相 + 85° + 25Hz
    f4, b4 = base.copy(), base.copy()
    b4["phase"] = np.pi
    f4["alpha_deg"] = 85.0
    b4["alpha_deg"] = 85.0
    f4["f"] = 25.0
    b4["f"] = 25.0
    cases.append(("Case 4: Anti-phase + alpha=85 + f=25Hz", f4, b4, "temp/case4_v2.png"))

    # Case 5: 反相 + 前翅大攻角后翅小攻角
    f5, b5 = base.copy(), base.copy()
    b5["phase"] = np.pi
    f5["alpha_deg"] = 85.0
    b5["alpha_deg"] = 30.0
    cases.append(("Case 5: Anti-phase + diff alpha", f5, b5, "temp/case5_v2.png"))

    # Case 6: 小相位差 90°
    f6, b6 = base.copy(), base.copy()
    b6["phase"] = np.pi / 2
    cases.append(("Case 6: 90 deg phase diff", f6, b6, "temp/case6_v2.png"))

    results = []
    for title, fp, bp, fn in cases:
        res = run_case(title, fp, bp, fn, t_span=(0, 2.0))
        if res:
            results.append(res)
            print(f"  Lift: {res['avg_Fz']:+.2f} mN | Thrust: {res['avg_Fx']:+.2f} mN | L/W: {res['ratio']:.3f}")

    # 汇总
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
