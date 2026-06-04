#!/usr/bin/env python3
"""
俯仰动力学验证：前后翅相位差产生俯仰力矩，通过俯仰旋转转化负升力为推力。

模型假设：
1. 蝴蝶固定在空间某点（不移动），只有俯仰自由度 θ_p(t)
2. 四翅对称，左右翅膀同相拍动
3. 前后翅可以有独立参数（相位、频率、攻角、a、偏移）
4. 气动力包含垂直（升力）和水平（推力）分量
5. 使用简化准定常模型
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
from pathlib import Path

# 导入 mechanism.py
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mechanism import wing_kinematics

# ==================== 参数 ====================
PHYS = {
    "rho": 1.225,          # 空气密度 kg/m³
    "g": 9.81,             # 重力加速度 m/s²
    "m_total": 0.020,      # 总质量 kg
    "I_yy": 3e-5,          # 俯仰转动惯量 kg·m²（估算）
    "x_front": 0.025,      # 前翅到俯仰轴距离 m（前方为正）
    "x_back": -0.025,      # 后翅到俯仰轴距离 m（后方为负）
    "d_cg": 0.015,         # 重心到俯仰轴垂直距离 m（下方为正）
    "c_damp": 1e-4,        # 俯仰阻尼 N·m·s/rad
}

# 翅膀几何参数（从 DXF 分析获得，单位 m）
WING_FRONT = {
    "S": 16166.6e-6,       # 面积 m²
    "R": 154.3e-3,         # 半展长 m
    "c_avg": 104.8e-3,     # 平均弦长 m
    "r1": 0.4227,          # 面积一阶矩系数
    "r2_sq": 0.2382,       # 面积二阶矩系数
}

WING_BACK = {
    "S": 15537.6e-6,
    "R": 147.4e-3,
    "c_avg": 105.4e-3,
    "r1": 0.4798,
    "r2_sq": 0.2876,
}

# 气动参数
AERO = {
    "alpha_front_deg": 45.0,   # 前翅攻角
    "alpha_back_deg": 45.0,    # 后翅攻角
    "k_3d": 0.7,               # 三维修正
}


def cl_cd(alpha_deg):
    """升阻力系数（alpha 单位：度）"""
    C_L = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
    C_D = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))
    return C_L, C_D


def get_wing_state(t, f, phase, a, phi_offset_deg, params_dict):
    """获取单个翅膀在时刻 t 的状态（考虑相位偏移）"""
    # 周期
    T = 1.0 / f
    # 将时间映射到一个周期内，加上相位
    t_eff = (t + phase / (2 * np.pi * f)) % T

    # 获取一个周期的运动学（用较多点保证精度）
    t_cycle, phi, phi_dot, phi_ddot, info = wing_kinematics(
        f=f, a=a, phi_offset_deg=phi_offset_deg,
        n_points=2000
    )

    # 插值到当前时刻
    idx = np.argmin(np.abs(t_cycle - t_eff))
    # 更精确的插值
    if t_eff <= t_cycle[0]:
        idx = 0
    elif t_eff >= t_cycle[-1]:
        idx = len(t_cycle) - 2
    else:
        idx = np.searchsorted(t_cycle, t_eff) - 1

    frac = (t_eff - t_cycle[idx]) / (t_cycle[idx+1] - t_cycle[idx])
    frac = np.clip(frac, 0, 1)

    phi_t = phi[idx] + frac * (phi[idx+1] - phi[idx])
    phi_dot_t = phi_dot[idx] + frac * (phi_dot[idx+1] - phi_dot[idx])

    return phi_t, phi_dot_t


def compute_forces(phi, phi_dot, theta_p, theta_dot, alpha_deg, wing):
    """
    计算单翅气动力（Fx, Fz）在惯性系中的分量。

    翅膀绝对角度 ψ = φ + θ_p（φ 为相对身体的拍动角）
    绝对角速度 Ω = φ_dot + θ_dot

    升力和阻力基于准定常模型，方向考虑翅膀姿态。
    """
    psi = phi + theta_p
    Omega = phi_dot + theta_dot

    if abs(Omega) < 1e-6:
        return 0.0, 0.0

    U = abs(Omega) * wing["R"]

    # 攻角方向：基于绝对角速度方向
    if Omega <= 0:
        # 绝对运动方向：顺时针（翅膀向下/向后）
        C_L, C_D = cl_cd(alpha_deg)
    else:
        # 绝对运动方向：逆时针（翅膀向上/向前）
        C_L, C_D = cl_cd(-alpha_deg)

    # 气动力大小（含 3D 修正）
    const = 0.5 * PHYS["rho"] * U**2 * wing["S"] * wing["r2_sq"] * AERO["k_3d"]

    sign_Omega = -1 if Omega <= 0 else 1

    # Fx: 水平方向（向前为正）
    # Fz: 垂直方向（向上为正）
    Fx = const * np.sin(psi) * (sign_Omega * C_D - C_L)
    Fz = const * np.cos(psi) * (C_L - sign_Omega * C_D)

    return Fx, Fz


def pitch_dynamics(t, state, front_params, back_params):
    """
    俯仰动力学方程。
    state = [theta_p, theta_dot]
    """
    theta_p, theta_dot = state

    # 获取前后翅状态
    phi_f, phi_dot_f = get_wing_state(
        t, front_params["f"], front_params["phase"],
        front_params["a"], front_params["phi_offset_deg"], front_params
    )
    phi_b, phi_dot_b = get_wing_state(
        t, back_params["f"], back_params["phase"],
        back_params["a"], back_params["phi_offset_deg"], back_params
    )

    # 计算四翅气动力（左右对称，乘 2）
    Fx_f, Fz_f = compute_forces(
        phi_f, phi_dot_f, theta_p, theta_dot,
        front_params["alpha_deg"], WING_FRONT
    )
    Fx_b, Fz_b = compute_forces(
        phi_b, phi_dot_b, theta_p, theta_dot,
        back_params["alpha_deg"], WING_BACK
    )

    # 四翅合力
    Fx_total = 2 * (Fx_f + Fx_b)
    Fz_total = 2 * (Fz_f + Fz_b)

    # 俯仰力矩（前后翅升力差异）
    M_aero = 2 * (Fz_f * PHYS["x_front"] + Fz_b * PHYS["x_back"])

    # 重力恢复力矩（重心在俯仰轴下方）
    M_gravity = -PHYS["m_total"] * PHYS["g"] * PHYS["d_cg"] * np.sin(theta_p)

    # 阻尼力矩
    M_damp = -PHYS["c_damp"] * theta_dot

    # 俯仰角加速度
    theta_ddot = (M_aero + M_gravity + M_damp) / PHYS["I_yy"]

    return [theta_dot, theta_ddot]


def simulate(front_params, back_params, t_span=(0, 2.0), n_points=4000):
    """仿真多个周期，返回时间序列"""
    # 初始状态：[theta_p, theta_dot]
    y0 = [0.0, 0.0]

    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(
        lambda t, y: pitch_dynamics(t, y, front_params, back_params),
        t_span, y0, t_eval=t_eval, method="RK45", max_step=1e-4
    )

    return sol


def analyze_simulation(sol, front_params, back_params):
    """分析仿真结果，计算平均升力、推力、俯仰角等"""
    t = sol.t
    theta_p = sol.y[0]
    theta_dot = sol.y[1]

    # 重新计算各时刻的力（避免存储中间结果）
    n = len(t)
    Fx_total = np.zeros(n)
    Fz_total = np.zeros(n)
    M_aero_arr = np.zeros(n)

    for i in range(n):
        phi_f, phi_dot_f = get_wing_state(
            t[i], front_params["f"], front_params["phase"],
            front_params["a"], front_params["phi_offset_deg"], front_params
        )
        phi_b, phi_dot_b = get_wing_state(
            t[i], back_params["f"], back_params["phase"],
            back_params["a"], back_params["phi_offset_deg"], back_params
        )

        Fx_f, Fz_f = compute_forces(
            phi_f, phi_dot_f, theta_p[i], theta_dot[i],
            front_params["alpha_deg"], WING_FRONT
        )
        Fx_b, Fz_b = compute_forces(
            phi_b, phi_dot_b, theta_p[i], theta_dot[i],
            back_params["alpha_deg"], WING_BACK
        )

        Fx_total[i] = 2 * (Fx_f + Fx_b)
        Fz_total[i] = 2 * (Fz_f + Fz_b)
        M_aero_arr[i] = 2 * (Fz_f * PHYS["x_front"] + Fz_b * PHYS["x_back"])

    # 只取稳态（后半段）
    steady = slice(n // 2, n)

    results = {
        "t": t,
        "theta_p_deg": np.degrees(theta_p),
        "theta_dot": theta_dot,
        "Fx_total": Fx_total,
        "Fz_total": Fz_total,
        "M_aero": M_aero_arr,
        "avg_Fx_mN": np.mean(Fx_total[steady]) * 1000,
        "avg_Fz_mN": np.mean(Fz_total[steady]) * 1000,
        "avg_M_aero_uNm": np.mean(M_aero_arr[steady]) * 1e6,
        "max_theta_p_deg": np.max(np.abs(theta_p[steady])) * 180 / np.pi,
        "weight_mN": PHYS["m_total"] * PHYS["g"] * 1000,
    }

    return results


def plot_results(results, title, filename):
    """绘制结果"""
    t = results["t"]
    t_ms = t * 1000

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Row 0: 俯仰角和俯仰角速度
    ax = axes[0, 0]
    ax.plot(t_ms, results["theta_p_deg"], "b-", lw=1.5)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Pitch angle (deg)")
    ax.set_title("Body Pitch Angle")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_ms, results["theta_dot"], "r-", lw=1.5)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Pitch rate (rad/s)")
    ax.set_title("Body Pitch Rate")
    ax.grid(True, alpha=0.3)

    # Row 1: 升力和推力
    ax = axes[1, 0]
    ax.plot(t_ms, results["Fz_total"] * 1000, "g-", lw=1.5, label="Lift")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axhline(results["avg_Fz_mN"], color="g", ls="--", lw=1, alpha=0.7, label=f"Avg={results['avg_Fz_mN']:.2f} mN")
    ax.axhline(results["weight_mN"], color="r", ls=":", lw=1, alpha=0.7, label=f"Weight={results['weight_mN']:.1f} mN")
    ax.set_ylabel("Vertical force (mN)")
    ax.set_title("Total Lift (4 wings)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_ms, results["Fx_total"] * 1000, "m-", lw=1.5, label="Thrust")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axhline(results["avg_Fx_mN"], color="m", ls="--", lw=1, alpha=0.7, label=f"Avg={results['avg_Fx_mN']:.2f} mN")
    ax.set_ylabel("Horizontal force (mN)")
    ax.set_title("Total Thrust (4 wings)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: 力矩和 theta_p vs 升力
    ax = axes[2, 0]
    ax.plot(t_ms, results["M_aero"] * 1e6, "orange", lw=1.5)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axhline(results["avg_M_aero_uNm"], color="orange", ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("Aero moment (uN·m)")
    ax.set_xlabel("Time (ms)")
    ax.set_title("Pitch Moment (aero)")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(results["theta_p_deg"], results["Fz_total"] * 1000, ".", ms=2, alpha=0.5, color="purple")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Pitch angle (deg)")
    ax.set_ylabel("Lift (mN)")
    ax.set_title("Lift vs Pitch Angle")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def main():
    print("=" * 70)
    print("俯仰动力学验证")
    print("=" * 70)

    # 基础参数
    base_front = {
        "f": 15.0,
        "phase": 0.0,
        "a": 7.92,
        "phi_offset_deg": -50.84,
        "alpha_deg": 45.0,
    }
    base_back = {
        "f": 15.0,
        "phase": 0.0,
        "a": 7.92,
        "phi_offset_deg": -50.84,
        "alpha_deg": 45.0,
    }

    # 测试用例
    cases = []

    # Case 1: 无相位差（基准）
    f1, b1 = base_front.copy(), base_back.copy()
    cases.append(("Case 1: No phase diff (baseline)", f1, b1, "temp/case1_baseline.png"))

    # Case 2: 前后翅反相（180°）
    f2, b2 = base_front.copy(), base_back.copy()
    b2["phase"] = np.pi
    cases.append(("Case 2: Front-Back anti-phase (180 deg)", f2, b2, "temp/case2_antiphase.png"))

    # Case 3: 反相 + 前翅大攻角
    f3, b3 = base_front.copy(), base_back.copy()
    b3["phase"] = np.pi
    f3["alpha_deg"] = 85.0
    b3["alpha_deg"] = 85.0
    cases.append(("Case 3: Anti-phase + alpha=85 deg", f3, b3, "temp/case3_antiphase_alpha85.png"))

    # Case 4: 反相 + 不同 a
    f4, b4 = base_front.copy(), base_back.copy()
    b4["phase"] = np.pi
    f4["a"] = 10.0
    b4["a"] = 7.92
    cases.append(("Case 4: Anti-phase + diff a", f4, b4, "temp/case4_antiphase_diffa.png"))

    # Case 5: 反相 + 不同频率
    f5, b5 = base_front.copy(), base_back.copy()
    b5["phase"] = np.pi
    f5["f"] = 15.0
    b5["f"] = 15.5  # 微小频率差
    cases.append(("Case 5: Anti-phase + diff freq", f5, b5, "temp/case5_antiphase_difff.png"))

    # Case 6: 反相 + 85° + 高频率
    f6, b6 = base_front.copy(), base_back.copy()
    b6["phase"] = np.pi
    f6["alpha_deg"] = 85.0
    b6["alpha_deg"] = 85.0
    f6["f"] = 25.0
    b6["f"] = 25.0
    cases.append(("Case 6: Anti-phase + alpha=85 + f=25Hz", f6, b6, "temp/case6_highfreq.png"))

    all_results = []

    for title, fp, bp, filename in cases:
        print(f"\n--- {title} ---")
        sol = simulate(fp, bp, t_span=(0, 3.0), n_points=6000)
        res = analyze_simulation(sol, fp, bp)
        plot_results(res, title, filename)

        print(f"  Avg Lift: {res['avg_Fz_mN']:+.2f} mN (weight: {res['weight_mN']:.1f} mN)")
        print(f"  Avg Thrust: {res['avg_Fx_mN']:+.2f} mN")
        print(f"  Max |Pitch|: {res['max_theta_p_deg']:.2f} deg")
        print(f"  Lift/Weight: {res['avg_Fz_mN']/res['weight_mN']:.3f}")

        all_results.append((title, res))

    # 汇总
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Case':<40} {'Lift(mN)':>10} {'Thrust(mN)':>12} {'L/W':>8} {'|Pitch|max':>12}")
    print("-" * 90)
    for title, res in all_results:
        short = title.split(":")[0]
        print(f"{short:<40} {res['avg_Fz_mN']:>+10.2f} {res['avg_Fx_mN']:>+12.2f} "
              f"{res['avg_Fz_mN']/res['weight_mN']:>8.3f} {res['max_theta_p_deg']:>11.2f} deg")

    print("\nDone!")


if __name__ == "__main__":
    main()
