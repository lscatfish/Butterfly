#!/usr/bin/env python3
"""
俯仰动力学验证 v5-moving：水平移动 + 俯仰自由度 + 四分量准定常模型

在 v5-fixed 基础上引入水平前飞速度 Vx，从 0 开始。
状态向量：[theta_p, theta_dot, x, Vx]

简化假设：
- 前飞速度 Vx 初始为 0，由气动力 Fx 驱动加速
- 气动力计算仍采用固定点模型（忽略前飞速度对相对速度方向的修正）
  这是小前飞速度下的合理近似（悬停/低速过渡）。
  严格处理需要向量合成相对速度：v_rel = v_flap + v_body，后续可升级。
- 垂直方向固定（假设高度控制或简化为二维运动）

坐标系：同 v5-fixed（惯性系/机体坐标系对齐，无平移时）
当蝴蝶移动后，机体坐标系原点在空间中移动，但气动力仍在机体坐标系中计算。
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
    "alpha_max_deg": 60.0,
}


def load_geometry():
    json_path = Path(__file__).parent.parent / "data" / "wing_analysis_results.json"
    with open(json_path) as f:
        data = json.load(f)
    geo = {}
    for g in data["geometry"]:
        geo[g["name"]] = {
            "S": g["S"], "R": g["R"], "c_avg": g["c_avg"],
            "r1": g["r1"], "r2_sq": g["r2_sq"], "AR": g["AR"],
        }
    return geo


GEO = load_geometry()
WING_FRONT = GEO["Front"]
WING_BACK = GEO["Back"]


def cl_cd(alpha_deg):
    C_L = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
    C_D = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))
    return C_L, C_D


def clamp_alpha(alpha):
    return np.clip(alpha, -AERO["alpha_max_deg"], AERO["alpha_max_deg"])


def precompute_kinematics(f, a, phi_offset_deg, n_points=2000):
    t, phi, phi_dot, phi_ddot, info = wing_kinematics(
        f=f, a=a, phi_offset_deg=phi_offset_deg, n_points=n_points
    )
    T = 1.0 / f
    t_ext = np.concatenate([[t[-1] - T], t, [t[0] + T]])
    phi_ext = np.concatenate([[phi[-1]], phi, [phi[0]]])
    phi_dot_ext = np.concatenate([[phi_dot[-1]], phi_dot, [phi[0]]])
    phi_ddot_ext = np.concatenate([[phi_ddot[-1]], phi_ddot, [phi[0]]])
    return t_ext, phi_ext, phi_dot_ext, phi_ddot_ext, T


def get_states(t_arr, t_ext, phi_ext, phi_dot_ext, phi_ddot_ext, phase, T):
    t_eff = np.mod(t_arr + phase / (2 * np.pi * T), T)
    phi = np.interp(t_eff, t_ext, phi_ext)
    phi_dot = np.interp(t_eff, t_ext, phi_dot_ext)
    phi_ddot = np.interp(t_eff, t_ext, phi_ddot_ext)
    return phi, phi_dot, phi_ddot


def compute_forces(phi, phi_dot, phi_ddot, theta_p, theta_dot,
                   alpha_down, alpha_up, wing, rho=1.225):
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

    L_trans = const * C_L
    D_trans = const * C_D

    alpha_rad = np.deg2rad(alpha_eff)
    F_AM = -(rho * np.pi * c_avg**2 / 4.0) * phi_ddot * R * r1 * np.sin(alpha_rad)

    alpha_dot = np.zeros_like(phi)
    F_rot = rho * AERO["C_rot"] * alpha_dot * phi_dot * c_avg**2 * R * AERO["r_rot"]

    phi_dot_peak = np.max(np.abs(phi_dot))
    reversal_threshold = 0.1 * phi_dot_peak
    in_reversal = np.abs(phi_dot) < reversal_threshold
    k_clap = np.where(in_reversal, AERO["k_clap"], 1.0)

    L_eff = (L_trans + F_AM + F_rot) * k_clap
    D_eff = D_trans * k_clap

    Fx = np.sin(psi) * (sign_Omega * D_eff - L_eff)
    Fz = np.cos(psi) * (L_eff - sign_Omega * D_eff)

    mask_still = np.abs(Omega) < 1e-6
    Fx = np.where(mask_still, 0, Fx)
    Fz = np.where(mask_still, 0, Fz)

    F_body = np.zeros((n, 3))
    F_body[:, 0] = Fx
    F_body[:, 2] = Fz

    M_body = np.zeros((n, 3))

    info = {
        "L_trans": L_trans, "D_trans": D_trans, "F_AM": F_AM,
        "L_eff": L_eff, "D_eff": D_eff,
        "C_L": C_L, "C_D": C_D, "alpha_eff": alpha_eff,
        "in_reversal": in_reversal, "k_clap": k_clap,
    }
    return F_body, M_body, info


def compute_rhs(theta_p, theta_dot, phi_f, phi_dot_f, phi_ddot_f,
                phi_b, phi_dot_b, phi_ddot_b, fp, bp):
    """计算 RHS: [theta_dot, theta_ddot, Vx, Vx_dot]"""
    F_f, _, _ = compute_forces(
        phi_f, phi_dot_f, phi_ddot_f, theta_p, theta_dot,
        fp["alpha_down"], fp["alpha_up"], WING_FRONT
    )
    F_b, _, _ = compute_forces(
        phi_b, phi_dot_b, phi_ddot_b, theta_p, theta_dot,
        bp["alpha_down"], bp["alpha_up"], WING_BACK
    )

    Fx_total = 2 * (F_f[:, 0] + F_b[:, 0])
    Fz_total = 2 * (F_f[:, 2] + F_b[:, 2])
    M_aero = 2 * (-PHYS["x_front"] * F_f[:, 2] - PHYS["x_back"] * F_b[:, 2])

    M_gravity = -PHYS["m_total"] * PHYS["g"] * PHYS["d_cg"] * np.sin(theta_p)
    M_damp = -PHYS["c_damp"] * theta_dot

    theta_ddot = (M_aero + M_gravity + M_damp) / PHYS["I_yy"]
    Vx_dot = Fx_total / PHYS["m_total"]

    if np.ndim(theta_ddot) > 0:
        theta_ddot = float(theta_ddot[0])
        Vx_dot = float(Vx_dot[0])
        Fx_total = float(Fx_total[0])
        Fz_total = float(Fz_total[0])
        M_aero = float(M_aero[0])

    return float(theta_dot), float(theta_ddot), float(Vx_dot), Fx_total, Fz_total, M_aero


def rk4_step(y, dt, *args):
    """单步 RK4，状态 [theta_p, theta_dot, x, Vx]"""
    theta_p, theta_dot, x, Vx = y

    k1_tp, k1_td, k1_vxd, _, _, _ = compute_rhs(theta_p, theta_dot, *args)
    k2_tp, k2_td, k2_vxd, _, _, _ = compute_rhs(
        theta_p + 0.5*dt*k1_tp, theta_dot + 0.5*dt*k1_td, *args
    )
    k3_tp, k3_td, k3_vxd, _, _, _ = compute_rhs(
        theta_p + 0.5*dt*k2_tp, theta_dot + 0.5*dt*k2_td, *args
    )
    k4_tp, k4_td, k4_vxd, _, _, _ = compute_rhs(
        theta_p + dt*k3_tp, theta_dot + dt*k3_td, *args
    )

    theta_p_new = theta_p + dt * (k1_tp + 2*k2_tp + 2*k3_tp + k4_tp) / 6
    theta_dot_new = theta_dot + dt * (k1_td + 2*k2_td + 2*k3_td + k4_td) / 6
    x_new = x + dt * Vx + dt**2 * (k1_vxd + 2*k2_vxd + 2*k3_vxd + k4_vxd) / 12
    Vx_new = Vx + dt * (k1_vxd + 2*k2_vxd + 2*k3_vxd + k4_vxd) / 6

    return theta_p_new, theta_dot_new, x_new, Vx_new


def run_case(title, fp, bp, filename, t_end=3.0, n_steps=60000):
    print(f"\nPrecomputing: {title} ...")

    t_ext_f, phi_ext_f, phi_dot_ext_f, phi_ddot_ext_f, T_f = precompute_kinematics(
        fp["f"], fp["a"], fp["phi_offset_deg"]
    )
    t_ext_b, phi_ext_b, phi_dot_ext_b, phi_ddot_ext_b, T_b = precompute_kinematics(
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
    x_pos = np.zeros(n_steps)
    Vx = np.zeros(n_steps)
    Fx_total = np.zeros(n_steps)
    Fz_total = np.zeros(n_steps)
    M_aero = np.zeros(n_steps)

    for i in range(n_steps - 1):
        _, _, _, Fx_t, Fz_t, M_t = compute_rhs(
            theta_p[i], theta_dot[i],
            phi_f[i:i+1], phi_dot_f[i:i+1], phi_ddot_f[i:i+1],
            phi_b[i:i+1], phi_dot_b[i:i+1], phi_ddot_b[i:i+1],
            fp, bp
        )
        Fx_total[i] = Fx_t
        Fz_total[i] = Fz_t
        M_aero[i] = M_t

        theta_p[i+1], theta_dot[i+1], x_pos[i+1], Vx[i+1] = rk4_step(
            [theta_p[i], theta_dot[i], x_pos[i], Vx[i]], dt,
            phi_f[i:i+1], phi_dot_f[i:i+1], phi_ddot_f[i:i+1],
            phi_b[i:i+1], phi_dot_b[i:i+1], phi_ddot_b[i:i+1],
            fp, bp
        )

    _, _, _, Fx_t, Fz_t, M_t = compute_rhs(
        theta_p[-1], theta_dot[-1],
        phi_f[-1:], phi_dot_f[-1:], phi_ddot_f[-1:],
        phi_b[-1:], phi_dot_b[-1:], phi_ddot_b[-1:],
        fp, bp
    )
    Fx_total[-1] = Fx_t
    Fz_total[-1] = Fz_t
    M_aero[-1] = M_t

    steady = slice(n_steps // 2, n_steps)
    weight_mN = PHYS["m_total"] * PHYS["g"] * 1000

    res = {
        "title": title,
        "t": t, "t_ms": t * 1000,
        "theta_p_deg": np.degrees(theta_p),
        "x_mm": x_pos * 1000,
        "Vx_m_s": Vx,
        "Fx_total_mN": Fx_total * 1000,
        "Fz_total_mN": Fz_total * 1000,
        "M_aero_uNm": M_aero * 1e6,
        "avg_Fx": np.mean(Fx_total[steady]) * 1000,
        "avg_Fz": np.mean(Fz_total[steady]) * 1000,
        "avg_M": np.mean(M_aero[steady]) * 1e6,
        "max_theta": np.max(np.abs(np.degrees(theta_p[steady]))),
        "max_Vx": np.max(np.abs(Vx[steady])),
        "final_x_mm": x_pos[-1] * 1000,
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
    ax.plot(res["t_ms"], res["Vx_m_s"], "r-", lw=1.2)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Vx (m/s)")
    ax.set_title(f"Forward Speed | max={res['max_Vx']:.2f} m/s")
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
    ax.plot(res["x_mm"], res["theta_p_deg"], ".", ms=1.5, alpha=0.4, color="purple")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("Pitch (deg)")
    ax.set_title("Pitch vs Position")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")

    return res


def main():
    print("=" * 70)
    print("俯仰动力学验证 v5-moving (水平移动 + 俯仰 + 四分量模型)")
    print("=" * 70)

    # 使用从 v5-fixed 扫描得到的最佳参数（后续替换为实际最佳值）
    base = {
        "f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84,
        "alpha_down": 45.0, "alpha_up": -10.0
    }

    cases = []

    f1, b1 = base.copy(), base.copy()
    cases.append(("Case 1: Baseline", f1, b1, "temp/case1_v5_moving.png"))

    f2, b2 = base.copy(), base.copy()
    b2["phase"] = np.pi
    cases.append(("Case 2: Anti-phase 180", f2, b2, "temp/case2_v5_moving.png"))

    f3, b3 = base.copy(), base.copy()
    b3["phase"] = np.pi
    f3["f"] = 25.0
    b3["f"] = 25.0
    cases.append(("Case 3: Anti-phase f=25", f3, b3, "temp/case3_v5_moving.png"))

    f4, b4 = base.copy(), base.copy()
    b4["phase"] = np.pi / 2
    cases.append(("Case 4: 90° phase diff", f4, b4, "temp/case4_v5_moving.png"))

    results = []
    for title, fp, bp, fn in cases:
        res = run_case(title, fp, bp, fn, t_end=3.0, n_steps=60000)
        if res:
            results.append(res)
            print(f"  Lift: {res['avg_Fz']:+.2f} mN | Thrust: {res['avg_Fx']:+.2f} mN | "
                  f"L/W: {res['ratio']:.3f} | |Pitch|: {res['max_theta']:.1f}° | "
                  f"Vx_max: {res['max_Vx']:.2f} m/s | x_final: {res['final_x_mm']:.1f} mm")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Case':<35} {'Lift':>10} {'Thrust':>10} {'L/W':>8} {'|Pitch|':>8} {'Vx_max':>8} {'x_final':>10}")
    print("-" * 90)
    for r in results:
        short = r['title'].split(":")[0]
        print(f"{short:<35} {r['avg_Fz']:>+10.2f} {r['avg_Fx']:>+10.2f} {r['ratio']:>8.3f} "
              f"{r['max_theta']:>7.1f}° {r['max_Vx']:>7.2f} {r['final_x_mm']:>9.1f} mm")

    print("\nDone!")


if __name__ == "__main__":
    main()
