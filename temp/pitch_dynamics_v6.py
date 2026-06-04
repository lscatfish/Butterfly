#!/usr/bin/env python3
"""
俯仰动力学验证 v6：非对称刚性安装角 + 前飞速度 + 高攻角模型 + 四分量力

v5 → v6 核心改动：
1. 刚性翅膀模型：攻角由 α_install + ψ 实时计算（非恒定）
2. 前后翅独立 α_install（支持非对称扫描）
3. 前飞速度 Vx：状态 [θ_p, θ̇_p, x, Vx]，由 Fx 驱动加速
4. 高攻角扩展：|α| ≤ 60° 使用经验公式，60-90° 使用 flat-plate 过渡，
   90-180° 使用物理对称性
5. 攻角始终随 pitch 变化 → 俯仰恢复机制自然运作

简化（已知限制，后续逐步放松）：
- 小前飞速度下忽略 Vx 对相对来流方向的修正
  （Vx 增长到与拍动速度相当时需升级为向量合成 v_rel = v_flap + v_body）
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
    "alpha_max_deg": 60.0,   # 经验公式可信上限
    "alpha_stall_deg": 45.0,  # 失速起始角
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


# ==================== 气动力系数（含高攻角扩展） ====================

def cl_cd_empirical(alpha_deg):
    """
    经验升阻力系数（|α| ≤ 60° 可信）。
    来源：文献 [11][26]，dm 级扑翼实验拟合。
    """
    C_L = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
    C_D = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))
    return C_L, C_D


def cl_cd_plate(alpha_deg):
    """
    平板失速后模型（|α| > 45° 的物理极限行为）。

    薄平板在大攻角下的法向力系数：
        C_N ≈ C_N90 * sin(α)
    其中 C_N90 ≈ 2.0 是平板垂直于来流时的法向力系数（低 Re）。

    升力 = C_N*cos(α)、阻力 = C_N*sin(α)（忽略表面摩擦力）：
        C_L = C_N90 * sin(α) * cos(α) = (C_N90/2) * sin(2α)
        C_D = C_N90 * sin²(α)

    此模型在高攻角 (|α| > 60°) 提供物理正确的极限行为：
    - α = 90°: C_L ≈ 0, C_D ≈ C_N90 ≈ 2.0
    - α = 120°: C_L 为负, C_D > 0（翅膀反面迎风）
    """
    C_N90 = 2.0  # 平板在 90° 的法向力系数
    alpha_rad = np.deg2rad(alpha_deg)
    C_L = (C_N90 / 2.0) * np.sin(2.0 * alpha_rad)
    C_D = C_N90 * np.sin(alpha_rad)**2
    return C_L, C_D


def cl_cd_blended(alpha_deg):
    """
    混合模型：低攻角用经验公式，高攻角平滑过渡到平板模型。

    过渡区间：[40°, 70°]
    - |α| ≤ 40°: 100% 经验公式（LEV 效应可信）
    - 40° < |α| ≤ 70°: 经验公式 → 平板模型 平滑过渡
    - |α| > 70°: 平板模型主导（失速后物理）

    过渡使用 smoothstep（三次 Hermite 插值）避免梯度不连续。
    """
    abs_a = np.abs(alpha_deg)

    # 过渡权重：0 = 纯经验, 1 = 纯平板
    lo, hi = 40.0, 70.0
    t = np.clip((abs_a - lo) / (hi - lo), 0.0, 1.0)
    # smoothstep: 3t² - 2t³
    w = 3.0 * t**2 - 2.0 * t**3

    cl_e, cd_e = cl_cd_empirical(alpha_deg)
    cl_p, cd_p = cl_cd_plate(alpha_deg)

    C_L = (1.0 - w) * cl_e + w * cl_p
    C_D = (1.0 - w) * cd_e + w * cd_p
    return C_L, C_D


def precompute_kinematics(f, a, phi_offset_deg, n_points=2000):
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
    t_eff = np.mod(t_arr + phase / (2 * np.pi * T), T)
    phi = np.interp(t_eff, t_ext, phi_ext)
    phi_dot = np.interp(t_eff, t_ext, phi_dot_ext)
    phi_ddot = np.interp(t_eff, t_ext, phi_ddot_ext)
    return phi, phi_dot, phi_ddot


def compute_forces(phi, phi_dot, phi_ddot, theta_p, theta_dot,
                   alpha_install, wing, rho=1.225):
    """
    刚性翅膀气动力计算。

    Parameters
    ----------
    alpha_install : float
        翅膀安装角 [deg]，弦线相对机身水平面的夹角。出厂设定。
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

    # ---- 几何攻角计算 ----
    alpha_geom_rad = np.deg2rad(alpha_install) + psi
    mask_down = phi_dot <= 0
    alpha_eff_rad = np.where(mask_down, alpha_geom_rad, -alpha_geom_rad)
    alpha_eff_deg = np.rad2deg(alpha_eff_rad)

    # ---- 使用混合模型（含高攻角扩展） ----
    C_L, C_D = cl_cd_blended(alpha_eff_deg)

    # ---- 1. 平动分量 ----
    L_trans = const * C_L
    D_trans = const * C_D

    # ---- 2. 附加质量力 ----
    # 使用有效攻角的 sin（方向垂直于翅膀平面）
    F_AM = -(rho * np.pi * c_avg**2 / 4.0) * phi_ddot * R * r1 * np.sin(alpha_eff_rad)

    # ---- 3. 旋转力（alpha_dot = 0，暂无扭转） ----
    alpha_dot = np.zeros_like(phi)
    F_rot = rho * AERO["C_rot"] * alpha_dot * phi_dot * c_avg**2 * R * AERO["r_rot"]

    # ---- 4. Clap-and-Fling ----
    phi_dot_peak = np.max(np.abs(phi_dot))
    reversal_threshold = 0.1 * phi_dot_peak
    in_reversal = np.abs(phi_dot) < reversal_threshold
    k_clap = np.where(in_reversal, AERO["k_clap"], 1.0)

    # ---- 5. 总有效力 ----
    L_eff = (L_trans + F_AM + F_rot) * k_clap
    D_eff = D_trans * k_clap

    # ---- 6. 投影到机体坐标系 ----
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
        "C_L": C_L, "C_D": C_D, "alpha_eff_deg": alpha_eff_deg,
        "alpha_geom_deg": np.rad2deg(alpha_geom_rad),
        "psi_deg": np.rad2deg(psi),
        "in_reversal": in_reversal, "k_clap": k_clap,
    }
    return F_body, M_body, info


# ==================== 固定点版本（仅俯仰） ====================

def compute_rhs_fixed(theta_p, theta_dot, phi_f, phi_dot_f, phi_ddot_f,
                      phi_b, phi_dot_b, phi_ddot_b, fp, bp):
    """RHS: [theta_dot, theta_ddot] — 固定点模型"""
    F_f, _, _ = compute_forces(
        phi_f, phi_dot_f, phi_ddot_f, theta_p, theta_dot,
        fp["alpha_install"], WING_FRONT
    )
    F_b, _, _ = compute_forces(
        phi_b, phi_dot_b, phi_ddot_b, theta_p, theta_dot,
        bp["alpha_install"], WING_BACK
    )

    Fx_total = 2 * (F_f[:, 0] + F_b[:, 0])
    Fz_total = 2 * (F_f[:, 2] + F_b[:, 2])
    M_aero = 2 * (-PHYS["x_front"] * F_f[:, 2] - PHYS["x_back"] * F_b[:, 2])

    M_gravity = -PHYS["m_total"] * PHYS["g"] * PHYS["d_cg"] * np.sin(theta_p)
    M_damp = -PHYS["c_damp"] * theta_dot
    theta_ddot = (M_aero + M_gravity + M_damp) / PHYS["I_yy"]

    if np.ndim(theta_ddot) > 0:
        theta_ddot = float(theta_ddot[0])
        Fx_total = float(Fx_total[0])
        Fz_total = float(Fz_total[0])
        M_aero = float(M_aero[0])
    return float(theta_dot), float(theta_ddot), Fx_total, 0.0, Fz_total, M_aero


# ==================== 移动版本（俯仰 + 前飞） ====================

def compute_rhs_moving(theta_p, theta_dot, Vx,
                       phi_f, phi_dot_f, phi_ddot_f,
                       phi_b, phi_dot_b, phi_ddot_b, fp, bp):
    """RHS: [theta_dot, theta_ddot, Vx_dot] — 移动模型"""
    F_f, _, _ = compute_forces(
        phi_f, phi_dot_f, phi_ddot_f, theta_p, theta_dot,
        fp["alpha_install"], WING_FRONT
    )
    F_b, _, _ = compute_forces(
        phi_b, phi_dot_b, phi_ddot_b, theta_p, theta_dot,
        bp["alpha_install"], WING_BACK
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


def rk4_step_fixed(y, dt, *args):
    theta_p, theta_dot = y
    k1_td, k1_tdd, _, _, _, _ = compute_rhs_fixed(theta_p, theta_dot, *args)
    k2_td, k2_tdd, _, _, _, _ = compute_rhs_fixed(
        theta_p + 0.5*dt*k1_td, theta_dot + 0.5*dt*k1_tdd, *args)
    k3_td, k3_tdd, _, _, _, _ = compute_rhs_fixed(
        theta_p + 0.5*dt*k2_td, theta_dot + 0.5*dt*k2_tdd, *args)
    k4_td, k4_tdd, _, _, _, _ = compute_rhs_fixed(
        theta_p + dt*k3_td, theta_dot + dt*k3_tdd, *args)
    theta_p_new = theta_p + dt*(k1_td + 2*k2_td + 2*k3_td + k4_td)/6
    theta_dot_new = theta_dot + dt*(k1_tdd + 2*k2_tdd + 2*k3_tdd + k4_tdd)/6
    return theta_p_new, theta_dot_new


def rk4_step_moving(y, dt, *args):
    theta_p, theta_dot, x, Vx = y
    k1_tp, k1_td, k1_vxd, _, _, _ = compute_rhs_moving(
        theta_p, theta_dot, Vx, *args)
    k2_tp, k2_td, k2_vxd, _, _, _ = compute_rhs_moving(
        theta_p + 0.5*dt*k1_tp, theta_dot + 0.5*dt*k1_td, Vx + 0.5*dt*k1_vxd, *args)
    k3_tp, k3_td, k3_vxd, _, _, _ = compute_rhs_moving(
        theta_p + 0.5*dt*k2_tp, theta_dot + 0.5*dt*k2_td, Vx + 0.5*dt*k2_vxd, *args)
    k4_tp, k4_td, k4_vxd, _, _, _ = compute_rhs_moving(
        theta_p + dt*k3_tp, theta_dot + dt*k3_td, Vx + dt*k3_vxd, *args)
    theta_p_new = theta_p + dt*(k1_tp + 2*k2_tp + 2*k3_tp + k4_tp)/6
    theta_dot_new = theta_dot + dt*(k1_td + 2*k2_td + 2*k3_td + k4_td)/6
    x_new = x + dt*Vx + dt**2*(k1_vxd + 2*k2_vxd + 2*k3_vxd + k4_vxd)/12
    Vx_new = Vx + dt*(k1_vxd + 2*k2_vxd + 2*k3_vxd + k4_vxd)/6
    return theta_p_new, theta_dot_new, x_new, Vx_new


# ==================== 运行器 ====================

def run_case(title, fp, bp, filename, moving=False, t_end=2.0, n_steps=40000,
             theta0_deg=0.0):
    print(f"\nPrecomputing: {title} ...")
    mode = "moving" if moving else "fixed"

    t_ext_f, phi_ext_f, phi_dot_ext_f, phi_ddot_ext_f, T_f, _ = precompute_kinematics(
        fp["f"], fp["a"], fp["phi_offset_deg"])
    t_ext_b, phi_ext_b, phi_dot_ext_b, phi_ddot_ext_b, T_b, _ = precompute_kinematics(
        bp["f"], bp["a"], bp["phi_offset_deg"])

    dt = t_end / n_steps
    print(f"  [{mode}] dt={dt*1e6:.1f} us, n={n_steps}, θ₀={theta0_deg}°")

    t = np.linspace(0, t_end, n_steps)
    phi_f, phi_dot_f, phi_ddot_f = get_states(
        t, t_ext_f, phi_ext_f, phi_dot_ext_f, phi_ddot_ext_f, fp["phase"], T_f)
    phi_b, phi_dot_b, phi_ddot_b = get_states(
        t, t_ext_b, phi_ext_b, phi_dot_ext_b, phi_ddot_ext_b, bp["phase"], T_b)

    theta_p = np.zeros(n_steps); theta_p[0] = np.deg2rad(theta0_deg)
    theta_dot = np.zeros(n_steps)
    Fx_total = np.zeros(n_steps); Fz_total = np.zeros(n_steps)
    M_aero = np.zeros(n_steps)

    if moving:
        x_pos = np.zeros(n_steps)
        Vx_arr = np.zeros(n_steps)
        for i in range(n_steps - 1):
            _, _, _, Fx_t, Fz_t, M_t = compute_rhs_moving(
                theta_p[i], theta_dot[i], Vx_arr[i],
                phi_f[i:i+1], phi_dot_f[i:i+1], phi_ddot_f[i:i+1],
                phi_b[i:i+1], phi_dot_b[i:i+1], phi_ddot_b[i:i+1], fp, bp)
            Fx_total[i] = Fx_t; Fz_total[i] = Fz_t; M_aero[i] = M_t
            theta_p[i+1], theta_dot[i+1], x_pos[i+1], Vx_arr[i+1] = rk4_step_moving(
                [theta_p[i], theta_dot[i], x_pos[i], Vx_arr[i]], dt,
                phi_f[i:i+1], phi_dot_f[i:i+1], phi_ddot_f[i:i+1],
                phi_b[i:i+1], phi_dot_b[i:i+1], phi_ddot_b[i:i+1], fp, bp)
        _, _, _, Fx_t, Fz_t, M_t = compute_rhs_moving(
            theta_p[-1], theta_dot[-1], Vx_arr[-1],
            phi_f[-1:], phi_dot_f[-1:], phi_ddot_f[-1:],
            phi_b[-1:], phi_dot_b[-1:], phi_ddot_b[-1:], fp, bp)
        Fx_total[-1] = Fx_t; Fz_total[-1] = Fz_t; M_aero[-1] = M_t
    else:
        for i in range(n_steps - 1):
            _, _, Fx_t, _, Fz_t, M_t = compute_rhs_fixed(
                theta_p[i], theta_dot[i],
                phi_f[i:i+1], phi_dot_f[i:i+1], phi_ddot_f[i:i+1],
                phi_b[i:i+1], phi_dot_b[i:i+1], phi_ddot_b[i:i+1], fp, bp)
            Fx_total[i] = Fx_t; Fz_total[i] = Fz_t; M_aero[i] = M_t
            theta_p[i+1], theta_dot[i+1] = rk4_step_fixed(
                [theta_p[i], theta_dot[i]], dt,
                phi_f[i:i+1], phi_dot_f[i:i+1], phi_ddot_f[i:i+1],
                phi_b[i:i+1], phi_dot_b[i:i+1], phi_ddot_b[i:i+1], fp, bp)
        _, _, Fx_t, _, Fz_t, M_t = compute_rhs_fixed(
            theta_p[-1], theta_dot[-1],
            phi_f[-1:], phi_dot_f[-1:], phi_ddot_f[-1:],
            phi_b[-1:], phi_dot_b[-1:], phi_ddot_b[-1:], fp, bp)
        Fx_total[-1] = Fx_t; Fz_total[-1] = Fz_t; M_aero[-1] = M_t
        x_pos = None; Vx_arr = None

    steady = slice(n_steps // 2, n_steps)
    weight_mN = PHYS["m_total"] * PHYS["g"] * 1000

    res = {
        "title": title, "t": t, "t_ms": t * 1000,
        "theta_p_deg": np.degrees(theta_p), "theta_dot": theta_dot,
        "Fx_total_mN": Fx_total * 1000, "Fz_total_mN": Fz_total * 1000,
        "M_aero_uNm": M_aero * 1e6,
        "avg_Fx": np.mean(Fx_total[steady]) * 1000,
        "avg_Fz": np.mean(Fz_total[steady]) * 1000,
        "avg_M": np.mean(M_aero[steady]) * 1e6,
        "max_theta": np.max(np.abs(np.degrees(theta_p[steady]))),
        "weight": weight_mN,
        "ratio": np.mean(Fz_total[steady]) * 1000 / weight_mN,
        "moving": moving,
    }
    if moving:
        res["x_mm"] = x_pos * 1000
        res["Vx_m_s"] = Vx_arr
        res["max_Vx"] = np.max(np.abs(Vx_arr[steady]))
        res["final_x_mm"] = x_pos[-1] * 1000

    # ---- 绘图 ----
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(res["t_ms"], res["theta_p_deg"], "b-", lw=1.2)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Pitch (deg)")
    ax.set_title(f"Body Pitch | max={res['max_theta']:.1f} deg")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if moving:
        ax.plot(res["t_ms"], res["Vx_m_s"], "r-", lw=1.2)
        ax.set_ylabel("Vx (m/s)")
        ax.set_title(f"Forward Speed | max={res['max_Vx']:.2f} m/s, final_x={res['final_x_mm']:.0f}mm")
    else:
        ax.plot(res["t_ms"], theta_dot, "r-", lw=1.2)
        ax.set_ylabel("Pitch rate (rad/s)")
        ax.set_title("Pitch Rate")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(res["t_ms"], res["Fz_total_mN"], "g-", lw=1.2)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axhline(res["avg_Fz"], color="g", ls="--", alpha=0.7)
    ax.axhline(res["weight"], color="r", ls=":", alpha=0.7, label="Weight")
    ax.set_ylabel("Lift (mN)")
    ax.set_title(f"Lift | avg={res['avg_Fz']:+.2f} mN")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

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
    ax.set_ylabel("Moment (uN m)")
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Pitch Moment | avg={res['avg_M']:+.1f}")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    if moving:
        ax.plot(res["Vx_m_s"], res["theta_p_deg"], ".", ms=1.5, alpha=0.4, color="purple")
        ax.set_xlabel("Vx (m/s)")
        ax.set_title("Pitch vs Speed")
    else:
        ax.plot(res["theta_p_deg"], res["Fz_total_mN"], ".", ms=1.5, alpha=0.4, color="purple")
        ax.set_xlabel("Pitch angle (deg)")
        ax.set_title("Lift vs Pitch")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")
    return res


# ==================== 扫描 ====================

def scan_install_angles(moving=False):
    """
    扫描安装角组合：前翅 α_install_f × 后翅 α_install_b。
    同时跑同相和反相两种相位配置。
    """
    mode = "moving" if moving else "fixed"
    print(f"\n{'='*70}")
    print(f"安装角扫描 [{mode}]：前翅 × 后翅（含同相/反相）")
    print(f"{'='*70}")

    base = {"f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84}

    # 扫描范围
    ai_range = list(range(10, 65, 10))  # [10, 20, 30, 40, 50, 60]
    t_end = 2.0 if moving else 1.5
    n_steps = 40000 if moving else 30000

    results_in = []   # 同相
    results_anti = []  # 反相

    total = len(ai_range)**2 * 2
    cnt = 0

    for ai_f in ai_range:
        for ai_b in ai_range:
            # ---- 同相 ----
            cnt += 1
            fp = base.copy(); bp = base.copy()
            fp["alpha_install"] = ai_f; bp["alpha_install"] = ai_b
            title = f"α_f={ai_f}° α_b={ai_b}° [in]"
            fn = f"temp/v6_{'mv' if moving else 'fx'}_scan_in_{ai_f}_{ai_b}.png"
            res = run_case(title, fp, bp, fn, moving=moving, t_end=t_end, n_steps=n_steps)
            if res:
                results_in.append({
                    "ai_f": ai_f, "ai_b": ai_b,
                    "lift": res["avg_Fz"], "thrust": res["avg_Fx"],
                    "ratio": res["ratio"], "max_pitch": res["max_theta"],
                    "Vx_max": res.get("max_Vx", 0),
                })
                print(f"  [{cnt}/{total}] {title} | L={res['avg_Fz']:+.1f} T={res['avg_Fx']:+.1f} L/W={res['ratio']:.3f} θpk={res['max_theta']:.1f}°")

            # ---- 反相 180° ----
            cnt += 1
            fp2 = base.copy(); bp2 = base.copy()
            fp2["alpha_install"] = ai_f; bp2["alpha_install"] = ai_b
            bp2["phase"] = np.pi
            title2 = f"α_f={ai_f}° α_b={ai_b}° [anti]"
            fn2 = f"temp/v6_{'mv' if moving else 'fx'}_scan_anti_{ai_f}_{ai_b}.png"
            res2 = run_case(title2, fp2, bp2, fn2, moving=moving, t_end=t_end, n_steps=n_steps)
            if res2:
                results_anti.append({
                    "ai_f": ai_f, "ai_b": ai_b,
                    "lift": res2["avg_Fz"], "thrust": res2["avg_Fx"],
                    "ratio": res2["ratio"], "max_pitch": res2["max_theta"],
                    "Vx_max": res2.get("max_Vx", 0),
                })
                print(f"  [{cnt}/{total}] {title2} | L={res2['avg_Fz']:+.1f} T={res2['avg_Fx']:+.1f} L/W={res2['ratio']:.3f} θpk={res2['max_theta']:.1f}°")

    # ---- 排序输出 ----
    def print_top(label, results, n=15):
        print(f"\n{'='*70}")
        print(f"{label} — 按升重比排序 Top {n}")
        print(f"{'='*70}")
        results.sort(key=lambda x: x["ratio"], reverse=True)
        hdr = f"{'α_f':>5} {'α_b':>5} {'Lift':>10} {'Thrust':>10} {'L/W':>8} {'|Pitch|':>10}"
        if moving: hdr += f" {'Vx_max':>8}"
        print(hdr)
        for r in results[:n]:
            line = f"{r['ai_f']:5d}°{r['ai_b']:5d}° {r['lift']:>+10.1f} {r['thrust']:>+10.1f} {r['ratio']:>8.3f} {r['max_pitch']:>9.1f}°"
            if moving: line += f" {r['Vx_max']:>7.2f}"
            print(line)

    print_top("同相 (in-phase)", results_in)
    print_top("反相 (anti-phase)", results_anti)

    # 找两组都表现好的参数
    combined = []
    for ri in results_in:
        for ra in results_anti:
            if ri["ai_f"] == ra["ai_f"] and ri["ai_b"] == ra["ai_b"]:
                if ri["max_pitch"] < 30 and ra["max_pitch"] < 30:
                    combined.append({
                        "ai_f": ri["ai_f"], "ai_b": ri["ai_b"],
                        "ratio_in": ri["ratio"], "ratio_anti": ra["ratio"],
                        "lift_in": ri["lift"], "lift_anti": ra["lift"],
                        "pitch_in": ri["max_pitch"], "pitch_anti": ra["max_pitch"],
                    })
    if combined:
        combined.sort(key=lambda x: x["ratio_in"] + x["ratio_anti"], reverse=True)
        print(f"\n{'='*70}")
        print("两组都稳定 (|Pitch|<30°) 的参数")
        print(f"{'='*70}")
        print(f"{'α_f':>5} {'α_b':>5} {'L/W in':>10} {'L/W anti':>12} {'θ_in':>8} {'θ_anti':>8}")
        for c in combined[:10]:
            print(f"{c['ai_f']:5d}°{c['ai_b']:5d}° {c['ratio_in']:>10.3f} {c['ratio_anti']:>12.3f} {c['pitch_in']:>7.1f}° {c['pitch_anti']:>7.1f}°")

    return results_in, results_anti


# ==================== main ====================

def main():
    print("=" * 70)
    print("俯仰动力学验证 v6 — 非对称刚性安装角 + 前飞 + 高攻角扩展")
    print("=" * 70)
    print(f"模型: C_L/C_D blended (empirical 0-40° → plate 70-180°)")
    print(f"状态: fixed (仅俯仰) & moving (俯仰 + 前飞 Vx)")
    print()

    # ==== Phase 1: 固定点扫描 ====
    print("\n" + "█" * 70)
    print("PHASE 1: 固定点安装角扫描（仅俯仰自由度）")
    print("█" * 70)
    scan_fixed_in, scan_fixed_anti = scan_install_angles(moving=False)

    # ==== Phase 2: 移动模型扫描 ====
    print("\n" + "█" * 70)
    print("PHASE 2: 移动模型扫描（俯仰 + 前飞 Vx）")
    print("█" * 70)
    scan_move_in, scan_move_anti = scan_install_angles(moving=True)

    # ==== Phase 3: 关键 case 详细分析 ====
    print("\n" + "█" * 70)
    print("PHASE 3: 关键参数详细分析")
    print("█" * 70)

    base = {"f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84}

    # 选几组有代表性的参数
    key_configs = [
        # 对称大安装角
        {"label": "Sym large (60/60)", "ai_f": 60, "ai_b": 60},
        # 对称中等安装角
        {"label": "Sym medium (40/40)", "ai_f": 40, "ai_b": 40},
        # 前大后小（前翅产生更多升力）
        {"label": "Front>Back (60/30)", "ai_f": 60, "ai_b": 30},
        # 前小后大
        {"label": "Front<Back (30/60)", "ai_f": 30, "ai_b": 60},
        # 不对称中等
        {"label": "Asym (50/20)", "ai_f": 50, "ai_b": 20},
    ]

    for kc in key_configs:
        fp = base.copy(); bp = base.copy()
        fp["alpha_install"] = kc["ai_f"]
        bp["alpha_install"] = kc["ai_b"]

        # 固定点 + 同相
        res = run_case(f"Key: {kc['label']} [in, fixed]",
                       fp, bp, f"temp/v6_key_{kc['label'].replace(' ','_')}_in_fx.png",
                       moving=False, t_end=2.0, n_steps=40000)
        if res: print(f"  {kc['label']:25s} in/fixed:  L={res['avg_Fz']:+.1f} T={res['avg_Fx']:+.1f} L/W={res['ratio']:.3f} θ={res['max_theta']:.1f}°")

        # 固定点 + 反相
        bp2 = bp.copy(); bp2["phase"] = np.pi
        res = run_case(f"Key: {kc['label']} [anti, fixed]",
                       fp, bp2, f"temp/v6_key_{kc['label'].replace(' ','_')}_anti_fx.png",
                       moving=False, t_end=2.0, n_steps=40000)
        if res: print(f"  {kc['label']:25s} anti/fixed: L={res['avg_Fz']:+.1f} T={res['avg_Fx']:+.1f} L/W={res['ratio']:.3f} θ={res['max_theta']:.1f}°")

        # 移动 + 反相（最接近实际飞行）
        bp3 = bp.copy(); bp3["phase"] = np.pi
        res = run_case(f"Key: {kc['label']} [anti, moving]",
                       fp, bp3, f"temp/v6_key_{kc['label'].replace(' ','_')}_anti_mv.png",
                       moving=True, t_end=3.0, n_steps=60000)
        if res: print(f"  {kc['label']:25s} anti/move: L={res['avg_Fz']:+.1f} T={res['avg_Fx']:+.1f} L/W={res['ratio']:.3f} θ={res['max_theta']:.1f}° Vx={res.get('max_Vx',0):.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
