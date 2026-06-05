#!/usr/bin/env python3
"""
v6.1: 修复俯仰发散 — 增加气动俯仰阻尼 + 扫描净力矩平衡参数

根因：M_aero 时间均值不为零，重力恢复力矩不足以抵消 → 俯仰持续漂移。

修复策略：
1. 气动阻尼：引入 pitch rate 对前后翅相对速度的差异贡献
   当 θ̇_p > 0 (抬头)，前翅有效上拍速度增加、后翅有效下拍速度增加
   → 前翅产生更多向下力、后翅产生更多向上力 → 形成低头恢复力矩
   这是自然的"气动俯仰阻尼"，不需要调 c_damp。

2. 扫描更细粒度的 (α_f, α_b) 组合，找净力矩≈0同时升力>0的点。
"""
import numpy as np, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mechanism import wing_kinematics

PHYS = {
    "rho": 1.225, "g": 9.81, "m_total": 0.020, "I_yy": 3e-5,
    "x_front": 0.025, "x_back": -0.025, "d_cg": 0.015, "c_damp": 5e-4,
}
AERO = {
    "k_3d": 0.7, "C_rot": 1.5, "r_rot": 0.5, "k_clap": 1.3,
    "alpha_max_deg": 60.0,
}

def load_geometry():
    with open(Path(__file__).parent.parent / "data" / "wing_analysis_results.json") as f:
        data = json.load(f)
    geo = {}
    for g in data["geometry"]:
        geo[g["name"]] = {"S": g["S"], "R": g["R"], "c_avg": g["c_avg"],
                          "r1": g["r1"], "r2_sq": g["r2_sq"], "AR": g["AR"]}
    return geo
GEO = load_geometry()

def cl_cd_blended(alpha_deg):
    """v6.3 LEV理论 + Lee公式 + Dickinson低攻角匹配

    |alpha| <= 60deg: Dickinson 经验 (含LEV增强, 已验证)
    |alpha| >  60deg: LEV-theoretic C_L = A*sin(2a), Lee C_D = C_D0 + A_D*(1-cos(2a))

    连续条件 at |alpha|=60deg:
      Dickinson: C_L=1.616, C_D=2.515
      C_D at a=0: 0.393
      A_adj = 1.616 / sin(120deg) = 1.866
      C_D0 = 0.393
      A_D = (2.515 - 0.393) / (1 - cos(120deg)) = 2.122 / 1.5 = 1.414

    参考: [32] JRSI 2017 (LEV), [24] 机器人 2025 (Lee)
    """
    abs_a = np.abs(alpha_deg)
    alpha_rad = np.deg2rad(alpha_deg)

    # Dickinson 经验（所有 alpha，用于低攻角段）
    cl_d = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
    cd_d = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))

    # LEV/Lee 理论（所有 alpha，物理形式正确）
    A_adj = 1.866
    C_D0 = 0.393
    A_D = 1.414
    cl_lev = A_adj * np.sin(2.0 * alpha_rad)
    cd_lee = C_D0 + A_D * (1.0 - np.cos(2.0 * alpha_rad))

    # smoothstep 混合：|alpha| <= 55 (纯Dickinson) -> |alpha| >= 65 (纯LEV/Lee)
    lo, hi = 55.0, 65.0
    t = np.clip((abs_a - lo) / (hi - lo), 0.0, 1.0)
    w = 3.0 * t**2 - 2.0 * t**3

    C_L = (1.0 - w) * cl_d + w * cl_lev
    C_D = (1.0 - w) * cd_d + w * cd_lee
    return C_L, C_D

def precompute_kinematics(f, a, phi_offset_deg, n_points=2000):
    t, phi, phi_dot, phi_ddot, info = wing_kinematics(
        f=f, a=a, phi_offset_deg=phi_offset_deg, n_points=n_points)
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

def compute_forces_v61(phi, phi_dot, phi_ddot, theta_p, theta_dot,
                        alpha_install, wing, x_wing, rho=1.225):
    """
    v6.1: 包含气动俯仰阻尼。

    新增：θ̇_p × x_wing 对翅膀有效速度的贡献。
    当身体以 θ̇_p 抬头旋转时：
    - 前翅 (x_wing > 0) 获得额外的向上速度分量
    - 后翅 (x_wing < 0) 获得额外的向下速度分量
    → 前后翅力差异产生与 θ̇_p 反向的力矩 → 气动阻尼

    严格处理：翅膀的平动速度 = 拍动速度 + 俯仰旋转线速度
    v_total = φ̇ * R (拍动，翅膀展向积分) + θ̇_p * x_wing (俯仰，在拍动平面内)

    简化：将俯仰贡献作为等效攻角修正。
    当 θ̇_p > 0 且前翅 (x_wing > 0)：
        翅膀相对来流方向略变 → 前翅有效攻角增加、后翅减小
    """
    S = wing["S"]; R_w = wing["R"]; c_avg = wing["c_avg"]
    r1 = wing["r1"]; r2_sq = wing["r2_sq"]

    psi = phi + theta_p
    Omega = phi_dot + theta_dot
    U = np.abs(Omega) * R_w

    # ---- 俯仰气动阻尼：将 θ̇_p * x_wing 转换为等效攻角修正 ----
    # 当身体俯仰旋转时，翅膀处的线速度 = θ̇_p * x_wing
    # 这个速度在拍动平面内，叠加到来流速度上，改变有效攻角
    # Δα ≈ atan(θ̇_p * x_wing / U) ≈ θ̇_p * x_wing / U (小角度)
    v_pitch_at_wing = theta_dot * x_wing  # [m/s]，俯仰旋转在翅膀处的线速度
    # 避免除零
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_alpha_rad = np.arctan2(v_pitch_at_wing, U + 1e-6)
    delta_alpha_deg = np.rad2deg(delta_alpha_rad)

    const = 0.5 * rho * U**2 * S * r2_sq * AERO["k_3d"]
    sign_Omega = np.where(Omega <= 0, -1, 1)

    # ---- 几何攻角 + 俯仰阻尼修正 ----
    alpha_geom_rad = np.deg2rad(alpha_install) + psi
    mask_down = phi_dot <= 0
    alpha_eff_rad = np.where(mask_down, alpha_geom_rad + delta_alpha_rad,
                                        -(alpha_geom_rad + delta_alpha_rad))
    alpha_eff_deg = np.rad2deg(alpha_eff_rad)

    C_L, C_D = cl_cd_blended(alpha_eff_deg)

    L_trans = const * C_L
    D_trans = const * C_D

    alpha_eff_rad_clamped = np.deg2rad(np.clip(alpha_eff_deg, -90, 90))
    F_AM = -(rho * np.pi * c_avg**2 / 4.0) * phi_ddot * R_w * r1 * np.sin(alpha_eff_rad_clamped)

    alpha_dot = np.zeros_like(phi)
    F_rot = rho * AERO["C_rot"] * alpha_dot * phi_dot * c_avg**2 * R_w * AERO["r_rot"]

    phi_dot_peak = np.max(np.abs(phi_dot))
    in_reversal = np.abs(phi_dot) < 0.1 * phi_dot_peak
    k_clap = np.where(in_reversal, AERO["k_clap"], 1.0)

    L_eff = (L_trans + F_AM + F_rot) * k_clap
    D_eff = D_trans * k_clap

    Fx = np.sin(psi) * (sign_Omega * D_eff - L_eff)
    Fz = np.cos(psi) * (L_eff - sign_Omega * D_eff)

    mask_still = np.abs(Omega) < 1e-6
    Fx = np.where(mask_still, 0, Fx)
    Fz = np.where(mask_still, 0, Fz)

    F_body = np.zeros((phi.shape[0], 3))
    F_body[:, 0] = Fx; F_body[:, 2] = Fz
    return F_body, np.zeros((phi.shape[0], 3)), None

def compute_rhs_v61(theta_p, theta_dot, phi_f, phi_dot_f, phi_ddot_f,
                     phi_b, phi_dot_b, phi_ddot_b, fp, bp):
    F_f, _, _ = compute_forces_v61(
        phi_f, phi_dot_f, phi_ddot_f, theta_p, theta_dot,
        fp["alpha_install"], GEO["Front"], PHYS["x_front"])
    F_b, _, _ = compute_forces_v61(
        phi_b, phi_dot_b, phi_ddot_b, theta_p, theta_dot,
        bp["alpha_install"], GEO["Back"], PHYS["x_back"])

    Fx_total = 2 * (F_f[:, 0] + F_b[:, 0])
    Fz_total = 2 * (F_f[:, 2] + F_b[:, 2])
    M_aero = 2 * (-PHYS["x_front"] * F_f[:, 2] - PHYS["x_back"] * F_b[:, 2])

    M_gravity = -PHYS["m_total"] * PHYS["g"] * PHYS["d_cg"] * np.sin(theta_p)
    M_damp = -PHYS["c_damp"] * theta_dot
    theta_ddot = (M_aero + M_gravity + M_damp) / PHYS["I_yy"]

    if np.ndim(theta_ddot) > 0:
        theta_ddot = float(theta_ddot[0])
        Fx_total = float(Fx_total[0]); Fz_total = float(Fz_total[0])
        M_aero = float(M_aero[0])
    return float(theta_dot), float(theta_ddot), Fx_total, 0.0, Fz_total, M_aero

def rk4_step(y, dt, *args):
    tp, td = y
    k1_td, k1_tdd, _, _, _, _ = compute_rhs_v61(tp, td, *args)
    k2_td, k2_tdd, _, _, _, _ = compute_rhs_v61(tp+0.5*dt*k1_td, td+0.5*dt*k1_tdd, *args)
    k3_td, k3_tdd, _, _, _, _ = compute_rhs_v61(tp+0.5*dt*k2_td, td+0.5*dt*k2_tdd, *args)
    k4_td, k4_tdd, _, _, _, _ = compute_rhs_v61(tp+dt*k3_td, td+dt*k3_tdd, *args)
    tp_new = tp + dt*(k1_td + 2*k2_td + 2*k3_td + k4_td)/6
    td_new = td + dt*(k1_tdd + 2*k2_tdd + 2*k3_tdd + k4_tdd)/6
    return tp_new, td_new

def run_fixed(title, ai_f, ai_b, t_end=3.0, n_steps=60000, theta0_deg=0.0):
    """Run v6.1 with aerodynamic pitch damping, longer time"""
    base = {"f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84}
    fp = base.copy(); bp = base.copy()
    fp["alpha_install"] = ai_f; bp["alpha_install"] = ai_b

    t_ext_f, pef, pdf, pddf, Tf = precompute_kinematics(fp["f"], fp["a"], fp["phi_offset_deg"])
    t_ext_b, peb, pdb, pddb, Tb = precompute_kinematics(bp["f"], bp["a"], bp["phi_offset_deg"])

    dt = t_end / n_steps
    t = np.linspace(0, t_end, n_steps)
    pf = get_states(t, t_ext_f, pef, pdf, pddf, 0, Tf)[0]
    pdf_arr = get_states(t, t_ext_f, pef, pdf, pddf, 0, Tf)[1]
    pddf_arr = get_states(t, t_ext_f, pef, pdf, pddf, 0, Tf)[2]
    pb = get_states(t, t_ext_b, peb, pdb, pddb, 0, Tb)[0]
    pdb_arr = get_states(t, t_ext_b, peb, pdb, pddb, 0, Tb)[1]
    pddb_arr = get_states(t, t_ext_b, peb, pdb, pddb, 0, Tb)[2]

    tp = np.zeros(n_steps); tp[0] = np.deg2rad(theta0_deg)
    td = np.zeros(n_steps)
    Fz_hist = np.zeros(n_steps); Fx_hist = np.zeros(n_steps)

    for i in range(n_steps - 1):
        _, _, Fx, _, Fz, _ = compute_rhs_v61(
            tp[i], td[i], pf[i:i+1], pdf_arr[i:i+1], pddf_arr[i:i+1],
            pb[i:i+1], pdb_arr[i:i+1], pddb_arr[i:i+1], fp, bp)
        Fx_hist[i] = Fx; Fz_hist[i] = Fz
        tp[i+1], td[i+1] = rk4_step(
            [tp[i], td[i]], dt, pf[i:i+1], pdf_arr[i:i+1], pddf_arr[i:i+1],
            pb[i:i+1], pdb_arr[i:i+1], pddb_arr[i:i+1], fp, bp)

    tp_deg = np.degrees(tp)
    half = n_steps // 2
    weight_mN = PHYS["m_total"] * PHYS["g"] * 1000

    return {
        "title": title, "tp_deg": tp_deg, "t": t,
        "peak_all": np.max(np.abs(tp_deg)),
        "peak_2nd": np.max(np.abs(tp_deg[half:])),
        "avg_Fz": np.mean(Fz_hist[half:]) * 1000,
        "avg_Fx": np.mean(Fx_hist[half:]) * 1000,
        "L/W": np.mean(Fz_hist[half:]) * 1000 / weight_mN,
        "theta_dot_final": td[-1],
    }


if __name__ == "__main__":
    print("=" * 70)
    print("v6.1 俯仰发散修复测试：气动阻尼 + 长时仿真 (t=3s)")
    print("=" * 70)

    configs = [
        (60, 10, "best-stable"),
        (30, 60, "high-lift"),
        (20, 60, "highest-lift"),
        (50, 15, "refined-stable"),
        (55, 12, "refined-stable2"),
        (40, 60, "mid-lift"),
    ]

    for ai_f, ai_b, label in configs:
        res = run_fixed(label, ai_f, ai_b, t_end=3.0, n_steps=60000)
        status = "✅" if res["peak_all"] < 90 else ("⚠️" if res["peak_all"] < 180 else "❌")
        print(f"  {status} α_f={ai_f}°/α_b={ai_b}° {label:20s} | "
              f"peak_all={res['peak_all']:6.1f}° | peak_2nd={res['peak_2nd']:5.1f}° | "
              f"L/W={res['L/W']:.3f} | Fz={res['avg_Fz']:+6.1f}mN | Fx={res['avg_Fx']:+6.1f}mN | "
              f"θ̇_final={res['theta_dot_final']:.4f}")

    print("\nDone.")
