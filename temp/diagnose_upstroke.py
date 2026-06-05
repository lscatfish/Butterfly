#!/usr/bin/env python3
"""诊断上拍攻角和公式使用情况"""
import numpy as np, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from pitch_dynamics_v6_1 import *

def analyze_one_cycle(ai_f, ai_b, theta_p_fixed=None):
    """在固定 θ_p 下分析一个拍动周期的攻角和公式使用"""
    base = {"f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84}
    fp = base.copy(); bp = base.copy()
    fp["alpha_install"] = ai_f; bp["alpha_install"] = ai_b

    # 用固定 θ_p 计算一个周期
    t_ext_f, pef, pdf_arr, pddf_arr, Tf = precompute_kinematics(fp["f"], fp["a"], fp["phi_offset_deg"])
    t_ext_b, peb, pdb_arr, pddb_arr, Tb = precompute_kinematics(bp["f"], bp["a"], bp["phi_offset_deg"])

    n = 1000
    t = np.linspace(0, Tf, n)
    pf, pdotf, pddotf = get_states(t, t_ext_f, pef, pdf_arr, pddf_arr, 0, Tf)
    pb, pdotb, pddotb = get_states(t, t_ext_b, peb, pdb_arr, pddb_arr, 0, Tb)

    theta_p = theta_p_fixed if theta_p_fixed is not None else 0.0
    theta_dot = 0.0

    # 只分析前翅
    alpha_geom_rad = np.deg2rad(ai_f) + pf + theta_p
    mask_down = pdotf <= 0
    alpha_eff_rad = np.where(mask_down, alpha_geom_rad, -alpha_geom_rad)
    alpha_eff_deg = np.rad2deg(alpha_eff_rad)

    # 计算 C_L, C_D
    abs_a = np.abs(alpha_eff_deg)
    lo, hi = 40.0, 70.0
    t_w = np.clip((abs_a - lo) / (hi - lo), 0.0, 1.0)
    w = 3.0 * t_w**2 - 2.0 * t_w**3

    cl_e = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_eff_deg - 7.2))
    cd_e = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_eff_deg - 9.82))
    C_N90 = 2.0
    alpha_rad = np.deg2rad(alpha_eff_deg)
    cl_p = (C_N90 / 2.0) * np.sin(2.0 * alpha_rad)
    cd_p = C_N90 * np.sin(alpha_rad)**2
    C_L = (1.0 - w) * cl_e + w * cl_p
    C_D = (1.0 - w) * cd_e + w * cd_p

    # 计算力
    S = GEO["Front"]["S"]; R_w = GEO["Front"]["R"]
    r2_sq = GEO["Front"]["r2_sq"]; r1 = GEO["Front"]["r1"]
    c_avg = GEO["Front"]["c_avg"]
    psi = pf + theta_p
    Omega = pdotf + theta_dot
    U = np.abs(Omega) * R_w
    const = 0.5 * PHYS["rho"] * U**2 * S * r2_sq * AERO["k_3d"]
    sign_Omega = np.where(Omega <= 0, -1, 1)

    L_trans = const * C_L
    D_trans = const * C_D
    F_AM = -(PHYS["rho"] * np.pi * c_avg**2 / 4.0) * pddotf * R_w * r1 * np.sin(alpha_rad)
    k_clap = np.where(np.abs(pdotf) < 0.1 * np.max(np.abs(pdotf)), AERO["k_clap"], 1.0)
    L_eff = (L_trans + F_AM) * k_clap
    D_eff = D_trans * k_clap
    Fx = np.sin(psi) * (sign_Omega * D_eff - L_eff)
    Fz = np.cos(psi) * (L_eff - sign_Omega * D_eff)

    return {
        "t": t, "phi_deg": np.degrees(pf), "psi_deg": np.degrees(psi),
        "alpha_eff_deg": alpha_eff_deg, "w_plate": w,
        "C_L": C_L, "C_D": C_D, "cl_emp": cl_e, "cd_emp": cd_e,
        "cl_plate": cl_p, "cd_plate": cd_p,
        "L_eff": L_eff, "D_eff": D_eff, "Fz": Fz, "Fx": Fx,
        "mask_down": mask_down,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("上拍攻角 & 公式使用诊断")
    print("=" * 70)

    # 诊断 α_f=60° 在 θ_p=20° (稳态典型值) 下的一个完整周期
    d = analyze_one_cycle(60, 10, theta_p_fixed=np.deg2rad(20))

    # 区分上拍和下拍
    mask_up = ~d["mask_down"]
    t_up = d["t"][mask_up]
    alpha_up = d["alpha_eff_deg"][mask_up]
    w_up = d["w_plate"][mask_up]
    cl_up = d["C_L"][mask_up]
    cd_up = d["C_D"][mask_up]
    fz_up = d["Fz"][mask_up]
    fx_up = d["Fx"][mask_up]
    psi_up = d["psi_deg"][mask_up]

    mask_dn = d["mask_down"]
    alpha_dn = d["alpha_eff_deg"][mask_dn]
    w_dn = d["w_plate"][mask_dn]
    fz_dn = d["Fz"][mask_dn]
    fx_dn = d["Fx"][mask_dn]

    print(f"\n--- 下拍 (φ_dot ≤ 0, {np.sum(mask_dn)}/{len(d['t'])} 点) ---")
    print(f"  α_eff range: [{alpha_dn.min():.1f}°, {alpha_dn.max():.1f}°]")
    print(f"  α_eff mean:  {np.mean(alpha_dn):.1f}°")
    print(f"  平板权重 w: [{w_dn.min():.2f}, {w_dn.max():.2f}]")
    n_emp_dn = np.sum(w_dn < 0.1); n_blend_dn = np.sum((w_dn >= 0.1) & (w_dn < 0.9)); n_plate_dn = np.sum(w_dn >= 0.9)
    print(f"  公式使用: 纯经验={n_emp_dn}, 过渡={n_blend_dn}, 平板={n_plate_dn}")
    print(f"  Fz range:  [{fz_dn.min()*1000:.0f}, {fz_dn.max()*1000:.0f}] mN")
    print(f"  Fz mean:   {np.mean(fz_dn)*1000:.0f} mN")
    print(f"  Fx mean:   {np.mean(fx_dn)*1000:.0f} mN")

    print(f"\n--- 上拍 (φ_dot > 0, {np.sum(mask_up)}/{len(d['t'])} 点) ---")
    print(f"  α_eff range: [{alpha_up.min():.1f}°, {alpha_up.max():.1f}°]")
    print(f"  α_eff mean:  {np.mean(alpha_up):.1f}°")
    print(f"  平板权重 w: [{w_up.min():.2f}, {w_up.max():.2f}]")
    n_emp_up = np.sum(w_up < 0.1); n_blend_up = np.sum((w_up >= 0.1) & (w_up < 0.9)); n_plate_up = np.sum(w_up >= 0.9)
    print(f"  公式使用: 纯经验={n_emp_up}, 过渡={n_blend_up}, 平板={n_plate_up}")
    print(f"  C_L range:  [{cl_up.min():.2f}, {cl_up.max():.2f}]")
    print(f"  C_D range:  [{cd_up.min():.2f}, {cd_up.max():.2f}]")
    print(f"  ψ range:    [{psi_up.min():.1f}°, {psi_up.max():.1f}°]")
    print(f"  Fz range:   [{fz_up.min()*1000:.0f}, {fz_up.max()*1000:.0f}] mN")
    print(f"  Fz mean:    {np.mean(fz_up)*1000:.0f} mN")
    print(f"  Fx mean:    {np.mean(fx_up)*1000:.0f} mN")

    # 关键：上拍力分解
    print(f"\n--- 上拍力分解 (ψ≈{np.mean(psi_up):.0f}°) ---")
    # 取中间一个上拍点
    mid = len(t_up) // 2
    print(f"  取上拍中点: α={alpha_up[mid]:.1f}°, ψ={psi_up[mid]:.1f}°")
    print(f"  C_L={cl_up[mid]:.3f}, C_D={cd_up[mid]:.3f}")
    # Fz = cos(ψ)*(L_eff - D_eff)  (sign=+1 for upstroke)
    # Fx = sin(ψ)*(D_eff - L_eff)
    L_mid = d["L_eff"][mask_up][mid]
    D_mid = d["D_eff"][mask_up][mid]
    psi_rad = np.deg2rad(psi_up[mid])
    Fz_check = np.cos(psi_rad) * (L_mid - D_mid) * 1000
    Fx_check = np.sin(psi_rad) * (D_mid - L_mid) * 1000
    print(f"  L_eff={L_mid*1000:.1f} mN, D_eff={D_mid*1000:.1f} mN")
    print(f"  Fz=cos({psi_up[mid]:.0f}°)*({L_mid*1000:.0f}-{D_mid*1000:.0f})={Fz_check:.0f} mN")
    print(f"  Fx=sin({psi_up[mid]:.0f}°)*({D_mid*1000:.0f}-{L_mid*1000:.0f})={Fx_check:.0f} mN")

    # 一个周期的净力
    print(f"\n--- 周期净值 ---")
    print(f"  下拍平均 Fz: {np.mean(fz_dn)*1000:.0f} mN (持续 {np.sum(mask_dn)/len(d['t'])*100:.0f}%)")
    print(f"  上拍平均 Fz: {np.mean(fz_up)*1000:.0f} mN (持续 {np.sum(mask_up)/len(d['t'])*100:.0f}%)")
    print(f"  周期平均 Fz: {np.mean(d['Fz'])*1000:.0f} mN")
    print(f"  周期平均 Fx: {np.mean(d['Fx'])*1000:.0f} mN")

    print("\nDone.")
