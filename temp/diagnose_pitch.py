#!/usr/bin/env python3
"""诊断俯仰发散根因：检查 M_aero vs M_gravity 的量级对比"""
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pitch_dynamics_v6 import *

def diagnose(ai_f, ai_b, label, t_end=1.5, n_steps=30000):
    base = {"f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84}
    fp = base.copy(); bp = base.copy()
    fp["alpha_install"] = ai_f; bp["alpha_install"] = ai_b

    t_ext_f, pef, pdf, pddf, Tf, _ = precompute_kinematics(fp["f"], fp["a"], fp["phi_offset_deg"])
    t_ext_b, peb, pdb, pddb, Tb, _ = precompute_kinematics(bp["f"], bp["a"], bp["phi_offset_deg"])

    dt = t_end / n_steps
    t = np.linspace(0, t_end, n_steps)
    pf, pdf_arr, pddf_arr = get_states(t, t_ext_f, pef, pdf, pddf, 0, Tf)
    pb, pdb_arr, pddb_arr = get_states(t, t_ext_b, peb, pdb, pddb, 0, Tb)

    tp = np.zeros(n_steps); td = np.zeros(n_steps)
    M_hist = np.zeros(n_steps); M_grav_hist = np.zeros(n_steps)
    Fz_hist = np.zeros(n_steps)

    for i in range(n_steps - 1):
        _, _, Fx, _, Fz, M = compute_rhs_fixed(
            tp[i], td[i], pf[i:i+1], pdf_arr[i:i+1], pddf_arr[i:i+1],
            pb[i:i+1], pdb_arr[i:i+1], pddb_arr[i:i+1], fp, bp)
        M_hist[i] = M
        M_grav_hist[i] = -PHYS["m_total"] * PHYS["g"] * PHYS["d_cg"] * np.sin(tp[i])
        Fz_hist[i] = Fz
        tp[i+1], td[i+1] = rk4_step_fixed(
            [tp[i], td[i]], dt, pf[i:i+1], pdf_arr[i:i+1], pddf_arr[i:i+1],
            pb[i:i+1], pdb_arr[i:i+1], pddb_arr[i:i+1], fp, bp)

    tp_deg = np.degrees(tp)
    half = n_steps // 2

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: {label} (α_f={ai_f}°, α_b={ai_b}°)")
    print(f"{'='*60}")
    print(f"  Peak |θ_p| (full):        {np.max(np.abs(tp_deg)):10.1f} deg")
    print(f"  Peak |θ_p| (1st half):    {np.max(np.abs(tp_deg[:half])):10.1f} deg")
    print(f"  Peak |θ_p| (2nd half):    {np.max(np.abs(tp_deg[half:])):10.1f} deg")
    print(f"  Avg θ_p drift rate:       {np.mean(td[half:]):10.3f} rad/s")
    print(f"  Avg M_aero (2nd half):    {np.mean(M_hist[half:])*1e6:10.1f} uN.m")
    print(f"  Avg M_grav (2nd half):    {np.mean(M_grav_hist[half:])*1e6:10.1f} uN.m")
    print(f"  |M_grav|_max possible:    {PHYS['m_total']*PHYS['g']*PHYS['d_cg']*1e6:10.1f} uN.m")
    print(f"  M_damp typical:           {PHYS['c_damp']*np.std(td[half:])*1e6:10.1f} uN.m")
    print(f"  Avg Fz (2nd half):        {np.mean(Fz_hist[half:])*1000:10.1f} mN")
    L_W = np.mean(Fz_hist[half:])*1000 / (PHYS["m_total"]*PHYS["g"]*1000)
    print(f"  L/W (2nd half):           {L_W:10.3f}")

    # 关键诊断：M_aero 是否超过重力恢复极限
    M_aero_rms = np.sqrt(np.mean(M_hist[half:]**2))
    M_grav_max = PHYS["m_total"] * PHYS["g"] * PHYS["d_cg"]
    if np.abs(np.mean(M_hist[half:])) > M_grav_max:
        print(f"  ** FATAL: |avg M_aero| > M_grav_max → 俯仰必然发散！**")
    else:
        print(f"  OK: |avg M_aero| < M_grav_max → 理论上有平衡点")

    return tp_deg, M_hist, t

if __name__ == "__main__":
    # 只诊断有正升力的参数
    configs = [
        (60, 10, "best-stable (L/W=0.47)"),
        (30, 60, "high-lift (L/W=1.30)"),
        (20, 60, "highest-lift (L/W=1.48)"),
        (40, 60, "mid (L/W=0.82)"),
        (50, 10, "modest (L/W=0.18)"),
    ]
    for ai_f, ai_b, label in configs:
        diagnose(ai_f, ai_b, label, t_end=1.5, n_steps=30000)
    print("\nDone.")
