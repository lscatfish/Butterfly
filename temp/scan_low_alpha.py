#!/usr/bin/env python3
"""
v6.2: 低安装角扫描 — 让 α_eff 回到经验公式有效范围

目标: α_eff 在大部分拍动周期内 |α| ≤ 60°
策略: α_install ∈ [25°, 42°], 扫描非对称组合
      保持气动俯仰阻尼, t=3s 验证稳定性
"""
import numpy as np, sys, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from pitch_dynamics_v6_1 import *

def scan_and_verify(t_end=3.0, n_steps=60000):
    base = {"f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84}

    ai_f_range = [28, 30, 32, 35, 38, 40, 42, 45]
    ai_b_range = [10, 15, 18, 20, 22, 25, 28, 30]

    total = len(ai_f_range) * len(ai_b_range)
    results = []
    cnt = 0
    weight_mN = PHYS["m_total"] * PHYS["g"] * 1000

    print(f"扫描: α_f ∈ {ai_f_range}, α_b ∈ {ai_b_range}  ({total} 组合)")
    print(f"t_end={t_end}s, dt={t_end/n_steps*1e6:.0f}us")
    print(f"{'α_f':>5} {'α_b':>5} {'L/W':>8} {'Peak θ':>8} {'>90?':>6} {'Fz_avg':>8} {'Fx_avg':>8} {'θ̇_f':>8} {'α_max':>6}")
    print("-" * 80)

    for ai_f in ai_f_range:
        for ai_b in ai_b_range:
            cnt += 1
            fp = base.copy(); bp = base.copy()
            fp["alpha_install"] = ai_f; bp["alpha_install"] = ai_b

            # 预计算运动学
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

            tp = np.zeros(n_steps); td = np.zeros(n_steps)
            Fz_hist = np.zeros(n_steps); Fx_hist = np.zeros(n_steps)
            # 记录一个拍动周期内的 α_eff 范围
            alpha_max_seen = 0

            for i in range(n_steps - 1):
                _, _, Fx, _, Fz, _ = compute_rhs_v61(
                    tp[i], td[i], pf[i:i+1], pdf_arr[i:i+1], pddf_arr[i:i+1],
                    pb[i:i+1], pdb_arr[i:i+1], pddb_arr[i:i+1], fp, bp)
                Fx_hist[i] = Fx; Fz_hist[i] = Fz
                tp[i+1], td[i+1] = rk4_step(
                    [tp[i], td[i]], dt, pf[i:i+1], pdf_arr[i:i+1], pddf_arr[i:i+1],
                    pb[i:i+1], pdb_arr[i:i+1], pddb_arr[i:i+1], fp, bp)

                # 估计 α_eff 范围 (前翅)
                psi_f = pf[i] + tp[i]
                alpha_geom = np.rad2deg(np.deg2rad(ai_f) + psi_f)
                if abs(alpha_geom) > alpha_max_seen:
                    alpha_max_seen = abs(alpha_geom)

            tp_deg = np.degrees(tp)
            half = n_steps // 2
            peak_all = np.max(np.abs(tp_deg))
            peak_2nd = np.max(np.abs(tp_deg[half:]))
            n90 = int(np.sum(np.abs(tp_deg) > 90))
            avg_Fz = np.mean(Fz_hist[half:]) * 1000
            avg_Fx = np.mean(Fx_hist[half:]) * 1000
            lw = avg_Fz / weight_mN
            td_final = td[-1]

            flag = "✅" if (n90 == 0 and peak_all < 90) else ("⚠️" if n90 < 10 else "❌")
            results.append({
                "ai_f": ai_f, "ai_b": ai_b, "L/W": lw, "peak_all": peak_all,
                "n90": n90, "avg_Fz": avg_Fz, "avg_Fx": avg_Fx,
                "theta_dot_f": td_final, "alpha_max": alpha_max_seen, "flag": flag,
            })

            print(f"  {flag} {ai_f:4d} {ai_b:4d} {lw:8.3f} {peak_all:7.1f}° {n90:5d}  {avg_Fz:+7.0f} {avg_Fx:+7.0f} {td_final:+7.2f} {alpha_max_seen:5.0f}°")

    # 排序输出
    print(f"\n{'='*70}")
    print(f"按 L/W 排序 (只列 L/W>0 且 peak<90° 的)")
    print(f"{'='*70}")
    good = [r for r in results if r["L/W"] > 0 and r["n90"] == 0]
    good.sort(key=lambda x: x["L/W"], reverse=True)

    if good:
        print(f"{'α_f':>5} {'α_b':>5} {'L/W':>8} {'Peak θ':>8} {'Fz':>8} {'Fx':>8} {'α_max':>6}")
        for r in good[:20]:
            print(f"{r['ai_f']:5d} {r['ai_b']:5d} {r['L/W']:8.3f} {r['peak_all']:7.1f}° {r['avg_Fz']:+7.0f} {r['avg_Fx']:+7.0f} {r['alpha_max']:5.0f}°")
    else:
        print("  无符合条件的参数！尝试放宽条件...")
        loose = [r for r in results if r["L/W"] > 0 and r["n90"] < 10]
        loose.sort(key=lambda x: x["L/W"], reverse=True)
        for r in loose[:10]:
            print(f"  α_f={r['ai_f']} α_b={r['ai_b']} L/W={r['L/W']:.3f} peak={r['peak_all']:.0f}° n90={r['n90']} Fz={r['avg_Fz']:.0f}")

    return results

if __name__ == "__main__":
    results = scan_and_verify(t_end=3.0, n_steps=60000)
    print("\nDone.")
