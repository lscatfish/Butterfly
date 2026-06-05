#!/usr/bin/env python3
"""
v6.3 LEV/Lee C_L/C_D 全参数重新扫描

新公式: |a|<=55: Dickinson -> |a|>=65: LEV sin(2a)/Lee (1-cos(2a))
连续匹配在 60deg, C_D_max=3.221 (接近昆虫实测)
"""
import numpy as np, sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from pitch_dynamics_v6_1 import *

def scan(t_end=3.0, n_steps=60000):
    base = {"f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84}
    ai_f_range = [28, 32, 35, 38, 40, 42, 45, 48, 50, 55, 60]
    ai_b_range = [8, 10, 12, 15, 18, 20, 22, 25, 30]
    total = len(ai_f_range) * len(ai_b_range)
    weight_mN = PHYS["m_total"] * PHYS["g"] * 1000
    results = []
    cnt = 0

    print(f"v6.3 LEV/Lee scan: af in {ai_f_range}, ab in {ai_b_range} ({total} combos)")
    print(f"t={t_end}s, dt={t_end/n_steps*1e6:.0f}us")
    print(f"{'af':>4} {'ab':>4} {'L/W':>8} {'Peak':>7} {'n90':>4} {'Fz_avg':>8} {'Fx_avg':>8} {'td':>5}")
    print("-" * 70)

    for ai_f in ai_f_range:
        for ai_b in ai_b_range:
            cnt += 1
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

            tp = np.zeros(n_steps); td = np.zeros(n_steps)
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
            peak = np.max(np.abs(tp_deg))
            n90 = int(np.sum(np.abs(tp_deg) > 90))
            avg_Fz = np.mean(Fz_hist[half:]) * 1000
            avg_Fx = np.mean(Fx_hist[half:]) * 1000
            lw = avg_Fz / weight_mN
            td_mean = np.mean(np.abs(td[half:]))

            flag = "!!" if n90 > 0 else ("++" if lw > 0.8 else (" +" if lw > 0 else "  "))
            results.append({"ai_f": ai_f, "ai_b": ai_b, "L/W": lw, "peak": peak,
                           "n90": n90, "Fz": avg_Fz, "Fx": avg_Fx, "td": td_mean, "flag": flag})

            print(f"  {flag} {ai_f:3d} {ai_b:3d} {lw:8.3f} {peak:6.1f}deg {n90:3d}  {avg_Fz:+7.0f} {avg_Fx:+7.0f} {td_mean:5.1f}")

    # Output results
    print(f"\n{'='*70}")
    print(f"Positive L/W (L/W>0, n90=0) sorted:")
    good = [r for r in results if r["L/W"] > 0 and r["n90"] == 0]
    good.sort(key=lambda x: x["L/W"], reverse=True)
    print(f"{'af':>4} {'ab':>4} {'L/W':>8} {'Peak':>7} {'n90':>4} {'Fz':>8} {'Fx':>8}")
    for r in good[:20]:
        print(f"{'>>' if r['L/W']>0.8 else '  '} {r['ai_f']:3d} {r['ai_b']:3d} {r['L/W']:8.3f} {r['peak']:6.1f}deg {r['n90']:3d}  {r['Fz']:+7.0f} {r['Fx']:+7.0f}")

    if len(good) == 0:
        print("  NONE! Checking marginal (n90<5)...")
        marginal = [r for r in results if r["L/W"] > 0 and r["n90"] < 5]
        marginal.sort(key=lambda x: x["L/W"], reverse=True)
        for r in marginal[:10]:
            print(f"  {r['ai_f']:3d} {r['ai_b']:3d} {r['L/W']:8.3f} {r['peak']:6.0f}deg n90={r['n90']} {r['Fz']:+7.0f}")

    print(f"\nDone. {cnt} combos. {len(good)} stable positive-lift solutions.")

    # Auto-save JSON
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "v63_scan_results.json"
    summary = {
        "version": "v6.3",
        "formula": "LEV/Lee hybrid (Dickinson |a|<=55, smoothstep 55-65, LEV/Lee |a|>=65)",
        "scan_params": {"t_end_s": t_end, "n_steps": n_steps,
                        "alpha_f_range": ai_f_range, "alpha_b_range": ai_b_range},
        "total_combos": cnt, "positive_lift_count": len(good),
        "best": good[0] if good else None,
        "top_20": good[:20], "all_results": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"Saved JSON to {out_path}")

    return results

if __name__ == "__main__":
    scan(t_end=3.0, n_steps=60000)
