#!/usr/bin/env python3
"""v6.1 长时稳定性验证：t=10s，测试俯仰是否在 ±90° 内"""
import numpy as np, sys, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from pitch_dynamics_v6_1 import *

def verify_long(title, ai_f, ai_b, t_end=10.0, n_steps=200000, plot_file=None):
    """长时间验证，返回全时程俯仰数据"""
    base = {"f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84}
    fp = base.copy(); bp = base.copy()
    fp["alpha_install"] = ai_f; bp["alpha_install"] = ai_b

    t_ext_f, pef, pdf, pddf, Tf = precompute_kinematics(fp["f"], fp["a"], fp["phi_offset_deg"])
    t_ext_b, peb, pdb, pddb, Tb = precompute_kinematics(bp["f"], bp["a"], bp["phi_offset_deg"])

    dt = t_end / n_steps
    print(f"  dt={dt*1e6:.0f}us, steps={n_steps}, t_end={t_end}s")

    t = np.linspace(0, t_end, n_steps)
    pf = get_states(t, t_ext_f, pef, pdf, pddf, 0, Tf)[0]
    pdf_arr = get_states(t, t_ext_f, pef, pdf, pddf, 0, Tf)[1]
    pddf_arr = get_states(t, t_ext_f, pef, pdf, pddf, 0, Tf)[2]
    pb = get_states(t, t_ext_b, peb, pdb, pddb, 0, Tb)[0]
    pdb_arr = get_states(t, t_ext_b, peb, pdb, pddb, 0, Tb)[1]
    pddb_arr = get_states(t, t_ext_b, peb, pdb, pddb, 0, Tb)[2]

    tp = np.zeros(n_steps)
    td = np.zeros(n_steps)
    Fz_hist = np.zeros(n_steps)
    Fx_hist = np.zeros(n_steps)
    M_hist = np.zeros(n_steps)

    for i in range(n_steps - 1):
        _, _, Fx, _, Fz, M = compute_rhs_v61(
            tp[i], td[i], pf[i:i+1], pdf_arr[i:i+1], pddf_arr[i:i+1],
            pb[i:i+1], pdb_arr[i:i+1], pddb_arr[i:i+1], fp, bp)
        Fx_hist[i] = Fx; Fz_hist[i] = Fz; M_hist[i] = M
        tp[i+1], td[i+1] = rk4_step(
            [tp[i], td[i]], dt, pf[i:i+1], pdf_arr[i:i+1], pddf_arr[i:i+1],
            pb[i:i+1], pdb_arr[i:i+1], pddb_arr[i:i+1], fp, bp)

    tp_deg = np.degrees(tp)
    peak_all = np.max(np.abs(tp_deg))
    peak_last80pct = np.max(np.abs(tp_deg[int(n_steps*0.2):]))
    n_exceed_90 = np.sum(np.abs(tp_deg) > 90)
    n_exceed_180 = np.sum(np.abs(tp_deg) > 180)

    weight_mN = PHYS["m_total"] * PHYS["g"] * 1000
    quarter = n_steps // 4
    avg_Fz_q = [np.mean(Fz_hist[q*quarter:(q+1)*quarter]) * 1000 for q in range(4)]
    avg_Fx_q = [np.mean(Fx_hist[q*quarter:(q+1)*quarter]) * 1000 for q in range(4)]

    status = "✅ STABLE" if (n_exceed_90 == 0 and peak_all < 90) else \
             ("⚠️  MARGINAL" if n_exceed_180 == 0 else "❌ DIVERGED")

    print(f"  {status}")
    print(f"    peak_all={peak_all:.1f}°, peak_last80%={peak_last80pct:.1f}°")
    print(f"    exceed_90={n_exceed_90}, exceed_180={n_exceed_180}")
    print(f"    Fz quarters: Q1={avg_Fz_q[0]:+.0f} Q2={avg_Fz_q[1]:+.0f} Q3={avg_Fz_q[2]:+.0f} Q4={avg_Fz_q[3]:+.0f} mN")
    print(f"    Fx quarters: Q1={avg_Fx_q[0]:+.0f} Q2={avg_Fx_q[1]:+.0f} Q3={avg_Fx_q[2]:+.0f} Q4={avg_Fx_q[3]:+.0f} mN")

    if plot_file:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"{title} — {status}", fontsize=14, fontweight="bold")

        ax = axes[0, 0]
        ax.plot(t, tp_deg, "b-", lw=0.8)
        ax.axhline(90, color="r", ls="--", lw=0.8, alpha=0.5, label="±90°")
        ax.axhline(-90, color="r", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.set_ylabel("Pitch (deg)")
        ax.set_title(f"Pitch | peak={peak_all:.1f}°")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(t, Fz_hist * 1000, "g-", lw=0.6, alpha=0.7)
        ax.axhline(weight_mN, color="r", ls=":", alpha=0.5, label=f"Weight={weight_mN:.0f}mN")
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.set_ylabel("Lift (mN)")
        ax.set_title(f"Lift | avg={np.mean(Fz_hist)*1000:+.0f} mN")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(t, Fx_hist * 1000, "m-", lw=0.6, alpha=0.7)
        ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.set_ylabel("Thrust (mN)")
        ax.set_title(f"Thrust | avg={np.mean(Fx_hist)*1000:+.0f} mN")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(t, tp_deg, Fz_hist*1000, ".", ms=0.5, alpha=0.2, color="purple")
        ax.axhline(0, color="k", ls="--", lw=0.5); ax.axvline(0, color="k", ls="--", lw=0.5)
        ax.set_xlabel("Pitch (deg)"); ax.set_ylabel("Lift (mN)")
        ax.set_title("Lift vs Pitch")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {plot_file}")

    return {
        "status": status, "peak_all": peak_all, "peak_last80": peak_last80pct,
        "n_exceed_90": n_exceed_90, "avg_Fz": np.mean(Fz_hist) * 1000,
        "L/W": np.mean(Fz_hist) * 1000 / weight_mN,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("v6.1 长时稳定性验证 (t=10s, 200k steps)")
    print("=" * 70)

    configs = [
        (60, 10, "best-stable (L/W=1.07)"),
        (55, 12, "refined (L/W=0.85)"),
        (50, 15, "safe (L/W=0.61)"),
        (30, 60, "high-lift (L/W=0.58)"),
    ]

    results = []
    for ai_f, ai_b, label in configs:
        fn = f"temp/v61_long_{ai_f}_{ai_b}.png"
        print(f"\n--- {label} (α_f={ai_f}°, α_b={ai_b}°) ---")
        res = verify_long(label, ai_f, ai_b, t_end=10.0, n_steps=200000, plot_file=fn)
        results.append(res)

    print("\n" + "=" * 70)
    print("SUMMARY (t=10s)")
    print("=" * 70)
    print(f"{'Config':<30} {'Status':<15} {'Peak':>8} {'Last80%':>8} {'Avg Fz':>8} {'L/W':>6}")
    for (ai_f, ai_b, label), r in zip(configs, results):
        print(f"α_f={ai_f}°/α_b={ai_b}° {label:<15} {r['status']:<15} {r['peak_all']:>7.1f}° {r['peak_last80']:>7.1f}° {r['avg_Fz']:>+7.0f} {r['L/W']:>6.3f}")

    print("\nDone.")
