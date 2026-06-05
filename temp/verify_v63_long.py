#!/usr/bin/env python3
"""v6.3 长时稳定性验证 + 完整时程图: t=10s, 200k steps, top-3 参数"""
import numpy as np, sys, json, matplotlib, os, time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from pitch_dynamics_v6_1 import *

OUT_DIR = Path(__file__).parent / "v63_long"
OUT_DIR.mkdir(exist_ok=True)

def verify_and_plot(label, ai_f, ai_b, t_end=10.0, n_steps=200000):
    """10s长稳验证 + 6-panel完整时程图"""
    base = {"f": 15.0, "phase": 0.0, "a": 7.92, "phi_offset_deg": -50.84}
    fp = base.copy(); bp = base.copy()
    fp["alpha_install"] = ai_f; bp["alpha_install"] = ai_b

    t_ext_f, pef, pdf, pddf, Tf = precompute_kinematics(fp["f"], fp["a"], fp["phi_offset_deg"])
    t_ext_b, peb, pdb, pddb, Tb = precompute_kinematics(bp["f"], bp["a"], bp["phi_offset_deg"])

    dt = t_end / n_steps
    print(f"  dt={dt*1e6:.0f}us, t_end={t_end}s, steps={n_steps}")

    t = np.linspace(0, t_end, n_steps)
    pf = get_states(t, t_ext_f, pef, pdf, pddf, 0, Tf)[0]
    pdf_arr = get_states(t, t_ext_f, pef, pdf, pddf, 0, Tf)[1]
    pddf_arr = get_states(t, t_ext_f, pef, pdf, pddf, 0, Tf)[2]
    pb = get_states(t, t_ext_b, peb, pdb, pddb, 0, Tb)[0]
    pdb_arr = get_states(t, t_ext_b, peb, pdb, pddb, 0, Tb)[1]
    pddb_arr = get_states(t, t_ext_b, peb, pdb, pddb, 0, Tb)[2]

    tp = np.zeros(n_steps); td = np.zeros(n_steps)
    Fz_hist = np.zeros(n_steps); Fx_hist = np.zeros(n_steps); M_hist = np.zeros(n_steps)

    t0 = time.time()
    for i in range(n_steps - 1):
        _, _, Fx, _, Fz, M = compute_rhs_v61(
            tp[i], td[i], pf[i:i+1], pdf_arr[i:i+1], pddf_arr[i:i+1],
            pb[i:i+1], pdb_arr[i:i+1], pddb_arr[i:i+1], fp, bp)
        Fx_hist[i] = Fx; Fz_hist[i] = Fz; M_hist[i] = M
        tp[i+1], td[i+1] = rk4_step(
            [tp[i], td[i]], dt, pf[i:i+1], pdf_arr[i:i+1], pddf_arr[i:i+1],
            pb[i:i+1], pdb_arr[i:i+1], pddb_arr[i:i+1], fp, bp)
    elapsed = time.time() - t0
    print(f"  Sim took {elapsed:.0f}s")

    tp_deg = np.degrees(tp)
    peak_all = np.max(np.abs(tp_deg))
    peak_last80pct = np.max(np.abs(tp_deg[int(n_steps*0.2):]))
    n_exceed_90 = int(np.sum(np.abs(tp_deg) > 90))
    n_exceed_180 = int(np.sum(np.abs(tp_deg) > 180))

    weight_mN = PHYS["m_total"] * PHYS["g"] * 1000
    half = n_steps // 2
    quarter = n_steps // 4
    avg_Fz_q = [np.mean(Fz_hist[q*quarter:(q+1)*quarter]) * 1000 for q in range(4)]
    avg_Fx_q = [np.mean(Fx_hist[q*quarter:(q+1)*quarter]) * 1000 for q in range(4)]
    avg_Fz = np.mean(Fz_hist[half:]) * 1000
    avg_Fx = np.mean(Fx_hist[half:]) * 1000
    lw = avg_Fz / weight_mN
    td_range = (td.min(), td.max())
    td_mean_abs = np.mean(np.abs(td[half:]))

    status = "✅ STABLE" if (n_exceed_90 == 0 and peak_all < 90) else \
             ("⚠️  MARGINAL" if n_exceed_180 == 0 else "❌ DIVERGED")

    print(f"  {status}")
    print(f"    peak_all={peak_all:.1f}°, peak_last80%={peak_last80pct:.1f}°")
    print(f"    exceed_90={n_exceed_90}, exceed_180={n_exceed_180}")
    print(f"    Fz quarters: Q1={avg_Fz_q[0]:+.0f} Q2={avg_Fz_q[1]:+.0f} Q3={avg_Fz_q[2]:+.0f} Q4={avg_Fz_q[3]:+.0f} mN")
    print(f"    Fx quarters: Q1={avg_Fx_q[0]:+.0f} Q2={avg_Fx_q[1]:+.0f} Q3={avg_Fx_q[2]:+.0f} Q4={avg_Fx_q[3]:+.0f} mN")
    print(f"    L/W={lw:.3f}, |θ̇|_avg={td_mean_abs:.1f} rad/s, θ̇_range=[{td_range[0]:.0f},{td_range[1]:.0f}]")

    # ---- 6-panel plot ----
    mask_zoom = t > (t_end - 0.5)

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    title = f"v6.3 α_f={ai_f}° α_b={ai_b}°  |  L/W={lw:.3f}  peak={peak_all:.1f}°  n90={n_exceed_90}"
    fig.suptitle(f"{title} — {status}", fontsize=14, fontweight="bold")

    # (0,0) Pitch full
    ax = axes[0, 0]
    ax.plot(t, tp_deg, "b-", lw=0.6)
    ax.axhline(90, color="r", ls="--", lw=0.8, alpha=0.4)
    ax.axhline(-90, color="r", ls="--", lw=0.8, alpha=0.4)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("θ_p (deg)")
    ax.set_title(f"Pitch Angle | peak={peak_all:.1f}°")
    ax.grid(True, alpha=0.3)

    # (0,1) Pitch rate
    ax = axes[0, 1]
    ax.plot(t, td, "r-", lw=0.5, alpha=0.8)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("θ̇_p (rad/s)")
    ax.set_title(f"Pitch Rate | range=[{td_range[0]:.0f},{td_range[1]:.0f}] rad/s | |θ̇|_avg={td_mean_abs:.1f}")
    ax.grid(True, alpha=0.3)

    # (1,0) Lift
    ax = axes[1, 0]
    ax.plot(t, Fz_hist * 1000, "g-", lw=0.4, alpha=0.7)
    ax.axhline(weight_mN, color="r", ls=":", alpha=0.5, label=f"Weight={weight_mN:.0f}mN")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Lift Fz (mN)")
    ax.set_title(f"Lift | avg={avg_Fz:+.0f} mN | Q1={avg_Fz_q[0]:+.0f} Q2={avg_Fz_q[1]:+.0f} Q3={avg_Fz_q[2]:+.0f} Q4={avg_Fz_q[3]:+.0f}")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (1,1) Thrust
    ax = axes[1, 1]
    ax.plot(t, Fx_hist * 1000, "m-", lw=0.4, alpha=0.7)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Thrust Fx (mN)")
    ax.set_title(f"Thrust | avg={avg_Fx:+.0f} mN")
    ax.grid(True, alpha=0.3)

    # (2,0) Pitch zoom (last 0.5s)
    ax = axes[2, 0]
    ax.plot(t[mask_zoom], tp_deg[mask_zoom], "b-", lw=1.0)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("θ_p (deg)")
    ax.set_title(f"Pitch zoom (last 0.5s) | range=[{tp_deg[mask_zoom].min():.1f},{tp_deg[mask_zoom].max():.1f}]°")
    ax.grid(True, alpha=0.3)

    # (2,1) Phase portrait
    ax = axes[2, 1]
    ax.plot(tp_deg[half:], td[half:], ".", ms=0.3, alpha=0.3, color="purple")
    ax.axhline(0, color="k", ls="--", lw=0.5); ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("θ_p (deg)"); ax.set_ylabel("θ̇_p (rad/s)")
    ax.set_title("Phase Portrait (steady)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fn = OUT_DIR / f"v63_long_{ai_f}_{ai_b}.png"
    plt.savefig(str(fn), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fn}")

    # ---- M_aero time history plot (extra) ----
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(t, M_hist * 1e6, "b-", lw=0.5, alpha=0.7, label="M_aero")
    ax2.axhline(0, color="k", ls="--", lw=0.5)
    ax2.set_ylabel("M_aero (μN·m)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"v6.3 α_f={ai_f}° α_b={ai_b}° — Aerodynamic Moment | mean={np.mean(M_hist[half:])*1e6:+.0f} μN·m")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fn2 = OUT_DIR / f"v63_moment_{ai_f}_{ai_b}.png"
    plt.savefig(str(fn2), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"    Saved: {fn2}")

    return {
        "ai_f": ai_f, "ai_b": ai_b, "status": status,
        "peak_all": peak_all, "peak_last80": peak_last80pct,
        "n_exceed_90": n_exceed_90, "n_exceed_180": n_exceed_180,
        "avg_Fz": avg_Fz, "avg_Fx": avg_Fx, "L/W": lw,
        "avg_Fz_q": avg_Fz_q, "avg_Fx_q": avg_Fx_q,
        "td_range": td_range, "td_mean_abs": td_mean_abs,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    configs = [
        (60, 8,  "v6.3 #1 best α_f=60/α_b=8  L/W=0.943"),
        (60, 10, "v6.3 #2      α_f=60/α_b=10 L/W=0.903"),
        (60, 12, "v6.3 #3      α_f=60/α_b=12 L/W=0.859"),
    ]

    print("=" * 70)
    print("v6.3 LEV/Lee 长时稳定性验证 (t=10s, 200k steps)")
    print("=" * 70)

    results = []
    for ai_f, ai_b, label in configs:
        print(f"\n{'='*70}")
        print(f"--- {label} ---")
        print(f"{'='*70}")
        res = verify_and_plot(label, ai_f, ai_b)
        results.append(res)

    print("\n" + "=" * 70)
    print("SUMMARY: v6.3 10s Long-term Stability")
    print("=" * 70)
    print(f"{'Config':<25} {'Status':<15} {'Peak':>7} {'Last80%':>7} {'n90':>4} {'L/W':>6} {'Fz':>8} {'Fx':>8} {'|θ̇|_avg':>8} {'Elapsed':>7}")
    print("-" * 105)
    for r in results:
        print(f"α_f={r['ai_f']}°/α_b={r['ai_b']:<3}° {'':<6} {r['status']:<15} {r['peak_all']:>6.1f}° {r['peak_last80']:>6.1f}° {r['n_exceed_90']:>3d}  {r['L/W']:>5.3f} {r['avg_Fz']:>+7.0f} {r['avg_Fx']:>+7.0f} {r['td_mean_abs']:>7.1f} {r['elapsed_s']:>6.0f}s")

    # Fz quarter-by-quarter trend
    print(f"\n{'Config':<25} {'Q1_Fz':>8} {'Q2_Fz':>8} {'Q3_Fz':>8} {'Q4_Fz':>8} {'ΔQ4-Q1':>8}")
    print("-" * 65)
    for r in results:
        q = r['avg_Fz_q']
        dq = q[3] - q[0]
        print(f"α_f={r['ai_f']}°/α_b={r['ai_b']:<3}° {'':<6} {q[0]:>+7.0f} {q[1]:>+7.0f} {q[2]:>+7.0f} {q[3]:>+7.0f} {dq:>+7.0f}")

    # Save JSON summary
    import json as _json
    summary = []
    for r in results:
        summary.append({k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                           [float(x) for x in v] if isinstance(v, (list, np.ndarray)) else
                           tuple(float(x) for x in v) if isinstance(v, tuple) else v)
                        for k, v in r.items() if k not in ('avg_Fz_q', 'avg_Fx_q')})
    # Add quarters separately
    for i, r in enumerate(results):
        summary[i]['Fz_quarters'] = [float(x) for x in r['avg_Fz_q']]
        summary[i]['Fx_quarters'] = [float(x) for x in r['avg_Fx_q']]
    with open(OUT_DIR / "v63_long_summary.json", "w") as f:
        _json.dump(summary, f, indent=2)
    print(f"\nSaved summary JSON to {OUT_DIR / 'v63_long_summary.json'}")

    print("\nDone.")
