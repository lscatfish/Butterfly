#!/usr/bin/env python3
"""为所有正升力解绘制完整时程图：theta_p, theta_dot_p, Fz, Fx"""
import numpy as np, sys, json, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from pitch_dynamics_v6_1 import *

def plot_full(title, ai_f, ai_b, t_end=3.0, n_steps=60000, filename=None):
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

    tp = np.zeros(n_steps); td = np.zeros(n_steps)
    Fz_hist = np.zeros(n_steps); Fx_hist = np.zeros(n_steps)

    print(f"  Simulating...")
    for i in range(n_steps - 1):
        _, _, Fx, _, Fz, _ = compute_rhs_v61(
            tp[i], td[i], pf[i:i+1], pdf_arr[i:i+1], pddf_arr[i:i+1],
            pb[i:i+1], pdb_arr[i:i+1], pddb_arr[i:i+1], fp, bp)
        Fx_hist[i] = Fx; Fz_hist[i] = Fz
        tp[i+1], td[i+1] = rk4_step(
            [tp[i], td[i]], dt, pf[i:i+1], pdf_arr[i:i+1], pddf_arr[i:i+1],
            pb[i:i+1], pdb_arr[i:i+1], pddb_arr[i:i+1], fp, bp)

    tp_deg = np.degrees(tp)
    weight_mN = PHYS["m_total"] * PHYS["g"] * 1000
    peak = np.max(np.abs(tp_deg))
    n90 = int(np.sum(np.abs(tp_deg) > 90))
    half = n_steps // 2
    avg_Fz = np.mean(Fz_hist[half:]) * 1000
    avg_Fx = np.mean(Fx_hist[half:]) * 1000
    td_range = (td.min(), td.max())
    td_mean_abs = np.mean(np.abs(td[half:]))
    mask_zoom = t > (t_end - 0.1)

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle(f"{title}  |  peak={peak:.1f}deg  n90={n90}", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(t * 1000, tp_deg, "b-", lw=0.8)
    ax.axhline(90, color="r", ls="--", lw=0.8, alpha=0.4)
    ax.axhline(-90, color="r", ls="--", lw=0.8, alpha=0.4)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("theta_p (deg)")
    ax.set_title("Pitch Angle (full)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t * 1000, td, "r-", lw=0.6, alpha=0.8)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("theta_dot_p (rad/s)")
    ax.set_title(f"Pitch Rate  |  peak={np.max(np.abs(td)):.0f} rad/s  range=[{td_range[0]:.0f},{td_range[1]:.0f}]")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t * 1000, Fz_hist * 1000, "g-", lw=0.6, alpha=0.7)
    ax.axhline(weight_mN, color="r", ls=":", alpha=0.5, label=f"Weight={weight_mN:.0f}mN")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Lift Fz (mN)")
    ax.set_title(f"Lift  |  avg={avg_Fz:+.0f} mN")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t * 1000, Fx_hist * 1000, "m-", lw=0.6, alpha=0.7)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Thrust Fx (mN)")
    ax.set_title(f"Thrust  |  avg={avg_Fx:+.0f} mN")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(t[mask_zoom] * 1000, tp_deg[mask_zoom], "b-", lw=1.2)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("theta_p (deg)")
    ax.set_title(f"Pitch zoom (last 0.1s)  range=[{tp_deg[mask_zoom].min():.1f},{tp_deg[mask_zoom].max():.1f}]deg")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(tp_deg[half:], td[half:], ".", ms=0.3, alpha=0.3, color="purple")
    ax.axhline(0, color="k", ls="--", lw=0.5); ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("theta_p (deg)"); ax.set_ylabel("theta_dot_p (rad/s)")
    ax.set_title("Phase Portrait (steady)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")
    return {"peak": peak, "n90": n90, "avg_Fz": avg_Fz, "avg_Fx": avg_Fx,
            "td_range": td_range, "td_mean_abs": td_mean_abs}


if __name__ == "__main__":
    configs = [
        # v6.2 LEV range
        ("v6.2 #1 LEV a_f=45 a_b=10 L/W=0.615", 45, 10, "temp/all/v62_01_45_10.png"),
        ("v6.2 #2 LEV a_f=42 a_b=10 L/W=0.509", 42, 10, "temp/all/v62_02_42_10.png"),
        ("v6.2 #3 LEV a_f=45 a_b=15 L/W=0.429", 45, 15, "temp/all/v62_03_45_15.png"),
        ("v6.2 #4 LEV a_f=40 a_b=10 L/W=0.427", 40, 10, "temp/all/v62_04_40_10.png"),
        ("v6.2 #5 LEV a_f=38 a_b=10 L/W=0.347", 38, 10, "temp/all/v62_05_38_10.png"),
        ("v6.2 #6 LEV a_f=42 a_b=15 L/W=0.308", 42, 15, "temp/all/v62_06_42_15.png"),
        ("v6.2 #7 LEV a_f=35 a_b=10 L/W=0.212", 35, 10, "temp/all/v62_07_35_10.png"),
        ("v6.2 #8 LEV a_f=32 a_b=10 L/W=0.056", 32, 10, "temp/all/v62_08_32_10.png"),
        # v6.1 high-alpha
        ("v6.1 #1 HiA a_f=60 a_b=10 L/W=1.145", 60, 10, "temp/all/v61_01_60_10.png"),
        ("v6.1 #2 HiA a_f=55 a_b=12 L/W=0.962", 55, 12, "temp/all/v61_02_55_12.png"),
        ("v6.1 #3 HiA a_f=50 a_b=15 L/W=0.711", 50, 15, "temp/all/v61_03_50_15.png"),
        ("v6.1 #4 HiA a_f=30 a_b=60 L/W=0.573", 30, 60, "temp/all/v61_04_30_60.png"),
    ]
    os.makedirs("temp/all", exist_ok=True)

    results = []
    n = len(configs)
    for i, (title, ai_f, ai_b, fn) in enumerate(configs):
        print(f"\n[{i+1}/{n}] {title}")
        r = plot_full(title, ai_f, ai_b, t_end=3.0, n_steps=60000, filename=fn)
        results.append({"title": title, **r})

    print("\n" + "=" * 80)
    print(f"SUMMARY ({n} solutions)")
    print("=" * 80)
    print(f"{'Config':<45} {'Peak':>6} {'n90':>4} {'avgFz':>7} {'avgFx':>7} {'td_range':>16} {'|td|_avg':>8}")
    for r in results:
        short = r["title"].split("(")[0].strip()
        print(f"{short:<45} {r['peak']:5.1f}deg {r['n90']:3d}  {r['avg_Fz']:+6.0f}  {r['avg_Fx']:+6.0f}  "
              f"[{r['td_range'][0]:4.0f},{r['td_range'][1]:4.0f}]  {r['td_mean_abs']:5.1f}")

    print(f"\nDone. All plots in temp/all/")
