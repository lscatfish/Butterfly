#!/usr/bin/env python3
"""为 v6.2 最佳参数绘制完整时程图：θ_p, θ̇_p, Fz, Fx"""
import numpy as np, sys, json, matplotlib
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

    print(f"Simulating {title}...")
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

    # 最后 0.1 秒放大
    mask_zoom = t > (t_end - 0.1)

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle(f"{title}  |  Peak θ={peak:.1f}°  n90={n90}", fontsize=14, fontweight="bold")

    # (0,0) Pitch angle
    ax = axes[0, 0]
    ax.plot(t * 1000, tp_deg, "b-", lw=0.8)
    ax.axhline(90, color="r", ls="--", lw=0.8, alpha=0.4)
    ax.axhline(-90, color="r", ls="--", lw=0.8, alpha=0.4)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("θ_p (deg)")
    ax.set_title("Pitch Angle (full)")
    ax.grid(True, alpha=0.3)

    # (0,1) Pitch rate
    ax = axes[0, 1]
    ax.plot(t * 1000, td, "r-", lw=0.6, alpha=0.8)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("θ̇_p (rad/s)")
    ax.set_title(f"Pitch Rate  |  peak={np.max(np.abs(td)):.0f} rad/s")
    ax.grid(True, alpha=0.3)

    # (1,0) Lift
    ax = axes[1, 0]
    ax.plot(t * 1000, Fz_hist * 1000, "g-", lw=0.6, alpha=0.7)
    ax.axhline(weight_mN, color="r", ls=":", alpha=0.5, label=f"Weight={weight_mN:.0f}mN")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Lift Fz (mN)")
    ax.set_title(f"Lift  |  avg={np.mean(Fz_hist[n_steps//2:])*1000:+.0f} mN")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,1) Thrust
    ax = axes[1, 1]
    ax.plot(t * 1000, Fx_hist * 1000, "m-", lw=0.6, alpha=0.7)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Thrust Fx (mN)")
    ax.set_title(f"Thrust  |  avg={np.mean(Fx_hist[n_steps//2:])*1000:+.0f} mN")
    ax.grid(True, alpha=0.3)

    # (2,0) Pitch zoom (last 0.1s)
    ax = axes[2, 0]
    ax.plot(t[mask_zoom] * 1000, tp_deg[mask_zoom], "b-", lw=1.2)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("θ_p (deg)")
    ax.set_title(f"Pitch (last 0.1s zoom)  |  range=[{tp_deg[mask_zoom].min():.1f}°, {tp_deg[mask_zoom].max():.1f}°]")
    ax.grid(True, alpha=0.3)

    # (2,1) Phase portrait θ̇ vs θ
    ax = axes[2, 1]
    ax.plot(tp_deg[n_steps//2:], td[n_steps//2:], ".", ms=0.3, alpha=0.3, color="purple")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("θ_p (deg)"); ax.set_ylabel("θ̇_p (rad/s)")
    ax.set_title("Phase Portrait (steady)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")
    print(f"  θ̇_p range: [{td.min():.1f}, {td.max():.1f}] rad/s")
    print(f"  θ̇_p mean abs: {np.mean(np.abs(td[n_steps//2:])):.1f} rad/s")

if __name__ == "__main__":
    # v6.2 best + v6.1 high-lift
    configs = [
        ("v6.2 LEV-best: α_f=45°/α_b=10°", 45, 10, "temp/v62_best_45_10.png"),
        ("v6.1 high-lift: α_f=60°/α_b=10°", 60, 10, "temp/v61_high_60_10.png"),
    ]
    for title, ai_f, ai_b, fn in configs:
        plot_full(title, ai_f, ai_b, t_end=3.0, n_steps=60000, filename=fn)
    print("\nDone.")
