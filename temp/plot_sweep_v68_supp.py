#!/usr/bin/env python3
"""
v6.8 补充图表 — 物理合理性判定 + α_f 趋势 + 合理区放大
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")

# 字体
_FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
_found = None
for _f in _FONT_CANDIDATES:
    for _fm in font_manager.fontManager.ttflist:
        if _f.lower() in _fm.name.lower():
            _found = _fm.name
            break
    if _found: break
if _found:
    plt.rcParams["font.family"] = _found
else:
    plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False

SWEEP_DIRS = [
    Path("F:/重大作业考试/26秋/机械原理/全链路气动仿真/temp/stability/sweep_cartesian"),
    Path("temp/stability/sweep_cartesian"),
]
FIG_DIR = Path("output/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    records = []
    seen = set()
    for SWEEP_DIR in SWEEP_DIRS:
        if not SWEEP_DIR.exists(): continue
        for d in sorted(SWEEP_DIR.iterdir()):
            if not d.is_dir(): continue
            if d.name in seen: continue
            seen.add(d.name)
            sm = d / "summary.json"
            if not sm.exists(): continue
            try:
                with open(sm) as f: s = json.load(f)
            except: continue
            combo = s.pop("_combo", {})
            s.update(combo)
            records.append(s)
    df = pd.DataFrame(records)
    df["stable"] = df["n_exceed_90"] == 0
    return df

print("Loading...")
df = load_data()
stable = df[df.stable]
print(f"{len(df)} total, {len(stable)} stable")

COLORS = {
    0.3: "#2196F3", 0.5: "#4CAF50", 0.8: "#FF9800", 1.0: "#F44336", 1.5: "#9C27B0",
    2.00: "#2196F3", 2.25: "#4CAF50", 2.50: "#F44336",
    6: "#2196F3", 8: "#F44336",
}
AF_LIST = sorted(stable["alpha_front_deg"].unique())
KC_LIST = sorted(stable["k_clap"].unique())
R_LIST = sorted(stable["mech_R"].unique())

# ====================================================================
# Figure 8: 物理合理性全景 — α_f 趋势 + 约束分区
# ====================================================================
print("Figure 8: Physical reasonableness")

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle("Physical Reasonableness Analysis — v6.8 Full Dataset (11,622 combos)",
             fontsize=16, fontweight="bold", y=0.98)

# 8a: L/W max by α_f for each k_clap (stable only)
ax = axes[0, 0]
for kc in KC_LIST:
    g = stable[stable["k_clap"] == kc]
    lw_max = g.groupby("alpha_front_deg")["L/W"].max()
    ax.plot(lw_max.index, lw_max.values, "o-", color=COLORS.get(kc, "#999"),
            label=f"kc={kc:.1f}", markersize=8, linewidth=1.5)
ax.set_xlabel("α_f (°)")
ax.set_ylabel("Max L/W")
ax.set_title("Max L/W vs α_f by k_clap")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
# Add physical boundary
ax.axvspan(30, 50, alpha=0.08, color="green", label="Physically Reasonable")
ax.axvspan(50, 71, alpha=0.08, color="red", label="Non-Physical")
ax.legend(fontsize=7, loc="lower right")

# 8b: Stability rate by α_f
ax = axes[0, 1]
for kc in KC_LIST:
    rates = []
    for af in AF_LIST:
        g = df[(df["alpha_front_deg"] == af) & (df["k_clap"] == kc)]
        if len(g) > 0:
            rates.append(100 * (g["n_exceed_90"] == 0).sum() / len(g))
        else:
            rates.append(0)
    ax.plot(AF_LIST, rates, "o-", color=COLORS.get(kc, "#999"),
            label=f"kc={kc:.1f}", markersize=7, linewidth=1.5)
ax.set_xlabel("α_f (°)")
ax.set_ylabel("Stability Rate (%)")
ax.set_title("Stability Rate vs α_f")
ax.set_ylim(0, 100)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.axvspan(30, 50, alpha=0.08, color="green")
ax.axvspan(50, 71, alpha=0.08, color="red")

# 8c: Mean peak_θ by α_f
ax = axes[0, 2]
for kc in KC_LIST:
    g = stable[stable["k_clap"] == kc]
    theta_mean = g.groupby("alpha_front_deg")["peak_theta_deg"].mean()
    ax.plot(theta_mean.index, theta_mean.values, "o-", color=COLORS.get(kc, "#999"),
            label=f"kc={kc:.1f}", markersize=7, linewidth=1.5)
ax.set_xlabel("α_f (°)")
ax.set_ylabel("Mean peak_θ (°)")
ax.set_title("Mean peak_θ vs α_f")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.axhline(y=30, color="green", linestyle="--", alpha=0.5, label="θ=30°")
ax.axvspan(30, 50, alpha=0.08, color="green")
ax.axvspan(50, 71, alpha=0.08, color="red")

# 8d: Physical regime scatter — L/W vs α_f, colored by α_b, size by stability
ax = axes[1, 0]
for ab in sorted(stable["alpha_back_deg"].unique()):
    g = stable[stable["alpha_back_deg"] == ab].groupby("alpha_front_deg").agg(
        L_W_max=("L/W", "max"), n=("L/W", "count")
    ).reset_index()
    sizes = np.clip(g["n"], 20, 300)
    ax.scatter(g["alpha_front_deg"], g["L_W_max"], s=sizes, alpha=0.6,
               label=f"α_b={ab:.0f}")
ax.set_xlabel("α_f (°)")
ax.set_ylabel("Max L/W")
ax.set_title("Max L/W vs α_f by α_b (bubble size = sample count)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
ax.axvspan(30, 50, alpha=0.08, color="green")
ax.axvspan(50, 71, alpha=0.08, color="red")

# 8e: L/W distribution in reasonable (α_f=40-50) vs unreasonable (α_f=60-70) range
ax = axes[1, 1]
reasonable = stable[stable["alpha_front_deg"].isin([40, 50])]
unreasonable = stable[stable["alpha_front_deg"].isin([60, 70])]
ax.hist(reasonable["L/W"], bins=30, alpha=0.6, color="green",
        label=f"α_f=40-50° (n={len(reasonable)})", density=True)
ax.hist(unreasonable["L/W"], bins=30, alpha=0.6, color="red",
        label=f"α_f=60-70° (n={len(unreasonable)})", density=True)
ax.set_xlabel("L/W")
ax.set_ylabel("Density")
ax.set_title("L/W Distribution: Reasonable vs Non-Physical α_f")
ax.legend(fontsize=9)
ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5)

# 8f: Best combo per α_f — bar chart with α_eff indicator
ax = axes[1, 2]
best_per_af = []
for af in AF_LIST:
    g = stable[stable["alpha_front_deg"] == af]
    if len(g) == 0: continue
    best = g.loc[g["L/W"].idxmax()]
    best_per_af.append({
        "α_f": af, "L/W": best["L/W"], "α_b": best["alpha_back_deg"],
        "θ": best["peak_theta_deg"], "ph": best["phase_diff_deg"],
        "kc": best["k_clap"],
    })
best_df = pd.DataFrame(best_per_af)
colors_bar = ["green" if af <= 50 else "red" for af in best_df["α_f"]]
bars = ax.bar(range(len(best_df)), best_df["L/W"], color=colors_bar, alpha=0.7,
              edgecolor="white")
ax.set_xticks(range(len(best_df)))
ax.set_xticklabels([f"α_f={af:.0f}\nα_b={ab:.0f}\nph={ph:.0f},kc={kc:.1f}"
                    for af, ab, ph, kc in zip(best_df["α_f"], best_df["α_b"],
                                              best_df["ph"], best_df["kc"])],
                   fontsize=7)
ax.set_ylabel("Max L/W")
ax.set_title("Best per α_f (green=reasonable, red=non-physical)")
for bar, val in zip(bars, best_df["L/W"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(FIG_DIR / "fig8_physical_reasonableness.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure 8 done")

# ====================================================================
# Figure 9: 合理区放大 — α_f=40-50 局部精调
# ====================================================================
print("Figure 9: Reasonable range zoom")

reasonable = stable[stable["alpha_front_deg"].isin([40, 50])]

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle("Physically Reasonable Range (α_f=40-50°) — High-Resolution View",
             fontsize=16, fontweight="bold", y=0.98)

# 9a: α_f=40 vs 50: L/W boxplot by α_b
ax = axes[0, 0]
ab_vals = sorted(reasonable["alpha_back_deg"].unique())
positions, data, colors_ab = [], [], []
for i, ab in enumerate(ab_vals):
    for j, af in enumerate([40, 50]):
        g = reasonable[(reasonable["alpha_back_deg"] == ab) & (reasonable["alpha_front_deg"] == af)]
        if len(g) > 0:
            pos = i * 3 + j
            positions.append(pos)
            data.append(g["L/W"].values)
            colors_ab.append("#4CAF50" if af == 40 else "#2196F3")
bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
for patch, c in zip(bp["boxes"], colors_ab):
    patch.set_facecolor(c)
    patch.set_alpha(0.5)
# Legend
from matplotlib.patches import Patch
ax.legend([Patch(facecolor="#4CAF50", alpha=0.5), Patch(facecolor="#2196F3", alpha=0.5)],
          ["α_f=40°", "α_f=50°"], fontsize=9)
ax.set_xticks([i * 3 + 0.5 for i in range(len(ab_vals))])
ax.set_xticklabels([f"α_b={ab:.0f}" for ab in ab_vals])
ax.set_ylabel("L/W")
ax.set_title("α_f=40° vs 50°: L/W by α_b")
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)
ax.grid(axis="y", alpha=0.2)

# 9b: α_f=40 best params detail
ax = axes[0, 1]
af40 = reasonable[reasonable["alpha_front_deg"] == 40]
for ab in sorted(af40["alpha_back_deg"].unique()):
    g = af40[af40["alpha_back_deg"] == ab]
    if len(g) == 0: continue
    for kc in [0.3, 0.5]:
        gg = g[g["k_clap"] == kc]
        if len(gg) == 0: continue
        best = gg.loc[gg["L/W"].idxmax()]
        marker = "o" if kc == 0.3 else "s"
        ax.scatter(best["phase_diff_deg"], best["L/W"], marker=marker,
                   s=100, alpha=0.7, label=f"α_b={ab:.0f},kc={kc:.1f}")
ax.set_xlabel("phase (°)")
ax.set_ylabel("Best L/W")
ax.set_title("α_f=40°: Best per (α_b, ph, kc)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.2)

# 9c: α_f=50 best params detail
ax = axes[0, 2]
af50 = reasonable[reasonable["alpha_front_deg"] == 50]
for ab in sorted(af50["alpha_back_deg"].unique()):
    g = af50[af50["alpha_back_deg"] == ab]
    if len(g) == 0: continue
    for kc in [0.3, 0.5]:
        gg = g[g["k_clap"] == kc]
        if len(gg) == 0: continue
        best = gg.loc[gg["L/W"].idxmax()]
        marker = "o" if kc == 0.3 else "s"
        ax.scatter(best["phase_diff_deg"], best["L/W"], marker=marker,
                   s=100, alpha=0.7, label=f"α_b={ab:.0f},kc={kc:.1f}")
ax.set_xlabel("phase (°)")
ax.set_ylabel("Best L/W")
ax.set_title("α_f=50°: Best per (α_b, ph, kc)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.2)

# 9d: Reasonable range — L/W vs peak_θ scatter with DESIGN markers
ax = axes[1, 0]
for af in [40, 50]:
    for ab in [3, 5, 8, 10, 15]:
        g = reasonable[(reasonable["alpha_front_deg"] == af) & (reasonable["alpha_back_deg"] == ab)]
        if len(g) == 0: continue
        c = {3: "#E91E63", 5: "#FF9800", 8: "#4CAF50", 10: "#2196F3", 15: "#9C27B0"}[ab]
        ax.scatter(g["peak_theta_deg"], g["L/W"], c=c, alpha=0.3, s=8,
                   edgecolors="none")
# Mark DESIGN points
designs = {
    "v6.8 (α_f=45)": (32.9, 2.447, "red", "*"),
    "v6.8_safe (α_f=40)": (25.4, 2.128, "green", "D"),
    "v6.8_agg (α_f=50)": (28.4, 2.807, "orange", "s"),
}
for label, (x, y, color, marker) in designs.items():
    ax.scatter([x], [y], c=color, s=250, marker=marker, edgecolors="black",
               linewidth=1.5, zorder=5, label=label)
ax.set_xlabel("peak_θ (°)")
ax.set_ylabel("L/W")
ax.set_title("Reasonable Range: Performance Space")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)

# 9e: α_f=40 vs 50 — φ_off effect on L/W
ax = axes[1, 1]
for af in [40, 50]:
    g = reasonable[reasonable["alpha_front_deg"] == af]
    po_means = g.groupby("phi_offset_deg")["L/W"].agg(["mean", "max"])
    ax.plot(po_means.index, po_means["mean"], "o-",
            color="#4CAF50" if af == 40 else "#2196F3",
            label=f"α_f={af}° mean", linewidth=1.5)
    ax.fill_between(po_means.index, po_means["mean"] - g.groupby("phi_offset_deg")["L/W"].std(),
                    po_means["mean"] + g.groupby("phi_offset_deg")["L/W"].std(),
                    alpha=0.15, color="#4CAF50" if af == 40 else "#2196F3")
ax.set_xlabel("φ_off (°)")
ax.set_ylabel("Mean L/W")
ax.set_title("φ_off Effect on L/W (α_f=40° vs 50°)")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

# 9f: R comparison in reasonable range
ax = axes[1, 2]
x = np.arange(2)
width = 0.25
for ri, rval in enumerate(R_LIST):
    bests_40, bests_50 = [], []
    for af in [40, 50]:
        g = reasonable[(reasonable["mech_R"] == rval) & (reasonable["alpha_front_deg"] == af)]
        bests_40.append(g["L/W"].max() if len(g) > 0 else 0) if af == 40 else bests_50.append(g["L/W"].max() if len(g) > 0 else 0)
    g40 = reasonable[(reasonable["mech_R"] == rval) & (reasonable["alpha_front_deg"] == 40)]
    g50 = reasonable[(reasonable["mech_R"] == rval) & (reasonable["alpha_front_deg"] == 50)]
    bests_40 = [g40["L/W"].max() if len(g40) > 0 else 0]
    bests_50 = [g50["L/W"].max() if len(g50) > 0 else 0]
    # Redo properly
# Redo 9f properly
# Actually let me redo this part cleanly
ax.clear()
for ri, rval in enumerate(R_LIST):
    bests = []
    for ai, af in enumerate([40, 50]):
        g = reasonable[(reasonable["mech_R"] == rval) & (reasonable["alpha_front_deg"] == af)]
        bests.append(g["L/W"].max() if len(g) > 0 else 0)
    ax.bar(x + ri * width, bests, width, color=COLORS.get(rval, "#999"),
           alpha=0.8, label=f"R={rval:.2f}")
ax.set_xticks(x + width)
ax.set_xticklabels(["α_f=40°", "α_f=50°"])
ax.set_xlabel("α_f")
ax.set_ylabel("Max L/W")
ax.set_title("Max L/W by R in Reasonable Range")
ax.legend(fontsize=9)
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(FIG_DIR / "fig9_reasonable_range_zoom.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure 9 done")

# ====================================================================
# Figure 10: 全量散点 — 极限性能全景 (α_f=30-70)
# ====================================================================
print("Figure 10: Full range performance scatter")

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
fig.suptitle("Full Performance Space — All 7,909 Stable Combos (α_f=30-70°)",
             fontsize=16, fontweight="bold")

# Color by α_f
af_colors = {30: "#0D47A1", 40: "#2196F3", 50: "#4CAF50", 55: "#8BC34A",
             60: "#FF9800", 70: "#F44336"}
for af in sorted(stable["alpha_front_deg"].unique()):
    g = stable[stable["alpha_front_deg"] == af]
    ax.scatter(g["peak_theta_deg"], g["L/W"], c=af_colors.get(af, "#999"),
               alpha=0.4, s=8, label=f"α_f={af:.0f}° (n={len(g)})",
               edgecolors="none")

# Physical boundary
ax.axvline(x=30, color="green", linestyle="--", alpha=0.4, linewidth=1.5)
ax.annotate("θ=30°", xy=(30, 0.5), fontsize=9, color="green")

# DESIGN markers
designs = [
    (29.5, 2.904, "DESIGN_v68 (α_f=55) [OLD]", "blue", "s"),
    (32.9, 2.447, "DESIGN_v68 (α_f=45) [NEW]", "green", "D"),
    (24.2, 2.381, "v68_safe (α_f=40)", "cyan", "o"),
]
for x, y, label, c, m in designs:
    ax.scatter([x], [y], c=c, s=300, marker=m, edgecolors="black",
               linewidth=2, zorder=10, label=label)

# Shade non-physical region
ax.axvspan(60, 91, alpha=0.06, color="red")
ax.annotate("Non-Physical\n(mean α_eff>60°)", xy=(75, 3.5), fontsize=11,
            color="red", ha="center", fontweight="bold")
ax.axvspan(0, 50, alpha=0.04, color="green")
ax.annotate("Physically\nReasonable\n(mean α_eff<50°)", xy=(25, 3.5), fontsize=11,
            color="green", ha="center", fontweight="bold")

ax.set_xlabel("peak_θ (°)", fontsize=13)
ax.set_ylabel("L/W", fontsize=13)
ax.legend(fontsize=8, ncol=2, loc="lower left")
ax.grid(alpha=0.15)
ax.set_xlim(0, 95)
ax.set_ylim(0, 4.0)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(FIG_DIR / "fig10_full_performance_landscape.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure 10 done")

print(f"\nAll 3 supplementary figures saved to {FIG_DIR}")
