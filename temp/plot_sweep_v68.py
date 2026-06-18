#!/usr/bin/env python3
"""
v6.8 扫参分析可视化 — 基于全量 11,622 组数据 (α_f=30-70)
输出所有图表到 output/figures/
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# ============================================================
# 字体设置
# ============================================================
_FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                     "WenQuanYi Micro Hei", "DejaVu Sans"]
_found = None
for _f in _FONT_CANDIDATES:
    for _fm in font_manager.fontManager.ttflist:
        if _f.lower() in _fm.name.lower():
            _found = _fm.name
            break
    if _found:
        break

# 如果找不到中文字体，设置为 sans-serif
if _found:
    plt.rcParams["font.family"] = _found
else:
    # Fallback: 使用 sans-serif，但中文标签用英文替代
    plt.rcParams["font.family"] = "sans-serif"
    print("Warning: No CJK font found, using English labels")
    _USE_EN = True
_USE_EN = _found is None

plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 路径
# ============================================================
SWEEP_DIRS = [
    Path("F:/重大作业考试/26秋/机械原理/全链路气动仿真/temp/stability/sweep_cartesian"),
    Path("temp/stability/sweep_cartesian"),
]
FIG_DIR = Path("output/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 数据加载
# ============================================================
def load_data():
    records = []
    seen = set()
    for SWEEP_DIR in SWEEP_DIRS:
        if not SWEEP_DIR.exists():
            continue
        for d in sorted(SWEEP_DIR.iterdir()):
            if not d.is_dir():
                continue
            if d.name in seen:
                continue
            seen.add(d.name)
            sm = d / "summary.json"
            if not sm.exists():
                continue
            try:
                with open(sm) as f:
                    s = json.load(f)
            except Exception:
                continue
            combo = s.pop("_combo", {})
            s.update(combo)
            records.append(s)
    df = pd.DataFrame(records)
    df["stable"] = df["n_exceed_90"] == 0
    df["L_W_bin"] = pd.cut(df["L/W"], bins=20)
    return df

print("Loading data...")
df = load_data()
stable = df[df.stable]
print(f"Loaded {len(df)} results, {len(stable)} stable")

# ============================================================
# 辅助函数
# ============================================================
def label(text):
    """返回中文标签，如果无字体则返回英文"""
    _EN_MAP = {
        "k_clap": "k_clap", "稳定率 (%)": "Stability (%)",
        "L/W": "L/W", "L/W 均值": "L/W mean",
        "L/W max": "L/W max", "L/W ≥ 2.0 组数": "L/W ≥ 2.0 count",
        "L/W 分布": "L/W distribution", "概率密度": "Density",
        "稳定": "Stable", "不稳定": "Unstable",
        "前翅安装角 α_f (°)": "α_f (°)",
        "后翅安装角 α_b (°)": "α_b (°)",
        "峰俯仰角 peak_θ (°)": "peak_θ (°)",
        "相位差 phase (°)": "phase (°)",
        "摇杆半径 R (mm)": "R (mm)",
        "曲柄半径 a (mm)": "a (mm)",
        "安装偏角 φ_off (°)": "φ_off (°)",
        "频率 f (Hz)": "f (Hz)",
        "分组": "Group",
        "数量": "Count",
        "最大俯仰峰": "Max peak θ",
        "平均俯仰峰": "Mean peak θ",
    }
    if _USE_EN:
        return _EN_MAP.get(text, text)
    return text

def save(name):
    path = FIG_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")

# ============================================================
# 颜色方案
# ============================================================
COLORS = {
    0.3: "#2196F3", 0.5: "#4CAF50", 0.8: "#FF9800",
    1.0: "#F44336", 1.5: "#9C27B0",
    2.00: "#2196F3", 2.25: "#4CAF50", 2.50: "#F44336",
    6: "#2196F3", 7: "#FF9800", 8: "#F44336",
    15: "#2196F3", 17: "#F44336",
}
KC_LIST = sorted(df["k_clap"].unique())
R_LIST = sorted(df["mech_R"].unique())
A_LIST = sorted(df["mech_a"].unique())

def kc_color(kc):
    return COLORS.get(kc, "#999999")

def r_color(r):
    return COLORS.get(r, "#999999")

# ====================================================================
# Figure 1: k_clap 敏感性全景 (2×3)
# ====================================================================
print("\nFigure 1: k_clap sensitivity")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("k_clap Sensitivity Analysis — v6.8 Sweep (10,326 combos)",
             fontsize=16, fontweight="bold", y=0.98)

# 1a: L/W boxplot by k_clap
ax = axes[0, 0]
data_by_kc = [stable[stable["k_clap"] == kc]["L/W"].values for kc in KC_LIST]
bp = ax.boxplot(data_by_kc, labels=[f"{kc:.1f}" for kc in KC_LIST],
                patch_artist=True, widths=0.6)
for i, (kc, patch) in enumerate(zip(KC_LIST, bp["boxes"])):
    patch.set_facecolor(kc_color(kc))
    patch.set_alpha(0.6)
ax.set_xlabel("k_clap")
ax.set_ylabel("L/W")
ax.set_title("L/W Distribution by k_clap (Stable Only)")
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5, label="L/W=2.0")
ax.legend(fontsize=8)

# 1b: Stability rate bar
ax = axes[0, 1]
rates = []
for kc in KC_LIST:
    g = df[df["k_clap"] == kc]
    rates.append(100 * (g["n_exceed_90"] == 0).sum() / len(g))
bars = ax.bar(range(len(KC_LIST)), rates, color=[kc_color(k) for k in KC_LIST],
              edgecolor="white", linewidth=1.2)
ax.set_xticks(range(len(KC_LIST)))
ax.set_xticklabels([f"{k:.1f}" for k in KC_LIST])
ax.set_xlabel("k_clap")
ax.set_ylabel("Stability Rate (%)")
ax.set_title("Stability Rate by k_clap")
ax.set_ylim(0, 100)
for bar, r in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{r:.1f}%", ha="center", fontsize=10, fontweight="bold")

# 1c: Percentage of L/W ≥ 2.0
ax = axes[0, 2]
lw2_counts, lw2_pcts = [], []
for kc in KC_LIST:
    g = stable[stable["k_clap"] == kc]
    lw2_counts.append((g["L/W"] >= 2.0).sum())
    lw2_pcts.append(100 * (g["L/W"] >= 2.0).sum() / len(g))
bars = ax.bar(range(len(KC_LIST)), lw2_counts, color=[kc_color(k) for k in KC_LIST],
              edgecolor="white", linewidth=1.2)
ax.set_xticks(range(len(KC_LIST)))
ax.set_xticklabels([f"{k:.1f}" for k in KC_LIST])
ax.set_xlabel("k_clap")
ax.set_ylabel("Count")
ax.set_title("Combos with L/W ≥ 2.0 by k_clap")
for bar, c, p in zip(bars, lw2_counts, lw2_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{c}\n({p:.1f}%)", ha="center", fontsize=9)

# 1d: L/W histogram (all k_clap overlaid)
ax = axes[1, 0]
for kc in KC_LIST:
    g = stable[stable["k_clap"] == kc]
    ax.hist(g["L/W"], bins=30, alpha=0.5, color=kc_color(kc),
            label=f"kc={kc:.1f}", density=True)
ax.set_xlabel("L/W")
ax.set_ylabel("Density")
ax.set_title("L/W Distribution (Stable, by k_clap)")
ax.legend(fontsize=8, ncol=2)
ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5)

# 1e: L/W mean + std errorbar
ax = axes[1, 1]
means, stds = [], []
for kc in KC_LIST:
    g = stable[stable["k_clap"] == kc]
    means.append(g["L/W"].mean())
    stds.append(g["L/W"].std())
ax.errorbar(range(len(KC_LIST)), means, yerr=stds, fmt="o-", capsize=8,
            markersize=10, linewidth=2, color="#333333")
for i, (kc, m) in enumerate(zip(KC_LIST, means)):
    ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points",
                xytext=(0, 15), ha="center", fontsize=9, fontweight="bold",
                color=kc_color(kc))
ax.set_xticks(range(len(KC_LIST)))
ax.set_xticklabels([f"{k:.1f}" for k in KC_LIST])
ax.set_xlabel("k_clap")
ax.set_ylabel("L/W")
ax.set_title("Mean L/W ± Std by k_clap (Stable)")
ax.set_ylim(bottom=0.8)

# 1f: L/W max + peak_θ max
ax = axes[1, 2]
lw_maxs, theta_maxs = [], []
for kc in KC_LIST:
    g = stable[stable["k_clap"] == kc]
    lw_maxs.append(g["L/W"].max())
    theta_maxs.append(g["peak_theta_deg"].max())
x = np.arange(len(KC_LIST))
width = 0.35
bars1 = ax.bar(x - width/2, lw_maxs, width, color=[kc_color(k) for k in KC_LIST],
               alpha=0.7, label="L/W max")
ax.set_ylabel("L/W max", color="#333")
ax.set_xlabel("k_clap")
ax.set_xticks(x)
ax.set_xticklabels([f"{k:.1f}" for k in KC_LIST])
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, theta_maxs, width, color="gray", alpha=0.3,
                label="peak_θ max (°)")
ax2.set_ylabel("peak_θ max (°)", color="gray")
ax.set_title("Max L/W & peak_θ by k_clap")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save("fig1_kclap_sensitivity.png")
plt.close()
print("  Figure 1 done")

# ====================================================================
# Figure 2: α_f × α_b 热力图 (2×3, 5个k_clap + 汇总)
# ====================================================================
print("\nFigure 2: α_f × α_b heatmaps")

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle("α_f × α_b Heatmaps — L/W (Stable, by k_clap)",
             fontsize=16, fontweight="bold", y=0.98)

af_vals = sorted(stable["alpha_front_deg"].unique())
ab_vals = sorted(stable["alpha_back_deg"].unique())

for idx, kc in enumerate(KC_LIST):
    ax = axes[idx // 3, idx % 3]
    g = stable[stable["k_clap"] == kc]
    pivot = g.pivot_table(values="L/W", index="alpha_back_deg",
                           columns="alpha_front_deg", aggfunc="max")
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   cmap="RdYlGn", vmin=0.5, vmax=3.2)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.0f}" for v in pivot.index])
    ax.set_xlabel("α_f (°)")
    ax.set_ylabel("α_b (°)")
    ax.set_title(f"k_clap = {kc:.1f}")
    # 标注数值
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 2.0 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

# 汇总面板
ax = axes[1, 2]
pivot_all = stable.pivot_table(values="L/W", index="alpha_back_deg",
                                columns="alpha_front_deg", aggfunc="max")
im = ax.imshow(pivot_all.values, aspect="auto", origin="lower",
               cmap="RdYlGn", vmin=0.5, vmax=3.2)
ax.set_xticks(range(len(pivot_all.columns)))
ax.set_xticklabels([f"{v:.0f}" for v in pivot_all.columns])
ax.set_yticks(range(len(pivot_all.index)))
ax.set_yticklabels([f"{v:.0f}" for v in pivot_all.index])
ax.set_xlabel("α_f (°)")
ax.set_ylabel("α_b (°)")
ax.set_title("All k_clap Combined (max L/W)")
for i in range(len(pivot_all.index)):
    for j in range(len(pivot_all.columns)):
        val = pivot_all.values[i, j]
        if not np.isnan(val):
            color = "white" if val > 2.0 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])
save("fig2_alpha_heatmap.png")
plt.close()
print("  Figure 2 done")

# ====================================================================
# Figure 3: 机构参数比较 — R × a (2×2)
# ====================================================================
print("\nFigure 3: Mechanism parameters")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Mechanism Parameter Analysis (R, a) — v6.8 Sweep",
             fontsize=16, fontweight="bold", y=0.98)

# 3a: L/W by R (boxplot, stable only, grouped by k_clap)
ax = axes[0, 0]
positions = []
data_r, colors_r, labels_r = [], [], []
for ri, rval in enumerate(R_LIST):
    for ki, kc in enumerate(KC_LIST):
        g = stable[(stable["mech_R"] == rval) & (stable["k_clap"] == kc)]
        if len(g) > 0:
            pos = ri * (len(KC_LIST) + 1) + ki
            positions.append(pos)
            data_r.append(g["L/W"].values)
            colors_r.append(kc_color(kc))
            labels_r.append(f"R={rval:.2f}\nkc={kc:.1f}")
bp = ax.boxplot(data_r, positions=positions, patch_artist=True,
                widths=0.5, manage_ticks=False)
for patch, color in zip(bp["boxes"], colors_r):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
# Add R group separators
for ri in range(1, len(R_LIST)):
    ax.axvline(x=ri * (len(KC_LIST) + 1) - 0.5, color="gray",
               linestyle="--", alpha=0.3, linewidth=0.8)
ax.set_xticks(positions[::len(KC_LIST)])
ax.set_xticklabels([f"R={r:.2f}" for r in R_LIST], fontsize=11, fontweight="bold")
ax.set_ylabel("L/W (Stable)")
ax.set_title("L/W by R and k_clap")
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)

# 3b: Stability rate by R and k_clap
ax = axes[0, 1]
x = np.arange(len(R_LIST))
width = 0.15
for ki, kc in enumerate(KC_LIST):
    rates_r = []
    for rval in R_LIST:
        g = df[(df["mech_R"] == rval) & (df["k_clap"] == kc)]
        rates_r.append(100 * (g["n_exceed_90"] == 0).sum() / len(g) if len(g) > 0 else 0)
    ax.bar(x + ki * width, rates_r, width, color=kc_color(kc), alpha=0.8,
           label=f"kc={kc:.1f}")
ax.set_xticks(x + width * 2)
ax.set_xticklabels([f"{r:.2f}" for r in R_LIST])
ax.set_xlabel("R (mm)")
ax.set_ylabel("Stability Rate (%)")
ax.set_title("Stability Rate by R and k_clap")
ax.set_ylim(0, 100)
ax.legend(fontsize=8, ncol=3)

# 3c: L/W by a (boxplot)
ax = axes[1, 0]
data_a = [stable[stable["mech_a"] == a]["L/W"].values for a in A_LIST]
bp = ax.boxplot(data_a, labels=[f"a={a:.0f}" for a in A_LIST],
                patch_artist=True, widths=0.5)
for patch, a_val in zip(bp["boxes"], A_LIST):
    patch.set_facecolor(COLORS.get(a_val, "#999"))
    patch.set_alpha(0.6)
ax.set_xlabel("a (mm)")
ax.set_ylabel("L/W (Stable)")
ax.set_title("L/W by mech_a")
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)

# 3d: L/W max by R × a
ax = axes[1, 1]
x = np.arange(len(R_LIST))
width = 0.25
for ai, a_val in enumerate(A_LIST):
    maxes = []
    for rval in R_LIST:
        g = stable[(stable["mech_R"] == rval) & (stable["mech_a"] == a_val)]
        maxes.append(g["L/W"].max() if len(g) > 0 else 0)
    ax.bar(x + ai * width, maxes, width, color=COLORS.get(a_val, "#999"),
           alpha=0.8, label=f"a={a_val:.0f}")
ax.set_xticks(x + width)
ax.set_xticklabels([f"{r:.2f}" for r in R_LIST])
ax.set_xlabel("R (mm)")
ax.set_ylabel("Max L/W (Stable)")
ax.set_title("Max L/W by R and a")
ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save("fig3_mechanism_params.png")
plt.close()
print("  Figure 3 done")

# ====================================================================
# Figure 4: Phase 和 φ_off 敏感性 (2×3)
# ====================================================================
print("\nFigure 4: Phase and phi_offset effects")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Phase & φ_offset Sensitivity — v6.8 Sweep",
             fontsize=16, fontweight="bold", y=0.98)

phase_list = sorted(stable["phase_diff_deg"].unique())
po_list = sorted(stable["phi_offset_deg"].unique())

# 4a: L/W by phase for each k_clap
ax = axes[0, 0]
for kc in KC_LIST:
    g = stable[stable["k_clap"] == kc].groupby("phase_diff_deg")["L/W"]
    means = g.mean()
    ax.plot(means.index, means.values, "o-", color=kc_color(kc),
            label=f"kc={kc:.1f}", markersize=7, linewidth=1.5)
ax.set_xlabel("phase (°)")
ax.set_ylabel("Mean L/W")
ax.set_title("L/W vs phase by k_clap")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 4b: Stability rate by phase for each k_clap
ax = axes[0, 1]
for kc in KC_LIST:
    rates_ph = []
    for ph in phase_list:
        g = df[(df["phase_diff_deg"] == ph) & (df["k_clap"] == kc)]
        rates_ph.append(100 * (g["n_exceed_90"] == 0).sum() / len(g) if len(g) > 0 else 0)
    ax.plot(phase_list, rates_ph, "o-", color=kc_color(kc),
            label=f"kc={kc:.1f}", markersize=7, linewidth=1.5)
ax.set_xlabel("phase (°)")
ax.set_ylabel("Stability Rate (%)")
ax.set_title("Stability vs phase by k_clap")
ax.set_ylim(0, 100)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 4c: Mean peak_θ by phase
ax = axes[0, 2]
for kc in KC_LIST:
    g = stable[stable["k_clap"] == kc].groupby("phase_diff_deg")["peak_theta_deg"]
    means = g.mean()
    ax.plot(means.index, means.values, "o-", color=kc_color(kc),
            label=f"kc={kc:.1f}", markersize=7, linewidth=1.5)
ax.set_xlabel("phase (°)")
ax.set_ylabel("Mean peak θ (°)")
ax.set_title("Mean peak_θ vs phase by k_clap")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 4d: L/W by φ_off for each k_clap
ax = axes[1, 0]
for kc in KC_LIST:
    g = stable[stable["k_clap"] == kc].groupby("phi_offset_deg")["L/W"]
    means = g.mean()
    ax.plot(means.index, means.values, "o-", color=kc_color(kc),
            label=f"kc={kc:.1f}", markersize=7, linewidth=1.5)
ax.set_xlabel("φ_off (°)")
ax.set_ylabel("Mean L/W")
ax.set_title("L/W vs φ_off by k_clap")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 4e: Stability rate by φ_off
ax = axes[1, 1]
for kc in KC_LIST:
    rates_po = []
    for po in po_list:
        g = df[(df["phi_offset_deg"] == po) & (df["k_clap"] == kc)]
        rates_po.append(100 * (g["n_exceed_90"] == 0).sum() / len(g) if len(g) > 0 else 0)
    ax.plot(po_list, rates_po, "o-", color=kc_color(kc),
            label=f"kc={kc:.1f}", markersize=7, linewidth=1.5)
ax.set_xlabel("φ_off (°)")
ax.set_ylabel("Stability Rate (%)")
ax.set_title("Stability vs φ_off by k_clap")
ax.set_ylim(0, 100)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 4f: L/W by phase × φ_off (heatmap, kc=0.3)
ax = axes[1, 2]
g = stable[stable["k_clap"] == 0.3]
pivot = g.pivot_table(values="L/W", index="phi_offset_deg",
                       columns="phase_diff_deg", aggfunc="max")
im = ax.imshow(pivot.values, aspect="auto", origin="lower",
               cmap="RdYlGn", vmin=1.0, vmax=3.2)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"{v:.0f}" for v in pivot.index])
ax.set_xlabel("phase (°)")
ax.set_ylabel("φ_off (°)")
ax.set_title("L/W: φ_off × phase (kc=0.3)")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            color = "white" if val > 2.0 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save("fig4_phase_phioff.png")
plt.close()
print("  Figure 4 done")

# ====================================================================
# Figure 5: L/W vs peak_θ 散点图 + R=2.25 最优区 (2×2)
# ====================================================================
print("\nFigure 5: L/W vs peak_θ scatter")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Performance Space: L/W vs peak_θ — v6.8 Sweep",
             fontsize=16, fontweight="bold", y=0.98)

# 5a: All stable combos, colored by k_clap
ax = axes[0, 0]
for kc in KC_LIST:
    g = stable[stable["k_clap"] == kc]
    ax.scatter(g["peak_theta_deg"], g["L/W"], c=kc_color(kc),
               alpha=0.4, s=12, label=f"kc={kc:.1f}", edgecolors="none")
ax.set_xlabel("peak_θ (°)")
ax.set_ylabel("L/W")
ax.set_title("Stable Combos — Colored by k_clap")
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5)
ax.axvline(x=30, color="green", linestyle="--", alpha=0.3, label="θ=30°")
ax.legend(fontsize=8, markerscale=2)
ax.grid(alpha=0.2)

# 5b: Stable combos, colored by R
ax = axes[0, 1]
for rval in R_LIST:
    g = stable[stable["mech_R"] == rval]
    ax.scatter(g["peak_theta_deg"], g["L/W"], c=r_color(rval),
               alpha=0.4, s=12, label=f"R={rval:.2f}", edgecolors="none")
ax.set_xlabel("peak_θ (°)")
ax.set_ylabel("L/W")
ax.set_title("Stable Combos — Colored by R")
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5)
ax.axvline(x=30, color="green", linestyle="--", alpha=0.3, label="θ=30°")
ax.legend(fontsize=8, markerscale=2)
ax.grid(alpha=0.2)

# 5c: R=2.25 zoom, colored by α_b
ax = axes[1, 0]
g = stable[stable["mech_R"] == 2.25]
ab_list = sorted(g["alpha_back_deg"].unique())
ab_colors = {3: "#E91E63", 5: "#FF9800", 8: "#4CAF50", 10: "#2196F3", 15: "#9C27B0"}
for ab in ab_list:
    gg = g[g["alpha_back_deg"] == ab]
    ax.scatter(gg["peak_theta_deg"], gg["L/W"], c=ab_colors.get(ab, "#999"),
               alpha=0.5, s=15, label=f"α_b={ab:.0f}", edgecolors="none")
ax.set_xlabel("peak_θ (°)")
ax.set_ylabel("L/W")
ax.set_title("R=2.25 Stable Combos — Colored by α_b")
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5)
ax.legend(fontsize=8, markerscale=2)
ax.grid(alpha=0.2)
# Mark DESIGN_v68_R225
ax.scatter([54.0], [2.370], c="red", s=200, marker="*", edgecolors="black",
           linewidth=1.5, zorder=5, label="DESIGN_v68_R225")
ax.legend(fontsize=8, markerscale=1)

# 5d: R=2.50 zoom, colored by α_b
ax = axes[1, 1]
g = stable[stable["mech_R"] == 2.50]
for ab in ab_list:
    gg = g[g["alpha_back_deg"] == ab]
    ax.scatter(gg["peak_theta_deg"], gg["L/W"], c=ab_colors.get(ab, "#999"),
               alpha=0.5, s=15, label=f"α_b={ab:.0f}", edgecolors="none")
ax.set_xlabel("peak_θ (°)")
ax.set_ylabel("L/W")
ax.set_title("R=2.50 Stable Combos — Colored by α_b")
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5)
ax.legend(fontsize=8, markerscale=2)
ax.grid(alpha=0.2)
# Mark DESIGN_v68
ax.scatter([29.5], [2.904], c="red", s=200, marker="*", edgecolors="black",
           linewidth=1.5, zorder=5, label="DESIGN_v68")
ax.legend(fontsize=8, markerscale=1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save("fig5_performance_scatter.png")
plt.close()
print("  Figure 5 done")

# ====================================================================
# Figure 6: Top 参数频率 + R=2.25 详细 (2×3)
# ====================================================================
print("\nFigure 6: Top combo analysis")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Top Combos & DESIGN_v68_R225 Detail — v6.8 Sweep",
             fontsize=16, fontweight="bold", y=0.98)

# 6a: Parameter distribution in Top 100 L/W
ax = axes[0, 0]
top100 = stable.nlargest(100, "L/W")
param_counts = {}
for col, label_name in [("alpha_front_deg", "α_f"), ("alpha_back_deg", "α_b"),
                          ("k_clap", "kc"), ("mech_a", "a"), ("mech_R", "R"),
                          ("phase_diff_deg", "ph"), ("phi_offset_deg", "φ_off"),
                          ("f", "f")]:
    counts = top100[col].value_counts().sort_index()
    for v, c in counts.items():
        param_counts[f"{label_name}={v}"] = c
# Plot top 15 most common
top_params = sorted(param_counts.items(), key=lambda x: -x[1])[:15]
bars = ax.barh(range(len(top_params)), [x[1] for x in top_params],
               color="#2196F3", edgecolor="white")
ax.set_yticks(range(len(top_params)))
ax.set_yticklabels([x[0] for x in top_params])
ax.set_xlabel("Count in Top 100")
ax.set_title("Parameter Frequency in Top 100 L/W")
ax.invert_yaxis()

# 6b: R=2.25, α_f=50, kc=0.3: α_b vs L/W with peak_θ annotations
ax = axes[0, 1]
g_r25 = stable[(stable["mech_R"] == 2.25) & (stable["alpha_front_deg"] == 50) &
               (stable["k_clap"] == 0.3) & (stable["f"] == 17) & (stable["mech_a"] == 6)]
for ab in sorted(g_r25["alpha_back_deg"].unique()):
    gg = g_r25[g_r25["alpha_back_deg"] == ab]
    if len(gg) == 0: continue
    best = gg.loc[gg["L/W"].idxmax()]
    ax.scatter(ab, best["L/W"], s=120, c=ab_colors.get(ab, "#999"),
               edgecolors="black", linewidth=1, zorder=3)
    ax.annotate(f"θ={best['peak_theta_deg']:.0f}°\nph={best['phase_diff_deg']:.0f}\nφ={best['phi_offset_deg']:.0f}",
                (ab, best["L/W"]), textcoords="offset points",
                xytext=(10, 0), fontsize=7, alpha=0.8)
ax.set_xlabel("α_b (°)")
ax.set_ylabel("Best L/W")
ax.set_title("R=2.25, α_f=50°, kc=0.3: Best per α_b")
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)
ax.grid(alpha=0.2)

# 6c: R=2.50, α_f=55, kc=0.3: α_b vs L/W
ax = axes[0, 2]
g_r50 = stable[(stable["mech_R"] == 2.50) & (stable["alpha_front_deg"] == 55) &
               (stable["k_clap"] == 0.3) & (stable["f"] == 17) & (stable["mech_a"] == 6)]
for ab in sorted(g_r50["alpha_back_deg"].unique()):
    gg = g_r50[g_r50["alpha_back_deg"] == ab]
    if len(gg) == 0: continue
    best = gg.loc[gg["L/W"].idxmax()]
    ax.scatter(ab, best["L/W"], s=120, c=ab_colors.get(ab, "#999"),
               edgecolors="black", linewidth=1, zorder=3)
    ax.annotate(f"θ={best['peak_theta_deg']:.0f}°\nph={best['phase_diff_deg']:.0f}\nφ={best['phi_offset_deg']:.0f}",
                (ab, best["L/W"]), textcoords="offset points",
                xytext=(10, 0), fontsize=7, alpha=0.8)
ax.set_xlabel("α_b (°)")
ax.set_ylabel("Best L/W")
ax.set_title("R=2.50, α_f=55°, kc=0.3: Best per α_b")
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)
ax.grid(alpha=0.2)

# 6d: DESIGN comparison bar chart
ax = axes[1, 0]
designs = {
    "DESIGN_v67\n(α_f=60,α_b=10,R=2.25)": (2.15, 37.6),
    "DESIGN_v68\n(α_f=55,α_b=3,R=2.50)": (2.90, 29.5),
    "v68_conservative\n(α_f=50,α_b=10,R=2.50)": (2.77, 26.1),
    "v68_R225\n(α_f=50,α_b=10,R=2.25)": (2.37, 54.0),
}
x = np.arange(len(designs))
width = 0.35
bars1 = ax.bar(x - width/2, [d[0] for d in designs.values()], width,
               color="#4CAF50", alpha=0.8, label="L/W")
ax.set_ylabel("L/W", color="#4CAF50")
ax.set_xticks(x)
ax.set_xticklabels(list(designs.keys()), fontsize=8)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, [d[1] for d in designs.values()], width,
                color="#FF9800", alpha=0.6, label="peak_θ (°)")
ax2.set_ylabel("peak_θ (°)", color="#FF9800")
ax.set_title("Design Parameter Comparison")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

# 6e: L/W histogram by R
ax = axes[1, 1]
for rval in R_LIST:
    g = stable[stable["mech_R"] == rval]
    ax.hist(g["L/W"], bins=25, alpha=0.5, color=r_color(rval),
            label=f"R={rval:.2f}", density=True)
ax.set_xlabel("L/W")
ax.set_ylabel("Density")
ax.set_title("L/W Distribution by R (Stable)")
ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5)
ax.legend(fontsize=9)

# 6f: Unstable vs Stable ratio pie
ax = axes[1, 2]
total = len(df)
stable_n = len(stable)
unstable_n = total - stable_n
ax.pie([stable_n, unstable_n], labels=["Stable", "Unstable"],
       colors=["#4CAF50", "#F44336"], autopct="%1.1f%%",
       explode=(0, 0.05), startangle=90, textprops={"fontsize": 12})
ax.set_title(f"Overall Stability ({total} combos)", fontsize=13)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save("fig6_top_combos_designs.png")
plt.close()
print("  Figure 6 done")

# ====================================================================
# Figure 7: R=2.25 专题 — α_b × φ_off × phase 三维展开
# ====================================================================
print("\nFigure 7: R=2.25 deep dive")

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle("R=2.25 Deep Dive — v6.8 Sweep",
             fontsize=16, fontweight="bold", y=0.98)

g225 = stable[(stable["mech_R"] == 2.25) & (stable["mech_a"] == 6) & (stable["f"] == 17)]

# 7a: α_b × phase L/W heatmap (kc=0.3)
ax = axes[0, 0]
g = g225[g225["k_clap"] == 0.3]
pivot = g.pivot_table(values="L/W", index="alpha_back_deg",
                       columns="phase_diff_deg", aggfunc="max")
im = ax.imshow(pivot.values, aspect="auto", origin="lower",
               cmap="RdYlGn", vmin=1.0, vmax=2.5)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"{v:.0f}" for v in pivot.index])
ax.set_xlabel("phase (°)")
ax.set_ylabel("α_b (°)")
ax.set_title("R=2.25, kc=0.3: L/W by α_b × phase")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            color = "white" if val > 1.8 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)

# 7b: α_b × φ_off L/W heatmap (kc=0.3)
ax = axes[0, 1]
pivot = g.pivot_table(values="L/W", index="alpha_back_deg",
                       columns="phi_offset_deg", aggfunc="max")
im = ax.imshow(pivot.values, aspect="auto", origin="lower",
               cmap="RdYlGn", vmin=1.0, vmax=2.5)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"{v:.0f}" for v in pivot.index])
ax.set_xlabel("φ_off (°)")
ax.set_ylabel("α_b (°)")
ax.set_title("R=2.25, kc=0.3: L/W by α_b × φ_off")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            color = "white" if val > 1.8 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)

# 7c: peak_θ by α_b × phase (kc=0.3)
ax = axes[0, 2]
pivot = g.pivot_table(values="peak_theta_deg", index="alpha_back_deg",
                       columns="phase_diff_deg", aggfunc="max")
im = ax.imshow(pivot.values, aspect="auto", origin="lower",
               cmap="RdYlGn_r", vmin=10, vmax=90)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"{v:.0f}" for v in pivot.index])
ax.set_xlabel("phase (°)")
ax.set_ylabel("α_b (°)")
ax.set_title("R=2.25, kc=0.3: peak_θ by α_b × phase")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)

# 7d: α_b=8 vs α_b=10 L/W comparison across all φ_off (kc=0.3)
ax = axes[1, 0]
for ab, marker in [(8, "o"), (10, "s")]:
    gg = g[(g["alpha_back_deg"] == ab) & (g["k_clap"] == 0.3)]
    for ph in sorted(gg["phase_diff_deg"].unique()):
        sub = gg[gg["phase_diff_deg"] == ph].sort_values("phi_offset_deg")
        ax.plot(sub["phi_offset_deg"], sub["L/W"], f"{marker}-",
                label=f"α_b={ab},ph={ph:.0f}", markersize=5, alpha=0.7)
ax.set_xlabel("φ_off (°)")
ax.set_ylabel("L/W")
ax.set_title("R=2.25, α_b=8 vs 10: L/W by φ_off and phase")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.2)
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)

# 7e: α_b=8,10,15 L/W vs peak_θ scatter (all k_clap)
ax = axes[1, 1]
for ab in [8, 10, 15]:
    gg = g225[g225["alpha_back_deg"] == ab]
    ax.scatter(gg["peak_theta_deg"], gg["L/W"], c=ab_colors.get(ab, "#999"),
               alpha=0.5, s=20, label=f"α_b={ab:.0f}", edgecolors="none")
ax.set_xlabel("peak_θ (°)")
ax.set_ylabel("L/W")
ax.set_title("R=2.25: α_b=8,10,15 in Performance Space")
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5)
ax.legend(fontsize=9, markerscale=2)
ax.grid(alpha=0.2)

# 7f: Best per k_clap at R=2.25 — L/W comparison
ax = axes[1, 2]
x = np.arange(len(KC_LIST))
width = 0.2
for ai, ab in enumerate([3, 5, 8, 10]):
    bests = []
    for kc in KC_LIST:
        gg = g225[(g225["alpha_back_deg"] == ab) & (g225["k_clap"] == kc)]
        bests.append(gg["L/W"].max() if len(gg) > 0 else 0)
    ax.bar(x + ai * width, bests, width, color=ab_colors.get(ab, "#999"),
           alpha=0.8, label=f"α_b={ab:.0f}")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f"{k:.1f}" for k in KC_LIST])
ax.set_xlabel("k_clap")
ax.set_ylabel("Max L/W")
ax.set_title("R=2.25: Max L/W by k_clap and α_b")
ax.legend(fontsize=8)
ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save("fig7_R225_deepdive.png")
plt.close()
print("  Figure 7 done")

# ============================================================
# Done
# ============================================================
print(f"\n{'='*60}")
print(f"All 7 figures saved to {FIG_DIR}")
print(f"  fig1_kclap_sensitivity.png   — k_clap 全局敏感性")
print(f"  fig2_alpha_heatmap.png       — α_f × α_b 热力图")
print(f"  fig3_mechanism_params.png    — R, a 机构参数")
print(f"  fig4_phase_phioff.png        — phase, φ_off 影响")
print(f"  fig5_performance_scatter.png — 性能空间散点")
print(f"  fig6_top_combos_designs.png  — Top参数 + 设计对比")
print(f"  fig7_R225_deepdive.png       — R=2.25 专题深化")
print(f"{'='*60}")
