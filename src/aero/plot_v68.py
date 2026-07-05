#!/usr/bin/env python3
"""
v6.8 扫参可视化 + 设计点曲线 — 融合版 (历史数据分析, v6.9 已废弃).

从旧 sweep_cartesian (v6.8 机构参数) 输出生成图表。
⚠️ 新机构 v6.9 需重新扫参后再用此模块分析。

  fig1-fig7:  扫参主图表 (plot_sweep_v68.py 原版)
  fig8-fig10: 物理合理性补充 (plot_sweep_v68_supp.py 原版)
  fig11:      旧 DESIGN_v68 设计点曲线 (plot_design_v68.py 原版)

用法:
  python -m src.aero.plot_v68 --all
  python -m src.aero.plot_v68 --sweep --supp
  python -m src.aero.plot_v68 --design
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 路径 & 常量
# ============================================================
SWEEP_DIRS = [
    Path("F:/重大作业考试/26秋/机械原理/全链路气动仿真/temp/stability/sweep_cartesian"),
    Path("temp/stability/sweep_cartesian"),
]
FIG_DIR = Path("output/figures/aero/v68")
DESIGN_DATA_DIR = Path("temp/design_v68_detail")

COLORS = {
    0.3: "#2196F3", 0.5: "#4CAF50", 0.8: "#FF9800",
    1.0: "#F44336", 1.5: "#9C27B0",
    2.00: "#2196F3", 2.25: "#4CAF50", 2.50: "#F44336",
    6: "#2196F3", 7: "#FF9800", 8: "#F44336",
    15: "#2196F3", 17: "#F44336",
}
AB_COLORS = {3: "#E91E63", 5: "#FF9800", 8: "#4CAF50", 10: "#2196F3", 15: "#9C27B0"}
AF_COLORS = {30: "#0D47A1", 40: "#2196F3", 50: "#4CAF50", 55: "#8BC34A",
             60: "#FF9800", 70: "#F44336"}

_USE_EN = False


# ============================================================
# 字体 & 初始化
# ============================================================
def configure_matplotlib():
    """设置中文字体，找不到则 fallback 英文."""
    global _USE_EN
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "WenQuanYi Micro Hei", "DejaVu Sans"]
    found = None
    for f in candidates:
        for fm in font_manager.fontManager.ttflist:
            if f.lower() in fm.name.lower():
                found = fm.name
                break
        if found:
            break
    if found:
        plt.rcParams["font.family"] = found
    else:
        plt.rcParams["font.family"] = "sans-serif"
        _USE_EN = True
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.unicode_minus"] = False


def label(text: str) -> str:
    """中文标签 fallback."""
    if not _USE_EN:
        return text
    _EN_MAP = {
        "k_clap": "k_clap", "稳定率 (%)": "Stability (%)",
        "L/W": "L/W", "L/W 均值": "L/W mean", "L/W max": "L/W max",
        "L/W ≥ 2.0 组数": "L/W ≥ 2.0 count", "L/W 分布": "L/W distribution",
        "概率密度": "Density", "稳定": "Stable", "不稳定": "Unstable",
        "前翅安装角 α_f (°)": "α_f (°)", "后翅安装角 α_b (°)": "α_b (°)",
        "峰俯仰角 peak_θ (°)": "peak_θ (°)", "相位差 phase (°)": "phase (°)",
        "摇杆半径 R (mm)": "R (mm)", "曲柄半径 a (mm)": "a (mm)",
        "安装偏角 φ_off (°)": "φ_off (°)", "频率 f (Hz)": "f (Hz)",
        "分组": "Group", "数量": "Count", "最大俯仰峰": "Max peak θ",
        "平均俯仰峰": "Mean peak θ",
    }
    return _EN_MAP.get(text, text)


def save(name: str):
    path = FIG_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")


# ============================================================
# 数据加载
# ============================================================
def load_sweep_data():
    """加载所有扫参 summary.json，返回 DataFrame."""
    records = []
    seen = set()
    for sweep_dir in SWEEP_DIRS:
        if not sweep_dir.exists():
            continue
        for d in sorted(sweep_dir.iterdir()):
            if not d.is_dir():
                continue
            if d.name in seen:
                continue
            seen.add(d.name)
            sm = d / "summary.json"
            if not sm.exists():
                continue
            try:
                with open(sm, encoding="utf-8") as f:
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


def load_design_data():
    """加载 DESIGN_v68 设计点时序数据."""
    data = np.load(DESIGN_DATA_DIR / "timeseries_2cycles.npz")
    with open(DESIGN_DATA_DIR / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    return data, summary


# ============================================================
# Figure 1-7: 扫参主图表
# ============================================================
def plot_sweep_main(df: pd.DataFrame):
    stable = df[df.stable]
    kc_list = sorted(df["k_clap"].unique())
    r_list = sorted(df["mech_R"].unique())
    a_list = sorted(df["mech_a"].unique())

    def kc_color(kc):
        return COLORS.get(kc, "#999999")

    def r_color(r):
        return COLORS.get(r, "#999999")

    # ---- Figure 1 ----
    print("\nFigure 1: k_clap sensitivity")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("k_clap Sensitivity Analysis — v6.8 Sweep",
                 fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    data_by_kc = [stable[stable["k_clap"] == kc]["L/W"].values for kc in kc_list]
    bp = ax.boxplot(data_by_kc, labels=[f"{kc:.1f}" for kc in kc_list],
                    patch_artist=True, widths=0.6)
    for kc, patch in zip(kc_list, bp["boxes"]):
        patch.set_facecolor(kc_color(kc))
        patch.set_alpha(0.6)
    ax.set_xlabel("k_clap")
    ax.set_ylabel("L/W")
    ax.set_title("L/W Distribution by k_clap (Stable Only)")
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5)

    ax = axes[0, 1]
    rates = [100 * (df[df["k_clap"] == kc]["n_exceed_90"] == 0).sum() / max(len(df[df["k_clap"] == kc]), 1)
             for kc in kc_list]
    bars = ax.bar(range(len(kc_list)), rates, color=[kc_color(k) for k in kc_list],
                  edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(len(kc_list)))
    ax.set_xticklabels([f"{k:.1f}" for k in kc_list])
    ax.set_xlabel("k_clap")
    ax.set_ylabel("Stability Rate (%)")
    ax.set_title("Stability Rate by k_clap")
    ax.set_ylim(0, 100)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{r:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax = axes[0, 2]
    lw2_counts = [(stable[stable["k_clap"] == kc]["L/W"] >= 2.0).sum() for kc in kc_list]
    lw2_pcts = [100 * c / max(len(stable[stable["k_clap"] == kc]), 1) for c, kc in zip(lw2_counts, kc_list)]
    bars = ax.bar(range(len(kc_list)), lw2_counts, color=[kc_color(k) for k in kc_list],
                  edgecolor="white", linewidth=1.2)
    ax.set_xticks(range(len(kc_list)))
    ax.set_xticklabels([f"{k:.1f}" for k in kc_list])
    ax.set_xlabel("k_clap")
    ax.set_ylabel("Count")
    ax.set_title("Combos with L/W ≥ 2.0 by k_clap")
    for bar, c, p in zip(bars, lw2_counts, lw2_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{c}\n({p:.1f}%)", ha="center", fontsize=9)

    ax = axes[1, 0]
    for kc in kc_list:
        g = stable[stable["k_clap"] == kc]
        ax.hist(g["L/W"], bins=30, alpha=0.5, color=kc_color(kc),
                label=f"kc={kc:.1f}", density=True)
    ax.set_xlabel("L/W")
    ax.set_ylabel("Density")
    ax.set_title("L/W Distribution (Stable, by k_clap)")
    ax.legend(fontsize=8, ncol=2)
    ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5)

    ax = axes[1, 1]
    means, stds = [], []
    for kc in kc_list:
        g = stable[stable["k_clap"] == kc]
        means.append(g["L/W"].mean())
        stds.append(g["L/W"].std())
    ax.errorbar(range(len(kc_list)), means, yerr=stds, fmt="o-", capsize=8,
                markersize=10, linewidth=2, color="#333333")
    for i, (kc, m) in enumerate(zip(kc_list, means)):
        ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=9, fontweight="bold",
                    color=kc_color(kc))
    ax.set_xticks(range(len(kc_list)))
    ax.set_xticklabels([f"{k:.1f}" for k in kc_list])
    ax.set_xlabel("k_clap")
    ax.set_ylabel("L/W")
    ax.set_title("Mean L/W ± Std by k_clap (Stable)")
    ax.set_ylim(bottom=0.8)

    ax = axes[1, 2]
    lw_maxs = [stable[stable["k_clap"] == kc]["L/W"].max() for kc in kc_list]
    theta_maxs = [stable[stable["k_clap"] == kc]["peak_theta_deg"].max() for kc in kc_list]
    x = np.arange(len(kc_list))
    width = 0.35
    ax.bar(x - width/2, lw_maxs, width, color=[kc_color(k) for k in kc_list],
           alpha=0.7, label="L/W max")
    ax.set_ylabel("L/W max", color="#333")
    ax.set_xlabel("k_clap")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k:.1f}" for k in kc_list])
    ax2 = ax.twinx()
    ax2.bar(x + width/2, theta_maxs, width, color="gray", alpha=0.3,
            label="peak_θ max (°)")
    ax2.set_ylabel("peak_θ max (°)", color="gray")
    ax.set_title("Max L/W & peak_θ by k_clap")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save("fig1_kclap_sensitivity.png")
    plt.close()
    print("  Figure 1 done")

    # ---- Figure 2 ----
    print("\nFigure 2: α_f × α_b heatmaps")
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle("α_f × α_b Heatmaps — L/W (Stable, by k_clap)",
                 fontsize=16, fontweight="bold", y=0.98)

    for idx, kc in enumerate(kc_list):
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

    # ---- Figure 3 ----
    print("\nFigure 3: Mechanism parameters")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Mechanism Parameter Analysis (R, a) — v6.8 Sweep",
                 fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    positions, data_r, colors_r, labels_r = [], [], [], []
    for ri, rval in enumerate(r_list):
        for ki, kc in enumerate(kc_list):
            g = stable[(stable["mech_R"] == rval) & (stable["k_clap"] == kc)]
            if len(g) > 0:
                pos = ri * (len(kc_list) + 1) + ki
                positions.append(pos)
                data_r.append(g["L/W"].values)
                colors_r.append(kc_color(kc))
                labels_r.append(f"R={rval:.2f}\nkc={kc:.1f}")
    bp = ax.boxplot(data_r, positions=positions, patch_artist=True,
                    widths=0.5, manage_ticks=False)
    for patch, color in zip(bp["boxes"], colors_r):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for ri in range(1, len(r_list)):
        ax.axvline(x=ri * (len(kc_list) + 1) - 0.5, color="gray",
                   linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xticks(positions[::len(kc_list)])
    ax.set_xticklabels([f"R={r:.2f}" for r in r_list], fontsize=11, fontweight="bold")
    ax.set_ylabel("L/W (Stable)")
    ax.set_title("L/W by R and k_clap")
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)

    ax = axes[0, 1]
    x = np.arange(len(r_list))
    width = 0.15
    for ki, kc in enumerate(kc_list):
        rates_r = []
        for rval in r_list:
            g = df[(df["mech_R"] == rval) & (df["k_clap"] == kc)]
            rates_r.append(100 * (g["n_exceed_90"] == 0).sum() / len(g) if len(g) > 0 else 0)
        ax.bar(x + ki * width, rates_r, width, color=kc_color(kc), alpha=0.8,
               label=f"kc={kc:.1f}")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"{r:.2f}" for r in r_list])
    ax.set_xlabel("R (mm)")
    ax.set_ylabel("Stability Rate (%)")
    ax.set_title("Stability Rate by R and k_clap")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, ncol=3)

    ax = axes[1, 0]
    data_a = [stable[stable["mech_a"] == a]["L/W"].values for a in a_list]
    bp = ax.boxplot(data_a, labels=[f"a={a:.0f}" for a in a_list],
                    patch_artist=True, widths=0.5)
    for patch, a_val in zip(bp["boxes"], a_list):
        patch.set_facecolor(COLORS.get(a_val, "#999"))
        patch.set_alpha(0.6)
    ax.set_xlabel("a (mm)")
    ax.set_ylabel("L/W (Stable)")
    ax.set_title("L/W by mech_a")
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)

    ax = axes[1, 1]
    x = np.arange(len(r_list))
    width = 0.25
    for ai, a_val in enumerate(a_list):
        maxes = []
        for rval in r_list:
            g = stable[(stable["mech_R"] == rval) & (stable["mech_a"] == a_val)]
            maxes.append(g["L/W"].max() if len(g) > 0 else 0)
        ax.bar(x + ai * width, maxes, width, color=COLORS.get(a_val, "#999"),
               alpha=0.8, label=f"a={a_val:.0f}")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{r:.2f}" for r in r_list])
    ax.set_xlabel("R (mm)")
    ax.set_ylabel("Max L/W (Stable)")
    ax.set_title("Max L/W by R and a")
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save("fig3_mechanism_params.png")
    plt.close()
    print("  Figure 3 done")

    # ---- Figure 4 ----
    print("\nFigure 4: Phase and phi_offset effects")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Phase & φ_offset Sensitivity — v6.8 Sweep",
                 fontsize=16, fontweight="bold", y=0.98)

    phase_list = sorted(stable["phase_diff_deg"].unique())
    po_list = sorted(stable["phi_offset_deg"].unique())

    ax = axes[0, 0]
    for kc in kc_list:
        means = stable[stable["k_clap"] == kc].groupby("phase_diff_deg")["L/W"].mean()
        ax.plot(means.index, means.values, "o-", color=kc_color(kc),
                label=f"kc={kc:.1f}", markersize=7, linewidth=1.5)
    ax.set_xlabel("phase (°)")
    ax.set_ylabel("Mean L/W")
    ax.set_title("L/W vs phase by k_clap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for kc in kc_list:
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

    ax = axes[0, 2]
    for kc in kc_list:
        means = stable[stable["k_clap"] == kc].groupby("phase_diff_deg")["peak_theta_deg"].mean()
        ax.plot(means.index, means.values, "o-", color=kc_color(kc),
                label=f"kc={kc:.1f}", markersize=7, linewidth=1.5)
    ax.set_xlabel("phase (°)")
    ax.set_ylabel("Mean peak θ (°)")
    ax.set_title("Mean peak_θ vs phase by k_clap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for kc in kc_list:
        means = stable[stable["k_clap"] == kc].groupby("phi_offset_deg")["L/W"].mean()
        ax.plot(means.index, means.values, "o-", color=kc_color(kc),
                label=f"kc={kc:.1f}", markersize=7, linewidth=1.5)
    ax.set_xlabel("φ_off (°)")
    ax.set_ylabel("Mean L/W")
    ax.set_title("L/W vs φ_off by k_clap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for kc in kc_list:
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

    # ---- Figure 5 ----
    print("\nFigure 5: L/W vs peak_θ scatter")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Performance Space: L/W vs peak_θ — v6.8 Sweep",
                 fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    for kc in kc_list:
        g = stable[stable["k_clap"] == kc]
        ax.scatter(g["peak_theta_deg"], g["L/W"], c=kc_color(kc),
                   alpha=0.4, s=12, label=f"kc={kc:.1f}", edgecolors="none")
    ax.set_xlabel("peak_θ (°)")
    ax.set_ylabel("L/W")
    ax.set_title("Stable Combos — Colored by k_clap")
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=30, color="green", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    for rval in r_list:
        g = stable[stable["mech_R"] == rval]
        ax.scatter(g["peak_theta_deg"], g["L/W"], c=r_color(rval),
                   alpha=0.4, s=12, label=f"R={rval:.2f}", edgecolors="none")
    ax.set_xlabel("peak_θ (°)")
    ax.set_ylabel("L/W")
    ax.set_title("Stable Combos — Colored by R")
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=30, color="green", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.2)

    ab_list_plot = sorted(stable["alpha_back_deg"].unique())
    for ax_idx, rval, title, star_y, star_lw, label in [
        (axes[1, 0], 2.25, "R=2.25 Stable Combos — Colored by α_b", 54.0, 2.370, "DESIGN_v68_R225"),
        (axes[1, 1], 2.50, "R=2.50 Stable Combos — Colored by α_b", 29.5, 2.904, "DESIGN_v68"),
    ]:
        ax = ax_idx
        g = stable[stable["mech_R"] == rval]
        for ab in ab_list_plot:
            gg = g[g["alpha_back_deg"] == ab]
            ax.scatter(gg["peak_theta_deg"], gg["L/W"], c=AB_COLORS.get(ab, "#999"),
                       alpha=0.5, s=15, label=f"α_b={ab:.0f}", edgecolors="none")
        ax.set_xlabel("peak_θ (°)")
        ax.set_ylabel("L/W")
        ax.set_title(title)
        ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5)
        ax.scatter([star_y], [star_lw], c="red", s=200, marker="*", edgecolors="black",
                   linewidth=1.5, zorder=5, label=label)
        ax.legend(fontsize=8, markerscale=1)
        ax.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save("fig5_performance_scatter.png")
    plt.close()
    print("  Figure 5 done")

    # ---- Figure 6 ----
    print("\nFigure 6: Top combo analysis")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Top Combos & DESIGN_v68 Detail — v6.8 Sweep",
                 fontsize=16, fontweight="bold", y=0.98)

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
    top_params = sorted(param_counts.items(), key=lambda x: -x[1])[:15]
    ax.barh(range(len(top_params)), [x[1] for x in top_params], color="#2196F3")
    ax.set_yticks(range(len(top_params)))
    ax.set_yticklabels([x[0] for x in top_params])
    ax.set_xlabel("Count in Top 100")
    ax.set_title("Parameter Frequency in Top 100 L/W")
    ax.invert_yaxis()

    for ax_idx, rval, af, title in [
        (axes[0, 1], 2.25, 50, "R=2.25, α_f=50°, kc=0.3: Best per α_b"),
        (axes[0, 2], 2.50, 55, "R=2.50, α_f=55°, kc=0.3: Best per α_b"),
    ]:
        ax = ax_idx
        g_r = stable[(stable["mech_R"] == rval) & (stable["alpha_front_deg"] == af) &
                     (stable["k_clap"] == 0.3) & (stable["f"] == 17) & (stable["mech_a"] == 6)]
        for ab in sorted(g_r["alpha_back_deg"].unique()):
            gg = g_r[g_r["alpha_back_deg"] == ab]
            if len(gg) == 0:
                continue
            best = gg.loc[gg["L/W"].idxmax()]
            ax.scatter(ab, best["L/W"], s=120, c=AB_COLORS.get(ab, "#999"),
                       edgecolors="black", linewidth=1, zorder=3)
            ax.annotate(f"θ={best['peak_theta_deg']:.0f}°\nph={best['phase_diff_deg']:.0f}\nφ={best['phi_offset_deg']:.0f}",
                        (ab, best["L/W"]), textcoords="offset points",
                        xytext=(10, 0), fontsize=7, alpha=0.8)
        ax.set_xlabel("α_b (°)")
        ax.set_ylabel("Best L/W")
        ax.set_title(title)
        ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)
        ax.grid(alpha=0.2)

    ax = axes[1, 0]
    designs = {
        "DESIGN_v67\n(α_f=60,α_b=10,R=2.25)": (2.15, 37.6),
        "DESIGN_v68\n(α_f=55,α_b=3,R=2.50)": (2.90, 29.5),
        "v68_conservative\n(α_f=50,α_b=10,R=2.50)": (2.77, 26.1),
        "v68_R225\n(α_f=50,α_b=10,R=2.25)": (2.37, 54.0),
    }
    x = np.arange(len(designs))
    width = 0.35
    ax.bar(x - width/2, [d[0] for d in designs.values()], width,
           color="#4CAF50", alpha=0.8, label="L/W")
    ax.set_ylabel("L/W", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(list(designs.keys()), fontsize=8)
    ax2 = ax.twinx()
    ax2.bar(x + width/2, [d[1] for d in designs.values()], width,
            color="#FF9800", alpha=0.6, label="peak_θ (°)")
    ax2.set_ylabel("peak_θ (°)", color="#FF9800")
    ax.set_title("Design Parameter Comparison")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[1, 1]
    for rval in r_list:
        g = stable[stable["mech_R"] == rval]
        ax.hist(g["L/W"], bins=25, alpha=0.5, color=r_color(rval),
                label=f"R={rval:.2f}", density=True)
    ax.set_xlabel("L/W")
    ax.set_ylabel("Density")
    ax.set_title("L/W Distribution by R (Stable)")
    ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    ax = axes[1, 2]
    total = len(df)
    stable_n = len(stable)
    ax.pie([stable_n, total - stable_n], labels=["Stable", "Unstable"],
           colors=["#4CAF50", "#F44336"], autopct="%1.1f%%",
           explode=(0, 0.05), startangle=90, textprops={"fontsize": 12})
    ax.set_title(f"Overall Stability ({total} combos)", fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save("fig6_top_combos_designs.png")
    plt.close()
    print("  Figure 6 done")

    # ---- Figure 7 ----
    print("\nFigure 7: R=2.25 deep dive")
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle("R=2.25 Deep Dive — v6.8 Sweep",
                 fontsize=16, fontweight="bold", y=0.98)

    g225 = stable[(stable["mech_R"] == 2.25) & (stable["mech_a"] == 6) & (stable["f"] == 17)]
    g = g225[g225["k_clap"] == 0.3]

    for ax_idx, value, index, columns, cmap, vmin, vmax, title, fmt, color_thresh in [
        (axes[0, 0], "L/W", "alpha_back_deg", "phase_diff_deg", "RdYlGn", 1.0, 2.5,
         "R=2.25, kc=0.3: L/W by α_b × phase", "{:.3f}", 1.8),
        (axes[0, 1], "L/W", "alpha_back_deg", "phi_offset_deg", "RdYlGn", 1.0, 2.5,
         "R=2.25, kc=0.3: L/W by α_b × φ_off", "{:.3f}", 1.8),
        (axes[0, 2], "peak_theta_deg", "alpha_back_deg", "phase_diff_deg", "RdYlGn_r", 10, 90,
         "R=2.25, kc=0.3: peak_θ by α_b × phase", "{:.0f}", 60),
    ]:
        ax = ax_idx
        pivot = g.pivot_table(values=value, index=index, columns=columns, aggfunc="max")
        im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.0f}" for v in pivot.index])
        ax.set_xlabel(columns.replace("phi_offset_deg", "φ_off (°)").replace("phase_diff_deg", "phase (°)"))
        ax.set_ylabel("α_b (°)")
        ax.set_title(title)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = "white" if val > color_thresh else "black"
                    ax.text(j, i, fmt.format(val), ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

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

    ax = axes[1, 1]
    for ab in [8, 10, 15]:
        gg = g225[g225["alpha_back_deg"] == ab]
        ax.scatter(gg["peak_theta_deg"], gg["L/W"], c=AB_COLORS.get(ab, "#999"),
                   alpha=0.5, s=20, label=f"α_b={ab:.0f}", edgecolors="none")
    ax.set_xlabel("peak_θ (°)")
    ax.set_ylabel("L/W")
    ax.set_title("R=2.25: α_b=8,10,15 in Performance Space")
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(alpha=0.2)

    ax = axes[1, 2]
    x = np.arange(len(kc_list))
    width = 0.2
    for ai, ab in enumerate([3, 5, 8, 10]):
        bests = []
        for kc in kc_list:
            gg = g225[(g225["alpha_back_deg"] == ab) & (g225["k_clap"] == kc)]
            bests.append(gg["L/W"].max() if len(gg) > 0 else 0)
        ax.bar(x + ai * width, bests, width, color=AB_COLORS.get(ab, "#999"),
               alpha=0.8, label=f"α_b={ab:.0f}")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{k:.1f}" for k in kc_list])
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
# Figure 8-10: 物理合理性补充
# ============================================================
def plot_sweep_supp(df: pd.DataFrame):
    stable = df[df.stable]
    kc_list = sorted(df["k_clap"].unique())
    af_list = sorted(stable["alpha_front_deg"].unique())
    r_list = sorted(df["mech_R"].unique())

    def kc_color(kc):
        return COLORS.get(kc, "#999999")

    # ---- Figure 8 ----
    print("\nFigure 8: Physical reasonableness")
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle("Physical Reasonableness Analysis — v6.8 Full Dataset",
                 fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    for kc in kc_list:
        g = stable[stable["k_clap"] == kc]
        lw_max = g.groupby("alpha_front_deg")["L/W"].max()
        ax.plot(lw_max.index, lw_max.values, "o-", color=kc_color(kc),
                label=f"kc={kc:.1f}", markersize=8, linewidth=1.5)
    ax.set_xlabel("α_f (°)")
    ax.set_ylabel("Max L/W")
    ax.set_title("Max L/W vs α_f by k_clap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axvspan(30, 50, alpha=0.08, color="green")
    ax.axvspan(50, 71, alpha=0.08, color="red")

    ax = axes[0, 1]
    for kc in kc_list:
        rates = []
        for af in af_list:
            g = df[(df["alpha_front_deg"] == af) & (df["k_clap"] == kc)]
            rates.append(100 * (g["n_exceed_90"] == 0).sum() / len(g) if len(g) > 0 else 0)
        ax.plot(af_list, rates, "o-", color=kc_color(kc),
                label=f"kc={kc:.1f}", markersize=7, linewidth=1.5)
    ax.set_xlabel("α_f (°)")
    ax.set_ylabel("Stability Rate (%)")
    ax.set_title("Stability Rate vs α_f")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axvspan(30, 50, alpha=0.08, color="green")
    ax.axvspan(50, 71, alpha=0.08, color="red")

    ax = axes[0, 2]
    for kc in kc_list:
        g = stable[stable["k_clap"] == kc]
        theta_mean = g.groupby("alpha_front_deg")["peak_theta_deg"].mean()
        ax.plot(theta_mean.index, theta_mean.values, "o-", color=kc_color(kc),
                label=f"kc={kc:.1f}", markersize=7, linewidth=1.5)
    ax.set_xlabel("α_f (°)")
    ax.set_ylabel("Mean peak_θ (°)")
    ax.set_title("Mean peak_θ vs α_f")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(y=30, color="green", linestyle="--", alpha=0.5)
    ax.axvspan(30, 50, alpha=0.08, color="green")
    ax.axvspan(50, 71, alpha=0.08, color="red")

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
    ax.set_title("Max L/W vs α_f by α_b")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.axvspan(30, 50, alpha=0.08, color="green")
    ax.axvspan(50, 71, alpha=0.08, color="red")

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

    ax = axes[1, 2]
    best_per_af = []
    for af in af_list:
        g = stable[stable["alpha_front_deg"] == af]
        if len(g) == 0:
            continue
        best = g.loc[g["L/W"].idxmax()]
        best_per_af.append({
            "α_f": af, "L/W": best["L/W"], "α_b": best["alpha_back_deg"],
            "θ": best["peak_theta_deg"], "ph": best["phase_diff_deg"],
            "kc": best["k_clap"],
        })
    best_df = pd.DataFrame(best_per_af)
    colors_bar = ["green" if af <= 50 else "red" for af in best_df["α_f"]]
    bars = ax.bar(range(len(best_df)), best_df["L/W"], color=colors_bar, alpha=0.7)
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
    save("fig8_physical_reasonableness.png")
    plt.close()
    print("  Figure 8 done")

    # ---- Figure 9 ----
    print("\nFigure 9: Reasonable range zoom")
    reasonable = stable[stable["alpha_front_deg"].isin([40, 50])]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle("Physically Reasonable Range (α_f=40-50°) — High-Resolution View",
                 fontsize=16, fontweight="bold", y=0.98)

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
    ax.legend([Patch(facecolor="#4CAF50", alpha=0.5), Patch(facecolor="#2196F3", alpha=0.5)],
              ["α_f=40°", "α_f=50°"], fontsize=9)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(ab_vals))])
    ax.set_xticklabels([f"α_b={ab:.0f}" for ab in ab_vals])
    ax.set_ylabel("L/W")
    ax.set_title("α_f=40° vs 50°: L/W by α_b")
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.2)

    for ax_idx, af, title in [
        (axes[0, 1], 40, "α_f=40°: Best per (α_b, ph, kc)"),
        (axes[0, 2], 50, "α_f=50°: Best per (α_b, ph, kc)"),
    ]:
        ax = ax_idx
        af_df = reasonable[reasonable["alpha_front_deg"] == af]
        for ab in sorted(af_df["alpha_back_deg"].unique()):
            g = af_df[af_df["alpha_back_deg"] == ab]
            for kc in [0.3, 0.5]:
                gg = g[g["k_clap"] == kc]
                if len(gg) == 0:
                    continue
                best = gg.loc[gg["L/W"].idxmax()]
                marker = "o" if kc == 0.3 else "s"
                ax.scatter(best["phase_diff_deg"], best["L/W"], marker=marker,
                           s=100, alpha=0.7, label=f"α_b={ab:.0f},kc={kc:.1f}")
        ax.set_xlabel("phase (°)")
        ax.set_ylabel("Best L/W")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.2)

    ax = axes[1, 0]
    for af in [40, 50]:
        for ab in [3, 5, 8, 10, 15]:
            g = reasonable[(reasonable["alpha_front_deg"] == af) & (reasonable["alpha_back_deg"] == ab)]
            if len(g) == 0:
                continue
            c = AB_COLORS[ab]
            ax.scatter(g["peak_theta_deg"], g["L/W"], c=c, alpha=0.3, s=8,
                       edgecolors="none")
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

    ax = axes[1, 1]
    for af in [40, 50]:
        g = reasonable[reasonable["alpha_front_deg"] == af]
        po_means = g.groupby("phi_offset_deg")["L/W"].agg(["mean", "max"])
        ax.plot(po_means.index, po_means["mean"], "o-",
                color="#4CAF50" if af == 40 else "#2196F3",
                label=f"α_f={af}° mean", linewidth=1.5)
        stds = g.groupby("phi_offset_deg")["L/W"].std()
        ax.fill_between(po_means.index, po_means["mean"] - stds,
                        po_means["mean"] + stds,
                        alpha=0.15, color="#4CAF50" if af == 40 else "#2196F3")
    ax.set_xlabel("φ_off (°)")
    ax.set_ylabel("Mean L/W")
    ax.set_title("φ_off Effect on L/W (α_f=40° vs 50°)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    ax = axes[1, 2]
    x = np.arange(2)
    width = 0.25
    for ri, rval in enumerate(r_list):
        bests = []
        for af in [40, 50]:
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
    save("fig9_reasonable_range_zoom.png")
    plt.close()
    print("  Figure 9 done")

    # ---- Figure 10 ----
    print("\nFigure 10: Full range performance scatter")
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.suptitle("Full Performance Space — All Stable Combos",
                 fontsize=16, fontweight="bold")

    for af in sorted(stable["alpha_front_deg"].unique()):
        g = stable[stable["alpha_front_deg"] == af]
        ax.scatter(g["peak_theta_deg"], g["L/W"], c=AF_COLORS.get(af, "#999"),
                   alpha=0.4, s=8, label=f"α_f={af:.0f}° (n={len(g)})",
                   edgecolors="none")

    ax.axvline(x=30, color="green", linestyle="--", alpha=0.4, linewidth=1.5)
    ax.annotate("θ=30°", xy=(30, 0.5), fontsize=9, color="green")

    designs = [
        (29.5, 2.904, "DESIGN_v68 (α_f=55) [OLD]", "blue", "s"),
        (32.9, 2.447, "DESIGN_v68 (α_f=45) [NEW]", "green", "D"),
        (24.2, 2.381, "v68_safe (α_f=40)", "cyan", "o"),
    ]
    for x, y, lab, c, m in designs:
        ax.scatter([x], [y], c=c, s=300, marker=m, edgecolors="black",
                   linewidth=2, zorder=10, label=lab)

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
    save("fig10_full_performance_landscape.png")
    plt.close()
    print("  Figure 10 done")


# ============================================================
# Figure 11: DESIGN_v68 设计点曲线
# ============================================================
def plot_design_point():
    print("\nFigure 11: DESIGN_v68 design-point curves")
    data, summary = load_design_data()
    t = data["t"]

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle(f"DESIGN_v68 Performance Curves — α_f=45°, α_b=8°, "
                 f"L/W={summary['L/W']:.3f}, peak_θ={summary['peak_theta_deg']:.1f}°",
                 fontsize=15, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    ax.plot(t*1000, np.rad2deg(data["theta_p"]), "#2196F3", linewidth=1.2)
    ax.set_ylabel("θ_p (°)")
    ax.set_xlabel("t (ms)")
    ax.set_title(f"Pitch Angle (mean={summary['steady_theta_mean_deg']:.1f}°, "
                 f"amp=±{summary['steady_theta_amplitude_deg']:.1f}°)")
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t*1000, data["theta_dot"], "#FF9800", linewidth=1.2)
    ax.set_ylabel("θ̇_p (rad/s)")
    ax.set_xlabel("t (ms)")
    ax.set_title("Pitch Rate")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t*1000, data["Fz_world_total"]*1000, "#4CAF50", linewidth=1.2, label="Fz_world")
    ax.axhline(y=196.2, color="gray", linestyle="--", alpha=0.5, label="Weight (196 mN)")
    ax.set_ylabel("Force (mN)")
    ax.set_xlabel("t (ms)")
    ax.set_title(f"Total Lift (World) — mean={summary['mean_Fz_world_mN']:.0f} mN")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t*1000, data["Fz_body_FL"]*1000, "#E91E63", linewidth=0.8, label="FL")
    ax.plot(t*1000, data["Fz_body_BL"]*1000, "#2196F3", linewidth=0.8, label="BL")
    ax.set_ylabel("Fz_body (mN)")
    ax.set_xlabel("t (ms)")
    ax.set_title("Body-frame Vertical Force per Wing")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t*1000, data["Fx_body_FL"]*1000, "#E91E63", linewidth=0.8, label="FL")
    ax.plot(t*1000, data["Fx_body_BL"]*1000, "#2196F3", linewidth=0.8, label="BL")
    ax.set_ylabel("Fx_body (mN)")
    ax.set_xlabel("t (ms)")
    ax.set_title("Body-frame Horizontal Force per Wing")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(t*1000, data["alpha_eff_FL"], "#E91E63", linewidth=0.8,
            label=f"FL (mean|α|={summary['mean_alpha_eff_FL_deg']:.1f}°)")
    ax.plot(t*1000, data["alpha_eff_BL"], "#2196F3", linewidth=0.8, label="BL")
    ax.axhline(y=70, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axhline(y=-70, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axhline(y=40, color="orange", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axhline(y=-40, color="orange", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_ylabel("α_eff (°)")
    ax.set_xlabel("t (ms)")
    ax.set_title(f"Effective AoA — |α|>70°: {summary['pct_alpha_above_70']:.1f}% time")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[2, 0]
    ax.plot(t*1000, data["CL_FL"], "#4CAF50", linewidth=0.8,
            label=f"C_L (mean={summary['mean_CL_FL']:.3f})")
    ax.plot(t*1000, data["CD_FL"], "#F44336", linewidth=0.8,
            label=f"C_D (mean={summary['mean_CD_FL']:.3f})")
    ax.set_ylabel("Coefficient")
    ax.set_xlabel("t (ms)")
    ax.set_title("Front Wing C_L / C_D")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t*1000, np.rad2deg(data["phi_FL"]), "#2196F3", linewidth=1.2, label="φ (°)")
    ax2 = ax.twinx()
    ax2.plot(t*1000, data["phi_dot_FL"], "#FF9800", linewidth=0.8, alpha=0.7, label="φ̇ (rad/s)")
    ax.set_ylabel("φ (°)")
    ax2.set_ylabel("φ̇ (rad/s)", color="#FF9800")
    ax.set_xlabel("t (ms)")
    ax.set_title("Front Wing Kinematics (2 cycles)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2, 2]
    ax.plot(t*1000, data["rocker_pv_x_FL"], "#E91E63", linewidth=0.8, label="PV_x")
    ax.plot(t*1000, data["rocker_pv_z_FL"], "#2196F3", linewidth=0.8, label="PV_z")
    ax2 = ax.twinx()
    ax2.plot(t*1000, data["rocker_pm_y_FL"]*1e3, "#4CAF50", linewidth=0.8, alpha=0.6,
            label="PM_y (mN·m)")
    ax.set_ylabel("Principal Vector (N)")
    ax2.set_ylabel("Principal Moment (mN·m)", color="#4CAF50")
    ax.set_xlabel("t (ms)")
    ax.set_title("FL Rocker Principal Vector & Moment")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save("fig11_design_v68_curves.png")
    plt.close()
    print("  Figure 11 done")


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="v6.8 sweep & design-point plotting")
    parser.add_argument("--all", action="store_true", help="Generate all 11 figures")
    parser.add_argument("--sweep", action="store_true", help="Generate fig1-fig7 (sweep main)")
    parser.add_argument("--supp", action="store_true", help="Generate fig8-fig10 (physical reasonableness)")
    parser.add_argument("--design", action="store_true", help="Generate fig11 (design point)")
    args = parser.parse_args()

    if not any([args.all, args.sweep, args.supp, args.design]):
        args.all = True

    configure_matplotlib()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.sweep or args.supp:
        df = load_sweep_data()
        print(f"Loaded {len(df)} results, {(df.stable).sum()} stable")

    if args.all or args.sweep:
        plot_sweep_main(df)

    if args.all or args.supp:
        plot_sweep_supp(df)

    if args.all or args.design:
        plot_design_point()

    print(f"\n{'='*60}")
    print(f"Figures saved to {FIG_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
