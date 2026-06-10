#!/usr/bin/env python3
"""
v6.8 扫参分析 — 聚合 10,326 组数据，找最优参数
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SWEEP_DIR = Path("temp/stability/sweep_cartesian")
OUT_DIR = Path("output/figures")

# ============================================================
# Step 1: 数据聚合
# ============================================================
print("=" * 60)
print("Step 1: 数据聚合")
records = []
for d in sorted(SWEEP_DIR.iterdir()):
    if not d.is_dir():
        continue
    sm = d / "summary.json"
    if not sm.exists():
        continue
    try:
        with open(sm) as f:
            s = json.load(f)
    except Exception:
        continue
    # _combo 包含所有扫描参数
    combo = s.pop("_combo", {})
    s.update(combo)
    s["_dir"] = d.name
    records.append(s)

df = pd.DataFrame(records)
print(f"Loaded {len(df)} results")

# 关键参数列
PARAM_COLS = ["alpha_front_deg", "alpha_back_deg", "phase_diff_deg",
              "mech_a", "mech_R", "phi_offset_deg", "f", "c_damp",
              "rotation", "k_clap"]

# 报告缺失数据
for col in PARAM_COLS:
    if col not in df.columns:
        print(f"  WARNING: {col} not in data!")

# 过滤稳定样本
stable = df[df["n_exceed_90"] == 0].copy()
print(f"Stable (n_exceed_90=0): {len(stable)} / {len(df)} ({100*len(stable)/len(df):.1f}%)")
print(f"Unstable: {len(df) - len(stable)}")

# ============================================================
# Step 2: k_clap 敏感性
# ============================================================
print("\n" + "=" * 60)
print("Step 2: k_clap 敏感性分析")

print("\n--- 全部数据按 k_clap 分组 ---")
for kc in sorted(df["k_clap"].unique()):
    g = df[df["k_clap"] == kc]
    gw = g[g["n_exceed_90"] == 0]
    print(f"  k_clap={kc:.1f}: n={len(g)}, stable={len(gw)} ({100*len(gw)/len(g):.0f}%), "
          f"L/W mean={g['L/W'].mean():.3f}, L/W max={g['L/W'].max():.3f}, "
          f"peak_θ max={g['peak_theta_deg'].max():.1f}°")

print("\n--- 稳定样本 L/W > 2.0 的 k_clap 分布 ---")
high_lw = stable[stable["L/W"] >= 2.0]
for kc in sorted(df["k_clap"].unique()):
    g = high_lw[high_lw["k_clap"] == kc]
    print(f"  k_clap={kc:.1f}: {len(g)} combos, L/W max={g['L/W'].max():.3f}" if len(g) > 0 else f"  k_clap={kc:.1f}: none")

# ============================================================
# Step 3: α_f × α_b 热力图数据
# ============================================================
print("\n" + "=" * 60)
print("Step 3: α_f × α_b 热力图")

# 按 k_clap 分组，找最优安装角组合
for kc in sorted(df["k_clap"].unique()):
    g = stable[stable["k_clap"] == kc]
    if len(g) == 0:
        continue
    # α_f × α_b 的 L/W 均值
    pivot = g.pivot_table(values="L/W", index="alpha_back_deg",
                           columns="alpha_front_deg", aggfunc="mean")
    print(f"\nk_clap={kc:.1f}: L/W pivot (mean)")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

# 所有 k_clap 汇总
print("\n--- 全部稳定样本 α_f × α_b 热力图 ---")
pivot_all = stable.pivot_table(values="L/W", index="alpha_back_deg",
                                columns="alpha_front_deg", aggfunc="max")
print(pivot_all.to_string(float_format=lambda x: f"{x:.3f}"))

print("\n--- 全部稳定样本 peak_θ 热力图 (max) ---")
pivot_theta = stable.pivot_table(values="peak_theta_deg", index="alpha_back_deg",
                                  columns="alpha_front_deg", aggfunc="max")
print(pivot_theta.to_string(float_format=lambda x: f"{x:.0f}"))

# ============================================================
# Step 4: phase 影响
# ============================================================
print("\n" + "=" * 60)
print("Step 4: phase_diff 对稳定性的影响")

# 固定最优 α 范围，看 phase 的效应
for kc in sorted(df["k_clap"].unique()):
    g = stable[stable["k_clap"] == kc]
    if len(g) == 0:
        continue
    print(f"\nk_clap={kc:.1f}: L/W by phase")
    pp = g.groupby("phase_diff_deg").agg(
        L_W_mean=("L/W", "mean"),
        L_W_max=("L/W", "max"),
        peak_theta_max=("peak_theta_deg", "max"),
        n=("L/W", "count")
    )
    print(pp.to_string(float_format=lambda x: f"{x:.3f}"))

# ============================================================
# Step 5: 最优参数组合
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 最优参数组合")

# 稳定性优先：n_exceed_90=0, L/W 高
print("\n--- Top 30 L/W (stable only) ---")
top30 = stable.nlargest(30, "L/W")
for i, (_, r) in enumerate(top30.iterrows()):
    print(f"  #{i+1}: L/W={r['L/W']:.3f}, peak_θ={r['peak_theta_deg']:.1f}°, "
          f"α_f={r['alpha_front_deg']:.0f}, α_b={r['alpha_back_deg']:.0f}, "
          f"ph={r['phase_diff_deg']:.0f}, kc={r['k_clap']:.1f}, "
          f"a={r['mech_a']:.0f}, R={r['mech_R']:.2f}, φ_off={r['phi_offset_deg']:.0f}, "
          f"f={r['f']:.0f}")

print("\n--- Top 10 L/W with peak_θ < 30° ---")
super_stable = stable[stable["peak_theta_deg"] < 30]
top10_safe = super_stable.nlargest(10, "L/W")
for i, (_, r) in enumerate(top10_safe.iterrows()):
    print(f"  #{i+1}: L/W={r['L/W']:.3f}, peak_θ={r['peak_theta_deg']:.1f}°, "
          f"α_f={r['alpha_front_deg']:.0f}, α_b={r['alpha_back_deg']:.0f}, "
          f"ph={r['phase_diff_deg']:.0f}, kc={r['k_clap']:.1f}, "
          f"a={r['mech_a']:.0f}, R={r['mech_R']:.2f}, φ_off={r['phi_offset_deg']:.0f}, "
          f"f={r['f']:.0f}")

# ============================================================
# Step 6: 最优参数统计
# ============================================================
print("\n" + "=" * 60)
print("Step 6: 最优参数区域统计")

# 哪些参数组合最稳定
print("\n--- 稳定率最高 (>95%) 的参数区域 ---")
for col in ["alpha_front_deg", "alpha_back_deg", "k_clap", "phase_diff_deg", "mech_a", "mech_R", "phi_offset_deg"]:
    print(f"\n  {col}:")
    stats = df.groupby(col).agg(
        n=("n_exceed_90", "count"),
        stable_pct=("n_exceed_90", lambda x: 100 * (x == 0).sum() / len(x)),
        L_W_max=("L/W", "max"),
        L_W_mean=("L/W", "mean"),
        peak_theta_max=("peak_theta_deg", "max"),
    )
    # 只显示稳定率 > 50% 的
    stats_filtered = stats[stats["stable_pct"] > 50]
    if len(stats_filtered) > 0:
        print(stats_filtered.to_string(float_format=lambda x: f"{x:.1f}"))

# ============================================================
# Step 7: k_clap vs α_f × α_b 交互
# ============================================================
print("\n" + "=" * 60)
print("Step 7: k_clap × α_f × α_b 最优交互")

# 每个 k_clap 下的最优 (α_f, α_b) 组合
for kc in sorted(df["k_clap"].unique()):
    g = stable[stable["k_clap"] == kc]
    if len(g) == 0:
        continue
    best = g.loc[g["L/W"].idxmax()]
    print(f"  k_clap={kc:.1f}: best L/W={best['L/W']:.3f} @ α_f={best['alpha_front_deg']:.0f}, "
          f"α_b={best['alpha_back_deg']:.0f}, ph={best['phase_diff_deg']:.0f}, "
          f"a={best['mech_a']:.0f}, R={best['mech_R']:.2f}, φ_off={best['phi_offset_deg']:.0f}, "
          f"f={best['f']:.0f}, peak_θ={best['peak_theta_deg']:.1f}°")

# ============================================================
# 汇总报告
# ============================================================
print("\n" + "=" * 60)
print("汇总报告")
print("=" * 60)

# 最佳设计点
design_candidates = stable[stable["peak_theta_deg"] < 50].nlargest(5, "L/W")
print("\n推荐 DESIGN_v68 候选:")
for i, (_, r) in enumerate(design_candidates.iterrows()):
    print(f"\n  候选 #{i+1}:")
    print(f"    α_front={r['alpha_front_deg']:.0f}°, α_back={r['alpha_back_deg']:.0f}°, "
          f"phase={r['phase_diff_deg']:.0f}°")
    print(f"    a={r['mech_a']:.0f}mm, R={r['mech_R']:.2f}mm, φ_off={r['phi_offset_deg']:.0f}°, "
          f"f={r['f']:.0f}Hz")
    print(f"    k_clap={r['k_clap']:.1f}, rotation={r['rotation']}, c_damp={r['c_damp']}")
    print(f"    L/W={r['L/W']:.3f}, peak_θ={r['peak_theta_deg']:.1f}°")

# k_clap 最优值
print("\n--- k_clap 推荐 ---")
for kc in sorted(df["k_clap"].unique()):
    g = stable[stable["k_clap"] == kc]
    if len(g) == 0:
        continue
    lw_avg = g["L/W"].mean()
    lw_max = g["L/W"].max()
    theta_avg = g["peak_theta_deg"].mean()
    count_lw2 = (g["L/W"] >= 2.0).sum()
    print(f"  k_clap={kc:.1f}: avg_L/W={lw_avg:.3f}, max_L/W={lw_max:.3f}, "
          f"avg_peak_θ={theta_avg:.1f}°, L/W≥2.0 count={count_lw2}")

print("\nDone. Analysis complete.")
