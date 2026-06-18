# v6.8 扫参分析计划

## 数据概况

- **位置**: `temp/stability/sweep_cartesian/`
- **已完成**: 10,326 / 45,000 组（扫描被中断，约 23%）
- **网格**: 10 参数，k_clap=[0.3,0.5,0.8,1.0,1.5]，α_f=[50-70]，α_b=[3-15] 等
- **每组数据**: config.json + summary.json + timeseries.npz

## 分析目标

1. **k_clap 标定**: 找到使 L/W 最优且俯仰稳定的 k_clap 值
2. **α_f/α_b 最优**: 新 clap-fling 模型下最优安装角组合
3. **确认可行域**: 哪些参数组合稳定（peak θ < 90°, n90=0）

## 分析步骤

### Step 1: 数据聚合
```python
# 读取所有 summary.json，构建 DataFrame
import json, glob, pandas as pd
records = []
for d in glob.glob("temp/stability/sweep_cartesian/*/"):
    with open(d + "summary.json") as f:
        s = json.load(f)
    s["combo_id"] = d.name
    records.append(s)
df = pd.DataFrame(records)
```

### Step 2: k_clap 敏感性
- 按 k_clap 分组，计算 L/W 均值/峰值、稳定率
- 目标：L/W ≥ 2.0 且 peak_θ < 90° 的组合中，哪个 k_clap 最多？
- 对比 k_clap=0.3/0.5/0.8/1.0/1.5 的表现差异

### Step 3: α_f × α_b 热力图（固定 k_clap）
- 每个 k_clap 值下，α_f × α_b 的 L/W 和 peak_θ 热力图
- 找出 L/W 峰值区 → 这是新设计点

### Step 4: phase 影响
- α_f/α_b 固定下，phase 对稳定性的影响
- clap-fling 增强在端点附近 → phase 改变前后翅到达端点的时间差

### Step 5: 最优参数组合
- 综合考虑 L/W、稳定性、峰值扭矩
- 选出 v6.8 推荐设计参数

### Step 6（可选）: 如果 10k 数据不够覆盖最优区
- 缩小网格在峰值区加密
- 补充扫描未覆盖的组合

## 快速分析脚本

```python
# analyze_sweep_v68.py
import json, glob, numpy as np
from pathlib import Path

SWEEP_DIR = Path("temp/stability/sweep_cartesian")

# 读取所有结果
results = []
for d in sorted(SWEEP_DIR.iterdir()):
    if not d.is_dir(): continue
    sm = d / "summary.json"
    if not sm.exists(): continue
    with open(sm) as f:
        s = json.load(f)
    # 从 config 提取参数
    with open(d / "config.json") as f:
        cfg = json.load(f)
    s.update(cfg)
    results.append(s)

print(f"Loaded {len(results)} results")

# k_clap 分组统计
import pandas as pd
df = pd.DataFrame(results)
stable = df[df["n_exceed_90"] == 0]
print("\n=== k_clap 分组 ===")
for kc, g in stable.groupby("k_clap"):
    print(f"k_clap={kc:.1f}: n={len(g)}, L/W mean={g['L/W'].mean():.3f}, max={g['L/W'].max():.3f}, peak_θ max={g['peak_theta_deg'].max():.1f}°")

# 最佳组合
print("\n=== Top 10 L/W (stable) ===")
top = stable.nlargest(10, "L/W")
for _, r in top.iterrows():
    print(f"L/W={r['L/W']:.3f} peak_θ={r['peak_theta_deg']:.1f}° α_f={r['alpha_front_deg']} α_b={r['alpha_back_deg']} kc={r['k_clap']:.1f} ph={r['phase_diff_deg']} a={r['mech_a']} R={r['mech_R']}")
```

## 预期输出

- k_clap 最优值（L/W 最高 + 稳定的 k_clap 范围）
- α_f/α_b 最优组合（热力图峰值坐标）
- v6.8 推荐设计参数（写入 AGENTS.md DESIGN_v68）
