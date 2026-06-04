# 安装角 α 扫描模块（alpha_scan.py）

> 本文件说明 `src/alpha_scan.py` 的功能与使用方法。  
> 用于分析固定攻角机械蝴蝶在不同安装角 α 下的气动性能。

---

## 1. 功能概述

对翅膀安装角 α 进行参数扫描，评估：

- **净升力**（符号平均，含下拍/上拍抵消）
- **平均阻力**
- **升阻比 L/D**
- **与重量的比值**

输出 2×2 图表，包含升力、阻力、升阻比、效率评价。

---

## 2. 依赖关系

```
alpha_scan.py
    ├── dynamic_analysis.py  (load_geometry, simulate_cycle, AERO_PARAMS)
    └── matplotlib
```

输出：`output/figures/alpha_scan.png`

---

## 3. 扫描范围

```python
ALPHA_RANGE = np.arange(5, 86, 2)  # 5° ~ 85°，步长 2°
```

扫描逻辑：对每个 α，复制 `AERO_PARAMS` 并修改 `alpha_deg`，然后调用 `simulate_cycle` 计算气动力。

---

## 4. 核心函数

### `scan_alpha(geo, alphas)`

返回四组数组：
- `front_net`, `front_drag` — 前翅净升力、阻力 [mN]
- `back_net`, `back_drag` — 后翅净升力、阻力 [mN]

### `plot_alpha_scan(...)`

绘制 2×2 子图：
1. 单翅净升力 vs α
2. 单翅阻力 vs α
3. 四翅总升力 vs α（含重量参考线）
4. 升阻比 vs α（标注最优效率点）

---

## 5. 使用示例

```bash
python src/alpha_scan.py
```

输出：`output/figures/alpha_scan.png`

---

## 6. 注意事项

- 固定攻角假设：每个 α 值在整个周期内不变
- 净升力为符号平均，已包含上拍负升力的抵消效应
- 当前模型未计入旋转力（Kramer 效应）和 Clap-and-Fling

---

## 7. 文件位置

| 文件 | 路径 |
|---|---|
| 扫描脚本 | `src/alpha_scan.py` |
| 输出图表 | `output/figures/alpha_scan.png` |
