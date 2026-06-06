# Butterfly MAV 方案二：9参数全笛卡尔积扫描 — 实验报告

**项目**: 仿生蝴蝶扑翼微型飞行器俯仰动力学仿真
**报告日期**: 2026-06-06
**作者**: lscatfish
**版本**: v6.6 Cart-1

---

## 摘要

在方案一（6参数单变量偏离扫描）的基础上，方案二将扫描扩展到 **9参数粗网格全笛卡尔积**，共 3,456 组组合，使用 **numba JIT 加速**（4.8× 单仿真加速）和 **joblib 多进程并行**（16 核），总耗时 **56.7 分钟**。扫描发现了一个卓越的参数区域，世界系升重比 L/W 从基线 2.505 提升至 **12.625**（5.0× 提升），峰值俯仰角仅 35.1°。核心结论：最优性能需要在 **曲柄半径 a=6mm、摇杆半径 R=3.0mm、拍动频率 f=17Hz、相位差 -20°、偏置角 -30°、顺时针旋转** 的狭窄参数窗口内实现。

---

## 1. 方法

### 1.1 物理模型

与方案一相同，使用 v6.3 LEV/Lee 混合 C_L/C_D 模型，RK4 俯仰积分（dt=50μs, t_end=5.0s），4 分量力模型（平动+附加质量+Clap-and-Fling+旋转力），气动俯仰阻尼。世界系 L/W 定义为：

$$\text{L/W} = \frac{\langle F_{z,\text{world}}\rangle_{\text{steady}}}{mg}$$

即严格垂直方向升力分量除以重量（m=0.020kg, W=196.2mN）。

### 1.2 计算加速

| 组件 | 技术 | 加速比 |
|------|------|--------|
| 单仿真 RK4 热循环 | numba @njit(cache=True, fastmath=True) | **4.8×** |
| 多组合并行 | joblib.Parallel(backend='loky', n_jobs=16) | **~16×** |
| 合并加速 | numba + joblib | **~77×** vs 串行 Python |

numba 编译 4 个标量函数：`cl_cd_blended_scalar` → `_wing_forces_scalar` → `_pitch_rhs_numba` → `_rk4_step_numba`。numba 与 Python 结果的 L/W 偏差 < 0.16%，在工程容差内。

单仿真 5s@50μs (100,000 步)：Python ~53s → numba ~11s → 16核并行等效 ~0.7s/组。

### 1.3 扫描网格

| # | 参数 | 值 | N | 说明 |
|---|------|----|---|------|
| 1 | α_front [°] | 60, 68 | 2 | 前翅安装角 |
| 2 | α_back [°] | 3, 5 | 2 | 后翅安装角 |
| 3 | phase_diff [°] | -20, -10 | 2 | 前后翅相位差 |
| 4 | **mech_a** [mm] | 6, 8, 10, 12 | 4 | 曲柄半径（机构摇杆枢轴 y 坐标） |
| 5 | **mech_R** [mm] | 2.5, 3.0, 3.25 | 3 | 摇杆半径 |
| 6 | φ_offset [°] | -50, -40, -30 | 3 | 翅膀安装偏角 |
| 7 | **f** [Hz] | 13, 15, 17 | 3 | 拍动频率（新参数） |
| 8 | **c_damp** [N·m·s/rad] | 1e-4, 5e-4 | 2 | 人工俯仰阻尼系数（新参数） |
| 9 | **rotation** | cw, ccw | 2 | 曲柄转向（新参数） |

总组合数: 2×2×2×4×3×3×3×2×2 = **3,456 组**

### 1.4 数据输出

每组保存三个文件（与方案一兼容）：
- `config.json` — 完整仿真参数
- `summary.json` — 标量摘要指标（L/W, peak θ, 力/力矩统计等）
- `timeseries.npz` — 全时程数据（34 通道 × 100,000 步）

汇总文件 `sweep_summary.json`（2.2MB）包含所有组合的标量指标。全量数据占用 82GB（3,456 组 × ~25MB/组）。

---

## 2. 结果

### 2.1 全局统计

| 指标 | 数值 |
|------|------|
| 总组合数 | 3,456 |
| 稳定（n_exceed_90=0） | **811 (23.5%)** |
| 发散（n_exceed_90>0） | 2,645 (76.5%) |
| 最佳稳定 L/W | **12.625** |
| 基线 L/W（方案一 v6.5） | 2.505 |
| L/W 提升倍数 | **5.0×** |

### 2.2 Top 20 最佳稳定组合

| # | L/W | Peak [°] | α_f | α_b | Phase | a | R | φ_off | f | c_damp | rot |
|---|------|----------|-----|-----|-------|---|---|-------|---|--------|-----|
| 1 | **12.625** | 35.1 | 60 | 5 | -20 | 6 | 3.0 | -30 | 17 | 5e-4 | cw |
| 2 | 12.609 | 34.9 | 68 | 5 | -20 | 6 | 3.0 | -30 | 17 | 5e-4 | cw |
| 3 | 12.538 | 38.5 | 68 | 5 | -20 | 6 | 3.0 | -30 | 17 | 1e-4 | cw |
| 4 | 12.521 | 35.0 | 68 | 3 | -20 | 6 | 3.0 | -30 | 17 | 5e-4 | cw |
| 5 | 12.506 | 39.1 | 60 | 5 | -20 | 6 | 3.0 | -30 | 17 | 1e-4 | cw |
| 6 | 12.495 | 35.1 | 60 | 3 | -20 | 6 | 3.0 | -30 | 17 | 5e-4 | cw |
| 7 | 12.494 | 39.5 | 68 | 3 | -20 | 6 | 3.0 | -30 | 17 | 1e-4 | cw |
| 8 | 12.452 | 40.3 | 60 | 3 | -20 | 6 | 3.0 | -30 | 17 | 1e-4 | cw |
| 9 | 12.204 | 49.8 | 60 | 5 | -10 | 6 | 3.0 | -30 | 17 | 5e-4 | cw |
| 10 | 12.204 | 46.7 | 68 | 5 | -10 | 6 | 3.0 | -30 | 17 | 5e-4 | cw |
| 11 | 12.190 | 66.6 | 60 | 5 | -10 | 6 | 3.0 | -30 | 17 | 1e-4 | cw |
| 12 | 12.182 | 50.7 | 60 | 3 | -10 | 6 | 3.0 | -30 | 17 | 5e-4 | cw |
| 13 | 12.179 | 61.7 | 68 | 5 | -10 | 6 | 3.0 | -30 | 17 | 1e-4 | cw |
| 14 | 12.154 | 69.1 | 60 | 3 | -10 | 6 | 3.0 | -30 | 17 | 1e-4 | cw |
| 15 | 12.131 | 45.1 | 68 | 3 | -10 | 6 | 3.0 | -30 | 17 | 5e-4 | cw |
| 16 | 12.121 | 63.5 | 68 | 3 | -10 | 6 | 3.0 | -30 | 17 | 1e-4 | cw |

**Top 16 全部共享**：a=6, R=3.0, φ_offset=-30, f=17, rotation=cw。相位差 (-10/-20) 和 α_front/α_back 在次优级变化。

### 2.3 最佳组合详细指标

**组合**: α_f=60°, α_b=5°, phase=-20°, a=6mm, R=3.0mm, φ_offset=-30°, f=17Hz, c_damp=5e-4, rotation=cw

| 指标 | 数值 | 基线对比 |
|------|------|---------|
| **L/W (world)** | **12.625** | 2.505 (5.0×) |
| L/W_body | 13.311 | 4.703 |
| Peak θ_p | 35.1° | 64.8° |
| n_exceed_90 | 0 | 0 |
| mean Fz_world | +2,459 mN | +461 mN (5.3×) |
| mean Fz_body | +2,619 mN | +923 mN |
| mean Fx_body | -8.6 mN | — |
| mean M_aero | +1,257 μN·m | — |
| peak |θ̇_p| | 61.8 rad/s | — |
| peak α_eff_FL | 189.0° | — |
| peak α_eff_BL | 125.7° | — |
| mean C_L (FL) | -0.283 | — |
| mean C_D (FL) | 2.004 | — |

### 2.4 参数敏感度分析

#### 2.4.1 综合分级总表

| 参数 | 敏感度 | 趋势 | 最优值 | 说明 |
|------|--------|------|--------|------|
| mech_a | ⭐⭐⭐⭐⭐ | 越小越好 | **6mm** (网格最小) | a≥8 性能崩溃 |
| rotation | ⭐⭐⭐⭐⭐ | 二元决定 | **cw** | ccw 几乎不可行 (4%稳定) |
| mech_R | ⭐⭐⭐⭐ | 非单调 | **3.0mm** | 3.25 稳定性骤降 |
| f | ⭐⭐⭐⭐ | 单调↑ | **17Hz** (网格最大) | 更高频率待探索 |
| c_damp | ⭐⭐⭐⭐ | 权衡 | 5e-4 (稳) / 1e-4 (性) | 稳定 vs 性能取舍 |
| phase_diff | ⭐⭐ | 非单调 | **-20°** | -15° 未采样 |
| φ_offset | ⭐⭐ | 平坦 | **-40°** (均) / **-30°** (峰) | 差异小 |
| α_front | ⭐ | 极平坦 | 60° 或 68° | 全域 <2% 差异 |
| α_back | ⭐ | 极平坦 | 3° 或 5° | 全域 <2% 差异 |

#### 2.4.2 完整分组统计（仅稳定组合 n90=0）

##### mech_a — 主导参数，唯一有效值为 6mm

| 取值 | 单元 | N_total | N_stable | 稳定率 | 平均 L/W | 标准差 |
|------|------|---------|----------|--------|----------|--------|
| **6** | mm | 864 | **283** | 33% | **5.68** | 3.20 |
| 8 | mm | 864 | 266 | 31% | 1.11 | 0.65 |
| 10 | mm | 864 | 182 | 21% | 0.45 | 0.22 |
| 12 | mm | 864 | 80 | 9% | 0.46 | 0.16 |

##### f — 频率单调递增，上界未探明

| 取值 | 单元 | N_total | N_stable | 稳定率 | 平均 L/W | 标准差 |
|------|------|---------|----------|--------|----------|--------|
| 13 | Hz | 1,152 | **364** | 32% | 1.58 | 1.95 |
| 15 | Hz | 1,152 | 258 | 22% | 2.59 | 2.93 |
| **17** | Hz | 1,152 | 189 | **16%** | **4.10** | 4.05 |

##### mech_R — 非单调，3.0mm 最优

| 取值 | 单元 | N_total | N_stable | 稳定率 | 平均 L/W | 标准差 |
|------|------|---------|----------|--------|----------|--------|
| 2.5 | mm | 1,152 | **402** | 35% | 1.42 | 1.33 |
| **3.0** | mm | 1,152 | 290 | **25%** | **4.52** | 4.08 |
| 3.25 | mm | 1,152 | 119 | 10% | 1.16 | 0.87 |

##### rotation — cw 必需，ccw 灾难

| 取值 | N_total | N_stable | 稳定率 | 平均 L/W | 标准差 |
|------|---------|----------|--------|----------|--------|
| **cw** | 1,728 | **740** | **43%** | **2.68** | 3.12 |
| ccw | 1,728 | 71 | 4% | 0.55 | 0.50 |

##### c_damp — 稳定性与性能的权衡

| 取值 | 单元 | N_total | N_stable | 稳定率 | 平均 L/W | 标准差 |
|------|------|---------|----------|--------|----------|--------|
| 1e-4 | N·m·s/rad | 1,728 | 179 | 10% | **4.61** | 3.45 |
| **5e-4** | N·m·s/rad | 1,728 | **632** | **37%** | 1.89 | 2.62 |

##### phase_diff — -20° 优于 -10°

| 取值 | 单元 | N_total | N_stable | 稳定率 | 平均 L/W | 标准差 |
|------|------|---------|----------|--------|----------|--------|
| -10 | ° | 1,728 | **454** | 26% | 2.17 | 2.88 |
| **-20** | ° | 1,728 | 357 | **21%** | **2.90** | 3.19 |

##### φ_offset — 差异较小

| 取值 | 单元 | N_total | N_stable | 稳定率 | 平均 L/W | 标准差 |
|------|------|---------|----------|--------|----------|--------|
| -50 | ° | 1,152 | 200 | 17% | 2.31 | 2.71 |
| **-40** | ° | 1,152 | 265 | **23%** | **2.58** | 3.04 |
| -30 | ° | 1,152 | **346** | **30%** | 2.52 | 3.21 |

##### α_front — 极平坦

| 取值 | 单元 | N_total | N_stable | 稳定率 | 平均 L/W | 标准差 |
|------|------|---------|----------|--------|----------|--------|
| 60 | ° | 1,728 | 412 | 24% | 2.42 | 3.01 |
| 68 | ° | 1,728 | 399 | 23% | 2.56 | 3.07 |

##### α_back — 极平坦

| 取值 | 单元 | N_total | N_stable | 稳定率 | 平均 L/W | 标准差 |
|------|------|---------|----------|--------|----------|--------|
| 3 | ° | 1,728 | 394 | 23% | 2.55 | 3.05 |
| 5 | ° | 1,728 | 417 | 24% | 2.43 | 3.03 |

### 2.5 交互效应：mech_a × φ_offset

```
           φ_offset = -50°    -40°    -30°
mech_a = 6              4.70    5.78    6.45
mech_a = 8              0.58    1.10    1.48
mech_a = 10             0.13    0.41    0.59
mech_a = 12             0.29    0.37    0.51
```

交互效应显著：（a=6, φ_off=-30°）的组合 L/W=6.45，是单独 a=6 平均 (5.68) 的 1.14×，是 φ_off=-30° (2.52) 的 2.56×。**最优性能需要 a=6 和 φ_off=-30° 同时满足**。

---

## 3. 讨论

### 3.1 方案一 vs 方案二对比

| 维度 | 方案一（单变量） | 方案二（笛卡尔积） |
|------|----------------|-------------------|
| 参数数 | 6 | **9** |
| 总组合 | 48 | **3,456** |
| 检测交互效应 | ❌ | ✅ |
| 最佳 L/W | 8.246 (a=5) | **12.625** |
| 新发现参数 | — | f, c_damp, rotation |
| 关键新结论 | — | ccw=灾难, f↑=好, a≤6必需 |

方案一的最优结论（a 越小越好, φ_off 趋 0, phase=-15°）被方案二**部分修正**：
- a 越小越好 ✅ 确认（但网格未达 a=5）
- φ_off 趋 0 ✅ 确认（-30° 优于 -50°）
- phase=-15° ⚠️ 未覆盖（仅采样 -10° 和 -20°，后者更优）
- **新增**: f 越大越好（趋势），R=3.0 最优（非单调），ccw 不可行，c_damp 需权衡

### 3.2 物理机制解读

**曲柄半径 a 的关键作用**：a 是摇杆枢轴的 y 坐标（机构几何）。更小的 a 使摇杆更靠近曲柄中心，增大拍动摆幅。a=6mm 时摆幅最大（网格内），速度最大 → 平动力 ∝ U²。

**频率 f 的单调效应**：升力 ∝ Ω² ∝ f²，频率从 15→17Hz (+13%) 对应理论升力增加 28%。实测 L/W 从 2.6→4.1 (+58%)，超出简单的 f² 标度，暗示附加质量力（∝ φ̈ ∝ f²）也在增强。

**ccw 的灾难性影响**：ccw 反转急回方向 → 下拍时间占比从 57% 缩至 43% → 下拍产生的主要升力减少 → 同时上拍攻角失序 → 俯仰发散。**实际蝴蝶必须使用 cw 旋转方向**。

**R=3.0 的非单调性**：R=2.5→3.0 升力显著增加（更大的摇杆半径 → 更大的翅膀力臂），但 R=3.25 时俯仰稳定性恶化（M_aero 波动增大）。存在一个最优 R 使气动效率和俯仰稳定性同时最大化。

### 3.3 局限性

1. **粗网格未捕获最优值边界**：a=6 是网格最小点，但方案一显示 a=5 时 L/W=8.2。实际最优可能在 a=5-6 之间。同理 f=17 是网格最大点，最优可能在 f=18-20Hz。
2. **c_damp 不确定性**：c_damp=5e-4 是人工阻尼系数，物理对应性不确定。需要 CFD 或实验标定。
3. **phase=-15° 漏采样**：方案一指出 phase=-15° 是尖锐最优，但方案二只采样了 -10° 和 -20°。不过 Top 1 使用 -20°，说明在 a=6/R=3.0 的最优区域，-20° 可能优于 -15°——交互效应修正了单变量结论。
4. **82GB 数据量**：仅 `sweep_summary.json` (2.2MB) 用于分析。全量 NPZ 用于后续绘图，发散组的 timeseries 可考虑清理。

---

## 4. 结论

1. **最佳参数组合**：α_f=60°, α_b=5°, phase=-20°, a=6mm, R=3.0mm, φ_offset=-30°, f=17Hz, c_damp=5e-4, rotation=cw。**L/W=12.625**（基线 2.505 的 5.0×）。
2. **a=6mm 是唯一有效值**——所有 Top 30 稳定组合均要求 a=6。
3. **频率 f 是未开发的控制维度**——从 13→17Hz 性能持续提升，17Hz 以上未探明。
4. **rotation=ccw 完全不可行**——4% 稳定率，必须使用 cw。
5. **R 存在最优值 3.0mm**——不是越大越好也不是越小越好。
6. **α_front 和 α_back 极不敏感**——在 ±8° 范围内对 L/W 几乎无影响，方案一结论被大规模验证。

### 后续建议

- **精化扫描**：a=5-7mm（步长 0.25）, f=17-22Hz, R=2.75-3.25, phase=-25°~-12° 加密
- **清理数据**：发散组 (2,645) 的 timeseries.npz 可选择性删除以回收 ~60GB
- **绘图**：基于 sweep_summary.json 生成交互热力图和平行坐标图
- **numba 回退验证**：确认 fastmath 导致的 0.15% 偏差不影响结论排序

---

## 附录

### A. 输出文件命名规范与数据读取

#### A.1 目录结构

```
temp/stability/sweep_cartesian/
├── sweep_summary.json          # 全量标量汇总（2.2MB，首读此文件）
├── <combo_id>/                 # 每个参数组合一个子目录
│   ├── config.json             # 该组合的完整仿真参数
│   ├── summary.json            # 该组合的标量摘要指标
│   └── timeseries.npz          # 全时程数据（34通道 × 100k步）
└── ... (3,456 个组合，共 82GB)
```

#### A.2 combo_id 命名规则

目录名（combo_id）由所有 9 个扫描参数的值编码而成，**可直接解码还原**：

```
格式: af{值}_ab{值}_ph{值}_a{值}_R{值}_po{值}_f{值}_cd{值}_rot{值}

编码规则:
  - 负号 "-"  → "n"   (negative)
  - 小数点 "." → "p"   (point)
  - 末尾零自动去除
  - 字符串值保持原样

参数缩写:
  af   = alpha_front_deg    (前翅安装角)
  ab   = alpha_back_deg     (后翅安装角)
  ph   = phase_diff_deg     (前后翅相位差)
  a    = mech_a             (曲柄半径/摇杆枢轴y坐标 [mm])
  R    = mech_R             (摇杆半径 [mm])
  po   = phi_offset_deg     (翅膀安装偏角 [°])
  f    = f                  (拍动频率 [Hz])
  cd   = c_damp             (俯仰阻尼系数 [N·m·s/rad])
  rot  = rotation           (曲柄转向: cw/ccw)
```

**解码示例**：

| combo_id | 解码结果 |
|----------|---------|
| `af60_ab5_phn20_a6_R3_pon30_f17_cd0p0005_rotcw` | α_f=60°, α_b=5°, phase=-20°, a=6mm, R=3.0mm, φ_off=-30°, f=17Hz, cd=5e-4, rot=cw |
| `af68_ab3_phn10_a10_R2p5_pon40_f13_cd0p0001_rotccw` | α_f=68°, α_b=3°, phase=-10°, a=10mm, R=2.5mm, φ_off=-40°, f=13Hz, cd=1e-4, rot=ccw |

**编程解码**（`src/sweep_cartesian.py` 中的函数）：

```python
from sweep_cartesian import combo_to_id, id_to_combo, PARAM_SHORT

# 编码: dict → id
combo_id = combo_to_id({"alpha_front_deg": 60, "mech_a": 6, "rotation": "cw", ...})
# → "af60_ab5_phn20_a6_R3_pon30_f17_cd0p0005_rotcw"

# 解码: id → dict
combo = id_to_combo("af60_ab5_phn20_a6_R3_pon30_f17_cd0p0005_rotcw", 
                     ["alpha_front_deg","alpha_back_deg","phase_diff_deg",
                      "mech_a","mech_R","phi_offset_deg","f","c_damp","rotation"])
# → {"alpha_front_deg": 60, "alpha_back_deg": 5, ...}
```

#### A.3 数据读取方法与 Python API

##### 方式一：读取全量汇总（推荐首选，2.2MB）

```python
import json

with open("temp/stability/sweep_cartesian/sweep_summary.json") as f:
    data = json.load(f)

# data 是一个字典，key 为指标名，value 为 3456 长度的 list
# 指标名: "L/W", "peak_theta_deg", "n_exceed_90", 
#         "mean_Fz_world_mN", "mean_Fz_body_mN", "mean_Fx_body_mN",
#         "mean_M_aero_uNm", "peak_M_aero_uNm", ...
# 参数列: "_param_alpha_front_deg", "_param_mech_a", ...

n = data["_n_combos"]
param_keys = data["_param_keys"]  # 扫描参数列表

# 示例：找出所有稳定组合，按 L/W 排序
stable = []
for i in range(n):
    if data["n_exceed_90"][i] == 0:
        stable.append({
            "id": data["_combo_id"][i],
            "L/W": data["L/W"][i],
            "peak_deg": data["peak_theta_deg"][i],
            "af": data["_param_alpha_front_deg"][i],
            "ab": data["_param_alpha_back_deg"][i],
            "a": data["_param_mech_a"][i],
            "R": data["_param_mech_R"][i],
            "po": data["_param_phi_offset_deg"][i],
            "f": data["_param_f"][i],
            "cd": data["_param_c_damp"][i],
            "rot": data["_param_rotation"][i],
        })
stable.sort(key=lambda x: x["L/W"], reverse=True)

for s in stable[:10]:
    print(f"L/W={s['L/W']:.3f} a={s['a']} R={s['R']} f={s['f']} {s['rot']}")
```

##### 方式二：读取单个组合的摘要

```python
import json

combo_id = "af60_ab5_phn20_a6_R3_pon30_f17_cd0p0005_rotcw"
summary_path = f"temp/stability/sweep_cartesian/{combo_id}/summary.json"

with open(summary_path) as f:
    sm = json.load(f)

print(f"L/W = {sm['L/W']:.3f}")
print(f"Peak θ = {sm['peak_theta_deg']:.1f}°")
print(f"Mean Fz_world = {sm['mean_Fz_world_mN']:.1f} mN")
print(f"n_exceed_90 = {sm['n_exceed_90']}")
```

##### 方式三：读取单个组合的全时程数据（25MB/组）

```python
import numpy as np
import json

combo_id = "af60_ab5_phn20_a6_R3_pon30_f17_cd0p0005_rotcw"
base = f"temp/stability/sweep_cartesian/{combo_id}"

# 加载时程（34个通道）
ts = np.load(f"{base}/timeseries.npz")

# 可用通道:
#   t, theta_p, theta_dot, theta_ddot          # 俯仰状态
#   Fz_body_total, Fx_body_total, Fz_world_total  # 合力
#   M_aero, M_grav, M_damp                     # 力矩分量
#   FL_Fz_body, FR_Fz_body, BL_Fz_body, BR_Fz_body  # 单翅力
#   FL_alpha_eff, FR_alpha_eff, BL_alpha_eff, BR_alpha_eff  # 有效攻角
#   FL_C_L, FL_C_D, ...                         # 气动系数
#   FL_phi, FR_phi, ...                         # 拍动角

t = ts["t"]
theta = np.rad2deg(ts["theta_p"])
Fz_world = ts["Fz_world_total"] * 1000  # → mN

# 稳态段（后一半）
half = len(t) // 2
avg_Fz = np.mean(Fz_world[half:])
print(f"Steady Fz_world = {avg_Fz:.1f} mN")
```

##### 方式四：用模块工具函数批量遍历

```python
from sweep_cartesian import (
    build_cartesian_grid, run_one_combo, combo_to_id, OUT_ROOT, DEFAULT_GRID
)

# 获取所有 combo_id
grid = DEFAULT_GRID  # 或自定义
combos = build_cartesian_grid(grid)
combo_ids = [combo_to_id(c) for c in combos]

# 遍历已有结果
import json
for cid in combo_ids:
    summary_path = OUT_ROOT / cid / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            sm = json.load(f)
        # 处理 sm ...
```

#### A.4 文件格式详解

##### config.json

```json
{
  "alpha_front_deg": 60,
  "alpha_back_deg": 5,
  "phase_diff_deg": -20,
  "mech_a": 6.0,
  "mech_R": 3.0,
  "phi_offset_deg": -30,
  "f": 17.0,
  "c_damp": 0.0005,
  "rotation": "cw",
  "m_total": 0.02,
  "I_yy": 3e-5,
  "dt": 5e-05,
  "t_end": 5.0,
  ... (全部 SimulationConfig 字段)
}
```

##### summary.json

```json
{
  "_combo_id": "af60_ab5_phn20_a6_R3_pon30_f17_cd0p0005_rotcw",
  "_combo": {"alpha_front_deg": 60, "mech_a": 6, ...},
  "L/W": 12.625,
  "L/W_body": 13.311,
  "peak_theta_deg": 35.15,
  "n_exceed_90": 0,
  "weight_mN": 196.2,
  "mean_Fz_body_mN": 2618.77,
  "mean_Fz_world_mN": 2458.84,
  "mean_Fx_body_mN": -8.56,
  "mean_M_aero_uNm": 1257.47,
  "peak_M_aero_uNm": 945109.64,
  "mean_M_grav_uNm": ...,
  "mean_M_damp_uNm": ...,
  "mean_abs_thetadot_rads": 7.49,
  "peak_abs_thetadot_rads": 61.80,
  "mean_abs_thetaddot_rads2": 3953.52,
  "peak_alpha_eff_FL_deg": 189.02,
  "peak_alpha_eff_BL_deg": 125.69,
  "mean_CL_FL": -0.283,
  "mean_CD_FL": 2.004
}
```

##### timeseries.npz

```
34 个通道，每通道 100,000 个 float64 (5s / 50μs)

通道清单:
  运动学:  t, theta_p, theta_dot, theta_ddot
  合力:    Fz_body_total, Fx_body_total, Fz_world_total
  力矩:    M_aero, M_grav, M_damp
  单翅力:  FL_Fz_body, FR_Fz_body, BL_Fz_body, BR_Fz_body,
           FL_Fx_body, FR_Fx_body, BL_Fx_body, BR_Fx_body
  攻角:    FL_alpha_eff, FR_alpha_eff, BL_alpha_eff, BR_alpha_eff
  气动:    FL_C_L, FL_C_D, FR_C_L, FR_C_D, BL_C_L, BL_C_D, BR_C_L, BR_C_D
  拍动:    FL_phi, FR_phi, BL_phi, BR_phi

加载: np.load("timeseries.npz")  → 类 dict 对象 (惰性加载)
      np.load("timeseries.npz")["theta_p"]  → (100000,) float64 数组
```

#### A.5 筛选示例

```python
"""快速筛选: 找出所有 rotation=cw, a=6, f=17 的稳定组合. """
import json, numpy as np

with open("temp/stability/sweep_cartesian/sweep_summary.json") as f:
    d = json.load(f)

results = []
for i in range(d["_n_combos"]):
    if (d["n_exceed_90"][i] == 0 
        and d["_param_rotation"][i] == "cw"
        and d["_param_mech_a"][i] == 6
        and d["_param_f"][i] == 17):
        
        results.append({
            "id": d["_combo_id"][i],
            "L/W": d["L/W"][i],
            "peak": d["peak_theta_deg"][i],
            "ph": d["_param_phase_diff_deg"][i],
            "R": d["_param_mech_R"][i],
            "po": d["_param_phi_offset_deg"][i],
        })

results.sort(key=lambda x: x["L/W"], reverse=True)
for r in results:
    print(f"L/W={r['L/W']:.3f} R={r['R']} po={r['po']} ph={r['ph']} | {r['id']}")
```

### B. 计算方法

```bash
# 完整方案二扫描
python src/sweep_cartesian.py --n-jobs 16 --t-end 5.0 --dt 50e-6

# 查看网格
python src/sweep_cartesian.py --list-grid

# 自定义网格 (JSON 文件)
python src/sweep_cartesian.py --grid my_grid.json --n-jobs 8
```

### C. 网格定义

编辑 `src/sweep_cartesian.py` 中的 `DEFAULT_GRID` 字典即可调整扫描范围和分辨率：

```python
DEFAULT_GRID = {
    "alpha_front_deg":  [60, 68],
    "alpha_back_deg":   [3, 5],
    "phase_diff_deg":   [-20, -10],
    "mech_a":           [6, 8, 10, 12],
    "mech_R":           [2.5, 3.0, 3.25],
    "phi_offset_deg":   [-50, -40, -30],
    "f":                [13, 15, 17],
    "c_damp":           [1e-4, 5e-4],
    "rotation":         ["cw", "ccw"],
}
```

### D. 输出文件结构

```
temp/stability/sweep_cartesian/
├── sweep_summary.json          # 汇总（2.2MB）
├── af60_ab5_phn10_a6_R3_pon30_f17_cd0p0005_rotcw/
│   ├── config.json
│   ├── summary.json
│   └── timeseries.npz
└── ... (3456 个组合目录)
```

### E. 与方案一结果对比

| 参数 | 方案一最优 | 方案二最优 | 一致性 |
|------|----------|----------|--------|
| α_front | 68° | 60° | 平坦（<2%差异） |
| α_back | 5° | 5° | ✅ |
| phase_diff | **-15°** | **-20°** ⚠️ | 接近，但未采样-15° |
| mech_a | 5.0 | 6.0 (网格最小) | 趋势一致（越小越好） |
| mech_R | 3.0 | 3.0 | ✅ |
| φ_offset | -30° | -30° | ✅ |
| f | N/A | 17 | 新参数 |
| c_damp | N/A | 5e-4 | 新参数 |
| rotation | N/A | cw | 新参数 |

### F. 模块 API 参考

#### F.1 `sweep_cartesian.py` — 方案二笛卡尔积扫描

```python
# ---- 网格生成 ----
build_cartesian_grid(grid_spec: dict = None) -> list[dict]
"""从 {param: [values]} 生成所有笛卡尔积组合.
   默认使用 DEFAULT_GRID.
   返回: [{"alpha_front_deg": 60, "mech_a": 6, ...}, ...]
"""

# ---- combo_id 编解码 ----
combo_to_id(combo: dict) -> str
"""参数组合 → 紧凑文件夹名. 可逆.
   例: {"alpha_front_deg": 60, "mech_a": 6} → "af60_ab5_..._a6_..."
"""

id_to_combo(combo_id: str, grid_keys: list) -> dict
"""文件夹名 → 参数组合 (反向解码).
   例: "af60_a6_R3_pon30_f17_rotcw" → {"alpha_front_deg": 60, "mech_a": 6, ...}
"""

# ---- 单组合运行 (joblib worker) ----
run_one_combo(combo: dict, out_root: Path = None,
              base_overrides: dict = None,
              t_end: float = 5.0, dt: float = 50e-6,
              verbose: bool = False) -> dict
"""运行单个参数组合, 保存 {config,summary,timeseries} 到磁盘.
   已完成则跳过 (断点续跑).
   返回: summary dict (含 _combo_id, _combo, L/W, ...)
"""

# ---- 主入口 ----
sweep_cartesian(grid_spec: dict = None,
                base_overrides: dict = None,
                n_jobs: int = -1,
                t_end: float = 5.0, dt: float = 50e-6,
                out_root: Path = None) -> list[dict]
"""笛卡尔积参数扫描 — joblib 并行.
   Args:
     grid_spec: {param: [values]}, 默认 DEFAULT_GRID
     n_jobs: 并行 worker 数, -1=全部核心
   Returns:
     [{summary}] 所有组合的结果
"""

# ---- 网格定义 (可直接编辑) ----
DEFAULT_GRID: dict
PARAM_SHORT: dict  # 参数名 → 缩写映射

# ---- 输出路径 ----
OUT_ROOT: Path  # temp/stability/sweep_cartesian/
```

**CLI 使用**:
```bash
python src/sweep_cartesian.py --n-jobs 16 --t-end 5.0 --dt 50e-6
python src/sweep_cartesian.py --list-grid                        # 查看默认网格
python src/sweep_cartesian.py --grid my_grid.json --n-jobs 8    # 自定义网格
```

#### F.2 `stability_analysis.py` — 方案一单变量扫描

```python
# ---- 基线 ----
run_baseline(config_overrides: dict = None, out_dir: Path = None,
             t_end: float = 5.0, dt: float = 50e-6) -> dict
"""运行基线分析, 保存到 temp/stability/baseline/.
   返回: summary dict
"""

# ---- 单变量偏离 ----
sweep_parameter(param_name: str, values: list = None,
                base_overrides: dict = None,
                t_end: float = 5.0, dt: float = 50e-6) -> dict
"""单变量偏离扫描. 从 BASELINE_CONFIG 出发, 每次只改 param_name.
   支持断点续跑 (checkpoint.json).
   Args:
     param_name: 参数名 (必须在 SWEEP_RANGES 中)
     values: 参数值列表, 默认用 SWEEP_RANGES[param_name]
   返回: sweep_summary dict
"""

# ---- 扫描范围 ----
SWEEP_RANGES: dict  # {param_name: [values]}
BASELINE_CONFIG: dict
PARAM_FORMATS: dict  # 参数名 → 短格式

# ---- 输出路径 ----
OUT_ROOT: Path       # temp/stability/
BASELINE_DIR: Path   # temp/stability/baseline/
```

**CLI 使用**:
```bash
python src/stability_analysis.py --baseline
python src/stability_analysis.py --sweep mech_a
```

#### F.3 两模块共用输出格式

```
temp/stability/
├── baseline/                          # 方案一基线
│   ├── config.json / summary.json / timeseries.npz
├── sweep_<param>/                     # 方案一单变量
│   ├── checkpoint.json
│   ├── sweep_summary.json
│   └── <value>/
│       ├── config.json / summary.json / timeseries.npz
└── sweep_cartesian/                   # 方案二笛卡尔积
    ├── sweep_summary.json
    └── <combo_id>/
        ├── config.json / summary.json / timeseries.npz
```

两模块的 `config.json` / `summary.json` / `timeseries.npz` 格式完全一致，绘图模块可跨方案复用。

