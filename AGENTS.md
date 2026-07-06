# Butterfly Aerodynamic Analysis — Agent Notes

## Project Overview

仿生蝴蝶扑翼微型飞行器（MAV）俯仰动力学仿真。质量 **23 g**，四连杆机构驱动，15 Hz 扑动频率。验证俯仰稳定性、升力/重量平衡、以及非对称安装角的优化。

**坐标系已与 SolidWorks 对齐**：

- 体轴原点 = 总质心 CG
- **X**: 左右（右为正）
- **Y**: 上下（上为正）
- **Z**: 前后（前为正）
- **翅膀拍动轴**: 平行于 Z（左右翅共轴）
- **机身俯仰轴**: 平行于 X（过 CG，因此重力对俯仰力矩为 0）

## Current State (2026-07-06)

**当前主力模型：v6.9** — 新四连杆机构 + YAML 统一配置 + DESIGN_v69，**坐标系已与 SolidWorks 对齐**。

### 版本演进

| 版本 | 文件 | 核心特点 | 关键结果 |
|------|------|---------|---------|
| v6 | `temp/pitch_dynamics_v6.py` | 刚性 wing, 非对称 α_install 扫描 | L/W=1.48 但俯仰全部发散 |
| **v6.7** | `src/aero/butterfly_forces.py` | **修正攻角定义**[32]: α=η (相对拍动平面) + tanh平滑过渡 | **L/W=2.15**, peak θ=37.6°, 物理自洽 ✅ |
| **v6.8** | `src/aero/butterfly_forces.py` | **速度耦合 clap-fling** + 全网格扫参 + DESIGN_v68 | **L/W=2.45**, peak θ=32.9°, 物理合理 ✅ |
| **v6.9** | `config/design_v69.yaml` + `src/config.py` | **新机构参数**（摆幅99°）+ **YAML 单一权威源** | **L/W=4.43**, peak θ=65.8°, α_f=60/α_b=3/phase=-15 |
| **v6.9-SW** | `src/aero/butterfly_forces.py` 等 | **坐标系对齐 SolidWorks**：X=右/Y=上/Z=前，拍动轴=Z，俯仰轴=X，新质量/惯量/铰链位置 | 2500 组扫参，180 个稳定组合，L/W 范围 -2.63~2.95 |

### 最佳参数 (v6.9 DESIGN_v69)

**参数唯一来源**：`config/design_v69.yaml`。所有仿真脚本通过 `src/config.py` 统一读取。

| 参数 | v6.8 (旧) | v6.9 (新) | 说明 |
|------|-----------|-----------|------|
| α_f | 45° | **60°** | 前翅安装角 |
| α_b | 8° | **3°** | 后翅安装角 |
| phase | -20° | **-15°** | 前后翅相位差 |
| mech_a | 6.0 mm | **7.6 mm** | 翅膀转轴 A 的 y 坐标 |
| mech_b | — | **1.71 mm** | 圆心 B 的 x 坐标（新机构） |
| mech_R | 2.50 mm | **3.8 mm** | 曲柄半径（+52%，摆幅 ~99°） |
| mech_c | — | **7.1 mm** | 连杆 P1-P2 长度 |
| mech_l | — | **5.0 mm** | 摇杆/翅膀杆 A-P2 长度 |
| φ_offset | -30° | **0°** | 翅膀安装偏角（新机构无偏移） |
| f | 17 Hz | 17 Hz | 扑动频率 |
| k_clap | 0.3 | 0.3 | 速度耦合 clap-fling k_max |
| c_damp | 5e-4 | 5e-4 | 俯仰气动阻尼 |
| Grashof | — | **3.8+7.79=11.59 ≤ 7.1+5.0=12.10 ✅** | 曲柄摇杆条件满足 |

**性能**: L/W_world=4.43, peak θ=65.8°, 摆幅 ~99°, K≈1.0

> **注意**：上述 v6.9 性能基于旧坐标系（X=前/Y=右/Z=上）。坐标系对齐 SolidWorks 后，设计点需在新坐标系下重新验证。

### 参数配置系统

```python
from src.config import get_design, get_mech_params, get_version, get_sweep_grid

design = get_design()       # → dict，同原 DESIGN_v69
mech   = get_mech_params()  # → dict，机构参数
grid   = get_sweep_grid()   # → dict，扫参网格（列表=扫参，标量=固定）
```

`config/design_v69.yaml` 四个键：`mechanism`, `aero`, `physical`, `numerical` + `sweep` 段（扫参网格 + `_n_jobs`/`_out_dir` 引擎配置）。修改 YAML 即可影响所有模块。

### 对外力输出模块: `src/aero/butterfly_forces.py`

对外调用接口，提供完整仿真管线（运动学→俯仰ODE→力输出），**全参数可配置**：

```python
from src.aero.butterfly_forces import SimulationConfig, ButterflyForceModel, scan_parameters

# 单次仿真
cfg = SimulationConfig(alpha_front_deg=60, alpha_back_deg=8, phase_diff_deg=0,
                        dt=10e-6, t_end=10.0)
model = ButterflyForceModel(cfg)
out = model.simulate()
# → out.wings["FL"].force_body     # (N,3) 前翅左体轴力
# → out.wings["FL"].rocker_principal_vec     # 摇杆主矢
# → out.wings["FL"].rocker_principal_moment  # 摇杆主矩
# → out.summary["L/W"]             # 升力/重量比

# 参数扫描（替代 temp 扫描脚本）
results = scan_parameters(cfg, {
    "alpha_front_deg": [28,32,35,38,40,42,45,48,50,55,60],
    "alpha_back_deg": [8,10,12,15,18,20,22,25,30],
})
```

**可配置参数**：α_install (前后), 相位差, 机构 a/b/R/c/l, φ_offset, 频率 f, 时间步长 dt, 物理常数, 气动系数等。详见 `SimulationConfig` dataclass。

## Physical Model

### 角度体系（新坐标系）

- **α_install**: 翅膀弦线相对机身水平面（XY 平面）的固定安装角（出厂设定，飞行中不变）
- **φ**: 机构拍动角，新机构摆幅 ~99°（范围 ~[-49.5°, +49.5°]）
- **θ_p**: 机身俯仰角，绕 X 轴旋转（动态变量）
- **α_eff**: 有效气动攻角 = -sign(φ̇) × (α_install + Δα_damp)，沿用文献[32]约定
- **坐标变换**: `body_to_world(theta_p) = R_x(theta_p)`，θ_p=0 时体轴与世界系重合

### 力投影（新坐标系）

翅膀绕 Z 轴拍动，弦线在 XY 平面扫掠：

```text
Fx = sin(φ) × (sign(φ̇) × D − L)
Fy = cos(φ) × (L − sign(φ̇) × D)
Fz = 0
```

- 升力 L、阻力 D 定义在翅膀局部坐标系
- `sign(φ̇)` 上下拍符号翻转（实际用 tanh 平滑）
- 世界系力通过 `R_x(θ_p)` 旋转得到

### 俯仰力矩（新坐标系）

俯仰轴为 X 轴，过总质心：

```text
M_aero = 2 × (−z_front × Fy_f − z_back × Fy_b)
M_damp = −c_damp × θ̇_p
θ̈_p = (M_aero + M_damp) / I_xx
```

- 重力作用线过俯仰轴，因此**无重力俯仰力矩**
- `z_front > 0`（前翅力作用点在前），`z_back < 0`（后翅力作用点在后）
- 前翅向上力 `Fy_f > 0` 产生低头力矩（M_aero < 0）

## File Structure

```
├── src/
│   ├── aero/
│   │   ├── butterfly_forces.py       # ★ 对外力输出模块 (v6.9-SW 坐标系)
│   │   ├── stability_analysis.py     # 方案一: 单变量偏离扫描
│   │   ├── stability_plot.py         # 方案一: 稳定性绘图
│   │   ├── sweep_cartesian.py        # 方案二: 笛卡尔积全扫描 (joblib并行, 支持追加合并)
│   │   ├── analyze_sweep_results.py  # 扫参结果分析：同步磁盘/json、去重、生成 Markdown/HTML 报告
│   │   └── plot_wing_shape.py        # 翅膀平面形状绘制
│   ├── struct/
│   │   ├── mechanism.py              # 曲柄摇杆四连杆 → φ(t), φ̇(t), φ̈(t)
│   │   ├── mechanism_plot.py         # 机构运动学可视化
│   │   ├── four_bar_analysis.py      # 四连杆动力学分析
│   │   ├── analyze_dxf.py            # DXF 几何提取 → 翅膀面积/展长/面积矩
│   │   ├── mechanical_principles_analysis.py  # 机械原理分析（力流/力矩）
│   │   └── generate_mechanical_principles_assets.py  # 机械原理报告配图生成
│   └── gear/
│       └── gear_analysis.py          # 两级定轴齿轮减速器分析
├── data/
│   ├── WingFront.DXF / WingBack.DXF / WingsAxis.DXF
│   └── wing_analysis_results.json
├── temp/
│   ├── pitch_dynamics_v6_1.py    # v6.1-v6.3 物理引擎 (历史)
│   ├── scan_v6_3.py              # v6.3 99组扫描脚本
│   ├── verify_v63_long.py        # v6.3 10s长稳验证
│   ├── discover_stable/          # v6 偏离发现脚本
│   ├── v63_long/ v6_fixed_scan/ v6_moving_scan/ v61_long_stability/
│   └── stability/                # ☆ 扫参输出 (旧bug公式数据已废弃, 可清)
├── docs/
│   ├── 仿生蝴蝶翅膀空气动力学分析文献综述.md  # ★ 文献综述 (公式来源, [11][24][32])
│   ├── butterfly_forces_使用说明.md       # 模块 API 文档 (v6.8)
│   ├── mechanism.md / mechanism_plot.md   # 机构文档
│   ├── analyze_dxf.md / gear_analysis.md  # 分析工具文档
└── AGENTS.md                     # 本文件
```

## Planned: 水平运动 (Vx/前飞)

**状态：计划中，尚未实施**

当前 fixed 模型假设蝴蝶身体不移动（Vx=Vz=0），仅 pitch 自由。下一步加入水平移动：

1. **新增状态变量**：body_x, body_Vx（RK4 扩至 4 维）
2. **来流修正**：相对速度 v_rel = v_flap + v_body，影响有效攻角
3. **预期效果**：前飞时来流水平分量打破上拍/下拍对称性 → 自然提升净升力
4. **接口兼容**：`butterfly_forces.py` 已预留 `BodyState.velocity` 和 `body_to_world(position=...)`

实施步骤：

- 在 `butterfly_forces.py` 中添加 `simulate_moving()` 方法或通过 `SimulationConfig.mode='moving'` 切换
- 扫描 Vx 稳定解（预期 Vx 稳态 ≈ 3-5 m/s）
- 验证 moving 模型下的 L/W 是否进一步提升

## Stability Analysis System

`src/aero/stability_analysis.py` + `src/aero/stability_plot.py` — 插拔式稳定性分析管线:

```text
analysis (stability_analysis.py) ──[JSON+NPZ]──> plot (stability_plot.py)
         │                                              │
         └── temp/stability/ ───────────────────────────┘
              ├── baseline/{config,summary,timeseries}
              └── sweep_<param>/<value>/{config,summary,timeseries}
```

- **分析模块**: 单变量偏离扫描, 每完成一个值立即保存 checkpoint, 支持断点续跑
- **绘图模块**: 完全独立, 只读文件。基线 5 张图 + 每参数 6 张偏离图
- **v6.8 基线**: `DESIGN_v68` (α_f=45/α_b=8, phase=-20, a=6.0, R=2.5, φ_offset=-30, k_clap=0.3, L/W_world=2.447, peak θ=32.9°)
- **方案一扫描**: 7参数32值真正单变量偏离（其他参数锁死 DESIGN_v68），全部完成 (~7 min)。结果见 `docs/v68_final_report.md`

### v6.6 方案二: 9参数全笛卡尔积扫描

`src/aero/sweep_cartesian.py` — joblib 并行 + numba JIT 全组合扫描:

```text
sweep_cartesian.py
  ├─ build_cartesian_grid(grid_spec) → [{param: value}, ...]
  ├─ combo_to_id(combo) → "af60_ab5_phn20_a6_R3_..."
  ├─ run_one_combo(combo) → 保存 {config,summary,timeseries}
  └─ sweep_cartesian(grid_spec, n_jobs=-1) → sweep_summary.json
```

- **3456 组** 9 参数粗网格全扫描，56.7 min (16 核, numba 4.8×) ⚠️ 基于 v6.6 bug 公式, 已废弃
- **v6.8 扫参**: **11,809 组** 10 参数 (含 k_clap), 存储于 `F:\...\temp\stability\sweep_cartesian\`, DESIGN_v68 已确定
- **v6.9-SW 扫参**: **2500 组** 基于新坐标系（X=右/Y=上/Z=前），存储于 `F:\重大作业考试\26秋\机械原理\sweep_all`
- 输出与方案一完全兼容 (config/summary/timeseries)，绘图模块可复用
- 详细报告见 `docs/v68_final_report.md`

### v6.9-SW 笛卡尔积扫参关键发现

基于新坐标系（X=右/Y=上/Z=前，拍动轴=Z，俯仰轴=X）的 2500 组全扫描：

| 参数 | 扫描范围 |
|------|---------|
| α_f | [-15, 0, 15, 30, 45] ° |
| α_b | [-15, 0, 15, 30, 45] ° |
| phase | [-180, -90, 0, 90, 180] ° |
| k_clap | [0.3, 0.5] |
| φ_offset | [-30, -15, 0, 15, 30] ° |
| f | [10, 15] Hz |

**结果统计**:

- 总组合数：2500
- 稳定组合数（n_exceed_90 == 0）：180
- 稳定组合 L/W 范围：-2.63 ~ 2.95
- 稳定组合中 L/W > 1 的数量：3

**Top 3 稳定组合**:

| 排名 | combo_id | L/W | peak θ | α_f | α_b | phase | k_clap | φ_offset | f |
|------|----------|-----|--------|-----|-----|-------|--------|----------|---|
| 1 | afn15_ab45_ph90_kc3en01_po0_f15 | 2.949 | 85.2° | -15 | 45 | 90 | 0.3 | 0 | 15 |
| 2 | afn15_ab45_ph90_kc3en01_po0_f10 | 1.264 | 83.3° | -15 | 45 | 90 | 0.3 | 0 | 10 |
| 3 | af0_ab45_ph0_kc3en01_po0_f10 | 1.119 | 87.9° | 0 | 45 | 0 | 0.3 | 0 | 10 |

**关键观察**:

- 小安装角（α_f≈0°, α_b≈0°）易稳定但升力不足
- 高 L/W 组合集中在 α_b=45° 且 φ_offset=0° 附近
- 新坐标系下仍需进一步细化网格寻找 L/W>1 且 peak θ<60° 的设计点

### v6.8 方案一关键发现（真正单变量偏离）

### 扫参结果分析工具

| 工具 | 说明 |
|------|------|
| `src/aero/analyze_sweep_results.py` | 读取 `sweep_summary.json`，同步磁盘 combo 文件夹与 JSON（追加/删除/去重），生成 Markdown 报告 |
| `src/aero/sweep_cartesian.py` | 全参数笛卡尔积扫描；已有 `sweep_summary.json` 时自动追加合并，而非覆盖 |

```bash
# 同步并生成报告（默认目录来自 config/design_v69.yaml 的 _out_dir）
python src/aero/analyze_sweep_results.py --sync

# 指定目录并限制 peak theta
python src/aero/analyze_sweep_results.py "F:/重大作业考试/26秋/机械原理/sweep_all" --sync --max-peak-theta 60
```

**结论**: `DESIGN_v68` 本身稳定，但参数容差很窄，邻近组合大量发散。

| 参数 | 扫描范围 | 稳定子范围 | 发散范围 | DESIGN_v68 位置 |
|------|---------|-----------|---------|----------------|
| α_f | 30,40,50,55,60,70° | 50-70° | 30-40° | 45°（设计点在稳定区边缘） |
| α_b | 3,5,8,10,15° | 3-8° | 10-15° | 8°（上限边界） |
| phase | -30,-25,-20,-15,-10° | -20 ~ -10° | -30,-25° | -20°（下限边界） |
| mech_a | 6,7,8 mm | 6 | 7,8 | 6（唯一稳定值） |
| mech_R | 2.0,2.25,2.5 mm | 2.5 | 2.0,2.25 | 2.5（唯一稳定值） |
| φ_off | -40,-35,-30,-25° | -40 ~ -30° | -25° | -30°（上限边界） |
| k_clap | 0.3,0.5,0.8,1.0,1.5 | 0.3 | 0.5-1.5 | 0.3（唯一稳定值） |

**与全网格平均结果的区别**: 全网格平均显示 α_f=60/70 等仍有高 L/W，但那是因为其他参数（如低 α_b、特定 phase）组合被一起平均。真正单变量偏离（其他参数锁死 DESIGN_v68）表明，单独提高 α_f 到 50° 以上反而比设计点更稳定且 L/W 更高；而 α_f=30/40 会发散。

```bash
# 基线分析
python src/aero/stability_analysis.py --baseline

# 单变量偏离 (e.g. mech_a)
python src/aero/stability_analysis.py --sweep mech_a

# 批量全部扫描
python temp/run_all_sweeps.py

# 出图
python src/aero/stability_plot.py --baseline
python src/aero/stability_plot.py --sweep mech_a
python src/aero/stability_plot.py --all
```

### TODO

- [x] **numba JIT 加速** ✅ — 标量 @njit 编译 RK4 热循环, 4.8× 单仿真加速
- [x] **全参数笛卡尔积扫描** ✅ — 11,809 组, 16核 joblib
- [x] **攻角公式修正** ✅ — 文献[32]标准: α=η (相对拍动平面), tanh平滑
- [x] **Clap-and-Fling 物理模型修正** ✅ — 速度-位置耦合, Lighthill公式启发
- [x] **k_clap 语义变更** ✅ — k_max=0.3, 速度耦合
- [x] **扫参分析** ✅ — 11,809 组, 物理最优 α_f=50°/α_b=8°, L/W=2.81
- [x] **物理合理性校准** ✅ — 约束 α_eff≤50°, 排除 α_f≥60° 非物理结果
- [x] **严格单变量稳定性分析** ✅ — 7参数32值, DESIGN_v68 容差窄
- [x] 用扫描结果更新默认参数和推荐配置 ✅ — DESIGN_v68
- [ ] **方案二绘图** — 交互热力图、平行坐标图、Sobol 敏感性指数

---

## v6.8: Clap-and-Fling 速度耦合修正 (已实现 + 已分析)

**状态**: 代码已实现 ✅ | k_max 标定完成 ✅ | 扫参 10,326 组 ✅ | 扩扫 α_f=60-70 待做 ⏳

### 实现

```python
# butterfly_forces.py — 三处均已更新:
# ① compute_clap_fling_window(phi, phi_dot, edge_width=0.10)
#    → 返回 k_extra = |φ̇|/φ̇_peak × cos²窗(距端点距离)
# ② compute_wing_forces_vec: k_clap = 1.0 + (config.k_clap - 1.0) * k_extra
# ③ RK4 预计算: kcl_f/kcl_b 同步更新
```

### 当前表现 (k_clap=0.3, L/W=2.45, peak_θ=32.9°, mean_α_eff=45.9°) — v6.8 物理合理范围

- 端点处 k_clap=1.0 (φ̇→0 增强自动归零)
- 峰值 k_clap≈1.06 在 crank≈225° (端点前, 速度尚存)
- 力曲线无 kink
- 增强很弱 (~6%), 与 Lighthill 公式的物理量级一致
- **扫参结论**: k_clap=0.3 最优, 不可更高; 过高 k_clap 破坏俯仰稳定性
- **最优参数**: 见上 DESIGN_v68

### 问题

当前实现 (`butterfly_forces.py:452-453`):

```python
in_reversal = np.abs(phi_dot) < 0.1 * phi_dot_peak
k_clap = np.where(in_reversal, config.k_clap, 1.0)
```

三个缺陷：

1. **硬二值开关**: k_clap 在阈值处瞬间跳变 30%，曲线出现 kink
2. **速度判据倒置**: 在 |φ̇|≈0 处增强 30%，但此时气动力∝U²≈0，增强几乎无实际作用
3. **非物理**: 文献[36-39] 明确 clap-and-fling 增强来自张开速度产生的额外环量

### 文献依据

Lighthill(1973) 环量公式: Γ = g(λ)·φ̇·c²

- Γ: 额外环量，正比于**张开角速度 φ̇**
- g(λ) ≈ 2 (张开角 < 30°)，随翅间距增大而衰减
- 粘性修正 (Maxworthy 1979): Γ ≈ 6Ωc² (8.7× 增强)

物理本质: 两翅分离时，间隙中的射流产生额外环量。**速度越快、翅越近，增强越强**。

### 实现方案

将 clap-and-fling 建模为对 C_L/C_D 的速度-位置耦合增强：

```
k_clap_extra = k_max · (|φ̇| / φ̇_peak) · window(φ)
k_clap = 1.0 + k_clap_extra
```

- **k_max**: 最大增强系数 (待标定，初始 ~0.5, 使平均 L/W 增加 ~30%)
- **|φ̇|/φ̇_peak**: 速度归一化 — 端点处 φ̇→0, 增强自然归零
- **window(φ)**: 位置余弦平方窗 — 翅越靠近端点, 左右翅间距越小, 增强越强
  - edge_width = 0.10 (端点 10% 拍动范围内激活, ~5.6°)
  - 余弦平方平滑过渡, 无硬开关

预期效果:

- 增强峰值出现在**端点附近且速度尚存**的位置 (而非端点处)
- 与位置窗对比: 增强作用于速度>0 的区域, 真正参与动力学
- 需要重新标定 k_max 使平均增强匹配文献的 ~30%

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `src/aero/butterfly_forces.py` | ① 新增 `compute_clap_fling_window()` 辅助函数 |
| | ② `compute_wing_forces_vec` 中替换 k_clap 计算 |
| | ③ 预计算 k_clap 数组 (numa + Python 两路径) |
| | ④ `SimulationConfig.k_clap` 语义变为 k_max |
| `src/aero/stability_analysis.py` | BASELINE_CONFIG 参数不变 (k_clap 仍为初始值) |
| `src/aero/sweep_cartesian.py` | BASELINE_CONFIG 同步 |
| `AGENTS.md` | 更新模型描述 + DESIGN 参数 |

### 参数重扫描 ✅ (10,326 组完成, 待扩扫)

新 clap-fling 模型下扫参已完成 10,326/45,000 组，覆盖 α_f=50-55°:

1. **扫参完成**: α_f=[50,55], α_b=[3,5,8,10,15], phase=[-30~-10], a=[6,7,8], R=[2.0,2.25,2.50], φ_off=[-40~-25], f=[15,17], k_clap=[0.3,0.5,0.8,1.0,1.5]
2. **分析完成**: `python temp/analyze_sweep_v68.py` → DESIGN_v68 已确定
3. **待扩扫**: α_f=[60,65,70] + α_b 加密 [1,2,4,6] — 确认 α_f 更大时 L/W 是否继续提升

### 验证标准

- [ ] 力/力矩曲线无 kink
- [ ] k_clap 在端点处 = 1.0 (速度→0, 增强归零)
- [ ] 平均 L/W 在合理范围 (1.5-3.0)
- [ ] 俯仰稳定 (peak θ < 90°, n90=0)
- [x] 新 DESIGN 参数 L/W=2.45，符合物理预期

## Key Parameters

**参数唯一来源**: `config/design_v69.yaml` → `src/config.py` → 所有脚本

```python
from src.config import get_design, get_mech_params
design = get_design()          # DESIGN_v69 完整 dict
mech   = get_mech_params()     # 机构参数子集
```

### v6.9-SW 物理参数（SolidWorks 对齐坐标系）

```python
# 坐标系：X=右, Y=上, Z=前；拍动轴=Z, 俯仰轴=X（过 CG）
PHYS = {
    "m_total": 0.023,          # 总质量 [kg]（SW 实测 23 g）
    "I_xx": 4.0668e-5,         # 绕 X 轴（俯仰轴）转动惯量 [kg·m²]
    "x_hinge_right": 0.009625, # 右翅铰链 X 坐标（相对 CG，右为正）[m]
    "x_hinge_left": -0.005935, # 左翅铰链 X 坐标（相对 CG，右为正）[m]
    "y_hinge_rel": 0.002171,   # 铰链 Y 坐标（相对 CG，上为正）[m]
    "z_front": 0.036515,       # 前翅力作用点 Z 坐标（相对 CG，前为正）[m]
    "z_back": -0.048012,       # 后翅力作用点 Z 坐标（相对 CG，前为正）[m]
    "g": 9.81,                 # 重力加速度 [m/s²]，方向 -Y
    "rho": 1.225,              # 空气密度 [kg/m³]
}
```

### v6.9 设计参数（新四连杆 + 扫参气动点）

```python
# v6.9 DESIGN_v69 (L/W_world=4.43, peak θ=65.8°, 摆幅 ~99°, K≈1.0)
# 注意：坐标系重构后，旧设计点需在新坐标系下重新验证
DESIGN_v69 = {
    # 气动安装角
    "alpha_front_deg": 60.0,   # 前翅安装角 [°]
    "alpha_back_deg": 3.0,     # 后翅安装角 [°]
    "phase_diff_deg": -15.0,   # 前后翅相位差 [°]
    # 机构参数（新四连杆）
    "mech_a": 7.6,             # 翅膀转轴 A 的 y 坐标 [mm]
    "mech_b": 1.71,            # 圆心 B 的 x 坐标 [mm]
    "mech_R": 3.8,             # 曲柄半径 [mm]
    "mech_c": 7.1,             # 连杆 P1-P2 长度 [mm]
    "mech_l": 5.0,             # 摇杆/翅膀杆 A-P2 长度 [mm]
    "phi_offset_deg": 0.0,     # 翅膀安装基准偏角 [°]
    "rotation": "cw",          # 曲柄旋转方向
    # 气动
    "f": 17.0,                 # 扑动频率 [Hz]
    "k_clap": 0.3,             # Clap-and-Fling k_max
    "c_damp": 5e-4,            # 俯仰阻尼 [N·m·s/rad]
    "k_3d": 0.7,
    "C_rot": 1.5,
    "r_rot": 0.5,
}
```

<details>
<summary>历史: v6.8 设计参数 (已归档)</summary>

以下 v6.8 参数基于旧机构（a=6.0, R=2.5, φ_offset=-30），已被新机构（a=7.6, R=3.8, φ_offset=0）取代。v6.8 物理约束（|α_eff| ≤ 50°）在新机构下放宽，设计点 α_f=60° 是合理的。

### 设计参数 (v6.8) ⭐ FINAL — 物理合理性约束

v6.8 扫参基于**速度耦合 clap-fling** (Lighthill 公式启发)。
粗网格 α_f=[30,40,60,70] (1,296 组) + α_f=45-50 定点测试。

**物理约束**: 有效攻角均值应 ≤ 50°（Dickinson 经验公式/过渡区），α_eff > 70°（平板失速）占比 < 5%。

| α_f | 均值 \|α_eff\| | >70°占比 | L/W | 判定 |
|-----|----------------|---------|-----|------|
| 40° | 40.9° | 1.5% | 2.13 | ✅ Dickinson范围 |
| 45° | 45.9° | 2.3% | 2.45 | ✅ 安全 |
| **50°** | **50.8°** | **3.1%** | **2.81** | ✅ **过渡区上界** |
| 60° | 60.5° | 6.8% | 3.61 | ❌ 过度依赖失速模型 |
| 70° | 70.5° | 31.1% | 3.89 | ❌ 非物理 |

```python
DESIGN_v68 = {
    "alpha_front_deg": 45, "alpha_back_deg": 8, "phase_diff_deg": -20,
    "mech_a": 6.0, "mech_R": 2.50, "phi_offset_deg": -30,
    "f": 17, "c_damp": 5e-4, "rotation": "cw", "k_clap": 0.3,
}
# L/W=2.447, peak_θ=32.9°, mean_α_eff=45.9°, >70°%=2.3%, n_exceed_90=0
```

</details>

<details>
<summary>v6.8 扫参历史发现 (已归档)</summary>

#### v6.8 扫参关键发现 (终版 — 物理合理性约束)

##### 物理合理性判定标准

| 标准 | 阈值 | 含义 |
|------|------|------|
| 均值 \|α_eff\| | ≤ 40° | Dickinson 经验公式直接适用范围 |
| 均值 \|α_eff\| | 40-50° | 过渡区，LEV/Lee 混合模型可用 |
| 均值 \|α_eff\| | > 50° | 过度依赖平板失速模型，非物理 |
| α_eff > 70° 时间占比 | < 5% | 可接受；> 5% 说明模型被推至边界 |

⚠️ **α_f ≥ 60° 时均值 α_eff ≥ 60°，翅膀整个拍动周期在深度失速区，L/W 虚高来自平板失速模型的非物理行为，不可采纳。**

##### 扫描范围

| 阶段 | α_f 范围 | 组数 | 状态 |
|------|----------|------|------|
| 细网格 | 50, 55 | 10,326 | ✅ (已迁移移动硬盘) |
| 粗网格补扫 | 30, 40, 60, 70 | 1,296 | ✅ |
| **合计** | **30-70** | **11,622** | |

##### α_f 扩展扫描: 核心发现

```
α_f   稳定率   平均L/W   最大L/W   θ_range    最优 α_b
30°   49.4%    0.651     1.594     62-86°      3
40°   48.8%    0.944     2.128     25-87°      8
50°   67.3%    1.253     3.039     21-89°      3/8/10  (细网格)
55°   —        1.389     3.081     29-90°      3      (细网格, 仅 α_b=3)
60°   82.7%    1.778     3.607     30-90°     15
70°   90.1%    1.898     3.892     20-90°     15
```

**α_f 越大越好** — L/W 单调递增，且 α_f≥60 后稳定率大幅上升（83→90%）。
**α_b 最优反转** — 低 α_f(30-55) 最优 α_b=3°，高 α_f(60-70) 最优 α_b=15°。物理原因：高 α_f 产生巨大前翅升力，需要高 α_b 维持俯仰力矩平衡。

##### L/W 峰值演进

```
α_f=55, α_b=3:  L/W=3.081  (旧上限 — 细网格)
α_f=60, α_b=15: L/W=3.607  (+17%)
α_f=70, α_b=15: L/W=3.892  (+26% vs 旧上限)
```

##### 机构参数 (不变)

- **a=6** 绝对最优，a=8 的 L/W 峰值不到 a=6 的 1/3
- **R=2.50 >> 2.25 >> 2.00**: R 每降一档 L/W 降 ~30%
- **φ_off=-25°** 一致最优
- **f=17Hz** 一致

##### k_clap (不变)

- 0.3 和 0.5 几乎并列（L/W 峰值差 <0.01），0.3 稳定率略高 (71% vs 65%)
- ≥0.8 显著恶化
- 推荐 0.5（L/W 略高）或 0.3（稳定性略高）

##### 设计参数对比

| 指标 | **v6.8 (主推)** | v6.8 (保守) | v6.8 (激进) |
|------|----------------|------------|------------|
| α_f | **45°** | 40° | 50° |
| α_b | **8°** | 8° | 8° |
| ph | **-20°** | -10° | -20° |
| φ_off | **-30°** | -25° | -25° |
| R | **2.50mm** | 2.50mm | 2.50mm |
| k_clap | **0.3** | 0.3 | 0.3 |
| L/W | **2.45** | 2.13 | 2.81 |
| peak_θ | **32.9°** | 25.4° | 28.4° |
| mean α_eff | **45.9°** | 40.9° | 50.8° |

##### 快速分析脚本

```bash
python temp/analyze_sweep_v68.py   # 完整六步分析，输出 Top 30 和推荐参数
python temp/plot_sweep_v68.py      # 7 张综合图表 → output/figures/
```

```python
PHYS = {"rho":1.225, "g":9.81, "m_total":0.020, "I_yy":3e-5,
        "x_front":0.025, "x_back":-0.025, "d_cg":0.015, "c_damp":5e-4}

MECH = {"a":6.0, "b":6.97, "R":2.25, "c":14.00, "l":8.00,
        "phi_offset_deg":-30, "f":17.0, "rotation":"cw"}

WING_FRONT = {"S":0.01617, "R":0.1543, "c_avg":0.1048, "r1":0.4227, "r2_sq":0.2382}
WING_BACK  = {"S":0.01554, "R":0.1474, "c_avg":0.1054, "r1":0.4798, "r2_sq":0.2876}

W = 196.2 mN  # 重量 (0.020 × 9.81 × 1000)
```

## Markdown → Word 报告生成工具

### 转换脚本: `tools/convert_to_docx.py`

将 `report/` 目录下的多章节 Markdown 报告合并并转换为规范格式的 Word 文档。

**特性**:

- 自动合并章节（摘要 → 绪论 → ... → 结论 → 心得体会）
- Pandoc 转换 LaTeX 公式 → Word OMML
- 自动格式处理:
  - A4 页面设置 + 页眉页脚
  - 三级标题格式（黑体+字号分级）
  - 正文（宋体小四号，首行缩进，固定行间距）
  - 表格三线表 + 居中
  - 图片自动缩放（最大宽度 14cm）
  - 公式字体 Cambria Math
- 图片路径自动修正（相对路径 → 绝对路径）
- 直引号自动转为中文弯引号

### 辅助脚本: `tools/fix_docx_math_fonts.py`

独立修复已生成 docx 中的公式字体，无需重新转换。

### 使用方法

```bash
# 完整转换（推荐）
python tools/convert_to_docx.py

# 仅修复公式字体
python tools/fix_docx_math_fonts.py output/仿生蝴蝶扑翼MAV机械原理分析报告.docx output/output_fixed.docx
```

**输出**: `output/仿生蝴蝶扑翼MAV机械原理分析报告.docx`

## 稳定性分析工具

### 单变量偏离扫描工具

| 工具 | 说明 |
|------|------|
| `tools/analysis/run_all_sweeps.py` | 批量运行所有单变量偏离扫描（其他参数锁死 DESIGN_v68），包含：α_f、α_b、phase_diff、mech_a、mech_R、φ_offset、k_clap |
| `src/aero/stability_analysis.py` | 单变量偏离扫描主程序（基线 + 指定参数扫描） |
| `src/aero/stability_plot.py` | 稳定性结果绘图（基线图 + 偏离图） |

### 笛卡尔积扫参分析工具

| 工具 | 说明 |
|------|------|
| `src/aero/sweep_cartesian.py` | 全参数笛卡尔积扫描（joblib并行），11,809组 |
| `tools/analysis/rebuild_sweep_summary.py` | 重建扫参摘要 JSON，从目录结构解析参数组合 |
| `tools/analysis/extract_sweep_slices.py` | 从扫参数据中提取特定参数切片，生成对比数据 |
| `tools/analysis/analyze_sweep_v68.py` | **完整六步分析**：k_clap敏感性、α_f×α_b交互、phase影响、Top 30排名、推荐设计参数 |

### 使用示例

```bash
# 单变量偏离扫描（7参数，约7分钟）
python tools/analysis/run_all_sweeps.py

# 生成稳定性图表
python src/aero/stability_plot.py --baseline
python src/aero/stability_plot.py --sweep alpha_front_deg
python src/aero/stability_plot.py --all

# 全扫参分析（输出 Top 30 和推荐参数）
python tools/analysis/analyze_sweep_v68.py
```

</details>

## Run Commands

```bash
# v6.3 力输出模块（对外接口）
python -m src.aero.butterfly_forces      # 自带验证测试

# 调用示例
python -c "from src.aero.butterfly_forces import *; cfg=SimulationConfig(); m=ButterflyForceModel(cfg); out=m.simulate(); print(out.summary)"

# v6.3 全参数扫描 (99组)
python temp/scan_v6_3.py

# v6.3 10s 长稳验证
python temp/verify_v63_long.py

# 机构运动学自测
python -m src.struct.mechanism
```

## Critical Warnings

1. **C_L/C_D 使用 v6.3 LEV/Lee 混合模型** (|α|≤55°: Dickinson, ≥65°: LEV/Lee)。不再使用平板模型
2. **必须检查每步 |θ_p| < 90°**，不能只看稳态平均
3. **气动俯仰阻尼 (v6.1 新增) 是稳定性关键**，不可移除
4. **对称安装角 (α_f=α_b) 净升力全为负** — 不对称是必要的
5. **坐标系已对齐 SolidWorks**：体轴 X=右、Y=上、Z=前；升力/重量方向沿 Y 轴；L/W 用体轴 Fy 或世界系 Fy 计算
6. **俯仰轴为 X 轴且过 CG**，重力不产生俯仰力矩
7. **默认 dt=10μs** 用于精细力输出。扫描时可用 50μs 提速
