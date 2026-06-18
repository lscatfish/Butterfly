# Butterfly Aerodynamic Analysis — Agent Notes

## Project Overview
仿生蝴蝶扑翼微型飞行器（MAV）俯仰动力学仿真。质量 20g，四连杆机构驱动，15Hz 扑动频率。验证俯仰稳定性、升力/推力平衡、以及非对称安装角的优化。

## Current State (2026-06-05)

**当前主力模型：v6.7** — 文献[32]标准攻角定义 + tanh平滑过渡 + numba JIT

### 版本演进

| 版本 | 文件 | 核心特点 | 关键结果 |
|------|------|---------|---------|
| v6 | `temp/pitch_dynamics_v6.py` | 刚性 wing, 非对称 α_install 扫描 | L/W=1.48 但俯仰全部发散 |
| v6.1 | `temp/pitch_dynamics_v6_1.py` | +气动俯仰阻尼 Δα(θ̇_p) | L/W=1.145, 10s 全稳定, 但全在平板区 |
| v6.2 | `temp/scan_low_alpha.py` | +低 α_install (28-45°) | L/W=0.615, LEV 范围, 64 组全稳定 |
| v6.3 | `src/aero/butterfly_forces.py` | +LEV/Lee C_L/C_D (C_D_max=3.22) | **L/W=1.033**, 10s STABLE, 达悬停条件 ✅ |
| **v6.7** | `src/aero/butterfly_forces.py` | **修正攻角定义**[32]: α=η (相对拍动平面) + tanh平滑过渡 | **L/W=2.15**, peak θ=37.6°, 物理自洽 ✅ |

### 最佳参数 (v6.3)

| α_f | α_b | L/W (3s) | L/W (10s) | Peak θ | n90 | Fz_body | 状态 |
|-----|-----|----------|-----------|--------|-----|---------|------|
| **60°** | **8°** | 0.943 | **1.033** | 46.7° | 0 | +203 mN | ✅ 悬停达成 |
| 60° | 10° | 0.903 | 0.993 | 45.0° | 0 | +195 mN | ✅ |
| 60° | 12° | 0.859 | 0.948 | 43.1° | 0 | +186 mN | ✅ |

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

### 角度体系
- **α_install**: 翅膀弦线相对机身水平面的固定安装角（出厂设定，飞行中不变）
- **φ**: 机构拍动角，范围 [-22.2°, +22.2°]
- **θ_p**: 机身俯仰角（动态变量）
- **ψ = φ + θ_p**: 翅膀在惯性系中的总角度
- **α_eff**: 有效气动攻角 = ±(α_install + ψ + Δα_damp)

### 核心 bug（已修复）
- **双重投影** (v4): 不要参考
- **常量攻角假设** (v5): α_down/α_up 隐含主动扭转，已废弃
- **俯仰发散** (v6): M_aero 时均不为零 → v6.1 引入气动阻尼修复
- **高 α_install 使经验公式失效** (v6.1) → v6.2 降低 α_install

### 四分量力模型
- F_trans: 平动升力/阻力（基于瞬时速度 + 混合 C_L/C_D 模型）
- F_AM: 附加质量力 ∝ φ̈
- F_clap: Clap-and-Fling（反转点增强 1.3x）
- F_rot: 旋转力（当前 α̇=0，自然为 0）

### C_L/C_D 混合模型
```
|α| ≤ 40°: 经验公式 (Sane & Dickinson 2002, 含 LEV 效应)
40°<|α|≤70°: smoothstep 过渡
|α| > 70°: 平板失速模型
```

### 力投影（v3 验证）
```
Fx = sin(ψ) × (sign×D − L)
Fz = cos(ψ) × (L − sign×D)
单次投影，无双重投影 bug
```

### 气动俯仰阻尼 (v6.1+)
```
Δα ≈ atan(θ̇_p × x_wing / U)
→ θ̇_p > 0: 前翅 α↑, 后翅 α↓ → 恢复力矩
```

## File Structure

```
├── src/
│   ├── aero/
│   │   ├── butterfly_forces.py       # ★ 对外力输出模块 (v6.3+, v6.7攻角修正)
│   │   ├── stability_analysis.py     # 方案一: 单变量偏离扫描
│   │   ├── stability_plot.py         # 方案一: 稳定性绘图
│   │   ├── sweep_cartesian.py        # 方案二: 笛卡尔积全扫描 (joblib并行)
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
│   ├── butterfly_forces_使用说明.md       # 模块 API 文档 (v6.7)
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

## v6.7 攻角公式修正 (2026-06-08)

### Bug 根因

v6.3-v6.6 的攻角定义有误：
```
α_geom = α_install + φ + θ_p    ← 错误！φ 是拍动位置，不应进入攻角
```

文献[32]标准定义：攻角 = 翅膀弦线相对**拍动平面**（体轴 XZ 平面）的角度。拍动平面随 body 整体倾斜，θ_p 影响的是力投影方向而非攻角本身。

### 修正

```
α_eff = -tanh(φ̇) · (α_install + δα)
```

- **η = α_install**：翅膀相对拍动平面的安装角（不含 φ、不含 θ_p）
- **-tanh(φ̇)**：平滑符号翻转（上下拍来流方向相反）
- **δα = atan2(θ̇_p·x_wing, |Ω|·R)**：俯仰气动阻尼
- **sign_Ω = tanh(Ω)**：阻力体轴投影平滑过渡

### 设计参数 (v6.7)

```python
DESIGN_v67 = {
    "alpha_front_deg": 60,    # 前翅安装角 (Dickinson CL峰值区)
    "alpha_back_deg": 10,     # 后翅安装角
    "phase_diff_deg": -20,    # 前后翅相位差
    "mech_a": 6,              # 曲柄半径 [mm]
    "mech_R": 2.25,           # 摇杆半径 [mm] (机构默认)
    "phi_offset_deg": -30,    # 翅膀安装偏角
    "f": 17,                  # 拍动频率 [Hz]
    "c_damp": 5e-4,           # 俯仰阻尼 [N·m·s/rad]
    "rotation": "cw",         # 曲柄转向 (ccw不可行)
}
# L/W_world=2.15, Fz_world=+422mN, peak θ=37.6°, pitch=15~27°, 5s稳定
```

### 修正效果对比

| 指标 | v6.6 (Bug) | v6.7 (修正) |
|------|-----------|------------|
| α_eff 范围 | [-170°, 240°] 需钳制 | **[-72°, 155°]** 自然范围 |
| Peak θ_p | 87° | **37.6°** |
| Pitch 稳态 | 64~76° | **15~27°** |
| L/W (world) | 2.89 (bug) | **2.15** |
| 攻角公式 | α_install+φ+θ_p | **α_install** (文献[32]) |
| 符号翻转 | if/else 硬切换 | **tanh 平滑** |

### 废弃数据

v6.6 的 82GB 扫参数据（`temp/stability/sweep_cartesian/`）基于 bug 公式生成，已废弃。

---

## Stability Analysis System

`src/aero/stability_analysis.py` + `src/aero/stability_plot.py` — 插拔式稳定性分析管线:

```
analysis (stability_analysis.py) ──[JSON+NPZ]──> plot (stability_plot.py)
         │                                              │
         └── temp/stability/ ───────────────────────────┘
              ├── baseline/{config,summary,timeseries}
              └── sweep_<param>/<value>/{config,summary,timeseries}
```

- **分析模块**: 单变量偏离扫描, 每完成一个值立即保存 checkpoint, 支持断点续跑
- **绘图模块**: 完全独立, 只读文件。基线 5 张图 + 每参数 6 张偏离图
- **v6.5 基线**: α_f=68/α_b=5, phase=-15, a=6.0, R=2.5, φ_offset=-50.84 (L/W_world=2.505)
- **方案一扫描**: 6参数共48组，全部完成 (41.4 min)。详细报告见 `docs/v6_5_stability_sweep_report.md`

### v6.6 方案二: 9参数全笛卡尔积扫描

`src/aero/sweep_cartesian.py` — joblib 并行 + numba JIT 全组合扫描:

```
sweep_cartesian.py
  ├─ build_cartesian_grid(grid_spec) → [{param: value}, ...]
  ├─ combo_to_id(combo) → "af60_ab5_phn20_a6_R3_..."
  ├─ run_one_combo(combo) → 保存 {config,summary,timeseries}
  └─ sweep_cartesian(grid_spec, n_jobs=-1) → sweep_summary.json
```

- **3456 组** 9 参数粗网格全扫描，56.7 min (16 核, numba 4.8×) ⚠️ 基于 v6.6 bug 公式, 已废弃
- **v6.8 扫参**: **10,326/45,000 组** 10 参数 (新增 k_clap [0.3,0.5,0.8,1.0,1.5]), 分析完成, DESIGN_v68 已确定
- 输出与方案一完全兼容 (config/summary/timeseries)，绘图模块可复用
- 详细报告见 `docs/v6_6_cartesian_sweep_report.md`

### v6.5 方案一关键发现

| 参数 | 敏感度 | 趋势 | 最优(L/W) |
|------|--------|------|-----------|
| **mech_a** | ⭐⭐⭐⭐⭐ | 单调↓ | 5.0mm → 8.246 |
| **mech_R** | ⭐⭐⭐⭐ | 单调↑ | 3.0mm → 6.734 (3.5mm发散) |
| **phi_offset** | ⭐⭐⭐ | 单调→0 | -30° → 3.728 |
| **phase_diff** | ⭐⭐⭐ | 尖锐最优 | -15° → 2.505 |
| α_back | ⭐ | 平坦 | 3-8° ~2.50 |
| α_front | ⭐ | 极平坦 | 55-70° ~2.47-2.51 |

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

- [x] **numba JIT 加速** ✅ — 标量 @njit 编译 RK4 热循环, 4.8× 单仿真加速 (v6.6)
- [x] **方案二: 全参数笛卡尔积扫描** ✅ — 9参数 3456 组, 16核 joblib, 56.7 min (v6.6)
- [x] **攻角公式修正** ✅ — 文献[32]标准: α=η (相对拍动平面), tanh平滑 (v6.7)
- [x] 用扫描结果更新默认参数和推荐配置 ✅ — DESIGN_v67 (v6.7)
- [ ] **清理 v6.6 废弃数据** — `temp/stability/sweep_cartesian/` 82GB 基于 bug 公式, 可全删
- [x] **v6.8 Clap-and-Fling 物理模型修正** ✅ — 速度-位置耦合, Lighthill公式启发
- [x] **v6.8 k_clap 语义变更** ✅ — 默认 0.5 (k_max), 扫参网格 45,000 组
- [x] **v6.8 扫参分析** ✅ — 11,622 组 + α_f=45-50 定点, 物理最优 α_f=50°/α_b=8°, L/W=2.81
- [x] **v6.8 物理合理性校准** ✅ — 约束 α_eff≤50°, 排除 α_f≥60° 非物理结果 (均值 α_eff>60°)
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
- [ ] 新 DESIGN 参数 L/W ≥ 当前 DESIGN_v67 的 2.15

## Key Parameters

```python
# v6.8 设计参数 (L/W_world=2.45, Fz_world=+469mN, peak θ=32.9°, 5s稳定)
# 攻角定义修正为文献[32]标准: α=η (翅膀相对拍动平面)
# k_clap=0.3 为 v6.8 速度耦合 clap-fling k_max (已扫参标定)
DESIGN_v68 = {"alpha_front_deg":45, "alpha_back_deg":8, "phase_diff_deg":-20,
              "mech_a":6.0, "mech_R":2.50, "phi_offset_deg":-30,
              "f":17.0, "c_damp":5e-4, "rotation":"cw", "k_clap":0.3}
```

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
    # 气动安装角
    "alpha_front_deg": 45,    # 前翅安装角 — 均值 α_eff=45.9°, 过渡区中部
    "alpha_back_deg": 8,      # 后翅安装角
    # 翅膀相位
    "phase_diff_deg": -20,    # 前后翅相位差
    # 机构参数
    "mech_a": 6.0,            # 曲柄半径 [mm]
    "mech_R": 2.50,           # 摇杆半径 [mm]
    "phi_offset_deg": -30,    # 翅膀安装偏角
    # 拍动
    "f": 17,                  # 拍动频率 [Hz]
    # 阻尼
    "c_damp": 5e-4,           # 俯仰阻尼
    "rotation": "cw",         # 曲柄转向
    # Clap-fling
    "k_clap": 0.3,            # k_max (速度耦合系数)
}
# L/W=2.447, peak_θ=32.9°, mean_α_eff=45.9°, >70°%=2.3%, n_exceed_90=0

# 备选 — 更保守 (均值 α_eff=41°, 完全在 Dickinson 范围内):
DESIGN_v68_safe = {
    "alpha_front_deg": 40, "alpha_back_deg": 8, "phase_diff_deg": -10,
    "mech_a": 6.0, "mech_R": 2.50, "phi_offset_deg": -25,
    "f": 17, "c_damp": 5e-4, "rotation": "cw", "k_clap": 0.3,
}
# L/W=2.128, peak_θ=25.4°, mean_α_eff=40.9°, >70°%=1.5%

# 激进方案 (均值 α_eff=51°, 过渡区上界):
DESIGN_v68_aggressive = {
    "alpha_front_deg": 50, "alpha_back_deg": 8, "phase_diff_deg": -20,
    "mech_a": 6.0, "mech_R": 2.50, "phi_offset_deg": -25,
    "f": 17, "c_damp": 5e-4, "rotation": "cw", "k_clap": 0.3,
}
# L/W=2.807, peak_θ=28.4°, mean_α_eff=50.8°, >70°%=3.1%
```

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

##### 与 DESIGN_v67 对比
| 指标 | v6.7 | **v6.8 (主推)** | v6.8 (保守) | v6.8 (激进) |
|------|------|----------------|------------|------------|
| α_f | 60° | **45°** | 40° | 50° |
| α_b | 10° | **8°** | 8° | 8° |
| ph | -20° | **-20°** | -10° | -20° |
| φ_off | -30° | **-30°** | -25° | -25° |
| R | 2.25mm | **2.50mm** | 2.50mm | 2.50mm |
| k_clap | 0.5 | **0.3** | 0.3 | 0.3 |
| L/W | 2.15 | **2.45** | 2.13 | 2.81 |
| peak_θ | 37.6° | **32.9°** | 25.4° | 28.4° |
| mean α_eff | — | **45.9°** | 40.9° | 50.8° |

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
5. **体轴系 vs 世界系**：L/W 用体轴系 Fz 计算（与现有 scan 一致）。世界系 Fz 约体轴系的 50%（因 pitch 振荡导致投影损耗）。物理悬停需要世界系 Fz ≥ 重量
6. **默认 dt=10μs** 用于精细力输出。扫描时可用 50μs 提速
