# Butterfly Aerodynamic Analysis — Agent Notes

## Project Overview
仿生蝴蝶扑翼微型飞行器（MAV）俯仰动力学仿真。质量 20g，四连杆机构驱动，15Hz 扑动频率。验证俯仰稳定性、升力/推力平衡、以及非对称安装角的优化。

## Current State (2026-06-05)

**当前主力模型：v6.3** — 刚性翅膀 + 非对称 α_install + 气动俯仰阻尼 + LEV/Lee C_L/C_D 理论公式

### 版本演进

| 版本 | 文件 | 核心特点 | 关键结果 |
|------|------|---------|---------|
| v6 | `temp/pitch_dynamics_v6.py` | 刚性 wing, 非对称 α_install 扫描 | L/W=1.48 但俯仰全部发散 |
| v6.1 | `temp/pitch_dynamics_v6_1.py` | +气动俯仰阻尼 Δα(θ̇_p) | L/W=1.145, 10s 全稳定, 但全在平板区 |
| v6.2 | `temp/scan_low_alpha.py` | +低 α_install (28-45°) | L/W=0.615, LEV 范围, 64 组全稳定 |
| **v6.3** | `src/butterfly_forces.py` | +LEV/Lee C_L/C_D (C_D_max=3.22) | **L/W=1.033**, 10s STABLE, 达悬停条件 ✅ |

### 最佳参数 (v6.3)

| α_f | α_b | L/W (3s) | L/W (10s) | Peak θ | n90 | Fz_body | 状态 |
|-----|-----|----------|-----------|--------|-----|---------|------|
| **60°** | **8°** | 0.943 | **1.033** | 46.7° | 0 | +203 mN | ✅ 悬停达成 |
| 60° | 10° | 0.903 | 0.993 | 45.0° | 0 | +195 mN | ✅ |
| 60° | 12° | 0.859 | 0.948 | 43.1° | 0 | +186 mN | ✅ |

详细记录见 `docs/v6_3_CL_CD_formula.md`，完整扫描数据见 `data/v63_scan_results.json`

### 对外力输出模块: `src/butterfly_forces.py`

对外调用接口，提供完整仿真管线（运动学→俯仰ODE→力输出），**全参数可配置**：

```python
from butterfly_forces import SimulationConfig, ButterflyForceModel, scan_parameters

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
│   ├── mechanism.py              # 曲柄摇杆四连杆 → φ(t), φ̇(t), φ̈(t)
│   ├── butterfly_forces.py       # ★ 对外力输出模块 (v6.3+)
│   ├── dynamic_analysis.py       # 气动力仿真 (v3)
│   ├── analyze_dxf.py            # DXF 几何提取 → 翅膀面积/展长/面积矩
│   ├── alpha_scan.py             # 安装角扫描
│   └── gear_analysis.py          # 齿轮减速比分析
├── data/
│   ├── WingFront.DXF / WingBack.DXF / WingsAxis.DXF
│   ├── wing_analysis_results.json
│   └── v63_scan_results.json     # v6.3 99组扫描完整数据
├── temp/
│   ├── pitch_dynamics_v6_1.py    # v6.1 物理引擎 (v6.2/v6.3 共用)
│   ├── scan_v6_3.py              # v6.3 99组扫描脚本
│   ├── verify_v63_long.py        # v6.3 10s长稳验证
│   ├── scan_low_alpha.py         # v6.2 低 α LEV 范围扫描
│   ├── verify_long_stability.py  # v6.1 10s长稳验证
│   ├── diagnose_pitch.py         # 俯仰发散根因诊断
│   ├── diagnose_upstroke.py      # 上拍攻角 & 公式使用分析
│   ├── plot_pitch_rate.py        # pitch rate 时程图
│   ├── debug_*.py                # 调试脚本
│   ├── v63_long/                 # v6.3 10s 验证图 (6+3 张)
│   ├── v6_fixed_scan/            # v6 固定点扫描结果图 (60 张)
│   ├── v6_moving_scan/           # v6 移动模型扫描结果图 (72 张)
│   └── v61_long_stability/       # v6.1 10s 验证图 (4 张)
├── docs/
│   ├── v6_3_CL_CD_formula.md     # v6.3 完整实验报告
│   ├── v6_scan_report.md         # v6 初始扫描详细报告
│   ├── v6_experiment_conclusions.md  # v6-v6.2 实验总结
│   └── v6_summary_and_roadmap.md # 路线图
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

`src/stability_analysis.py` + `src/stability_plot.py` — 插拔式稳定性分析管线:

```
analysis (stability_analysis.py) ──[JSON+NPZ]──> plot (stability_plot.py)
         │                                              │
         └── temp/stability/ ───────────────────────────┘
              ├── baseline/{config,summary,timeseries}
              └── sweep_<param>/<value>/{config,summary,timeseries}
```

- **分析模块**: 单变量偏离扫描, 每完成一个值立即保存 checkpoint, 支持断点续跑
- **绘图模块**: 完全独立, 只读文件。基线 5 张图 + 每参数 6 张偏离图
- 基线参数: α_f=60/α_b=8, phase=0, a=7.92, R=2.25, phi_offset=-50.84 (L/W≈1)

```bash
# 基线分析
python src/stability_analysis.py --baseline

# 单变量偏离 (e.g. mech_a)
python src/stability_analysis.py --sweep mech_a

# 出图
python src/stability_plot.py --baseline
python src/stability_plot.py --sweep mech_a
python src/stability_plot.py --all
```

### TODO: 全参数笛卡尔积扫描

- [ ] 6 参数粗网格全扫描 (α_f × α_b × phase × a × R × phi_offset)
- [ ] 在最优区域附近精化扫描
- [ ] 用扫描结果更新默认参数和推荐配置
- [ ] 写入 `stability_analysis.py` 的 `sweep_all()` 方法

## Key Parameters

```python
# v6.4 基线 (L/W≈1.03, 5s)
BASE = {"alpha_front_deg":60, "alpha_back_deg":8, "phase_diff_deg":0,
        "mech_a":7.92, "mech_R":2.25, "phi_offset_deg":-50.84}

# v6.4 最优趋势
# α_f=68, α_b=5, phase=-15~-30, a=5.0, R=2.5+ → L/W=10.44
```

```python
PHYS = {"rho":1.225, "g":9.81, "m_total":0.020, "I_yy":3e-5,
        "x_front":0.025, "x_back":-0.025, "d_cg":0.015, "c_damp":5e-4}

MECH = {"a":7.92, "b":6.97, "R":2.25, "c":14.00, "l":8.00,
        "phi_offset_deg":-50.84, "f":15.0, "rotation":"cw"}
# 偏移后 φ ∈ [-22.2°, +22.2°], 急回: 下拍 57%/上拍 43%

WING_FRONT = {"S":0.01617, "R":0.1543, "c_avg":0.1048, "r1":0.4227, "r2_sq":0.2382}
WING_BACK  = {"S":0.01554, "R":0.1474, "c_avg":0.1054, "r1":0.4798, "r2_sq":0.2876}

W = 196.2 mN  # 重量 (0.020 × 9.81 × 1000)
```

## Run Commands

```bash
# v6.3 力输出模块（对外接口）
python src/butterfly_forces.py           # 自带验证测试

# 调用示例
python -c "from butterfly_forces import *; cfg=SimulationConfig(); m=ButterflyForceModel(cfg); out=m.simulate(); print(out.summary)"

# v6.3 全参数扫描 (99组)
python temp/scan_v6_3.py

# v6.3 10s 长稳验证
python temp/verify_v63_long.py

# 机构运动学自测
python src/mechanism.py
```

## Critical Warnings

1. **C_L/C_D 使用 v6.3 LEV/Lee 混合模型** (|α|≤55°: Dickinson, ≥65°: LEV/Lee)。不再使用平板模型
2. **必须检查每步 |θ_p| < 90°**，不能只看稳态平均
3. **气动俯仰阻尼 (v6.1 新增) 是稳定性关键**，不可移除
4. **对称安装角 (α_f=α_b) 净升力全为负** — 不对称是必要的
5. **体轴系 vs 世界系**：L/W 用体轴系 Fz 计算（与现有 scan 一致）。世界系 Fz 约体轴系的 50%（因 pitch 振荡导致投影损耗）。物理悬停需要世界系 Fz ≥ 重量
6. **默认 dt=10μs** 用于精细力输出。扫描时可用 50μs 提速
