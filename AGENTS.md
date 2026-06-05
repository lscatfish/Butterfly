# Butterfly Aerodynamic Analysis — Agent Notes

## Project Overview
仿生蝴蝶扑翼微型飞行器（MAV）俯仰动力学仿真。质量 20g，四连杆机构驱动，15Hz 扑动频率。验证俯仰稳定性、升力/推力平衡、以及非对称安装角的优化。

## Current State (2026-06-05)

**当前主力模型：v6.2** — 刚性翅膀 + 非对称 α_install + 气动俯仰阻尼 + LEV 经验公式范围

### 版本演进

| 版本 | 文件 | 核心特点 | 关键结果 |
|------|------|---------|---------|
| v6 | `temp/pitch_dynamics_v6.py` | 刚性 wing, 非对称 α_install 扫描 | L/W=1.48 但俯仰全部发散 |
| v6.1 | `temp/pitch_dynamics_v6_1.py` | +气动俯仰阻尼 Δα(θ̇_p) | L/W=1.145, 10s 全稳定, 但全在平板区 |
| **v6.2** | `temp/scan_low_alpha.py` | +低 α_install (28-45°) | L/W=0.615, LEV 范围, 64 组全稳定 |

### 最佳参数

| 场景 | α_f | α_b | L/W | Peak θ | 公式范围 |
|------|-----|-----|-----|--------|---------|
| 最大升力 (v6.1) | 60° | 10° | 1.145 | 46° | 全平板 ❌ |
| LEV 物理 (v6.2) | 45° | 10° | 0.615 | 35° | 经验公式 ✅ |

详细记录见 `docs/v6_experiment_conclusions.md`

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
│   ├── mechanism.py          # 曲柄摇杆四连杆 → φ(t), φ̇(t), φ̈(t)
│   └── analyze_dxf.py        # DXF 几何提取 → 翅膀面积/展长/面积矩
├── data/
│   ├── WingFront.DXF / WingBack.DXF / WingsAxis.DXF
│   └── wing_analysis_results.json
├── temp/
│   ├── pitch_dynamics_v6.py       # v6 刚性翅膀 + 不对称扫描
│   ├── pitch_dynamics_v6_1.py     # v6.1 +气动俯仰阻尼（导入用）
│   ├── scan_low_alpha.py          # v6.2 低 α_install LEV 范围扫描
│   ├── verify_long_stability.py   # 10s 长时稳定性验证
│   ├── diagnose_pitch.py          # 俯仰发散根因诊断
│   ├── diagnose_upstroke.py       # 上拍攻角 & 公式使用分析
│   ├── debug_alpha_sweep.py       # 攻角诊断
│   ├── debug_forces.py            # 力分量诊断
│   ├── debug_v4_forces.py         # v4 bug 诊断（历史参考）
│   ├── v6_fixed_scan/             # v6 固定点扫描结果图 (60 张)
│   ├── v6_moving_scan/            # v6 移动模型扫描结果图 (72 张)
│   └── v61_long_stability/        # v6.1 10s 验证图 (4 张)
├── docs/
│   ├── v6_scan_report.md          # v6 初始扫描详细报告
│   └── v6_experiment_conclusions.md # 完整实验总结
└── AGENTS.md                      # 本文件
```

## Key Parameters

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
# v6.2 低安装角扫描
python temp/scan_low_alpha.py

# v6.1 长时稳定性验证（需先确保 temp/pitch_dynamics_v6_1.py 存在）
python temp/verify_long_stability.py

# 机构运动学自测
python src/mechanism.py

# 翅膀几何提取
python src/analyze_dxf.py
```

## Critical Warnings

1. **α_install 必须 ≤ 45°** 才能让中段 α_eff 在经验公式有效范围 (±60°) 内
2. **必须检查每步 |θ_p| < 90°**，不能只看稳态平均
3. **气动俯仰阻尼 (v6.1 新增) 是稳定性关键**，不可移除
4. **反相 180° 在当前机构下不适用** — 俯仰全部 >90°
5. **对称安装角 (α_f=α_b) 净升力全为负** — 不对称是必要的
