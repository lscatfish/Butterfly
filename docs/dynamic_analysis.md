# 仿生蝴蝶翅膀动态气动力分析（dynamic_analysis.py）

> 本文件说明 `src/dynamic_analysis.py` 的功能、气动模型、接口与输出。  
> 基于准定常模型（Dickinson/Sane 2002 分解），使用 `mechanism.py` 的实际运动学输出。

---

## 1. 功能概述

1. **单周期气动力仿真**：平动升力/阻力、旋转力（Kramer 效应）、附加质量力、Clap-and-Fling
2. **功率需求分析**：气动功率 + 惯性功率
3. **参数扫描**：频率 f、机构参数 a、攻角 α 对升/阻力的影响
4. **可视化**：力-转角曲线、时间域力曲线、加速度、功率、参数扫描图
5. **Markdown 报告生成**：`output/reports/气动分析报告.md`

---

## 2. 气动模型

### 2.1 力分解（Sane & Dickinson 2002）

| 力分量 | 公式 | 状态 |
|---|---|---|
| **平动升力** | $F_{L,tr} = \frac{1}{2}\rho C_L \dot{\phi}^2 R^2 S \hat{r}_2^2$ | ✅ 已实现 |
| **平动阻力** | $F_{D,tr} = \frac{1}{2}\rho C_D \dot{\phi}^2 R^2 S \hat{r}_2^2$ | ✅ 已实现 |
| **旋转力** | $F_{rot} = \rho C_{rot} \dot{\alpha} \dot{\phi} c^2 R \hat{r}_{rot}$ | ✅ 公式保留（当前 $\dot{\alpha}=0$，故为 0） |
| **附加质量力** | $F_{AM} = -\frac{\rho\pi c^2}{4} \ddot{\phi} R \hat{r}_1 \sin(\alpha)$ | ✅ 已实现 |
| **Clap-and-Fling** | 简化模型：reversal 附近升力增强 $k_{clap}$ 倍 | ✅ 已修正 |
| **三维展向效应** | 整体气动力乘以修正系数 $k_{3d}$ | ✅ 已修正 |

### 2.2 升阻力系数

```
C_L(α) = 0.255 + 1.58·sin(2.13·α - 7.2°)
C_D(α) = 1.92 - 1.55·cos(2.04·α - 9.82°)
```

**攻角方向判定**：
- φ̇ ≤ 0（下拍）→ 有效攻角 +α → C_L(+α)
- φ̇ > 0（上拍）→ 有效攻角 -α → C_L(-α)

---

## 3. 用户设计参数（AERO_PARAMS）

| 参数 | 数值 | 说明 |
|---|---|---|
| `rho` | 1.225 | 空气密度 [kg/m³] |
| `m_total` | 0.020 | 总质量 [kg] |
| `m_wing_total` | 0.004 | 四翅总质量 [kg] |
| `f` | 15.0 | 典型频率 [Hz] |
| `alpha_deg` | 45.0 | 固定攻角 [°]（单次周期内不变，可外部扫描）|
| `mech_a` | 7.92 | 机构参数 a [mm] |
| `phi_offset_deg` | -50.84 | 翅膀安装基准偏移 [°]（出厂固定）|
| `C_rot` | 1.5 | 旋转力系数（Kramer effect）|
| `r_rot` | 0.5 | 旋转力作用点半径系数 |
| `k_clap` | 1.3 | Clap-and-Fling 升力增强系数 |
| `k_3d` | 0.7 | 三维展向效应修正系数 |

---

## 4. 核心函数

### `simulate_cycle(geo_item, params, n_points=2000)`

模拟一个完整周期的气动力。

**返回 dict 关键字段**：

| 字段 | 说明 |
|---|---|
| `t` | 时间 [s] |
| `phi_deg` | 翅膀拍动角 [°] |
| `phi_dot` | 角速度 [rad/s] |
| `phi_ddot` | 角加速度 [rad/s²] |
| `F_trans_lift` | 平动升力 [N] |
| `F_trans_drag` | 平动阻力 [N] |
| `F_rot` | 旋转力 [N]（当前为 0）|
| `F_AM` | 附加质量力 [N] |
| `F_lift` | 总升力 [N]（含 Clap-and-Fling 和 3D 修正）|
| `F_drag` | 总阻力 [N]（含 3D 修正）|
| `P_aero` | 气动功率 [W] |
| `P_inertial` | 惯性功率 [W] |
| `P_total` | 总功率 [W] |

### `param_scan(geo_item, param_name, param_range, base_params)`

单参数扫描，返回时均/峰值升阻力。

### 绘图函数

| 函数 | 输出文件 |
|---|---|
| `plot_force_vs_phi` | `force_vs_phi.png` |
| `plot_time_domain` | `force_time_domain.png` |
| `plot_acceleration` | `wing_acceleration.png` |
| `plot_power_time_domain` | `power_time_domain.png` |
| `plot_param_scans` | `param_scan.png` |

---

## 5. 使用示例

```bash
python src/dynamic_analysis.py
```

输出：
- `output/figures/force_vs_phi.png`
- `output/figures/force_time_domain.png`
- `output/figures/wing_acceleration.png`
- `output/figures/power_time_domain.png`
- `output/figures/param_scan.png`
- `output/reports/气动分析报告.md`

---

## 6. 已知问题与修复历史

| 问题 | 状态 | 修复方式 |
|---|---|---|
| `split_stroke()` 用 `phi` 而非 `phi_dot` 分类 stroke | ✅ 已修复 | 改用角速度符号判定下拍/上拍 |
| `param_scan` 扫描无效的 `phi_down_deg` | ✅ 已修复 | 替换为有效的 `mech_a` 扫描 |
| `mech_info['raw_span_deg']` 键不存在 | ✅ 已修复 | 改为 `phi_span_deg` |
| 旋转力（Kramer 效应）未实现 | ✅ 已修复 | 公式保留，当前 α̇=0 时自然为 0 |
| Clap-and-Fling 完全忽略 | ✅ 已修复 | 简化模型：reversal 附近升力增强 1.3 倍 |
| 三维展向效应仅注释提及 | ✅ 已修复 | 整体气动力乘以 0.7 修正系数 |
| 固定攻角导致下拍/上拍升力抵消 | 🟡 设计约束 | 实际机械无 pitch reversal，只能通过 α 扫描寻找最优值 |

---

## 7. 文件位置

| 文件 | 路径 |
|---|---|
| 分析脚本 | `src/dynamic_analysis.py` |
| 依赖的运动学模块 | `src/mechanism.py` |
| 输入几何数据 | `data/wing_analysis_results.json` |
| 输出图表 | `output/figures/*.png` |
| 输出报告 | `output/reports/气动分析报告.md` |
