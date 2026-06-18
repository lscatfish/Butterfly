# v6.8 图表说明与模拟方法

本文件说明仿生蝴蝶扑翼飞行器 v6.8 仿真中生成的主要图表含义，以及对应的物理模型、数值方法和实验/计算流程。

---

## 1. 研究对象与几何

- **机体**: 20g 微型扑翼飞行器，四连杆机构驱动双翼。
- **翅膀**: 前翅 + 后翅，由 DXF 文件 (`data/WingFront.DXF`, `data/WingBack.DXF`) 提取面积、展长、面积矩。
- **机构**: 曲柄摇杆四连杆，曲柄半径 `a=6mm`，摇杆半径 `R=2.5mm`，拍动频率 `f=17Hz`。
- **坐标系**: 体轴系 (X 向前，Z 向上)，世界系随机身俯仰 `θ_p` 旋转。

---

## 2. 物理模型

### 2.1 运动学

- 曲柄匀速转动 → 摇杆拍动角 `φ(t)`，范围约 `[-22.2°, +22.2°]`。
- 机身俯仰角 `θ_p` 为动态变量，由俯仰 ODE 决定。
- 翅膀在惯性系中的总角度：`ψ = φ + θ_p`。

### 2.2 攻角定义 (v6.7 修正)

按文献 [32] 标准：

```
α_eff = -tanh(φ̇) · (α_install + δα)
δα = atan2(θ̇_p · x_wing, |Ω| · R)
```

- `α_install`: 翅膀弦线相对拍动平面的安装角。
- `-tanh(φ̇)`: 上下拍来流方向平滑翻转。
- `δα`: 俯仰气动阻尼带来的附加攻角。

### 2.3 气动力模型

四分量力模型：

1. **平动升力/阻力** (`F_trans`): 基于瞬时速度、混合 `C_L/C_D` 模型。
2. **附加质量力** (`F_AM`): ∝ `φ̈`。
3. **Clap-and-Fling** (`F_clap`): 速度-位置耦合增强，`k_clap = 0.3`。
4. **旋转力** (`F_rot`): 当前 `α̇=0`，自然为 0。

`C_L/C_D` 混合模型：

- `|α| ≤ 55°`: Dickinson 经验公式。
- `55° < |α| ≤ 65°`: smoothstep 过渡。
- `|α| > 65°`: LEV/Lee 理论模型。

### 2.4 俯仰动力学

```
I_yy · θ̈_p = M_aero + M_grav + M_damp
M_aero = -x_front · Fz_front - x_back · Fz_back
M_grav = -m · g · d_cg · sin(θ_p)
M_damp = -c_damp · θ̇_p
```

俯仰气动阻尼 `c_damp = 5e-4 N·m·s/rad` 是稳定性关键。

---

## 3. 数值方法

- **时间积分**: RK4，默认 `dt = 50μs`（扫描）/ `10μs`（力输出）。
- **加速**: numba JIT 编译 RK4 热循环。
- **并行**: 方案二使用 joblib (`n_jobs=-1`)。
- **稳定性判据**: `n_exceed_90 = 0`（5s 内 `|θ_p|` 不超过 90°）且 `L/W > 1`。

---

## 4. 数值计算流程

```
1. DXF 提取翅膀几何 (analyze_dxf.py)
2. 四连杆运动学求解 (mechanism.py)
3. 气动力 + 俯仰 ODE 求解 (butterfly_forces.py)
4. 参数扫描
   ├── 方案一: 单变量偏离 (stability_analysis.py)
   └── 方案二: 笛卡尔积全扫描 (sweep_cartesian.py)
5. 后处理与绘图 (stability_plot.py, plot_v68.py)
```

---

## 5. 图表说明

### 5.1 基线图 (`output/figures/stability/baseline/`)

由 `stability_plot.py --baseline` 生成，展示 `DESIGN_v68` 设计点的单次 5s 仿真。

| 图名 | 内容 | 解读要点 |
|------|------|---------|
| `baseline_overview.png` | 全时程概览：θ_p、θ̇_p、θ̈_p、Fz、Fx、M | 观察是否稳定、升力是否覆盖重量、力矩是否平衡 |
| `baseline_phase.png` | 相图 θ̇_p vs θ_p | 稳定时相轨迹收敛到极限环；发散时螺旋外扩 |
| `baseline_wings.png` | 四翅力时程（最后 0.3s） | 前/后翅、左/右翅的升力/推力对比 |
| `baseline_aero.png` | α_eff、C_L、C_D 时程 | 检查攻角是否超出物理合理范围 |
| `baseline_rocker.png` | 摇杆主矢 + 主矩 | 机构载荷，用于结构强度校核 |

### 5.2 单变量偏离图 (`output/figures/stability/sweep_<param>/`)

由 `stability_plot.py --all` 生成。每次只改一个参数，其他参数锁死 `DESIGN_v68`。

| 图名 | 内容 | 解读要点 |
|------|------|---------|
| `L_W_peak.png` | L/W 与 peak θ_p 随参数变化 | 稳定区、L/W 峰值、设计点位置 |
| `forces.png` | Fz_body、Fz_world、Fx 随参数变化 | 升力/推力对参数的敏感度 |
| `moments.png` | M_aero、M_grav、M_damp 随参数变化 | 力矩平衡与阻尼作用 |
| `pitch_rates.png` | 俯仰角速度/加速度随参数变化 | 动态响应剧烈程度 |
| `accel.png` | 俯仰加速度峰值随参数变化 | 平稳性指标 |
| `phase_grid.png` | 各参数值下的相图叠加 | 稳定/发散模式可视化 |

### 5.3 v6.8 全量扫参图 (`output/figures/aero/v68/`)

由 `plot_v68.py` 生成，基于 F 盘 11,809 组笛卡尔积数据。

| 图名 | 内容 | 解读要点 |
|------|------|---------|
| `fig1_kclap_sensitivity.png` | k_clap 敏感性全景 | k_clap 对 L/W、稳定率、α_eff 的影响 |
| `fig2_alpha_heatmap.png` | α_f × α_b L/W 热力图 | 最优 α_f/α_b 组合区域 |
| `fig3_mechanism_params.png` | 机构参数 R × a 分析 | R 和 a 对 L/W、稳定率的联合影响 |
| `fig4_phase_phioff.png` | phase & φ_off 效应 | 相位和安装偏角的折衷 |
| `fig5_performance_scatter.png` | 性能空间散点 | 7,909 稳定点的 L/W vs peak θ |
| `fig6_top_combos_designs.png` | Top 参数频率 + 方案对比 | 高 L/W 组合的参数分布 |
| `fig7_R225_deepdive.png` | R=2.25 专题深化 | R=2.25 时的参数敏感性 |
| `fig8_physical_reasonableness.png` | 物理合理性：α_f 趋势 + 分区 | α_f 与 mean α_eff、>70° 占比关系 |
| `fig9_reasonable_range_zoom.png` | 合理区放大 (α_f=40-50°) | 推荐设计区 |
| `fig10_full_performance_landscape.png` | 全景散点 (物理/非物理分区) | 物理合理与非物理结果的分界 |
| `fig11_design_v68_curves.png` | DESIGN_v68 设计点曲线 | 设计点的力/力矩/攻角时程 |

### 5.4 机构与翅膀图 (`output/figures/mechanism/`)

#### 代码生成图（随 DESIGN_v68 自动重绘）

| 图名 | 内容 | 解读要点 |
|------|------|---------|
| `mechanism_analysis.png` | 四连杆运动学分析 | φ(t)、φ̇(t)、φ̈(t) 时程 |
| `equivalent_output_torque.png` | 等效输出扭矩 | 机构驱动所需扭矩 |
| `wing_analysis.png` | 翅膀几何分析 | 面积、展长、面积矩 |
| `wing_shape.png` / `wing_poses.png` | 翅膀平面形状与姿态 | 安装角、拍动姿态 |
| `gear_force_diagram.png` / `gear_mesh_forces_vs_crank_angle.png` | 齿轮啮合力分析 | 减速器载荷 |
| `rocker_moment_vs_crank_angle.png` / `torque_chain_vs_crank_angle.png` | 摇杆力矩 / 扭矩链 | 机构动力传递 |

#### 静态示意图（尚未按 DESIGN_v68 重绘）

| 图名 | 内容 | 状态 |
|------|------|------|
| `曲柄摇杆受力简图.png` | 机构受力简图 | 静态图，仅作示意 |
| `机构简图.png` | 四连杆示意图 | 静态图，仅作示意 |
| `齿轮运动简图.png` | 齿轮减速器简图 | 静态图，仅作示意 |

---

## 6. 关键结论

1. **DESIGN_v68 是稳定的窄解**：L/W=2.447，peak θ=32.9°，n90=0。
2. **参数容差很小**：α_b≥10、phase≤-25、a≠6、R≠2.5、φ_off=-25、k_clap≥0.5 都会发散。
3. **提高 α_f 可增加 L/W**：在固定其他参数为 DESIGN_v68 时，α_f=50-70° 均稳定，L/W 从 2.70 增至 3.40，但 α_f≥60° 进入非物理失速区。
4. **物理合理性约束**：推荐 α_f≤50°，mean α_eff≤50°，>70° 占比 <5%。

---

## 7. 复现清单

| 输出 | 命令 |
|------|------|
| 基线图 | `python src/aero/stability_plot.py --baseline` |
| 单变量偏离图 | `python temp/run_all_sweeps.py && python src/aero/stability_plot.py --all` |
| v6.8 扫参图 | `python temp/analyze_sweep_v68.py && python temp/plot_sweep_v68.py` |
| 机构图 | `python src/struct/generate_mechanical_principles_assets.py` |
| 翅膀图 | `python src/aero/plot_wing_shape.py` |
