# Butterfly — 仿生蝴蝶扑翼 MAV 气动分析

仿生蝴蝶扑翼微型飞行器（MAV）俯仰动力学仿真。质量 20g，曲柄摇杆四连杆机构驱动，15–17 Hz 扑动频率。

## 快速开始

```bash
# 单次仿真
python -m src.aero.butterfly_forces

# 调用示例
python -c "
from src.aero.butterfly_forces import SimulationConfig, ButterflyForceModel
cfg = SimulationConfig(alpha_front_deg=60, alpha_back_deg=3, phase_diff_deg=-15)
model = ButterflyForceModel(cfg)
out = model.simulate()
print(out.summary)
"

# 参数扫描
python src/aero/sweep_cartesian.py
```

## 安装

```bash
pip install numpy scipy pyyaml joblib numba
```

## 项目结构

```
├── config/
│   └── design_v69.yaml           # 参数唯一来源（机构+气动+物理+扫参网格）
├── src/
│   ├── config.py                 # YAML 参数加载器
│   ├── aero/
│   │   ├── butterfly_forces.py   # 主仿真管线（运动学→ODE→力输出）
│   │   ├── sweep_cartesian.py    # 笛卡尔积全参数扫描（joblib并行）
│   │   ├── stability_analysis.py # 单变量偏离稳定性分析
│   │   └── stability_plot.py     # 稳定性结果绘图
│   └── struct/
│       ├── mechanism.py          # 曲柄摇杆四连杆运动学
│       ├── mechanism_plot.py     # 机构运动学可视化
│       ├── analyze_dxf.py        # DXF 翅膀几何提取
│       └── ...
├── data/                         # 翅膀 DXF 图纸 + 面积/展长数据
├── docs/                         # 文献综述、模块文档、分析报告
└── temp/                         # 历史脚本 + 扫参输出
```

## 核心设计参数 (v6.9 DESIGN_v69)

参数唯一来源：`config/design_v69.yaml`

| 参数 | 值 | 说明 |
|------|-----|------|
| α_f | 60° | 前翅安装角 |
| α_b | 3° | 后翅安装角 |
| phase | -15° | 前后翅相位差 |
| f | 17 Hz | 扑动频率 |
| 摆幅 | ~99° | 新四连杆机构 |
| k_clap | 0.3 | Clap-and-Fling 增强系数 |

**性能**: L/W = 4.43, peak θ = 65.8°

## 物理模型

- **四分量力模型**: 平动升力/阻力 + 附加质量力 + Clap-and-Fling + 旋转力
- **C_L/C_D 混合模型**: LEV/Lee 理论 + Dickinson 经验公式，tanh 平滑过渡
- **攻角定义**: α = η（相对拍动平面），文献[32]标准
- **气动俯仰阻尼**: Δα ∝ θ̇_p × x_wing / U，维持俯仰稳定性

## 参数配置

修改 `config/design_v69.yaml` 即影响所有模块：

```python
from src.config import get_design, get_sweep_grid

design = get_design()          # 完整参数 dict
grid   = get_sweep_grid()     # 扫参网格（列表=扫参，标量=固定）
```

`config/design_v69.yaml` 包含四个主键 (`mechanism`, `aero`, `physical`, `numerical`) 和 `sweep` 段。Sweep 段中列表值参与笛卡尔积扫描，标量值固定，`_n_jobs` / `_out_dir` 为扫参引擎配置。

## 许可

内部研究项目。
