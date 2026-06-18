# 机械原理报告配套图复现说明

本说明用于复现以下两个文档中的机械原理配套图片和汇总数据：

```text
output/reports/机械原理结构系统分析.md
output/reports/机械原理曲线图读图说明.md
```

## 运行命令

在项目根目录运行：

```bash
python src/struct/generate_mechanical_principles_assets.py
```

该命令默认只重新生成图片和 JSON 汇总文件，不覆盖已经手工修改过的 Markdown 报告。

## 生成内容

```text
output/figures/mechanism_schematic_reference.png
output/figures/crank_rocker_reference.png
output/figures/crank_rocker_fbd_reference.png
output/figures/rocker_moment_vs_crank_angle.png
output/figures/torque_chain_vs_crank_angle.png
output/figures/gear_mesh_forces_vs_crank_angle.png
output/tables/mechanical_principles_summary.json
```

其中三张线图分别为：

| 图 | 说明 |
|:---|:---|
| `rocker_moment_vs_crank_angle.png` | 摇杆端气动、惯性和合外载力矩随曲柄角变化 |
| `torque_chain_vs_crank_angle.png` | 曲柄轴和电机轴扭矩随曲柄角变化 |
| `gear_mesh_forces_vs_crank_angle.png` | 两级齿轮啮合力随曲柄角变化 |

## 计算说明

配套计算入口位于：

```text
src/struct/mechanical_principles_analysis.py
```

主要流程为：

```text
四杆机构运动学
→ 翅膀气动力矩和惯性力矩
→ 摇杆端等效外载 M_wing
→ 功率等效换算曲柄轴扭矩
→ 曲柄摇杆效率和轮系效率折算电机轴扭矩
→ 按齿轮公式换算啮合力
```

当前为减少气动分段公式带来的小尖角，对气动力矩 `M_aero` 采用周期 Hann 窗平滑：

```text
aero_moment_smoothing = periodic_hann
window_points = 21
window_deg = 10.5
```

曲柄摇杆等效效率和轮系效率为：

```text
η_l = 0.90
η_g = 0.9319
```

## 可选：重新生成报告模板

如果确实需要用脚本模板覆盖生成主分析报告，可运行：

```bash
python src/struct/generate_mechanical_principles_assets.py --write-report-template
```

注意：该选项会重新写入：

```text
output/reports/机械原理结构系统分析.md
```

因此如果报告已经经过手工修改，通常不要使用这个选项。

