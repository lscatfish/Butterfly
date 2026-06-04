# 翅膀平面形状绘制工具（plot_wing_shape.py）

> 本文件说明 `src/plot_wing_shape.py` 的功能与使用方法。  
> 用于可视化仿生蝴蝶翅膀的平面形状（planform）及不同拍动姿态。

---

## 1. 功能概述

读取 DXF 文件，在同一坐标系中绘制：

- **转轴**（`WingsAxis.DXF` 中的两个圆心连线）
- **前翅轮廓**（`WingFront.DXF`）
- **后翅轮廓**（`WingBack.DXF`）

支持指定拍动角 φ，将翅膀绕转轴端点旋转后绘制，用于展示不同拍动姿态。

---

## 2. 依赖关系

```
plot_wing_shape.py
    ├── analyze_dxf.py  (parse_dxf, connect_entities, read_axis_from_dxf)
    └── matplotlib / numpy
```

输出目录：`output/figures/`

---

## 3. 旋转方法

绕 hinge line（转轴）进行三维旋转后取 XY 平面投影：

1. 计算轴中点 `c` 和方向单位向量 `unit_dir`
2. 对点集 `P`：分解为平行于轴和垂直于轴的分量
3. 旋转后，垂直分量在 XY 投影中被压缩 `cos(φ)` 倍

数学上：
```
P' = c + v_parallel + v_perp * cos(φ)
```

这是一种**简化的三维旋转投影**，适用于小角度拍动的平面可视化。

---

## 4. 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--phi` | float | 0.0 | 单姿态拍动角 [deg] |
| `--poses` | float[] | None | 多姿态拍动角列表 |
| `--output` | str | None | 输出文件名 |

---

## 5. 使用示例

### 5.1 默认绘制（φ=0）

```bash
python src/plot_wing_shape.py
```

### 5.2 指定单姿态

```bash
python src/plot_wing_shape.py --phi 30 --output wing_pose_up.png
python src/plot_wing_shape.py --phi -30 --output wing_pose_down.png
```

### 5.3 多姿态叠加

```bash
python src/plot_wing_shape.py --poses -60 -30 0 30 60 --output wing_poses.png
```

---

## 6. 输出文件

| 文件 | 说明 |
|---|---|
| `wing_shape.png` | 默认单姿态输出 |
| `wing_poses.png` | 多姿态叠加输出 |
| `wing_analysis.png` | 翅膀几何分析图（由 analyze_dxf.py 生成）|

---

## 7. 文件位置

| 文件 | 路径 |
|---|---|
| 绘制脚本 | `src/plot_wing_shape.py` |
| DXF 数据 | `data/WingFront.DXF`, `data/WingBack.DXF`, `data/WingsAxis.DXF` |
