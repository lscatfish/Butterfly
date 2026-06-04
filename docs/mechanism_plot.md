# 前置机构运动学可视化工具（mechanism_plot.py）

> 本文件说明 `src/mechanism_plot.py` 的功能、命令行接口、输出图表与使用方法。  
> 该脚本依赖 `src/mechanism.py` 的运动学求解器，用于绘制四连杆机构的运动学特性。

---

## 1. 功能概述

`mechanism_plot.py` 是 `mechanism.py` 的可视化前端，负责：

1. **机构几何示意图** — 固定点、轨迹圆、若干机构姿态
2. **顺/逆时针运动学对比** — 一个周期内的角度 φ、角速度 φ̇、角加速度 φ̈
3. **a 参数扫描重叠图** — 不同 a 值下的 φ(t) 与 φ̇(t) 曲线叠加

所有运动学数据均通过 `mechanism.wing_kinematics()` 获取，**不使用任何正弦近似**。

---

## 2. 依赖关系

```
mechanism_plot.py
    ├── mechanism.py  (DEFAULT_PARAMS, solve_phi, wing_kinematics)
    └── matplotlib    ( Agg backend, 中文显示 )
```

输出目录：`output/figures/mechanism_analysis.png`

---

## 3. 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--a` | float | 7.92 | 机构参数 a（A 点 y 坐标），飞行中可调 |
| `--a-list` | float[] | `[6,7,7.92,9,10,11]` | a 扫描列表，用于重叠图 |
| `--freq` | float | 15.0 | 振翅频率 [Hz] |
| `--phi-offset` | float | `None` | 角度偏移 [deg]。**默认使用 `mechanism.py` 的 `DEFAULT_PARAMS['phi_offset_deg']`**（当前 -50.84°）。传 `0` 可显式关闭偏移 |
| `--no-plot` | flag | False | 仅打印分析结果，不生成图片 |

**关键行为**：`--phi-offset` 为 `None` 时，脚本自动读取 `mechanism.DEFAULT_PARAMS['phi_offset_deg']`（-50.84°）。这是出厂固定的翅膀折弯角，**不随 a 的改变而变化**。

---

## 4. 输出图表说明

`output/figures/mechanism_analysis.png` 为 2×3 布局，共 6 个子图：

| 子图 | 内容 | 说明 |
|---|---|---|
| 1 | **机构几何示意图** | 固定点 A/B、摇杆圆（绿色虚线）、曲柄圆（蓝色虚线）、8 个机构姿态的连杆/摇杆/曲柄 |
| 2 | **φ(t) cw vs ccw** | 顺/逆时针一个周期内的角度变化。默认偏移后应关于 0° 对称 |
| 3 | **φ̇(t) cw vs ccw** | 角速度对比。峰值相同，时间分布相反（急回特性） |
| 4 | **φ̈(t) cw vs ccw** | 角加速度对比。峰值相同，相位相反 |
| 5 | **a 扫描 φ(t)** | 不同 a 值的角度曲线重叠。仅 a=7.92 时关于 0° 对称（因偏移固定） |
| 6 | **a 扫描 φ̇(t)** | 不同 a 值的角速度曲线重叠。摆幅随 a 增大而减小 |

---

## 5. 使用示例

### 5.1 默认运行（自动应用对称偏移）

```bash
python src/mechanism_plot.py
```

输出示例：
```
rotation='cw', offset=-50.84°: span=44.47°, |φ̇|max=44.21, |φ̈|max=5730.36
rotation='ccw', offset=-50.84°: span=44.47°, |φ̇|max=44.21, |φ̈|max=5730.36
Saved: output/figures/mechanism_analysis.png
```

### 5.2 查看原始机构输出（无偏移）

```bash
python src/mechanism_plot.py --phi-offset 0
```

原始机构角度范围约 `[28.6°, 73.1°]`，全在上拍区。

### 5.3 指定频率和 a 值

```bash
python src/mechanism_plot.py --freq 20 --a 9.0
```

### 5.4 自定义 a 扫描列表

```bash
python src/mechanism_plot.py --a-list 6 7 8 9 10
```

---

## 6. 与 mechanism.py 的衔接

- **纯几何分析** (`analyze_geometry`) 直接调用 `solve_phi`，与频率无关，用于统计摆幅和角度范围
- **运动学分析** 调用 `wing_kinematics`，自动继承 `DEFAULT_PARAMS` 中的默认偏移
- **机构几何图** 使用 `solve_phi` 反求 P2 坐标，与运动学计算保持一致

---

## 7. 文件位置

| 文件 | 路径 |
|---|---|
| 可视化脚本 | `src/mechanism_plot.py` |
| 依赖的运动学模块 | `src/mechanism.py` |
| 输出图片 | `output/figures/mechanism_analysis.png` |
