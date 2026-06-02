# DXF 翅膀几何分析模块（analyze_dxf.py）

> 本文件说明 `src/analyze_dxf.py` 的功能、DXF 解析逻辑与几何参数计算方法。  
> 负责从 SolidWorks DXF 导出文件中提取翅膀平面形状参数，供气动分析使用。

---

## 1. 功能概述

1. **DXF 解析**：读取 SPLINE / LINE / CIRCLE 实体
2. **轮廓重建**：连接离散实体为连续翅膀轮廓
3. **几何参数计算**：面积、展长、平均弦长、展弦比、面积矩
4. **JSON 输出**：保存为 `data/wing_analysis_results.json`

---

## 2. 输入文件

| 文件 | 内容 | 来源 |
|---|---|---|
| `data/WingFront.DXF` | 前翅轮廓 | SolidWorks 草图导出 |
| `data/WingBack.DXF` | 后翅轮廓 | SolidWorks 草图导出 |
| `data/WingsAxis.DXF` | 转轴端点（两个圆）| SolidWorks 草图导出 |

---

## 3. 核心流程

```
parse_dxf(filepath)
    → 提取 SPLINE 控制点 / LINE 端点 / CIRCLE 圆心

connect_entities(entities, scale=1e-3)
    → 按首尾相连排序，生成闭合轮廓

calculate_planform_properties(points_mm, axis)
    → 面积 S、展长 R、平均弦长 c_avg
    → 一阶矩 r1、二阶矩 r2_sq、展弦比 AR
```

---

## 4. 关键参数定义

| 参数 | 符号 | 计算方式 |
|---|---|---|
| 面积 | S | 多边形 shoelace 公式 [m²] |
| 展长 | R | 转轴到最远点距离 [m] |
| 平均弦长 | c_avg | S / R [m] |
| 展弦比 | AR | R / c_avg |
| 一阶矩位置 | r̂₁ | ∫r·dS / (R·S) |
| 二阶矩 | r̂₂² | ∫r²·dS / (R²·S) |

---

## 5. 输出 JSON 结构

```json
{
  "geometry": [
    {
      "name": "Front",
      "S": 0.01617,
      "R": 0.1543,
      "c_avg": 0.1048,
      "r1": 0.4227,
      "r2_sq": 0.2382,
      "AR": 1.47
    }
  ]
}
```

---

## 6. 已知问题

- **正弦假设**：静态估算中使用了 `φ̇_max = 2πfΦ`，与 `mechanism.py` 实际运动学脱节
- **phi_down_deg / phi_up_deg**：基于旧模型的静态估算参数，新四连杆模型下已不适用

---

## 7. 使用示例

```bash
python src/analyze_dxf.py
```

输出：`data/wing_analysis_results.json` + `output/figures/wing_analysis.png`

---

## 8. 文件位置

| 文件 | 路径 |
|---|---|
| 分析脚本 | `src/analyze_dxf.py` |
| 输出 JSON | `data/wing_analysis_results.json` |
| 输出图表 | `output/figures/wing_analysis.png` |
