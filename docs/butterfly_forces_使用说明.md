# butterfly_forces.py 使用说明

> 蝴蝶扑翼力输出模块 — 让傻子都能看懂的使用指南

---

## 0. 这玩意儿是干啥的？

一句话：**输入蝴蝶的设计参数，输出每只翅膀上受到的力和力矩。**

你要分析摇杆（四连杆机构的输出端）受多大的力？曲柄需要多大扭矩？翅膀上气动力怎么分布的？——调这个模块就行。

---

## 1. 坐标系（很重要，先看这里）

### 1.1 体轴系 (Body Frame)

想象你坐在蝴蝶身上：

```
         Z (上)
         ↑
         |
         +----→ X (前，头冲的方向)
        /
       ↙
      Y (右，伸开右边翅膀的方向)
```

- **原点**：蝴蝶的重心 (CG)
- **X**：头指向的方向 = "前面"
- **Y**：右边翅膀伸开的方向 = "展向"
- **Z**：后背指向的方向 = "上面"

**体轴系是跟着蝴蝶一起转的。** 蝴蝶低头抬头，这个坐标系就跟着转。

### 1.2 世界系 (World Frame)

```
         Z_w (真正的上，重力反方向)
         ↑
         |
         +----→ X_w (水平前方)
```

- 蝴蝶平飞（θ_p=0）时，世界系和体轴系重合
- 世界系**不跟着蝴蝶转**
- Z_w 永远是重力的反方向

### 1.3 为什么有两个坐标系？

| 你要干什么 | 用哪个 |
|-----------|--------|
| 分析摇杆受力、设计机构强度 | **体轴系**（力是翅膀直接传到摇杆上的） |
| 算蝴蝶能不能飞起来、升力够不够 | **世界系**（升力必须克服重力） |
| 看翅膀的气动特性（C_L, C_D, 攻角） | **体轴系** |

**当前状态**：蝴蝶身体假设不动（固定在空中），只分析机构受力。所以体轴系是最常用的。

### 1.4 机构平面 & 摇杆分解

四连杆机构在 **体轴 XZ 平面** 内运动。摇杆（输出杆 A→P2）在这个平面内来回摆。

```
机构坐标  →  体轴坐标
  机构 x   →  体轴 X (前)
  机构 y   →  体轴 Z (上)
  旋转轴   →  体轴 Y (展向)
```

**摇杆主矢**：翅膀力沿摇杆杆方向的分量（这个力通过连杆传到曲柄上）

**摇杆主矩**：翅膀力对摇杆枢轴 A 产生的扭矩（在 Y 轴上的分量，是驱动/制动摇杆的有效扭矩）

```
摇杆枢轴 A 位置（机构坐标）：(0, a) mm，其中 a 默认=7.92mm
摇杆枢轴 A 位置（体轴）：(x_wing, y_hinge, a/1000) m
摇杆长度 l = 8mm
```

---

## 2. 安装

不需要安装。把 `src/butterfly_forces.py` 放到你的项目里，然后：

```python
import sys
sys.path.insert(0, 'path/to/Butterfly/src')
from butterfly_forces import *
```

**依赖**：numpy, matplotlib（仅绘图需要）, `src/mechanism.py`（已包含）, `data/wing_analysis_results.json`（已包含）

---

## 3. 快速上手（5 分钟学会）

### 3.1 最基本用法

```python
from butterfly_forces import SimulationConfig, ButterflyForceModel

# 用默认参数（已经是最优的 α_f=60°, α_b=8°）
cfg = SimulationConfig()
model = ButterflyForceModel(cfg)

# 跑仿真！
out = model.simulate()

# 看结果
print(out.summary)
# → {'L/W': 1.033, 'peak_theta_deg': 46.7, 'n_exceed_90': 0, ...}

# 拿前翅左的体轴力
F_FL = out.wings["FL"].force_body   # shape: (N, 3) — [Fx, Fy, Fz] 每行
```

### 3.2 改参数

```python
cfg = SimulationConfig(
    alpha_front_deg=55,   # 前翅安装角 55°
    alpha_back_deg=12,    # 后翅安装角 12°
    phase_diff_deg=30,    # 前后翅相位差 30°
    f=18.0,               # 频率改 18Hz
    dt=10e-6,             # 时间步长 10 微秒
    t_end=10.0,           # 仿真 10 秒
)
```

**所有能改的参数见第 5 节。**

### 3.3 只看稳态数据（5 秒之后）

```python
out = model.simulate()

# 找到 5 秒对应的索引
t = out.t
mask_steady = t >= 5.0  # 5 秒后算稳态

# 取稳态段的前翅左 Fz
Fz_FL_steady = out.wings["FL"].force_body[mask_steady, 2]  # Z 分量
print(f"稳态平均 Fz = {Fz_FL_steady.mean()*1000:.1f} mN")
```

### 3.4 参数扫描（找最优参数）

```python
from butterfly_forces import scan_parameters

cfg = SimulationConfig()

results = scan_parameters(cfg, {
    "alpha_front_deg": [55, 60, 65],
    "alpha_back_deg": [8, 10, 12],
    "mech_a": [7.5, 7.92],      # 可以扫机构参数！
})
# → 返回 list[dict]，按 L/W 从高到低排

for r in results:
    print(f"α_f={r['alpha_front_deg']} α_b={r['alpha_back_deg']} "
          f"L/W={r['L/W']:.3f} peak={r['peak_deg']:.1f}°")
```

---

## 4. 输出结构详解

### 4.1 `out` 的顶层字段

```python
out.t              # np.ndarray (N,)  时间序列 [s]
out.theta_p        # np.ndarray (N,)  俯仰角 [rad]
out.theta_dot      # np.ndarray (N,)  俯仰角速度 [rad/s]
out.theta_ddot     # np.ndarray (N,)  俯仰角加速度 [rad/s²]

out.wings          # dict, key = "FL", "FR", "BL", "BR"
                   #   FL = Front Left  (前翅左)
                   #   FR = Front Right (前翅右)
                   #   BL = Back Left   (后翅左)
                   #   BR = Back Right  (后翅右)

out.summary        # dict, 聚合统计
out.config         # SimulationConfig, 回记用的参数
```

### 4.2 每只翅膀的字段 (`out.wings["FL"].xxx`)

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `force_body` | (N,3) | N | **体轴力** [Fx, Fy, Fz]。Fz>0 = 向上 |
| `force_world` | (N,3) | N | **世界力**。Fz_w>0 = 克服重力 |
| `cop_body` | (N,3) | m | **气动中心位置**（体轴）。力作用在这个点上 |
| `cop_world` | (N,3) | m | 气动中心位置（世界） |
| `moment_body` | (N,3) | N·m | 对 CG 的**力矩**（体轴）。M = r_cop × F |
| `moment_world` | (N,3) | N·m | 对 CG 的力矩（世界） |
| `rocker_principal_vec` | (N,3) | N | **摇杆主矢**。沿摇杆方向的力分量 |
| `rocker_principal_moment` | (N,3) | N·m | **摇杆主矩**。对摇杆枢轴的有效扭矩 |
| `rocker_angle_rad` | (N,) | rad | 摇杆在机构平面内的角度 |
| `alpha_eff_deg` | (N,) | ° | 有效气动攻角（翅膀实际感受到的） |
| `C_L` | (N,) | - | 升力系数 |
| `C_D` | (N,) | - | 阻力系数 |
| `phi` | (N,) | rad | 翅膀拍动角 |
| `phi_dot` | (N,) | rad/s | 拍动角速度 |
| `phi_ddot` | (N,) | rad/s² | 拍动角加速度 |

### 4.3 `out.summary` 的字段

| 字段 | 说明 |
|------|------|
| `L/W` | 升力/重量比（体轴 Fz）。≥1.0 = 能悬停 |
| `avg_Fz_body_mN` | 稳态平均体轴升力 [mN] |
| `avg_Fz_world_mN` | 稳态平均世界升力 [mN] |
| `avg_Fx_body_mN` | 稳态平均体轴推力 [mN] |
| `weight_mN` | 蝴蝶重量 [mN]（~196 mN） |
| `peak_theta_deg` | 俯仰角峰值 [°] |
| `n_exceed_90` | 俯仰超过 ±90° 的步数（>0 = 不稳定） |
| `n_steps` | 仿真总步数 |
| `dt_s` | 时间步长 [s] |

---

## 5. 所有可配置参数

```python
@dataclass
class SimulationConfig:
    # ===== 翅膀安装 =====
    alpha_front_deg: float = 60.0     # 前翅安装角 [°] — 翅膀弦线与机身水平面的夹角
    alpha_back_deg: float = 8.0       # 后翅安装角 [°]
    phase_diff_deg: float = 0.0       # 前后翅相位差 [°] — 0=同相, 180=反相

    # ===== 四连杆机构 =====
    mech_a: float = 7.92              # 铰链 A 的 y 坐标 [mm] — 改这个影响摆幅和急回比
    mech_b: float = 6.97              # 曲柄中心 x 坐标 [mm]
    mech_R: float = 2.25              # 曲柄半径 [mm] — 改这个直接改摆幅
    mech_c: float = 14.00             # 连杆长度 [mm]
    mech_l: float = 8.00              # 摇杆长度 [mm]
    phi_offset_deg: float = -50.84    # 翅膀在摇杆上的安装偏角 [°]
    rotation: str = 'cw'              # 曲柄转向 'cw' 或 'ccw'

    # ===== 物理 =====
    f: float = 15.0                   # 扑动频率 [Hz]
    rho: float = 1.225                # 空气密度 [kg/m³]
    m_total: float = 0.020            # 总质量 [kg]
    I_yy: float = 3e-5                # 俯仰转动惯量 [kg·m²]
    d_cg: float = 0.015               # CG 在铰链下方距离 [m]
    x_front: float = 0.025            # 前翅铰链 x 位置 [m]（CG 前方为正）
    x_back: float = -0.025            # 后翅铰链 x 位置 [m]（CG 后方为负）
    g: float = 9.81                   # 重力加速度 [m/s²]

    # ===== 数值 =====
    dt: float = 10e-6                 # 时间步长 [s] — 10μs 用于精细力分析
    t_end: float = 10.0               # 仿真总时间 [s] — 5s 后为稳态
    theta0_deg: float = 0.0           # 初始俯仰角 [°]
    steady_start: float = 5.0         # 视为稳态的起始时间 [s]

    # ===== 气动系数（一般不用改） =====
    k_3d: float = 0.7                 # 3D 展向修正
    C_rot: float = 1.5                # 旋转力系数
    r_rot: float = 0.5                # 旋转力展向位置
    k_clap: float = 1.3               # Clap-and-Fling 增强系数
    c_damp: float = 5e-4              # 俯仰阻尼 [N·m·s/rad]
```

---

## 6. 常见场景

### 场景 A：机构分析（摇杆/连杆受力）

```python
cfg = SimulationConfig(alpha_front_deg=60, alpha_back_deg=8, dt=10e-6, t_end=10.0)
model = ButterflyForceModel(cfg)
out = model.simulate()

# 只取稳态（5 秒后）
mask = out.t >= 5.0

# 前翅左摇杆主矢
F_rocker_FL = out.wings["FL"].rocker_principal_vec[mask]
# 后翅右摇杆主矩（扭矩）
M_rocker_BR = out.wings["BR"].rocker_principal_moment[mask]

# 一个周期内的平均值
import numpy as np
print(f"前翅左摇杆主矢均值: {np.mean(np.linalg.norm(F_rocker_FL, axis=1)):.4f} N")
print(f"后翅右摇杆主矩均值: {np.mean(M_rocker_BR[:, 1]):.6f} N·m")
```

### 场景 B：验证新参数是否稳定

```python
cfg = SimulationConfig(alpha_front_deg=62, alpha_back_deg=6)
model = ButterflyForceModel(cfg)
out = model.simulate()

if out.summary['n_exceed_90'] == 0:
    print(f"✅ 稳定！L/W={out.summary['L/W']:.3f}")
else:
    print(f"❌ 不稳定！俯仰超过90°的次数: {out.summary['n_exceed_90']}")
```

### 场景 C：扫描最优安装角

```python
from butterfly_forces import scan_parameters

results = scan_parameters(
    SimulationConfig(),
    {"alpha_front_deg": range(40, 65, 5), "alpha_back_deg": range(5, 20, 2)},
    t_end=3.0, dt=50e-6,   # 扫描用粗步长提速
)
# 导出 CSV
import json, csv
with open('scan_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader(); w.writerows(results)
```

### 场景 D：只看力，不跑俯仰仿真（固定 θ_p=0）

```python
# 把初始俯仰设 0，I_yy 设很大（让身体几乎不动）
cfg = SimulationConfig(
    theta0_deg=0, I_yy=1e10, t_end=1.0, dt=50e-6,
)
model = ButterflyForceModel(cfg)
out = model.simulate()
# 此时 theta_p ≈ 0 全程，纯翅膀力
```

---

## 7. 注意事项

1. **时间步长选择**：精细力分析用 `dt=10e-6`（10μs），参数扫描用 `dt=50e-6`（50μs）提速 5 倍
2. **稳态判断**：5 秒后数据视为稳态。`t_end` 必须 > `steady_start`
3. **L/W 的体轴 vs 世界**：`summary["L/W"]` 用体轴 Fz 算的。当前固定模型下，世界系 Fz 只有体轴的一半左右（因为身体 pitch 振荡导致投影损失）
4. **翅膀编号**：FL=前左, FR=前右, BL=后左, BR=后右。左右翅力对称（Fy 符号相反）
5. **内存**：10s@10μs 产生约 1M 步 × 4 翅 × 3 分量 ≈ 96MB 数据。内存不够时用 `dt=50e-6`

---

## 8. C_L/C_D 模型说明

本模块使用 **v6.3 LEV/Lee 混合模型**：

```
|α| ≤ 55°: Dickinson 经验公式（含 LEV 增强效应）
55°~65°:   smoothstep 平滑过渡
|α| ≥ 65°: C_L = 1.866·sin(2α), C_D = 0.393 + 1.414·(1−cos(2α))
```

- C_D_max = 3.221（接近昆虫实测 ~3.0−3.5）
- C_L 在 α=90° 自然归零，α>90° 自动变号（物理正确）
- 参考：文献 [32] JRSI 2017 (LEV), [24] 机器人 2025 (Lee)

---

## 9. 力分量模型

每只翅膀的力由四部分组成：

| 分量 | 来源 | 公式要点 |
|------|------|---------|
| **平动力** (F_trans) | 翅膀拍动产生的升力/阻力 | ∝ U²·S·C_L/C_D |
| **附加质量力** (F_AM) | 翅膀加减速推动周围空气 | ∝ φ̈·c²·R·sin(α) |
| **旋转力** (F_rot) | 翅膀主动扭转 | ∝ α̇·φ̇（当前 α̇=0，此项为 0） |
| **Clap-and-Fling** | 翅膀在反转点拍合/打开 | 反转点附近 ×1.3 |

所有力在反转点（|φ̇| < 10% 峰值）乘 `k_clap=1.3`。

---

## 10. 完整示例脚本

```python
#!/usr/bin/env python3
"""示例：输出蝴蝶翅膀力数据做机构分析"""
import sys, numpy as np
sys.path.insert(0, 'D:/code/Butterfly/src')
from butterfly_forces import *

# 1. 配置
cfg = SimulationConfig(
    alpha_front_deg=60, alpha_back_deg=8,
    dt=10e-6, t_end=10.0, steady_start=5.0,
)

# 2. 仿真
print("仿真中...")
model = ButterflyForceModel(cfg)
out = model.simulate()

# 3. 提取稳态数据
mask = out.t >= 5.0
t_steady = out.t[mask]
dt_avg = np.mean(np.diff(t_steady))

# 4. 输出每翅力（一个周期的平均）
print(f"\n{'翅':<6} {'Fz_body(mN)':>12} {'Fx_body(mN)':>12} {'|M_rocker|(N·m)':>16}")
print("-" * 50)
for name in ["FL", "FR", "BL", "BR"]:
    w = out.wings[name]
    fz = np.mean(w.force_body[mask, 2]) * 1000
    fx = np.mean(w.force_body[mask, 0]) * 1000
    mr = np.mean(np.abs(w.rocker_principal_moment[mask, 1]))
    print(f"{name:<6} {fz:>+12.1f} {fx:>+12.1f} {mr:>16.8f}")

# 5. 汇总
s = out.summary
print(f"\n总重: {s['weight_mN']:.0f} mN")
print(f"体轴 Fz: {s['avg_Fz_body_mN']:+.0f} mN | L/W: {s['L/W']:.3f}")
print(f"峰值俯仰: {s['peak_theta_deg']:.1f}° | 超90°次数: {s['n_exceed_90']}")
print(f"状态: {'✅ 稳定' if s['n_exceed_90']==0 else '❌ 不稳定'}")
```
