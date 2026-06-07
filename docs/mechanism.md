# 前置机构运动学模块（mechanism.py）

> 本文件说明 `src/mechanism.py` 的设计原理、几何定义、接口与使用方法。
> 对应 SolidWorks 装配图中的曲柄摇杆四连杆机构，以及 MATLAB 脚本 `angularvelocity2dimensions.m` 的运动学模型。

---

## 1. 机构类型

**曲柄摇杆四连杆机构（Crank-Rocker 4-Bar Linkage）**

```
        A(0,a)                          P2（翅膀端点）
         ●─────────────────────────────────●
         │            摇杆 l = 8.00
         │
         │  连杆 c = 14.00
         │
         │
    ─────┼───────────────────────────────────────→ x
         │
         │            曲柄 R = 2.25
         │         ╱
    ─────┴────────●──────────────────────────────
              B(b,0)                    P1（主点/曲柄端点）
```

- 曲柄 BP1 绕固定点 B 匀速旋转（电机驱动）
- 摇杆 AP2 绕固定点 A 摆动（驱动翅膀）
- 连杆 P1P2 传递运动
- 曲柄转一圈 = 翅膀一个完整拍动周期

---

## 2. 坐标系定义

以 SolidWorks 装配图中的两条参考轴为基准：

- **y 轴**：过 A 点的竖直虚线（翅膀转轴所在直线）
- **x 轴**：过 B 点（圆心）的水平轴
- **原点 O**：x 轴与 y 轴的交点

因此：

| 点 | 坐标 | 含义 |
|---|---|---|
| A | (0, a) | 翅膀转轴（摇杆支点） |
| B | (b, 0) | 曲柄圆心（crank 支点） |
| P1 | B + R(cosθ, sinθ) | 曲柄端点（主点） |
| P2 | 由几何约束求解 | 摇杆端点（翅膀杆端点） |

**翅膀转角 φ**：向量 A→P2 与 +x 轴的夹角，向上为正，向下为负。

---

## 3. 机构参数

| 符号 | 默认值 | 单位 | 含义 | 可调性 |
|---|---|---|---|---|
| a | 7.92 | mm | A 点 y 坐标（翅膀转轴高度） | **主调参数** |
| b | 6.97 | mm | B 点 x 坐标（曲柄圆心水平位置） | 固定 |
| R | 2.25 | mm | 曲柄半径（主点圆 Ø4.50 的一半） | 固定 |
| c | 14.00 | mm | 连杆 P1-P2 长度 | 固定 |
| l | 8.00 | mm | 摇杆/翅膀杆 A-P2 长度 | 固定 |
| φ_offset | -50.84 | ° | 翅膀安装基准偏移（出厂固定折弯角） | 固定 |

参数对应关系（与 MATLAB `angularvelocity2dimensions.m` 一致）：
- `a` = `y0`（A 点竖直坐标）
- `b` = `x0`（B 点水平坐标）
- `R` = `R`（主点半径）
- `c` = `line_c`（直线方程常数，新模型中为连杆长度）
- `l` = `l`（固定圆半径，新模型中为摇杆长度）
- `phi_offset_deg` 为出厂固定折弯角，飞行中不随 a 改变

---

## 4. 核心函数

### 4.1 `solve_phi(theta, params)`

给定曲柄角 θ，求解翅膀转角 φ。

**几何方法**：三角形余弦定理

在 ΔA-P1-P2 中，已知三边 AP2=l、P1P2=c、AP1=d(θ)=|P1-A|，由余弦定理：

$$
\cos(\angle P_2AP_1) = \frac{l^2 + d^2 - c^2}{2 \cdot l \cdot d}
$$

向量 A→P1 的方向角：θ_AP1 = atan2(P1y - a, P1x)

翅膀角 φ = θ_AP1 + ∠P2AP1（取上方/外侧交点，保证机构连续）

**参数**：
- `theta`：曲柄角 [rad]，顺时针旋转对应 θ 从 0 向 -2π 变化
- `params`：dict，需包含 a, b, R, c, l

**返回**：φ [rad]，若机构无解返回 `np.nan`

---

### 4.2 `wing_kinematics(f, a, rotation, phi_offset_deg, params, n_points)`

生成完整翅膀运动学（时间序列）。

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `f` | float | 必填 | 振翅频率 [Hz]，曲柄转一圈 = 翅膀一拍 |
| `a` | float | None | 翅膀转轴 A 的 y 坐标 [mm]，优先级最高，覆盖 params |
| `rotation` | str | 'cw' | 主点旋转方向：'cw'=顺时针，'ccw'=逆时针 |
| `phi_offset_deg` | float | -50.84 | 翅膀安装基准固定偏移 [deg]。默认从 `DEFAULT_PARAMS` 读取，使 a=7.92 时关于 0° 对称。传 0 可显式关闭偏移 |
| `params` | dict | None | 其他机构参数（b, R, c, l, phi_offset_deg），可部分覆盖 `DEFAULT_PARAMS` |
| `n_points` | int | 2000 | 每周期时间采样点数 |

**重要说明**：
- `phi_offset_deg` 在 `np.unwrap` 之后、微分之前施加，因此 **不改变角速度和角加速度**
- 旋转方向改变翅膀下拍/上拍的先后顺序和角速度分布（急回特性）

**返回**：`(t, phi, phi_dot, phi_ddot, info)`

| 返回值 | 形状 | 说明 |
|---|---|---|
| `t` | (n,) | 时间 [s]，0 → T = 1/f |
| `phi` | (n,) | 翅膀拍动角 [rad]，向上为正 |
| `phi_dot` | (n,) | 角速度 [rad/s] |
| `phi_ddot` | (n,) | 角加速度 [rad/s²] |
| `info` | dict | 机构信息（含频率、参数 a、旋转方向、角度范围、峰值角速度/加速度等） |

`info` 字典关键字段：

```python
{
    'f_Hz': 15.0,                    # 频率
    'T_s': 0.0667,                   # 周期
    'params': {...},                  # 完整参数
    'a': 7.92,                       # 实际使用的 a
    'rotation': 'cw',                # 旋转方向
    'phi_offset_deg': -50.84,        # 角度偏移（默认使 a=7.92 关于 0° 对称）
    'phi_range_deg': (-22.2, 22.2),  # 角度范围 [°]
    'phi_span_deg': 44.5,            # 总摆幅 [°]
    'phi_dot_max_rad_s': 44.2,       # 峰值角速度
    'phi_ddot_max_rad_s2': 5730.0,   # 峰值角加速度
}
```

---

## 5. 使用示例

### 5.1 基本调用

```python
from mechanism import wing_kinematics

t, phi, phi_dot, phi_ddot, info = wing_kinematics(f=15.0)
print(f"角度范围: {info['phi_range_deg']}")
print(f"峰值角速度: {info['phi_dot_max_rad_s']:.2f} rad/s")
```

### 5.2 默认调用（自动应用对称偏移）

```python
t, phi, phi_dot, phi_ddot, info = wing_kinematics(f=15.0)
# 默认使用 DEFAULT_PARAMS['phi_offset_deg'] = -50.84°
# 输出范围约 [-22.2°, 22.2°]，关于 0° 水平对称
```

### 5.3 显式关闭偏移（查看原始机构输出）

```python
t, phi, phi_dot, phi_ddot, info = wing_kinematics(
    f=15.0, phi_offset_deg=0)
# 输出范围约 [28.6°, 73.1°]，原始机构角度（全在上拍区）
```

### 5.3 参数扫描

```python
for a_val in [6.0, 7.0, 7.92, 9.0, 10.0, 11.0]:
    _, _, _, _, info = wing_kinematics(f=15.0, a=a_val)
    print(f"a={a_val}: span={info['phi_span_deg']:.1f}°")
```

---

## 6. 与气动分析的衔接

`dynamic_analysis.py` 调用 `wing_kinematics` 获取运动学数据后，进行准定常气动力计算：

1. **运动学输入**：`phi(t)`, `phi_dot(t)`, `phi_ddot(t)`
2. **攻角 α**：由翅膀安装角决定（当前为固定值）
3. **升力/阻力方向**：由 `phi_dot` 符号决定（下拍 vs 上拍）
4. **附加质量力**：与 `phi_ddot` 成正比

**关键设计决策**：
- 所有气动计算必须使用 `mechanism.py` 的实际输出，不得使用正弦假设
- `phi_offset_deg` 用于补偿翅膀折弯角度，使机构输出关于水平位置对称
- 固定攻角 α = 45° 时，由于下拍正升力与上拍负升力几乎抵消，净升重比极低（~0.1），需通过攻角主动控制或旋转力机制改善

---

## 7. 与 MATLAB 脚本的关系

原 MATLAB 脚本 `angularvelocity2dimensions.m` 使用直线-圆交点模型（将连杆约束简化为直线方程），而新 Python 模型使用完整的四连杆几何。

**符号约定**：
- MATLAB：`phi = atan2(y_intersect, x_intersect)`（无负号）
- 旧 Python 代码：`-np.arctan2(ry, rx)`（符号相反，**已修正**）
- 新 Python 代码：`phi = theta_AP1 + angle`（与 MATLAB 同向）

**坐标系修正历史**：
- 旧模型将 B 点放在 (2b, 0)，新模型修正为 (b, 0)
- 旧模型将 A 点放在 (-b, a)，新模型修正为 (0, a)

---

## 8. 文件位置

| 文件 | 路径 | 说明 |
|---|---|---|
| 运动学模块 | `src/mechanism.py` | 核心求解器 |
| 可视化工具 | `src/mechanism_plot.py` | 机构轨迹、运动学曲线、a 扫描 |
| 气动分析 | `src/dynamic_analysis.py` | 调用 wing_kinematics，输出气动力报告 |
| MATLAB 参考 | `angularvelocity2dimensions.m` | 原始参考实现（直线-圆模型） |
