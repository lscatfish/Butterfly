#!/usr/bin/env python3
"""
前置机构运动学模块（曲柄摇杆四连杆机构 → 翅膀拍动角）

机构几何（参见 SolidWorks 装配图 angularvelocity2dimensions.m）：
  ├─ 固定点 A(0, a)：翅膀转轴（摇杆支点）
  ├─ 固定点 B(b, 0)：曲柄圆心（crank 支点）
  ├─ 曲柄 BP1：长度 R = 2.25，绕 B 顺时针匀速转动
  ├─ 连杆 P1P2：长度 c = 14.00
  └─ 摇杆 AP2：长度 l = 8.00，即翅膀杆

  翅膀转角 φ 定义为向量 A→P2 与 +x 轴的夹角，
  向上为正，向下为负。

坐标系定义：
  以过 A 的竖直虚线为 y 轴，以过 B（圆心）的水平轴为 x 轴
  原点 O = (0, 0) 在两轴交点
  x 轴水平向右，y 轴垂直向上。
  因此：A = (0, a)，B = (b, 0)

参数对应（与原 .m 文件一致）：
  a  = 7.92   → 翅膀转轴 A 的 y 坐标
  b  = 6.97   → 圆心 B 的 x 坐标
  R  = 2.25   → 曲柄半径（主点圆 Ø4.50 的一半）
  c  = 14.00  → 连杆长度
  l  = 8.00   → 摇杆/翅膀杆长度
"""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ==================== 机构几何参数 ====================
# 所有长度单位与 SolidWorks / .m 文件一致（mm）
DEFAULT_PARAMS = {
    "a": 7.92,     # 翅膀转轴 A 的 y 坐标
    "b": 6.97,     # 圆心 B 的 x 坐标
    "R": 2.25,     # 曲柄半径（主点圆直径 4.50 的一半）
    "c": 14.00,    # 连杆 P1-P2 长度
    "l": 8.00,     # 摇杆/翅膀杆 A-P2 长度
}


def solve_phi(theta: float, params: dict) -> float:
    """
    给定曲柄角 θ，求解翅膀转角 φ。

    几何关系
    --------
      P1 = B + R*(cosθ, sinθ)        # 主点（曲柄端点）
      P2 满足 |P2 - A| = l 且 |P2 - P1| = c

    求解方法：三角形余弦定理
      在 ΔA-P1-P2 中，已知三边：
        AP2 = l,   P1P2 = c,   AP1 = d(θ) = |P1 - A|
      由余弦定理：
        cos(∠P2AP1) = (l² + d² - c²) / (2*l*d)
      向量 A→P1 的方向角：θ_AP1 = atan2(P1y - a, P1x)
      翅膀角 φ = θ_AP1 + ∠P2AP1          （取上方/外侧交点，使机构连续）

    参数
    ----
    theta : float
        曲柄角 [rad]。顺时针旋转对应 θ 从 0 向 -2π 变化。
    params : dict
        机构参数 dict，需包含 a, b, R, c, l。

    返回
    ----
    phi : float
        翅膀转角 [rad]，向上为正。若机构无解返回 np.nan。
    """
    a = params['a']
    b = params['b']
    R = params['R']
    c_len = params['c']   # 避免与数学常数 e 混淆，本地变量用 c_len
    l_len = params['l']

    # 固定点坐标（按坐标系定义：A 在 y 轴上，B 在 x 轴上）
    A = np.array([0.0, a])
    B = np.array([b, 0.0])

    # 主点 P1（曲柄端点）
    P1 = B + R * np.array([np.cos(theta), np.sin(theta)])

    # 向量 A → P1
    AP1 = P1 - A
    d = np.linalg.norm(AP1)

    # 机构可达性检查（Grashof 条件已满足时通常不会触发）
    min_reach = abs(c_len - l_len)
    max_reach = c_len + l_len
    if d < min_reach - 1e-6 or d > max_reach + 1e-6:
        return np.nan

    # 向量 A→P1 的方向角
    theta_AP1 = np.arctan2(AP1[1], AP1[0])

    # 余弦定理求夹角 ∠P2AP1
    cos_angle = (l_len**2 + d**2 - c_len**2) / (2.0 * l_len * d)
    # 数值截断，防止浮点误差导致 |cos| > 1
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # 选择上方（外侧）交点，使得 P2 位于 AP1 的逆时针方向。
    # 这样在整个顺时针曲柄运动过程中，φ 连续变化，无需分支切换。
    phi = theta_AP1 + angle

    return phi


def wing_kinematics(f: float, params: dict = None, n_points: int = 2000):
    """
    生成翅膀运动学。

    曲柄顺时针匀速转动，转一圈 = 翅膀一个完整拍动周期。

    参数
    ----
    f : float
        振翅频率 [Hz]。曲柄转一圈对应翅膀一拍。
    params : dict
        机构参数（可部分覆盖），None 则用 DEFAULT_PARAMS。
    n_points : int
        每周期时间采样点数。

    返回
    ----
    t : (n,) ndarray
        时间 [s]，0 → T = 1/f
    phi : (n,) ndarray
        翅膀拍动角 [rad]，向上为正
    phi_dot : (n,) ndarray
        角速度 [rad/s]
    phi_ddot : (n,) ndarray
        角加速度 [rad/s²]
    info : dict
        机构信息（角度范围、摆幅、峰值角速度/加速度等）
    """
    full_params = DEFAULT_PARAMS.copy()
    if params is not None:
        full_params.update(params)
    p = full_params

    T = 1.0 / f
    t = np.linspace(0, T, n_points)
    dt = t[1] - t[0]

    # 曲柄顺时针旋转：θ 从 0 线性减小到 -2π（一个周期）
    theta = np.linspace(0, -2.0 * np.pi, n_points, endpoint=False)

    # 逐点求解每个曲柄角对应的翅膀角
    phi = np.array([solve_phi(th, p) for th in theta])

    # 处理可能的无效解（机构死点或参数越界）
    valid = ~np.isnan(phi)
    if not np.all(valid):
        n_invalid = np.sum(~valid)
        print(f"Warning: {n_invalid}/{n_points} 个位置机构无解（参数可能需要调整）")
        if np.any(valid):
            valid_idx = np.where(valid)[0]
            phi = np.interp(
                np.arange(n_points), valid_idx, phi[valid_idx], period=n_points
            )

    # 解缠绕：消除 ±2π 跳变，保证连续微分正确
    phi = np.unwrap(phi)

    # 时间微分得到角速度和角加速度
    phi_dot = np.gradient(phi, dt)
    phi_ddot = np.gradient(phi_dot, dt)

    info = {
        'f_Hz': f,
        'T_s': T,
        'params': p,
        'phi_range_rad': (float(np.min(phi)), float(np.max(phi))),
        'phi_range_deg': (float(np.rad2deg(np.min(phi))), float(np.rad2deg(np.max(phi)))),
        'phi_span_deg': float(np.rad2deg(np.max(phi) - np.min(phi))),
        'phi_dot_max_rad_s': float(np.max(np.abs(phi_dot))),
        'phi_ddot_max_rad_s2': float(np.max(np.abs(phi_ddot))),
        'n_points': n_points,
    }

    return t, phi, phi_dot, phi_ddot, info


# ==================== 自测 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("前置曲柄摇杆机构运动学自测")
    print("=" * 60)

    f = 15.0  # Hz
    t, phi, phi_dot, phi_ddot, info = wing_kinematics(f=f)

    phi_min, phi_max = info['phi_range_deg']

    print(f"\n--- 基本参数 ---")
    print(f"  频率 f      = {info['f_Hz']:.1f} Hz")
    print(f"  周期 T      = {info['T_s']*1000:.2f} ms")
    print(f"  采样点数    = {info['n_points']}")

    print(f"\n--- 翅膀运动范围 ---")
    print(f"  φ_min       = {phi_min:.2f}°")
    print(f"  φ_max       = {phi_max:.2f}°")
    print(f"  总摆幅      = {info['phi_span_deg']:.2f}°")

    print(f"\n--- 角速度 / 角加速度 ---")
    print(f"  |φ̇|_max     = {info['phi_dot_max_rad_s']:.2f} rad/s")
    print(f"  |φ̈|_max     = {info['phi_ddot_max_rad_s2']:.2f} rad/s²")

    # 统计 phi>0 / phi<0 的时间占比（检验正负对称性）
    print(f"\n--- 上下拍时间占比（以 φ=0° 为界） ---")
    print(f"  φ > 0 (上拍侧)  : {np.mean(phi > 0)*100:.1f}%")
    print(f"  φ < 0 (下拍侧)  : {np.mean(phi < 0)*100:.1f}%")
    print(f"  φ = 0           : {np.mean(np.isclose(phi, 0, atol=1e-6))*100:.1f}%")

    print(f"\n--- 角速度符号分布（以 φ̇=0 为界） ---")
    print(f"  φ̇ > 0           : {np.mean(phi_dot > 0)*100:.1f}%")
    print(f"  φ̇ < 0           : {np.mean(phi_dot < 0)*100:.1f}%")

    print("\nDone!")
