#!/usr/bin/env python3
"""
前置机构运动学模块（crank-rocker 平面连杆机构 → 翅膀拍动角）

机构原理：
  主点 P1 绕偏心圆 (x-a)²+(y-b)²=R² 匀速转动（曲柄一周 = 翅膀一拍）
  动态直线过 P1 且满足 A·x+B·y=C，与固定圆 x²+y²=l² 相交
  取 x 最大交点 P_int，其极角 φ = atan2(y_int, x_int) 即为翅膀转角

用法:
    from mechanism import wing_kinematics, DEFAULT_PARAMS

    t, phi, phi_dot, phi_ddot = wing_kinematics(f=17.5, R=2.5)
    # 返回一个振翅周期的运动学数组
"""

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DEFAULT_PARAMS = {
    "a": 8.0,        # 主点轨迹圆圆心 x
    "b": 6.0,        # 主点轨迹圆圆心 y
    "R": 2.25,       # 主点轨迹圆半径（控制摆幅和运动不对称性）
    "c": 14.0,       # 直线方程常数参数
    "l": 8.0,        # 固定圆 x²+y²=l² 半径
}


def solve_intersection(A: float, B: float, C: float, l: float):
    """直线 A·x + B·y = C 与圆 x² + y² = l² 的实数交点"""
    if abs(A) < 1e-10 and abs(B) < 1e-10:
        return np.array([]), np.array([])

    if abs(B) > 1e-10:
        coeff = [A ** 2 + B ** 2, -2 * A * C, C ** 2 - B ** 2 * l ** 2]
        roots_x = np.roots(coeff)
        real_mask = np.abs(np.imag(roots_x)) < 1e-10
        real_x = np.real(roots_x[real_mask])
        real_y = (C - A * real_x) / B
    else:
        x0 = C / A
        y_sq = l ** 2 - x0 ** 2
        if y_sq >= -1e-10:
            y_sq = max(y_sq, 0.0)
            real_y = np.array([-np.sqrt(y_sq), np.sqrt(y_sq)])
            real_x = x0 * np.ones_like(real_y)
        else:
            real_x = np.array([])
            real_y = np.array([])

    return real_x, real_y


def mechanism_cycle(params: dict = None, n_points: int = 2000):
    """
    计算机构一个完整曲柄周期（theta: 0 → 2π）的运动学

    参数
    ----
    params : 机构参数 dict（可部分覆盖），None 用默认值
    n_points : 采样点数

    返回 dict: crank_theta, phi_stroke, phi_raw, x_intersect, y_intersect,
               span_rad, span_deg, phi_min_rad, phi_max_rad, n_valid
    """
    full_params = DEFAULT_PARAMS.copy()
    if params is not None:
        full_params.update(params)
    params = full_params

    a, b, R, c, l = params['a'], params['b'], params['R'], params['c'], params['l']

    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    x_intersect = np.full(n_points, np.nan)
    y_intersect = np.full(n_points, np.nan)
    phi_raw = np.full(n_points, np.nan)

    for i in range(n_points):
        th = theta[i]
        x1 = a + R * np.cos(th)
        y1 = b + R * np.sin(th)

        A, B = x1, y1
        C_val = (x1 ** 2 + y1 ** 2 - c) / 2.0

        rx, ry = solve_intersection(A, B, C_val, l)

        if rx.size > 0:
            idx = np.argmax(rx)
            x_intersect[i] = rx[idx]
            y_intersect[i] = ry[idx]
            phi_raw[i] = np.arctan2(ry[idx], rx[idx])

    valid = ~np.isnan(phi_raw)
    if valid.sum() < 2:
        raise RuntimeError("机构无有效交点，检查参数（特别是 R 不要太大）")

    # 解缠绕用于连续微分
    phi_unwrap = np.unwrap(phi_raw)
    # 原始振荡角（直接 atan2 取值，已正确处理 [-π,π] 区间）
    # 居中：使摆动围绕 0
    phi_center = 0.5 * (np.nanmin(phi_raw[valid]) + np.nanmax(phi_raw[valid]))
    phi_stroke = phi_raw - phi_center

    phi_min = np.nanmin(phi_stroke)
    phi_max = np.nanmax(phi_stroke)
    span = phi_max - phi_min

    return {
        'crank_theta': theta,
        'phi_stroke': phi_stroke,
        'phi_raw': phi_raw,
        'x_intersect': x_intersect,
        'y_intersect': y_intersect,
        'span_rad': span,
        'span_deg': np.rad2deg(span),
        'phi_min_rad': phi_min,
        'phi_max_rad': phi_max,
        'n_valid': int(valid.sum()),
    }


def wing_kinematics(f: float, params: dict = None, n_points: int = 2000,
                    phi_down_deg: float = None, phi_up_deg: float = None):
    """
    生成翅膀运动学：根据机构形状 + 目标频率 + 目标幅度

    机构决定 φ(t) 的波形（不对称性、加减速特征），再通过时间缩放和
    幅度缩放匹配设计要求的频率和拍动幅度。

    参数
    ----
    f : float
        振翅频率 [Hz]，一个完整周期对应曲柄转一圈
    params : dict
        机构参数（可只传部分 key，其余用默认值）；None 则全用默认
    n_points : int
        时间采样点数
    phi_down_deg : float
        目标下拍幅度 [deg]（向下），None 则不缩放幅度
    phi_up_deg : float
        目标上拍幅度 [deg]（向上），None 则不缩放幅度

    返回
    ----
    t : (n,)          时间 [s]，从 0 到 T = 1/f
    phi : (n,)         翅膀拍动角 [rad]，向上为正
    phi_dot : (n,)     角速度 [rad/s]
    phi_ddot : (n,)    角加速度 [rad/s²]
    info : dict        机构信息（跨度、缩放因子等）
    """
    full_params = DEFAULT_PARAMS.copy()
    if params is not None:
        full_params.update(params)
    params = full_params

    mech = mechanism_cycle(params, n_points)

    T = 1.0 / f
    t = np.linspace(0, T, n_points)

    # 填充 NaN（如果存在）用插值
    phi_stroke = mech['phi_stroke'].copy()
    nan_mask = np.isnan(phi_stroke)
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        phi_stroke[nan_mask] = np.interp(
            np.where(nan_mask)[0], valid_idx, phi_stroke[valid_idx],
            period=n_points)

    # 幅度缩放：保持机构波形的不对称特性，同时匹配目标幅值
    scale = 1.0
    offset = 0.0
    if phi_down_deg is not None and phi_up_deg is not None:
        raw_down = abs(mech['phi_min_rad'])
        raw_up = abs(mech['phi_max_rad'])
        target_down_rad = np.deg2rad(phi_down_deg)
        target_up_rad = np.deg2rad(phi_up_deg)
        target_span = target_down_rad + target_up_rad
        raw_span = mech['span_rad']
        if raw_span > 1e-6:
            scale = target_span / raw_span
        # 零点偏移：使拍动幅度不对称
        # 缩放后 down = scale * raw_down, up = scale * raw_up（仍保持机构原始比率）
        # 再平移中心使得缩放后的 [-scale*raw_down, +scale*raw_up] 变为 [-target_down, +target_up]
        scaled_down = scale * raw_down
        scaled_up = scale * raw_up
        offset = (target_up_rad - scaled_up)  # 向上移动中心
        phi_stroke = phi_stroke * scale + offset

    # 时间微分：phi_dot = dφ/dt（T 秒内走完一个机构周期）
    dt = t[1] - t[0]
    phi_dot = np.gradient(phi_stroke, dt)
    phi_ddot = np.gradient(phi_dot, dt)

    info = {
        'f_Hz': f,
        'T_s': T,
        'mechanism_params': params,
        'raw_span_deg': mech['span_deg'],
        'raw_phi_range_deg': (np.rad2deg(mech['phi_min_rad']), np.rad2deg(mech['phi_max_rad'])),
        'scale_factor': scale,
        'n_points': n_points,
    }

    if phi_down_deg is not None and phi_up_deg is not None:
        info['target_phi_deg'] = (phi_down_deg, phi_up_deg)
        eff_down = abs(np.min(phi_stroke))
        eff_up = abs(np.max(phi_stroke))
        info['effective_phi_deg'] = (np.rad2deg(eff_down), np.rad2deg(eff_up))

    return t, phi_stroke, phi_dot, phi_ddot, info


# ==================== self-test ====================
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 60)
    print("前置机构运动学测试")
    print("=" * 60)

    # 扫描不同 R
    for R in [2.0, 2.25, 3.0, 4.0, 5.0]:
        p = {**DEFAULT_PARAMS, 'R': R}
        try:
            mech = mechanism_cycle(p, n_points=720)
            print(f"R={R:.2f}: span={mech['span_deg']:.1f}°  "
                  f"range=[{np.rad2deg(mech['phi_min_rad']):.1f}, {np.rad2deg(mech['phi_max_rad']):.1f}]°  "
                  f"valid={mech['n_valid']}")
        except RuntimeError as e:
            print(f"R={R:.2f}: ERROR - {e}")

    # 生成翅膀运动学
    print("\n--- wing_kinematics ---")
    t, phi, phi_dot, phi_ddot, info = wing_kinematics(
        f=17.5, params={'R': 2.25}, phi_down_deg=80, phi_up_deg=60)
    print(f"f={info['f_Hz']} Hz, T={info['T_s']*1000:.1f} ms")
    print(f"Raw span={info['raw_span_deg']:.1f}°, scale={info['scale_factor']:.3f}")
    print(f"Effective: down={info['effective_phi_deg'][0]:.1f}°, up={info['effective_phi_deg'][1]:.1f}°")
    print(f"phi_dot_max={np.max(phi_dot):.1f} rad/s")
    print(f"phi_ddot_abs_max={np.max(np.abs(phi_ddot)):.1f} rad/s²")

    # 对比图
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Mechanism-Based Wing Kinematics (f=17.5 Hz, R=2.25)', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(t * 1000, np.rad2deg(phi), 'b-', lw=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Stroke angle φ (°)')
    ax.set_title('Wing Stroke Angle')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)

    ax = axes[0, 1]
    ax.plot(t * 1000, phi_dot, 'g-', lw=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Angular velocity (rad/s)')
    ax.set_title('Angular Velocity')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)

    ax = axes[1, 0]
    ax.plot(t * 1000, phi_ddot, 'r-', lw=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Angular acceleration (rad/s²)')
    ax.set_title('Angular Acceleration')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)

    ax = axes[1, 1]
    ax.plot(np.rad2deg(phi), phi_dot, 'm-', lw=2)
    ax.set_xlabel('Stroke angle φ (°)')
    ax.set_ylabel('Angular velocity (rad/s)')
    ax.set_title('Velocity vs Angle (Phase Portrait)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PROJECT_ROOT / 'output' / 'figures' / 'mechanism_kinematics.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()
    print("\nDone!")
