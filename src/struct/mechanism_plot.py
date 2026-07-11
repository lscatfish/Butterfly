#!/usr/bin/env python3
"""
前置机构运动学可视化工具（参数来源：config/design_v69.yaml → mechanism.DEFAULT_PARAMS）

功能：
1. 机构几何示意图
2. 顺/逆时针（cw/ccw）两个周期内的角度、角速度、角加速度对比

用法:
    python src/struct/mechanism_plot.py                 # 生成图
    python src/struct/mechanism_plot.py --no-plot       # 仅打印数值
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "output" / "figures" / "mechanism"

from src.struct.mechanism import DEFAULT_PARAMS, solve_phi, wing_kinematics
from src.config import get_design


def analyze_geometry(params: dict, n_frames: int = 360):
    """纯几何分析（与频率无关）：扫描曲柄一整圈求角度范围"""
    theta = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    phi = np.array([solve_phi(th, params) for th in theta])
    valid = ~np.isnan(phi)
    phi_valid = phi[valid]

    info = {
        'phi_min_rad': float(np.min(phi_valid)) if np.any(valid) else np.nan,
        'phi_max_rad': float(np.max(phi_valid)) if np.any(valid) else np.nan,
        'span_deg': float(np.rad2deg(np.max(phi_valid) - np.min(phi_valid))) if np.any(valid) else 0.0,
        'n_valid': int(np.sum(valid)),
        'theta': theta,
        'phi': phi,
    }
    return info, valid


def get_p2_coords(phi, params):
    """由翅膀角 phi 反求摇杆端点 P2 坐标"""
    a = params['a']
    l = params['l']
    return np.array([l * np.cos(phi), a + l * np.sin(phi)])


def get_p1_coords(theta, params):
    """由曲柄角 theta 求曲柄端点 P1 坐标"""
    b = params['b']
    R = params['R']
    return np.array([b + R * np.cos(theta), R * np.sin(theta)])


def plot_mechanism_geometry(ax, params, n_poses=8):
    """绘制机构几何示意图（固定点、轨迹圆、若干姿态）"""
    a, b, R, c, l = params['a'], params['b'], params['R'], params['c'], params['l']

    # 固定点
    ax.plot(0, a, 'ko', ms=8, label='A (hinge)')
    ax.plot(b, 0, 'bs', ms=8, label='B (crank center)')

    # 摇杆可达圆（以 A 为中心）
    theta_c = np.linspace(0, 2 * np.pi, 100)
    ax.plot(l * np.cos(theta_c), a + l * np.sin(theta_c),
            'g--', lw=1, alpha=0.5, label=f'Swing circle r={l}')

    # 曲柄圆（以 B 为中心）
    ax.plot(b + R * np.cos(theta_c), R * np.sin(theta_c),
            'b--', lw=1, alpha=0.5, label=f'Crank circle R={R}')

    # 画几个机构姿态（曲柄顺时针方向）
    thetas = np.linspace(0, -2 * np.pi, n_poses, endpoint=False)
    for th in thetas:
        phi = solve_phi(th, params)
        if np.isnan(phi):
            continue
        P1 = get_p1_coords(th, params)
        P2 = get_p2_coords(phi, params)
        # 连杆 P1-P2
        ax.plot([P1[0], P2[0]], [P1[1], P2[1]], 'm-', lw=1, alpha=0.4)
        # 摇杆 A-P2
        ax.plot([0, P2[0]], [a, P2[1]], 'g-', lw=1, alpha=0.4)
        # 曲柄 B-P1
        ax.plot([b, P1[0]], [0, P1[1]], 'b-', lw=1, alpha=0.4)
        ax.plot(P1[0], P1[1], 'b.', ms=4)
        ax.plot(P2[0], P2[1], 'g.', ms=4)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title('Mechanism Geometry')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')


def plot_all(params, geo, valid, f, phi_offset_deg):
    """统一绘图：4 个子图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting skipped: {e}", file=sys.stderr)
        return

    fig = plt.figure(figsize=(14, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    colors_rot = {'cw': '#1f77b4', 'ccw': '#d62728'}

    # ---- 1. 机构几何示意图 ----
    ax1 = fig.add_subplot(2, 2, 1)
    plot_mechanism_geometry(ax1, params)

    # ---- 获取 cw / ccw 运动学数据（2 个周期） ----
    kin_data = {}
    for rot in ['cw', 'ccw']:
        t1, phi1, phi_dot1, phi_ddot1, info = wing_kinematics(
            f=f, a=params['a'], rotation=rot,
            phi_offset_deg=phi_offset_deg, params=params, n_points=2000)
        T = 1.0 / f
        kin_data[rot] = {
            't': np.concatenate([t1, t1 + T]),
            'phi': np.concatenate([phi1, phi1]),
            'phi_dot': np.concatenate([phi_dot1, phi_dot1]),
            'phi_ddot': np.concatenate([phi_ddot1, phi_ddot1]),
            'info': info
        }

    t_ms = lambda t: t * 1000

    # ---- 2. cw vs ccw: 角度 φ(t) ----
    ax2 = fig.add_subplot(2, 2, 2)
    for rot in ['cw', 'ccw']:
        d = kin_data[rot]
        rng = d['info']['phi_range_deg']
        ax2.plot(t_ms(d['t']), np.rad2deg(d['phi']),
                 color=colors_rot[rot], lw=1.5,
                 label=f"{rot} [{rng[0]:.1f}°, {rng[1]:.1f}°]")
    ax2.axhline(0, color='gray', ls='--', lw=0.8)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Stroke angle φ (°)')
    ax2.set_title(f'φ(t) — 2 periods @ {f} Hz')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- 3. cw vs ccw: 角速度 φ̇(t) ----
    ax3 = fig.add_subplot(2, 2, 3)
    for rot in ['cw', 'ccw']:
        d = kin_data[rot]
        peak = d['info']['phi_dot_max_rad_s']
        ax3.plot(t_ms(d['t']), d['phi_dot'],
                 color=colors_rot[rot], lw=1.5,
                 label=f"{rot} (peak={peak:.1f})")
    ax3.axhline(0, color='gray', ls='--', lw=0.8)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Angular velocity (rad/s)')
    ax3.set_title(f'Angular velocity @ {f} Hz')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- 4. cw vs ccw: 角加速度 φ̈(t) ----
    ax4 = fig.add_subplot(2, 2, 4)
    for rot in ['cw', 'ccw']:
        d = kin_data[rot]
        peak = d['info']['phi_ddot_max_rad_s2']
        ax4.plot(t_ms(d['t']), d['phi_ddot'],
                 color=colors_rot[rot], lw=1.5,
                 label=f"{rot} (peak={peak:.1f})")
    ax4.axhline(0, color='gray', ls='--', lw=0.8)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel(r'Angular acceleration ($rad/s^2$)')
    ax4.set_title(f'Angular acceleration @ {f} Hz')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    offset_str = f", offset={phi_offset_deg}°" if phi_offset_deg is not None else ""
    fig.suptitle(f'Mechanism Kinematics: f={f} Hz, a={params["a"]}{offset_str}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "mechanism_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def plot_rocker_kinematics(params, f, phi_offset_deg):
    """摇杆（输出杆件）运动学分析：顺时针/逆时针各一张图，角量与线量分轴"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting skipped: {e}", file=sys.stderr)
        return

    l_m = params['l'] / 1000.0
    T = 1.0 / f
    t_ms = lambda t: t * 1000

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    rotations = {
        'cw':  ('顺时针 (cw)', '#2878B5'),
        'ccw': ('逆时针 (ccw)', '#C82423'),
    }

    for rot, (rot_label, rot_color) in rotations.items():
        t1, phi1, phi_dot1, phi_ddot1, info = wing_kinematics(
            f=f, a=params['a'], rotation=rot,
            phi_offset_deg=phi_offset_deg, params=params, n_points=2000)

        t2 = np.concatenate([t1, t1 + T])
        phi_dot2 = np.concatenate([phi_dot1, phi_dot1])
        phi_ddot2 = np.concatenate([phi_ddot1, phi_ddot1])
        v2 = l_m * np.abs(phi_dot2)
        a2 = l_m * np.sqrt(phi_ddot2**2 + phi_dot2**4)

        t = t_ms(t2)
        peak_phi_dot = float(np.max(np.abs(phi_dot2)))
        peak_phi_ddot = float(np.max(np.abs(phi_ddot2)))
        peak_v = float(np.max(v2))
        peak_a = float(np.max(a2))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

        c_phi_dot = '#3498DB'   # 角速度: 蓝
        c_phi_ddot = '#E74C3C'  # 角加速度: 红
        c_v = '#27AE60'         # 线速度: 绿
        c_a = '#F39C12'         # 线加速度: 橙

        # ---- 子图1: 角量 (双 y 轴) ----
        line1, = ax1.plot(t, phi_dot2, color=c_phi_dot, lw=1.8,
                          label=f'角速度 (峰值 {peak_phi_dot:.1f} rad/s)')
        ax1.set_ylabel('角速度 (rad/s)', color=c_phi_dot)
        ax1.tick_params(axis='y', labelcolor=c_phi_dot)
        ax1.set_xlabel('时间 (ms)')

        ax1b = ax1.twinx()
        line2, = ax1b.plot(t, phi_ddot2, color=c_phi_ddot, lw=1.8,
                           label=f'角加速度 (峰值 {peak_phi_ddot:.0f} rad/s^2)')
        ax1b.set_ylabel('角加速度 (rad/s^2)', color=c_phi_ddot)
        ax1b.tick_params(axis='y', labelcolor=c_phi_ddot)

        lines1 = [line1, line2]
        labels1 = [l.get_label() for l in lines1]
        ax1.set_title(f'摇杆角量 — {rot_label}', fontsize=12)
        ax1.grid(True, alpha=0.2)
        ax1.axhline(0, color='gray', ls='--', lw=0.6)
        ax1.set_zorder(ax1b.get_zorder() + 1)
        ax1.patch.set_visible(False)
        ax1.legend(lines1, labels1, loc='lower right', fontsize=9, framealpha=0.9).set_zorder(10)

        # ---- 子图2: 线量 (双 y 轴) ----
        line3, = ax2.plot(t, v2, color=c_v, lw=1.8,
                          label=f'线速度 (峰值 {peak_v:.2f} m/s)')
        ax2.set_ylabel('线速度 (m/s)', color=c_v)
        ax2.tick_params(axis='y', labelcolor=c_v)
        ax2.set_xlabel('时间 (ms)')

        ax2b = ax2.twinx()
        line4, = ax2b.plot(t, a2, color=c_a, lw=1.8,
                           label=f'线加速度 (峰值 {peak_a:.1f} m/s^2)')
        ax2b.set_ylabel('线加速度 (m/s^2)', color=c_a)
        ax2b.tick_params(axis='y', labelcolor=c_a)

        lines2 = [line3, line4]
        labels2 = [l.get_label() for l in lines2]
        ax2.set_title(f'端点 P2 线量 (摇杆长 l={params["l"]:.1f} mm) — {rot_label}', fontsize=12)
        ax2.grid(True, alpha=0.2)
        ax2.set_zorder(ax2b.get_zorder() + 1)
        ax2.patch.set_visible(False)
        ax2.legend(lines2, labels2, loc='lower right', fontsize=9, framealpha=0.9).set_zorder(10)

        fig.suptitle(f'摇杆运动学分析  (f={f} Hz, a={params["a"]} mm) — {rot_label}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = f'摇杆运动学分析-{rot}.svg'
        out_path = OUTPUT_DIR / fname
        plt.savefig(out_path, format='svg', bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    design = get_design()
    f = design["f"]
    params = dict(DEFAULT_PARAMS)
    phi_offset_deg = params.get('phi_offset_deg', 0.0)

    parser = argparse.ArgumentParser(description="Mechanism kinematics visualization")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    geo, valid = analyze_geometry(params)
    print("=" * 60)
    print("机构几何分析（纯几何，与频率无关）")
    print("=" * 60)
    print(f"  a={params['a']}, b={params['b']}, R={params['R']}, "
          f"c={params['c']}, l={params['l']}")
    print(f"  摆幅: {geo['span_deg']:.2f}°")
    print(f"  角度范围: [{np.rad2deg(geo['phi_min_rad']):.2f}, "
          f"{np.rad2deg(geo['phi_max_rad']):.2f}]°")
    print(f"  有效帧: {geo['n_valid']} / {len(valid)}")

    print(f"\n运动学分析 @ {f} Hz:")
    for rot in ['cw', 'ccw']:
        _, phi, phi_dot, phi_ddot, info = wing_kinematics(
            f=f, a=params['a'], rotation=rot,
            phi_offset_deg=phi_offset_deg, params=params)
        span = info['phi_span_deg']
        peak_vel = info['phi_dot_max_rad_s']
        peak_acc = info['phi_ddot_max_rad_s2']
        print(f"  rotation='{rot}', offset={phi_offset_deg:.2f}°: "
              f"span={span:.2f}°, |φ̇|max={peak_vel:.2f}, |φ̈|max={peak_acc:.2f}")

    if not args.no_plot:
        plot_all(params, geo, valid, f, phi_offset_deg)
        plot_rocker_kinematics(params, f, phi_offset_deg)

    print("\nDone!")


if __name__ == "__main__":
    main()
