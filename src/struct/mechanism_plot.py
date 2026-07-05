#!/usr/bin/env python3
"""
前置机构运动学可视化工具（参数来源：config/design_v69.yaml → mechanism.DEFAULT_PARAMS）

功能：
1. 机构几何示意图
2. 顺/逆时针（cw/ccw）一个周期内的角度、角速度、角加速度对比

用法:
    python src/struct/mechanism_plot.py                    # 默认 f=17 Hz, a=7.6
    python src/struct/mechanism_plot.py --a 9.0            # 指定 a
    python src/struct/mechanism_plot.py --freq 15          # 指定频率
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

    # ---- 获取 cw / ccw 运动学数据 ----
    kin_data = {}
    for rot in ['cw', 'ccw']:
        t, phi, phi_dot, phi_ddot, info = wing_kinematics(
            f=f, a=params['a'], rotation=rot,
            phi_offset_deg=phi_offset_deg, params=params, n_points=2000)
        kin_data[rot] = {
            't': t, 'phi': phi, 'phi_dot': phi_dot,
            'phi_ddot': phi_ddot, 'info': info
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
    ax2.set_title(f'φ(t) @ {f} Hz (cw vs ccw)')
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


def main():
    design = get_design()
    default_freq = design["f"]

    parser = argparse.ArgumentParser(description="Mechanism kinematics visualization")
    parser.add_argument("--a", type=float, default=DEFAULT_PARAMS['a'],
                        help="Parameter a (wing hinge y-coordinate)")
    parser.add_argument("--freq", type=float, default=default_freq,
                        help=f"Flapping frequency in Hz (default: {default_freq})")
    parser.add_argument("--phi-offset", type=float, default=None,
                        help="Fixed angle offset (deg). "
                             "Default uses mechanism.py DEFAULT_PARAMS['phi_offset_deg'].")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    params = {**DEFAULT_PARAMS, 'a': args.a}

    effective_offset = args.phi_offset
    if effective_offset is None:
        effective_offset = DEFAULT_PARAMS.get('phi_offset_deg', 0.0)

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

    print(f"\n运动学分析 @ {args.freq} Hz:")
    for rot in ['cw', 'ccw']:
        _, phi, phi_dot, phi_ddot, info = wing_kinematics(
            f=args.freq, a=args.a, rotation=rot,
            phi_offset_deg=args.phi_offset, params=params)
        span = info['phi_span_deg']
        peak_vel = info['phi_dot_max_rad_s']
        peak_acc = info['phi_ddot_max_rad_s2']
        print(f"  rotation='{rot}', offset={effective_offset:.2f}°: "
              f"span={span:.2f}°, |φ̇|max={peak_vel:.2f}, |φ̈|max={peak_acc:.2f}")

    if not args.no_plot:
        plot_all(params, geo, valid, args.freq, effective_offset)

    print("\nDone!")


if __name__ == "__main__":
    main()
