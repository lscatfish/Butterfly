#!/usr/bin/env python3
"""
前置机构运动学可视化工具
融合 analyze_matlab_constraint.py + plot_phi_t_various_R.py

功能：
1. 默认参数机构仿真（轨迹、极角、角速度、角加速度）
2. 主点半径 R 扫描（φ-t 曲线对比）
3. 基于 mechanism.py 统一计算，不再重复 solve_intersection

用法:
    python src/mechanism_plot.py                     # 默认参数分析 + R 扫描
    python src/mechanism_plot.py --R-list 2.25 3 4 5  # 自定义 R 扫描
    python src/mechanism_plot.py --a 8 --b 6 --l 8    # 修改机构参数
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "figures"

from mechanism import DEFAULT_PARAMS, mechanism_cycle


def analyze_default(params: dict, n_frames: int = 360, dt: float = 0.05):
    """默认参数机构分析（原 analyze_matlab_constraint 功能）"""
    mech = mechanism_cycle(params, n_points=n_frames)

    crank_theta = mech['crank_theta']
    phi_stroke = mech['phi_stroke']
    span_deg = mech['span_deg']
    valid = ~np.isnan(phi_stroke)

    # 角运动计算（dt 缩放对应物理时间）
    phi_unwrap = np.unwrap(mech['phi_raw'])
    omega = np.gradient(phi_unwrap, dt)
    alpha = np.gradient(omega, dt)
    time_arr = np.arange(n_frames) * dt

    # 统计
    print("=" * 60)
    print("机构运动学约束分析")
    print("=" * 60)
    print(f"  主点圆心: ({params['a']}, {params['b']})")
    print(f"  主点半径 R: {params['R']}")
    print(f"  固定圆半径 l: {params['l']}")
    print(f"  摆动幅度: {span_deg:.2f}°")
    print(f"  极角范围: [{np.rad2deg(mech['phi_min_rad']):.2f}, {np.rad2deg(mech['phi_max_rad']):.2f}]°")
    print(f"  omega 范围: [{np.min(omega):.3f}, {np.max(omega):.3f}] rad/s (at dt={dt}s)")
    print(f"  alpha 范围: [{np.min(alpha):.1f}, {np.max(alpha):.1f}] rad/s²")

    # 关键帧
    key_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4]
    key_labels = ["0°", "90°", "180°", "270°"]
    print("\n关键帧 (crank angle → phi, omega, alpha):")
    for idx, label in zip(key_indices, key_labels):
        if valid[idx]:
            print(f"  {label:>6}: phi={np.rad2deg(phi_stroke[idx]):7.2f}°  "
                  f"omega={omega[idx]:8.3f} rad/s  alpha={alpha[idx]:8.1f} rad/s²")
        else:
            print(f"  {label:>6}: (no valid intersection)")

    return mech, time_arr, omega, alpha, valid


def scan_R(params: dict, R_list: list, n_frames: int = 360, dt: float = 0.05):
    """扫描不同主点半径 R（原 plot_phi_t_various_R 功能）"""
    results = []
    print(f"\n{'R':>6} {'span':>10} {'phi_min':>10} {'phi_max':>10} {'valid':>8}")
    print("-" * 48)
    for R in R_list:
        p = {**params, 'R': R}
        try:
            mech = mechanism_cycle(p, n_points=n_frames)
            results.append((R, mech))
            print(f"{R:6.2f} {mech['span_deg']:10.1f} "
                  f"{np.rad2deg(mech['phi_min_rad']):10.1f} "
                  f"{np.rad2deg(mech['phi_max_rad']):10.1f} "
                  f"{mech['n_valid']:8d}")
        except RuntimeError as e:
            print(f"{R:6.2f} {'ERROR':>10} {str(e)[:50]}")
    return results


def plot_all(params, mech, time_arr, omega, alpha, valid, R_results, dt):
    """统一绘图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting skipped: {e}", file=sys.stderr)
        return

    fig = plt.figure(figsize=(16, 12))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    colors_r = ["k", "b", "g", "r", "m", "orange", "c", "purple"]

    # ---- 1. 机构轨迹 ----
    ax1 = fig.add_subplot(2, 3, 1)
    theta_c = np.linspace(0, 2 * np.pi, 360)
    l_val = params['l']
    ax1.plot(l_val * np.cos(theta_c), l_val * np.sin(theta_c), 'g--', lw=1, label=f'Circle r={l_val}')
    x1 = params['a'] + params['R'] * np.cos(mech['crank_theta'])
    y1 = params['b'] + params['R'] * np.sin(mech['crank_theta'])
    ax1.plot(x1, y1, 'b-', lw=1, label='Master point')
    ax1.plot(mech['x_intersect'][valid], mech['y_intersect'][valid], 'm:', lw=1, label='Intersection')
    ax1.plot(x1[0], y1[0], 'bo', ms=6)
    ax1.plot(mech['x_intersect'][0], mech['y_intersect'][0], 'ms', ms=5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7)
    ax1.set_title(f'Mechanism Trajectory (R={params["R"]})')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')

    # ---- 2. 极角 ----
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(time_arr, np.rad2deg(mech['phi_stroke']), 'k-', lw=1.5)
    ax2.axhline(0, color='gray', ls='--', lw=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Stroke angle φ (°)')
    ax2.set_title(f'Wing Stroke Angle (span={mech["span_deg"]:.1f}°)')
    ax2.grid(True, alpha=0.3)

    # ---- 3. 角速度 ----
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(time_arr, omega, 'b-', lw=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angular velocity (rad/s)')
    ax3.set_title('Angular Velocity (at dt={}s)'.format(dt))
    ax3.grid(True, alpha=0.3)

    # ---- 4. 角加速度 ----
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(time_arr, alpha, 'r-', lw=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angular acceleration (rad/s²)')
    ax4.set_title('Angular Acceleration')
    ax4.grid(True, alpha=0.3)

    # ---- 5. R 扫描叠加 ----
    ax5 = fig.add_subplot(2, 3, 5)
    for idx, (R_val, mech_r) in enumerate(R_results):
        color = colors_r[idx % len(colors_r)]
        t_r = np.arange(mech_r['crank_theta'].size) * dt
        phi_r = mech_r['phi_stroke']
        ax5.plot(t_r, np.rad2deg(phi_r), color=color, lw=1.5, label=f'R={R_val}')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Stroke angle φ (°)')
    ax5.set_title('φ vs time for various R')
    ax5.legend(fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.3)

    # ---- 6. span vs R ----
    ax6 = fig.add_subplot(2, 3, 6)
    R_vals = [r[0] for r in R_results]
    spans = [r[1]['span_deg'] for r in R_results]
    ax6.plot(R_vals, spans, 'o-', color='#d62728', lw=2, ms=6)
    ax6.set_xlabel('R')
    ax6.set_ylabel('Span (°)')
    ax6.set_title('Stroke Amplitude vs R')
    ax6.grid(True, alpha=0.3)

    fig.suptitle(f'Mechanism Kinematics: a={params["a"]}, b={params["b"]}, c={params["c"]}, l={params["l"]}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "mechanism_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Mechanism kinematics visualization")
    parser.add_argument("--a", type=float, default=DEFAULT_PARAMS['a'])
    parser.add_argument("--b", type=float, default=DEFAULT_PARAMS['b'])
    parser.add_argument("--R", type=float, default=DEFAULT_PARAMS['R'])
    parser.add_argument("--c", type=float, default=DEFAULT_PARAMS['c'])
    parser.add_argument("--l", type=float, default=DEFAULT_PARAMS['l'])
    parser.add_argument("--n-frames", type=int, default=360, help="Frames per crank revolution")
    parser.add_argument("--dt", type=float, default=0.05, help="Frame interval (s)")
    parser.add_argument("--R-list", type=float, nargs="+",
                        default=[2.0, 2.25, 3.0, 4.0, 5.0, 6.0],
                        help="R values to scan")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    params = {'a': args.a, 'b': args.b, 'R': args.R, 'c': args.c, 'l': args.l}

    # 1. 默认参数分析
    mech, time_arr, omega, alpha, valid = analyze_default(params, args.n_frames, args.dt)

    # 2. R 扫描
    R_results = scan_R(params, args.R_list, args.n_frames, args.dt)

    if not args.no_plot:
        plot_all(params, mech, time_arr, omega, alpha, valid, R_results, args.dt)

    print("\nDone!")


if __name__ == "__main__":
    main()
