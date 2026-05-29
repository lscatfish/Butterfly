#!/usr/bin/env python3
"""
前置机构运动学可视化工具

功能：
1. 机构几何轨迹（纯几何，与频率无关）
2. 翅膀运动学（φ, φ̇, φ̈ @ target frequency, 由 wing_kinematics 生成）
3. 主点圆心 x 坐标 a 扫描（φ-t 曲线 + span vs a）

用法:
    python src/mechanism_plot.py                          # 默认 f=17.5 Hz, a=8
    python src/mechanism_plot.py --freq 15 --a 10         # f=15 Hz, a=10
    python src/mechanism_plot.py --a-list 6 8 10 12       # 自定义 a 列表
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "figures"

from mechanism import DEFAULT_PARAMS, mechanism_cycle, wing_kinematics


def analyze_default(params: dict, n_frames: int = 360):
    """机构几何统计（与频率无关）"""
    mech = mechanism_cycle(params, n_points=n_frames)
    valid = ~np.isnan(mech['phi_stroke'])

    print("=" * 60)
    print("机构运动学约束分析（纯几何）")
    print("=" * 60)
    print(f"  主点圆心: ({params['a']}, {params['b']})")
    print(f"  主点半径 R: {params['R']}    固定圆半径 l: {params['l']}")
    print(f"  摆动幅度: {mech['span_deg']:.2f}°")
    print(f"  极角范围: [{np.rad2deg(mech['phi_min_rad']):.2f}, "
          f"{np.rad2deg(mech['phi_max_rad']):.2f}]°")
    print(f"  有效帧数: {mech['n_valid']} / {n_frames}")

    return mech, valid


def scan_a(params: dict, a_list: list, n_frames: int = 360):
    """扫描不同主点圆心 x 坐标 a（几何跨度）"""
    results = []
    print(f"\n{'a':>6} {'span':>10} {'phi_min':>10} {'phi_max':>10} {'valid':>8}")
    print("-" * 48)
    for a_val in a_list:
        p = {**params, 'a': a_val}
        try:
            mech = mechanism_cycle(p, n_points=n_frames)
            results.append((a_val, mech))
            print(f"{a_val:6.1f} {mech['span_deg']:10.1f} "
                  f"{np.rad2deg(mech['phi_min_rad']):10.1f} "
                  f"{np.rad2deg(mech['phi_max_rad']):10.1f} "
                  f"{mech['n_valid']:8d}")
        except RuntimeError as e:
            print(f"{a_val:6.1f} {'ERROR':>10} {str(e)[:50]}")
    return results


def simulate_a_kinematics(a_list: list, f: float, phi_down: float, phi_up: float,
                           n_points: int = 500):
    """对每个 a 值运行 wing_kinematics"""
    results = []
    for a_val in a_list:
        try:
            t, phi, phi_dot, phi_ddot, info = wing_kinematics(
                f=f, params={'a': a_val}, n_points=n_points,
                phi_down_deg=phi_down, phi_up_deg=phi_up)
            results.append({
                'a': a_val,
                't': t,
                'phi': phi,
                'phi_dot': phi_dot,
                'phi_ddot': phi_ddot,
                'info': info,
            })
        except Exception:
            pass
    return results


def plot_all(params, mech, valid, a_results, a_kin_results, f, phi_down, phi_up):
    """统一绘图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting skipped: {e}", file=sys.stderr)
        return

    fig = plt.figure(figsize=(17, 12))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    colors_a = ["k", "b", "g", "r", "m", "orange", "c", "purple", "brown"]

    # ---- 1. 机构几何轨迹 ----
    ax1 = fig.add_subplot(2, 3, 1)
    theta_c = np.linspace(0, 2 * np.pi, 360)
    l_val = params['l']
    ax1.plot(l_val * np.cos(theta_c), l_val * np.sin(theta_c), 'g--', lw=1,
             label=f'Circle r={l_val}')
    x1 = params['a'] + params['R'] * np.cos(mech['crank_theta'])
    y1 = params['b'] + params['R'] * np.sin(mech['crank_theta'])
    ax1.plot(x1, y1, 'b-', lw=1, label='Master point')
    ax1.plot(mech['x_intersect'][valid], mech['y_intersect'][valid],
             'm:', lw=1, label='Intersection')
    ax1.plot(x1[0], y1[0], 'bo', ms=6)
    ax1.plot(mech['x_intersect'][0], mech['y_intersect'][0], 'ms', ms=5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7)
    ax1.set_title(f'Mechanism Geometry (R={params["R"]}, span={mech["span_deg"]:.1f}°)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # ---- 获取默认 a 值的运动学 ----
    default_kin = None
    for r in a_kin_results:
        if abs(r['a'] - params['a']) < 0.01:
            default_kin = r
            break
    if default_kin is None and a_kin_results:
        default_kin = a_kin_results[0]

    # ---- 2. 翅膀转角 φ (实际频率) ----
    ax2 = fig.add_subplot(2, 3, 2)
    if default_kin:
        t_ms = default_kin['t'] * 1000
        ax2.plot(t_ms, np.rad2deg(default_kin['phi']), 'k-', lw=2)
        ax2.axhline(0, color='gray', ls='--', lw=0.8)
        span_eff = (np.rad2deg(abs(np.min(default_kin['phi'])))
                    + np.rad2deg(abs(np.max(default_kin['phi']))))
        ax2.set_title(f'Wing Stroke Angle @ {f} Hz (span≈{span_eff:.0f}°)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Stroke angle φ (°)')
    ax2.grid(True, alpha=0.3)

    # ---- 3. 角速度 φ̇ (实际频率) ----
    ax3 = fig.add_subplot(2, 3, 3)
    if default_kin:
        ax3.plot(t_ms, default_kin['phi_dot'], 'b-', lw=2)
        ax3.axhline(0, color='gray', ls='--', lw=0.8)
        ax3.set_title(f'Angular Velocity @ {f} Hz '
                      f'(peak={np.max(np.abs(default_kin["phi_dot"])):.0f} rad/s)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Angular velocity (rad/s)')
    ax3.grid(True, alpha=0.3)

    # ---- 4. 角加速度 φ̈ (实际频率) ----
    ax4 = fig.add_subplot(2, 3, 4)
    if default_kin:
        ax4.plot(t_ms, default_kin['phi_ddot'], 'r-', lw=2)
        ax4.axhline(0, color='gray', ls='--', lw=0.8)
        ax4.set_title(f'Angular Acceleration @ {f} Hz '
                      f'(peak={np.max(np.abs(default_kin["phi_ddot"])):.0f} rad/s²)')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Angular acceleration (rad/s²)')
    ax4.grid(True, alpha=0.3)

    # ---- 5. a 扫描 φ-t 叠加 (实际频率) ----
    ax5 = fig.add_subplot(2, 3, 5)
    for idx, r in enumerate(a_kin_results):
        color = colors_a[idx % len(colors_a)]
        ax5.plot(r['t'] * 1000, np.rad2deg(r['phi']),
                 color=color, lw=1.5, label=f'a={r["a"]}')
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Stroke angle φ (°)')
    ax5.set_title(f'φ vs time @ {f} Hz for various a')
    ax5.legend(fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.3)

    # ---- 6. span vs a ----
    ax6 = fig.add_subplot(2, 3, 6)
    a_vals = [r[0] for r in a_results]
    spans = [r[1]['span_deg'] for r in a_results]
    ax6.plot(a_vals, spans, 'o-', color='#d62728', lw=2, ms=6)
    ax6.set_xlabel('a')
    ax6.set_ylabel('Span (°)')
    ax6.set_title('Raw Stroke Amplitude vs a')
    ax6.grid(True, alpha=0.3)

    fig.suptitle(f'Mechanism Kinematics: f={f} Hz, '
                 f'φ_down={phi_down}° φ_up={phi_up}°, '
                 f'b={params["b"]}, R={params["R"]}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "mechanism_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Mechanism kinematics visualization")
    parser.add_argument("--a", type=float, default=DEFAULT_PARAMS['a'],
                        help="Master circle center x (default: 8)")
    parser.add_argument("--a-list", type=float, nargs="+",
                        default=[6, 7, 8, 9, 10, 11, 12],
                        help="a values to scan")
    parser.add_argument("--freq", type=float, default=17.5,
                        help="Flapping frequency in Hz (default: 17.5)")
    parser.add_argument("--phi-down", type=float, default=80.0,
                        help="Downstroke amplitude deg (default: 80)")
    parser.add_argument("--phi-up", type=float, default=60.0,
                        help="Upstroke amplitude deg (default: 60)")
    parser.add_argument("--n-frames", type=int, default=2000,
                        help="Points per cycle (default: 2000)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    params = {**DEFAULT_PARAMS, 'a': args.a}

    # 1. 机构几何
    mech, valid = analyze_default(params)

    # 2. a 几何扫描
    a_results = scan_a(params, args.a_list)

    # 3. 实际频率运动学
    print(f"\n实际频率运动学 @ {args.freq} Hz:")
    a_kin_results = simulate_a_kinematics(
        args.a_list, args.freq, args.phi_down, args.phi_up,
        n_points=args.n_frames)
    for r in a_kin_results:
        print(f"  a={r['a']}: span≈{r['info']['raw_span_deg']:.1f}°, "
              f"φ̇_max={np.max(np.abs(r['phi_dot'])):.0f} rad/s, "
              f"φ̈_max={np.max(np.abs(r['phi_ddot'])):.0f} rad/s²")

    if not args.no_plot:
        plot_all(params, mech, valid, a_results, a_kin_results,
                 args.freq, args.phi_down, args.phi_up)

    print("\nDone!")


if __name__ == "__main__":
    main()
