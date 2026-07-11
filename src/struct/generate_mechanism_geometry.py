#!/usr/bin/env python3
"""
生成曲柄摇杆机构几何尺寸图（单图，按 config/design_v69.yaml 参数）。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.struct.mechanism import DEFAULT_PARAMS, solve_phi


def get_p2_coords(phi, params):
    a = params['a']
    l = params['l']
    return np.array([l * np.cos(phi), a + l * np.sin(phi)])


def get_p1_coords(theta, params):
    b = params['b']
    R = params['R']
    return np.array([b + R * np.cos(theta), R * np.sin(theta)])


def plot_geometry(params, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting skipped: {e}", file=sys.stderr)
        return

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    a, b, R, c, l = params['a'], params['b'], params['R'], params['c'], params['l']

    fig, ax = plt.subplots(figsize=(10, 10))

    # 固定铰链
    ax.plot(0, a, 'ko', ms=10, label='A (wing hinge)')
    ax.plot(b, 0, 'bs', ms=10, label='B (crank center)')

    # 摇杆可达圆
    theta_c = np.linspace(0, 2 * np.pi, 200)
    ax.plot(l * np.cos(theta_c), a + l * np.sin(theta_c),
            'g--', lw=1, alpha=0.5, label=f'Swing circle r={l:.2f} mm')

    # 曲柄圆
    ax.plot(b + R * np.cos(theta_c), R * np.sin(theta_c),
            'b--', lw=1, alpha=0.5, label=f'Crank circle R={R:.2f} mm')

    # 机架
    ax.plot([0, b], [a, 0], 'k-', lw=1.5, alpha=0.4)

    # 多个姿态
    thetas = np.linspace(0, -2 * np.pi, 8, endpoint=False)
    for th in thetas:
        phi = solve_phi(th, params)
        if np.isnan(phi):
            continue
        P1 = get_p1_coords(th, params)
        P2 = get_p2_coords(phi, params)
        ax.plot([b, P1[0]], [0, P1[1]], 'b-', lw=1.2, alpha=0.35)
        ax.plot([P1[0], P2[0]], [P1[1], P2[1]], 'm-', lw=1.2, alpha=0.35)
        ax.plot([0, P2[0]], [a, P2[1]], 'g-', lw=1.2, alpha=0.35)
        ax.plot(P1[0], P1[1], 'b.', ms=5)
        ax.plot(P2[0], P2[1], 'g.', ms=5)

    # 标注尺寸
    ax.annotate('', xy=(b, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    ax.text(b / 2, -0.8, f'b={b:.2f} mm', ha='center', fontsize=10, color='gray')

    ax.annotate('', xy=(0, a), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    ax.text(-0.9, a / 2, f'a={a:.1f} mm', ha='right', fontsize=10, color='gray', rotation=90, va='center')

    ax.annotate('', xy=(b + R, 0), xytext=(b, 0),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1))
    ax.text(b + R / 2, -0.8, f'R={R:.2f} mm', ha='center', fontsize=10, color='blue')

    ax.annotate('', xy=(l, a), xytext=(0, a),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1))
    ax.text(l / 2, a + 0.4, f'l={l:.2f} mm', ha='center', fontsize=10, color='green')

    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Crank-Rocker Geometry (a={a:.1f}, b={b:.2f}, R={R:.2f}, l={l:.2f}, c={c:.2f} mm)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    params = DEFAULT_PARAMS.copy()
    output_path = ROOT / "output" / "figures" / "mechanism" / "曲柄摇杆几何尺寸图.png"
    plot_geometry(params, output_path)


if __name__ == "__main__":
    main()
