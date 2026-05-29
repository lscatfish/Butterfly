"""
扫描不同主点半径 R，绘制翅膀拍动角 phi-t 曲线

复现 angularvelocity2dimensions.m 的机构运动学，固定其他参数，
仅改变主点轨迹圆半径 R，观察交点极角（翅膀拍动角）随时间的变化。

用法:
    python src/plot_phi_t_various_R.py
    python src/plot_phi_t_various_R.py --R-list 2.25 4 5 6 --a 8 --b 6 --l 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "figures"


def solve_intersection(A: float, B: float, C: float, l: float):
    """直线 A*x + B*y = C 与圆 x² + y² = l² 的实数交点。"""
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


def simulate_phi(a: float, b: float, R: float, c: float, l: float, n: int, dt: float):
    """
    对给定 R 计算一周期内交点极角 phi（度）。
    返回 (time, phi_deg, phi_min, phi_max, span, valid_count)。
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x1 = a + R * np.cos(theta)
    y1 = b + R * np.sin(theta)
    phi = np.full(n, np.nan)

    for i in range(n):
        A, B = x1[i], y1[i]
        C = (x1[i] ** 2 + y1[i] ** 2 - c) / 2.0
        real_x, real_y = solve_intersection(A, B, C, l)
        if real_x.size > 0:
            idx = np.argmax(real_x)
            phi[i] = np.arctan2(real_y[idx], real_x[idx])

    phi_unwrap = np.unwrap(phi)
    phi_deg = np.rad2deg(phi_unwrap)
    time = np.arange(n) * dt

    valid = ~np.isnan(phi_deg)
    if valid.sum() == 0:
        return time, phi_deg, 0.0, 0.0, 0.0, 0

    phi_min = float(np.nanmin(phi_deg))
    phi_max = float(np.nanmax(phi_deg))
    span = phi_max - phi_min
    return time, phi_deg, phi_min, phi_max, span, int(valid.sum())


def main():
    parser = argparse.ArgumentParser(
        description="Plot wing flapping angle phi-t curves for various master-point radii R"
    )
    parser.add_argument("--a", type=float, default=8.0, help="Master circle center x")
    parser.add_argument("--b", type=float, default=6.0, help="Master circle center y")
    parser.add_argument("--c", type=float, default=14.0, help="Line parameter c")
    parser.add_argument("--l", type=float, default=8.0, help="Fixed circle radius")
    parser.add_argument("--n-frames", type=int, default=360, help="Frames per cycle")
    parser.add_argument("--dt", type=float, default=0.05, help="Frame interval [s]")
    parser.add_argument(
        "--R-list",
        type=float,
        nargs="+",
        default=[2.25, 3.0, 4.0, 5.0, 6.0, 7.0],
        help="List of R values to scan",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    R_values = args.R_list
    colors = ["k", "b", "g", "r", "m", "orange", "c", "purple", "brown", "pink"]

    # ---- 数值计算 ----
    results = []
    print(f"{'R':>6} {'phi_min':>10} {'phi_max':>10} {'span':>10} {'valid':>8}")
    print("-" * 50)
    for R in R_values:
        time, phi_deg, phi_min, phi_max, span, valid = simulate_phi(
            args.a, args.b, R, args.c, args.l, args.n_frames, args.dt
        )
        results.append((R, time, phi_deg, phi_min, phi_max, span, valid))
        print(f"{R:6.2f} {phi_min:10.2f} {phi_max:10.2f} {span:10.2f} {valid:8d}")

    if args.no_plot:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting skipped: {e}", file=sys.stderr)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 图1: 子图阵列 ----
    n_curves = len(R_values)
    n_cols = 2
    n_rows = (n_curves + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5.5 * n_rows))
    if n_rows == 1:
        axes = np.array(axes).reshape(1, -1)
    axes = axes.flatten()

    for idx, (R, time, phi_deg, phi_min, phi_max, span, valid) in enumerate(results):
        ax = axes[idx]
        color = colors[idx % len(colors)]
        ax.plot(time, phi_deg, color=color, linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"R = {R}", fontsize=12)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("phi (deg)")
        ax.grid(True)
        ax.text(
            0.98,
            0.98,
            f"Range: [{phi_min:.1f}, {phi_max:.1f}]\nSpan: {span:.1f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # 隐藏多余子图
    for idx in range(n_curves, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Wing Flapping Angle vs Time for Different Master Point Radii (R)",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = OUTPUT_DIR / "phi_t_various_R.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out1}")

    # ---- 图2: 叠加对比 ----
    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, (R, time, phi_deg, _, _, _, _) in enumerate(results):
        color = colors[idx % len(colors)]
        ax.plot(time, phi_deg, color=color, linewidth=1.5, label=f"R={R}")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Flapping Angle phi (deg)", fontsize=12)
    ax.set_title("Comparison: phi-t curves for various R", fontsize=13)
    ax.legend(loc="best")
    ax.grid(True)

    out2 = OUTPUT_DIR / "phi_t_comparison.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
