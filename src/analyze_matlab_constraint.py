"""
分析 angularvelocity2dimensions.m 的机构运动学约束

复现 MATLAB 中的平面连杆机构动态过程：
- 主点沿偏心圆匀速转动
- 动态直线与固定圆求交点（始终取 x 最大者）
- 计算交点的极角、角速度、角加速度
- 输出关键帧统计和运动轨迹图

用法:
    python src/analyze_matlab_constraint.py
    python src/analyze_matlab_constraint.py --a 8 --b 6 --R 2.25 --c 14 --l 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "figures"


def solve_intersection(A: float, B: float, C: float, l: float):
    """
    求解直线 A*x + B*y = C 与圆 x² + y² = l² 的交点。
    返回所有实数交点的 (x, y) 数组；无交点时返回空数组。
    """
    if abs(A) < 1e-10 and abs(B) < 1e-10:
        return np.array([]), np.array([])

    if abs(B) > 1e-10:
        # 消去 y，解关于 x 的二次方程
        coeff = [A ** 2 + B ** 2, -2 * A * C, C ** 2 - B ** 2 * l ** 2]
        roots_x = np.roots(coeff)
        real_mask = np.abs(np.imag(roots_x)) < 1e-10
        real_x = np.real(roots_x[real_mask])
        real_y = (C - A * real_x) / B
    else:
        # B ≈ 0，直线为垂直线 x = C/A
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


def simulate(
    a: float = 8.0,
    b: float = 6.0,
    R: float = 2.25,
    c: float = 14.0,
    l: float = 8.0,
    n_frames: int = 360,
    dt: float = 0.05,
    save_plot: bool = True,
):
    """
    运行机构运动学仿真。

    参数
    ----
    a, b, R : 主点轨迹圆 (x-a)² + (y-b)² = R²
    c       : 直线方程常数相关参数
    l       : 固定圆 x² + y² = l² 半径
    n_frames: 总帧数（主点旋转圈数对应 n_frames/360 圈）
    dt      : 帧间隔（秒）
    save_plot: 是否保存图像

    返回
    ----
    dict: 包含时间序列和统计量的结果字典
    """
    theta_rad = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

    x1 = np.zeros(n_frames)
    y1 = np.zeros(n_frames)
    x_intersect = np.full(n_frames, np.nan)
    y_intersect = np.full(n_frames, np.nan)
    phi = np.full(n_frames, np.nan)

    for i in range(n_frames):
        theta = theta_rad[i]
        x1[i] = a + R * np.cos(theta)
        y1[i] = b + R * np.sin(theta)

        A, B = x1[i], y1[i]
        C = (x1[i] ** 2 + y1[i] ** 2 - c) / 2.0

        real_x, real_y = solve_intersection(A, B, C, l)

        if real_x.size > 0:
            # 始终取 x 坐标最大的交点（与 MATLAB 一致）
            select_idx = np.argmax(real_x)
            x_intersect[i] = real_x[select_idx]
            y_intersect[i] = real_y[select_idx]
            phi[i] = np.arctan2(y_intersect[i], x_intersect[i])

    # 角运动计算
    phi_unwrap = np.unwrap(phi)
    omega = np.gradient(phi_unwrap, dt)
    alpha = np.gradient(omega, dt)
    time = np.arange(n_frames) * dt

    valid = ~np.isnan(x_intersect)

    # ---- 统计输出 ----
    stats = {
        "主点圆心": (a, b),
        "主点半径": R,
        "固定圆半径": l,
        "总周期_s": n_frames * dt,
        "有效帧数": int(valid.sum()),
        "交点_x范围": (float(x_intersect[valid].min()), float(x_intersect[valid].max())),
        "交点_y范围": (float(y_intersect[valid].min()), float(y_intersect[valid].max())),
        "极角范围_deg": (float(np.rad2deg(phi[valid].min())), float(np.rad2deg(phi[valid].max()))),
        "摆动幅度_deg": float(np.rad2deg(phi[valid].max() - phi[valid].min())),
        "omega_max_rad_s": float(np.nanmax(omega)),
        "omega_min_rad_s": float(np.nanmin(omega)),
        "alpha_max_rad_s2": float(np.nanmax(alpha)),
        "alpha_min_rad_s2": float(np.nanmin(alpha)),
        "alpha_rms_rad_s2": float(np.sqrt(np.nanmean(alpha ** 2))),
    }

    print("=" * 60)
    print("机构运动学约束分析结果")
    print("=" * 60)
    for k, v in stats.items():
        if isinstance(v, tuple):
            if all(isinstance(x, (int, float)) for x in v):
                print(f"  {k}: [{v[0]:.4f}, {v[1]:.4f}]")
            else:
                print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")

    # ---- 关键帧 ----
    key_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4]
    key_labels = ["0°", "90°", "180°", "270°"]
    print("\n关键帧:")
    print(f"{'theta':>8} {'x1':>10} {'y1':>10} {'xi':>10} {'yi':>10} {'|r|':>8} {'omega':>10} {'alpha':>10}")
    for idx, label in zip(key_indices, key_labels):
        r_norm = np.sqrt(x_intersect[idx] ** 2 + y_intersect[idx] ** 2)
        print(
            f"{label:>8} {x1[idx]:10.3f} {y1[idx]:10.3f} "
            f"{x_intersect[idx]:10.3f} {y_intersect[idx]:10.3f} "
            f"{r_norm:8.3f} {omega[idx]:10.4f} {alpha[idx]:10.4f}"
        )

    # ---- 保存图像 ----
    if save_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 1. 轨迹
            ax = axes[0, 0]
            circle_t = np.linspace(0, 2 * np.pi, 360)
            ax.plot(l * np.cos(circle_t), l * np.sin(circle_t), "g--", label="Fixed circle")
            ax.plot(x1, y1, "b-", label="Master point")
            ax.plot(x_intersect[valid], y_intersect[valid], "m:", label="Intersection")
            ax.plot(x1[0], y1[0], "bo", markersize=8)
            ax.plot(x_intersect[0], y_intersect[0], "ms", markersize=7)
            ax.set_aspect("equal")
            ax.grid(True)
            ax.legend()
            ax.set_title("Mechanism Trajectory")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # 2. 极角
            ax = axes[0, 1]
            ax.plot(time, np.rad2deg(phi_unwrap), "k-")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Polar angle (deg)")
            ax.set_title("Unwrapped Polar Angle")
            ax.grid(True)

            # 3. 角速度
            ax = axes[1, 0]
            ax.plot(time, omega, "b-", linewidth=1.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Angular velocity (rad/s)")
            ax.set_title("Angular Velocity")
            ax.grid(True)

            # 4. 角加速度
            ax = axes[1, 1]
            ax.plot(time, alpha, "r-", linewidth=1.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Angular acceleration (rad/s2)")
            ax.set_title("Angular Acceleration")
            ax.grid(True)

            plt.tight_layout()
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DIR / "angularvelocity2d_analysis.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\nPlot saved to: {out_path}")
        except Exception as e:
            print(f"\nPlotting skipped: {e}", file=sys.stderr)

    return {
        "time": time,
        "theta": theta_rad,
        "x1": x1,
        "y1": y1,
        "x_intersect": x_intersect,
        "y_intersect": y_intersect,
        "phi": phi,
        "phi_unwrap": phi_unwrap,
        "omega": omega,
        "alpha": alpha,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze angularvelocity2dimensions.m mechanism kinematics"
    )
    parser.add_argument("--a", type=float, default=8.0, help="Master circle center x")
    parser.add_argument("--b", type=float, default=6.0, help="Master circle center y")
    parser.add_argument("--R", type=float, default=2.25, help="Master circle radius")
    parser.add_argument("--c", type=float, default=14.0, help="Line parameter c")
    parser.add_argument("--l", type=float, default=8.0, help="Fixed circle radius")
    parser.add_argument("--n-frames", type=int, default=360, help="Number of frames")
    parser.add_argument("--dt", type=float, default=0.05, help="Frame interval (s)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    simulate(
        a=args.a,
        b=args.b,
        R=args.R,
        c=args.c,
        l=args.l,
        n_frames=args.n_frames,
        dt=args.dt,
        save_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
