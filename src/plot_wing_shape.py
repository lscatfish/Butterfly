#!/usr/bin/env python3
"""
绘制翅膀与转轴的平面形状（planform）

读取 DXF 文件，在同一坐标系中绘制：
- 转轴（WingsAxis.DXF 中的两个圆心连线）
- 前翅轮廓（WingFront.DXF）
- 后翅轮廓（WingBack.DXF）

支持指定拍动角 phi，将翅膀绕转轴端点（p0）旋转后绘制，
用于展示不同拍动姿态。可叠加多个姿态于同一张图。

用法:
    python src/plot_wing_shape.py
    python src/plot_wing_shape.py --phi 30 --output wing_pose_up.png
    python src/plot_wing_shape.py --phi -30 --output wing_pose_down.png
    python src/plot_wing_shape.py --poses -60 -30 0 30 60 --output wing_poses.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"

# 复用 analyze_dxf 的解析函数
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from analyze_dxf import parse_dxf, connect_entities, read_axis_from_dxf


def rotate_around_axis(pts, axis, phi_deg):
    """
    将点集绕 hinge line（三维旋转）后取 XY 平面投影。

    hinge line 由轴中点 c 和方向 unit_dir 定义。
    对于点 P：
      v = P - c
      v_parallel = (v · unit_dir) * unit_dir    (沿轴，不变)
      v_perp     = v - v_parallel               (垂直于轴)
    绕轴旋转 phi 后，XY 投影中 v_perp 被压缩 cos(phi) 倍：
      P' = c + v_parallel + v_perp * cos(phi)
    """
    c = (axis["p0"] + axis["p1"]) / 2.0
    unit_dir = axis["unit_dir"]
    phi = np.deg2rad(phi_deg)
    cos_p = np.cos(phi)

    v = pts - c
    v_dot_dir = v @ unit_dir
    v_parallel = np.outer(v_dot_dir, unit_dir)
    v_perp = v - v_parallel

    return c + v_parallel + v_perp * cos_p


def plot_single_pose(axis, front_pts, back_pts, phi=0.0, ax=None,
                     front_color="blue", back_color="green", axis_color="red",
                     alpha=0.35, lw=1.5, label_prefix=""):
    """在指定 ax 上绘制轴和翅膀（单姿态，三维旋转投影），返回 ax"""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if phi != 0:
        front_pts_rot = rotate_around_axis(front_pts, axis, phi)
        back_pts_rot = rotate_around_axis(back_pts, axis, phi)
    else:
        front_pts_rot = front_pts
        back_pts_rot = back_pts

    p0, p1 = axis["p0"], axis["p1"]

    # 轴（仅在单姿态时绘制，多姿态叠加时由调用方控制）
    if label_prefix == "":
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=axis_color, lw=3, label="Axis", zorder=5)
        ax.plot(p0[0], p0[1], "o", color=axis_color, markersize=8, zorder=5)
        ax.plot(p1[0], p1[1], "s", color=axis_color, markersize=8, zorder=5)

    # 翅膀
    ax.fill(front_pts_rot[:, 0], front_pts_rot[:, 1], alpha=alpha, color=front_color, zorder=2)
    ax.plot(front_pts_rot[:, 0], front_pts_rot[:, 1], color=front_color, lw=lw,
            label=f"{label_prefix}Front".strip(), zorder=3)

    ax.fill(back_pts_rot[:, 0], back_pts_rot[:, 1], alpha=alpha, color=back_color, zorder=2)
    ax.plot(back_pts_rot[:, 0], back_pts_rot[:, 1], color=back_color, lw=lw,
            label=f"{label_prefix}Back".strip(), zorder=3)

    return ax


def main():
    parser = argparse.ArgumentParser(description="Plot wing and axis planform from DXF")
    parser.add_argument("--phi", type=float, default=0.0, help="Flapping angle (deg), CCW positive")
    parser.add_argument("--poses", type=float, nargs="+", default=None,
                        help="Multiple poses to overlay, e.g. --poses -60 -30 0 30 60")
    parser.add_argument("--output", type=str, default="wing_shape.png", help="Output filename")
    args = parser.parse_args()

    # 读取 DXF
    axis = read_axis_from_dxf(DATA_DIR / "WingsAxis.DXF")
    front_entities = parse_dxf(DATA_DIR / "WingFront.DXF")
    back_entities = parse_dxf(DATA_DIR / "WingBack.DXF")
    front_pts = connect_entities(front_entities)
    back_pts = connect_entities(back_entities)

    if front_pts is None or back_pts is None:
        print("Error: Failed to parse wing DXF files")
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / args.output

    if args.poses is not None:
        # 多姿态叠加
        fig, ax = plt.subplots(figsize=(12, 10))
        n = len(args.poses)
        colors_front = plt.cm.Blues(np.linspace(0.35, 0.9, n))
        colors_back = plt.cm.Greens(np.linspace(0.35, 0.9, n))

        for i, phi in enumerate(args.poses):
            plot_single_pose(axis, front_pts, back_pts, phi=phi, ax=ax,
                             front_color=colors_front[i], back_color=colors_back[i],
                             alpha=0.18, lw=1.0,
                             label_prefix=f"phi={phi:.0f}° ")

        # 轴置顶层
        p0, p1 = axis["p0"], axis["p1"]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "r-", lw=3, label="Axis", zorder=10)
        ax.plot(p0[0], p0[1], "ro", markersize=8, zorder=10)
        ax.plot(p1[0], p1[1], "rs", markersize=8, zorder=10)

        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Wing Planform — Multiple Poses")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_single_pose(axis, front_pts, back_pts, phi=args.phi, ax=ax)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"Wing Planform  (phi = {args.phi:.1f}°)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
