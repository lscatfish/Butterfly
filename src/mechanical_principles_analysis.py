#!/usr/bin/env python3
"""
Mechanical-principles oriented analysis for the butterfly flapping mechanism.

This script reorganizes the existing aerodynamic, kinematic, and gear-train
results into the kind of simplified motion/force analysis usually needed for
a mechanical principles course assignment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynamic_analysis import AERO_PARAMS, load_geometry, simulate_cycle
from gear_analysis import FixedAxisGearTrain
from mechanism import DEFAULT_PARAMS, mechanism_cycle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
DATA_DIR = PROJECT_ROOT / "data"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def four_wing_series(front: dict, back: dict) -> dict:
    """Combine one front wing and one back wing into four-wing totals."""
    return {
        "P_aero": 2.0 * (front["P_aero"] + back["P_aero"]),
        "P_inertial": 2.0 * (front["P_inertial"] + back["P_inertial"]),
        "P_total": 2.0 * (front["P_total"] + back["P_total"]),
        "M_aero": 2.0 * (
            front["F_drag"] * front["R_equiv"] + back["F_drag"] * back["R_equiv"]
        ),
        "M_inertial": 2.0 * (
            front["I_w"] * front["phi_ddot"] + back["I_w"] * back["phi_ddot"]
        ),
    }


def enrich_sim_with_moments(sim: dict, geo_item: dict) -> dict:
    """Add equivalent lever arm and hinge moments to a wing simulation."""
    sim = dict(sim)
    r_eff = geo_item["R"] * geo_item["r1"]
    sim["R_equiv"] = r_eff
    sim["M_aero"] = sim["F_drag"] * r_eff
    sim["M_inertial"] = sim["I_w"] * sim["phi_ddot"]
    sim["M_total_abs_model"] = np.abs(sim["M_aero"]) + np.abs(sim["M_inertial"])
    return sim


def compute_results() -> dict:
    geo = load_geometry()
    params = AERO_PARAMS.copy()
    front = enrich_sim_with_moments(simulate_cycle(geo["Front"], params), geo["Front"])
    back = enrich_sim_with_moments(simulate_cycle(geo["Back"], params), geo["Back"])
    combined = four_wing_series(front, back)

    gear = FixedAxisGearTrain()
    mech = mechanism_cycle(DEFAULT_PARAMS, n_points=720)

    f = params["f"]
    omega_crank = 2.0 * math.pi * f
    i_total = abs(gear.i_total)
    eta_total = gear.eta_total

    p_aero_avg = float(np.mean(combined["P_aero"]))
    p_total_peak = float(np.max(np.abs(combined["P_total"])))
    p_inertial_abs_avg = float(np.mean(np.abs(combined["P_inertial"])))
    p_inertial_peak = float(np.max(np.abs(combined["P_inertial"])))

    t_out_avg = p_aero_avg / omega_crank
    t_out_peak = p_total_peak / omega_crank
    t_motor_avg = t_out_avg / (i_total * eta_total)
    t_motor_peak = t_out_peak / (i_total * eta_total)

    g1 = gear.gear1
    g3 = gear.gear3
    d1_m = g1.d / 1000.0
    d3_m = g3.d / 1000.0
    pressure_angle = gear.alpha

    ft1_peak = 2.0 * t_motor_peak / d1_m
    fr1_peak = ft1_peak * math.tan(pressure_angle)
    fn1_peak = ft1_peak / math.cos(pressure_angle)
    ft3_peak = 2.0 * t_out_peak / d3_m
    fr3_peak = ft3_peak * math.tan(pressure_angle)
    fn3_peak = ft3_peak / math.cos(pressure_angle)

    weight_n = params["m_total"] * 9.81

    return {
        "params": params,
        "geo": geo,
        "front": front,
        "back": back,
        "combined": combined,
        "gear": gear,
        "mech": mech,
        "kinematics": {
            "omega_crank_rad_s": omega_crank,
            "period_s": 1.0 / f,
            "raw_span_deg": mech["span_deg"],
            "phi_min_deg": math.degrees(mech["phi_min_rad"]),
            "phi_max_deg": math.degrees(mech["phi_max_rad"]),
            "front_phi_dot_peak": float(np.max(np.abs(front["phi_dot"]))),
            "front_phi_ddot_peak": float(np.max(np.abs(front["phi_ddot"]))),
        },
        "power_torque": {
            "p_aero_avg_W": p_aero_avg,
            "p_total_peak_W": p_total_peak,
            "p_inertial_abs_avg_W": p_inertial_abs_avg,
            "p_inertial_peak_W": p_inertial_peak,
            "t_out_avg_Nm": t_out_avg,
            "t_out_peak_Nm": t_out_peak,
            "t_motor_avg_Nm": t_motor_avg,
            "t_motor_peak_Nm": t_motor_peak,
            "weight_N": weight_n,
        },
        "gear_forces": {
            "mesh_12_peak": {"Ft_N": ft1_peak, "Fr_N": fr1_peak, "Fn_N": fn1_peak},
            "mesh_2p3_peak": {"Ft_N": ft3_peak, "Fr_N": fr3_peak, "Fn_N": fn3_peak},
        },
        "gear_summary": {
            "i_total": i_total,
            "eta_total": eta_total,
            "d1_mm": g1.d,
            "d3_mm": g3.d,
            "alpha_deg": gear.alpha_deg,
        },
    }


def add_arrow(ax, xy_from, xy_to, text, color="#1f77b4", lw=2.0, fs=10):
    ax.annotate(
        text,
        xy=xy_to,
        xytext=xy_from,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
    )


def draw_box(ax, xy, w, h, text, fc="#f8f9fb", ec="#333333", fs=10):
    rect = plt.Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, lw=1.5)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def plot_system_force_flow(results: dict) -> Path:
    path = FIG_DIR / "mechanical_system_force_flow.png"
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")

    labels = [
        ("电机\n旋转输入", 0.4),
        ("两级定轴\n齿轮减速器", 2.6),
        ("曲柄\n匀速转动", 4.9),
        ("连杆机构\n传递力", 7.0),
        ("摇杆/翅根\n往复摆动", 9.1),
        ("翅膀\n气动力+惯性力", 11.2),
    ]
    for label, x in labels:
        draw_box(ax, (x, 1.35), 1.45, 1.0, label, fc="#ffffff")

    for i in range(len(labels) - 1):
        x0 = labels[i][1] + 1.45
        x1 = labels[i + 1][1]
        ax.annotate("", xy=(x1, 1.85), xytext=(x0, 1.85),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color="#1f77b4"))

    pt = results["power_torque"]
    gs = results["gear_summary"]
    ax.text(3.35, 3.1, f"传动比 i = {gs['i_total']:.3f}, 效率 eta = {gs['eta_total']:.3f}",
            ha="center", fontsize=10)
    ax.text(9.7, 0.55,
            f"输出轴峰值扭矩约 {pt['t_out_peak_Nm']*1000:.2f} N.mm\n"
            f"电机峰值扭矩约 {pt['t_motor_peak_Nm']*1000:.3f} N.mm",
            ha="center", fontsize=10)
    ax.set_title("机械功率与力传递路径", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_wing_fbd(results: dict) -> Path:
    path = FIG_DIR / "wing_free_body_diagram.png"
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.0, 7.0)
    ax.set_ylim(-2.6, 3.6)

    root = np.array([0.0, 0.0])
    wing = np.array([[0.0, -0.35], [5.3, -1.15], [5.9, 1.15], [0.0, 0.35]])
    ax.fill(wing[:, 0], wing[:, 1], color="#d7ebff", ec="#1f77b4", lw=2)
    ax.plot(root[0], root[1], "ko", ms=7)
    ax.text(-0.2, -0.55, "翅根铰点", ha="right", fontsize=10)

    add_arrow(ax, (-0.75, 1.65), (0.0, 0.2), "铰点反力\nRx, Ry", color="#333333")
    add_arrow(ax, (3.15, 2.75), (3.15, 0.8), "升力 F_L", color="#2ca02c")
    add_arrow(ax, (5.45, -2.05), (3.8, -0.55), "阻力 F_D", color="#d62728")
    ax.annotate("气动力矩 M_aero", xy=(0.45, 0.55), xytext=(1.15, 1.45),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.5", lw=2, color="#d62728"),
                color="#d62728", fontsize=11)
    ax.annotate("惯性力矩 M_inertia = I_w * phi_ddot", xy=(0.65, -0.55), xytext=(0.95, -2.15),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.45", lw=2, color="#9467bd"),
                color="#9467bd", fontsize=11)

    geo = results["geo"]
    front = results["front"]
    back = results["back"]
    ax.text(
        0.2,
        3.15,
        "翅根等效力矩:\n"
        "M_wing = F_drag * R * r1 + I_w * phi_ddot\n"
        f"前翅等效力臂 = {front['R_equiv']*1000:.1f} mm, 后翅等效力臂 = {back['R_equiv']*1000:.1f} mm\n"
        f"前翅面积 = {geo['Front']['S_mm2']:.0f} mm^2, 后翅面积 = {geo['Back']['S_mm2']:.0f} mm^2",
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="#cccccc", pad=6),
    )
    ax.set_title("单翅等效自由体图", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_linkage_force_diagram(results: dict) -> Path:
    path = FIG_DIR / "linkage_force_diagram.png"
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.5, 8.0)
    ax.set_ylim(-2.2, 4.8)

    o2 = np.array([0.0, 0.0])
    a = np.array([1.6, 1.1])
    b = np.array([4.6, 2.35])
    o4 = np.array([5.4, 0.0])
    wing_tip = np.array([7.2, 3.4])

    ax.plot([o2[0], a[0]], [o2[1], a[1]], "o-", lw=4, color="#1f77b4")
    ax.plot([a[0], b[0]], [a[1], b[1]], "o-", lw=4, color="#ff7f0e")
    ax.plot([o4[0], b[0]], [o4[1], b[1]], "o-", lw=4, color="#2ca02c")
    ax.plot([b[0], wing_tip[0]], [b[1], wing_tip[1]], "-", lw=3, color="#2ca02c")
    ax.plot(o2[0], o2[1], "ks", ms=8)
    ax.plot(o4[0], o4[1], "ks", ms=8)

    ax.text(o2[0], o2[1] - 0.35, "O2", ha="center")
    ax.text(o4[0], o4[1] - 0.35, "O4 / 翅根", ha="center")
    ax.text(a[0], a[1] + 0.25, "A", ha="center")
    ax.text(b[0], b[1] + 0.25, "B", ha="center")

    ax.annotate("曲柄输入扭矩 T_crank", xy=(0.55, 0.2), xytext=(0.25, -1.2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.6", lw=2, color="#1f77b4"),
                color="#1f77b4", fontsize=11)
    add_arrow(ax, (2.4, 3.7), (2.9, 1.75), "连杆力 F_link", color="#ff7f0e")
    ax.annotate("翅根力矩 M_wing", xy=(5.15, 0.7), xytext=(6.55, 0.9),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.45", lw=2, color="#d62728"),
                color="#d62728", fontsize=11)
    add_arrow(ax, (7.25, 4.35), (6.15, 3.0), "气动载荷", color="#d62728")
    add_arrow(ax, (-0.8, 1.35), (0.0, 0.0), "机架反力 R_O2", color="#333333")
    add_arrow(ax, (4.35, -1.25), (5.4, 0.0), "机架反力 R_O4", color="#333333")

    pt = results["power_torque"]
    ax.text(
        -1.1,
        4.4,
        "等效动态静力传递链:\n"
        "翅根力矩 -> 摇杆 -> 连杆力 -> 曲柄扭矩\n"
        f"峰值输出扭矩约 {pt['t_out_peak_Nm']*1000:.2f} N.mm",
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="#cccccc", pad=6),
    )
    ax.set_title("扑翼连杆机构受力简图", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gear_force_diagram(results: dict) -> Path:
    path = FIG_DIR / "gear_force_diagram.png"
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0.0, 10.2)
    ax.set_ylim(-1.8, 5.2)

    def draw_bearing(x: float, y: float, side: str = "left") -> None:
        width, height = 0.95, 0.35
        x0 = x - width - 0.18 if side == "left" else x + 0.18
        rect = plt.Rectangle(
            (x0, y - height / 2),
            width,
            height,
            facecolor="#f6f6f6",
            edgecolor="#333333",
            hatch="///",
            lw=1.3,
        )
        ax.add_patch(rect)
        ax.plot([x0, x0 + width], [y, y], color="#333333", lw=1.2)

    def draw_gear_bar(x: float, y: float, half_width: float, label: str, color: str) -> None:
        ax.add_patch(
            plt.Rectangle(
                (x - half_width, y - 0.18),
                2 * half_width,
                0.36,
                facecolor=color,
                edgecolor="#222222",
                lw=1.8,
            )
        )
        ax.text(x, y + 0.42, label, ha="center", va="bottom", fontsize=11)

    def draw_mesh_marker(x: float, y: float, label: str) -> None:
        ax.plot([x, x], [y - 0.38, y + 0.38], color="#d62728", lw=2.0)
        ax.plot([x - 0.08, x + 0.08], [y - 0.22, y - 0.38], color="#d62728", lw=1.5)
        ax.plot([x - 0.08, x + 0.08], [y + 0.22, y + 0.38], color="#d62728", lw=1.5)
        ax.text(x, y - 0.62, label, ha="center", va="top", fontsize=9, color="#d62728")

    # 三根平行定轴。中间轴上齿轮 2 和 2' 为双联固连。
    x1, x2, x3 = 1.8, 4.3, 6.8
    y_mesh12, y_mesh23 = 3.35, 1.2
    shaft_top, shaft_bottom = 4.45, 0.25

    for x in (x1, x2, x3):
        ax.plot([x, x], [shaft_bottom, shaft_top], color="#222222", lw=2.4)
        ax.plot([x - 0.16, x + 0.16], [shaft_top, shaft_top], color="#222222", lw=2.0)
        ax.plot([x - 0.16, x + 0.16], [shaft_bottom, shaft_bottom], color="#222222", lw=2.0)

    # 轴承支承，画成机械原理简图里常用的剖线块。
    draw_bearing(x1, 4.0, "left")
    draw_bearing(x1, 0.75, "left")
    draw_bearing(x2, 4.0, "right")
    draw_bearing(x2, 0.75, "right")
    draw_bearing(x3, 4.0, "right")
    draw_bearing(x3, 0.75, "right")

    # 齿轮用宽窄横条表示。横条边缘相接，突出啮合关系。
    draw_gear_bar(x1, y_mesh12, 0.65, "齿轮1\nz1=7", "#d7ebff")
    draw_gear_bar(x2, y_mesh12, 1.85, "齿轮2\nz2=40", "#f9e0b7")
    draw_gear_bar(x2, y_mesh23, 0.65, "齿轮2'\nz2'=7", "#d7ebff")
    draw_gear_bar(x3, y_mesh23, 1.85, "齿轮3\nz3=40", "#f9e0b7")

    draw_mesh_marker(x1 + 0.65, y_mesh12, "1-2 啮合")
    draw_mesh_marker(x2 + 0.65, y_mesh23, "2'-3 啮合")

    ax.text(x1, 4.78, "轴 I\n输入轴", ha="center", va="bottom", fontsize=11)
    ax.text(x2, 4.78, "轴 II\n中间双联轴", ha="center", va="bottom", fontsize=11)
    ax.text(x3, 4.78, "轴 III\n输出轴", ha="center", va="bottom", fontsize=11)
    ax.annotate(
        "输入转矩 T1",
        xy=(x1 - 0.35, 3.85),
        xytext=(0.45, 4.55),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#1f77b4"),
        fontsize=10,
        color="#1f77b4",
    )
    ax.annotate(
        "输出转矩 T3",
        xy=(x3 + 0.4, 1.75),
        xytext=(7.95, 2.55),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#1f77b4"),
        fontsize=10,
        color="#1f77b4",
    )

    gf = results["gear_forces"]
    gs = results["gear_summary"]
    ax.text(
        0.45,
        -0.85,
        f"峰值啮合 1-2: Ft={gf['mesh_12_peak']['Ft_N']:.2f} N, "
        f"Fr={gf['mesh_12_peak']['Fr_N']:.2f} N, Fn={gf['mesh_12_peak']['Fn_N']:.2f} N\n"
        f"峰值啮合 2'-3: Ft={gf['mesh_2p3_peak']['Ft_N']:.2f} N, "
        f"Fr={gf['mesh_2p3_peak']['Fr_N']:.2f} N, Fn={gf['mesh_2p3_peak']['Fn_N']:.2f} N\n"
        f"总传动比 i={gs['i_total']:.3f}, 输出轴与输入轴同向, 压力角={gs['alpha_deg']:.1f} deg",
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="#cccccc", pad=6),
    )
    ax.set_title("两级定轴齿轮轮系结构简图", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_annotated_existing_gear_motion_diagram(results: dict) -> Path:
    """Overlay force annotations on the existing hand-style gear train diagram."""
    src_path = OUTPUT_DIR / "fig" / "齿轮运动简图.png"
    path = FIG_DIR / "齿轮运动简图_受力标注.png"
    if not src_path.exists():
        return plot_gear_force_diagram(results)

    img = plt.imread(src_path)
    fig, ax = plt.subplots(figsize=(10, 8.2))
    ax.imshow(img)
    ax.axis("off")

    h, w = img.shape[:2]

    def p(x_frac: float, y_frac: float) -> tuple[float, float]:
        return x_frac * w, y_frac * h

    def arrow(text: str, start, end, color: str, fs: int = 13) -> None:
        ax.annotate(
            text,
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=2.5, color=color),
            fontsize=fs,
            color=color,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
        )

    def curved(text: str, center, text_xy, rad: float, color: str) -> None:
        ax.annotate(
            text,
            xy=center,
            xytext=text_xy,
            arrowprops=dict(arrowstyle="->", lw=2.5, color=color, connectionstyle=f"arc3,rad={rad}"),
            fontsize=13,
            color=color,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
        )

    # Upper mesh: radial force along center line, tangential force perpendicular
    # to center line, normal force as their resultant.
    upper_mesh = p(0.55, 0.36)
    arrow("Ft1", p(0.50, 0.25), p(0.50, 0.34), "#d62728")
    arrow("Fr1", p(0.66, 0.34), p(0.57, 0.34), "#1f77b4")
    arrow("Fn1", p(0.67, 0.25), upper_mesh, "#9467bd")
    ax.text(*p(0.61, 0.40), "第一级啮合", fontsize=12, color="#333333",
            ha="center", bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2))

    # Lower mesh.
    lower_mesh = p(0.56, 0.70)
    arrow("Ft2", p(0.49, 0.79), p(0.49, 0.71), "#d62728")
    arrow("Fr2", p(0.33, 0.69), p(0.45, 0.69), "#1f77b4")
    arrow("Fn2", p(0.34, 0.79), lower_mesh, "#9467bd")
    ax.text(*p(0.40, 0.63), "第二级啮合", fontsize=12, color="#333333",
            ha="center", bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2))

    # Input / output torques.
    curved("输入转矩 T1", p(0.45, 0.29), p(0.35, 0.20), 0.45, "#2ca02c")
    curved("输出转矩 T3", p(0.60, 0.18), p(0.73, 0.14), -0.45, "#2ca02c")

    # Bearing reactions at representative supports.
    arrow("轴承反力 R_A", p(0.12, 0.29), p(0.24, 0.34), "#333333", fs=11)
    arrow("轴承反力 R_B", p(0.86, 0.30), p(0.72, 0.34), "#333333", fs=11)
    arrow("轴承反力 R_C", p(0.12, 0.74), p(0.24, 0.68), "#333333", fs=11)
    arrow("轴承反力 R_D", p(0.79, 0.78), p(0.62, 0.72), "#333333", fs=11)

    gf = results["gear_forces"]
    gs = results["gear_summary"]
    ax.text(
        *p(0.04, 0.94),
        "受力标注说明: Fr 沿两轮中心线, Ft 与中心线垂直, Fn 为法向合力\n"
        f"第一级峰值: Ft={gf['mesh_12_peak']['Ft_N']:.2f} N, "
        f"Fr={gf['mesh_12_peak']['Fr_N']:.2f} N, Fn={gf['mesh_12_peak']['Fn_N']:.2f} N\n"
        f"第二级峰值: Ft={gf['mesh_2p3_peak']['Ft_N']:.2f} N, "
        f"Fr={gf['mesh_2p3_peak']['Fr_N']:.2f} N, Fn={gf['mesh_2p3_peak']['Fn_N']:.2f} N; "
        f"总传动比 i={gs['i_total']:.3f}",
        fontsize=12,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.9, pad=6),
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_output_torque_time(results: dict) -> Path:
    path = FIG_DIR / "equivalent_output_torque.png"
    c = results["combined"]
    f = results["params"]["f"]
    omega_crank = 2.0 * math.pi * f
    t_ms = results["front"]["t"] * 1000.0
    torque_from_power = c["P_total"] / omega_crank
    torque_aero = c["M_aero"]
    torque_inertial = c["M_inertial"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(t_ms, torque_from_power * 1000, "k-", lw=2.2, label="由 P/omega 得到的输出扭矩")
    ax.plot(t_ms, torque_aero * 1000, color="#d62728", lw=1.7, label="气动翅根力矩")
    ax.plot(t_ms, torque_inertial * 1000, color="#9467bd", lw=1.7, label="惯性翅根力矩")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("扭矩 / 力矩 (N.mm)")
    ax.set_title("等效输出扭矩与翅根力矩")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary_json(results: dict) -> Path:
    path = OUTPUT_DIR / "tables" / "mechanical_principles_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "kinematics": results["kinematics"],
        "power_torque": results["power_torque"],
        "gear_forces": results["gear_forces"],
        "gear_summary": results["gear_summary"],
        "geometry": {
            name: {k: float(v) for k, v in vals.items() if isinstance(v, (int, float))}
            for name, vals in results["geo"].items()
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def generate_report(results: dict, figure_paths: dict[str, Path], summary_json: Path) -> Path:
    path = REPORT_DIR / "机械原理结构系统分析.md"
    p = results["params"]
    kin = results["kinematics"]
    pt = results["power_torque"]
    gs = results["gear_summary"]
    gf = results["gear_forces"]
    geo = results["geo"]

    md = f"""# 仿生蝴蝶扑翼结构系统机械原理分析

> 分析脚本: `src/mechanical_principles_analysis.py`  
> 分析定位: 机械原理课程作业用的等效运动学与动态静力分析  
> 说明: 本报告不再讨论能否真实起飞，只把已有气动力作为机构外载荷来分析传动与受力关系。

---

## 1. 系统组成与运动传递路线

该结构系统可分为五个部分:

1. 电机输入部分: 提供连续旋转运动。
2. 两级定轴齿轮减速器: 降低转速、提高输出扭矩。
3. 曲柄输入部分: 将齿轮输出轴的连续旋转作为扑翼机构输入。
4. 连杆/摇杆扑翼机构: 将连续转动转化为翅膀往复摆动。
5. 翅膀执行构件: 承受气动力与惯性力，并通过翅根把载荷传回机构。

![系统力流图](../figures/{figure_paths['force_flow'].name})

运动与力传递链为:

```text
电机转矩 -> 齿轮减速器 -> 曲柄转矩 -> 连杆力 -> 摇杆/翅根力矩 -> 翅膀气动力与惯性力
```

---

## 2. 自由度分析

实际 CAD 与代码中的扑翼输出机构采用“主点圆周运动 + 几何约束交点”的方式生成翅膀转角。为了满足机械原理课程分析，本文将其等效为平面曲柄-连杆-摇杆型单输入机构。

平面机构自由度公式为:

```text
F = 3n - 2PL - PH
```

其中:

- `n`: 活动构件数
- `PL`: 低副数，包括转动副、移动副
- `PH`: 高副数

对扑翼执行机构作课程等效建模:

| 项目 | 数值 | 说明 |
|---|---:|---|
| 活动构件数 `n` | 3 | 曲柄、连杆、摇杆/翅根输出件 |
| 低副数 `PL` | 4 | 机架-曲柄、曲柄-连杆、连杆-摇杆、摇杆-机架 |
| 高副数 `PH` | 0 | 按等效四杆机构处理 |

因此:

```text
F = 3*3 - 2*4 - 0 = 1
```

结论: 扑翼机构自由度为 1，使用一个电机输入即可驱动机构完成周期性扑翼运动。齿轮轮系只改变转速、转矩和转向关系，不增加独立输入自由度。

---

## 3. 轮系传动分析

轮系采用两级定轴外啮合圆柱齿轮传动，齿数为:

```text
z1 = 7, z2 = 40, z2' = 7, z3 = 40
```

各级传动比:

```text
i12  = -z2/z1  = -40/7 = -5.7143
i2'3 = -z3/z2' = -40/7 = -5.7143
i13  = i12*i2'3 = +32.6531
```

轮系结论:

| 项目 | 数值 |
|---|---:|
| 总传动比 `|i13|` | {gs['i_total']:.4f} |
| 总效率 `eta` | {gs['eta_total']:.4f} |
| 输出/输入转向 | 相同 |
| 模数 | 0.3 mm |
| 压力角 | {gs['alpha_deg']:.1f} deg |
| 齿轮1分度圆直径 | {gs['d1_mm']:.3f} mm |
| 齿轮3分度圆直径 | {gs['d3_mm']:.3f} mm |

![轮系受力标注图](../figures/{figure_paths['gear_annotated'].name})

---

## 4. 扑翼机构运动分析

机构参数为:

| 参数 | 数值 | 含义 |
|---|---:|---|
| `a` | 8.0 | 主点圆心 x 坐标 |
| `b` | 6.97 | 主点圆心 y 坐标 |
| `R` | 2.25 | 主点轨迹圆半径 |
| `c` | 14 | 直线约束常数 |
| `l` | 8 | 固定圆半径 |

由 `mechanism.py` 计算得到:

| 项目 | 数值 |
|---|---:|
| 扑翼频率 | {p['f']:.1f} Hz |
| 曲柄角速度 | {kin['omega_crank_rad_s']:.2f} rad/s |
| 周期 | {kin['period_s']*1000:.2f} ms |
| 扑翼角范围 | {kin['phi_min_deg']:.2f} deg ~ {kin['phi_max_deg']:.2f} deg |
| 总摆幅 | {kin['raw_span_deg']:.2f} deg |
| 翅膀角速度峰值 | {kin['front_phi_dot_peak']:.2f} rad/s |
| 翅膀角加速度峰值 | {kin['front_phi_ddot_peak']:.0f} rad/s^2 |

该机构的特点是: 输入曲柄匀速转动，但输出翅膀角速度和角加速度不均匀，换向附近存在较大角加速度。因此在动态静力分析中，除气动力外还必须考虑翅膀惯性力矩。

---

## 5. 翅膀等效受力分析

翅膀几何参数来自 DXF 解析结果:

| 翅膀 | 面积 S(mm^2) | 展长 R(mm) | 平均弦长(mm) | r1 | r2_sq |
|---|---:|---:|---:|---:|---:|
| 前翅 | {geo['Front']['S_mm2']:.1f} | {geo['Front']['R_mm']:.1f} | {geo['Front']['c_avg']*1000:.1f} | {geo['Front']['r1']:.4f} | {geo['Front']['r2_sq']:.4f} |
| 后翅 | {geo['Back']['S_mm2']:.1f} | {geo['Back']['R_mm']:.1f} | {geo['Back']['c_avg']*1000:.1f} | {geo['Back']['r1']:.4f} | {geo['Back']['r2_sq']:.4f} |

单翅自由体图如下:

![单翅自由体图](../figures/{figure_paths['wing'].name})

在机械原理层面，将分布气动力简化为作用于等效力臂 `R*r1` 处的集中力，翅根等效力矩为:

```text
M_aero = F_drag * R * r1
M_inertia = I_w * phi_ddot
M_wing = M_aero + M_inertia
```

其中翅膀转动惯量采用:

```text
I_w = m_w * R^2 * r2_sq
```

这一步的意义是把复杂的翼面载荷转换成可传入连杆机构的翅根等效阻力矩。

---

## 6. 连杆机构动态静力分析

机构受力简图如下:

![连杆受力图](../figures/{figure_paths['linkage'].name})

动态静力分析采用从输出端向输入端反推的思路:

```text
翅膀等效力矩 M_wing
-> 摇杆输出力矩
-> 连杆轴向力 F_link
-> 曲柄输入转矩 T_crank
-> 齿轮输出轴转矩 T_output
```

由于当前项目尚未建立完整的杆长、铰点尺寸和瞬时压力角模型，本报告先采用功率等效法求输出轴扭矩:

```text
T_output = P_output / omega_crank
```

得到:

| 项目 | 数值 |
|---|---:|
| 平均气动功率 | {pt['p_aero_avg_W']*1000:.1f} mW |
| 平均惯性功率绝对值 | {pt['p_inertial_abs_avg_W']*1000:.1f} mW |
| 峰值惯性功率 | {pt['p_inertial_peak_W']*1000:.1f} mW |
| 峰值总功率 | {pt['p_total_peak_W']*1000:.1f} mW |
| 平均输出扭矩 | {pt['t_out_avg_Nm']*1000:.2f} N.mm |
| 峰值输出扭矩 | {pt['t_out_peak_Nm']*1000:.2f} N.mm |
| 平均电机扭矩 | {pt['t_motor_avg_Nm']*1000:.3f} N.mm |
| 峰值电机扭矩 | {pt['t_motor_peak_Nm']*1000:.3f} N.mm |

![等效输出扭矩](../figures/{figure_paths['torque'].name})

---

## 7. 齿轮啮合力计算

由输出扭矩和电机端扭矩可估算两级齿轮啮合力。公式为:

```text
Ft = 2T / d
Fr = Ft * tan(alpha)
Fn = Ft / cos(alpha)
```

其中 `alpha = 20 deg`。

峰值工况下:

| 啮合副 | Ft(N) | Fr(N) | Fn(N) |
|---|---:|---:|---:|
| 齿轮1-齿轮2 | {gf['mesh_12_peak']['Ft_N']:.3f} | {gf['mesh_12_peak']['Fr_N']:.3f} | {gf['mesh_12_peak']['Fn_N']:.3f} |
| 齿轮2'-齿轮3 | {gf['mesh_2p3_peak']['Ft_N']:.3f} | {gf['mesh_2p3_peak']['Fr_N']:.3f} | {gf['mesh_2p3_peak']['Fn_N']:.3f} |

其中第二级啮合力明显大于第一级，这是因为齿轮减速后输出轴扭矩增大。

---

## 8. 结论

1. 该扑翼结构可按单自由度平面机构处理，单电机输入即可驱动周期扑翼。
2. 两级定轴轮系总传动比为 `{gs['i_total']:.3f}`，实现降速增扭，输出轴与输入轴同向。
3. 扑翼机构将连续旋转转换为翅膀往复摆动，当前原始摆幅约 `{kin['raw_span_deg']:.1f} deg`。
4. 输出运动不是匀速摆动，换向附近角加速度较大，因此结构受力分析需要计入惯性力矩。
5. 翅膀载荷可等效为翅根气动力矩与惯性力矩，并沿“翅膀 -> 摇杆 -> 连杆 -> 曲柄 -> 齿轮 -> 电机”的路径传递。
6. 按功率等效法估算，峰值输出扭矩约 `{pt['t_out_peak_Nm']*1000:.2f} N.mm`，峰值电机扭矩约 `{pt['t_motor_peak_Nm']*1000:.3f} N.mm`。
7. 峰值工况下，第二级齿轮啮合圆周力约 `{gf['mesh_2p3_peak']['Ft_N']:.2f} N`，可作为后续轴、轴承和机架受力分析的输入。

---

## 9. 生成文件

计算摘要 JSON:

```text
{summary_json.relative_to(PROJECT_ROOT)}
```

新增图表:

```text
{figure_paths['force_flow'].relative_to(PROJECT_ROOT)}
{figure_paths['wing'].relative_to(PROJECT_ROOT)}
{figure_paths['linkage'].relative_to(PROJECT_ROOT)}
{figure_paths['gear'].relative_to(PROJECT_ROOT)}
{figure_paths['gear_annotated'].relative_to(PROJECT_ROOT)}
{figure_paths['torque'].relative_to(PROJECT_ROOT)}
```
"""

    path.write_text(md, encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    results = compute_results()
    figure_paths = {
        "force_flow": plot_system_force_flow(results),
        "wing": plot_wing_fbd(results),
        "linkage": plot_linkage_force_diagram(results),
        "gear": plot_gear_force_diagram(results),
        "gear_annotated": plot_annotated_existing_gear_motion_diagram(results),
        "torque": plot_output_torque_time(results),
    }
    summary_json = write_summary_json(results)
    report = generate_report(results, figure_paths, summary_json)

    print("Generated mechanical-principles analysis:")
    print(f"  {report}")
    print(f"  {summary_json}")
    for p in figure_paths.values():
        print(f"  {p}")


if __name__ == "__main__":
    main()
