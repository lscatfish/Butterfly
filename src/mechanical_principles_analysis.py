#!/usr/bin/env python3
"""
仿生蝴蝶扑翼结构的机械原理分析脚本。

本脚本把项目中已有的气动力、四杆机构运动学和两级齿轮传动结果
整理为机械原理课程作业所需的等效运动分析、力矩传递分析和曲线图。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from butterfly_forces import (
    SimulationConfig,
    WingGeometry,
    compute_cop_vec,
    compute_wing_forces_vec,
    rocker_decompose,
)
from gear_analysis import FixedAxisGearTrain
from mechanism import wing_kinematics


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


def baseline_wing_geometry() -> dict:
    """Return the documented baseline wing geometry in SI and report units."""
    items = {
        "Front": {"S": 0.01617, "R": 0.1543, "c_avg": 0.1048, "r1": 0.4227, "r2_sq": 0.2382},
        "Back": {"S": 0.01554, "R": 0.1474, "c_avg": 0.1054, "r1": 0.4798, "r2_sq": 0.2876},
    }
    for item in items.values():
        item["S_mm2"] = item["S"] * 1e6
        item["R_mm"] = item["R"] * 1000.0
        item["c_avg_mm"] = item["c_avg"] * 1000.0
        item["AR"] = item["R"] ** 2 / item["S"]
    return items


def to_wing_geometry(name: str, item: dict) -> WingGeometry:
    return WingGeometry(
        name=name,
        S=item["S"],
        R=item["R"],
        c_avg=item["c_avg"],
        r1=item["r1"],
        r2_sq=item["r2_sq"],
        AR=item["AR"],
    )


def periodic_hann_smooth(values: np.ndarray, window: int = 21) -> np.ndarray:
    """Smooth one periodic cycle without creating endpoint edge artifacts."""
    if window <= 2:
        return np.asarray(values, dtype=float).copy()
    if window % 2 == 0:
        window += 1
    values = np.asarray(values, dtype=float)
    if window >= len(values):
        return values.copy()

    kernel = np.hanning(window)
    kernel /= kernel.sum()
    pad = window // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]])
    return np.convolve(padded, kernel, mode="valid")


def simulate_mechanical_wing(
    name: str,
    geo_item: dict,
    wing_geo: WingGeometry,
    cfg: SimulationConfig,
    phi: np.ndarray,
    phi_dot: np.ndarray,
    phi_ddot: np.ndarray,
    alpha_install_deg: float,
    x_wing: float,
) -> dict:
    """Map the new force model into the simplified fields used by this report."""
    theta_p = np.zeros_like(phi)
    theta_dot = np.zeros_like(phi)
    force_result = compute_wing_forces_vec(
        phi,
        phi_dot,
        phi_ddot,
        theta_p,
        theta_dot,
        alpha_install_deg,
        wing_geo,
        x_wing,
        cfg,
    )
    cop = compute_cop_vec(phi, wing_geo, x_wing, y_hinge=0.010, z_hinge=0.0, side_sign=1.0)
    _, rocker_moment, _ = rocker_decompose(
        force_result["F_body"], cop, phi, cfg, x_wing, y_hinge=0.010
    )

    r_eff = geo_item["R"] * geo_item["r1"]
    m_wing = cfg.m_total / 4.0
    i_w = m_wing * geo_item["R"] ** 2 * geo_item["r2_sq"]
    m_aero_raw = rocker_moment[:, 1]
    m_aero = periodic_hann_smooth(m_aero_raw, window=21)
    m_inertial = i_w * phi_ddot
    p_aero = m_aero * phi_dot
    p_inertial = m_inertial * phi_dot

    return {
        "name": name,
        "t": np.linspace(0.0, 1.0 / cfg.f, len(phi), endpoint=False),
        "phi": phi,
        "phi_dot": phi_dot,
        "phi_ddot": phi_ddot,
        "F_body": force_result["F_body"],
        "F_drag": np.divide(m_aero, r_eff, out=np.zeros_like(m_aero), where=r_eff != 0.0),
        "R_equiv": r_eff,
        "I_w": i_w,
        "M_aero_raw": m_aero_raw,
        "M_aero": m_aero,
        "M_inertial": m_inertial,
        "P_aero": p_aero,
        "P_inertial": p_inertial,
        "P_total": p_aero + p_inertial,
        "alpha_eff_deg": force_result["alpha_eff_deg"],
        "C_L": force_result["C_L"],
        "C_D": force_result["C_D"],
    }


def four_wing_series(front: dict, back: dict) -> dict:
    """Combine one front wing and one back wing into four-wing totals."""
    return {
        "P_aero": 2.0 * (front["P_aero"] + back["P_aero"]),
        "P_inertial": 2.0 * (front["P_inertial"] + back["P_inertial"]),
        "P_total": 2.0 * (front["P_total"] + back["P_total"]),
        "M_aero": 2.0 * (front["M_aero"] + back["M_aero"]),
        "M_inertial": 2.0 * (front["M_inertial"] + back["M_inertial"]),
    }


def compute_results() -> dict:
    cfg = SimulationConfig(
        alpha_front_deg=60.0,
        alpha_back_deg=10.0,
        phase_diff_deg=-20.0,
        mech_a=6.0,
        mech_R=2.25,
        phi_offset_deg=-30.0,
        f=17.0,
        rotation="cw",
        c_damp=5e-4,
        t_end=1.0 / 17.0,
        dt=(1.0 / 17.0) / 720.0,
        steady_start=0.0,
    )
    params = {"f": cfg.f, "m_total": cfg.m_total, "rho": cfg.rho}
    geo = baseline_wing_geometry()
    front_geo = to_wing_geometry("Front", geo["Front"])
    back_geo = to_wing_geometry("Back", geo["Back"])

    gear = FixedAxisGearTrain()
    mech_t, mech_phi, mech_phi_dot, mech_phi_ddot, mech_info = wing_kinematics(
        f=cfg.f,
        a=cfg.mech_a,
        phi_offset_deg=cfg.phi_offset_deg,
        rotation=cfg.rotation,
        params={"a": cfg.mech_a, "b": cfg.mech_b, "R": cfg.mech_R, "c": cfg.mech_c, "l": cfg.mech_l},
        n_points=720,
    )
    phase_sec_b = math.radians(cfg.phase_diff_deg) / (2.0 * math.pi) * (1.0 / cfg.f)
    back_t_eff = np.mod(mech_t + phase_sec_b, 1.0 / cfg.f)
    back_phi = np.interp(back_t_eff, mech_t, mech_phi, period=1.0 / cfg.f)
    back_phi_dot = np.interp(back_t_eff, mech_t, mech_phi_dot, period=1.0 / cfg.f)
    back_phi_ddot = np.interp(back_t_eff, mech_t, mech_phi_ddot, period=1.0 / cfg.f)
    front = simulate_mechanical_wing(
        "Front",
        geo["Front"],
        front_geo,
        cfg,
        mech_phi,
        mech_phi_dot,
        mech_phi_ddot,
        cfg.alpha_front_deg,
        cfg.x_front,
    )
    back = simulate_mechanical_wing(
        "Back",
        geo["Back"],
        back_geo,
        cfg,
        back_phi,
        back_phi_dot,
        back_phi_ddot,
        cfg.alpha_back_deg,
        cfg.x_back,
    )
    combined = four_wing_series(front, back)

    mech = {
        "t": mech_t,
        "phi": mech_phi,
        "phi_dot": mech_phi_dot,
        "phi_ddot": mech_phi_ddot,
        "span_deg": mech_info["phi_span_deg"],
        "phi_min_rad": float(np.min(mech_phi)),
        "phi_max_rad": float(np.max(mech_phi)),
    }

    f = params["f"]
    omega_crank = 2.0 * math.pi * f
    i_total = abs(gear.i_total)
    eta_gear = gear.eta_total
    eta_linkage = 0.90
    eta_system = eta_gear * eta_linkage

    p_aero_avg = float(np.mean(combined["P_aero"]))
    p_total_peak = float(np.max(np.abs(combined["P_total"])))
    p_inertial_abs_avg = float(np.mean(np.abs(combined["P_inertial"])))
    p_inertial_peak = float(np.max(np.abs(combined["P_inertial"])))

    t_out_ideal_avg = p_aero_avg / omega_crank
    t_out_ideal_peak = p_total_peak / omega_crank
    t_out_avg = t_out_ideal_avg / eta_linkage
    t_out_peak = t_out_ideal_peak / eta_linkage
    t_motor_avg = t_out_avg / (i_total * eta_gear)
    t_motor_peak = t_out_peak / (i_total * eta_gear)

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

    n = len(mech_t)
    crank_angle_deg = np.linspace(0.0, 360.0, n, endpoint=False)
    m_wing_total = combined["M_aero"] + combined["M_inertial"]
    t_crank_ideal = combined["P_total"] / omega_crank
    t_crank = t_crank_ideal / eta_linkage
    t_motor = t_crank / (i_total * eta_gear)
    ft1 = 2.0 * np.abs(t_motor) / d1_m
    fr1 = ft1 * math.tan(pressure_angle)
    fn1 = ft1 / math.cos(pressure_angle)
    ft2 = 2.0 * np.abs(t_crank) / d3_m
    fr2 = ft2 * math.tan(pressure_angle)
    fn2 = ft2 / math.cos(pressure_angle)

    weight_n = params["m_total"] * 9.81

    return {
        "params": params,
        "config": {
            "alpha_front_deg": cfg.alpha_front_deg,
            "alpha_back_deg": cfg.alpha_back_deg,
            "phase_diff_deg": cfg.phase_diff_deg,
            "mech_a": cfg.mech_a,
            "mech_b": cfg.mech_b,
            "mech_R": cfg.mech_R,
            "mech_c": cfg.mech_c,
            "mech_l": cfg.mech_l,
            "phi_offset_deg": cfg.phi_offset_deg,
            "rotation": cfg.rotation,
            "c_damp": cfg.c_damp,
            "aero_moment_smoothing": "periodic_hann",
            "aero_moment_smoothing_window_points": 21,
            "aero_moment_smoothing_window_deg": 21 * 360.0 / len(mech_t),
        },
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
            "t_out_ideal_avg_Nm": t_out_ideal_avg,
            "t_out_ideal_peak_Nm": t_out_ideal_peak,
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
        "load_chain": {
            "crank_angle_deg": crank_angle_deg,
            "M_aero_Nm": combined["M_aero"],
            "M_inertial_Nm": combined["M_inertial"],
            "M_wing_total_Nm": m_wing_total,
            "P_total_W": combined["P_total"],
            "T_crank_ideal_Nm": t_crank_ideal,
            "T_crank_Nm": t_crank,
            "T_motor_Nm": t_motor,
            "Ft12_N": ft1,
            "Fr12_N": fr1,
            "Fn12_N": fn1,
            "Ft2p3_N": ft2,
            "Fr2p3_N": fr2,
            "Fn2p3_N": fn2,
        },
        "gear_summary": {
            "i_total": i_total,
            "eta_total": eta_gear,
            "eta_linkage": eta_linkage,
            "eta_system": eta_system,
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
    fig = plt.figure(figsize=(9.5, 12.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.35, 1.0, 1.0, 1.0], hspace=0.45)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]

    def setup(ax, xlim=(-1.4, 6.2), ylim=(-1.0, 4.1)) -> None:
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    def joint(ax, xy, r=0.09) -> None:
        ax.add_patch(plt.Circle(xy, r, facecolor="white", edgecolor="#222222", lw=2.0, zorder=5))

    def ground_pin(ax, xy) -> None:
        x, y = xy
        joint(ax, xy)
        ax.plot([x - 0.45, x + 0.45], [y - 0.28, y - 0.28], color="#222222", lw=2.0)
        for i in range(5):
            xi = x - 0.38 + i * 0.18
            ax.plot([xi, xi + 0.15], [y - 0.45, y - 0.28], color="#222222", lw=1.2)

    def wall_pin(ax, xy) -> None:
        x, y = xy
        joint(ax, xy)
        ax.plot([x - 0.28, x - 0.28], [y - 0.45, y + 0.45], color="#222222", lw=2.0)
        for i in range(5):
            yi = y - 0.38 + i * 0.18
            ax.plot([x - 0.45, x - 0.28], [yi - 0.12, yi], color="#222222", lw=1.2)

    def force(ax, text, start, end, color="#333333", fs=12) -> None:
        ax.annotate(
            text,
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=2.2, color=color),
            color=color,
            fontsize=fs,
            ha="center",
            va="center",
        )

    def moment(ax, text, xy, text_xy, rad=0.55, color="#d62728") -> None:
        ax.annotate(
            text,
            xy=xy,
            xytext=text_xy,
            arrowprops=dict(arrowstyle="->", lw=2.2, color=color, connectionstyle=f"arc3,rad={rad}"),
            color=color,
            fontsize=13,
            ha="center",
            va="center",
        )

    # Textbook-like reference geometry.
    o1 = np.array([2.15, 0.45])   # fixed crank pivot
    a = np.array([3.0, 1.25])     # crank-link joint
    b = np.array([4.65, 2.35])    # link-rocker joint
    o3 = np.array([3.9, 3.35])    # rocker fixed pivot

    ax = axes[0]
    setup(ax, xlim=(1.0, 5.7), ylim=(0.0, 4.05))
    ground_pin(ax, o1)
    wall_pin(ax, o3)
    ax.plot([o1[0], a[0]], [o1[1], a[1]], color="#1f77b4", lw=3.2)
    ax.plot([a[0], b[0]], [a[1], b[1]], color="#ff7f0e", lw=3.2)
    ax.plot([o3[0], b[0]], [o3[1], b[1]], color="#2ca02c", lw=3.2)
    joint(ax, a)
    joint(ax, b)
    ax.text((o1[0] + a[0]) / 2 - 0.15, (o1[1] + a[1]) / 2 - 0.15, "1", fontsize=13)
    ax.text((a[0] + b[0]) / 2 + 0.15, (a[1] + b[1]) / 2 + 0.12, "2", fontsize=13)
    ax.text((o3[0] + b[0]) / 2 - 0.35, (o3[1] + b[1]) / 2, "3", fontsize=13)
    ax.text(o1[0], o1[1] - 0.65, "4", fontsize=13, ha="center")
    ax.text(o1[0] - 0.05, o1[1] + 0.3, "O1", fontsize=10, ha="right")
    ax.text(o3[0] - 0.1, o3[1] + 0.35, "O3", fontsize=10, ha="right")
    moment(ax, "外载力矩 M", b, (4.95, 3.65), rad=-0.55)
    ax.set_title("曲柄摇杆机构简图", fontsize=15, fontweight="bold")

    # Link 2: two-force member.
    ax = axes[1]
    setup(ax, xlim=(0.55, 5.8), ylim=(-0.25, 2.45))
    p1 = np.array([1.6, 0.55])
    p2 = np.array([4.8, 1.75])
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#ff7f0e", lw=3.2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color="#888888", lw=1.2)
    joint(ax, p1)
    joint(ax, p2)
    direction = (p2 - p1) / np.linalg.norm(p2 - p1)
    force(ax, "R12", p1 - direction * 0.95 + np.array([0.0, -0.12]), p1 - direction * 0.15, "#333333")
    force(ax, "R32", p2 + direction * 0.75 + np.array([0.08, 0.22]), p2 + direction * 0.15, "#333333")
    ax.text(3.1, 2.12, "构件2为二力杆: R12 与 R32 等大、反向、共线", fontsize=11, ha="center")
    ax.set_title("构件 2 受力图", fontsize=14, fontweight="bold")

    # Rocker 3: load moment plus two reactions.
    ax = axes[2]
    setup(ax, xlim=(0.35, 5.25), ylim=(-0.25, 2.95))
    q_fixed = np.array([1.5, 0.65])
    q_joint = np.array([3.9, 2.25])
    ax.plot([q_fixed[0], q_joint[0]], [q_fixed[1], q_joint[1]], color="#2ca02c", lw=3.2)
    wall_pin(ax, q_fixed)
    joint(ax, q_joint)
    force(ax, "R23", q_joint + np.array([1.0, -0.75]), q_joint + np.array([0.18, -0.12]), "#333333")
    force(ax, "R43", q_fixed + np.array([-0.9, 0.65]), q_fixed + np.array([-0.15, 0.12]), "#333333")
    moment(ax, "M", q_joint + np.array([0.05, 0.05]), q_joint + np.array([0.55, 0.95]), rad=-0.55)
    ax.text(3.25, 0.25, "R23 = -R32", fontsize=11)
    ax.set_title("构件 3 受力图", fontsize=14, fontweight="bold")

    # Crank 1: driving moment plus two reactions.
    ax = axes[3]
    setup(ax, xlim=(0.45, 4.55), ylim=(-0.5, 2.8))
    c_fixed = np.array([1.55, 0.65])
    c_joint = np.array([3.15, 1.85])
    ax.plot([c_fixed[0], c_joint[0]], [c_fixed[1], c_joint[1]], color="#1f77b4", lw=3.2)
    ground_pin(ax, c_fixed)
    joint(ax, c_joint)
    force(ax, "R21", c_joint + np.array([0.9, 0.65]), c_joint + np.array([0.14, 0.1]), "#333333")
    force(ax, "R41", c_fixed + np.array([-0.65, -0.55]), c_fixed + np.array([-0.12, -0.1]), "#333333")
    moment(ax, "驱动力矩 Md", c_fixed + np.array([0.35, 0.35]), c_fixed + np.array([1.35, 0.05]), rad=0.55, color="#1f77b4")
    ax.text(3.2, 0.25, "R21 = -R12", fontsize=11)
    ax.set_title("构件 1 受力图", fontsize=14, fontweight="bold")

    pt = results["power_torque"]
    fig.text(
        0.5,
        0.018,
        f"载荷换算: 外载力矩 M 为翅根等效气动/惯性力矩；驱动力矩 Md 由齿轮输出轴提供。"
        f" 峰值输出扭矩约 {pt['t_out_peak_Nm']*1000:.2f} N.mm。",
        ha="center",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="#cccccc", pad=5),
    )
    fig.suptitle("曲柄摇杆机构拆构件受力分析图", fontsize=16, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.95, bottom=0.07, hspace=0.72)
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


def plot_crank_rocker_reference_crops() -> dict[str, Path]:
    src = FIG_DIR / "曲柄摇杆受力简图.png"
    paths = {
        "mechanism": FIG_DIR / "crank_rocker_reference.png",
        "fbd": FIG_DIR / "crank_rocker_fbd_reference.png",
    }
    if not src.exists():
        return {}

    img = plt.imread(src)
    h = img.shape[0]
    crops = {
        "mechanism": img[: int(h * 0.32), :, :],
        "fbd": img[int(h * 0.30):, :, :],
    }

    for key, crop in crops.items():
        fig, ax = plt.subplots(figsize=(4.0, 5.6 if key == "fbd" else 3.0))
        ax.imshow(crop)
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(paths[key], dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    return paths


def plot_mechanism_schematic_reference() -> Path | None:
    src = FIG_DIR / "机构简图.png"
    if not src.exists():
        return None
    path = FIG_DIR / "mechanism_schematic_reference.png"
    img = plt.imread(src)
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def plot_rocker_moment_vs_crank_angle(results: dict) -> Path:
    path = FIG_DIR / "rocker_moment_vs_crank_angle.png"
    lc = results["load_chain"]
    theta = lc["crank_angle_deg"]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(theta, lc["M_aero_Nm"] * 1000, color="#d62728", lw=1.8, label="气动等效力矩")
    ax.plot(theta, lc["M_inertial_Nm"] * 1000, color="#9467bd", lw=1.8, label="惯性等效力矩")
    ax.plot(theta, lc["M_wing_total_Nm"] * 1000, color="#111111", lw=2.2, label="摇杆外载合力矩")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlim(0, 360)
    ax.set_xlabel("曲柄转角 θ (deg)")
    ax.set_ylabel("力矩 (N.mm)")
    ax.set_title("摇杆端等效外载力矩随曲柄转角变化")
    ax.grid(True, alpha=0.28)
    ax.legend(ncol=3, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_torque_chain_vs_crank_angle(results: dict) -> Path:
    path = FIG_DIR / "torque_chain_vs_crank_angle.png"
    lc = results["load_chain"]
    theta = lc["crank_angle_deg"]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    axes[0].plot(theta, lc["T_crank_Nm"] * 1000, color="#111111", lw=2.0)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_ylabel("曲柄轴扭矩\n(N.mm)")
    axes[0].set_title("曲柄轴与电机轴扭矩随曲柄转角变化")
    axes[0].grid(True, alpha=0.28)

    axes[1].plot(theta, lc["T_motor_Nm"] * 1000, color="#1f77b4", lw=2.0)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_xlim(0, 360)
    axes[1].set_xlabel("曲柄转角 θ (deg)")
    axes[1].set_ylabel("电机轴扭矩\n(N.mm)")
    axes[1].grid(True, alpha=0.28)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gear_mesh_forces_vs_crank_angle(results: dict) -> Path:
    path = FIG_DIR / "gear_mesh_forces_vs_crank_angle.png"
    lc = results["load_chain"]
    theta = lc["crank_angle_deg"]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=True)
    series = [
        (axes[0], "第一级啮合：齿轮 1-2", "Ft12_N", "Fr12_N", "Fn12_N"),
        (axes[1], "第二级啮合：齿轮 2'-3", "Ft2p3_N", "Fr2p3_N", "Fn2p3_N"),
    ]
    for ax, title, ft, fr, fn in series:
        ax.plot(theta, lc[ft], color="#d62728", lw=1.8, label="圆周力 Ft")
        ax.plot(theta, lc[fr], color="#1f77b4", lw=1.6, label="径向力 Fr")
        ax.plot(theta, lc[fn], color="#2ca02c", lw=1.6, label="法向力 Fn")
        ax.set_title(title)
        ax.set_ylabel("力 (N)")
        ax.grid(True, alpha=0.28)
        ax.legend(ncol=3, fontsize=10)

    axes[-1].set_xlim(0, 360)
    axes[-1].set_xlabel("曲柄转角 θ (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary_json(results: dict) -> Path:
    path = OUTPUT_DIR / "tables" / "mechanical_principles_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "config": results["config"],
        "kinematics": results["kinematics"],
        "power_torque": results["power_torque"],
        "gear_forces": results["gear_forces"],
        "gear_summary": results["gear_summary"],
        "load_chain_peaks": {
            "M_wing_peak_Nmm": float(np.max(np.abs(results["load_chain"]["M_wing_total_Nm"])) * 1000),
            "T_crank_peak_Nmm": float(np.max(np.abs(results["load_chain"]["T_crank_Nm"])) * 1000),
            "T_motor_peak_Nmm": float(np.max(np.abs(results["load_chain"]["T_motor_Nm"])) * 1000),
            "Ft12_peak_N": float(np.max(results["load_chain"]["Ft12_N"])),
            "Ft2p3_peak_N": float(np.max(results["load_chain"]["Ft2p3_N"])),
        },
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


def generate_report_v2(results: dict, figure_paths: dict[str, Path], summary_json: Path) -> Path:
    path = REPORT_DIR / "机械原理结构系统分析.md"
    p = results["params"]
    cfg = results["config"]
    kin = results["kinematics"]
    pt = results["power_torque"]
    gs = results["gear_summary"]
    gf = results["gear_forces"]
    geo = results["geo"]

    md = f"""# 仿生蝴蝶扑翼结构系统机械原理分析

> 分析脚本：`src/mechanical_principles_analysis.py`  
> 依据文件：`docs/mechanism.md`、`docs/gear_analysis.md`、`docs/butterfly_forces_使用说明.md`、`docs/v6_6_cartesian_sweep_report.md`  
> 分析定位：机械原理课程作业。本文重点说明机构组成、自由度、轮系传动、曲柄摇杆运动转换、构件受力和等效扭矩，不讨论真实飞行可行性。

---

## 1. 系统组成和传动路线

该装置可以按“电机 - 齿轮减速 - 曲柄摇杆 - 翅膀负载”的路线理解：

```text
电机连续转动
→ 两级定轴外啮合齿轮减速器
→ 曲柄 BP1 作匀速圆周运动
→ 连杆 P1P2 传递运动和力
→ 摇杆 AP2 带动翅膀绕 A 点往复摆动
→ 翅膀气动力、惯性力矩反作用到摇杆和传动系统
```

机械原理分析的核心是把翅膀侧的等效外载力矩施加到摇杆端，再反推连杆、曲柄、齿轮和电机需要承受的力与扭矩。

![系统力流图](../figures/{figure_paths['force_flow'].name})

---

## 2. 曲柄摇杆机构参数含义

根据 `docs/mechanism.md` 和 `src/mechanism.py`，机构坐标定义如下：

| 点 | 坐标 | 物理含义 |
|---|---|---|
| A | `(0, a)` | 翅膀转轴，也是摇杆 AP2 的固定铰链 |
| B | `(b, 0)` | 曲柄 BP1 的固定转轴 |
| P1 | `B + R(cosθ, sinθ)` | 曲柄端点，即主动点 |
| P2 | 由 `AP2 = l` 和 `P1P2 = c` 的几何约束求出 | 摇杆端点，也是连杆与摇杆铰接点 |

各参数的正确含义为：

| 参数 | 本次取值 | 单位 | 正确含义 |
|---|---:|---|---|
| `a` / `mech_a` | {cfg['mech_a']:.2f} | mm | A 点 y 坐标，即摇杆/翅膀转轴相对 B 点水平基准线的高度 |
| `b` / `mech_b` | {cfg['mech_b']:.2f} | mm | B 点 x 坐标，即曲柄固定转轴的水平位置 |
| `R` / `mech_R` | {cfg['mech_R']:.2f} | mm | 曲柄 BP1 的半径 |
| `c` / `mech_c` | {cfg['mech_c']:.2f} | mm | 连杆 P1P2 长度 |
| `l` / `mech_l` | {cfg['mech_l']:.2f} | mm | 摇杆 AP2 长度 |
| `phi_offset_deg` | {cfg['phi_offset_deg']:.1f} | deg | 翅膀安装偏角，用来把机构输出角转换为实际翅膀拍动角 |

特别注意：`a` 不是曲柄半径，也不是主动点圆心的 x 坐标；`R` 才是曲柄半径。后续运动学和受力分析均按这个定义进行。

---

## 3. 自由度分析

将扑翼执行部分按平面曲柄摇杆四杆机构处理：

| 项目 | 数值 | 说明 |
|---|---:|---|
| 活动构件数 `n` | 3 | 曲柄、连杆、摇杆 |
| 低副数 `PL` | 4 | A、B、P1、P2 四个转动副 |
| 高副数 `PH` | 0 | 等效四杆机构中不计高副 |

平面机构自由度公式：

```text
F = 3n - 2PL - PH = 3×3 - 2×4 - 0 = 1
```

结论：扑翼机构为单自由度机构。只要电机通过齿轮系给曲柄一个连续转动输入，摇杆和翅膀的往复摆动就由几何约束唯一决定。

---

## 4. 轮系传动分析

轮系采用两级定轴外啮合圆柱齿轮传动。原始轮系简图如下：

![轮系运动简图](../fig/齿轮运动简图.png)

齿数关系为：

```text
z1 = 7, z2 = 40, z2' = 7, z3 = 40
```

齿轮 2 和齿轮 2' 固连在同一根中间轴上，所以二者角速度相同。两级均为外啮合：

```text
i12  = -z2/z1  = -40/7
i2'3 = -z3/z2' = -40/7
i13  = i12 · i2'3 = (+)1600/49 = 32.653
```

两个负号相乘为正，因此输出轴与输入轴转向相同。主要结果为：

| 项目 | 数值 |
|---|---:|
| 总传动比 `|i13|` | {gs['i_total']:.4f} |
| 总效率 `eta` | {gs['eta_total']:.4f} |
| 模数 `m` | 0.3 mm |
| 压力角 `alpha` | {gs['alpha_deg']:.1f} deg |
| 齿轮 1 分度圆直径 | {gs['d1_mm']:.3f} mm |
| 齿轮 3 分度圆直径 | {gs['d3_mm']:.3f} mm |

这部分重点写清楚三件事：轮系为定轴轮系；2 和 2' 为双联齿轮、同轴同速；两级外啮合导致总传动比为正，达到降速增扭的目的。

---

## 5. 运动学分析

本次按项目当前推荐的 v6.6 设计参数计算：`a=6 mm, R=2.25 mm, phi_offset=-30 deg, f=17 Hz, rotation=cw`。

`mechanism.py` 的求解过程是：给定曲柄角 `theta` 后，先求 P1 点位置，再由三角形 A-P1-P2 的几何约束求 P2 点，最后得到摇杆角 `phi`。该机构的输出不是正弦假设，而是由实际四杆几何关系决定。

| 项目 | 数值 |
|---|---:|
| 扑翼频率 | {p['f']:.1f} Hz |
| 曲柄角速度 | {kin['omega_crank_rad_s']:.2f} rad/s |
| 周期 | {kin['period_s']*1000:.2f} ms |
| 翅膀角范围 | {kin['phi_min_deg']:.2f} deg ~ {kin['phi_max_deg']:.2f} deg |
| 总摆幅 | {kin['raw_span_deg']:.2f} deg |
| 峰值角速度 | {kin['front_phi_dot_peak']:.2f} rad/s |
| 峰值角加速度 | {kin['front_phi_ddot_peak']:.0f} rad/s^2 |

这说明即使曲柄匀速转动，摇杆输出仍是非匀速摆动。换向附近角加速度较大，因此做受力分析时不能只考虑气动力，还要考虑翅膀和摇杆等效转动惯性带来的惯性力矩。

---

## 6. 翅膀外载的等效处理

翅膀几何参数采用项目文档中记录的 DXF 分析基准值：

| 翅膀 | 面积 S(mm²) | 展长 Rw(mm) | 平均弦长(mm) | r1 | r2_sq |
|---|---:|---:|---:|---:|---:|
| 前翅 | {geo['Front']['S_mm2']:.1f} | {geo['Front']['R_mm']:.1f} | {geo['Front']['c_avg_mm']:.1f} | {geo['Front']['r1']:.4f} | {geo['Front']['r2_sq']:.4f} |
| 后翅 | {geo['Back']['S_mm2']:.1f} | {geo['Back']['R_mm']:.1f} | {geo['Back']['c_avg_mm']:.1f} | {geo['Back']['r1']:.4f} | {geo['Back']['r2_sq']:.4f} |

这里为了避免和机构曲柄半径 `R` 混淆，表中把翅膀展长记为 `Rw`。在 `butterfly_forces.py` 中，翅膀气动力被进一步分解为体轴力、对重心力矩，以及对摇杆枢轴 A 的有效主矩 `rocker_principal_moment[:, 1]`。

机械原理分析中不放翅膀自由体图，只把上层气动模型给出的翅膀负载等效成作用在摇杆上的外载力矩：

```text
M_wing = M_aero_about_A + I_w · phi_ddot
M_aero_about_A 取自 rocker_principal_moment 的 Y 分量
I_w ≈ m_w · Rw² · r2_sq
```

这个等效量就是曲柄摇杆机构受力分析中的外载力矩 `M`。

---

## 7. 曲柄摇杆机构受力分析

根目录中的受力简图可作为本节的受力分析图使用：

![曲柄摇杆受力简图](../../曲柄摇杆受力简图.png)

按课本的拆构件方法，可作如下分析：

| 构件 | 受力特点 | 平衡关系 |
|---|---|---|
| 构件 2：连杆 P1P2 | 两端为转动副，忽略自重和惯性时可近似为二力杆 | `R12` 与 `R32` 等大、反向、共线 |
| 构件 3：摇杆 AP2 | 受连杆作用力 `R23`、机架反力 `R43` 和外载力矩 `M` | 由 `ΣFx=0, ΣFy=0, ΣM=0` 求铰链反力和外载平衡 |
| 构件 1：曲柄 BP1 | 受连杆反力 `R21`、机架反力 `R41` 和驱动力矩 `Md` | 驱动力矩 `Md` 用来克服由摇杆端传回的负载 |

作用反作用关系为：

```text
R23 = -R32
R21 = -R12
```

若要严格求每个铰链反力，需要在每一瞬时知道构件 1、2、3 的几何位置和惯性项。本报告先采用机械原理课程中常用的“等效功率/等效扭矩”方法估计驱动需求：

```text
T_output = P_output / omega_crank
T_motor  = T_output / (i_total · eta)
```

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

## 8. 齿轮啮合受力估算

齿轮啮合力按圆柱直齿轮的基本受力关系估算：

```text
Ft = 2T / d
Fr = Ft · tan(alpha)
Fn = Ft / cos(alpha)
```

其中 `Ft` 为圆周力，`Fr` 为径向力，`Fn` 为法向啮合力，压力角 `alpha = 20 deg`。

| 啮合副 | Ft(N) | Fr(N) | Fn(N) |
|---|---:|---:|---:|
| 齿轮 1 - 齿轮 2 | {gf['mesh_12_peak']['Ft_N']:.3f} | {gf['mesh_12_peak']['Fr_N']:.3f} | {gf['mesh_12_peak']['Fn_N']:.3f} |
| 齿轮 2' - 齿轮 3 | {gf['mesh_2p3_peak']['Ft_N']:.3f} | {gf['mesh_2p3_peak']['Fr_N']:.3f} | {gf['mesh_2p3_peak']['Fn_N']:.3f} |

第二级啮合力更大，是因为齿轮减速后输出端扭矩增大。后续如果要做轴和轴承的强度分析，应把两级啮合处的 `Ft`、`Fr` 作为中间轴和输出轴的外载输入。

---

## 9. 结论

1. 该扑翼执行机构可按单自由度曲柄摇杆机构分析，电机只需提供一个连续转动输入。
2. 机构参数中，`a` 是摇杆固定铰链 A 的 y 坐标，`R` 才是曲柄半径；这是本次修正的重点。
3. 轮系为两级定轴外啮合齿轮传动，2 和 2' 为双联齿轮，总传动比约为 `{gs['i_total']:.3f}`，输出与输入同向。
4. 曲柄匀速转动时，摇杆输出角速度和角加速度并不均匀，因此受力分析必须考虑惯性力矩。
5. 翅膀侧复杂气动力在机械原理层面等效为作用于摇杆的外载力矩 `M_wing`，再由摇杆、连杆、曲柄逐级传回齿轮和电机。
6. 按当前 v6.6 推荐设计参数估算，峰值输出扭矩约 `{pt['t_out_peak_Nm']*1000:.2f} N.mm`，峰值电机扭矩约 `{pt['t_motor_peak_Nm']*1000:.3f} N.mm`。
7. 峰值工况下第二级齿轮圆周力约 `{gf['mesh_2p3_peak']['Ft_N']:.2f} N`，是后续轴、轴承和机架受力分析的关键载荷。

---

## 10. 输出文件

```text
{summary_json.relative_to(PROJECT_ROOT)}
{figure_paths['force_flow'].relative_to(PROJECT_ROOT)}
{figure_paths['torque'].relative_to(PROJECT_ROOT)}
```
"""

    path.write_text(md, encoding="utf-8")
    return path


def generate_report_v3(results: dict, figure_paths: dict[str, Path], summary_json: Path) -> Path:
    path = REPORT_DIR / "机械原理结构系统分析.md"
    p = results["params"]
    cfg = results["config"]
    kin = results["kinematics"]
    pt = results["power_torque"]
    gs = results["gear_summary"]
    gf = results["gear_forces"]
    geo = results["geo"]
    peaks = {
        "M_wing": float(np.max(np.abs(results["load_chain"]["M_wing_total_Nm"])) * 1000),
        "T_crank": float(np.max(np.abs(results["load_chain"]["T_crank_Nm"])) * 1000),
        "T_motor": float(np.max(np.abs(results["load_chain"]["T_motor_Nm"])) * 1000),
    }

    md = f"""# 扑翼结构系统机械原理分析

## 一、分析对象

本结构的机械传动路线为：电机输入，经两级定轴齿轮减速后驱动曲柄，曲柄通过连杆带动摇杆摆动，摇杆再带动翅膀绕 A 点往复拍动。本文只分析结构系统的运动与受力传递，不讨论能否真实飞行。

```text
电机 → 两级定轴齿轮 → 曲柄 BP1 → 连杆 P1P2 → 摇杆 AP2 → 翅膀外载
```

---

## 二、曲柄摇杆机构

实际机构简图如下。图中 `R=2.25` 对应曲柄半径 BP1，`c=14` 对应连杆 P1P2，`a` 为 A 点相对基准线的高度。

![实际机构简图](../figures/{figure_paths['mechanism_schematic'].name if figure_paths.get('mechanism_schematic') else figure_paths['crank_ref'].name})

![曲柄摇杆机构参考图](../figures/{figure_paths['crank_ref'].name})

为便于机械原理受力分析，将实际机构抽象为下图所示的平面曲柄摇杆机构。机构简图中的中间固定转轴对应 B 点；曲柄端小圆点对应 P1；上方连杆与摇杆连接处对应 P2；左侧摇杆固定铰对应 A 点。

| 参数 | 数值 | 含义 |
|:---|---:|:---|
| `a` | {cfg['mech_a']:.2f} mm | A 点 y 坐标，即摇杆/翅膀转轴高度 |
| `b` | {cfg['mech_b']:.2f} mm | B 点 x 坐标，即曲柄固定轴水平位置 |
| `R` | {cfg['mech_R']:.2f} mm | 曲柄 BP1 半径 |
| `c` | {cfg['mech_c']:.2f} mm | 连杆 P1P2 长度 |
| `l` | {cfg['mech_l']:.2f} mm | 摇杆 AP2 长度 |
| `phi_offset` | {cfg['phi_offset_deg']:.1f}° | 翅膀安装偏角 |

其中 `a` 表示 A 点高度，`R` 才表示曲柄半径，这一点在后续计算中必须区分。

平面机构自由度为：

$$F = 3n - 2P_L - P_H = 3 \\times 3 - 2 \\times 4 - 0 = 1$$

因此，该扑翼执行机构为单自由度机构。电机只需提供一个连续转动输入，摇杆摆角由四杆几何关系唯一确定。

运动学计算采用当前推荐设计参数：`a=6 mm, R=2.25 mm, phi_offset=-30°，f=17 Hz`。计算结果如下：

| 项目 | 数值 |
|:---|---:|
| 曲柄角速度 | {kin['omega_crank_rad_s']:.2f} rad/s |
| 运动周期 | {kin['period_s']*1000:.2f} ms |
| 翅膀摆角范围 | {kin['phi_min_deg']:.2f}° ~ {kin['phi_max_deg']:.2f}° |
| 总摆幅 | {kin['raw_span_deg']:.2f}° |
| 峰值角速度 | {kin['front_phi_dot_peak']:.2f} rad/s |
| 峰值角加速度 | {kin['front_phi_ddot_peak']:.0f} rad/s² |

---

## 三、曲柄摇杆受力模型

![曲柄摇杆受力简图](../figures/{figure_paths['crank_fbd'].name})

按课本拆构件法，连杆 2 可近似看作二力杆，摇杆 3 承受翅膀传回的外载力矩 `M`，曲柄 1 承受驱动力矩 `Md`。图中各铰链处的红圈表示摩擦圆；考虑转动副摩擦时，铰链反力的作用线不再穿过销轴中心，而应与摩擦圆相切。若只作理想低副分析，可把摩擦圆半径近似取为 0，反力通过铰链中心。

各构件受力关系如下：

| 构件 | 主要受力 | 说明 |
|:---|:---|:---|
| 连杆 2 | `R12`、`R32` | 二力杆，二力等大、反向、共线 |
| 摇杆 3 | `R23`、`R43`、`M` | `M` 为翅膀侧等效外载力矩 |
| 曲柄 1 | `R21`、`R41`、`Md` | `Md` 为驱动曲柄所需力矩 |

作用反作用关系为：

$$R_{{23}}=-R_{{32}}, \\qquad R_{{21}}=-R_{{12}}$$

翅膀侧外载采用等效力矩表示：

$$M_{{wing}} = M_{{aero,A}} + I_w \\ddot\\phi$$

其中 `M_aero,A` 取自气动模型中关于摇杆枢轴 A 的主矩，`Iw` 为翅膀等效转动惯量。为避免气动分段公式造成不符合真实流场连续性的小尖角，绘图和峰值统计前对气动力矩采用周期 Hann 窗平滑处理。这样可把复杂翅膀受力简化成作用在摇杆上的外载力矩。

![摇杆端等效外载力矩](../figures/{figure_paths['rocker_moment'].name})

图中气动等效力矩随拍动方向和有效攻角变化而改变符号；惯性力矩主要由 `phi_ddot` 决定，因此在摇杆换向附近幅值较大。局部尖峰主要来自两点：一是四杆机构换向附近角加速度变化快；二是气动模型中拍动方向、附加质量和 clap 修正按分段公式计算，跨过分段边界时曲线会出现较陡变化。这些尖峰对应传动系统需要重点校核的冲击载荷区间。

---

## 四、轮系传动分析

![轮系运动简图](../fig/齿轮运动简图.png)

轮系为两级定轴外啮合齿轮传动。齿轮 2 与齿轮 2' 固连在同一中间轴上。

$$i_{{12}}=-\\frac{{z_2}}{{z_1}}=-\\frac{{40}}{{7}}=-5.7143$$

$$i_{{2'3}}=-\\frac{{z_3}}{{z_{{2'}}}}=-\\frac{{40}}{{7}}=-5.7143$$

$$i_{{13}}=i_{{12}}i_{{2'3}}=+32.6531$$

两级外啮合各改变一次转向，总传动比为正，所以输出轴与输入轴同向。

| 项目 | 数值 |
|:---|---:|
| 总传动比 `|i13|` | {gs['i_total']:.4f} |
| 轮系总效率 `η_g` | {gs['eta_total']:.4f} |
| 模数 `m` | 0.3 mm |
| 压力角 `α` | {gs['alpha_deg']:.1f}° |
| 齿轮 1 分度圆直径 | {gs['d1_mm']:.3f} mm |
| 齿轮 3 分度圆直径 | {gs['d3_mm']:.3f} mm |

---

## 五、力矩逐级传递

采用功率等效法从摇杆外载反推到电机端。摇杆侧瞬时功率为：

$$P = M_{{wing}}\\dot\\phi$$

简化考虑曲柄摇杆转动副摩擦、摩擦圆和装配损失，取曲柄摇杆等效效率 `η_l=0.90`。因此曲柄轴所需输入扭矩为：

$$T_{{crank}} = \\frac{{P}}{{\\omega_{{crank}}\\eta_l}}$$

齿轮减速器再折算到电机轴：

$$T_{{motor}} = \\frac{{T_{{crank}}}}{{i_{{13}}\\eta_g}} = \\frac{{P}}{{\\omega_{{crank}}i_{{13}}\\eta_l\\eta_g}}$$

计算得到：

| 项目 | 数值 |
|:---|---:|
| 摇杆外载合力矩峰值 | {peaks['M_wing']:.2f} N.mm |
| 曲柄摇杆等效效率 `η_l` | {gs['eta_linkage']:.2f} |
| 曲柄轴峰值扭矩 | {peaks['T_crank']:.2f} N.mm |
| 电机轴峰值扭矩 | {peaks['T_motor']:.3f} N.mm |
| 平均气动功率 | {pt['p_aero_avg_W']*1000:.1f} mW |
| 峰值总功率 | {pt['p_total_peak_W']*1000:.1f} mW |

![曲柄轴与电机轴扭矩](../figures/{figure_paths['torque_chain'].name})

两条曲线形状相同，是因为电机轴扭矩由曲柄轴扭矩按固定传动比和轮系效率折算得到；幅值变小，反映两级减速器在电机端具有“高转速、低扭矩”的输入特点。正负号表示该瞬时载荷对曲柄转动方向的助推或阻碍。

---

## 六、齿轮啮合力

齿轮啮合力按直齿圆柱齿轮基本关系估算：

$$F_t=\\frac{{2T}}{{d}}, \\qquad F_r=F_t\\tan\\alpha, \\qquad F_n=\\frac{{F_t}}{{\\cos\\alpha}}$$

其中，`Ft` 为圆周力，方向沿分度圆切线，是传递扭矩的主要分量；`Fr` 为径向力，方向沿两齿轮中心连线，会压向轴承和机架；`Fn` 为法向啮合力，方向沿齿廓接触法线，是齿面真实接触力，可分解为 `Ft` 和 `Fr`。

峰值工况下：

| 啮合副 | Ft | Fr | Fn |
|:---|---:|---:|---:|
| 齿轮 1-2 | {gf['mesh_12_peak']['Ft_N']:.3f} N | {gf['mesh_12_peak']['Fr_N']:.3f} N | {gf['mesh_12_peak']['Fn_N']:.3f} N |
| 齿轮 2'-3 | {gf['mesh_2p3_peak']['Ft_N']:.3f} N | {gf['mesh_2p3_peak']['Fr_N']:.3f} N | {gf['mesh_2p3_peak']['Fn_N']:.3f} N |

![齿轮啮合力随曲柄转角变化](../figures/{figure_paths['gear_forces_angle'].name})

第二级啮合力大于第一级，原因是减速后输出端扭矩增大。该结果可作为后续轴、轴承和机架受力分析的输入。

---

## 七、结论

该结构可按单自由度曲柄摇杆机构进行机械原理分析。翅膀气动与惯性载荷先等效为摇杆端外载力矩，再通过功率等效法和曲柄摇杆等效效率换算到曲柄轴，最后按齿轮传动比和轮系效率折算到电机端。当前设计参数下，曲柄轴峰值扭矩约 `{peaks['T_crank']:.2f} N.mm`，电机轴峰值扭矩约 `{peaks['T_motor']:.3f} N.mm`；齿轮第二级峰值圆周力约 `{gf['mesh_2p3_peak']['Ft_N']:.2f} N`。

输出数据见：

```text
{summary_json.relative_to(PROJECT_ROOT)}
```
"""

    path.write_text(md, encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    results = compute_results()
    reference_paths = plot_crank_rocker_reference_crops()
    mechanism_schematic = plot_mechanism_schematic_reference()
    figure_paths = {
        "force_flow": plot_system_force_flow(results),
        "mechanism_schematic": mechanism_schematic,
        "crank_ref": reference_paths["mechanism"],
        "crank_fbd": reference_paths["fbd"],
        "rocker_moment": plot_rocker_moment_vs_crank_angle(results),
        "torque_chain": plot_torque_chain_vs_crank_angle(results),
        "gear_forces_angle": plot_gear_mesh_forces_vs_crank_angle(results),
        "torque": plot_output_torque_time(results),
    }
    summary_json = write_summary_json(results)
    report = generate_report_v3(results, figure_paths, summary_json)

    print("Generated mechanical-principles analysis:")
    print(f"  {report}")
    print(f"  {summary_json}")
    for p in figure_paths.values():
        print(f"  {p}")


if __name__ == "__main__":
    main()
