#!/usr/bin/env python3
"""
稳定性分析绘图模块 — 独立于分析逻辑，仅读取文件。

读取 temp/stability/ 中的数据，生成基线图和单变量偏离图。
通过文件与分析模块通信，可独立运行。
"""
import json, sys, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ))

from src.aero.butterfly_forces import DESIGN_v69

STABILITY_DIR = _PROJ / "temp" / "stability"
OUT_DIR = _PROJ / "output" / "figures" / "stability"
BASELINE_DIR = STABILITY_DIR / "baseline"

# 基线参数 (与 stability_analysis.py / DESIGN_v69 保持一致)
BASELINE_CONFIG = {**DESIGN_v69, "dt": 50e-6, "t_end": 5.0, "steady_start": 3.0}

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配色
C = {
    "blue": "#2166ac", "red": "#b2182b", "green": "#4dac26",
    "orange": "#f4a582", "purple": "#762a83", "teal": "#008080",
    "grey": "#888888", "pink": "#d6604d",
}


def _load_npz(path: Path) -> dict:
    return dict(np.load(path))


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ============================================================
# Baseline 图 (5 张)
# ============================================================

def plot_baseline_overview(baseline_dir: Path = None, out_dir: Path = None):
    """图1: 全时程概览 — θ_p, θ̇_p, θ̈_p, Fz, Fx, M."""
    bd = baseline_dir or BASELINE_DIR
    od = out_dir or (OUT_DIR / "baseline")
    od.mkdir(parents=True, exist_ok=True)

    ts = _load_npz(bd / "timeseries.npz")
    sm = _load_json(bd / "summary.json")

    t = ts["t"]
    half = len(t) // 2
    weight_mN = sm["weight_mN"]

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle(f"Baseline 全时程概览 | L/W={sm['L/W']:.3f}  peak={sm['peak_theta_deg']:.1f}°  n90={sm['n_exceed_90']}",
                 fontsize=14, fontweight="bold")

    # (0,0) θ_p
    ax = axes[0, 0]
    ax.plot(t, np.rad2deg(ts["theta_p"]), color=C["blue"], lw=0.6)
    ax.axhline(90, color="r", ls="--", lw=0.6, alpha=0.4)
    ax.axhline(-90, color="r", ls="--", lw=0.6, alpha=0.4)
    ax.axhline(0, color="k", ls="--", lw=0.4)
    ax.set_ylabel(r"$\theta_p$ [°]")
    ax.set_title(f"Pitch Angle | steady peak={sm['peak_abs_thetadot_rads']:.1f}")
    ax.grid(True, alpha=0.3)

    # (0,1) θ̇_p
    ax = axes[0, 1]
    ax.plot(t, ts["theta_dot"], color=C["red"], lw=0.5, alpha=0.8)
    ax.axhline(0, color="k", ls="--", lw=0.4)
    ax.set_ylabel(r"$\dot{\theta}_p$ [rad/s]")
    td_mean_label = "$|\\dot{\\theta}_p|$"
    ax.set_title(f"Pitch Rate | mean{td_mean_label}={sm['mean_abs_thetadot_rads']:.1f} peak={sm['peak_abs_thetadot_rads']:.0f}")
    ax.grid(True, alpha=0.3)

    # (1,0) Fz
    ax = axes[1, 0]
    ax.plot(t, ts["Fz_body_total"] * 1000, color=C["green"], lw=0.4, alpha=0.7, label="Fz_body")
    ax.plot(t, ts["Fz_world_total"] * 1000, color=C["teal"], lw=0.4, alpha=0.7, label="Fz_world")
    ax.axhline(weight_mN, color="r", ls=":", alpha=0.5, label=f"Weight={weight_mN:.0f}mN")
    ax.axhline(0, color="k", ls="--", lw=0.4)
    ax.set_ylabel("Fz [mN]")
    ax.set_title(f"Lift | body mean={sm['mean_Fz_body_mN']:+.0f} world mean={sm['mean_Fz_world_mN']:+.0f}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,1) Fx
    ax = axes[1, 1]
    ax.plot(t, ts["Fx_body_total"] * 1000, color=C["purple"], lw=0.4, alpha=0.7)
    ax.axhline(0, color="k", ls="--", lw=0.4)
    ax.set_ylabel("Fx [mN]")
    ax.set_title(f"Thrust | body mean={sm['mean_Fx_body_mN']:+.0f}")
    ax.grid(True, alpha=0.3)

    # (2,0) Moments
    ax = axes[2, 0]
    ax.plot(t, ts["M_aero"] * 1e6, color=C["blue"], lw=0.4, alpha=0.7, label="M_aero")
    ax.plot(t, ts["M_grav"] * 1e6, color=C["red"], lw=0.4, alpha=0.5, label="M_grav")
    ax.plot(t, ts["M_damp"] * 1e6, color=C["grey"], lw=0.3, alpha=0.5, label="M_damp")
    ax.axhline(0, color="k", ls="--", lw=0.4)
    ax.set_ylabel("M [μN·m]")
    ax.set_title(f"Moments | M_aero mean={sm['mean_M_aero_uNm']:+.0f} peak={sm['peak_M_aero_uNm']:.0f}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (2,1) θ̈_p
    ax = axes[2, 1]
    ax.plot(t, ts["theta_ddot"], color=C["orange"], lw=0.4, alpha=0.7)
    ax.axhline(0, color="k", ls="--", lw=0.4)
    ax.set_ylabel(r"$\ddot{\theta}_p$ [rad/s²]")
    tdd_label = "$|\\ddot{\\theta}_p|$"
    ax.set_title(f"Pitch Accel | mean{tdd_label}={sm['mean_abs_thetaddot_rads2']:.0f}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = od / "baseline_overview.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_baseline_phase(baseline_dir: Path = None, out_dir: Path = None):
    """图2: 相图 θ̇_p vs θ_p."""
    bd = baseline_dir or BASELINE_DIR
    od = out_dir or (OUT_DIR / "baseline")
    od.mkdir(parents=True, exist_ok=True)

    ts = _load_npz(bd / "timeseries.npz")
    sm = _load_json(bd / "summary.json")
    t = ts["t"]
    tp = np.rad2deg(ts["theta_p"])
    td = ts["theta_dot"]
    half = len(t) // 2

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Baseline Phase Portrait | L/W={sm['L/W']:.3f} peak={sm['peak_theta_deg']:.1f}°",
                 fontsize=14, fontweight="bold")

    # 全时程, 按时间着色
    ax = axes[0]
    sc = ax.scatter(tp, td, c=t, s=0.3, alpha=0.5, cmap="viridis", linewidths=0)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel(r"$\theta_p$ [°]")
    ax.set_ylabel(r"$\dot{\theta}_p$ [rad/s]")
    ax.set_title("Full time (color=time)")
    plt.colorbar(sc, ax=ax, label="t [s]")

    # 稳态 zoom
    ax = axes[1]
    ax.plot(tp[half:], td[half:], ".", ms=0.5, alpha=0.4, color=C["purple"])
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel(r"$\theta_p$ [°]")
    ax.set_ylabel(r"$\dot{\theta}_p$ [rad/s]")
    ax.set_title("Steady state (last 50%)")

    plt.tight_layout()
    fp = od / "baseline_phase.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_baseline_wings(baseline_dir: Path = None, out_dir: Path = None):
    """图3: 四翅力时程."""
    bd = baseline_dir or BASELINE_DIR
    od = out_dir or (OUT_DIR / "baseline")
    od.mkdir(parents=True, exist_ok=True)

    ts = _load_npz(bd / "timeseries.npz")
    t = ts["t"]
    mask = t > (t[-1] - 0.3)  # 最后 0.3s

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Baseline — Per-wing Forces (last 0.3s zoom)", fontsize=14, fontweight="bold")

    for idx, (wn, color) in enumerate([("FL", C["blue"]), ("FR", C["red"]),
                                         ("BL", C["green"]), ("BR", C["orange"])]):
        ax = axes[idx // 2, idx % 2]
        ax.plot(t[mask] * 1000, ts[f"{wn}_Fz_body"][mask] * 1000,
                color=color, lw=0.8, alpha=0.8, label="Fz_body")
        ax.plot(t[mask] * 1000, ts[f"{wn}_Fx_body"][mask] * 1000,
                color="grey", lw=0.4, alpha=0.5, label="Fx_body")
        ax.axhline(0, color="k", ls="--", lw=0.4)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Force [mN]")
        ax.set_title(f"{wn} — Fz peak={np.max(np.abs(ts[f'{wn}_Fz_body'][mask]))*1000:.0f}mN")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = od / "baseline_wings.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_baseline_aero(baseline_dir: Path = None, out_dir: Path = None):
    """图4: α_eff, C_L, C_D 时程."""
    bd = baseline_dir or BASELINE_DIR
    od = out_dir or (OUT_DIR / "baseline")
    od.mkdir(parents=True, exist_ok=True)

    ts = _load_npz(bd / "timeseries.npz")
    t = ts["t"]
    mask = t > (t[-1] - 0.3)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle("Baseline — Aerodynamic Quantities (last 0.3s)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(t[mask] * 1000, ts["FL_alpha_eff"][mask], color=C["blue"], lw=0.6, label="Front")
    ax.plot(t[mask] * 1000, ts["BL_alpha_eff"][mask], color=C["red"], lw=0.6, label="Back")
    ax.axhline(0, color="k", ls="--", lw=0.4)
    ax.set_ylabel("α_eff [°]")
    ax.set_title("Effective AoA")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t[mask] * 1000, ts["FL_C_L"][mask], color=C["blue"], lw=0.5, label="Front C_L")
    ax.plot(t[mask] * 1000, ts["BL_C_L"][mask], color=C["red"], lw=0.5, label="Back C_L")
    ax.axhline(0, color="k", ls="--", lw=0.4)
    ax.set_ylabel("C_L")
    ax.set_title("Lift Coefficient")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t[mask] * 1000, ts["FL_C_D"][mask], color=C["blue"], lw=0.5, label="Front C_D")
    ax.plot(t[mask] * 1000, ts["BL_C_D"][mask], color=C["red"], lw=0.5, label="Back C_D")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("C_D")
    ax.set_title("Drag Coefficient")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = od / "baseline_aero.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_baseline_rocker(baseline_dir: Path = None, out_dir: Path = None):
    """图5: 摇杆主矢+主矩."""
    bd = baseline_dir or BASELINE_DIR
    od = out_dir or (OUT_DIR / "baseline")
    od.mkdir(parents=True, exist_ok=True)

    ts = _load_npz(bd / "timeseries.npz")
    t = ts["t"]
    mask = t > (t[-1] - 0.3)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Baseline — Rocker Forces & Moments (last 0.3s)", fontsize=14, fontweight="bold")

    colors = {"FL": C["blue"], "FR": C["red"], "BL": C["green"], "BR": C["orange"]}
    for idx, wn in enumerate(["FL", "FR", "BL", "BR"]):
        ax = axes[idx // 2, idx % 2]
        ax2 = ax.twinx()
        # 主矢大小 = sqrt(pv_x^2 + pv_z^2), 摇杆在 XZ 平面内
        pv_mag = np.sqrt(ts[f"{wn}_rocker_pv_x"][mask]**2 + ts[f"{wn}_rocker_pv_z"][mask]**2)
        ax.plot(t[mask] * 1000, pv_mag,
                color=colors[wn], lw=0.8, label="主矢 |F|")
        ax2.plot(t[mask] * 1000, ts[f"{wn}_rocker_pm_y"][mask] * 1000,
                 color=C["grey"], lw=0.4, alpha=0.7, label="主矩 My")
        ax.axhline(0, color="k", ls="--", lw=0.4)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Force [N]", color=colors[wn])
        ax2.set_ylabel("Moment [mN·m]", color=C["grey"])
        ax.set_title(f"{wn}")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = od / "baseline_rocker.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_all_baseline(baseline_dir: Path = None, out_dir: Path = None):
    """生成全部 5 张基线图."""
    print("[baseline plots]")
    plot_baseline_overview(baseline_dir, out_dir)
    plot_baseline_phase(baseline_dir, out_dir)
    plot_baseline_wings(baseline_dir, out_dir)
    plot_baseline_aero(baseline_dir, out_dir)
    plot_baseline_rocker(baseline_dir, out_dir)


# ============================================================
# 单变量偏离图 (每个参数 6 张)
# ============================================================

def plot_sweep_LW_peak(param_name: str, sweep_dir: Path = None, out_dir: Path = None):
    """偏离图1: L/W + Peak θ vs 参数值."""
    sd = sweep_dir or (STABILITY_DIR / f"sweep_{param_name}")
    od = out_dir or (OUT_DIR / f"sweep_{param_name}")
    od.mkdir(parents=True, exist_ok=True)

    ss = _load_json(sd / "sweep_summary.json")
    x = ss["_value"]
    lw = ss["L/W"]
    peak = ss["peak_theta_deg"]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(x, lw, "o-", color=C["blue"], lw=2, markersize=8, label="L/W")
    ax1.axhline(1.0, color=C["green"], ls="--", lw=1, alpha=0.6, label="L/W=1 (hover)")
    ax1.axhline(0, color="k", ls="--", lw=0.5)
    ax2.plot(x, peak, "s--", color=C["red"], lw=1.5, markersize=6, label=r"Peak $\theta_p$")

    # 标注基线
    baseline_val = _get_baseline_val(param_name)
    if baseline_val is not None:
        ax1.axvline(baseline_val, color="grey", ls=":", lw=1, alpha=0.5, label=f"baseline={baseline_val}")

    ax1.set_xlabel(param_name)
    ax1.set_ylabel("L/W", color=C["blue"])
    ax2.set_ylabel(r"Peak $\theta_p$ [°]", color=C["red"])
    ax1.set_title(f"{param_name} → L/W & Pitch Stability")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = od / "L_W_peak.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_sweep_forces(param_name: str, sweep_dir: Path = None, out_dir: Path = None):
    """偏离图2: Fz_body, Fz_world, Fx vs 参数."""
    sd = sweep_dir or (STABILITY_DIR / f"sweep_{param_name}")
    od = out_dir or (OUT_DIR / f"sweep_{param_name}")
    od.mkdir(parents=True, exist_ok=True)

    ss = _load_json(sd / "sweep_summary.json")
    x = ss["_value"]
    weight = BASELINE_CONFIG.get("m_total", 0.02) * 9.81 * 1000

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, ss["mean_Fz_body_mN"], "o-", color=C["blue"], lw=2, markersize=7, label="Fz_body (mean)")
    ax.plot(x, ss["mean_Fz_world_mN"], "s--", color=C["teal"], lw=2, markersize=7, label="Fz_world (mean)")
    ax.plot(x, ss["mean_Fx_body_mN"], "d-.", color=C["purple"], lw=1.5, markersize=6, label="Fx_body (mean)")
    ax.axhline(weight, color="r", ls=":", lw=1, alpha=0.5, label=f"Weight={weight:.0f}mN")
    ax.axhline(0, color="k", ls="--", lw=0.5)

    baseline_val = _get_baseline_val(param_name)
    if baseline_val is not None:
        ax.axvline(baseline_val, color="grey", ls=":", lw=1, alpha=0.5)

    ax.set_xlabel(param_name)
    ax.set_ylabel("Force [mN]")
    ax.set_title(f"{param_name} → Forces")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = od / "forces.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_sweep_moments(param_name: str, sweep_dir: Path = None, out_dir: Path = None):
    """偏离图3: M_aero, M_grav, M_damp 均值+峰值."""
    sd = sweep_dir or (STABILITY_DIR / f"sweep_{param_name}")
    od = out_dir or (OUT_DIR / f"sweep_{param_name}")
    od.mkdir(parents=True, exist_ok=True)

    ss = _load_json(sd / "sweep_summary.json")
    x = ss["_value"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{param_name} → Moments", fontsize=13, fontweight="bold")

    # 均值
    ax1.plot(x, ss["mean_M_aero_uNm"], "o-", color=C["blue"], lw=2, markersize=7, label="M_aero")
    ax1.plot(x, ss["mean_M_grav_uNm"], "s-", color=C["red"], lw=1.5, markersize=6, label="M_grav")
    ax1.plot(x, ss["mean_M_damp_uNm"], "d-", color=C["grey"], lw=1.5, markersize=6, label="M_damp")
    ax1.axhline(0, color="k", ls="--", lw=0.5)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel("Moment [μN·m]")
    ax1.set_title("Mean (steady)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 峰值
    ax2.plot(x, ss["peak_M_aero_uNm"], "o-", color=C["blue"], lw=2, markersize=7, label="M_aero peak")
    ax2.plot(x, ss["peak_M_grav_uNm"], "s-", color=C["red"], lw=1.5, markersize=6, label="M_grav peak")
    ax2.plot(x, ss["peak_M_damp_uNm"], "d-", color=C["grey"], lw=1.5, markersize=6, label="M_damp peak")
    ax2.axhline(0, color="k", ls="--", lw=0.5)
    ax2.set_xlabel(param_name)
    ax2.set_ylabel("Moment [μN·m]")
    ax2.set_title("Peak")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = od / "moments.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_sweep_pitch_rates(param_name: str, sweep_dir: Path = None, out_dir: Path = None):
    """偏离图4: θ̇_p 统计量 vs 参数."""
    sd = sweep_dir or (STABILITY_DIR / f"sweep_{param_name}")
    od = out_dir or (OUT_DIR / f"sweep_{param_name}")
    od.mkdir(parents=True, exist_ok=True)

    ss = _load_json(sd / "sweep_summary.json")
    x = ss["_value"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, ss["mean_abs_thetadot_rads"], "o-", color=C["blue"], lw=2, markersize=7, label=r"mean $|\dot{\theta}_p|$")
    ax.plot(x, ss["peak_abs_thetadot_rads"], "s--", color=C["red"], lw=2, markersize=7, label=r"peak $|\dot{\theta}_p|$")
    ax.set_xlabel(param_name)
    ax.set_ylabel(r"$|\dot{\theta}_p|$ [rad/s]")
    ax.set_title(f"{param_name} → Pitch Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = od / "pitch_rates.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_sweep_accel(param_name: str, sweep_dir: Path = None, out_dir: Path = None):
    """偏离图5: θ̈_p 统计量 vs 参数."""
    sd = sweep_dir or (STABILITY_DIR / f"sweep_{param_name}")
    od = out_dir or (OUT_DIR / f"sweep_{param_name}")
    od.mkdir(parents=True, exist_ok=True)

    ss = _load_json(sd / "sweep_summary.json")
    x = ss["_value"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, ss["mean_abs_thetaddot_rads2"], "o-", color=C["blue"], lw=2, markersize=7, label=r"mean $|\ddot{\theta}_p|$")
    ax.plot(x, ss["peak_abs_thetaddot_rads2"], "s--", color=C["red"], lw=2, markersize=7, label=r"peak $|\ddot{\theta}_p|$")
    ax.set_xlabel(param_name)
    ax.set_ylabel(r"$|\ddot{\theta}_p|$ [rad/s²]")
    ax.set_title(f"{param_name} → Pitch Acceleration")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = od / "accel.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_sweep_phase_grid(param_name: str, sweep_dir: Path = None, out_dir: Path = None):
    """偏离图6: N 格相图, 每个参数值一格."""
    sd = sweep_dir or (STABILITY_DIR / f"sweep_{param_name}")
    od = out_dir or (OUT_DIR / f"sweep_{param_name}")
    od.mkdir(parents=True, exist_ok=True)

    # 读取所有已保存的 runs
    val_dirs = sorted(
        [d for d in sd.iterdir() if d.is_dir() and (d / "timeseries.npz").exists()],
        key=lambda d: float(d.name.replace('n', '-').replace('p', '.'))
    )

    n = len(val_dirs)
    if n == 0:
        print(f"  No data found in {sd}")
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f"{param_name} → Phase Portraits (steady state)", fontsize=14, fontweight="bold")

    for idx, vd in enumerate(val_dirs):
        ax = axes[idx // cols, idx % cols]
        ts = _load_npz(vd / "timeseries.npz")
        sm = _load_json(vd / "summary.json")
        half = len(ts["t"]) // 2

        ax.plot(np.rad2deg(ts["theta_p"][half:]), ts["theta_dot"][half:],
                ".", ms=0.5, alpha=0.4, color=C["purple"])
        ax.axhline(0, color="k", ls="--", lw=0.4)
        ax.axvline(0, color="k", ls="--", lw=0.4)
        ax.set_title(f"{param_name}={sm.get('_value','?')} | L/W={sm['L/W']:.3f}", fontsize=9)
        ax.set_xlabel(r"$\theta_p$ [°]", fontsize=7)
        ax.set_ylabel(r"$\dot{\theta}_p$ [rad/s]", fontsize=7)
        ax.tick_params(labelsize=7)

    # 隐藏多余的格子
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout()
    fp = od / "phase_grid.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp}")


def plot_all_sweep(param_name: str, sweep_dir: Path = None, out_dir: Path = None):
    """生成某个参数的全部 6 张偏离图."""
    sd = sweep_dir or (STABILITY_DIR / f"sweep_{param_name}")
    od = out_dir or (OUT_DIR / f"sweep_{param_name}")

    if not (sd / "sweep_summary.json").exists():
        print(f"[{param_name}] No sweep_summary.json — run analysis first")
        return

    print(f"\n[{param_name}] plotting...")
    plot_sweep_LW_peak(param_name, sd, od)
    plot_sweep_forces(param_name, sd, od)
    plot_sweep_moments(param_name, sd, od)
    plot_sweep_pitch_rates(param_name, sd, od)
    plot_sweep_accel(param_name, sd, od)
    plot_sweep_phase_grid(param_name, sd, od)


def _get_baseline_val(param_name: str) -> float:
    return BASELINE_CONFIG.get(param_name, None)


# ============================================================
# __main__
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="Plot all baseline figures")
    ap.add_argument("--sweep", type=str, default=None, help="Parameter name to plot sweep figures for")
    ap.add_argument("--all", action="store_true", help="Plot baseline + all available sweeps")
    args = ap.parse_args()

    if args.baseline or args.all:
        plot_all_baseline()

    if args.sweep:
        plot_all_sweep(args.sweep)

    if args.all:
        for dn in sorted(STABILITY_DIR.iterdir()):
            if dn.is_dir() and dn.name.startswith("sweep_"):
                pn = dn.name.replace("sweep_", "")
                if (dn / "sweep_summary.json").exists():
                    plot_all_sweep(pn)

    if not any([args.baseline, args.sweep, args.all]):
        print("Usage: python stability_plot.py --baseline [--sweep PARAM] [--all]")
        print(f"Available: baseline + {[d.name for d in STABILITY_DIR.iterdir() if d.is_dir() and d.name.startswith('sweep_')]}")
