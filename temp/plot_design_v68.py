#!/usr/bin/env python3
"""DESIGN_v68 设计点曲线图"""
import numpy as np, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 字体
for _f in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    for _fm in font_manager.fontManager.ttflist:
        if _f.lower() in _fm.name.lower():
            plt.rcParams["font.family"] = _fm.name
            break
    else: continue
    break
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = Path("output/figures")
DATA_DIR = Path("temp/design_v68_detail")

# Load data
data = np.load(DATA_DIR / "timeseries_2cycles.npz")
t = data["t"]
with open(DATA_DIR / "summary.json") as f:
    summary = json.load(f)

fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle(f"DESIGN_v68 Performance Curves — α_f=45°, α_b=8°, L/W={summary['L/W']:.3f}, peak_θ={summary['peak_theta_deg']:.1f}°",
             fontsize=15, fontweight="bold", y=0.98)

# 1: Pitch angle
ax = axes[0, 0]
ax.plot(t*1000, np.rad2deg(data["theta_p"]), "#2196F3", linewidth=1.2)
ax.set_ylabel("θ_p (°)")
ax.set_xlabel("t (ms)")
ax.set_title(f"Pitch Angle (mean={summary['steady_theta_mean_deg']:.1f}°, amp=±{summary['steady_theta_amplitude_deg']:.1f}°)")
ax.grid(alpha=0.3)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

# 2: θ_dot
ax = axes[0, 1]
ax.plot(t*1000, data["theta_dot"], "#FF9800", linewidth=1.2)
ax.set_ylabel("θ̇_p (rad/s)")
ax.set_xlabel("t (ms)")
ax.set_title("Pitch Rate")
ax.grid(alpha=0.3)

# 3: Body forces (world)
ax = axes[0, 2]
ax.plot(t*1000, data["Fz_world_total"]*1000, "#4CAF50", linewidth=1.2, label="Fz_world")
ax.axhline(y=196.2, color="gray", linestyle="--", alpha=0.5, label="Weight (196 mN)")
ax.set_ylabel("Force (mN)")
ax.set_xlabel("t (ms)")
ax.set_title(f"Total Lift (World) — mean={summary['mean_Fz_world_mN']:.0f} mN")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 4: Per-wing Fz_body
ax = axes[1, 0]
ax.plot(t*1000, data["Fz_body_FL"]*1000, "#E91E63", linewidth=0.8, label="FL")
ax.plot(t*1000, data["Fz_body_BL"]*1000, "#2196F3", linewidth=0.8, label="BL")
ax.set_ylabel("Fz_body (mN)")
ax.set_xlabel("t (ms)")
ax.set_title("Body-frame Vertical Force per Wing")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 5: Per-wing Fx_body
ax = axes[1, 1]
ax.plot(t*1000, data["Fx_body_FL"]*1000, "#E91E63", linewidth=0.8, label="FL")
ax.plot(t*1000, data["Fx_body_BL"]*1000, "#2196F3", linewidth=0.8, label="BL")
ax.set_ylabel("Fx_body (mN)")
ax.set_xlabel("t (ms)")
ax.set_title("Body-frame Horizontal Force per Wing")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 6: Effective AoA
ax = axes[1, 2]
ax.plot(t*1000, data["alpha_eff_FL"], "#E91E63", linewidth=0.8, label=f"FL (mean|α|={summary['mean_alpha_eff_FL_deg']:.1f}°)")
ax.plot(t*1000, data["alpha_eff_BL"], "#2196F3", linewidth=0.8, label="BL")
ax.axhline(y=70, color="red", linestyle="--", alpha=0.3, linewidth=0.8, label="α=70° (stall)")
ax.axhline(y=-70, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
ax.axhline(y=40, color="orange", linestyle="--", alpha=0.3, linewidth=0.8, label="α=40° (Dickinson)")
ax.axhline(y=-40, color="orange", linestyle="--", alpha=0.3, linewidth=0.8)
ax.set_ylabel("α_eff (°)")
ax.set_xlabel("t (ms)")
ax.set_title(f"Effective AoA — |α|>70°: {summary['pct_alpha_above_70']:.1f}% time")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# 7: C_L, C_D (FL)
ax = axes[2, 0]
ax.plot(t*1000, data["CL_FL"], "#4CAF50", linewidth=0.8, label=f"C_L (mean={summary['mean_CL_FL']:.3f})")
ax.plot(t*1000, data["CD_FL"], "#F44336", linewidth=0.8, label=f"C_D (mean={summary['mean_CD_FL']:.3f})")
ax.set_ylabel("Coefficient")
ax.set_xlabel("t (ms)")
ax.set_title("Front Wing C_L / C_D")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 8: Flapping kinematics (FL)
ax = axes[2, 1]
ax.plot(t*1000, np.rad2deg(data["phi_FL"]), "#2196F3", linewidth=1.2, label="φ (°)")
ax2 = ax.twinx()
ax2.plot(t*1000, data["phi_dot_FL"], "#FF9800", linewidth=0.8, alpha=0.7, label="φ̇ (rad/s)")
ax.set_ylabel("φ (°)")
ax2.set_ylabel("φ̇ (rad/s)", color="#FF9800")
ax.set_xlabel("t (ms)")
ax.set_title("Front Wing Kinematics (2 cycles)")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax.grid(alpha=0.3)

# 9: Rocker principal vector & moment (FL)
ax = axes[2, 2]
ax.plot(t*1000, data["rocker_pv_x_FL"], "#E91E63", linewidth=0.8, label="PV_x")
ax.plot(t*1000, data["rocker_pv_z_FL"], "#2196F3", linewidth=0.8, label="PV_z")
ax2 = ax.twinx()
ax2.plot(t*1000, data["rocker_pm_y_FL"]*1e3, "#4CAF50", linewidth=0.8, alpha=0.6, label="PM_y (mN·m)")
ax.set_ylabel("Principal Vector (N)")
ax2.set_ylabel("Principal Moment (mN·m)", color="#4CAF50")
ax.set_xlabel("t (ms)")
ax.set_title("FL Rocker Principal Vector & Moment")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(FIG_DIR / "fig11_design_v68_curves.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {FIG_DIR / 'fig11_design_v68_curves.png'}")
