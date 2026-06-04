#!/usr/bin/env python3
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mechanism import wing_kinematics

PHYS = {"rho": 1.225, "g": 9.81, "m_total": 0.020, "I_yy": 3e-5,
        "x_front": 0.025, "x_back": -0.025, "d_cg": 0.015, "c_damp": 5e-4}
WING = {"S": 16166.6e-6, "R": 154.3e-3, "r2_sq": 0.2382}
AERO = {"k_3d": 0.7}

def cl_cd(alpha_deg):
    C_L = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
    C_D = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))
    return C_L, C_D

t, phi, phi_dot, phi_ddot, info = wing_kinematics(f=15.0, a=7.92, phi_offset_deg=-50.84, n_points=2000)

print("phi range (deg):", np.degrees(info['phi_range_rad']))
print("phi_dot > 0 ratio:", np.mean(phi_dot > 0))
print("phi_dot max:", np.max(phi_dot), "min:", np.min(phi_dot))

# 固定 theta_p=0, theta_dot=0 计算单翅气动力
theta_p, theta_dot = 0.0, 0.0
alpha_deg = 45.0

psi = phi + theta_p
Omega = phi_dot + theta_dot
U = np.abs(Omega) * WING["R"]
const = 0.5 * PHYS["rho"] * U**2 * WING["S"] * WING["r2_sq"] * AERO["k_3d"]
sign_Omega = np.where(Omega <= 0, -1, 1)

C_L = np.zeros_like(Omega)
C_D = np.zeros_like(Omega)
mask_down = phi_dot <= 0
C_L[mask_down], C_D[mask_down] = cl_cd(alpha_deg)
C_L[~mask_down], C_D[~mask_down] = cl_cd(-alpha_deg)

Fx = const * np.sin(psi) * (sign_Omega * C_D - C_L)
Fz = const * np.cos(psi) * (C_L - sign_Omega * C_D)
Fx = np.where(np.abs(Omega) < 1e-6, 0, Fx)
Fz = np.where(np.abs(Omega) < 1e-6, 0, Fz)

print("\n--- 固定 theta=0, theta_dot=0 ---")
print(f"Avg Fx per wing: {np.mean(Fx)*1000:.3f} mN")
print(f"Avg Fz per wing: {np.mean(Fz)*1000:.3f} mN")
print(f"Max Fz: {np.max(Fz)*1000:.3f} mN, Min Fz: {np.min(Fz)*1000:.3f} mN")

# 打印几个关键点的值
print("\n--- 时序样本 (前20个点) ---")
print(f"{'t(ms)':>8} {'phi(deg)':>10} {'phi_dot':>10} {'Omega':>8} {'sign':>5} {'C_L':>6} {'C_D':>6} {'Fz(mN)':>10}")
for i in range(0, 20):
    print(f"{t[i]*1000:8.2f} {np.degrees(phi[i]):10.2f} {phi_dot[i]:10.2f} {Omega[i]:8.2f} {int(sign_Omega[i]):5d} {C_L[i]:6.2f} {C_D[i]:6.2f} {Fz[i]*1000:10.3f}")

# 检查下拍和上拍的平均贡献
down_mask = phi_dot <= 0
up_mask = phi_dot > 0
print(f"\n下拍平均 Fz: {np.mean(Fz[down_mask])*1000:.3f} mN (占比 {np.mean(down_mask):.2%})")
print(f"上拍平均 Fz: {np.mean(Fz[up_mask])*1000:.3f} mN (占比 {np.mean(up_mask):.2%})")

# 验证 cl_cd 在 +/-45° 的值
cl_p, cd_p = cl_cd(45)
cl_n, cd_n = cl_cd(-45)
print(f"\ncl_cd(+45): C_L={cl_p:.3f}, C_D={cd_p:.3f}")
print(f"cl_cd(-45): C_L={cl_n:.3f}, C_D={cd_n:.3f}")
