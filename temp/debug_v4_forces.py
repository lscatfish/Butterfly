#!/usr/bin/env python3
"""诊断 v4 各力分量的量级"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mechanism import wing_kinematics

def cl_cd(alpha_deg):
    C_L = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
    C_D = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))
    return C_L, C_D

t, phi, phi_dot, phi_ddot, info = wing_kinematics(f=15.0, a=7.92, phi_offset_deg=-50.84, n_points=2000)

rho = 1.225
S = 16166.6e-6
R = 154.3e-3
c_avg = 104.8e-3
r1 = 0.4227
r2_sq = 0.2382
k_3d = 0.7
alpha_down, alpha_up = 45.0, -10.0

psi = phi  # theta_p = 0, theta_dot = 0
Omega = phi_dot
U = np.abs(Omega) * R
const = 0.5 * rho * U**2 * S * r2_sq * k_3d
sign_Omega = np.where(Omega <= 0, -1, 1)

mask_down = phi_dot <= 0
C_L = np.zeros_like(phi)
C_D = np.zeros_like(phi)
alpha_eff = np.zeros_like(phi)
C_L[mask_down], C_D[mask_down] = cl_cd(alpha_down)
C_L[~mask_down], C_D[~mask_down] = cl_cd(alpha_up)
alpha_eff[mask_down] = alpha_down
alpha_eff[~mask_down] = alpha_up

L_trans = const * C_L
D_trans = const * C_D

alpha_rad = np.deg2rad(alpha_eff)
F_AM = -(rho * np.pi * c_avg**2 / 4.0) * phi_ddot * R * r1 * np.sin(alpha_rad)

phi_dot_peak = np.max(np.abs(phi_dot))
reversal_threshold = 0.1 * phi_dot_peak
in_reversal = np.abs(phi_dot) < reversal_threshold
k_clap = np.where(in_reversal, 1.3, 1.0)

L_eff = (L_trans + F_AM) * k_clap
D_eff = D_trans * k_clap

Fx = np.sin(psi) * (sign_Omega * D_eff - L_eff)
Fz = np.cos(psi) * (L_eff - sign_Omega * D_eff)

# 静止时归零
mask_still = np.abs(Omega) < 1e-6
Fx = np.where(mask_still, 0, Fx)
Fz = np.where(mask_still, 0, Fz)

print("=== 力分量诊断 ===")
print(f"L_trans  avg: {np.mean(L_trans)*1000:+.3f} mN | max: {np.max(np.abs(L_trans))*1000:.3f} mN")
print(f"D_trans  avg: {np.mean(D_trans)*1000:+.3f} mN | max: {np.max(np.abs(D_trans))*1000:.3f} mN")
print(f"F_AM     avg: {np.mean(F_AM)*1000:+.3f} mN | max: {np.max(np.abs(F_AM))*1000:.3f} mN")
print(f"k_clap   avg: {np.mean(k_clap):.3f}")
print(f"L_eff    avg: {np.mean(L_eff)*1000:+.3f} mN")
print(f"D_eff    avg: {np.mean(D_eff)*1000:+.3f} mN")
print(f"Fx       avg: {np.mean(Fx)*1000:+.3f} mN")
print(f"Fz       avg: {np.mean(Fz)*1000:+.3f} mN")

print("\n=== 附加质量力时序（前20点）===")
print(f"{'t(ms)':>8} {'phi_dot':>10} {'phi_ddot':>12} {'F_AM(mN)':>10} {'L_trans':>10} {'Fz(mN)':>10}")
for i in range(20):
    print(f"{t[i]*1000:8.2f} {phi_dot[i]:10.2f} {phi_ddot[i]:12.2f} {F_AM[i]*1000:10.3f} {L_trans[i]*1000:10.3f} {Fz[i]*1000:10.3f}")

print("\n=== 反转区域（|phi_dot| < threshold）===")
print(f"threshold = {reversal_threshold:.2f} rad/s")
print(f"反转点占比: {np.mean(in_reversal):.2%}")
print(f"反转区 Fz 均值: {np.mean(Fz[in_reversal])*1000:.3f} mN")
print(f"非反转区 Fz 均值: {np.mean(Fz[~in_reversal])*1000:.3f} mN")
