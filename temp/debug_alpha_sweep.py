#!/usr/bin/env python3
"""扫描不同攻角组合，找到能产生正净升力的参数空间"""
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

t, phi, phi_dot, _, info = wing_kinematics(f=15.0, a=7.92, phi_offset_deg=-50.84, n_points=2000)
theta_p, theta_dot = 0.0, 0.0
psi = phi + theta_p
Omega = phi_dot + theta_dot
U = np.abs(Omega) * WING["R"]
const = 0.5 * PHYS["rho"] * U**2 * WING["S"] * WING["r2_sq"] * AERO["k_3d"]
sign_Omega = np.where(Omega <= 0, -1, 1)
down_mask = phi_dot <= 0
up_mask = phi_dot > 0

def net_lift(alpha_down, alpha_up):
    C_L = np.zeros_like(Omega)
    C_D = np.zeros_like(Omega)
    C_L[down_mask], C_D[down_mask] = cl_cd(alpha_down)
    C_L[up_mask], C_D[up_mask] = cl_cd(alpha_up)
    Fz = const * np.cos(psi) * (C_L - sign_Omega * C_D)
    Fz = np.where(np.abs(Omega) < 1e-6, 0, Fz)
    return np.mean(Fz) * 1000  # mN per wing

print("=== 对称攻角扫描 ===")
print(f"{'alpha':>6} {'Fz(mN)':>10}")
for a in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 85]:
    fz = net_lift(a, -a)
    print(f"{a:6d} {fz:10.3f}")

print("\n=== 非对称攻角扫描（上拍攻角绝对值更小）===")
print(f"{'down':>6} {'up':>6} {'Fz(mN)':>10}")
for ad in [30, 45, 60]:
    for au in [-5, -10, -15, -20, -25, -30, -45]:
        fz = net_lift(ad, au)
        print(f"{ad:6d} {au:6d} {fz:10.3f}")

print("\n=== 非对称攻角扫描（上拍攻角为正 = 始终正升力）===")
print(f"{'down':>6} {'up':>6} {'Fz(mN)':>10}")
for ad in [30, 45, 60]:
    for au in [5, 10, 15, 20, 30, 45]:
        fz = net_lift(ad, au)
        print(f"{ad:6d} {au:6d} {fz:10.3f}")
