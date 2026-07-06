#!/usr/bin/env python3
"""
蝴蝶力输出模块 — 对外调用接口。

提供:
  - ButterflyForceModel: 完整仿真（运动学→俯仰ODE→力输出）
  - scan_parameters(): 参数扫描
  - SimulationConfig: 所有可配置参数
  - SimulationOutput / WingOutput: 结构化输出

坐标系 (v6.9, 与 SolidWorks 对齐):
  体轴 (Body): 原点=CG, X=右, Y=上, Z=前
  世界 (World): θ_p=0时与体轴重合
  翅膀拍动轴: 平行于 Z (左右翅膀共轴)
  机身俯仰轴: 平行于 X (过总质心 CG)
  四连杆机构平面: XY 平面 (垂直于拍动轴 Z)

摇杆分解:
  主矢 = 合力沿摇杆方向(A→P2)的分量, 通过连杆传至曲柄
  主矩 = 对摇杆枢轴A的力矩在Z轴上的分量, 驱动/制动摇杆的有效扭矩

v6.9 坐标系重构: 翅膀拍动绕 Z, 机身俯仰绕 X, 两者垂直, 不再用 psi=phi+theta_p.
"""
import numpy as np
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        """No-op decorator when numba unavailable."""
        def _decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _decorator

# ---- project root for imports ----
_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ))
from src.struct.mechanism import wing_kinematics, DEFAULT_PARAMS, solve_phi
from src.config import get_design, get_version

# ============================================================
# Dataclasses
# ============================================================

@dataclass
class WingGeometry:
    """单翅几何参数 (SI单位)."""
    name: str
    S: float          # 面积 [m²]
    R: float          # 展长 [m]
    c_avg: float      # 平均弦长 [m]
    r1: float         # 一阶矩系数 [无量纲]
    r2_sq: float      # 二阶矩系数 [无量纲]
    AR: float         # 展弦比


# ---- 共享设计参数: 从 config/design_v69.yaml 加载（单一来源） ----
DESIGN_v69 = get_design()

# 向后兼容
DESIGN_v68 = DESIGN_v69


@dataclass
class SimulationConfig:
    """仿真全参数配置 — 所有参数可被scan覆盖."""

    # ---- 翅膀安装 ----
    alpha_front_deg: float = DESIGN_v69["alpha_front_deg"]
    alpha_back_deg: float = DESIGN_v69["alpha_back_deg"]
    phase_diff_deg: float = DESIGN_v69["phase_diff_deg"]

    # ---- 四连杆机构 ----
    mech_a: float = DESIGN_v69["mech_a"]
    mech_b: float = DESIGN_v69["mech_b"]
    mech_R: float = DESIGN_v69["mech_R"]
    mech_c: float = DESIGN_v69["mech_c"]
    mech_l: float = DESIGN_v69["mech_l"]
    phi_offset_deg: float = DESIGN_v69["phi_offset_deg"]
    rotation: str = DESIGN_v69["rotation"]

    # ---- 物理 (SW 对齐: X=右, Y=上, Z=前; 拍动轴=Z, 俯仰轴=X) ----
    f: float = DESIGN_v69["f"]
    rho: float = DESIGN_v69["rho"]
    m_total: float = DESIGN_v69["m_total"]
    I_xx: float = DESIGN_v69["I_xx"]
    x_hinge_right: float = DESIGN_v69["x_hinge_right"]
    x_hinge_left: float = DESIGN_v69["x_hinge_left"]
    y_hinge_rel: float = DESIGN_v69["y_hinge_rel"]
    z_front: float = DESIGN_v69["z_front"]
    z_back: float = DESIGN_v69["z_back"]
    g: float = DESIGN_v69["g"]

    # ---- 数值 ----
    dt: float = DESIGN_v69["dt"]
    t_end: float = DESIGN_v69["t_end"]
    theta0_deg: float = DESIGN_v69["theta0_deg"]
    steady_start: float = DESIGN_v69["steady_start"]

    # ---- 气动系数 ----
    k_3d: float = DESIGN_v69["k_3d"]
    C_rot: float = DESIGN_v69["C_rot"]
    r_rot: float = DESIGN_v69["r_rot"]
    k_clap: float = DESIGN_v69["k_clap"]
    c_damp: float = DESIGN_v69["c_damp"]

    def to_mech_params(self) -> dict:
        return {
            "a": self.mech_a, "b": self.mech_b, "R": self.mech_R,
            "c": self.mech_c, "l": self.mech_l,
            "phi_offset_deg": self.phi_offset_deg, "rotation": self.rotation,
            "f": self.f,
        }


@dataclass
class WingOutput:
    """单翅完整输出 (N_timesteps)."""
    name: str
    force_body: np.ndarray            # (N,3) 体轴力 [N]
    force_world: np.ndarray           # (N,3) 世界力 [N]
    cop_body: np.ndarray              # (N,3) 气动中心 (体轴) [m]
    cop_world: np.ndarray             # (N,3) 气动中心 (世界) [m]
    moment_body: np.ndarray           # (N,3) 对CG力矩 (体轴) [N·m]
    moment_world: np.ndarray          # (N,3) 对CG力矩 (世界) [N·m]
    rocker_principal_vec: np.ndarray  # (N,3) 摇杆主矢 [N]
    rocker_principal_moment: np.ndarray  # (N,3) 摇杆主矩 [N·m]
    rocker_angle_rad: np.ndarray      # (N,) 摇杆方向角 [rad]
    alpha_eff_deg: np.ndarray         # (N,) 有效攻角 [°]
    C_L: np.ndarray                   # (N,)
    C_D: np.ndarray                   # (N,)
    phi: np.ndarray                   # (N,) 拍动角 [rad]
    phi_dot: np.ndarray               # (N,) 拍动角速度 [rad/s]
    phi_ddot: np.ndarray              # (N,) 拍动角加速度 [rad/s²]


@dataclass
class SimulationOutput:
    """顶层仿真输出."""
    t: np.ndarray                     # (N,) 时间 [s]
    theta_p: np.ndarray               # (N,) 俯仰角 [rad]
    theta_dot: np.ndarray            # (N,) 俯仰角速度 [rad/s]
    theta_ddot: np.ndarray           # (N,) 俯仰角加速度 [rad/s²]
    wings: Dict[str, WingOutput]      # "FL","FR","BL","BR"
    summary: dict                     # L/W, avg_Fz, n90, peak_deg, ...
    config: SimulationConfig


# ============================================================
# Clap-and-Fling — 速度-位置耦合增强窗口 (文献[36-39])
# ============================================================

def compute_clap_fling_window(phi, phi_dot, edge_width=0.10):
    """Lighthill环量公式启发的clap-and-fling增强窗口.

    文献[36-39]: clap-and-fling效应来自两翅分离时间隙射流产生
    的额外环量. 增强正比于:
      - 张开速度 |φ̇| (Lighthill Γ=g(λ)φ̇c²)
      - 翅膀靠近程度 (位置余弦平方窗, 端点处最强)

    Args:
        phi: 翅膀拍动角数组 (rad)
        phi_dot: 拍动角速度数组 (rad/s)
        edge_width: 端点区域宽度 (占拍动幅度的比例, 默认10%)

    Returns:
        k_extra: (N,) 增强系数数组, 范围 [0, (|φ̇|/φ̇_peak)]
    """
    phi_max = np.max(phi); phi_min = np.min(phi)
    phi_range = phi_max - phi_min
    if phi_range < 1e-10:
        return np.zeros_like(phi)
    # 位置窗: 余弦平方, 端点=1, 远离=0
    dist_to_max = np.abs(phi - phi_max)
    dist_to_min = np.abs(phi - phi_min)
    dist_norm = np.minimum(dist_to_max, dist_to_min) / phi_range
    in_edge = dist_norm < edge_width
    pos_window = np.zeros_like(phi)
    pos_window[in_edge] = np.cos(dist_norm[in_edge] / edge_width * np.pi / 2.0)**2
    # 速度耦合: 增强正比于 |φ̇|, 端点处速度→0 增强自动归零
    phi_dot_peak = np.max(np.abs(phi_dot))
    if phi_dot_peak < 1e-6:
        return np.zeros_like(phi)
    vel_factor = np.abs(phi_dot) / phi_dot_peak
    return pos_window * vel_factor


# C_L / C_D — v6.3 LEV/Lee 混合模型
# ============================================================

def cl_cd_blended(alpha_deg):
    """v6.3 LEV理论 + Lee公式 + Dickinson低攻角匹配.

    |alpha| <= 55deg: Dickinson 经验 (LEV增强)
    55-65deg: smoothstep 过渡
    |alpha| >= 65deg: C_L = A*sin(2a), C_D = C_D0 + A_D*(1-cos(2a))

    参考: [32] JRSI 2017 (LEV), [24] 机器人 2025 (Lee)
    """
    abs_a = np.abs(alpha_deg)
    alpha_rad = np.deg2rad(alpha_deg)

    # Dickinson 经验
    cl_d = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
    cd_d = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))

    # LEV/Lee 理论
    A_adj = 1.866
    C_D0 = 0.393
    A_D = 1.414
    cl_lev = A_adj * np.sin(2.0 * alpha_rad)
    cd_lee = C_D0 + A_D * (1.0 - np.cos(2.0 * alpha_rad))

    # smoothstep 55-65 deg
    lo, hi = 55.0, 65.0
    t = np.clip((abs_a - lo) / (hi - lo), 0.0, 1.0)
    w = 3.0 * t**2 - 2.0 * t**3

    C_L = (1.0 - w) * cl_d + w * cl_lev
    C_D = (1.0 - w) * cd_d + w * cd_lee
    return C_L, C_D


# ============================================================
# Numba JIT 加速 — 标量热路径
# ============================================================

@njit(cache=True, fastmath=True)
def _cl_cd_blended_scalar(alpha_deg):
    """Numba标量版 cl_cd_blended — 与向量版数学一致."""
    abs_a = abs(alpha_deg)
    alpha_rad = alpha_deg * np.pi / 180.0

    # Dickinson 经验
    cl_d = 0.255 + 1.58 * np.sin((2.13 * alpha_deg - 7.2) * np.pi / 180.0)
    cd_d = 1.92 - 1.55 * np.cos((2.04 * alpha_deg - 9.82) * np.pi / 180.0)

    # LEV/Lee 理论
    cl_lev = 1.866 * np.sin(2.0 * alpha_rad)
    cd_lee = 0.393 + 1.414 * (1.0 - np.cos(2.0 * alpha_rad))

    # smoothstep 55-65 deg
    if abs_a <= 55.0:
        w = 0.0
    elif abs_a >= 65.0:
        w = 1.0
    else:
        t = (abs_a - 55.0) / 10.0
        w = 3.0 * t * t - 2.0 * t * t * t

    C_L = (1.0 - w) * cl_d + w * cl_lev
    C_D = (1.0 - w) * cd_d + w * cd_lee
    return C_L, C_D


@njit(cache=True, fastmath=True)
def _wing_forces_scalar(phi, phi_dot, phi_ddot, theta_p, theta_dot,
                         alpha_install_deg, S, R, c_avg, r1, r2_sq,
                         y_hinge_rel, rho, k_3d, k_clap):
    """Numba标量版单翅气动力 — 返回 (Fy_body, Fx_body).

    坐标系: X=右, Y=上, Z=前; 翅膀绕 Z 拍动, phi 在 XY 平面内。
    俯仰绕 X, 铰链有 Y 向偏移 y_hinge_rel, 产生 Δα 修正。
    """
    Omega = phi_dot
    abs_Omega = abs(Omega)
    U = abs_Omega * R

    # 俯仰气动阻尼: Δα = atan(θ̇_p·y_hinge_rel / U)
    v_pitch = theta_dot * y_hinge_rel
    if abs_Omega < 1e-6:
        delta_alpha_rad = 0.0
    else:
        delta_alpha_rad = np.arctan2(v_pitch, U)

    # 有效攻角：文献[32]约定，η = α_install，上下拍符号翻转
    eta_rad = alpha_install_deg * np.pi / 180.0
    sign_smooth = np.tanh(phi_dot / 2.0)
    alpha_eff_rad = -sign_smooth * (eta_rad + delta_alpha_rad)
    alpha_eff_deg = alpha_eff_rad * 180.0 / np.pi

    C_L, C_D = _cl_cd_blended_scalar(alpha_eff_deg)

    # 平动力
    const = 0.5 * rho * U * U * S * r2_sq * k_3d
    L_trans = const * C_L
    D_trans = const * C_D

    # 附加质量
    if alpha_eff_rad > np.pi / 2.0:
        alpha_eff_clamped = np.pi / 2.0
    elif alpha_eff_rad < -np.pi / 2.0:
        alpha_eff_clamped = -np.pi / 2.0
    else:
        alpha_eff_clamped = alpha_eff_rad
    F_AM = -(rho * np.pi * c_avg * c_avg / 4.0) * phi_ddot * R * r1 * np.sin(alpha_eff_clamped)

    # Clap-and-Fling
    L_eff = (L_trans + F_AM) * k_clap
    D_eff = D_trans * k_clap

    # 体轴系力: X=右, Y=上; phi=0 时翅膀弦线沿 +X, 升力沿 +Y
    sign_Omega = np.tanh(Omega / 2.0)
    Fx = np.sin(phi) * (sign_Omega * D_eff - L_eff)
    Fy = np.cos(phi) * (L_eff - sign_Omega * D_eff)
    Fz = 0.0

    if abs_Omega < 1e-6:
        Fx = 0.0
        Fy = 0.0
        Fz = 0.0

    return Fy, Fx, Fz


@njit(cache=True, fastmath=True)
def _pitch_rhs_numba(theta_p, theta_dot,
                      pf, pdf, pddf, pb, pdb, pddb,
                      k_clap_f, k_clap_b,
                      alpha_f_deg, alpha_b_deg,
                      S_f, R_f, c_avg_f, r1_f, r2_sq_f,
                      S_b, R_b, c_avg_b, r1_b, r2_sq_b,
                      y_hinge_rel,
                      z_front, z_back,
                      rho, k_3d,
                      m_total, g, I_xx, c_damp):
    """Numba标量版俯仰ODE右端项 — 返回 (theta_ddot, Fx_total, Fy_total, M_aero).

    俯仰轴 = X (过 CG), 气动力在 Y 方向, 力臂 = z_front/z_back。
    """
    # 前翅
    Fy_f, Fx_f, _ = _wing_forces_scalar(
        pf, pdf, pddf, theta_p, theta_dot,
        alpha_f_deg, S_f, R_f, c_avg_f, r1_f, r2_sq_f,
        y_hinge_rel, rho, k_3d, k_clap_f)

    # 后翅
    Fy_b, Fx_b, _ = _wing_forces_scalar(
        pb, pdb, pddb, theta_p, theta_dot,
        alpha_b_deg, S_b, R_b, c_avg_b, r1_b, r2_sq_b,
        y_hinge_rel, rho, k_3d, k_clap_b)

    # 左右对称 ×2
    Fx_total = 2.0 * (Fx_f + Fx_b)
    Fy_total = 2.0 * (Fy_f + Fy_b)
    # 前翅向上力 (Fy>0) 作用在 z_front>0, 产生低头力矩 (负); 后翅在 z_back<0, 产生抬头力矩 (正)
    M_aero = 2.0 * (-z_front * Fy_f - z_back * Fy_b)

    # 俯仰轴过 CG, 重力矩为 0
    M_damp = -c_damp * theta_dot
    theta_ddot = (M_aero + M_damp) / I_xx

    return theta_ddot, Fx_total, Fy_total, M_aero


@njit(cache=True, fastmath=True)
def _rk4_step_numba(tp, td, dt, pf, pdf, pddf, pb, pdb, pddb, kcl_f, kcl_b, *params):
    """Numba标量版单步RK4 — 返回 (tp_new, td_new)."""
    # k1
    k1_tdd, _, _, _ = _pitch_rhs_numba(
        tp, td, pf, pdf, pddf, pb, pdb, pddb, kcl_f, kcl_b, *params)

    # k2
    tp_k2 = tp + 0.5 * dt * td
    td_k2 = td + 0.5 * dt * k1_tdd
    k2_tdd, _, _, _ = _pitch_rhs_numba(
        tp_k2, td_k2, pf, pdf, pddf, pb, pdb, pddb, kcl_f, kcl_b, *params)

    # k3
    tp_k3 = tp + 0.5 * dt * td_k2
    td_k3 = td + 0.5 * dt * k2_tdd
    k3_tdd, _, _, _ = _pitch_rhs_numba(
        tp_k3, td_k3, pf, pdf, pddf, pb, pdb, pddb, kcl_f, kcl_b, *params)

    # k4
    tp_k4 = tp + dt * td_k3
    td_k4 = td + dt * k3_tdd
    k4_tdd, _, _, _ = _pitch_rhs_numba(
        tp_k4, td_k4, pf, pdf, pddf, pb, pdb, pddb, kcl_f, kcl_b, *params)

    tp_new = tp + dt * (td + 2.0 * td_k2 + 2.0 * td_k3 + td_k4) / 6.0
    td_new = td + dt * (k1_tdd + 2.0 * k2_tdd + 2.0 * k3_tdd + k4_tdd) / 6.0

    return tp_new, td_new


# ============================================================
# 翅膀几何加载
# ============================================================

def load_wing_geometry() -> Dict[str, WingGeometry]:
    """从 data/wing_analysis_results.json 加载翅膀几何."""
    geo_path = _PROJ / "data" / "wing_analysis_results.json"
    with open(geo_path, encoding="utf-8") as f:
        data = json.load(f)
    geo = {}
    for g in data["geometry"]:
        geo[g["name"]] = WingGeometry(
            name=g["name"],
            S=g["S"], R=g["R"], c_avg=g["c_avg"],
            r1=g["r1"], r2_sq=g["r2_sq"], AR=g["AR"],
        )
    return geo


_DEFAULT_GEO = None

def default_geometry() -> Dict[str, WingGeometry]:
    global _DEFAULT_GEO
    if _DEFAULT_GEO is None:
        _DEFAULT_GEO = load_wing_geometry()
    return _DEFAULT_GEO


# ============================================================
# 运动学
# ============================================================

def _precompute_one_period(config: SimulationConfig, n_period: int = 2000):
    """预计算一个周期的运动学."""
    mp = config.to_mech_params()
    t, phi, phi_dot, phi_ddot, info = wing_kinematics(
        f=mp["f"], a=mp["a"],
        phi_offset_deg=mp["phi_offset_deg"],
        rotation=mp["rotation"],
        params={"a": mp["a"], "b": mp["b"], "R": mp["R"], "c": mp["c"], "l": mp["l"]},
        n_points=n_period,
    )
    T = info["T_s"]
    t_ext = np.concatenate([[t[-1] - T], t, [t[0] + T]])
    phi_ext = np.concatenate([[phi[-1]], phi, [phi[0]]])
    phi_dot_ext = np.concatenate([[phi_dot[-1]], phi_dot, [phi_dot[0]]])
    phi_ddot_ext = np.concatenate([[phi_ddot[-1]], phi_ddot, [phi_ddot[0]]])
    return t_ext, phi_ext, phi_dot_ext, phi_ddot_ext, T


def _interpolate_kinematics(t_arr, t_ext, phi_ext, phi_dot_ext, phi_ddot_ext,
                             phase_sec, T):
    """从周期模板插值到任意时间网格, 支持相位偏移."""
    t_eff = np.mod(t_arr + phase_sec, T)
    phi = np.interp(t_eff, t_ext, phi_ext)
    phi_dot = np.interp(t_eff, t_ext, phi_dot_ext)
    phi_ddot = np.interp(t_eff, t_ext, phi_ddot_ext)
    return phi, phi_dot, phi_ddot


# ============================================================
# 力计算 — 向量化版本
# ============================================================

def compute_wing_forces_vec(phi, phi_dot, phi_ddot, theta_p, theta_dot,
                             alpha_install_deg, geo: WingGeometry, y_hinge_rel, config: SimulationConfig):
    """向量化单翅气动力计算 (v6.9 坐标系).

    坐标系: X=右, Y=上, Z=前; 翅膀绕 Z 拍动, phi 在 XY 平面内。
    俯仰绕 X, 铰链有 Y 向偏移 y_hinge_rel, 产生 Δα 修正。

    返回 dict:
      F_body: (N,3) 体轴力 (Fx, Fy, Fz)
      alpha_eff_deg: (N,)
      C_L, C_D: (N,)
      Omega: (N,) 拍动角速度
    """
    N = len(phi)
    Omega = phi_dot
    U = np.abs(Omega) * geo.R

    # 俯仰气动阻尼: Δα = atan(θ̇_p·y_hinge_rel / U)
    v_pitch = theta_dot * y_hinge_rel
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_alpha_rad = np.arctan2(v_pitch, U + 1e-6)

    # 有效攻角：文献[32]约定
    eta_rad = np.deg2rad(alpha_install_deg)
    sign_smooth = np.tanh(phi_dot / 2.0)
    alpha_eff_rad = -sign_smooth * (eta_rad + delta_alpha_rad)
    alpha_eff_deg = np.rad2deg(alpha_eff_rad)

    C_L, C_D = cl_cd_blended(alpha_eff_deg)

    # 平动力
    const = 0.5 * config.rho * U**2 * geo.S * geo.r2_sq * config.k_3d
    sign_Omega = np.tanh(Omega / 2.0)
    L_trans = const * C_L
    D_trans = const * C_D

    # 附加质量
    alpha_eff_clamped = np.deg2rad(np.clip(alpha_eff_deg, -90, 90))
    F_AM = -(config.rho * np.pi * geo.c_avg**2 / 4.0) * phi_ddot * geo.R * geo.r1 * np.sin(alpha_eff_clamped)

    # 旋转力 (当前 alpha_dot=0)
    alpha_dot = np.zeros(N)
    F_rot = config.rho * config.C_rot * alpha_dot * phi_dot * geo.c_avg**2 * geo.R * config.r_rot

    # Clap-and-Fling: 速度-位置耦合增强
    k_clap_extra = compute_clap_fling_window(phi, phi_dot, edge_width=0.10)
    k_clap = 1.0 + config.k_clap * k_clap_extra

    L_eff = (L_trans + F_AM + F_rot) * k_clap
    D_eff = D_trans * k_clap

    # 体轴系力: X=右, Y=上; phi=0 时翅膀弦线沿 +X, 升力沿 +Y
    Fx = np.sin(phi) * (sign_Omega * D_eff - L_eff)
    Fy = np.cos(phi) * (L_eff - sign_Omega * D_eff)
    Fz = np.zeros(N)

    mask_still = np.abs(Omega) < 1e-6
    Fx = np.where(mask_still, 0.0, Fx)
    Fy = np.where(mask_still, 0.0, Fy)
    Fz = np.where(mask_still, 0.0, Fz)

    F_body = np.zeros((N, 3))
    F_body[:, 0] = Fx
    F_body[:, 1] = Fy
    F_body[:, 2] = Fz

    return {
        "F_body": F_body,
        "alpha_eff_deg": alpha_eff_deg,
        "C_L": C_L, "C_D": C_D,
        "Omega": Omega, "k_clap": k_clap,
    }


def compute_cop_vec(phi, geo: WingGeometry, x_hinge, y_hinge_rel, z_hinge, side_sign):
    """向量化气动中心位置 (v6.9 坐标系).

    side_sign: +1 (右翅, +Z 展向) / -1 (左翅, -Z 展向)

    CoP = hinge_pos + chordwise_offset + spanwise_offset.
    - 弦向: c_avg/4 在 XY 平面内, 方向随 phi 变化
    - 展向: r1 * R 沿 Z
    """
    N = len(phi)
    cop = np.zeros((N, 3))
    cop[:, 0] = x_hinge + (geo.c_avg / 4.0) * np.cos(phi)   # 弦向 X
    cop[:, 1] = y_hinge_rel + (geo.c_avg / 4.0) * np.sin(phi)  # 弦向 Y
    cop[:, 2] = z_hinge + side_sign * geo.r1 * geo.R         # 展向 Z
    return cop


def rocker_decompose(F_body, cop_body, phi, config: SimulationConfig, x_hinge, y_hinge_rel):
    """摇杆力分解: 主矢 + 主矩 (v6.9 坐标系).

    摇杆枢轴 A 在体轴系: A_body = (x_hinge, y_hinge_rel, mech_a/1000)
    摇杆机构平面 = XY 平面 (垂直于拍动轴 Z).
    摇杆方向: 从 A 指向 P2, 在 XY 平面内变化, 方向角 = phi - phi_offset.

    主矢: F_body 沿摇杆方向的分量
    主矩: 对 A 的力矩在 Z 轴上的分量
    """
    N = len(phi)
    a_m = config.mech_a / 1000.0

    # 枢轴 A 在体轴系
    A_body = np.array([x_hinge, y_hinge_rel, a_m])

    # 摇杆机构角 (去除安装偏角后的原始机构角度)
    phi_mech = phi - np.deg2rad(config.phi_offset_deg)

    # 摇杆方向单位矢量 (在 XY 平面内)
    d_rocker = np.zeros((N, 3))
    d_rocker[:, 0] = np.cos(phi_mech)  # 体轴 X 分量
    d_rocker[:, 1] = np.sin(phi_mech)  # 体轴 Y 分量
    # 归一化
    norm = np.sqrt(d_rocker[:, 0]**2 + d_rocker[:, 1]**2)
    d_rocker[:, 0] /= norm
    d_rocker[:, 1] /= norm

    # ---- 主矢: 力沿摇杆方向的分量 ----
    F_dot_rocker = np.sum(F_body * d_rocker, axis=1)  # (N,)
    principal_vec = d_rocker * F_dot_rocker[:, np.newaxis]  # (N,3)

    # ---- 主矩: 对枢轴A的力矩在Z轴上的分量 ----
    r_from_A = cop_body - A_body[np.newaxis, :]  # (N,3) 从A到CoP的矢量
    M_about_A = np.cross(r_from_A, F_body)        # (N,3) 对A的力矩
    principal_moment = np.zeros((N, 3))
    principal_moment[:, 2] = M_about_A[:, 2]      # 仅保留Z分量

    # 摇杆方向角 (用于调用方参考)
    rocker_angle = phi_mech

    return principal_vec, principal_moment, rocker_angle


# ============================================================
# 坐标变换
# ============================================================

def body_to_world(vec_body, theta_p):
    """体轴系→世界系: R_x(θ_p) 旋转 (v6.9 俯仰绕 X).

    vec_body: (N,3) 或 (3,)
    theta_p: (N,) 或 scalar
    """
    vec = np.asarray(vec_body)
    tp = np.asarray(theta_p)
    scalar_tp = tp.ndim == 0
    if scalar_tp:
        tp = np.array([tp])
    if vec.ndim == 1:
        vec = vec[np.newaxis, :]
        was_1d = True
    else:
        was_1d = False

    cos_t = np.cos(tp)
    sin_t = np.sin(tp)
    N = vec.shape[0]

    vec_w = np.zeros_like(vec)
    vec_w[:, 0] = vec[:, 0]
    vec_w[:, 1] = cos_t * vec[:, 1] - sin_t * vec[:, 2]
    vec_w[:, 2] = sin_t * vec[:, 1] + cos_t * vec[:, 2]

    if was_1d:
        vec_w = vec_w[0]
    return vec_w


# ============================================================
# 俯仰 ODE (用于 RK4)
# ============================================================

def _compute_pitch_rhs_scalar(theta_p, theta_dot,
                                phi_f, phi_dot_f, phi_ddot_f,
                                phi_b, phi_dot_b, phi_ddot_b,
                                geo_f, geo_b, config):
    """单步俯仰 ODE 右端项 (标量)."""
    r_f = compute_wing_forces_vec(
        np.array([phi_f]), np.array([phi_dot_f]), np.array([phi_ddot_f]),
        np.array([theta_p]), np.array([theta_dot]),
        config.alpha_front_deg, geo_f, config.y_hinge_rel, config)
    r_b = compute_wing_forces_vec(
        np.array([phi_b]), np.array([phi_dot_b]), np.array([phi_ddot_b]),
        np.array([theta_p]), np.array([theta_dot]),
        config.alpha_back_deg, geo_b, config.y_hinge_rel, config)

    Fy_f = r_f["F_body"][0, 1]
    Fy_b = r_b["F_body"][0, 1]
    Fx_f = r_f["F_body"][0, 0]
    Fx_b = r_b["F_body"][0, 0]

    # 左右对称 x2
    Fx_total = 2.0 * (Fx_f + Fx_b)
    Fy_total = 2.0 * (Fy_f + Fy_b)
    # 前翅向上力 (Fy>0) 在 z_front>0 处产生低头力矩 (负)
    M_aero = 2.0 * (-config.z_front * Fy_f - config.z_back * Fy_b)

    # 俯仰轴过 CG, 重力矩为 0
    M_damp = -config.c_damp * theta_dot
    theta_ddot = (M_aero + M_damp) / config.I_xx

    return float(theta_ddot), float(Fx_total), float(Fy_total), float(M_aero)


def _rk4_step_full(tp, td, dt, phi_f, phi_dot_f, phi_ddot_f,
                    phi_b, phi_dot_b, phi_ddot_b, geo_f, geo_b, config):
    """单步 RK4 — 返回完整的 (tp_new, td_new)."""
    # k1
    k1_tdd, _, _, _ = _compute_pitch_rhs_scalar(
        tp, td, phi_f, phi_dot_f, phi_ddot_f, phi_b, phi_dot_b, phi_ddot_b, geo_f, geo_b, config)

    # k2
    tp_k2 = tp + 0.5 * dt * td
    td_k2 = td + 0.5 * dt * k1_tdd
    k2_tdd, _, _, _ = _compute_pitch_rhs_scalar(
        tp_k2, td_k2, phi_f, phi_dot_f, phi_ddot_f, phi_b, phi_dot_b, phi_ddot_b, geo_f, geo_b, config)

    # k3
    tp_k3 = tp + 0.5 * dt * td_k2
    td_k3 = td + 0.5 * dt * k2_tdd
    k3_tdd, _, _, _ = _compute_pitch_rhs_scalar(
        tp_k3, td_k3, phi_f, phi_dot_f, phi_ddot_f, phi_b, phi_dot_b, phi_ddot_b, geo_f, geo_b, config)

    # k4
    tp_k4 = tp + dt * td_k3
    td_k4 = td + dt * k3_tdd
    k4_tdd, _, _, _ = _compute_pitch_rhs_scalar(
        tp_k4, td_k4, phi_f, phi_dot_f, phi_ddot_f, phi_b, phi_dot_b, phi_ddot_b, geo_f, geo_b, config)

    tp_new = tp + dt * (td + 2*td_k2 + 2*td_k3 + td_k4) / 6.0
    td_new = td + dt * (k1_tdd + 2*k2_tdd + 2*k3_tdd + k4_tdd) / 6.0

    return tp_new, td_new


# ============================================================
# ButterflyForceModel
# ============================================================

class ButterflyForceModel:
    """蝴蝶力模型 — 完整仿真管线.

    Usage:
        >>> config = SimulationConfig(alpha_front_deg=45, alpha_back_deg=8)
        >>> model = ButterflyForceModel(config)
        >>> output = model.simulate()        # t=0→t_end, 含俯仰ODE
        >>> # output.wings["FL"].force_body  # (N,3) 前翅左体轴力
        >>> # output.summary["L/W"]          # 升力/重量比

        >>> # 参数扫描
        >>> results = model.scan([
        ...     {"alpha_front_deg": 40, "alpha_back_deg": 8},
        ...     {"alpha_front_deg": 50, "alpha_back_deg": 8},
        ... ])
    """

    def __init__(self, config: SimulationConfig = None,
                 geo_override: dict = None):
        self.config = config or SimulationConfig()
        geo = default_geometry()
        if geo_override:
            for k, v in geo_override.items():
                if k in geo:
                    for attr, val in v.items():
                        setattr(geo[k], attr, val)
        self.geo = geo

    def simulate(self, progress: bool = True, use_numba: bool = True) -> SimulationOutput:
        """运行完整仿真: 运动学预计算 → RK4俯仰积分 → 力批处理.

        Args:
            progress: 打印进度信息.
            use_numba: 启用 numba JIT 加速 RK4 热循环 (需安装 numba).

        Returns:
            SimulationOutput: 含所有时间序列的完整输出.
        """
        cfg = self.config
        geo_f = self.geo["Front"]
        geo_b = self.geo["Back"]

        # ---- 运动学预计算 ----
        if progress:
            print(f"[ButterflyForceModel] 预计算运动学...")
        t_ext_f, pef, pdf, pddf, Tf = _precompute_one_period(cfg)
        t_ext_b, peb, pdb, pddb, Tb = _precompute_one_period(cfg)

        phase_sec_b = np.deg2rad(cfg.phase_diff_deg) / (2.0 * np.pi) * Tb

        # ---- 时间网格 ----
        n_steps = int(cfg.t_end / cfg.dt)
        t = np.linspace(0, cfg.t_end, n_steps)
        dt = cfg.dt

        # 插值运动学到全时间网格
        phi_f, phi_dot_f, phi_ddot_f = _interpolate_kinematics(
            t, t_ext_f, pef, pdf, pddf, 0.0, Tf)
        phi_b, phi_dot_b, phi_ddot_b = _interpolate_kinematics(
            t, t_ext_b, peb, pdb, pddb, phase_sec_b, Tb)

        # ---- RK4 俯仰积分 ----
        use_nb = use_numba and _HAS_NUMBA
        if progress:
            backend = "numba" if use_nb else "Python"
            print(f"[ButterflyForceModel] RK4积分({backend}): t_end={cfg.t_end}s, dt={cfg.dt*1e6:.0f}us, steps={n_steps}")

        tp = np.zeros(n_steps)
        tp[0] = np.deg2rad(cfg.theta0_deg)
        td = np.zeros(n_steps)

        pf_arr = phi_f; pdf_arr = phi_dot_f; pddf_arr = phi_ddot_f
        pb_arr = phi_b; pdb_arr = phi_dot_b; pddb_arr = phi_ddot_b

        if use_nb:
            # ---- 预计算 k_clap 数组 (速度-位置耦合, Lighthill公式) ----
            kcl_f = 1.0 + cfg.k_clap * compute_clap_fling_window(pf_arr, pdf_arr)
            kcl_b = 1.0 + cfg.k_clap * compute_clap_fling_window(pb_arr, pdb_arr)

            # ---- 打包 numba 参数 ----
            nb_params = (
                cfg.alpha_front_deg, cfg.alpha_back_deg,
                geo_f.S, geo_f.R, geo_f.c_avg, geo_f.r1, geo_f.r2_sq,
                geo_b.S, geo_b.R, geo_b.c_avg, geo_b.r1, geo_b.r2_sq,
                cfg.y_hinge_rel,
                cfg.z_front, cfg.z_back,
                cfg.rho, cfg.k_3d,
                cfg.m_total, cfg.g, cfg.I_xx, cfg.c_damp,
            )
            for i in range(n_steps - 1):
                tp[i+1], td[i+1] = _rk4_step_numba(
                    tp[i], td[i], dt,
                    pf_arr[i], pdf_arr[i], pddf_arr[i],
                    pb_arr[i], pdb_arr[i], pddb_arr[i],
                    kcl_f[i], kcl_b[i],
                    *nb_params)
        else:
            for i in range(n_steps - 1):
                tp[i+1], td[i+1] = _rk4_step_full(
                    tp[i], td[i], dt,
                    pf_arr[i], pdf_arr[i], pddf_arr[i],
                    pb_arr[i], pdb_arr[i], pddb_arr[i],
                    geo_f, geo_b, cfg)

        # ---- theta_ddot (后处理) ----
        tdd = np.zeros(n_steps)
        for i in range(n_steps):
            tdd_i, _, _, _ = _compute_pitch_rhs_scalar(
                tp[i], td[i],
                pf_arr[i], pdf_arr[i], pddf_arr[i],
                pb_arr[i], pdb_arr[i], pddb_arr[i],
                geo_f, geo_b, cfg)
            tdd[i] = tdd_i

        # ---- 力批处理 (向量化, 四翅) ----
        if progress:
            print(f"[ButterflyForceModel] 向量化力计算 (四翅)...")

        wings_out = {}
        wing_specs = [
            ("FL", geo_f, cfg.alpha_front_deg, cfg.x_hinge_left,  cfg.y_hinge_rel, cfg.z_front, -1.0),
            ("FR", geo_f, cfg.alpha_front_deg, cfg.x_hinge_right, cfg.y_hinge_rel, cfg.z_front, +1.0),
            ("BL", geo_b, cfg.alpha_back_deg,  cfg.x_hinge_left,  cfg.y_hinge_rel, cfg.z_back,  -1.0),
            ("BR", geo_b, cfg.alpha_back_deg,  cfg.x_hinge_right, cfg.y_hinge_rel, cfg.z_back,  +1.0),
        ]

        for name, geo, alpha_inst, x_h, y_hinge_rel, z_hinge, side_sign in wing_specs:
            # 该翅的运动学 (FL/FR共享Front, BL/BR共享Back)
            if name.startswith("F"):
                p, pd, pdd = phi_f, phi_dot_f, phi_ddot_f
            else:
                p, pd, pdd = phi_b, phi_dot_b, phi_ddot_b

            r = compute_wing_forces_vec(
                p, pd, pdd, tp, td, alpha_inst, geo, y_hinge_rel, cfg)

            cop = compute_cop_vec(p, geo, x_h, y_hinge_rel, z_hinge, float(side_sign))
            # 力矩 = r × F (对CG, CG在原点)
            M_body = np.cross(cop, r["F_body"])

            # 世界系
            F_world = body_to_world(r["F_body"], tp)
            cop_w = body_to_world(cop, tp)
            M_world = body_to_world(M_body, tp)

            # 摇杆分解
            pv, pm, ra = rocker_decompose(r["F_body"], cop, p, cfg, x_h, y_hinge_rel)

            wings_out[name] = WingOutput(
                name=name,
                force_body=r["F_body"],
                force_world=F_world,
                cop_body=cop,
                cop_world=cop_w,
                moment_body=M_body,
                moment_world=M_world,
                rocker_principal_vec=pv,
                rocker_principal_moment=pm,
                rocker_angle_rad=ra,
                alpha_eff_deg=r["alpha_eff_deg"],
                C_L=r["C_L"], C_D=r["C_D"],
                phi=p, phi_dot=pd, phi_ddot=pdd,
            )

        # ---- Summary ----
        # v6.9: 升力沿体轴 +Y, 世界系经 R_x(theta_p) 变换
        Fy_body_total = np.zeros(n_steps)
        Fx_body_total = np.zeros(n_steps)
        Fy_world_total = np.zeros(n_steps)
        for name in ["FL", "FR", "BL", "BR"]:
            Fy_body_total += wings_out[name].force_body[:, 1]
            Fx_body_total += wings_out[name].force_body[:, 0]
            Fy_world_total += wings_out[name].force_world[:, 1]

        weight_N = cfg.m_total * cfg.g
        half = n_steps // 2
        steady_idx = int(cfg.steady_start / cfg.dt) if cfg.steady_start < cfg.t_end else half
        avg_Fy_body = np.mean(Fy_body_total[steady_idx:])
        avg_Fy_world = np.mean(Fy_world_total[steady_idx:])
        avg_Fx_body = np.mean(Fx_body_total[steady_idx:])
        lw_body = avg_Fy_body / weight_N
        lw_world = avg_Fy_world / weight_N  # 真正的物理升重比

        tp_deg = np.rad2deg(tp)
        peak_all = float(np.max(np.abs(tp_deg)))
        n90 = int(np.sum(np.abs(tp_deg) > 90))

        summary = {
            "L/W": float(lw_world),         # 世界系升重比 (物理悬停判据)
            "L/W_body": float(lw_body),     # 体轴系升重比 (机构分析用)
            "avg_Fy_body_mN": float(avg_Fy_body * 1000),
            "avg_Fy_world_mN": float(avg_Fy_world * 1000),
            "avg_Fx_body_mN": float(avg_Fx_body * 1000),
            "weight_mN": float(weight_N * 1000),
            "peak_theta_deg": peak_all,
            "n_exceed_90": n90,
            "n_steps": n_steps,
            "dt_s": float(dt),
            "t_end_s": float(cfg.t_end),
            "steady_start_s": float(cfg.steady_start),
        }

        if progress:
            status = "✅ STABLE" if n90 == 0 else "❌ DIVERGED"
            print(f"[ButterflyForceModel] {status} | L/W={lw_world:.3f} (world) / {lw_body:.3f} (body) | peak={peak_all:.1f}° | n90={n90} | Fy_body={avg_Fy_body*1000:+.0f}mN | Fy_world={avg_Fy_world*1000:+.0f}mN")

        return SimulationOutput(
            t=t, theta_p=tp, theta_dot=td, theta_ddot=tdd,
            wings=wings_out, summary=summary, config=cfg,
        )

    def scan(self, overrides: List[dict],
             t_end: float = None, dt: float = None,
             progress: bool = True) -> List[dict]:
        """参数扫描.

        Args:
            overrides: 参数覆盖列表, 每项是 {param_name: value} dict.
            t_end: 覆盖仿真时间 (默认用config值).
            dt: 覆盖时间步长 (默认用config值).
            progress: 打印进度.

        Returns:
            [{**override, "L/W": ..., "peak_deg": ..., "n90": ..., "Fz_mN": ...}, ...]
        """
        results = []
        for i, ov in enumerate(overrides):
            # 构建新配置
            cfg_dict = self.config.__dict__.copy()
            cfg_dict.update(ov)
            if t_end is not None:
                cfg_dict["t_end"] = t_end
            if dt is not None:
                cfg_dict["dt"] = dt
            cfg = SimulationConfig(**{k: v for k, v in cfg_dict.items()
                                       if k in SimulationConfig.__dataclass_fields__})

            if progress:
                print(f"\n[{i+1}/{len(overrides)}] {ov}")

            # 临时模型
            tmp_model = ButterflyForceModel(cfg)
            # 重用geo
            tmp_model.geo = self.geo
            out = tmp_model.simulate(progress=progress)
            s = out.summary
            res = {**ov,
                   "L/W": s["L/W"], "peak_deg": s["peak_theta_deg"],
                   "n90": s["n_exceed_90"],
                   "Fy_body_mN": s["avg_Fy_body_mN"],
                   "Fy_world_mN": s["avg_Fy_world_mN"],
                   }
            results.append(res)

        if progress:
            print(f"\nScan done. {len(results)} combos.")
        return results


# ============================================================
# 便捷函数
# ============================================================

def scan_parameters(base_config: SimulationConfig,
                    vary: Dict[str, list],
                    t_end: float = 3.0,
                    dt: float = 50e-6,
                    progress: bool = True) -> List[dict]:
    """参数扫描便捷函数 — 笛卡尔积遍历.

    Args:
        base_config: 基础配置.
        vary: 参数范围, e.g. {"alpha_front_deg": [55,60], "alpha_back_deg": [8,10]}.
        t_end: 扫描仿真时间.
        dt: 扫描时间步长.
        progress: 打印进度.

    Returns:
        结果列表, 按 L/W 降序排列.
    """
    import itertools
    keys = list(vary.keys())
    values = list(vary.values())
    overrides = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    model = ButterflyForceModel(base_config)
    results = model.scan(overrides, t_end=t_end, dt=dt, progress=progress)

    results.sort(key=lambda x: x["L/W"], reverse=True)
    return results


# ============================================================
# __main__ — 验证
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("butterfly_forces.py — 验证测试")
    print(f"  版本: v{get_version()}  |  配置文件: config/design_v69.yaml")
    print("=" * 70)

    # Test 1: 基线参数快速验证 (v6.9 DESIGN_v69 默认参数, 3s, 50us)
    print("\n--- Test 1: 默认参数 (DESIGN_v69, t=3s) ---")
    cfg1 = SimulationConfig(t_end=3.0, dt=50e-6)
    print(f"  默认参数: α_f={cfg1.alpha_front_deg}, α_b={cfg1.alpha_back_deg}, "
          f"R={cfg1.mech_R}, phase={cfg1.phase_diff_deg}, k_clap={cfg1.k_clap}, f={cfg1.f}")
    m1 = ButterflyForceModel(cfg1)
    out1 = m1.simulate(progress=True)
    s1 = out1.summary
    print(f"  L/W={s1['L/W']:.3f} (expected ~4.5) | peak={s1['peak_theta_deg']:.1f}° | n90={s1['n_exceed_90']}")
    print(f"  α_eff FL: [{np.min(out1.wings['FL'].alpha_eff_deg):.0f}°, {np.max(out1.wings['FL'].alpha_eff_deg):.0f}°]")
    print(f"  Fy_body={s1['avg_Fy_body_mN']:+.0f}mN | Fy_world={s1['avg_Fy_world_mN']:+.0f}mN | weight={s1['weight_mN']:.0f}mN")
    print(f"  Wings: {list(out1.wings.keys())}")
    for name, wo in out1.wings.items():
        print(f"    {name}: F_body max={np.max(np.abs(wo.force_body)):.4f}N  "
              f"CoP_z={wo.cop_body[0,2]:.4f}m  "
              f"rocker_pv_max={np.max(np.abs(wo.rocker_principal_vec)):.4f}N  "
              f"rocker_pm_max={np.max(np.abs(wo.rocker_principal_moment)):.6f}N·m")

    # Test 2: 相位差影响 (使用默认 v6.9 机构参数, 仅改变相位)
    print("\n--- Test 2: 相位差 -10° ---")
    cfg2 = SimulationConfig(phase_diff_deg=-10, t_end=1.0, dt=50e-6)
    m2 = ButterflyForceModel(cfg2)
    out2 = m2.simulate(progress=True)
    print(f"  L/W={out2.summary['L/W']:.3f} | peak={out2.summary['peak_theta_deg']:.1f}°")

    # Test 3: α_f 小范围扫描 (使用默认 v6.9 机构参数)
    print("\n--- Test 3: α_f/α_b 扫描 ---")
    results = scan_parameters(
        SimulationConfig(),
        {"alpha_front_deg": [50, 55, 60, 70], "alpha_back_deg": [3, 5, 8]},
        t_end=3.0, dt=50e-6, progress=True,
    )
    print(f"\n  Top results:")
    for r in results:
        print(f"    α_f={r['alpha_front_deg']} α_b={r['alpha_back_deg']}  L/W={r['L/W']:.3f}  peak={r['peak_deg']:.1f}°  n90={r['n90']}")

    print("\n✅ 所有测试完成.")
