#!/usr/bin/env python3
"""
蝴蝶力输出模块 — 对外调用接口。

提供:
  - ButterflyForceModel: 完整仿真（运动学→俯仰ODE→力输出）
  - scan_parameters(): 参数扫描
  - SimulationConfig: 所有可配置参数
  - SimulationOutput / WingOutput: 结构化输出

坐标系:
  体轴 (Body): 原点=CG, X=前, Y=右, Z=上
  世界 (World): θ_p=0时与体轴重合
  机构平面 = 体轴XZ平面 (四连杆运动平面)
  机构x→体轴X, 机构y→体轴Z, 机构法向→体轴Y(摇杆旋转轴)

摇杆分解:
  主矢 = 合力沿摇杆方向(A→P2)的分量, 通过连杆传至曲柄
  主矩 = 对摇杆枢轴A的力矩在Y轴上的分量, 驱动/制动摇杆的有效扭矩

v6.3 LEV/Lee混合C_L/C_D模型, 含气动俯仰阻尼, RK4积分.
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


# ---- 共享设计参数: v6.8 最终推荐 (物理合理性约束) ----
DESIGN_v68 = {
    "alpha_front_deg": 60.0,
    "alpha_back_deg": 3.0,
    "phase_diff_deg": -15.0,
    "mech_a": 7.6,            # A点y坐标
    "mech_b": 1.71,           # B点x坐标
    "mech_R": 3.8,            # 曲柄半径
    "mech_c": 7.1,            # 连杆长度
    "mech_l": 5.0,            # 摇杆长度
    "phi_offset_deg": 0.0,    # 新机构无偏移
    "rotation": "cw",
    "f": 17.0,
    "rho": 1.225,
    "m_total": 0.020,
    "I_yy": 3e-5,
    "d_cg": 0.015,
    "x_front": 0.025,
    "x_back": -0.025,
    "g": 9.81,
    "k_3d": 0.7,
    "C_rot": 1.5,
    "r_rot": 0.5,
    "k_clap": 0.3,
    "c_damp": 5e-4,
    "dt": 10e-6,
    "t_end": 10.0,
    "theta0_deg": 0.0,
    "steady_start": 5.0,
}


# ---- 共享设计参数: v6.8 最终推荐 (物理合理性约束) ----
DESIGN_v68 = {
    "alpha_front_deg": 45.0,
    "alpha_back_deg": 8.0,
    "phase_diff_deg": -20.0,
    "mech_a": 6.0,
    "mech_b": 6.97,
    "mech_R": 2.50,
    "mech_c": 14.00,
    "mech_l": 8.00,
    "phi_offset_deg": -30.0,
    "rotation": "cw",
    "f": 17.0,
    "rho": 1.225,
    "m_total": 0.020,
    "I_yy": 3e-5,
    "d_cg": 0.015,
    "x_front": 0.025,
    "x_back": -0.025,
    "g": 9.81,
    "k_3d": 0.7,
    "C_rot": 1.5,
    "r_rot": 0.5,
    "k_clap": 0.3,
    "c_damp": 5e-4,
    "dt": 10e-6,
    "t_end": 10.0,
    "theta0_deg": 0.0,
    "steady_start": 5.0,
}


@dataclass
class SimulationConfig:
    """仿真全参数配置 — 所有参数可被scan覆盖."""

    # ---- 翅膀安装 ----
    alpha_front_deg: float = DESIGN_v68["alpha_front_deg"]
    alpha_back_deg: float = DESIGN_v68["alpha_back_deg"]
    phase_diff_deg: float = DESIGN_v68["phase_diff_deg"]

    # ---- 四连杆机构 ----
    mech_a: float = DESIGN_v68["mech_a"]
    mech_b: float = DESIGN_v68["mech_b"]
    mech_R: float = DESIGN_v68["mech_R"]
    mech_c: float = DESIGN_v68["mech_c"]
    mech_l: float = DESIGN_v68["mech_l"]
    phi_offset_deg: float = DESIGN_v68["phi_offset_deg"]
    rotation: str = DESIGN_v68["rotation"]

    # ---- 物理 ----
    f: float = DESIGN_v68["f"]
    rho: float = DESIGN_v68["rho"]
    m_total: float = DESIGN_v68["m_total"]
    I_yy: float = DESIGN_v68["I_yy"]
    d_cg: float = DESIGN_v68["d_cg"]
    x_front: float = DESIGN_v68["x_front"]
    x_back: float = DESIGN_v68["x_back"]
    g: float = DESIGN_v68["g"]

    # ---- 数值 ----
    dt: float = DESIGN_v68["dt"]
    t_end: float = DESIGN_v68["t_end"]
    theta0_deg: float = DESIGN_v68["theta0_deg"]
    steady_start: float = DESIGN_v68["steady_start"]

    # ---- 气动系数 ----
    k_3d: float = DESIGN_v68["k_3d"]
    C_rot: float = DESIGN_v68["C_rot"]
    r_rot: float = DESIGN_v68["r_rot"]
    k_clap: float = DESIGN_v68["k_clap"]
    c_damp: float = DESIGN_v68["c_damp"]

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
                         x_wing, rho, k_3d, k_clap):
    """Numba标量版单翅气动力 — 返回 (Fz_body, Fx_body)."""
    psi = phi + theta_p
    Omega = phi_dot + theta_dot
    abs_Omega = abs(Omega)
    U = abs_Omega * R

    # 俯仰气动阻尼: Δα = atan(θ̇_p·x_wing / U)
    # U≈0时Δα可达±90°, 但此时F_trans∝U²≈0, 瞬态大攻角无实际力贡献
    v_pitch = theta_dot * x_wing
    if abs_Omega < 1e-6:
        delta_alpha_rad = 0.0
    else:
        delta_alpha_rad = np.arctan2(v_pitch, U)

    # 有效攻角：翅膀弦线相对拍动平面(体轴XZ平面)的角度
    # η = α_install (文献[32]: α=η下拍, α=π-η上拍)
    # θ_p 使拍动平面整体倾斜, 通过力投影和重力矩体现, 不进入气动攻角
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

    # 体轴系力: Fx=前, Fz=上
    # sign_Omega 用 tanh 平滑过渡, 避免拍动反转时力跳变
    sign_Omega = np.tanh(Omega / 2.0)
    Fx = np.sin(psi) * (sign_Omega * D_eff - L_eff)
    Fz = np.cos(psi) * (L_eff - sign_Omega * D_eff)

    if abs_Omega < 1e-6:
        Fx = 0.0
        Fz = 0.0

    return Fz, Fx


@njit(cache=True, fastmath=True)
def _pitch_rhs_numba(theta_p, theta_dot,
                      pf, pdf, pddf, pb, pdb, pddb,
                      k_clap_f, k_clap_b,
                      alpha_f_deg, alpha_b_deg,
                      S_f, R_f, c_avg_f, r1_f, r2_sq_f,
                      S_b, R_b, c_avg_b, r1_b, r2_sq_b,
                      x_front, x_back,
                      rho, k_3d,
                      m_total, g, d_cg, I_yy, c_damp):
    """Numba标量版俯仰ODE右端项 — 返回 (theta_ddot, Fx_total, Fz_total, M_aero)."""
    # 前翅
    Fz_f, Fx_f = _wing_forces_scalar(
        pf, pdf, pddf, theta_p, theta_dot,
        alpha_f_deg, S_f, R_f, c_avg_f, r1_f, r2_sq_f,
        x_front, rho, k_3d, k_clap_f)

    # 后翅
    Fz_b, Fx_b = _wing_forces_scalar(
        pb, pdb, pddb, theta_p, theta_dot,
        alpha_b_deg, S_b, R_b, c_avg_b, r1_b, r2_sq_b,
        x_back, rho, k_3d, k_clap_b)

    # 左右对称 ×2
    Fx_total = 2.0 * (Fx_f + Fx_b)
    Fz_total = 2.0 * (Fz_f + Fz_b)
    M_aero = 2.0 * (-x_front * Fz_f - x_back * Fz_b)

    M_grav = -m_total * g * d_cg * np.sin(theta_p)
    M_damp = -c_damp * theta_dot
    theta_ddot = (M_aero + M_grav + M_damp) / I_yy

    return theta_ddot, Fx_total, Fz_total, M_aero


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
                             alpha_install_deg, geo: WingGeometry, x_wing, config: SimulationConfig):
    """向量化单翅气动力计算.

    返回 dict:
      F_body: (N,3) 体轴力
      alpha_eff_deg: (N,)
      C_L, C_D: (N,)
      psi: (N,) 世界系中翅膀角度
      Omega: (N,) 总角速度
    """
    N = len(phi)
    psi = phi + theta_p
    Omega = phi_dot + theta_dot
    U = np.abs(Omega) * geo.R

    # 俯仰气动阻尼: Δα = atan(θ̇_p·x_wing / U)
    # U≈0时Δα可达±90°, 但此时F_trans∝U²≈0, 瞬态大攻角无实际力贡献
    v_pitch = theta_dot * x_wing
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_alpha_rad = np.arctan2(v_pitch, U + 1e-6)

    # 有效攻角：翅膀弦线相对拍动平面(体轴XZ平面)的角度
    # η = α_install (文献[32]: α=η下拍, α=π-η上拍)
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

    # Clap-and-Fling: 速度-位置耦合增强 (Lighthill Γ=g(λ)φ̇c²)
    # k_extra ∝ |φ̇|/φ̇_peak × cos²窗(距端点距离)
    # 端点处 φ̇→0 增强自动归零, 增强峰值在端点附近速度尚存处
    k_clap_extra = compute_clap_fling_window(phi, phi_dot, edge_width=0.10)
    k_clap = 1.0 + config.k_clap * k_clap_extra

    L_eff = (L_trans + F_AM + F_rot) * k_clap
    D_eff = D_trans * k_clap

    # 体轴系力: Fx=前, Fz=上
    Fx = np.sin(psi) * (sign_Omega * D_eff - L_eff)
    Fz = np.cos(psi) * (L_eff - sign_Omega * D_eff)

    mask_still = np.abs(Omega) < 1e-6
    Fx = np.where(mask_still, 0.0, Fx)
    Fz = np.where(mask_still, 0.0, Fz)

    F_body = np.zeros((N, 3))
    F_body[:, 0] = Fx
    F_body[:, 2] = Fz

    return {
        "F_body": F_body,
        "alpha_eff_deg": alpha_eff_deg,
        "C_L": C_L, "C_D": C_D,
        "psi": psi, "Omega": Omega, "k_clap": k_clap,
    }


def compute_cop_vec(phi, geo: WingGeometry, x_wing, y_hinge, z_hinge, side_sign):
    """向量化气动中心位置.

    side_sign: +1 (右翅, +Y) / -1 (左翅, -Y)

    CoP = hinge_pos + spanwise_offset + chordwise_offset.
    - 展向: r1 * R 沿 Y
    - 弦向: c_avg/4 在 XZ 平面内, 方向随 phi 变化
    """
    N = len(phi)
    cop = np.zeros((N, 3))
    cop[:, 0] = x_wing + (geo.c_avg / 4.0) * np.cos(phi)   # 弦向在X的投影
    cop[:, 1] = y_hinge + side_sign * geo.r1 * geo.R         # 展向
    cop[:, 2] = z_hinge + (geo.c_avg / 4.0) * np.sin(phi)   # 弦向在Z的投影
    return cop


def rocker_decompose(F_body, cop_body, phi, config: SimulationConfig, x_wing, y_hinge):
    """摇杆力分解: 主矢 + 主矩.

    摇杆枢轴 A (机构坐标): (0, mech_a) mm
    在体轴系: A_body = (x_wing, y_hinge, mech_a/1000)

    机构平面 = 体轴 XZ 平面.
    摇杆方向: 从 A 指向 P2, 在 XZ 平面内变化.
    摇杆机构角 = phi - phi_offset (即原始solve_phi输出).

    主矢: F_body 沿摇杆方向的分量
    主矩: 对 A 的力矩在 Y 轴上的分量
    """
    N = len(phi)
    a_m = config.mech_a / 1000.0
    l_m = config.mech_l / 1000.0

    # 枢轴 A 在体轴系
    A_body = np.array([x_wing, y_hinge, a_m])

    # 摇杆机构角 (去除安装偏角后的原始机构角度)
    phi_mech = phi - np.deg2rad(config.phi_offset_deg)

    # 摇杆方向单位矢量 (在 XZ 平面内)
    # 摇杆从A到P2, 方向角 = phi_mech (机构x→体轴X, 机构y→体轴Z)
    d_rocker = np.zeros((N, 3))
    d_rocker[:, 0] = np.cos(phi_mech)  # 体轴X分量
    d_rocker[:, 2] = np.sin(phi_mech)  # 体轴Z分量
    # 归一化 (应该已是单位矢量, 但确保)
    norm = np.sqrt(d_rocker[:, 0]**2 + d_rocker[:, 2]**2)
    d_rocker[:, 0] /= norm
    d_rocker[:, 2] /= norm

    # ---- 主矢: 力沿摇杆方向的分量 ----
    F_dot_rocker = np.sum(F_body * d_rocker, axis=1)  # (N,)
    principal_vec = d_rocker * F_dot_rocker[:, np.newaxis]  # (N,3)

    # ---- 主矩: 对枢轴A的力矩在Y轴上的分量 ----
    r_from_A = cop_body - A_body[np.newaxis, :]  # (N,3) 从A到CoP的矢量
    M_about_A = np.cross(r_from_A, F_body)        # (N,3) 对A的力矩
    # Y分量 = 机构平面内的有效扭矩
    principal_moment = np.zeros((N, 3))
    principal_moment[:, 1] = M_about_A[:, 1]      # 仅保留Y分量

    # 摇杆方向角 (用于调用方参考)
    rocker_angle = phi_mech

    return principal_vec, principal_moment, rocker_angle


# ============================================================
# 坐标变换
# ============================================================

def body_to_world(vec_body, theta_p):
    """体轴系→世界系: R_y(θ_p) 旋转.

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
    vec_w[:, 0] = cos_t * vec[:, 0] + sin_t * vec[:, 2]
    vec_w[:, 1] = vec[:, 1]
    vec_w[:, 2] = -sin_t * vec[:, 0] + cos_t * vec[:, 2]

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
        config.alpha_front_deg, geo_f, config.x_front, config)
    r_b = compute_wing_forces_vec(
        np.array([phi_b]), np.array([phi_dot_b]), np.array([phi_ddot_b]),
        np.array([theta_p]), np.array([theta_dot]),
        config.alpha_back_deg, geo_b, config.x_back, config)

    Fz_f = r_f["F_body"][0, 2]
    Fz_b = r_b["F_body"][0, 2]
    Fx_f = r_f["F_body"][0, 0]
    Fx_b = r_b["F_body"][0, 0]

    # 左右对称 x2
    Fx_total = 2.0 * (Fx_f + Fx_b)
    Fz_total = 2.0 * (Fz_f + Fz_b)
    M_aero = 2.0 * (-config.x_front * Fz_f - config.x_back * Fz_b)

    M_grav = -config.m_total * config.g * config.d_cg * np.sin(theta_p)
    M_damp = -config.c_damp * theta_dot
    theta_ddot = (M_aero + M_grav + M_damp) / config.I_yy

    return float(theta_ddot), float(Fx_total), float(Fz_total), float(M_aero)


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
                cfg.x_front, cfg.x_back,
                cfg.rho, cfg.k_3d,
                cfg.m_total, cfg.g, cfg.d_cg, cfg.I_yy, cfg.c_damp,
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
            ("FL", geo_f, cfg.alpha_front_deg, cfg.x_front, -0.010, 0.0, -1),
            ("FR", geo_f, cfg.alpha_front_deg, cfg.x_front, +0.010, 0.0, +1),
            ("BL", geo_b, cfg.alpha_back_deg,  cfg.x_back,  -0.010, 0.0, -1),
            ("BR", geo_b, cfg.alpha_back_deg,  cfg.x_back,  +0.010, 0.0, +1),
        ]

        for name, geo, alpha_inst, x_w, y_hinge, z_hinge, side_sign in wing_specs:
            # 该翅的运动学 (FL/FR共享Front, BL/BR共享Back)
            if name.startswith("F"):
                p, pd, pdd = phi_f, phi_dot_f, phi_ddot_f
            else:
                p, pd, pdd = phi_b, phi_dot_b, phi_ddot_b

            r = compute_wing_forces_vec(
                p, pd, pdd, tp, td, alpha_inst, geo, x_w, cfg)

            cop = compute_cop_vec(p, geo, x_w, y_hinge, z_hinge, float(side_sign))
            # 力矩 = r × F (对CG, CG在原点)
            M_body = np.cross(cop, r["F_body"])

            # 世界系
            F_world = body_to_world(r["F_body"], tp)
            cop_w = body_to_world(cop, tp)
            M_world = body_to_world(M_body, tp)

            # 摇杆分解
            pv, pm, ra = rocker_decompose(r["F_body"], cop, p, cfg, x_w, y_hinge)

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
        # 体轴系 Fz (与原始 scan_v6_3 一致)
        Fz_body_total = np.zeros(n_steps)
        Fx_body_total = np.zeros(n_steps)
        Fz_world_total = np.zeros(n_steps)
        for name in ["FL", "FR", "BL", "BR"]:
            Fz_body_total += wings_out[name].force_body[:, 2]
            Fx_body_total += wings_out[name].force_body[:, 0]
            Fz_world_total += wings_out[name].force_world[:, 2]

        weight_N = cfg.m_total * cfg.g
        half = n_steps // 2
        steady_idx = int(cfg.steady_start / cfg.dt) if cfg.steady_start < cfg.t_end else half
        avg_Fz_body = np.mean(Fz_body_total[steady_idx:])
        avg_Fz_world = np.mean(Fz_world_total[steady_idx:])
        avg_Fx_body = np.mean(Fx_body_total[steady_idx:])
        lw_body = avg_Fz_body / weight_N
        lw_world = avg_Fz_world / weight_N  # 真正的物理升重比

        tp_deg = np.rad2deg(tp)
        peak_all = float(np.max(np.abs(tp_deg)))
        n90 = int(np.sum(np.abs(tp_deg) > 90))

        summary = {
            "L/W": float(lw_world),         # 世界系升重比 (物理悬停判据)
            "L/W_body": float(lw_body),     # 体轴系升重比 (机构分析用)
            "avg_Fz_body_mN": float(avg_Fz_body * 1000),
            "avg_Fz_world_mN": float(avg_Fz_world * 1000),
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
            print(f"[ButterflyForceModel] {status} | L/W={lw_world:.3f} (world) / {lw_body:.3f} (body) | peak={peak_all:.1f}° | n90={n90} | Fz_body={avg_Fz_body*1000:+.0f}mN | Fz_world={avg_Fz_world*1000:+.0f}mN")

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
                   "Fz_body_mN": s["avg_Fz_body_mN"],
                   "Fz_world_mN": s["avg_Fz_world_mN"],
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
    print("=" * 70)

    # Test 1: 基线参数快速验证 (v6.8 DESIGN_v68 默认参数, 3s, 50us)
    print("\n--- Test 1: 默认参数 (DESIGN_v68, t=3s) ---")
    cfg1 = SimulationConfig(t_end=3.0, dt=50e-6)
    print(f"  默认参数: α_f={cfg1.alpha_front_deg}, α_b={cfg1.alpha_back_deg}, "
          f"R={cfg1.mech_R}, k_clap={cfg1.k_clap}, f={cfg1.f}")
    m1 = ButterflyForceModel(cfg1)
    out1 = m1.simulate(progress=True)
    s1 = out1.summary
    print(f"  L/W={s1['L/W']:.3f} (expected ~2.45) | peak={s1['peak_theta_deg']:.1f}° | n90={s1['n_exceed_90']}")
    print(f"  α_eff FL: [{np.min(out1.wings['FL'].alpha_eff_deg):.0f}°, {np.max(out1.wings['FL'].alpha_eff_deg):.0f}°]")
    print(f"  Fz_body={s1['avg_Fz_body_mN']:+.0f}mN | Fz_world={s1['avg_Fz_world_mN']:+.0f}mN | weight={s1['weight_mN']:.0f}mN")
    print(f"  Wings: {list(out1.wings.keys())}")
    for name, wo in out1.wings.items():
        print(f"    {name}: F_body max={np.max(np.abs(wo.force_body)):.4f}N  "
              f"CoP_y={wo.cop_body[0,1]:.4f}m  "
              f"rocker_pv_max={np.max(np.abs(wo.rocker_principal_vec)):.4f}N  "
              f"rocker_pm_max={np.max(np.abs(wo.rocker_principal_moment)):.6f}N·m")

    # Test 2: 相位差影响
    print("\n--- Test 2: 相位差 -10° ---")
    cfg2 = SimulationConfig(
        alpha_front_deg=45, alpha_back_deg=8,
        phase_diff_deg=-10,
        mech_a=6, mech_R=2.50, phi_offset_deg=-30,
        f=17, c_damp=5e-4, rotation='cw',
        t_end=1.0, dt=50e-6,
    )
    m2 = ButterflyForceModel(cfg2)
    out2 = m2.simulate(progress=True)
    print(f"  L/W={out2.summary['L/W']:.3f} | peak={out2.summary['peak_theta_deg']:.1f}°")

    # Test 3: α_f 小范围扫描
    print("\n--- Test 3: α_f/α_b 扫描 ---")
    results = scan_parameters(
        SimulationConfig(
            phase_diff_deg=-20, mech_a=6, mech_R=2.50,
            phi_offset_deg=-30, f=17, c_damp=5e-4, rotation='cw',
        ),
        {"alpha_front_deg": [40, 45, 50], "alpha_back_deg": [5, 8, 10]},
        t_end=3.0, dt=50e-6, progress=True,
    )
    print(f"\n  Top results:")
    for r in results:
        print(f"    α_f={r['alpha_front_deg']} α_b={r['alpha_back_deg']}  L/W={r['L/W']:.3f}  peak={r['peak_deg']:.1f}°  n90={r['n90']}")

    print("\n✅ 所有测试完成.")
