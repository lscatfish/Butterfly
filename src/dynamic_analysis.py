#!/usr/bin/env python3
"""
仿生蝴蝶翅膀动态气动力分析
功能：
1. 单周期时间域力曲线（平动升力/阻力、旋转力、附加质量力）
2. 参数扫描：频率、幅度、攻角对升/阻力的影响
3. 运动学由前置连杆机构（mechanism.py）生成
4. 生成高清图表 + Markdown 报告
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import integrate
from pathlib import Path
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from mechanism import wing_kinematics

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
MM_TO_M = 1e-3

# ==================== 用户设计参数 ====================
AERO_PARAMS = {
    "rho": 1.225,           # 空气密度 kg/m³
    "nu": 1.46e-5,          # 运动粘度 m²/s
    "m_total": 0.025,       # 总质量 25g
    "m_wing_total": 0.004,  # 四翅总质量 4g
    "f": 15.0,              # 典型频率 Hz (范围 15-20)
    # 机构角度不做缩放，使用 mechanism.py 原始输出
    # a=8.0 时原始范围: [-2.8°, 30.5°]（负值=下拍，正值=上拍）
    "alpha_deg": 45.0,      # 攻角 °（固定安装角）
    "mech_a": 8.0,          # 机构主点圆心 x（可调 6-14，控制摆幅）
}


def load_geometry():
    """从 JSON 读取 wing geometry"""
    json_path = DATA_DIR / 'wing_analysis_results.json'
    if not json_path.exists():
        raise FileNotFoundError(f"请先运行 analyze_dxf.py 生成 {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    geo = {}
    for g in data['geometry']:
        geo[g['name']] = {
            'S': g['S'],
            'R': g['R'],
            'c_avg': g['c_avg'],
            'r1': g['r1'],
            'r2_sq': g['r2_sq'],
            'AR': g['AR'],
            'S_mm2': g['S_mm2'],
            'R_mm': g['R_mm'],
        }
    return geo


def cl_cd(alpha_deg):
    """升阻力系数（alpha 单位：度）"""
    C_L = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
    C_D = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))
    return C_L, C_D


def simulate_cycle(geo_item, params, n_points=2000):
    """模拟一个完整周期的气动力
    
    运动学由前置连杆机构（mechanism.py）生成。
    返回 dict 包含 t, phi_deg, phi_dot, alpha_deg, 以及各力分量。
    """
    f = params['f']
    alpha0 = np.deg2rad(params['alpha_deg'])
    rho = params['rho']
    mech_a = params.get('mech_a', 8.0)

    S = geo_item['S']
    R = geo_item['R']
    c_avg = geo_item['c_avg']
    r1 = geo_item['r1']
    r2_sq = geo_item['r2_sq']

    # ========== 机构运动学 ==========
    # 输出原始机构角度，不做幅度缩放（保持前级机构特征）
    t, phi, phi_dot, phi_ddot, mech_info = wing_kinematics(
        f=f, params={'a': mech_a}, n_points=n_points)

    # ========== 固定攻角 ==========
    alpha = alpha0 * np.ones_like(t)
    alpha_dot = np.zeros_like(t)
    
    # 计算各力分量 — C_L 基于瞬时速度方向（φ̇ 符号决定有效攻角）
    C_L_arr = np.zeros_like(t)
    C_D_arr = np.zeros_like(t)
    for i in range(len(t)):
        if phi_dot[i] <= 0:
            # 翅膀向下运动 → 相对来流从下方 → 有效攻角 +α
            C_L_arr[i], C_D_arr[i] = cl_cd(np.degrees(alpha0))
        else:
            # 翅膀向上运动 → 相对来流从上方 → 有效攻角 -α
            C_L_arr[i], C_D_arr[i] = cl_cd(-np.degrees(alpha0))
    
    # 平动分量（与 phi_dot^2 成正比）
    # 注意：C_L 的符号已经包含了方向信息
    # 下拍 C_L > 0 → 升力向上；上拍 C_L < 0 → 升力向下
    F_trans_lift = 0.5 * rho * C_L_arr * (phi_dot * R)**2 * S * r2_sq
    F_trans_drag = 0.5 * rho * C_D_arr * (phi_dot * R)**2 * S * r2_sq
    
    # 旋转力 = 0（因为 alpha_dot = 0，翅膀不能扭转）
    F_rot = np.zeros_like(t)
    
    # 附加质量力：F_AM = -(ρπc²/4)·φ̈·R·r₁·sin(α)
    # 阻力加速度（a_n = φ̈·R·sinα），与 φ̇ 方向无关
    F_AM = -(rho * np.pi * c_avg**2 / 4.0) * phi_ddot * R * r1 * np.sin(alpha0)
    
    # 总力
    F_lift = F_trans_lift + F_rot + F_AM
    F_drag = np.abs(F_trans_drag)  # 阻力始终为正（与运动方向相反）
    
    # ========== 功率计算 ==========
    # 单翅质量（四翅均分）
    m_wing_total = params.get('m_wing_total', 0.004)
    m_w = m_wing_total / 4.0
    
    # 转动惯量（绕转轴，基于 r2_sq）
    I_w = m_w * R**2 * r2_sq
    
    # 气动功率：克服空气阻力的功率 = 阻力矩 × 角速度
    # 阻力分布在整个翼面，等效力臂 ≈ R × r1（一阶矩位置）
    # P_aero = F_drag × |φ̇| × R × r1
    P_aero = F_drag * np.abs(phi_dot) * R * r1
    
    # 惯性功率：加速/减速翅膀的功率 = 惯性力矩 × 角速度
    # P_inertial = I_w × φ̈ × φ̇
    P_inertial = I_w * phi_ddot * phi_dot
    
    # 总功率
    P_total = P_aero + P_inertial
    
    return {
        't': t,
        'phi_deg': np.degrees(phi),
        'phi_dot': phi_dot,
        'phi_ddot': phi_ddot,
        'alpha_deg': np.degrees(alpha),
        'alpha_dot': alpha_dot,
        'F_trans_lift': F_trans_lift,
        'F_trans_drag': F_trans_drag,
        'F_rot': F_rot,
        'F_AM': F_AM,
        'F_lift': F_lift,
        'F_drag': F_drag,
        'C_L': C_L_arr,
        'C_D': C_D_arr,
        'P_aero': P_aero,
        'P_inertial': P_inertial,
        'P_total': P_total,
        'I_w': I_w,
        'm_w': m_w,
        'mech_span': mech_info['raw_span_deg'],
    }


def param_scan(geo_item, param_name, param_range, base_params):
    """单参数扫描"""
    results = []
    for val in param_range:
        p = base_params.copy()
        p[param_name] = val
        sim = simulate_cycle(geo_item, p, n_points=500)
        # 时均力（绝对值平均）
        avg_lift = np.mean(np.abs(sim['F_lift']))
        avg_drag = np.mean(np.abs(sim['F_drag']))
        peak_lift = np.max(np.abs(sim['F_lift']))
        peak_drag = np.max(np.abs(sim['F_drag']))
        results.append({
            'val': val,
            'avg_lift_N': avg_lift,
            'avg_drag_N': avg_drag,
            'peak_lift_N': peak_lift,
            'peak_drag_N': peak_drag,
        })
    return results


def plot_force_vs_phi(front_sim, back_sim, params, output_dir):
    """绘制力随翅膀转角 φ 的变化（向上为正，向下为负）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Aerodynamic Forces vs Wing Stroke Angle (φ)\nUpward Positive, Downward Negative',
                 fontsize=14, fontweight='bold')
    
    colors = {'front': '#1f77b4', 'back': '#2ca02c'}
    
    # Helper: split into downstroke (phi < 0) and upstroke (phi > 0)
    def split_stroke(sim):
        phi = sim['phi_deg']
        ds = phi <= 0   # 下拍：phi <= 0（向下为负）
        us = phi >= 0   # 上拍：phi >= 0（向上为正）
        return ds, us
    
    ds_f, us_f = split_stroke(front_sim)
    ds_b, us_b = split_stroke(back_sim)
    
    # Row 0: Lift vs phi
    ax = axes[0, 0]
    ax.plot(front_sim['phi_deg'][ds_f], front_sim['F_lift'][ds_f]*1000, 'o-', color=colors['front'], 
            markersize=2, lw=1.5, label='Front downstroke')
    ax.plot(front_sim['phi_deg'][us_f], front_sim['F_lift'][us_f]*1000, 's--', color=colors['front'], 
            markersize=2, lw=1.5, alpha=0.7, label='Front upstroke')
    ax.axvline(x=0, color='k', linestyle='-', lw=0.5)
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    ax.set_xlabel('Stroke angle φ (°)  [Up+, Down-]')
    ax.set_ylabel('Lift (mN)')
    ax.set_title('Front Wing - Lift vs φ')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(back_sim['phi_deg'][ds_b], back_sim['F_lift'][ds_b]*1000, 'o-', color=colors['back'], 
            markersize=2, lw=1.5, label='Back downstroke')
    ax.plot(back_sim['phi_deg'][us_b], back_sim['F_lift'][us_b]*1000, 's--', color=colors['back'], 
            markersize=2, lw=1.5, alpha=0.7, label='Back upstroke')
    ax.axvline(x=0, color='k', linestyle='-', lw=0.5)
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    ax.set_xlabel('Stroke angle φ (°)  [Up+, Down-]')
    ax.set_ylabel('Lift (mN)')
    ax.set_title('Back Wing - Lift vs φ')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Row 1: Drag vs phi (key result)
    ax = axes[1, 0]
    ax.fill_between(front_sim['phi_deg'][ds_f], 0, front_sim['F_trans_drag'][ds_f]*1000, 
                    color='#f44336', alpha=0.3, label='Front downstroke')
    ax.fill_between(front_sim['phi_deg'][us_f], 0, front_sim['F_trans_drag'][us_f]*1000, 
                    color='#f44336', alpha=0.15, label='Front upstroke')
    ax.plot(front_sim['phi_deg'][ds_f], front_sim['F_trans_drag'][ds_f]*1000, 
            color='#f44336', lw=1.5)
    ax.plot(front_sim['phi_deg'][us_f], front_sim['F_trans_drag'][us_f]*1000, 
            '--', color='#f44336', lw=1.5, alpha=0.7)
    ax.axvline(x=0, color='k', linestyle='-', lw=0.5)
    ax.set_xlabel('Stroke angle φ (°)  [Up+, Down-]')
    ax.set_ylabel('Drag (mN)')
    ax.set_title('Front Wing - Drag vs φ (Key Result)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.fill_between(back_sim['phi_deg'][ds_b], 0, back_sim['F_trans_drag'][ds_b]*1000, 
                    color='#9C27B0', alpha=0.3, label='Back downstroke')
    ax.fill_between(back_sim['phi_deg'][us_b], 0, back_sim['F_trans_drag'][us_b]*1000, 
                    color='#9C27B0', alpha=0.15, label='Back upstroke')
    ax.plot(back_sim['phi_deg'][ds_b], back_sim['F_trans_drag'][ds_b]*1000, 
            color='#9C27B0', lw=1.5)
    ax.plot(back_sim['phi_deg'][us_b], back_sim['F_trans_drag'][us_b]*1000, 
            '--', color='#9C27B0', lw=1.5, alpha=0.7)
    ax.axvline(x=0, color='k', linestyle='-', lw=0.5)
    ax.set_xlabel('Stroke angle φ (°)  [Up+, Down-]')
    ax.set_ylabel('Drag (mN)')
    ax.set_title('Back Wing - Drag vs φ (Key Result)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'force_vs_phi.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {figures_dir / "force_vs_phi.png"}')
    plt.close()


def plot_time_domain(front_sim, back_sim, params, output_dir):
    """绘制时间域力曲线"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'Butterfly Wing Aerodynamic Forces (f={params["f"]}Hz, α={params["alpha_deg"]}°)\n'
                 f'原始机构角度，不做缩放（负值=下拍，正值=上拍）',
                 fontsize=14, fontweight='bold')
    
    t = front_sim['t']
    T = t[-1]
    
    # 转换为毫秒
    t_ms = t * 1000
    
    colors = {'front': '#1f77b4', 'back': '#2ca02c'}
    
    # Row 0: Kinematics
    ax = axes[0, 0]
    ax.plot(t_ms, front_sim['phi_deg'], color=colors['front'], lw=2, label='Front')
    ax.plot(t_ms, back_sim['phi_deg'], '--', color=colors['back'], lw=2, label='Back')
    ax.set_ylabel('Stroke angle φ (°)')
    ax.set_title('Flapping Kinematics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    ax = axes[0, 1]
    ax.plot(t_ms, front_sim['alpha_deg'], color=colors['front'], lw=2, label='Front')
    ax.plot(t_ms, back_sim['alpha_deg'], '--', color=colors['back'], lw=2, label='Back')
    ax.set_ylabel('Angle of attack α (°)')
    ax.set_title('Angle of Attack')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    # Row 1: Lift components (Rotational = 0 due to fixed AoA, omitted from plot)
    ax = axes[1, 0]
    ax.stackplot(t_ms, 
                 front_sim['F_trans_lift']*1000,
                 front_sim['F_AM']*1000,
                 labels=['Translational', 'Added Mass'],
                 colors=['#4CAF50', '#2196F3'],
                 alpha=0.8)
    ax.set_ylabel('Lift (mN)')
    ax.set_title('Front Wing - Lift Components\n(Fixed AoA: Rotational = 0)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    ax = axes[1, 1]
    ax.stackplot(t_ms,
                 back_sim['F_trans_lift']*1000,
                 back_sim['F_AM']*1000,
                 labels=['Translational', 'Added Mass'],
                 colors=['#4CAF50', '#2196F3'],
                 alpha=0.8)
    ax.set_ylabel('Lift (mN)')
    ax.set_title('Back Wing - Lift Components\n(Fixed AoA: Rotational = 0)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    # Row 2: Drag + Total comparison
    ax = axes[2, 0]
    ax.fill_between(t_ms, 0, front_sim['F_trans_drag']*1000, color='#f44336', alpha=0.4, label='Front drag')
    ax.fill_between(t_ms, 0, back_sim['F_trans_drag']*1000, color='#9C27B0', alpha=0.4, label='Back drag')
    ax.plot(t_ms, front_sim['F_trans_drag']*1000, color='#f44336', lw=2)
    ax.plot(t_ms, back_sim['F_trans_drag']*1000, color='#9C27B0', lw=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Drag (mN)')
    ax.set_title('Translational Drag (Key Result)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    ax = axes[2, 1]
    ax.plot(t_ms, front_sim['F_lift']*1000, color=colors['front'], lw=2, label='Front total lift')
    ax.plot(t_ms, back_sim['F_lift']*1000, color=colors['back'], lw=2, label='Back total lift')
    ax.plot(t_ms, front_sim['F_drag']*1000, '--', color='#f44336', lw=1.5, label='Front total drag')
    ax.plot(t_ms, back_sim['F_drag']*1000, '--', color='#9C27B0', lw=1.5, label='Back total drag')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Force (mN)')
    ax.set_title('Total Lift vs Drag')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    plt.tight_layout()
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'force_time_domain.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {figures_dir / "force_time_domain.png"}')
    plt.close()


def plot_acceleration(front_sim, back_sim, params, output_dir):
    """绘制翅膀角速度与角加速度（机构运动学）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Wing Kinematic Acceleration (f={params["f"]}Hz, '
                 f'R=2.25, a={params.get("mech_a", 8.0)}, '
                 f'α={params["alpha_deg"]}°)',
                 fontsize=14, fontweight='bold')

    t = front_sim['t']
    T = t[-1]
    t_ms = t * 1000

    phi_dot = front_sim['phi_dot']
    zero_cross = np.where(np.diff(np.sign(phi_dot)))[0]
    t_div = float(t_ms[len(t_ms)//2]) if len(zero_cross) < 2 else float(t_ms[zero_cross[1]])

    colors = {'front': '#1f77b4', 'back': '#2ca02c'}

    for row_idx, ((ax_l, ax_r), suffix) in enumerate([
        ((axes[0, 0], axes[0, 1]), ''),
        ((axes[1, 0], axes[1, 1]), ''),
    ]):
        # Left: rad/s or rad/s²
        ax = axes[row_idx, 0]
        y_data = front_sim['phi_dot'] if row_idx == 0 else front_sim['phi_ddot']
        y_data_b = back_sim['phi_dot'] if row_idx == 0 else back_sim['phi_ddot']
        ylabel = 'Angular velocity (rad/s)' if row_idx == 0 else 'Angular acceleration (rad/s²)'
        title = 'Angular Velocity vs Time' if row_idx == 0 else 'Angular Acceleration vs Time'
        xlabel = 'Time (ms)' if row_idx == 1 else ''

        ax.plot(t_ms, y_data, color=colors['front'], lw=2, label='Front')
        ax.plot(t_ms, y_data_b, '--', color=colors['back'], lw=2, label='Back')
        ax.axvline(x=t_div, color='gray', linestyle=':', lw=1.5, alpha=0.7)
        ax.axhline(y=0, color='black', lw=0.5)
        ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, T * 1000)

        # Right: deg/s or scaled
        ax = axes[row_idx, 1]
        if row_idx == 0:
            y_data_r = np.degrees(y_data)
            y_data_br = np.degrees(y_data_b)
            ylabel_r = 'Angular velocity (°/s)'
            title_r = 'Angular Velocity (deg/s)'
        else:
            y_data_r = y_data * 1000
            y_data_br = y_data_b * 1000
            ylabel_r = 'Angular acceleration (×10³ rad/s²)'
            title_r = 'Angular Acceleration (scaled)'
        xlabel_r = 'Time (ms)' if row_idx == 1 else ''

        ax.plot(t_ms, y_data_r, color=colors['front'], lw=2, label='Front')
        ax.plot(t_ms, y_data_br, '--', color=colors['back'], lw=2, label='Back')
        ax.axvline(x=t_div, color='gray', linestyle=':', lw=1.5, alpha=0.7)
        ax.axhline(y=0, color='black', lw=0.5)
        ax.set_ylabel(ylabel_r)
        if xlabel_r:
            ax.set_xlabel(xlabel_r)
        ax.set_title(title_r)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, T * 1000)

    plt.tight_layout()
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'wing_acceleration.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {figures_dir / "wing_acceleration.png"}')
    plt.close()


def plot_power_time_domain(front_sim, back_sim, params, output_dir):
    """绘制功率时间域曲线（气动 + 惯性 + 总功率）"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Butterfly Wing Power Requirements (f={params["f"]}Hz, α={params["alpha_deg"]}°)\n'
                 f'P_aero = F_drag × |φ̇| × R × r̂₁    |    P_inertial = I_w × φ̈ × φ̇',
                 fontsize=14, fontweight='bold')
    
    t = front_sim['t']
    T = t[-1]
    t_ms = t * 1000
    
    colors = {'front': '#1f77b4', 'back': '#2ca02c'}
    
    # ---- 左上：单翅功率分解（Front） ----
    ax = axes[0, 0]
    ax.fill_between(t_ms, 0, front_sim['P_aero']*1000, alpha=0.3, color='#f44336', label='Aerodynamic')
    ax.fill_between(t_ms, 0, front_sim['P_inertial']*1000, alpha=0.3, color='#2196F3', label='Inertial')
    ax.plot(t_ms, front_sim['P_aero']*1000, '-', color='#f44336', lw=2, label='_nolegend_')
    ax.plot(t_ms, front_sim['P_inertial']*1000, '-', color='#2196F3', lw=2, label='_nolegend_')
    ax.plot(t_ms, front_sim['P_total']*1000, 'k-', lw=2.5, label='Total')
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    ax.set_ylabel('Power (mW)')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Front Wing - Power Components')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    # ---- 右上：单翅功率分解（Back） ----
    ax = axes[0, 1]
    ax.fill_between(t_ms, 0, back_sim['P_aero']*1000, alpha=0.3, color='#f44336')
    ax.fill_between(t_ms, 0, back_sim['P_inertial']*1000, alpha=0.3, color='#2196F3')
    ax.plot(t_ms, back_sim['P_aero']*1000, '-', color='#f44336', lw=2)
    ax.plot(t_ms, back_sim['P_inertial']*1000, '-', color='#2196F3', lw=2)
    ax.plot(t_ms, back_sim['P_total']*1000, 'k-', lw=2.5, label='Total')
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    ax.set_ylabel('Power (mW)')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Back Wing - Power Components')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    # ---- 左下：四翅总功率 ----
    P_aero_4w = 2 * (front_sim['P_aero'] + back_sim['P_aero'])
    P_inertial_4w = 2 * (front_sim['P_inertial'] + back_sim['P_inertial'])
    P_total_4w = P_aero_4w + P_inertial_4w
    
    ax = axes[1, 0]
    ax.fill_between(t_ms, 0, P_aero_4w*1000, alpha=0.3, color='#f44336', label='Aerodynamic (4w)')
    ax.fill_between(t_ms, 0, P_inertial_4w*1000, alpha=0.3, color='#2196F3', label='Inertial (4w)')
    ax.plot(t_ms, P_aero_4w*1000, '-', color='#f44336', lw=2)
    ax.plot(t_ms, P_inertial_4w*1000, '-', color='#2196F3', lw=2)
    ax.plot(t_ms, P_total_4w*1000, 'k-', lw=2.5, label='Total (4w)')
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    # 标注峰值
    peak_total_idx = np.argmax(np.abs(P_total_4w))
    ax.annotate(f'peak={P_total_4w[peak_total_idx]*1000:.1f} mW',
                xy=(t_ms[peak_total_idx], P_total_4w[peak_total_idx]*1000),
                xytext=(t_ms[peak_total_idx]+5, P_total_4w[peak_total_idx]*1000*0.8),
                fontsize=9, color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.set_ylabel('Power (mW)')
    ax.set_xlabel('Time (ms)')
    ax.set_title('Total Power (4 Wings)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    # ---- 右下：功率统计柱状图 ----
    ax = axes[1, 1]
    
    # 计算统计值
    stats = {
        'Front Aero': np.mean(front_sim['P_aero'])*1000,
        'Front Inertial': np.mean(np.abs(front_sim['P_inertial']))*1000,
        'Back Aero': np.mean(back_sim['P_aero'])*1000,
        'Back Inertial': np.mean(np.abs(back_sim['P_inertial']))*1000,
    }
    
    colors_bar = ['#f44336', '#2196F3', '#f44336', '#2196F3']
    bars = ax.bar(range(len(stats)), list(stats.values()), color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(list(stats.keys()), rotation=15, ha='right')
    ax.set_ylabel('Avg / Mean Abs Power (mW)')
    ax.set_title('Power Summary per Wing')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 标注数值
    for bar, val in zip(bars, stats.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stats.values())*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'power_time_domain.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {figures_dir / "power_time_domain.png"}')
    plt.close()


def plot_param_scans(geo, params, output_dir):
    """参数扫描图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Parameter Scan: Effect on Avg Drag & Lift (per wing)', fontsize=14, fontweight='bold')
    
    scan_configs = [
        ('f', np.linspace(10, 25, 30), 'Frequency (Hz)'),
        ('phi_down_deg', np.linspace(40, 100, 30), 'Downstroke amplitude (°)'),
        ('alpha_deg', np.linspace(20, 70, 30), 'Angle of attack (°)'),
    ]
    
    for col, (pname, prange, xlabel) in enumerate(scan_configs):
        # Front wing
        res_front = param_scan(geo['Front'], pname, prange, params)
        vals = [r['val'] for r in res_front]
        avg_drag_f = [r['avg_drag_N']*1000 for r in res_front]
        avg_lift_f = [r['avg_lift_N']*1000 for r in res_front]
        
        # Back wing
        res_back = param_scan(geo['Back'], pname, prange, params)
        avg_drag_b = [r['avg_drag_N']*1000 for r in res_back]
        avg_lift_b = [r['avg_lift_N']*1000 for r in res_back]
        
        # Drag plot
        ax = axes[0, col]
        ax.plot(vals, avg_drag_f, 'o-', color='#1f77b4', lw=2, markersize=4, label='Front')
        ax.plot(vals, avg_drag_b, 's-', color='#2ca02c', lw=2, markersize=4, label='Back')
        ax.set_ylabel('Avg Drag (mN)')
        ax.set_xlabel(xlabel)
        ax.set_title(f'Drag vs {xlabel}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Lift plot
        ax = axes[1, col]
        ax.plot(vals, avg_lift_f, 'o-', color='#1f77b4', lw=2, markersize=4, label='Front')
        ax.plot(vals, avg_lift_b, 's-', color='#2ca02c', lw=2, markersize=4, label='Back')
        ax.set_ylabel('Avg Lift (mN)')
        ax.set_xlabel(xlabel)
        ax.set_title(f'Lift vs {xlabel}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'param_scan.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {figures_dir / "param_scan.png"}')
    plt.close()


def generate_markdown_report(geo, params, front_sim, back_sim, output_dir):
    """生成 Markdown 报告"""
    weight = params['m_total'] * 9.81 * 1000  # mN
    avg_lift_4w = 2 * (np.mean(np.abs(front_sim['F_lift'])) + np.mean(np.abs(back_sim['F_lift']))) * 1000
    avg_drag_4w = 2 * (np.mean(np.abs(front_sim['F_drag'])) + np.mean(np.abs(back_sim['F_drag']))) * 1000
    peak_lift_4w = 2 * (np.max(np.abs(front_sim['F_lift'])) + np.max(np.abs(back_sim['F_lift']))) * 1000
    peak_drag_4w = 2 * (np.max(np.abs(front_sim['F_drag'])) + np.max(np.abs(back_sim['F_drag']))) * 1000

    # net lift (signed)
    net_lift_f = np.mean(front_sim['F_lift']) * 1000
    net_lift_b = np.mean(back_sim['F_lift']) * 1000
    net_lift_4w = 2 * (net_lift_f + net_lift_b)

    # 功率统计
    avg_aero_power_4w = 2 * (np.mean(front_sim['P_aero']) + np.mean(back_sim['P_aero'])) * 1000
    peak_aero_power_4w = 2 * np.max(np.array([np.max(front_sim['P_aero']), np.max(back_sim['P_aero'])])) * 1000
    avg_inertial_power_4w = 2 * (np.mean(np.abs(front_sim['P_inertial'])) + np.mean(np.abs(back_sim['P_inertial']))) * 1000
    peak_inertial_power_4w = 2 * np.max(np.array([np.max(np.abs(front_sim['P_inertial'])), np.max(np.abs(back_sim['P_inertial']))])) * 1000
    peak_total_power_4w = 2 * np.max(np.array([np.max(np.abs(front_sim['P_total'])), np.max(np.abs(back_sim['P_total']))])) * 1000
    
    # 单翅功率
    front_avg_aero = np.mean(front_sim['P_aero']) * 1000
    front_peak_aero = np.max(front_sim['P_aero']) * 1000
    front_avg_inertial = np.mean(np.abs(front_sim['P_inertial'])) * 1000
    front_peak_inertial = np.max(np.abs(front_sim['P_inertial'])) * 1000
    back_avg_aero = np.mean(back_sim['P_aero']) * 1000
    back_peak_aero = np.max(back_sim['P_aero']) * 1000
    back_avg_inertial = np.mean(np.abs(back_sim['P_inertial'])) * 1000
    back_peak_inertial = np.max(np.abs(back_sim['P_inertial'])) * 1000

    mech_a = params.get('mech_a', 8.0)

    md = f"""# 仿生蝴蝶翅膀空气动力学分析报告

> 生成日期: 2026-05-29  
> 运动学: 前置连杆机构 (mechanism.py)  
> 分析脚本: dynamic_analysis.py

---

## 1. 设计参数

### 1.1 飞行参数
| 参数 | 数值 | 说明 |
|------|------|------|
| 总质量 | 25 g | 机身+翅膀 |
| 四翅总质量 | 4 g | 单翅 1 g |
| 扑动频率 | {params['f']} Hz | 范围 10-25 Hz |
| 机构角度范围 | [{np.min(front_sim['phi_deg']):.1f}°, {np.max(front_sim['phi_deg']):.1f}°] | 原始输出，不做缩放 |
| 攻角 α | {params['alpha_deg']}° | 固定安装角（刚性连接，无翻转） |
| 空气密度 | 1.225 kg/m³ | 海平面标准值 |

### 1.2 前置连杆机构参数
| 参数 | 数值 | 说明 |
|------|------|------|
| a | {mech_a} | 主点圆心 x（可调 6-14，控制摆幅） |
| b | 6.97 | 主点圆心 y（固定） |
| R | 2.25 | 主点轨迹圆半径（固定） |
| c | 14 | 直线方程常数（固定） |
| l | 8 | 固定圆 x²+y²=l² 半径（固定） |
| 机构摆幅 | {front_sim['mech_span']:.1f}° | 原始机构输出的 ± 范围 |
| φ̇ 峰值 | {np.max(np.abs(front_sim['phi_dot'])):.1f} rad/s | 角速度峰值 |
| φ̈ 峰值 | {np.max(np.abs(front_sim['phi_ddot'])):.0f} rad/s² | 角加速度峰值（stroke reversal） |

## 2. 几何参数（由 DXF 实测）

| 翅膀 | 面积(mm²) | 展长(mm) | 平均弦长(mm) | 展弦比 | r̂₁ | r̂₂² |
|------|-----------|----------|--------------|--------|-----|-----|
| 前翅 Front | {geo['Front']['S_mm2']:.1f} | {geo['Front']['R_mm']:.1f} | {geo['Front']['c_avg']*1000:.1f} | {geo['Front']['AR']:.2f} | {geo['Front']['r1']:.4f} | {geo['Front']['r2_sq']:.4f} |
| 后翅 Back | {geo['Back']['S_mm2']:.1f} | {geo['Back']['R_mm']:.1f} | {geo['Back']['c_avg']*1000:.1f} | {geo['Back']['AR']:.2f} | {geo['Back']['r1']:.4f} | {geo['Back']['r2_sq']:.4f} |

## 3. 气动力计算结果

### 3.1 四翅总计
| 项目 | 数值 | 备注 |
|------|------|------|
| 重量 | **{weight:.1f} mN** | mg |
| 时均升力 | {avg_lift_4w:.1f} mN | 绝对值平均 |
| 净升力（符号平均） | {net_lift_4w:.1f} mN | 含上拍负升力抵消 |
| **时均阻力** | **{avg_drag_4w:.1f} mN** | **重点指标** |
| 峰值升力 | {peak_lift_4w:.1f} mN | AM+trans 综合峰值 |
| 峰值阻力 | {peak_drag_4w:.1f} mN | 拍动中期 |
| 时均升重比 | {avg_lift_4w/weight:.1f} | 绝对值平均 / 重量 |
| 净升重比 | {net_lift_4w/weight:.1f} | 净升力 / 重量 |

### 3.2 单翅明细
| 翅膀 | 时均升力(mN) | 净升力(mN) | 时均阻力(mN) | 峰值升力(mN) |
|------|-------------|-----------|-------------|-------------|
| Front | {np.mean(np.abs(front_sim['F_lift']))*1000:.1f} | {net_lift_f:.1f} | {np.mean(np.abs(front_sim['F_drag']))*1000:.1f} | {np.max(np.abs(front_sim['F_lift']))*1000:.1f} |
| Back | {np.mean(np.abs(back_sim['F_lift']))*1000:.1f} | {net_lift_b:.1f} | {np.mean(np.abs(back_sim['F_drag']))*1000:.1f} | {np.max(np.abs(back_sim['F_lift']))*1000:.1f} |

> **注**：准定常模型理论估算。实际飞行中三维效应、涡脱落、翅膀柔性变形使真实力降低 30-50%。
> 机构运动学含天然急回特性，角加速度峰值高于正弦假设。

## 4. 功率需求分析

### 4.1 功率计算公式
- **气动功率**：`P_aero = F_drag × |φ̇| × R × r̂₁` — 克服空气阻力的功率
- **惯性功率**：`P_inertial = I_w × φ̈ × φ̇` — 加速/减速翅膀的功率（周期平均≈0）
- **转动惯量**：`I_w = m_w × R² × r̂₂²`，单翅 `m_w = 1 g`

### 4.2 四翅总功率
| 项目 | 数值 | 说明 |
|------|------|------|
| 时均气动功率 | **{avg_aero_power_4w:.1f} mW** | 克服空气阻力 |
| 峰值气动功率 | **{peak_aero_power_4w:.1f} mW** | 拍动中期 |
| 平均惯性功率（绝对值）| {avg_inertial_power_4w:.1f} mW | 加速减速翅膀 |
| 峰值惯性功率 | **{peak_inertial_power_4w:.1f} mW** | stroke reversal |
| **峰值总功率** | **{peak_total_power_4w:.1f} mW** | **电机需提供的最大功率** |
| 时均总功率 | {avg_aero_power_4w:.1f} mW | 惯性功率周期平均≈0 |

### 4.3 单翅功率明细
| 翅膀 | 时均气动(mW) | 峰值气动(mW) | 平均惯性(mW) | 峰值惯性(mW) |
|------|-------------|-------------|-------------|-------------|
| Front | {front_avg_aero:.1f} | {front_peak_aero:.1f} | {front_avg_inertial:.1f} | {front_peak_inertial:.1f} |
| Back | {back_avg_aero:.1f} | {back_peak_aero:.1f} | {back_avg_inertial:.1f} | {back_peak_inertial:.1f} |

> **功率评估**：
> - 时均气动功率 **{avg_aero_power_4w:.1f} mW** 是持续悬停的主要能耗
> - 峰值总功率 **{peak_total_power_4w:.1f} mW** 决定电机和减速器的选型要求
> - 惯性功率在 stroke reversal 达到峰值，与角加速度同步
> - 若考虑三维效应和涡脱落损失，实际功率需求可能增加 30-50%

## 5. 图表

### 图 1：力随翅膀转角 φ 的变化（向上为正，向下为负）
![力转角曲线](../figures/force_vs_phi.png)

### 图 2：单周期时间域力曲线
![力时间曲线](../figures/force_time_domain.png)

### 图 3：翅膀角速度与角加速度
![翅膀加速度](../figures/wing_acceleration.png)

### 图 4：功率时间域曲线
![功率曲线](../figures/power_time_domain.png)

### 图 5：参数扫描结果
![参数扫描](../figures/param_scan.png)

### 图 6：安装角 α 扫描（净升力/阻力/升阻比）
![安装角扫描](../figures/alpha_scan.png)

### 图 7：机构运动学（轨迹、a 扫描、span vs a）
![机构运动学](../figures/mechanism_analysis.png)

### 图 8：安装角 α 扫描（净升力/阻力/升阻比）
![安装角扫描](../figures/alpha_scan.png)

## 7. α 扫描结果
| α | 净升力(mN) | 阻力(mN) | L/D | vs 重量 | 评价 |
|---|-----------|----------|-----|---------|------|
| 17° | 286 | 1,391 | **0.206** | 1.2x | 最佳效率（但升力低） |
| 35° | 564 | 3,009 | 0.187 | 2.3x | 安全 |
| **45°** | **719** | 4,140 | 0.174 | **2.9x** | **当前设计** |
| 55° | 848 | 5,262 | 0.161 | 3.5x | |
| 75° | **960** | 6,934 | 0.138 | **3.9x** | 最大净升力 |
| 85° | 927 | 7,274 | 0.127 | 3.8x | 阻力过大，效率降 |

## 8. 关键假设

1. 几何数据来源于 SolidWorks DXF 导出（草图局部坐标）。
2. **运动学由前置连杆机构生成**（mechanism.py，a={mech_a}，b=6.97，R=2.25，c=14，l=8）。曲柄一周 = 翅膀一拍。
3. 准定常模型：平动力基于瞬时速度（Dickinson/Sane 2002 分解）。
4. 固定攻角：α = {params['alpha_deg']}°（刚性连接），α̇=0，旋转力=0。
5. C_L 基于 φ̇ 方向：φ̇≤0 → C_L(+α)；φ̇>0 → C_L(-α)。
6. F_AM = -(ρπc²/4)·φ̈·R·r̂₁·sin(α)，阻力加速度。
7. 未考虑翅膀柔性变形、三维展向流动、涡干扰等效应。

---

## 附录：SolidWorks 轴线 DXF 导出步骤

当前轴线端点：`(-13.39, -84.95)` 和 `(41.55, 44.73)`，长度 140.84 mm。

1. 打开 `Wings.SLDPRT`
2. 特征树 → **草图102**（Axis/hinge line）→ 编辑草图
3. 确认只有两个圆（轴线端点），无构造线
4. 文件 → 另存为 → **DXF (*.dxf)**，版本 R2000+
5. 选项：**仅输出激活草图**，坐标系：草图坐标
6. 保存为 `WingsAxis.DXF`，覆盖原文件
7. 重新运行 `python analyze_dxf.py` 和 `python dynamic_analysis.py`
"""
    
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / '气动分析报告.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Saved Markdown report: {report_path}')
    return report_path


def main():
    print("=" * 70)
    print("BUTTERFLY WING DYNAMIC AERODYNAMIC ANALYSIS")
    print("=" * 70)
    
    geo = load_geometry()
    print(f"\n[Geometry loaded] Front: S={geo['Front']['S_mm2']:.1f} mm², Back: S={geo['Back']['S_mm2']:.1f} mm²")
    
    params = AERO_PARAMS.copy()
    
    # Time domain simulation
    print("\n[1/3] Running time-domain simulation...")
    front_sim = simulate_cycle(geo['Front'], params)
    back_sim = simulate_cycle(geo['Back'], params)
    
    # Compute stats
    for name, sim in [('Front', front_sim), ('Back', back_sim)]:
        avg_lift = np.mean(np.abs(sim['F_lift']))
        avg_drag = np.mean(np.abs(sim['F_drag']))
        peak_lift = np.max(np.abs(sim['F_lift']))
        peak_drag = np.max(np.abs(sim['F_drag']))
        avg_aero_power = np.mean(sim['P_aero'])
        peak_aero_power = np.max(sim['P_aero'])
        avg_inertial_power = np.mean(np.abs(sim['P_inertial']))
        peak_inertial_power = np.max(np.abs(sim['P_inertial']))
        peak_total_power = np.max(np.abs(sim['P_total']))
        print(f"  {name}: avg_lift={avg_lift*1000:.1f} mN, avg_drag={avg_drag*1000:.1f} mN, "
              f"peak_lift={peak_lift*1000:.1f} mN, peak_drag={peak_drag*1000:.1f} mN")
        print(f"         avg_aero_power={avg_aero_power*1000:.1f} mW, peak_aero_power={peak_aero_power*1000:.1f} mW")
        print(f"         peak_inertial_power={peak_inertial_power*1000:.1f} mW, peak_total_power={peak_total_power*1000:.1f} mW")
    
    # Plot force vs phi
    print("\n[2/6] Generating force vs phi plots...")
    plot_force_vs_phi(front_sim, back_sim, params, OUTPUT_DIR)
    
    # Plot time domain
    print("\n[3/6] Generating time-domain plots...")
    plot_time_domain(front_sim, back_sim, params, OUTPUT_DIR)
    
    # Plot acceleration
    print("\n[4/6] Generating wing acceleration plots...")
    plot_acceleration(front_sim, back_sim, params, OUTPUT_DIR)
    
    # Plot power
    print("\n[5/6] Generating power plots...")
    plot_power_time_domain(front_sim, back_sim, params, OUTPUT_DIR)
    
    # Param scan
    print("\n[6/6] Running parameter scans...")
    plot_param_scans(geo, params, OUTPUT_DIR)
    
    # Markdown report
    print("\n[6/6] Generating Markdown report...")
    report_path = generate_markdown_report(geo, params, front_sim, back_sim, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("DONE! Generated files:")
    print(f"  - {OUTPUT_DIR / 'figures' / 'force_vs_phi.png'}")
    print(f"  - {OUTPUT_DIR / 'figures' / 'force_time_domain.png'}")
    print(f"  - {OUTPUT_DIR / 'figures' / 'wing_acceleration.png'}")
    print(f"  - {OUTPUT_DIR / 'figures' / 'power_time_domain.png'}")
    print(f"  - {OUTPUT_DIR / 'figures' / 'param_scan.png'}")
    print(f"  - {report_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
