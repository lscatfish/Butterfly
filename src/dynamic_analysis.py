#!/usr/bin/env python3
"""
仿生蝴蝶翅膀动态气动力分析
功能：
1. 单周期时间域力曲线（平动升力/阻力、旋转力、附加质量力）
2. 参数扫描：频率、幅度、攻角对升/阻力的影响
3. 生成高清图表 + Word 报告
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import integrate
from pathlib import Path
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
MM_TO_M = 1e-3

# ==================== 用户设计参数 ====================
# 运动学约定：翅膀水平时 φ=0，向上为正(+)，向下为(-)
AERO_PARAMS = {
    "rho": 1.225,           # 空气密度 kg/m³
    "nu": 1.46e-5,          # 运动粘度 m²/s
    "m_total": 0.025,       # 总质量 25g
    "m_wing_total": 0.004,  # 四翅总质量 4g
    "f": 17.5,              # 典型频率 Hz (范围 15-20)
    "phi_down_deg": 80.0,   # 下拍最大角度（向下，取正值表示幅度）
    "phi_up_deg": 60.0,     # 上拍最大角度（向上，取正值表示幅度）
    "alpha_deg": 45.0,      # 攻角 °
    "C_r": 1.5,             # 旋转力系数 (Dickinson 1.0-2.0)
    "flip_ratio": 0.08,     # 翻转占半拍时间的比例（反转过渡区）
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
    
    返回 dict 包含 t, phase, phi, phi_dot, phi_ddot, alpha_deg, 以及各力分量
    """
    f = params['f']
    phi_down = np.deg2rad(params['phi_down_deg'])
    phi_up = np.deg2rad(params['phi_up_deg'])
    alpha0 = np.deg2rad(params['alpha_deg'])  # 转为弧度计算
    rho = params['rho']
    C_r = params['C_r']
    flip_ratio = params['flip_ratio']
    
    S = geo_item['S']
    R = geo_item['R']
    c_avg = geo_item['c_avg']
    r1 = geo_item['r1']
    r2_sq = geo_item['r2_sq']
    
    T = 1.0 / f
    t = np.linspace(0, T, n_points)
    dt = t[1] - t[0]
    
    # ========== 非对称运动学模型 ==========
    # 约定：翅膀水平时 φ=0，向上为正(+)，向下为负(-)
    # 下拍最大 -80°，上拍最大 +60°
    # 下拍和上拍时间分配保证 phi_dot 在过渡点连续
    t_d = T * phi_down / (phi_down + phi_up)   # 下拍时间
    t_u = T * phi_up / (phi_down + phi_up)     # 上拍时间
    
    phi = np.zeros_like(t)
    phi_dot = np.zeros_like(t)
    phi_ddot = np.zeros_like(t)
    
    for i, ti in enumerate(t):
        if ti < t_d:
            # 下拍：phi 从 0 到 -phi_down 再到 0
            tau = ti / t_d
            phi[i] = -phi_down * np.sin(np.pi * tau)
            phi_dot[i] = -phi_down * (np.pi / t_d) * np.cos(np.pi * tau)
            phi_ddot[i] = phi_down * (np.pi / t_d)**2 * np.sin(np.pi * tau)
        else:
            # 上拍：phi 从 0 到 +phi_up 再到 0
            tau = (ti - t_d) / t_u
            phi[i] = phi_up * np.sin(np.pi * tau)
            phi_dot[i] = phi_up * (np.pi / t_u) * np.cos(np.pi * tau)
            phi_ddot[i] = -phi_up * (np.pi / t_u)**2 * np.sin(np.pi * tau)
    
    # ========== 固定攻角模型（翅膀安装角固定，不能翻转） ==========
    # 翅膀安装角始终为 +alpha0
    # 下拍时：来流相对向上，有效攻角 = +alpha0 → C_L > 0，升力向上
    # 上拍时：来流相对向下，有效攻角 = -alpha0 → C_L < 0，升力向下
    alpha = alpha0 * np.ones_like(t)  # 物理安装角始终 +alpha0
    alpha_dot = np.zeros_like(t)       # 无翻转，alpha_dot = 0
    
    # 计算各力分量
    C_L_arr = np.zeros_like(t)
    C_D_arr = np.zeros_like(t)
    for i in range(len(t)):
        if 0 <= t[i] < t_d:
            # 下拍：有效攻角 +alpha0
            C_L_arr[i], C_D_arr[i] = cl_cd(np.degrees(alpha0))
        else:
            # 上拍：来流方向反转，有效攻角 -alpha0
            C_L_arr[i], C_D_arr[i] = cl_cd(-np.degrees(alpha0))
    
    # 平动分量（与 phi_dot^2 成正比）
    # 注意：C_L 的符号已经包含了方向信息
    # 下拍 C_L > 0 → 升力向上；上拍 C_L < 0 → 升力向下
    F_trans_lift = 0.5 * rho * C_L_arr * (phi_dot * R)**2 * S * r2_sq
    F_trans_drag = 0.5 * rho * C_D_arr * (phi_dot * R)**2 * S * r2_sq
    
    # 旋转力 = 0（因为 alpha_dot = 0，翅膀不能扭转）
    F_rot = np.zeros_like(t)
    
    # 附加质量力（方向由 phi_ddot 和 sin(alpha0) 共同决定）
    # sin(alpha0) = sin(+45°) = +0.707 始终为正
    F_AM = (rho * np.pi * c_avg**2 / 4.0) * phi_ddot * R * r1 * np.sin(alpha0)
    
    # 总力
    F_lift = F_trans_lift + F_rot + F_AM
    F_drag = np.abs(F_trans_drag)  # 阻力始终为正（与运动方向相反）
    
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
    fig.suptitle(f'Butterfly Wing Aerodynamic Forces (f={params["f"]}Hz, φ=-{params["phi_down_deg"]}°~+{params["phi_up_deg"]}°, α={params["alpha_deg"]}°)',
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
    
    # Row 1: Lift components
    ax = axes[1, 0]
    ax.stackplot(t_ms, 
                 front_sim['F_trans_lift']*1000,
                 front_sim['F_rot']*1000,
                 front_sim['F_AM']*1000,
                 labels=['Translational', 'Rotational', 'Added Mass'],
                 colors=['#4CAF50', '#FF9800', '#2196F3'],
                 alpha=0.8)
    ax.set_ylabel('Lift (mN)')
    ax.set_title('Front Wing - Lift Components')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    ax = axes[1, 1]
    ax.stackplot(t_ms,
                 back_sim['F_trans_lift']*1000,
                 back_sim['F_rot']*1000,
                 back_sim['F_AM']*1000,
                 labels=['Translational', 'Rotational', 'Added Mass'],
                 colors=['#4CAF50', '#FF9800', '#2196F3'],
                 alpha=0.8)
    ax.set_ylabel('Lift (mN)')
    ax.set_title('Back Wing - Lift Components')
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
    """绘制翅膀角速度与角加速度"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Wing Kinematic Acceleration (f={params["f"]}Hz, φ=-{params["phi_down_deg"]}°~+{params["phi_up_deg"]}°)',
                 fontsize=14, fontweight='bold')
    
    t = front_sim['t']
    T = t[-1]
    t_ms = t * 1000
    
    # 下拍/上拍分界时间
    phi_down = np.deg2rad(params['phi_down_deg'])
    phi_up = np.deg2rad(params['phi_up_deg'])
    t_d = T * phi_down / (phi_down + phi_up)
    
    colors = {'front': '#1f77b4', 'back': '#2ca02c'}
    
    # Row 0: Angular velocity
    ax = axes[0, 0]
    ax.plot(t_ms, front_sim['phi_dot'], color=colors['front'], lw=2, label='Front')
    ax.plot(t_ms, back_sim['phi_dot'], '--', color=colors['back'], lw=2, label='Back')
    ax.axvline(x=t_d*1000, color='gray', linestyle=':', lw=1.5, alpha=0.7)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_ylabel('Angular velocity (rad/s)')
    ax.set_title('Angular Velocity vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    # 标注下拍/上拍
    ax.text(t_d*1000/2, ax.get_ylim()[1]*0.85, 'Downstroke', ha='center', fontsize=9, color='#666')
    ax.text(t_d*1000 + (T-t_d)*1000/2, ax.get_ylim()[1]*0.85, 'Upstroke', ha='center', fontsize=9, color='#666')
    
    ax = axes[0, 1]
    ax.plot(t_ms, np.degrees(front_sim['phi_dot']), color=colors['front'], lw=2, label='Front')
    ax.plot(t_ms, np.degrees(back_sim['phi_dot']), '--', color=colors['back'], lw=2, label='Back')
    ax.axvline(x=t_d*1000, color='gray', linestyle=':', lw=1.5, alpha=0.7)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_ylabel('Angular velocity (°/s)')
    ax.set_title('Angular Velocity (deg/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    # Row 1: Angular acceleration
    ax = axes[1, 0]
    ax.plot(t_ms, front_sim['phi_ddot'], color=colors['front'], lw=2, label='Front')
    ax.plot(t_ms, back_sim['phi_ddot'], '--', color=colors['back'], lw=2, label='Back')
    ax.axvline(x=t_d*1000, color='gray', linestyle=':', lw=1.5, alpha=0.7)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Angular acceleration (rad/s^2)')
    ax.set_title('Angular Acceleration vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    ax = axes[1, 1]
    ax.plot(t_ms, front_sim['phi_ddot']*1000, color=colors['front'], lw=2, label='Front')
    ax.plot(t_ms, back_sim['phi_ddot']*1000, '--', color=colors['back'], lw=2, label='Back')
    ax.axvline(x=t_d*1000, color='gray', linestyle=':', lw=1.5, alpha=0.7)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Angular acceleration (x10^3 rad/s^2)')
    ax.set_title('Angular Acceleration (scaled)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T*1000)
    
    plt.tight_layout()
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'wing_acceleration.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {figures_dir / "wing_acceleration.png"}')
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
    
    md = f"""# 仿生蝴蝶翅膀空气动力学分析报告

> 生成日期: 2026-05-29  
> 分析脚本: dynamic_analysis.py

---

## 1. 设计参数

| 参数 | 数值 | 说明 |
|------|------|------|
| 总质量 | 25 g | 机身+翅膀 |
| 四翅总质量 | 4 g | 单翅 1 g |
| 扑动频率 | {params['f']} Hz | 典型值，范围 10-25 Hz |
| 下拍幅度 | {params['phi_down_deg']}° | 向下最大角度 |
| 上拍幅度 | {params['phi_up_deg']}° | 向上最大角度 |
| 攻角 | {params['alpha_deg']}° | 典型值 |
| 空气密度 | 1.225 kg/m³ | 海平面 |
| 旋转力系数 C_r | {params['C_r']} | 文献范围 1.0-2.0 |

## 2. 几何参数（由 DXF 实测）

| 翅膀 | 面积(mm²) | 展长(mm) | 平均弦长(mm) | 展弦比 | r₁ | r₂² |
|------|-----------|----------|--------------|--------|-----|-----|
| 前翅 Front | {geo['Front']['S_mm2']:.1f} | {geo['Front']['R_mm']:.1f} | {geo['Front']['c_avg']*1000:.1f} | {geo['Front']['AR']:.2f} | {geo['Front']['r1']:.4f} | {geo['Front']['r2_sq']:.4f} |
| 后翅 Back | {geo['Back']['S_mm2']:.1f} | {geo['Back']['R_mm']:.1f} | {geo['Back']['c_avg']*1000:.1f} | {geo['Back']['AR']:.2f} | {geo['Back']['r1']:.4f} | {geo['Back']['r2_sq']:.4f} |

## 3. 气动力计算结果（四翅总计）

| 项目 | 数值 | 备注 |
|------|------|------|
| 重量 | **{weight:.1f} mN** | mg |
| 时均升力（四翅） | {avg_lift_4w:.1f} mN | 理论估算 |
| **时均阻力（四翅）** | **{avg_drag_4w:.1f} mN** | **重点指标** |
| 峰值升力（四翅） | {peak_lift_4w:.1f} mN | 拍动中期 |
| 峰值阻力（四翅） | {peak_drag_4w:.1f} mN | 拍动中期 |
| 升重比 | {avg_lift_4w/weight:.1f} | 理论值 |

> **注**：以上力值基于准定常模型估算。实际飞行中，三维效应、涡脱落、翅膀柔性变形等因素会使真实力降低 30-50%。

## 4. 图表

### 图 1：力随翅膀转角 φ 的变化（向上为正，向下为负）
![力转角曲线](../figures/force_vs_phi.png)

### 图 2：单周期时间域力曲线
![力时间曲线](../figures/force_time_domain.png)

### 图 3：翅膀角速度与角加速度
![翅膀加速度](../figures/wing_acceleration.png)

### 图 4：参数扫描结果
![参数扫描](../figures/param_scan.png)

## 5. 关键假设

1. 几何数据来源于 SolidWorks DXF 导出（草图局部坐标），已废弃早期的 VBA 全局坐标 CSV。
2. 准定常模型：平动力基于瞬时速度，旋转力基于攻角变化率。
3. 攻角在 stroke reversal 处平滑翻转（过渡区占周期 {params['flip_ratio']*100:.0f}%）。
4. 旋转力系数 C_r = {params['C_r']}，取自 Dickinson 文献范围 1.0-2.0。
5. 附加质量力采用二维薄翼近似。
6. 未考虑翅膀柔性变形、三维展向流动、涡干扰等效应。

---

## 附录：SolidWorks 轴线 DXF 重新导出步骤

若需修正轴线位置（当前轴线端点为 `(-13.39, -84.95)` 和 `(41.55, 44.73)`）：

1. 打开 `Wings.SLDPRT`
2. 在特征树中找到 **草图102**（Axis/hinge line）
3. 右键草图102 → **编辑草图**
4. 确认草图中只有两个圆（表示轴线端点），无其他构造线
5. 如需调整：删除现有圆，在翅膀根部重新绘制两个圆（直径约 5mm）
6. 文件 → 另存为 → 选择格式 **DXF (*.dxf)**
7. 在 DXF 选项中：
   - 版本：R2000 或更高
   - **仅输出激活草图**（关键！）
   - 坐标系：草图坐标
8. 保存为 `WingsAxis.DXF`，覆盖原文件
9. 重新运行 `python analyze_dxf.py` 和 `python dynamic_analysis.py`
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
        print(f"  {name}: avg_lift={avg_lift*1000:.1f} mN, avg_drag={avg_drag*1000:.1f} mN, "
              f"peak_lift={peak_lift*1000:.1f} mN, peak_drag={peak_drag*1000:.1f} mN")
    
    # Plot force vs phi
    print("\n[2/5] Generating force vs phi plots...")
    plot_force_vs_phi(front_sim, back_sim, params, OUTPUT_DIR)
    
    # Plot time domain
    print("\n[3/5] Generating time-domain plots...")
    plot_time_domain(front_sim, back_sim, params, OUTPUT_DIR)
    
    # Plot acceleration
    print("\n[4/5] Generating wing acceleration plots...")
    plot_acceleration(front_sim, back_sim, params, OUTPUT_DIR)
    
    # Param scan
    print("\n[5/5] Running parameter scans...")
    plot_param_scans(geo, params, OUTPUT_DIR)
    
    # Markdown report
    print("\n[5/5] Generating Markdown report...")
    report_path = generate_markdown_report(geo, params, front_sim, back_sim, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("DONE! Generated files:")
    print(f"  - {OUTPUT_DIR / 'figures' / 'force_vs_phi.png'}")
    print(f"  - {OUTPUT_DIR / 'figures' / 'force_time_domain.png'}")
    print(f"  - {OUTPUT_DIR / 'figures' / 'wing_acceleration.png'}")
    print(f"  - {OUTPUT_DIR / 'figures' / 'param_scan.png'}")
    print(f"  - {report_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
