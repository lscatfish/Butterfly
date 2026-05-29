#!/usr/bin/env python3
"""
仿生蝴蝶翅膀几何参数提取与气动力分解分析
读取 DXF (WingFront / WingBack / WingsAxis) → 面积、弦长分布、面积矩
准定常力分解：平动升力/阻力、旋转力、附加质量力、时均值
"""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy import integrate
import matplotlib.pyplot as plt
from pathlib import Path
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    "f": 17.5,              # 典型频率 Hz (范围 15-20)
    "phi_down_deg": 80.0,   # 下拍最大角度 °（向下）
    "phi_up_deg": 60.0,     # 上拍最大角度 °（向上）
    "alpha_deg": 45.0,      # 攻角 °
    "C_r": 1.5,             # 旋转力系数 (Dickinson 1.0-2.0)
    "flip_ratio": 0.08,     # 翻转占半拍比例 (过渡区占半拍 8%)
}


def parse_dxf(filepath):
    """解析 DXF，提取 SPLINE / LINE / CIRCLE 实体"""
    with open(filepath) as f:
        content = f.read()
    
    entities = []
    # SPLINE
    parts = content.split('\n  0\nSPLINE\n')
    for part in parts[1:]:
        lines_b = part.strip().split('\n')
        idx = 0
        data = {}
        ctrl_pts = []
        fit_pts = []
        knots = []
        while idx < len(lines_b) - 1:
            code = lines_b[idx].strip()
            if code == '0':
                break
            val = lines_b[idx + 1].strip()
            if code == '71':
                data['degree'] = int(float(val))
            elif code == '40':
                knots.append(float(val))
            elif code == '10' and idx + 3 < len(lines_b) and lines_b[idx + 2].strip() == '20':
                ctrl_pts.append([float(val), float(lines_b[idx + 3].strip())])
                idx += 2
            elif code == '11' and idx + 3 < len(lines_b) and lines_b[idx + 2].strip() == '21':
                fit_pts.append([float(val), float(lines_b[idx + 3].strip())])
                idx += 2
            idx += 2
        entities.append({
            'type': 'SPLINE',
            'degree': data.get('degree', 3),
            'ctrl_pts': np.array(ctrl_pts),
            'fit_pts': np.array(fit_pts),
            'knots': np.array(knots),
        })
    
    # LINE
    line_blocks = content.split('\n  0\nLINE\n')
    for block in line_blocks[1:]:
        d = {}
        lines_b = block.strip().split('\n')
        idx = 0
        while idx < len(lines_b) - 1:
            code = lines_b[idx].strip()
            if code == '0':
                break
            val = lines_b[idx + 1].strip()
            if code in ('10', '20', '30', '11', '21', '31'):
                d[code] = float(val)
            idx += 2
        if '10' in d and '20' in d and '11' in d and '21' in d:
            entities.append({
                'type': 'LINE',
                'start': np.array([d['10'], d['20']]),
                'end': np.array([d['11'], d['21']]),
            })
    
    # CIRCLE
    circle_blocks = content.split('\n  0\nCIRCLE\n')
    for block in circle_blocks[1:]:
        d = {}
        lines_b = block.strip().split('\n')
        idx = 0
        while idx < len(lines_b) - 1:
            code = lines_b[idx].strip()
            if code == '0':
                break
            val = lines_b[idx + 1].strip()
            if code in ('10', '20', '30', '40'):
                d[code] = float(val)
            idx += 2
        if '10' in d and '20' in d:
            entities.append({
                'type': 'CIRCLE',
                'center': np.array([d['10'], d['20']]),
                'radius': d.get('40', 0),
            })
    
    return entities


def sample_spline(entity, n=200):
    ctrl = entity['ctrl_pts']
    knots = entity['knots']
    k = entity['degree']
    if len(ctrl) == 0:
        return np.zeros((0, 2))
    fit = entity['fit_pts']
    if len(fit) >= 4:
        return fit
    if len(knots) < len(ctrl) + k + 1:
        knots = np.linspace(0, 1, len(ctrl) + k + 1)
    try:
        t = np.array(knots)
        c = ctrl
        if len(t) >= len(c) + k + 1:
            t = t[:len(c) + k + 1]
        if t[-1] > t[0]:
            t_norm = (t - t[0]) / (t[-1] - t[0])
        else:
            t_norm = t
        spl_x = BSpline(t_norm, c[:, 0], k)
        spl_y = BSpline(t_norm, c[:, 1], k)
        u = np.linspace(t_norm[k], t_norm[-k - 1], n)
        return np.column_stack([spl_x(u), spl_y(u)])
    except Exception as e:
        print(f"BSpline warn: {e}")
        return ctrl


def get_segment_points(entity, n=2000):
    if entity['type'] == 'SPLINE':
        return sample_spline(entity, n)
    elif entity['type'] == 'LINE':
        return np.vstack([entity['start'], entity['end']])
    return np.zeros((0, 2))


def connect_entities(entities):
    segs = []
    for e in entities:
        if e['type'] in ('SPLINE', 'LINE'):
            pts = get_segment_points(e, n=2000 if e['type'] == 'SPLINE' else 2)
            if len(pts) > 0:
                segs.append(pts)
    if len(segs) == 0:
        return None
    used = [False] * len(segs)
    ordered = [segs[0]]
    used[0] = True
    current_end = segs[0][-1]
    for _ in range(len(segs) - 1):
        best_idx, best_dist, best_reverse = -1, float('inf'), False
        for i in range(len(segs)):
            if used[i]:
                continue
            s, e = segs[i][0], segs[i][-1]
            d = np.linalg.norm(s - current_end)
            if d < best_dist:
                best_dist, best_idx, best_reverse = d, i, False
            d = np.linalg.norm(e - current_end)
            if d < best_dist:
                best_dist, best_idx, best_reverse = d, i, True
        if best_idx < 0 or best_dist > 1.0:
            break
        used[best_idx] = True
        ordered.append(segs[best_idx][::-1] if best_reverse else segs[best_idx])
        current_end = segs[best_idx][0] if best_reverse else segs[best_idx][-1]
    return np.vstack(ordered)


def read_axis_from_dxf(filepath):
    entities = parse_dxf(filepath)
    circles = [e for e in entities if e['type'] == 'CIRCLE']
    if len(circles) < 2:
        raise ValueError(f"轴线 DXF 需要 2 个圆，只找到 {len(circles)} 个")
    p0 = circles[0]['center']
    p1 = circles[1]['center']
    direction = p1 - p0
    dir_len = np.linalg.norm(direction)
    unit_dir = direction / dir_len if dir_len > 0 else np.array([1.0, 0.0])
    unit_perp = np.array([-unit_dir[1], unit_dir[0]])
    return {'p0': p0, 'p1': p1, 'unit_dir': unit_dir, 'unit_perp': unit_perp}


def calculate_wing(pts, axis, wing_name, n_bins=200):
    """计算翅膀几何参数（使用带符号 y，不取 abs）"""
    def shoelace(poly):
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    
    S_poly = shoelace(pts)
    
    # 局部坐标
    v = pts - axis['p0']
    r = v @ axis['unit_dir']    # 弦向（沿转轴）
    y = v @ axis['unit_perp']   # 展向（垂直转轴，带符号）
    
    # 展长 = y 的总跨度（单侧或双侧）
    y_min, y_max = y.min(), y.max()
    R = y_max - y_min
    
    if R < 1e-6:
        print(f"[{wing_name}] 展长过小")
        return None
    
    # 沿展向分条算弦长 c(y)
    bins = np.linspace(y_min, y_max, n_bins + 1)
    y_centers = 0.5 * (bins[:-1] + bins[1:])
    chords = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (y >= bins[i]) & (y < bins[i + 1])
        if np.any(mask):
            chords[i] = r[mask].max() - r[mask].min()
    
    S = integrate.simpson(chords, y_centers)
    if S <= 0:
        S = S_poly
    
    c_avg = S / R
    AR = R**2 / S if S > 0 else 0
    
    # 归一化
    c_hat = chords / c_avg if c_avg > 0 else np.zeros_like(chords)
    y_hat = (y_centers - y_min) / R
    
    # 面积矩（注意：使用原始 y 坐标，不取 abs）
    valid = chords > 1e-10
    if np.any(valid):
        # r̂ = (y - y_min) / R，在 [0,1] 区间
        r1 = integrate.simpson(y_hat[valid] * c_hat[valid], y_hat[valid])
        r2_sq = integrate.simpson(y_hat[valid]**2 * c_hat[valid], y_hat[valid])
    else:
        r1 = r2_sq = 0.0
    
    # 质心展向位置
    y_cg = integrate.simpson(y_centers * chords, y_centers) / S if S > 0 else 0.0
    y_cg_hat = (y_cg - y_min) / R
    
    return {
        'name': wing_name,
        'S': S * MM_TO_M**2,
        'S_mm2': S,
        'R': R * MM_TO_M,
        'R_mm': R,
        'c_avg': c_avg * MM_TO_M,
        'c_avg_mm': c_avg,
        'AR': AR,
        'y_cg': y_cg * MM_TO_M,
        'y_cg_hat': y_cg_hat,
        'r1': r1,
        'r2_sq': r2_sq,
        'y_hat': y_hat,
        'c_hat': c_hat,
        'y_centers_mm': y_centers,
        'chords_mm': chords,
        'pts': pts,
    }


def aerodynamic_estimate(props, params):
    rho = params['rho']
    f = params['f']
    phi_down = np.deg2rad(params['phi_down_deg'])
    phi_up = np.deg2rad(params['phi_up_deg'])
    alpha_deg = params['alpha_deg']
    m_total = params['m_total']
    m_wing_total = params['m_wing_total']
    nu = params['nu']
    # 运动学参数（使用下拍幅度 80° 计算峰值，下拍时间更长）
    Phi_max = phi_down  # 峰值出现在下拍
    phi_dot_max = 2 * np.pi * f * Phi_max
    phi_ddot_max = (2 * np.pi * f)**2 * Phi_max
    alpha_rad = np.deg2rad(alpha_deg)
    
    # 固定攻角：无翻转，alpha_dot = 0
    alpha_dot_max = 0.0
    
    # 升阻力系数
    C_L = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
    C_D = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))
    
    results = {}
    for p in props:
        name = p['name']
        S = p['S']
        R = p['R']
        c_avg = p['c_avg']
        r1 = p['r1']
        r2_sq = p['r2_sq']
        
        # 单翅质量（四翅均分）
        m_w = m_wing_total / 4.0
        
        # 翼尖速度
        u_tip_max = phi_dot_max * R
        u_mean = (2.0 / np.pi) * u_tip_max
        
        # 雷诺数、减缩频率
        Re = u_mean * c_avg / nu
        omega = 2 * np.pi * f
        k = omega * c_avg / (2 * u_mean) if u_mean > 0 else 0
        
        # ====== 力分解（峰值）======
        # 1. 平动升力（拍动中期，|φ̇|最大）
        F_trans_lift = 0.5 * rho * C_L * (phi_dot_max * R)**2 * S * r2_sq
        # 2. 平动阻力（拍动中期）
        F_trans_drag = 0.5 * rho * C_D * (phi_dot_max * R)**2 * S * r2_sq
        
        # 3. 旋转力：固定攻角，alpha_dot = 0，F_rot = 0
        F_rot = 0.0
        
        # 4. 附加质量力（反转点，|φ̈|最大）
        # F_AM = (ρ π c² / 4) φ̈ R r̂₁ sinα
        F_AM = (rho * np.pi * c_avg**2 / 4.0) * phi_ddot_max * R * r1 * np.sin(alpha_rad)
        
        # 5. 总峰值力（平动主导，旋转力=0）
        F_peak_total = F_trans_lift + abs(F_AM)
        
        # ====== 时间平均力 ======
        # 平动力：cos² 平均 = 1/2
        F_avg_trans_lift = F_trans_lift / 2.0
        F_avg_trans_drag = F_trans_drag / 2.0
        # 旋转力：固定攻角，为零
        F_avg_rot = 0.0
        # 附加质量力：周期对称，时均≈0
        F_avg_AM = 0.0
        # 总时均升力（静态估算仅含平动分量；动态分析含上拍负升力）
        F_avg_lift = F_avg_trans_lift + F_avg_rot + F_avg_AM
        
        # 转动惯量 & 功率
        I_w = m_w * R**2 * r2_sq
        KE = 0.5 * I_w * phi_dot_max**2
        P_inertial = 4 * KE * f
        
        results[name] = {
            'C_L': C_L,
            'C_D': C_D,
            'phi_dot_max_rad_s': phi_dot_max,
            'phi_ddot_max_rad_s2': phi_ddot_max,
            'alpha_dot_max_rad_s': alpha_dot_max,  # = 0 (fixed AoA)
            'u_tip_max_m_s': u_tip_max,
            'u_mean_m_s': u_mean,
            'Re': Re,
            'k': k,
            'm_w_g': m_w * 1000,
            'I_w_g_mm2': I_w * 1e9,
            
            # 峰值力 (N)
            'F_trans_lift_peak_N': F_trans_lift,
            'F_trans_drag_peak_N': F_trans_drag,
            'F_rot_peak_N': F_rot,
            'F_AM_peak_N': F_AM,
            'F_peak_total_N': F_peak_total,
            
            # 峰值力 (mN)
            'F_trans_lift_peak_mN': F_trans_lift * 1000,
            'F_trans_drag_peak_mN': F_trans_drag * 1000,
            'F_rot_peak_mN': F_rot * 1000,
            'F_AM_peak_mN': F_AM * 1000,
            'F_peak_total_mN': F_peak_total * 1000,
            
            # 时均力 (mN)
            'F_avg_trans_lift_mN': F_avg_trans_lift * 1000,
            'F_avg_trans_drag_mN': F_avg_trans_drag * 1000,
            'F_avg_rot_mN': F_avg_rot * 1000,
            'F_avg_lift_mN': F_avg_lift * 1000,
            
            'KE_mJ': KE * 1000,
            'P_inertial_mW': P_inertial * 1000,
        }
    
    # 四翅总力（2 front + 2 back）
    weight = m_total * 9.81
    total_peak_lift = 2 * sum(results[n]['F_peak_total_N'] for n in results)
    total_avg_lift = 2 * sum(results[n]['F_avg_lift_mN'] for n in results)
    total_avg_drag = 2 * sum(results[n]['F_avg_trans_drag_mN'] for n in results)
    
    results['total'] = {
        'weight_N': weight,
        'weight_mN': weight * 1000,
        'total_peak_lift_N': total_peak_lift,
        'total_peak_lift_mN': total_peak_lift * 1000,
        'total_avg_lift_mN': total_avg_lift,
        'total_avg_drag_mN': total_avg_drag,
        'avg_lift_to_weight': total_avg_lift / (weight * 1000) if weight > 0 else 0,
    }
    return results


def plot_results(axis, front_prop, back_prop, output_dir):
    fig = plt.figure(figsize=(18, 12))
    
    # 1. DXF Global
    ax1 = fig.add_subplot(2, 3, 1)
    for prop, color, lbl in [(front_prop, 'blue', 'Front'), (back_prop, 'green', 'Back')]:
        if prop:
            pts = prop['pts']
            ax1.fill(pts[:, 0], pts[:, 1], alpha=0.3, color=color)
            ax1.plot(pts[:, 0], pts[:, 1], color=color, lw=1.5, label=lbl)
    ax1.plot([axis['p0'][0], axis['p1'][0]], [axis['p0'][1], axis['p1'][1]], 'r--', lw=2, label='Axis')
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('Wing Planform (DXF)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Local coords
    ax2 = fig.add_subplot(2, 3, 2)
    for prop, color in [(front_prop, 'blue'), (back_prop, 'green')]:
        if prop:
            pts = prop['pts']
            v = pts - axis['p0']
            r = v @ axis['unit_dir']
            y = v @ axis['unit_perp']
            ax2.plot(y, r, color=color, lw=1.5, label=prop['name'])
    ax2.axvline(x=0, color='r', linestyle='--', lw=1)
    ax2.set_xlabel('y: spanwise (mm)')
    ax2.set_ylabel('r: chordwise (mm)')
    ax2.set_title('Local Coordinates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Chord distribution
    ax3 = fig.add_subplot(2, 3, 3)
    for prop, color in [(front_prop, 'blue'), (back_prop, 'green')]:
        if prop:
            ax3.plot(prop['y_hat'], prop['c_hat'], color=color, lw=2, label=prop['name'])
    ax3.axhline(y=1.0, color='k', linestyle='--', lw=1, alpha=0.5)
    ax3.set_xlabel('y_hat = (y-y_min)/R')
    ax3.set_ylabel('c_hat = c/c_avg')
    ax3.set_title('Chord Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Local filled
    ax4 = fig.add_subplot(2, 3, 4)
    for prop, color in [(front_prop, 'blue'), (back_prop, 'green')]:
        if prop:
            pts = prop['pts']
            v = pts - axis['p0']
            r = v @ axis['unit_dir']
            y = v @ axis['unit_perp']
            ax4.fill(y, r, alpha=0.3, color=color)
            ax4.plot(y, r, color=color, lw=0.5)
    ax4.axvline(x=0, color='r', linestyle='--', lw=1)
    ax4.set_xlabel('y (mm)')
    ax4.set_ylabel('r (mm)')
    ax4.set_title('Local Filled')
    ax4.grid(True, alpha=0.3)
    
    # 5. 参数表
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    rows = []
    for p in [front_prop, back_prop]:
        if p:
            rows.append([p['name'], f"{p['S_mm2']:.1f}", f"{p['R_mm']:.1f}",
                         f"{p['c_avg_mm']:.1f}", f"{p['AR']:.2f}", f"{p['r2_sq']:.4f}"])
    if rows:
        tbl = ax5.table(cellText=rows,
                        colLabels=['Wing', 'S(mm2)', 'R(mm)', 'c_avg(mm)', 'AR', 'r2_sq'],
                        loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)
    
    # 6. 力分解柱状图（峰值）
    ax6 = fig.add_subplot(2, 3, 6)
    names = []
    lift_vals = []
    drag_vals = []
    rot_vals = []
    am_vals = []
    for p in [front_prop, back_prop]:
        if p:
            names.append(p['name'])
    
    # 需要 aero 结果才能画图，这里简化留空或后续补
    ax6.text(0.5, 0.5, 'See console output\nfor force breakdown', ha='center', va='center', fontsize=12)
    ax6.set_title('Force Breakdown (console)')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wing_analysis.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {output_dir / "wing_analysis.png"}')
    plt.close()


def print_force_breakdown(aero):
    """在控制台清晰输出力分解（单翅 + 四翅总计）"""
    print("\n" + "=" * 70)
    print("FORCE BREAKDOWN")
    print("=" * 70)
    
    total = aero['total']
    print(f"\n[四翅总计]  Weight        = {total['weight_mN']:.2f} mN")
    print(f"[四翅总计]  Peak Lift     = {total['total_peak_lift_mN']:.2f} mN")
    print(f"[四翅总计]  Avg Lift      = {total['total_avg_lift_mN']:.2f} mN")
    print(f"[四翅总计]  Avg Drag      = {total['total_avg_drag_mN']:.2f} mN  <-- 重点")
    print(f"[四翅总计]  Lift/Weight   = {total['avg_lift_to_weight']:.2f}")
    
    for name in ['Front', 'Back']:
        if name not in aero:
            continue
        r = aero[name]
        print(f"\n--- {name.upper()} WING (单翅) ---")
        print(f"  Mass        = {r['m_w_g']:.2f} g")
        print(f"  Re          = {r['Re']:.0f}")
        print(f"  k           = {r['k']:.3f}")
        print(f"  u_tip_max   = {r['u_tip_max_m_s']:.2f} m/s")
        print(f"  phi_dot_max = {r['phi_dot_max_rad_s']:.1f} rad/s")
        print(f"  alpha_dot   = {r['alpha_dot_max_rad_s']:.1f} rad/s  (fixed AoA)")
        
        print(f"\n  [峰值力]")
        print(f"    Translational Lift = {r['F_trans_lift_peak_mN']:>10.2f} mN")
        print(f"    Translational Drag = {r['F_trans_drag_peak_mN']:>10.2f} mN  <--")
        print(f"    Rotational Force   = {r['F_rot_peak_mN']:>10.2f} mN  (fixed AoA, zero)")
        print(f"    Added Mass Force   = {r['F_AM_peak_mN']:>10.2f} mN")
        print(f"    Peak Total         = {r['F_peak_total_mN']:>10.2f} mN")
        
        print(f"\n  [时均力]")
        print(f"    Avg Lift (trans)   = {r['F_avg_trans_lift_mN']:>10.2f} mN")
        print(f"    Avg Drag (trans)   = {r['F_avg_trans_drag_mN']:>10.2f} mN  <--")
        print(f"    Avg Rotational     = {r['F_avg_rot_mN']:>10.2f} mN  (fixed AoA, zero)")
        print(f"    Avg Total Lift     = {r['F_avg_lift_mN']:>10.2f} mN")
        
        print(f"\n  [功率]")
        print(f"    Inertial Power     = {r['P_inertial_mW']:>10.2f} mW")
    
    print("\n" + "=" * 70)
    print("NOTE: 以上力值基于准定常模型估算，实际飞行中受三维效应、")
    print("      涡脱落、柔性变形等因素影响，真实力可能低 30-50%。")
    print("=" * 70)


def main():
    print("=" * 70)
    print("BUTTERFLY WING AERODYNAMIC ANALYSIS")
    print("=" * 70)
    
    axis = read_axis_from_dxf(DATA_DIR / 'WingsAxis.DXF')
    print(f"\n[Axis]  p0=({axis['p0'][0]:.2f},{axis['p0'][1]:.2f})  p1=({axis['p1'][0]:.2f},{axis['p1'][1]:.2f})  "
          f"L={np.linalg.norm(axis['p1']-axis['p0']):.2f} mm")
    
    front_entities = parse_dxf(DATA_DIR / 'WingFront.DXF')
    back_entities = parse_dxf(DATA_DIR / 'WingBack.DXF')
    
    front_pts = connect_entities(front_entities)
    back_pts = connect_entities(back_entities)
    
    front_prop = calculate_wing(front_pts, axis, "Front") if front_pts is not None else None
    back_prop = calculate_wing(back_pts, axis, "Back") if back_pts is not None else None
    
    print("\n" + "=" * 70)
    print("GEOMETRY")
    print("=" * 70)
    all_props = []
    for p in [front_prop, back_prop]:
        if p is None:
            continue
        all_props.append(p)
        print(f"\n{p['name']}: S={p['S_mm2']:.1f} mm2  R={p['R_mm']:.1f} mm  "
              f"c_avg={p['c_avg_mm']:.1f} mm  AR={p['AR']:.2f}  r2_sq={p['r2_sq']:.4f}")
    
    aero = aerodynamic_estimate(all_props, AERO_PARAMS)
    print_force_breakdown(aero)
    
    # 保存 JSON
    save_data = {
        'params': AERO_PARAMS,
        'axis': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in axis.items()},
        'geometry': [{k: float(v) if isinstance(v, (np.floating, float)) else v
                      for k, v in p.items() if k not in ('y_hat', 'c_hat', 'y_centers_mm', 'chords_mm', 'pts')}
                     for p in all_props],
        'aerodynamics': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                            for kk, vv in v.items()}
                        for k, v in aero.items()},
    }
    # 保存 JSON 到 data/
    json_path = DATA_DIR / 'wing_analysis_results.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved JSON: {json_path}")
    
    # 保存弦长分布到 output/tables/
    if front_prop and back_prop:
        chord_df = pd.DataFrame({
            'y_hat': front_prop['y_hat'],
            'c_hat_front': front_prop['c_hat'],
            'c_hat_back': np.interp(front_prop['y_hat'], back_prop['y_hat'], back_prop['c_hat'], left=0, right=0),
            'y_center_mm': front_prop['y_centers_mm'],
            'chord_front_mm': front_prop['chords_mm'],
            'chord_back_mm': np.interp(front_prop['y_centers_mm'], back_prop['y_centers_mm'], back_prop['chords_mm'], left=0, right=0),
        })
        tables_dir = OUTPUT_DIR / 'tables'
        tables_dir.mkdir(parents=True, exist_ok=True)
        chord_df.to_csv(tables_dir / 'chord_distribution.csv', index=False)
        print(f"Saved output/tables/chord_distribution.csv")
    
    # 保存图表到 output/figures/
    figures_dir = OUTPUT_DIR / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_results(axis, front_prop, back_prop, figures_dir)
    print("\nDone!")


if __name__ == '__main__':
    main()
