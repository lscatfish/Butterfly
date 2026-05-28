#!/usr/bin/env python3
"""
仿生蝴蝶翅膀几何参数提取与气动估算
读取 SolidWorks 导出的轮廓 CSV，计算面积、弦长分布、面积矩等
"""

import numpy as np
import pandas as pd
from scipy import integrate
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import json

# 修复中文乱码：优先用系统自带的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
DATA_DIR = Path(__file__).parent
MM_TO_M = 1e-3

AERO_PARAMS = {
    "rho": 1.225,
    "nu": 1.46e-5,
    "m_total": 0.0216,
    "f": 10.0,
    "Phi_max_deg": 80.0,
    "alpha_deg": 45.0,
}


def read_axis(filepath):
    df = pd.read_csv(filepath)
    p0 = df[df['Type'] == 0][['X', 'Y', 'Z']].values[0] * MM_TO_M
    p1 = df[df['Type'] == 1][['X', 'Y', 'Z']].values[0] * MM_TO_M
    direction = df[df['Type'] == 2][['X', 'Y', 'Z']].values[0] * MM_TO_M
    
    dir_len = np.linalg.norm(direction)
    unit_dir = direction / dir_len if dir_len > 0 else np.array([1.0, 0.0, 0.0])
    
    # XY 平面内垂直方向
    unit_perp = np.array([-unit_dir[1], unit_dir[0], 0.0])
    perp_len = np.linalg.norm(unit_perp)
    if perp_len > 0:
        unit_perp = unit_perp / perp_len
    else:
        unit_perp = np.array([0.0, 1.0, 0.0])
    
    return {'p0': p0, 'p1': p1, 'unit_dir': unit_dir, 'unit_perp': unit_perp}


def segment_area(seg_pts):
    """计算单条 segment 与其两端点到原点连线围成的三角形面积（用于近似）"""
    if len(seg_pts) < 2:
        return 0.0
    # 将 segment 端点与原点连成多边形
    poly = np.vstack([seg_pts, seg_pts[0]])
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))


def read_wing_by_segments(filepath):
    """按 segment 读取翅膀，返回每个 segment 的点列表"""
    df = pd.read_csv(filepath)
    segments = []
    for seg_id in sorted(df['SegmentIndex'].unique()):
        sub = df[df['SegmentIndex'] == seg_id][['X', 'Y', 'Z']].values * MM_TO_M
        if len(sub) > 1:
            # 去重
            dists = np.linalg.norm(np.diff(sub, axis=0), axis=1)
            mask = np.concatenate(([True], dists > 1e-8))
            sub = sub[mask]
            segments.append(sub)
    return segments


def transform_to_local(pts, axis):
    """局部坐标：r 沿转轴，y 垂直于转轴"""
    v = pts - axis['p0']
    r = v @ axis['unit_dir']
    y = v @ axis['unit_perp']
    return r, y


def polygon_from_segments(segments, axis):
    """
    将多个 segment 尝试拼接成连续多边形。
    策略：按端点距离排序，把首尾相近的 segment 连接起来。
    """
    if not segments:
        return None
    
    # 提取每条 segment 的起点和终点
    endpoints = []
    for i, seg in enumerate(segments):
        endpoints.append((i, 'start', seg[0]))
        endpoints.append((i, 'end', seg[-1]))
    
    # 贪心拼接：从第一条开始，找最近的未使用 segment
    used = [False] * len(segments)
    ordered_pts = [segments[0]]
    used[0] = True
    current_end = segments[0][-1]
    reverse_flag = [False] * len(segments)
    
    for _ in range(len(segments) - 1):
        best_dist = float('inf')
        best_idx = -1
        best_reverse = False
        
        for i in range(len(segments)):
            if used[i]:
                continue
            # 尝试 seg[i] 的起点接当前尾
            d = np.linalg.norm(segments[i][0] - current_end)
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_reverse = False
            # 尝试 seg[i] 的终点接当前尾（反向）
            d = np.linalg.norm(segments[i][-1] - current_end)
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_reverse = True
        
        if best_idx < 0:
            break
        
        used[best_idx] = True
        reverse_flag[best_idx] = best_reverse
        
        if best_reverse:
            ordered_pts.append(segments[best_idx][::-1])
            current_end = segments[best_idx][0]
        else:
            ordered_pts.append(segments[best_idx])
            current_end = segments[best_idx][-1]
    
    # 合并所有点
    all_pts = np.vstack(ordered_pts)
    
    # 检查首尾是否闭合
    gap = np.linalg.norm(all_pts[0] - all_pts[-1])
    
    return all_pts, gap


def calculate_wing_properties(segments, axis, wing_name, n_bins=200):
    """计算翅膀几何参数"""
    
    # 1. 尝试拼接多边形
    poly_result = polygon_from_segments(segments, axis)
    if poly_result is None:
        print(f"[{wing_name}] 无法构建多边形")
        return None
    
    poly_pts, gap = poly_result
    
    # 2. 多边形面积（鞋带公式）
    def shoelace(poly):
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    
    S_poly = shoelace(poly_pts)
    
    # 如果首尾 gap 太大，说明轮廓不闭合，用凸包面积作为 fallback
    if gap > 1e-3:  # 1mm
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(poly_pts[:, :2])
            S_hull = hull.volume  # 2D 凸包面积
        except:
            S_hull = 0.0
        print(f"[{wing_name}] 警告：轮廓未闭合 (gap={gap*1000:.2f} mm)，使用凸包面积")
        S = S_hull
    else:
        S = S_poly
    
    # 3. 所有点转换到局部坐标
    all_pts_flat = np.vstack(segments)
    r_all, y_all = transform_to_local(all_pts_flat, axis)
    
    # 4. 展长：展向 = 垂直于转轴的距离 |y|
    y_abs = np.abs(y_all)
    R = y_abs.max() - y_abs.min()
    
    if R < 1e-6:
        print(f"[{wing_name}] 警告：展长过小")
        return None
    
    # 5. 弦长分布：沿展向（|y|）分条，计算每条内 r 的跨度
    y_min, y_max = y_abs.min(), y_abs.max()
    bins = np.linspace(y_min, y_max, n_bins + 1)
    y_centers = 0.5 * (bins[:-1] + bins[1:])
    dy = bins[1] - bins[0]
    
    chords = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (y_abs >= bins[i]) & (y_abs < bins[i + 1])
        if np.any(mask):
            chords[i] = r_all[mask].max() - r_all[mask].min()
        else:
            chords[i] = 0.0
    
    # 6. 用弦长分布重新积分面积（作为校验）
    S_chord = integrate.simpson(chords, y_centers)
    
    # 取两种方法的合理值
    if S_poly > 0 and gap < 1e-3:
        S_final = S_poly
    else:
        S_final = max(S_chord, S_poly)
    
    if S_final <= 0:
        print(f"[{wing_name}] 警告：面积计算失败")
        return None
    
    # 7. 平均弦长、展弦比
    c_avg = S_final / R
    AR = R**2 / S_final
    
    # 8. 归一化分布
    c_hat = chords / c_avg if c_avg > 0 else np.zeros_like(chords)
    y_hat = (y_centers - y_min) / R
    
    # 9. 面积矩
    valid = chords > 1e-10
    if np.any(valid):
        r1 = integrate.simpson(y_hat[valid] * c_hat[valid], y_hat[valid])
        r2_sq = integrate.simpson(y_hat[valid]**2 * c_hat[valid], y_hat[valid])
    else:
        r1 = r2_sq = 0.0
    
    # 10. 质心
    y_cg = integrate.simpson(y_centers * chords, y_centers) / S_final if S_final > 0 else 0.0
    y_cg_hat = (y_cg - y_min) / R
    
    return {
        'name': wing_name,
        'S': S_final,
        'S_poly': S_poly,
        'S_chord': S_chord,
        'gap_mm': gap * 1000,
        'R': R,
        'c_avg': c_avg,
        'AR': AR,
        'y_cg': y_cg,
        'y_cg_hat': y_cg_hat,
        'r1': r1,
        'r2_sq': r2_sq,
        'y_hat': y_hat,
        'c_hat': c_hat,
        'y_centers': y_centers,
        'chords': chords,
        'poly_pts': poly_pts,
        'segments': segments,
    }


def aerodynamic_estimate(props, params):
    rho = params['rho']
    f = params['f']
    Phi_max = np.deg2rad(params['Phi_max_deg'])
    alpha_deg = params['alpha_deg']
    m_total = params['m_total']
    nu = params['nu']
    
    results = {}
    for p in props:
        name = p['name']
        S = p['S']
        R = p['R']
        c_avg = p['c_avg']
        r2_sq = p['r2_sq']
        
        phi_dot_max = 2 * np.pi * f * Phi_max
        C_L = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
        C_D = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))
        
        F_L_peak = 0.5 * rho * C_L * (phi_dot_max * R)**2 * S * r2_sq
        u_tip_max = phi_dot_max * R
        u_mean = (2.0 / np.pi) * u_tip_max
        Re = u_mean * c_avg / nu
        omega = 2 * np.pi * f
        k = omega * c_avg / (2 * u_mean) if u_mean > 0 else 0
        
        m_w = m_total * 0.05
        I_w = m_w * R**2 * r2_sq
        KE = 0.5 * I_w * phi_dot_max**2
        P_inertial = 4 * KE * f
        
        results[name] = {
            'C_L': C_L, 'C_D': C_D,
            'phi_dot_max_rad_s': phi_dot_max,
            'u_tip_max_m_s': u_tip_max,
            'u_mean_m_s': u_mean,
            'Re': Re, 'k': k,
            'F_L_peak_N': F_L_peak,
            'F_L_peak_mN': F_L_peak * 1000,
            'm_w_kg': m_w,
            'I_w_kg_m2': I_w,
            'KE_mJ': KE * 1000,
            'P_inertial_mW': P_inertial * 1000,
        }
    
    total_lift = sum(results[n]['F_L_peak_N'] for n in results)
    weight = m_total * 9.81
    results['total'] = {
        'F_L_peak_total_N': total_lift,
        'F_L_peak_total_mN': total_lift * 1000,
        'weight_N': weight,
        'weight_mN': weight * 1000,
        'lift_to_weight': total_lift / weight if weight > 0 else 0,
    }
    return results


def plot_results(axis, front_prop, back_prop, output_dir):
    fig = plt.figure(figsize=(16, 10))
    
    # ---------- 全局坐标 ----------
    ax1 = fig.add_subplot(2, 3, 1)
    for prop, color, label in [(front_prop, 'blue', 'Front'), (back_prop, 'green', 'Back')]:
        if prop:
            for seg in prop['segments']:
                ax1.plot(seg[:, 0]*1000, seg[:, 1]*1000, color=color, lw=1)
    t = np.linspace(-5000, 15000, 2)
    ax1.plot((axis['p0'][0] + t*axis['unit_dir'][0])*1000,
             (axis['p0'][1] + t*axis['unit_dir'][1])*1000, 'r--', lw=2, label='Axis')
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('Global XY')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ---------- 拼接多边形 ----------
    ax2 = fig.add_subplot(2, 3, 2)
    for prop, color in [(front_prop, 'blue'), (back_prop, 'green')]:
        if prop and 'poly_pts' in prop:
            poly = prop['poly_pts']
            ax2.fill(poly[:, 0]*1000, poly[:, 1]*1000, alpha=0.3, color=color)
            ax2.plot(poly[:, 0]*1000, poly[:, 1]*1000, color=color, lw=1)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Polygon (auto-connected)')
    ax2.grid(True, alpha=0.3)
    
    # ---------- 局部坐标系 (y=展向, r=弦向) ----------
    ax3 = fig.add_subplot(2, 3, 3)
    for prop, color, name in [(front_prop, 'blue', 'Front'), (back_prop, 'green', 'Back')]:
        if prop:
            for seg in prop['segments']:
                r, y = transform_to_local(seg, axis)
                ax3.plot(y*1000, r*1000, color=color, lw=1)
    ax3.axvline(x=0, color='r', linestyle='--', lw=1)
    ax3.set_xlabel('y: spanwise (mm)')
    ax3.set_ylabel('r: chordwise (mm)')
    ax3.set_title('Local coords (y=span, r=chord)')
    ax3.grid(True, alpha=0.3)
    
    # ---------- 弦长分布 ----------
    ax4 = fig.add_subplot(2, 3, 4)
    for prop, color in [(front_prop, 'blue'), (back_prop, 'green')]:
        if prop:
            ax4.plot(prop['y_hat'], prop['c_hat'], color=color, lw=2, label=prop['name'])
    ax4.axhline(y=1.0, color='k', linestyle='--', lw=1, alpha=0.5)
    ax4.set_xlabel('y_hat = y/R')
    ax4.set_ylabel('c_hat = c/c_avg')
    ax4.set_title('Chord distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ---------- 局部填充 ----------
    ax5 = fig.add_subplot(2, 3, 5)
    for prop, color in [(front_prop, 'blue'), (back_prop, 'green')]:
        if prop:
            all_seg = np.vstack(prop['segments'])
            r, y = transform_to_local(all_seg, axis)
            ax5.fill(y*1000, r*1000, alpha=0.3, color=color)
            ax5.plot(y*1000, r*1000, color=color, lw=0.5)
    ax5.axvline(x=0, color='r', linestyle='--', lw=1)
    ax5.set_xlabel('y: spanwise (mm)')
    ax5.set_ylabel('r: chordwise (mm)')
    ax5.set_title('Local filled')
    ax5.grid(True, alpha=0.3)
    
    # ---------- 参数表 ----------
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    rows = []
    for p in [front_prop, back_prop]:
        if p:
            rows.append([
                p['name'],
                f"{p['S']*1e6:.1f}",
                f"{p['R']*1000:.1f}",
                f"{p['c_avg']*1000:.1f}",
                f"{p['AR']:.2f}",
                f"{p['r2_sq']:.4f}",
                f"{p['gap_mm']:.3f}",
            ])
    if rows:
        tbl = ax6.table(
            cellText=rows,
            colLabels=['Wing', 'S (mm2)', 'R (mm)', 'c_avg (mm)', 'AR', 'r2_sq', 'gap (mm)'],
            loc='center', cellLoc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wing_geometry_analysis.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {output_dir / 'wing_geometry_analysis.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("Butterfly Wing Geometry & Aerodynamics")
    print("=" * 60)
    
    axis = read_axis(DATA_DIR / "wing_axis.csv")
    print(f"\n[Axis]")
    print(f"  p0: ({axis['p0'][0]*1000:.2f}, {axis['p0'][1]*1000:.2f}) mm")
    print(f"  dir: ({axis['unit_dir'][0]:.4f}, {axis['unit_dir'][1]:.4f})")
    
    front_segs = read_wing_by_segments(DATA_DIR / "wing_front.csv")
    back_segs = read_wing_by_segments(DATA_DIR / "wing_back.csv")
    
    print(f"\n[Segments]")
    print(f"  Front: {len(front_segs)} segments, {sum(len(s) for s in front_segs)} pts")
    for i, s in enumerate(front_segs):
        print(f"    seg {i}: {len(s)} pts, len={np.linalg.norm(s[0]-s[-1])*1000:.1f} mm")
    print(f"  Back:  {len(back_segs)} segments, {sum(len(s) for s in back_segs)} pts")
    for i, s in enumerate(back_segs):
        print(f"    seg {i}: {len(s)} pts, len={np.linalg.norm(s[0]-s[-1])*1000:.1f} mm")
    
    front_prop = calculate_wing_properties(front_segs, axis, "Front")
    back_prop = calculate_wing_properties(back_segs, axis, "Back")
    
    print("\n" + "=" * 60)
    print("Geometry Results")
    print("=" * 60)
    
    all_props = []
    for p in [front_prop, back_prop]:
        if p is None:
            continue
        all_props.append(p)
        print(f"\n--- {p['name']} ---")
        print(f"  Area S       = {p['S']*1e6:>12.3f} mm2  (poly={p['S_poly']*1e6:.1f}, chord={p['S_chord']*1e6:.1f})")
        print(f"  Span R       = {p['R']*1000:>12.2f} mm")
        print(f"  Avg chord    = {p['c_avg']*1000:>12.2f} mm")
        print(f"  Aspect AR    = {p['AR']:>12.3f}")
        print(f"  CG span      = {p['y_cg']*1000:>12.2f} mm  (y_hat={p['y_cg_hat']:.4f})")
        print(f"  r1           = {p['r1']:>12.4f}")
        print(f"  r2_sq        = {p['r2_sq']:>12.4f}")
        print(f"  Polygon gap  = {p['gap_mm']:>12.3f} mm")
    
    if not all_props:
        print("ERROR: No valid wing data!")
        return
    
    total_S = sum(p['S'] for p in all_props)
    print(f"\n  Total area   = {total_S*1e6:.3f} mm2")
    
    # 气动估算
    print("\n" + "=" * 60)
    print("Aerodynamic Estimate")
    print("=" * 60)
    aero = aerodynamic_estimate(all_props, AERO_PARAMS)
    
    for name, res in aero.items():
        if name == 'total':
            continue
        print(f"\n--- {name} ---")
        print(f"  C_L        = {res['C_L']:.3f}")
        print(f"  C_D        = {res['C_D']:.3f}")
        print(f"  Re         = {res['Re']:.0f}")
        print(f"  k          = {res['k']:.3f}")
        print(f"  F_L_peak   = {res['F_L_peak_mN']:.3f} mN")
        print(f"  I_wing     = {res['I_w_kg_m2']*1e9:.3f} kg.mm2")
        print(f"  P_inertial = {res['P_inertial_mW']:.2f} mW")
    
    total = aero['total']
    print(f"\n--- Total ---")
    print(f"  Lift (2 wings) = {total['F_L_peak_total_mN']:.3f} mN")
    print(f"  Weight         = {total['weight_mN']:.3f} mN")
    print(f"  L/W            = {total['lift_to_weight']:.2f}")
    
    # 保存
    save_data = {
        'axis': {'p0_mm': axis['p0'].tolist(), 'unit_dir': axis['unit_dir'].tolist()},
        'geometry': [{k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                      for k, v in p.items() if k not in ['poly_pts', 'segments', 'y_hat', 'c_hat', 'y_centers', 'chords']}
                     for p in all_props],
        'aerodynamics': {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv) 
                            for kk, vv in v.items()} 
                        for k, v in aero.items()},
    }
    json_path = DATA_DIR / 'wing_analysis_results.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved JSON: {json_path}")
    
    plot_results(axis, front_prop, back_prop, DATA_DIR)
    print("\nDone!")


if __name__ == '__main__':
    main()
