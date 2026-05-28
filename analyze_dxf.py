#!/usr/bin/env python3
"""
读取 WingFront.DXF / WingBack.DXF / WingsAxis.DXF
计算翅膀几何参数与气动估算
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


def parse_dxf(filepath):
    """解析 DXF，提取 SPLINE / LINE / CIRCLE 实体"""
    with open(filepath) as f:
        lines = [l.strip() for l in f.readlines()]
    
    entities = []
    i = 0
    while i < len(lines):
        if lines[i] == '0':
            etype = lines[i + 1] if i + 1 < len(lines) else ''
            if etype in ('SPLINE', 'LINE', 'CIRCLE'):
                j = i + 2
                data = {}
                ctrl_pts = []
                fit_pts = []
                knots = []
                
                while j < len(lines) and lines[j] != '0':
                    if j + 1 >= len(lines):
                        j += 1
                        continue
                    code = lines[j]
                    val = lines[j + 1]
                    j += 2
                    
                    if code == '71':
                        data['degree'] = int(float(val))
                    elif code == '72':
                        data['n_fit'] = int(float(val))
                    elif code == '73':
                        data['n_ctrl'] = int(float(val))
                    elif code == '40':
                        knots.append(float(val))
                    elif code == '10':
                        if j + 1 < len(lines) and lines[j] == '20':
                            ctrl_pts.append([float(val), float(lines[j + 1])])
                            j += 2
                    elif code == '11':
                        if j + 1 < len(lines) and lines[j] == '21':
                            fit_pts.append([float(val), float(lines[j + 1])])
                            j += 2
                    elif code in ('10', '20', '30', '11', '21', '31', '40'):
                        # 单个值存储
                        try:
                            data[code] = float(val)
                        except:
                            data[code] = val
                
                if etype == 'SPLINE':
                    entities.append({
                        'type': 'SPLINE',
                        'degree': data.get('degree', 3),
                        'ctrl_pts': np.array(ctrl_pts),
                        'fit_pts': np.array(fit_pts),
                        'knots': np.array(knots),
                    })
                elif etype == 'LINE':
                    # 需要重新读取起终点（上面的通用解析可能不够准，简化处理）
                    pass
            i += 1
        i += 1
    
    # 重新扫描 LINE 和 CIRCLE（更可靠）
    with open(filepath) as f:
        content = f.read()
    
    # LINE: 10,20,30 = start; 11,21,31 = end
    line_blocks = content.split('\n  0\nLINE\n')
    for block in line_blocks[1:]:
        d = {}
        lines_b = block.strip().split('\n')
        idx = 0
        while idx < len(lines_b) - 1:
            code = lines_b[idx].strip()
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
    
    # CIRCLE: 10,20,30 = center; 40 = radius
    circle_blocks = content.split('\n  0\nCIRCLE\n')
    for block in circle_blocks[1:]:
        d = {}
        lines_b = block.strip().split('\n')
        idx = 0
        while idx < len(lines_b) - 1:
            code = lines_b[idx].strip()
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
    """对 DXF SPLINE 采样 n 个点"""
    ctrl = entity['ctrl_pts']
    knots = entity['knots']
    k = entity['degree']
    
    if len(ctrl) == 0:
        return np.zeros((0, 2))
    
    # 如果拟合点存在且足够，直接用拟合点
    fit = entity['fit_pts']
    if len(fit) >= 4:
        return fit
    
    # 否则用 scipy BSpline 重建
    if len(knots) < len(ctrl) + k + 1:
        # 节点向量不够，用均匀节点
        knots = np.linspace(0, 1, len(ctrl) + k + 1)
    
    # scipy BSpline 要求节点向量有 k+1 个重复端点
    # DXF 的节点向量通常已经是规范的
    try:
        t = np.array(knots)
        c = ctrl
        # 确保维度匹配
        if len(t) >= len(c) + k + 1:
            # 取前 len(c)+k+1 个节点
            t = t[:len(c) + k + 1]
        
        # 归一化到 [0,1]
        if t[-1] > t[0]:
            t_norm = (t - t[0]) / (t[-1] - t[0])
        else:
            t_norm = t
        
        # 分别对 x, y 做 B-spline
        spl_x = BSpline(t_norm, c[:, 0], k)
        spl_y = BSpline(t_norm, c[:, 1], k)
        
        u = np.linspace(t_norm[k], t_norm[-k-1], n)
        x = spl_x(u)
        y = spl_y(u)
        return np.column_stack([x, y])
    except Exception as e:
        print(f"BSpline failed: {e}, fallback to ctrl_pts")
        return ctrl


def get_segment_points(entity, n=200):
    """获取任意 entity 的点序列"""
    if entity['type'] == 'SPLINE':
        return sample_spline(entity, n)
    elif entity['type'] == 'LINE':
        return np.vstack([entity['start'], entity['end']])
    return np.zeros((0, 2))


def segment_start_end(pts):
    if len(pts) == 0:
        return None, None
    return pts[0], pts[-1]


def connect_entities(entities):
    """按端点距离贪心连接所有实体"""
    # 只处理 SPLINE 和 LINE
    segs = []
    for e in entities:
        if e['type'] in ('SPLINE', 'LINE'):
            pts = get_segment_points(e, n=200 if e['type'] == 'SPLINE' else 2)
            if len(pts) > 0:
                segs.append(pts)
    
    if len(segs) == 0:
        return None
    
    # 贪心连接
    used = [False] * len(segs)
    ordered = [segs[0]]
    used[0] = True
    current_end = segs[0][-1]
    
    for _ in range(len(segs) - 1):
        best_idx = -1
        best_dist = float('inf')
        best_reverse = False
        
        for i in range(len(segs)):
            if used[i]:
                continue
            s, e = segs[i][0], segs[i][-1]
            # 正向：当前尾接 seg 头
            d = np.linalg.norm(s - current_end)
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_reverse = False
            # 反向：当前尾接 seg 尾
            d = np.linalg.norm(e - current_end)
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_reverse = True
        
        if best_idx < 0 or best_dist > 1.0:  # 1mm 容差
            break
        
        used[best_idx] = True
        if best_reverse:
            ordered.append(segs[best_idx][::-1])
            current_end = segs[best_idx][0]
        else:
            ordered.append(segs[best_idx])
            current_end = segs[best_idx][-1]
    
    all_pts = np.vstack(ordered)
    return all_pts


def read_axis_from_dxf(filepath):
    """从 WingsAxis.DXF 读取两个圆心作为轴线"""
    entities = parse_dxf(filepath)
    circles = [e for e in entities if e['type'] == 'CIRCLE']
    if len(circles) < 2:
        raise ValueError(f"轴线 DXF 中需要 2 个圆，只找到 {len(circles)} 个")
    
    p0 = circles[0]['center']
    p1 = circles[1]['center']
    direction = p1 - p0
    dir_len = np.linalg.norm(direction)
    unit_dir = direction / dir_len if dir_len > 0 else np.array([1.0, 0.0])
    unit_perp = np.array([-unit_dir[1], unit_dir[0]])
    
    return {
        'p0': p0,
        'p1': p1,
        'unit_dir': unit_dir,
        'unit_perp': unit_perp,
    }


def calculate_wing(pts, axis, wing_name, n_bins=200):
    """计算翅膀几何参数"""
    # 多边形面积（鞋带公式）
    def shoelace(poly):
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    
    S_poly = shoelace(pts)
    
    # 局部坐标转换
    v = pts - axis['p0']
    r = v @ axis['unit_dir']   # 沿转轴
    y = v @ axis['unit_perp']  # 垂直转轴（展向）
    
    # 展长 = |y| 范围
    y_abs = np.abs(y)
    R = y_abs.max() - y_abs.min()
    
    if R < 1e-6:
        print(f"[{wing_name}] 展长过小")
        return None
    
    # 沿展向分条算弦长
    y_min, y_max = y_abs.min(), y_abs.max()
    bins = np.linspace(y_min, y_max, n_bins + 1)
    y_centers = 0.5 * (bins[:-1] + bins[1:])
    chords = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (y_abs >= bins[i]) & (y_abs < bins[i + 1])
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
    
    # 面积矩
    valid = chords > 1e-10
    r1 = integrate.simpson(y_hat[valid] * c_hat[valid], y_hat[valid]) if np.any(valid) else 0
    r2_sq = integrate.simpson(y_hat[valid]**2 * c_hat[valid], y_hat[valid]) if np.any(valid) else 0
    
    # 质心
    y_cg = integrate.simpson(y_centers * chords, y_centers) / S if S > 0 else 0
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
    
    # 全局坐标
    ax1 = fig.add_subplot(2, 3, 1)
    for prop, color in [(front_prop, 'blue'), (back_prop, 'green')]:
        if prop:
            pts = prop['pts']
            ax1.fill(pts[:, 0], pts[:, 1], alpha=0.3, color=color)
            ax1.plot(pts[:, 0], pts[:, 1], color=color, lw=1.5, label=prop['name'])
    ax1.plot([axis['p0'][0], axis['p1'][0]], [axis['p0'][1], axis['p1'][1]], 'r--', lw=2, label='Axis')
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('DXF Global')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 局部坐标
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
    ax2.set_title('Local coords')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 弦长分布
    ax3 = fig.add_subplot(2, 3, 3)
    for prop, color in [(front_prop, 'blue'), (back_prop, 'green')]:
        if prop:
            ax3.plot(prop['y_hat'], prop['c_hat'], color=color, lw=2, label=prop['name'])
    ax3.axhline(y=1.0, color='k', linestyle='--', lw=1, alpha=0.5)
    ax3.set_xlabel('y_hat = y/R')
    ax3.set_ylabel('c_hat = c/c_avg')
    ax3.set_title('Chord distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 局部填充
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
    ax4.set_title('Local filled')
    ax4.grid(True, alpha=0.3)
    
    # 参数表
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    rows = []
    for p in [front_prop, back_prop]:
        if p:
            rows.append([
                p['name'],
                f"{p['S_mm2']:.1f}",
                f"{p['R_mm']:.1f}",
                f"{p['c_avg_mm']:.1f}",
                f"{p['AR']:.2f}",
                f"{p['r2_sq']:.4f}",
            ])
    if rows:
        tbl = ax5.table(
            cellText=rows,
            colLabels=['Wing', 'S(mm2)', 'R(mm)', 'c_avg(mm)', 'AR', 'r2_sq'],
            loc='center', cellLoc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)
    
    # 空位
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wing_analysis.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {output_dir / "wing_analysis.png"}')
    plt.close()


def main():
    print("=" * 60)
    print("DXF Wing Analysis")
    print("=" * 60)
    
    axis = read_axis_from_dxf(DATA_DIR / 'WingsAxis.DXF')
    print(f"\n[Axis]")
    print(f"  p0: ({axis['p0'][0]:.3f}, {axis['p0'][1]:.3f}) mm")
    print(f"  p1: ({axis['p1'][0]:.3f}, {axis['p1'][1]:.3f}) mm")
    print(f"  Length: {np.linalg.norm(axis['p1'] - axis['p0']):.3f} mm")
    
    front_entities = parse_dxf(DATA_DIR / 'WingFront.DXF')
    back_entities = parse_dxf(DATA_DIR / 'WingBack.DXF')
    
    print(f"\n[Entities]")
    print(f"  Front: {len([e for e in front_entities if e['type'] in ('SPLINE', 'LINE')])} segments")
    print(f"  Back:  {len([e for e in back_entities if e['type'] in ('SPLINE', 'LINE')])} segments")
    
    front_pts = connect_entities(front_entities)
    back_pts = connect_entities(back_entities)
    
    if front_pts is not None:
        print(f"  Front connected: {len(front_pts)} pts")
    if back_pts is not None:
        print(f"  Back connected:  {len(back_pts)} pts")
    
    front_prop = calculate_wing(front_pts, axis, "Front") if front_pts is not None else None
    back_prop = calculate_wing(back_pts, axis, "Back") if back_pts is not None else None
    
    print("\n" + "=" * 60)
    print("Geometry Results")
    print("=" * 60)
    
    all_props = []
    for p in [front_prop, back_prop]:
        if p is None:
            continue
        all_props.append(p)
        print(f"\n--- {p['name']} ---")
        print(f"  Area S       = {p['S_mm2']:.3f} mm2  ({p['S']*1e4:.4f} cm2)")
        print(f"  Span R       = {p['R_mm']:.2f} mm")
        print(f"  Avg chord    = {p['c_avg_mm']:.2f} mm")
        print(f"  Aspect AR    = {p['AR']:.3f}")
        print(f"  CG span      = {p['y_cg']*1000:.2f} mm")
        print(f"  r1           = {p['r1']:.4f}")
        print(f"  r2_sq        = {p['r2_sq']:.4f}")
    
    if not all_props:
        print("ERROR: No valid wings")
        return
    
    total_S = sum(p['S'] for p in all_props)
    print(f"\n  Total area   = {total_S*1e6:.3f} mm2")
    
    # 气动
    print("\n" + "=" * 60)
    print("Aerodynamics")
    print("=" * 60)
    aero = aerodynamic_estimate(all_props, AERO_PARAMS)
    
    for name, res in aero.items():
        if name == 'total':
            continue
        print(f"\n--- {name} ---")
        print(f"  C_L        = {res['C_L']:.3f}")
        print(f"  Re         = {res['Re']:.0f}")
        print(f"  F_L_peak   = {res['F_L_peak_mN']:.3f} mN")
        print(f"  P_inertial = {res['P_inertial_mW']:.2f} mW")
    
    total = aero['total']
    print(f"\n--- Total ---")
    print(f"  Lift (2w)  = {total['F_L_peak_total_mN']:.3f} mN")
    print(f"  Weight     = {total['weight_mN']:.3f} mN")
    print(f"  L/W        = {total['lift_to_weight']:.2f}")
    
    # 保存
    save_data = {
        'axis': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in axis.items()},
        'geometry': [{k: float(v) if isinstance(v, (np.floating, float)) else v 
                      for k, v in p.items() if k not in ('y_hat', 'c_hat', 'y_centers_mm', 'chords_mm', 'pts')}
                     for p in all_props],
        'aerodynamics': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                            for kk, vv in v.items()} 
                        for k, v in aero.items()},
    }
    json_path = DATA_DIR / 'wing_analysis_results.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved JSON: {json_path}")
    
    # 保存弦长分布 CSV
    if front_prop and back_prop:
        chord_df = pd.DataFrame({
            'y_hat': front_prop['y_hat'],
            'c_hat_front': front_prop['c_hat'],
            'c_hat_back': np.interp(
                front_prop['y_hat'],
                back_prop['y_hat'],
                back_prop['c_hat'],
                left=0, right=0
            ),
            'y_center_mm': front_prop['y_centers_mm'],
            'chord_front_mm': front_prop['chords_mm'],
            'chord_back_mm': np.interp(
                front_prop['y_centers_mm'],
                back_prop['y_centers_mm'],
                back_prop['chords_mm'],
                left=0, right=0
            ),
        })
        chord_path = DATA_DIR / 'chord_distribution.csv'
        chord_df.to_csv(chord_path, index=False)
        print(f"Saved chord: {chord_path}")
    
    plot_results(axis, front_prop, back_prop, DATA_DIR)
    print("Done!")


if __name__ == '__main__':
    main()
