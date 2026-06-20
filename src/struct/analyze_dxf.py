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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
FIGURES_DIR = OUTPUT_DIR / 'figures' / 'aero'
MM_TO_M = 1e-3

# ==================== 用户设计参数 ====================
AERO_PARAMS = {
    "rho": 1.225,           # 空气密度 kg/m³
    "nu": 1.46e-5,          # 运动粘度 m²/s
    "m_total": 0.020,       # 总质量 20g
    "m_wing_total": 0.004,  # 四翅总质量 4g
    "f": 15.0,              # 典型频率 Hz (范围 15-20)
    "alpha_deg": 45.0,      # 攻角 °（参考值，实际气动分析使用 dynamic_analysis.py）
    # 注：本脚本仅负责几何参数提取，气动力计算由 dynamic_analysis.py 基于 mechanism.py 实际运动学完成
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


def plot_results(axis, front_prop, back_prop, output_dir):
    fig = plt.figure(figsize=(12, 10))

    # 1. DXF Global
    ax1 = fig.add_subplot(2, 2, 1)
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

    # 2. Local coordinates (spanwise vs chordwise)
    ax2 = fig.add_subplot(2, 2, 2)
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
    ax3 = fig.add_subplot(2, 2, 3)
    for prop, color in [(front_prop, 'blue'), (back_prop, 'green')]:
        if prop:
            ax3.plot(prop['y_hat'], prop['c_hat'], color=color, lw=2, label=prop['name'])
    ax3.axhline(y=1.0, color='k', linestyle='--', lw=1, alpha=0.5)
    ax3.set_xlabel(r'$\hat{y}=(y-y_{\min})/R_w$')
    ax3.set_ylabel(r'$\hat{c}=c/\bar{c}$')
    ax3.set_title('Normalized Chord Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Local filled planform (rotated to spanwise-chordwise view)
    ax4 = fig.add_subplot(2, 2, 4)
    for prop, color in [(front_prop, 'blue'), (back_prop, 'green')]:
        if prop:
            pts = prop['pts']
            v = pts - axis['p0']
            r = v @ axis['unit_dir']
            y = v @ axis['unit_perp']
            ax4.fill(y, r, alpha=0.3, color=color)
            ax4.plot(y, r, color=color, lw=0.5)
    ax4.axvline(x=0, color='r', linestyle='--', lw=1)
    ax4.set_xlabel('y: spanwise (mm)')
    ax4.set_ylabel('r: chordwise (mm)')
    ax4.set_title('Local Filled Planform')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'wing_analysis.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def main():
    print("=" * 70)
    print("BUTTERFLY WING AERODYNAMIC ANALYSIS")
    print("=" * 70)
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
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
    
    # 保存 JSON（仅几何参数，气动力由 dynamic_analysis.py 基于 mechanism.py 实际运动学计算）
    save_data = {
        'params': AERO_PARAMS,
        'axis': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in axis.items()},
        'geometry': [{k: float(v) if isinstance(v, (np.floating, float)) else v
                      for k, v in p.items() if k not in ('y_hat', 'c_hat', 'y_centers_mm', 'chords_mm', 'pts')}
                     for p in all_props],
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
    
    # 保存图表到 output/figures/aero/
    plot_results(axis, front_prop, back_prop, FIGURES_DIR)
    print("\nDone!")


if __name__ == '__main__':
    main()
