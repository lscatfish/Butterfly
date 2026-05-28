#!/usr/bin/env python3
"""
仿生蝴蝶翅膀几何参数提取与气动估算
读取 SolidWorks 导出的轮廓 CSV，计算面积、弦长分布、面积矩等
"""

import numpy as np
import pandas as pd
from scipy import integrate
from pathlib import Path
import matplotlib.pyplot as plt
import json

# ==================== 配置 ====================
DATA_DIR = Path(__file__).parent  # 脚本所在目录
csv_front = DATA_DIR / "wing_front.csv"
csv_back  = DATA_DIR / "wing_back.csv"
csv_axis  = DATA_DIR / "wing_axis.csv"

# 单位转换：SolidWorks 导出为 mm，气动公式用 m
MM_TO_M = 1e-3

# 气动默认参数（基于文献综述典型值）
AERO_PARAMS = {
    "rho": 1.225,           # 空气密度 kg/m^3
    "nu": 1.46e-5,          # 运动粘度 m^2/s
    "m_total": 0.0216,      # 总质量 kg（可根据实际修改）
    "f": 10.0,              # 拍动频率 Hz
    "Phi_max_deg": 80.0,    # 拍动幅度（单向，度）
    "alpha_deg": 45.0,      # 攻角度
}


def read_axis(filepath):
    """读取转轴数据：起点、终点、方向向量"""
    df = pd.read_csv(filepath)
    p0 = df[df['Type'] == 0][['X', 'Y', 'Z']].values[0] * MM_TO_M
    p1 = df[df['Type'] == 1][['X', 'Y', 'Z']].values[0] * MM_TO_M
    direction = df[df['Type'] == 2][['X', 'Y', 'Z']].values[0] * MM_TO_M
    
    # 归一化方向向量
    dir_len = np.linalg.norm(direction)
    unit_dir = direction / dir_len if dir_len > 0 else np.array([1.0, 0.0, 0.0])
    
    # 在 XY 平面内的垂直方向（逆时针旋转 90°）
    unit_perp = np.array([-unit_dir[1], unit_dir[0], 0.0])
    perp_len = np.linalg.norm(unit_perp)
    if perp_len > 0:
        unit_perp = unit_perp / perp_len
    else:
        unit_perp = np.array([0.0, 1.0, 0.0])
    
    return {
        'p0': p0,
        'p1': p1,
        'unit_dir': unit_dir,
        'unit_perp': unit_perp,
        'length_m': dir_len * MM_TO_M,  # direction 已经是 mm，但用 p1-p0 更准确
    }


def read_wing_points(filepath):
    """读取翅膀轮廓点，去重，转换单位"""
    df = pd.read_csv(filepath)
    pts = df[['X', 'Y', 'Z']].values * MM_TO_M
    
    # 去重：去掉距离过近的点（避免宏中 segment 连接处重复）
    if len(pts) > 1:
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        mask = np.concatenate(([True], dists > 1e-7))
        pts = pts[mask]
    
    return pts


def transform_to_local(pts, axis):
    """将全局坐标转换到以转轴为基准的局部坐标系 (r, y)
    r: 沿转轴的带符号距离（展向）
    y: 垂直于转轴的带符号距离（弦向）
    """
    v = pts - axis['p0']  # 平移到转轴起点
    r = v @ axis['unit_dir']
    y = v @ axis['unit_perp']
    return r, y


def calculate_wing_properties(r, y, wing_name, n_bins=200):
    """计算翅膀几何参数"""
    
    # 1. 多边形面积（鞋带公式，假设点大致按轮廓顺序）
    def shoelace_area(r_coords, y_coords):
        return 0.5 * abs(
            np.dot(r_coords, np.roll(y_coords, -1)) 
            - np.dot(y_coords, np.roll(r_coords, -1))
        )
    
    S_poly = shoelace_area(r, y)
    
    # 2. 展向分条计算弦长分布
    r_min, r_max = r.min(), r.max()
    R = r_max - r_min  # 展长（半展长）
    
    if R < 1e-6:
        print(f"[{wing_name}] 警告：展长过小")
        return None
    
    # 分条边界
    bins = np.linspace(r_min, r_max, n_bins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    dr = bins[1] - bins[0]
    
    chords = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(mask):
            chords[i] = y[mask].max() - y[mask].min()
        else:
            chords[i] = 0.0
    
    # 3. 面积（数值积分：∫ c(r) dr）
    S = integrate.simpson(chords, r_centers)
    if S <= 0:
        S = S_poly  # fallback
    
    # 4. 平均弦长
    c_avg = S / R
    
    # 5. 归一化弦长分布
    c_hat = chords / c_avg if c_avg > 0 else np.zeros_like(chords)
    r_hat = (r_centers - r_min) / R  # 归一化展向坐标 [0, 1]
    
    # 6. 面积矩（归一化）
    valid = chords > 1e-10
    if np.any(valid):
        r1 = integrate.simpson(r_hat[valid] * c_hat[valid], r_hat[valid])
        r2_sq = integrate.simpson(r_hat[valid]**2 * c_hat[valid], r_hat[valid])
    else:
        r1 = r2_sq = 0.0
    
    # 7. 质心位置（展向）
    if S > 0:
        r_cg = integrate.simpson(r_centers * chords, r_centers) / S
        r_cg_hat = (r_cg - r_min) / R
    else:
        r_cg = r_cg_hat = 0.0
    
    # 8. 展弦比
    AR = R**2 / S if S > 0 else 0.0
    
    return {
        'name': wing_name,
        'S': S,                 # 单翼面积 m^2
        'R': R,                 # 展长 m
        'c_avg': c_avg,         # 平均弦长 m
        'AR': AR,               # 展弦比
        'r_cg': r_cg,           # 质心展向位置 m（全局 r 坐标）
        'r_cg_hat': r_cg_hat,   # 归一化质心位置
        'r1': r1,               # 一阶面积矩
        'r2_sq': r2_sq,         # 二阶面积矩
        'S_poly': S_poly,       # 多边形面积（校验）
        'r_hat': r_hat,         # 归一化展向坐标数组
        'c_hat': c_hat,         # 归一化弦长数组
        'r_centers': r_centers, # 展向坐标数组（全局，m）
        'chords': chords,       # 弦长数组（m）
    }


def aerodynamic_estimate(props, params):
    """基于准定常模型估算气动参数"""
    
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
        
        # 最大角速度
        phi_dot_max = 2 * np.pi * f * Phi_max
        
        # 升力系数（经验公式，alpha 用角度）
        C_L = 0.255 + 1.58 * np.sin(np.deg2rad(2.13 * alpha_deg - 7.2))
        C_D = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * alpha_deg - 9.82))
        
        # 平动升力峰值（使用面积矩简化）
        # F_L = 0.5 * rho * C_L * (phi_dot_max * R)^2 * S * r2_sq
        F_L_peak = 0.5 * rho * C_L * (phi_dot_max * R)**2 * S * r2_sq
        
        # 翼尖最大速度
        u_tip_max = phi_dot_max * R
        
        # 平均速度（简谐运动）
        u_mean = (2.0 / np.pi) * u_tip_max
        
        # 雷诺数
        Re = u_mean * c_avg / nu
        
        # 减缩频率（使用平均弦长为参考长度）
        omega = 2 * np.pi * f
        k = omega * c_avg / (2 * u_mean) if u_mean > 0 else 0
        
        # 转动惯量（假设质量均匀分布，厚度均匀）
        # 单翼质量估算：假设两翅占总质量 10%（薄膜+翅脉很轻）
        m_w = m_total * 0.05  # 5% 每翅，可调
        I_w = m_w * R**2 * r2_sq
        
        # 惯性功率
        KE = 0.5 * I_w * phi_dot_max**2
        P_inertial = 4 * KE * f
        
        results[name] = {
            'C_L': C_L,
            'C_D': C_D,
            'phi_dot_max_rad_s': phi_dot_max,
            'u_tip_max_m_s': u_tip_max,
            'u_mean_m_s': u_mean,
            'Re': Re,
            'k': k,
            'F_L_peak_N': F_L_peak,
            'F_L_peak_mN': F_L_peak * 1000,
            'm_w_kg': m_w,
            'I_w_kg_m2': I_w,
            'KE_mJ': KE * 1000,
            'P_inertial_mW': P_inertial * 1000,
        }
    
    # 总升力（两翅）
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


def plot_results(axis, front_pts, back_pts, front_prop, back_prop, output_dir):
    """绘制分析图表"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # ---------- 子图 1: 原始坐标系下的轮廓与转轴 ----------
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(front_pts[:, 0]*1000, front_pts[:, 1]*1000, 'b-', lw=1, label='前翅 WingFront')
    ax1.plot(back_pts[:, 0]*1000, back_pts[:, 1]*1000, 'g-', lw=1, label='后翅 WingBack')
    
    # 转轴
    t = np.linspace(-5000, 15000, 2)
    ax1.plot((axis['p0'][0] + t*axis['unit_dir'][0])*1000,
             (axis['p0'][1] + t*axis['unit_dir'][1])*1000,
             'r--', lw=2, label='转轴')
    ax1.scatter(*axis['p0'][:2]*1000, c='red', s=50, zorder=5, marker='o')
    ax1.scatter(*axis['p1'][:2]*1000, c='red', s=50, zorder=5, marker='s')
    
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('SolidWorks 全局坐标系')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ---------- 子图 2: 局部坐标系 (r, y) ----------
    ax2 = fig.add_subplot(2, 3, 2)
    
    # 重新计算局部坐标用于绘图
    r_f, y_f = transform_to_local(front_pts, axis)
    r_b, y_b = transform_to_local(back_pts, axis)
    
    ax2.plot(r_f*1000, y_f*1000, 'b-', lw=1, label='前翅')
    ax2.plot(r_b*1000, y_b*1000, 'g-', lw=1, label='后翅')
    ax2.axhline(y=0, color='r', linestyle='--', lw=1, label='转轴 (y=0)')
    ax2.set_xlabel('r: 展向距离 (mm)')
    ax2.set_ylabel('y: 弦向距离 (mm)')
    ax2.set_title('局部坐标系（以转轴为基准）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ---------- 子图 3: 弦长分布 ----------
    ax3 = fig.add_subplot(2, 3, 3)
    
    if front_prop:
        ax3.plot(front_prop['r_hat'], front_prop['c_hat'], 'b-', lw=2, label='前翅')
    if back_prop:
        ax3.plot(back_prop['r_hat'], back_prop['c_hat'], 'g-', lw=2, label='后翅')
    ax3.axhline(y=1.0, color='k', linestyle='--', lw=1, alpha=0.5, label='平均弦长')
    ax3.set_xlabel('归一化展向位置 $\\hat{r} = r/R$')
    ax3.set_ylabel('归一化弦长 $\\hat{c} = c/c_{avg}$')
    ax3.set_title('弦长分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ---------- 子图 4: 前翅局部放大 ----------
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.fill(r_f*1000, y_f*1000, alpha=0.3, color='blue')
    ax4.plot(r_f*1000, y_f*1000, 'b-', lw=1)
    ax4.axhline(y=0, color='r', linestyle='--', lw=1)
    ax4.set_aspect('equal')
    ax4.set_xlabel('r (mm)')
    ax4.set_ylabel('y (mm)')
    if front_prop:
        ax4.set_title(f"前翅: S={front_prop['S']*1e6:.1f} mm², AR={front_prop['AR']:.2f}")
    ax4.grid(True, alpha=0.3)
    
    # ---------- 子图 5: 后翅局部放大 ----------
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.fill(r_b*1000, y_b*1000, alpha=0.3, color='green')
    ax5.plot(r_b*1000, y_b*1000, 'g-', lw=1)
    ax5.axhline(y=0, color='r', linestyle='--', lw=1)
    ax5.set_aspect('equal')
    ax5.set_xlabel('r (mm)')
    ax5.set_ylabel('y (mm)')
    if back_prop:
        ax5.set_title(f"后翅: S={back_prop['S']*1e6:.1f} mm², AR={back_prop['AR']:.2f}")
    ax5.grid(True, alpha=0.3)
    
    # ---------- 子图 6: 几何参数汇总表 ----------
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    table_data = []
    for prop in [front_prop, back_prop]:
        if prop:
            table_data.append([
                prop['name'],
                f"{prop['S']*1e6:.1f}",
                f"{prop['R']*1000:.1f}",
                f"{prop['c_avg']*1000:.1f}",
                f"{prop['AR']:.2f}",
                f"{prop['r2_sq']:.4f}",
            ])
    
    if table_data:
        table = ax6.table(
            cellText=table_data,
            colLabels=['翅膀', '面积 (mm²)', '展长 (mm)', '平均弦长 (mm)', '展弦比 AR', 'r̂₂²'],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        ax6.set_title('几何参数汇总', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wing_geometry_analysis.png', dpi=200, bbox_inches='tight')
    print(f"图表已保存: {output_dir / 'wing_geometry_analysis.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("仿生蝴蝶翅膀几何参数提取与气动估算")
    print("=" * 60)
    
    # 1. 读取转轴
    axis = read_axis(csv_axis)
    print(f"\n[转轴信息]")
    print(f"  起点: ({axis['p0'][0]*1000:.2f}, {axis['p0'][1]*1000:.2f}, {axis['p0'][2]*1000:.2f}) mm")
    print(f"  终点: ({axis['p1'][0]*1000:.2f}, {axis['p1'][1]*1000:.2f}, {axis['p1'][2]*1000:.2f}) mm")
    print(f"  方向: ({axis['unit_dir'][0]:.4f}, {axis['unit_dir'][1]:.4f}, {axis['unit_dir'][2]:.4f})")
    print(f"  长度: {np.linalg.norm(axis['p1']-axis['p0'])*1000:.2f} mm")
    
    # 2. 读取翅膀轮廓
    front_pts = read_wing_points(csv_front)
    back_pts  = read_wing_points(csv_back)
    print(f"\n[轮廓点数]")
    print(f"  前翅: {len(front_pts)} 点")
    print(f"  后翅: {len(back_pts)} 点")
    
    # 3. 计算几何参数
    r_f, y_f = transform_to_local(front_pts, axis)
    r_b, y_b = transform_to_local(back_pts, axis)
    
    front_prop = calculate_wing_properties(r_f, y_f, "前翅 WingFront")
    back_prop  = calculate_wing_properties(r_b, y_b, "后翅 WingBack")
    
    # 4. 输出几何参数
    print("\n" + "=" * 60)
    print("几何参数计算结果")
    print("=" * 60)
    
    all_props = []
    for prop in [front_prop, back_prop]:
        if prop is None:
            continue
        all_props.append(prop)
        print(f"\n--- {prop['name']} ---")
        print(f"  单翼面积 S       = {prop['S']*1e6:>10.3f} mm²  ({prop['S']*1e4:.4f} cm²)")
        print(f"  展长 R           = {prop['R']*1000:>10.2f} mm")
        print(f"  平均弦长 c_avg   = {prop['c_avg']*1000:>10.2f} mm")
        print(f"  展弦比 AR        = {prop['AR']:>10.3f}")
        print(f"  质心位置 r_cg    = {prop['r_cg']*1000:>10.2f} mm (距根部)")
        print(f"  归一化质心 r̂_cg   = {prop['r_cg_hat']:>10.4f}")
        print(f"  一阶面积矩 r̂₁    = {prop['r1']:>10.4f}")
        print(f"  二阶面积矩 r̂₂²   = {prop['r2_sq']:>10.4f}")
        print(f"  [校验] 多边形面积 = {prop['S_poly']*1e6:.3f} mm²")
    
    # 总面积
    total_S = sum(p['S'] for p in all_props)
    print(f"\n  前后翅总面积     = {total_S*1e6:.3f} mm²")
    
    # 5. 气动估算
    print("\n" + "=" * 60)
    print("气动参数估算（基于准定常模型）")
    print("=" * 60)
    print(f"  输入参数: m={AERO_PARAMS['m_total']}kg, f={AERO_PARAMS['f']}Hz, "
          f"Φ={AERO_PARAMS['Phi_max_deg']}°, α={AERO_PARAMS['alpha_deg']}°")
    
    aero = aerodynamic_estimate(all_props, AERO_PARAMS)
    
    for name, res in aero.items():
        if name == 'total':
            continue
        print(f"\n--- {name} ---")
        print(f"  升力系数 C_L       = {res['C_L']:.3f}")
        print(f"  阻力系数 C_D       = {res['C_D']:.3f}")
        print(f"  最大角速度         = {res['phi_dot_max_rad_s']:.1f} rad/s")
        print(f"  翼尖最大速度       = {res['u_tip_max_m_s']:.2f} m/s")
        print(f"  平均雷诺数 Re      = {res['Re']:.0f}")
        print(f"  减缩频率 k         = {res['k']:.3f}")
        print(f"  峰值升力           = {res['F_L_peak_mN']:.3f} mN")
        print(f"  单翼质量(估)       = {res['m_w_kg']*1000:.3f} g")
        print(f"  转动惯量 I         = {res['I_w_kg_m2']*1e9:.3f} kg·mm²")
        print(f"  惯性功率           = {res['P_inertial_mW']:.2f} mW")
    
    total = aero['total']
    print(f"\n--- 总计 ---")
    print(f"  双翅峰值升力       = {total['F_L_peak_total_mN']:.3f} mN")
    print(f"  重量               = {total['weight_mN']:.3f} mN")
    print(f"  升重比             = {total['lift_to_weight']:.2f}")
    
    # 6. 保存数据
    output_dir = DATA_DIR
    
    # 保存 JSON
    save_data = {
        'axis': {
            'p0_mm': axis['p0'].tolist(),
            'p1_mm': axis['p1'].tolist(),
            'unit_dir': axis['unit_dir'].tolist(),
        },
        'geometry': [{
            'name': p['name'],
            'S_m2': float(p['S']),
            'R_m': float(p['R']),
            'c_avg_m': float(p['c_avg']),
            'AR': float(p['AR']),
            'r_cg_m': float(p['r_cg']),
            'r_cg_hat': float(p['r_cg_hat']),
            'r1': float(p['r1']),
            'r2_sq': float(p['r2_sq']),
        } for p in all_props],
        'aerodynamics': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv 
                            for kk, vv in v.items()} 
                        for k, v in aero.items()},
        'params': AERO_PARAMS,
    }
    
    json_path = output_dir / 'wing_analysis_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n数据已保存: {json_path}")
    
    # 保存弦长分布 CSV
    if front_prop:
        chord_df = pd.DataFrame({
            'r_hat': front_prop['r_hat'],
            'c_hat_front': front_prop['c_hat'],
        })
        if back_prop:
            # 插值到相同的 r_hat 网格
            c_back_interp = np.interp(
                front_prop['r_hat'],
                back_prop['r_hat'],
                back_prop['c_hat'],
                left=0, right=0
            )
            chord_df['c_hat_back'] = c_back_interp
        chord_df.to_csv(output_dir / 'chord_distribution.csv', index=False)
        print(f"弦长分布已保存: {output_dir / 'chord_distribution.csv'}")
    
    # 7. 绘图
    if all_props:
        plot_results(axis, front_pts, back_pts, front_prop, back_prop, output_dir)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
