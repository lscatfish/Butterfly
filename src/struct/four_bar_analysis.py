"""
四连杆机构运动学分析
曲柄摇杆机构：BP1（曲柄）- P2P1（连杆）- AP2（摇杆）
坐标系：x轴沿机架4（地面），y轴沿A点处竖直机架
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import os
import csv

# ==================== 机构参数 ====================
l1 = 3.8          # BP1 曲柄长度
l2 = 7.1          # P2P1 连杆长度
l3 = 5.0          # AP2 摇杆长度
bx = 1.71   # B点x坐标
h_default = 7.6   # 默认A点高度 (机架AB=sqrt(1.71^2+7.6^2)=7.79)
h_min, h_max, h_step = 5.0, 10.0, 0.5
h_values = np.arange(h_min, h_max + h_step/2, h_step)

omega1 = 1.0  #可改成动态角速度
# 曲柄匀角速度 (rad/s)

out_dir = r"../../output/four_bar_analysis"
os.makedirs(out_dir, exist_ok=True)


# ==================== 核心运动学求解 ====================
def solve_theta3_branches(theta1, h):
    """
    解析法求两个 theta3 分支。
    返回: (theta3_a, theta3_b, P1, valid)  角度范围 [-pi, pi]
    """
    B = np.array([bx, 0.0])
    A = np.array([0.0, h])
    P1 = B + l1 * np.array([np.cos(theta1), np.sin(theta1)])
    
    C_const = (l2**2 - l3**2 - l1**2 - (2*bx)*l1*np.cos(theta1)
               - bx**2 - h**2 + 2*h*l1*np.sin(theta1))
    A_const = -2*l3*(l1*np.cos(theta1) + bx)
    B_const = 2*l3*(h - l1*np.sin(theta1))
    
    R = np.sqrt(A_const**2 + B_const**2)
    if R == 0 or abs(C_const / R) > 1.0 + 1e-9:
        return None, None, P1, False
    
    C_clip = np.clip(C_const, -R, R)
    phi = np.arctan2(B_const, A_const)
    ac = np.arccos(C_clip / R)
    
    t3a = (phi - ac)
    t3b = (phi + ac)
    # 归一化到 [-pi, pi]
    t3a = ((t3a + np.pi) % (2*np.pi)) - np.pi
    t3b = ((t3b + np.pi) % (2*np.pi)) - np.pi
    
    return t3a, t3b, P1, True


def get_P2_from_theta3(theta3, h):
    """由 theta3 求 P2 坐标。"""
    return np.array([l3 * np.cos(theta3), h + l3 * np.sin(theta3)])


def angle_diff(a, b):
    """计算两个角度之间的最小差值（带符号）。"""
    d = a - b
    while d > np.pi:
        d -= 2*np.pi
    while d < -np.pi:
        d += 2*np.pi
    return d


def simulate_cycle(h, n=2000):
    """
    模拟一整周，用角度连续性保持分支。
    返回完整数据字典。
    """
    theta1s = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    data = {
        'theta1': theta1s,
        'P1x': [], 'P1y': [],
        'P2x': [], 'P2y': [],
        'theta3': [],
        'v_P2': [], 'a_P2': [],
        'pressure_angle': [],
        'omega3': [], 'alpha3': []
    }
    
    prev_t3 = None
    
    for t1 in theta1s:
        t3a, t3b, P1, valid = solve_theta3_branches(t1, h)
        if not valid:
            _fill_prev(data)
            continue
        
        # 选择分支：优先用角度连续性；首次选与曲柄同侧（开式）
        if prev_t3 is None:
            # 首次：计算两个分支的叉积，选开式（P2与P1在AB同侧）
            B = np.array([bx, 0.0])
            A = np.array([0.0, h])
            AB = B - A
            
            def side(t3):
                P2 = get_P2_from_theta3(t3, h)
                return AB[0]*(P2[1]-A[1]) - AB[1]*(P2[0]-A[0])
            
            cross_P1 = AB[0]*(P1[1]-A[1]) - AB[1]*(P1[0]-A[0])
            side_a = side(t3a) * cross_P1
            side_b = side(t3b) * cross_P1
            
            # 选开式（同侧）分支；若都同侧则选角度较小的
            if side_a > 0 and side_b <= 0:
                t3 = t3a
            elif side_b > 0 and side_a <= 0:
                t3 = t3b
            else:
                # 都同侧或都异侧，选 |t3| 较小的（更水平）
                t3 = t3a if abs(t3a) < abs(t3b) else t3b
        else:
            # 用角度连续性
            da = abs(angle_diff(t3a, prev_t3))
            db = abs(angle_diff(t3b, prev_t3))
            t3 = t3a if da <= db else t3b
        
        P2 = get_P2_from_theta3(t3, h)
        
        # 速度和加速度
        B = np.array([bx, 0.0])
        A = np.array([0.0, h])
        t1_corr = np.arctan2(P1[1]-B[1], P1[0]-B[0])
        t2 = np.arctan2(P2[1]-P1[1], P2[0]-P1[0])
        
        M = np.array([
            [ l2*np.sin(t2), -l3*np.sin(t3)],
            [-l2*np.cos(t2),  l3*np.cos(t3)]
        ])
        rhs = np.array([-l1*omega1*np.sin(t1_corr), l1*omega1*np.cos(t1_corr)])
        
        try:
            omegas = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            _fill_prev(data)
            continue
        omega2, omega3 = omegas
        
        v_P2 = abs(l3 * omega3)
        
        rhs_a = np.array([
            l1*omega1**2*np.cos(t1_corr) - l2*omega2**2*np.cos(t2) + l3*omega3**2*np.cos(t3),
            l1*omega1**2*np.sin(t1_corr) - l2*omega2**2*np.sin(t2) + l3*omega3**2*np.sin(t3)
        ])
        try:
            alphas = np.linalg.solve(M, rhs_a)
        except np.linalg.LinAlgError:
            _fill_prev(data)
            continue
        alpha2, alpha3 = alphas
        
        a_tang = l3 * abs(alpha3)
        a_norm = l3 * omega3**2
        a_P2 = np.sqrt(a_tang**2 + a_norm**2)
        
        # 压力角
        r_AP2 = P2 - A
        r_P2P1 = P1 - P2
        cos_a = np.dot(r_AP2, r_P2P1) / (np.linalg.norm(r_AP2)*np.linalg.norm(r_P2P1))
        cos_a = np.clip(cos_a, -1.0, 1.0)
        ang = np.arccos(cos_a)
        trans = min(ang, np.pi - ang)
        pa = np.degrees(np.pi/2 - trans)
        
        data['P1x'].append(P1[0]); data['P1y'].append(P1[1])
        data['P2x'].append(P2[0]); data['P2y'].append(P2[1])
        data['theta3'].append(t3)
        data['v_P2'].append(v_P2)
        data['a_P2'].append(a_P2)
        data['pressure_angle'].append(pa)
        data['omega3'].append(omega3)
        data['alpha3'].append(alpha3)
        
        prev_t3 = t3
    
    for k in data:
        if k != 'theta1':
            data[k] = np.array(data[k])
    return data


def _fill_prev(data):
    """用前一个值填充（保持数组长度一致）。"""
    if len(data['P1x']) > 0:
        for k in ['P1x','P1y','P2x','P2y','theta3','v_P2','a_P2','pressure_angle','omega3','alpha3']:
            data[k].append(data[k][-1])
    else:
        for k in ['P1x','P1y','P2x','P2y','theta3','v_P2','a_P2','pressure_angle','omega3','alpha3']:
            data[k].append(np.nan)


# ==================== 极限位置与急回特性（模拟法）====================
def find_limit_positions(h, n=4000):
    """
    用模拟数据确定极限位置。
    simulate_cycle 已通过角度连续性正确跟踪开式配置，
    直接取 theta3 的极值点即为两个极限位置。
    """
    data = simulate_cycle(h, n)
    
    # 找 theta3 极值点
    idx_min = np.nanargmin(data['theta3'])
    idx_max = np.nanargmax(data['theta3'])
    
    lim_max = {
        'theta1': data['theta1'][idx_max],
        'P1': np.array([data['P1x'][idx_max], data['P1y'][idx_max]]),
        'P2': np.array([data['P2x'][idx_max], data['P2y'][idx_max]]),
        'theta3': data['theta3'][idx_max],
    }
    lim_min = {
        'theta1': data['theta1'][idx_min],
        'P1': np.array([data['P1x'][idx_min], data['P1y'][idx_min]]),
        'P2': np.array([data['P2x'][idx_min], data['P2y'][idx_min]]),
        'theta3': data['theta3'][idx_min],
    }
    
    # 极位夹角：两极限位置曲柄转角之差取较小弧度，
    # 极位夹角 = 180° - 较小弧度（因为重叠共线时曲柄近乎反向）
    dt = abs(lim_max['theta1'] - lim_min['theta1'])
    if dt > np.pi:
        dt = 2*np.pi - dt
    theta_deg = 180.0 - np.degrees(dt)
    K = (180.0 + theta_deg) / (180.0 - theta_deg)
    
    return K, theta_deg, lim_min, lim_max


# ==================== 绘图 ====================
def set_style():
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
    except:
        pass


def plot_overview(h, lim_min, lim_max, K, theta_deg, save_dir=None):
    """机构简图 + 轨迹 + 极限位置（每个h单独保存）"""
    if save_dir is None:
        save_dir = out_dir
    data = simulate_cycle(h, n=1000)
    A = np.array([0.0, h])
    B = np.array([bx, 0.0])
    
    # 计算P2轨迹范围，自适应扩大视野
    p2x_min, p2x_max = np.nanmin(data['P2x']), np.nanmax(data['P2x'])
    p2y_min, p2y_max = np.nanmin(data['P2y']), np.nanmax(data['P2y'])
    p1x_min, p1x_max = np.nanmin(data['P1x']), np.nanmax(data['P1x'])
    p1y_min, p1y_max = np.nanmin(data['P1y']), np.nanmax(data['P1y'])
    
    all_x_min = min(p2x_min, p1x_min, 0, bx) - 10.0
    all_x_max = max(p2x_max, p1x_max, bx) + 5.0
    all_y_min = min(p2y_min, p1y_min, 0) - 3.0
    all_y_max = max(p2y_max, p1y_max, h) + 5.0
    
    fig, ax = plt.subplots(figsize=(20, 17))
    
    # 机架（地面）
    ax.axhline(y=0, color='black', linewidth=2)
    ground_x_max = max(all_x_max, 14)
    ax.fill_between([all_x_min, ground_x_max], -1.5, 0, color='lightgray', alpha=0.5)
    for xg in np.arange(all_x_min, ground_x_max, 0.6):
        ax.plot([xg, xg-0.4], [0, -0.4], 'k-', linewidth=0.5)
    # 竖直机架
    ax.axvline(x=0, color='black', linewidth=2, ymin=0, ymax=1)
    for yg in np.arange(0, h+2, 0.6):
        ax.plot([-0.4, 0], [yg, yg-0.4], 'k-', linewidth=0.5)
    
    # 轨迹
    ax.plot(data['P1x'], data['P1y'], 'b--', linewidth=1.2, alpha=0.7, label='P1轨迹（圆）')
    ax.plot(data['P2x'], data['P2y'], 'r-', linewidth=1.5, alpha=0.8, label='P2轨迹（摇杆端）')
    
    # 极限位置
    for i, lim in enumerate([lim_max, lim_min]):
        label = 'P2极限1' if i == 0 else 'P2极限2'
        P1e, P2e = lim['P1'], lim['P2']
        ax.plot([B[0], P1e[0]], [B[1], P1e[1]], 'b-', linewidth=3, alpha=0.35)
        ax.plot([P1e[0], P2e[0]], [P1e[1], P2e[1]], 'g-', linewidth=3, alpha=0.35)
        ax.plot([P2e[0], A[0]], [P2e[1], A[1]], 'm-', linewidth=3, alpha=0.35)
        ax.plot(P2e[0], P2e[1], 'ro', markersize=10)
        ax.plot(P1e[0], P1e[1], 'bo', markersize=8)
        dy = 0.8 if i == 0 else -1.2
        ax.annotate(label, xy=P2e, xytext=(P2e[0]+1.0, P2e[1]+dy),
                    fontsize=10, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
    
    # 中间位姿
    t_mid = np.pi / 3
    t3a, t3b, P1m, _ = solve_theta3_branches(t_mid, h)
    P2m = get_P2_from_theta3(t3a, h)  # 首次默认取a
    
    ax.plot([B[0], P1m[0]], [B[1], P1m[1]], 'b-', linewidth=2.5, label='杆1（曲柄）')
    ax.plot([P1m[0], P2m[0]], [P1m[1], P2m[1]], 'g-', linewidth=2.5, label='杆2（连杆）')
    ax.plot([P2m[0], A[0]], [P2m[1], A[1]], 'm-', linewidth=2.5, label='杆3（摇杆）')
    ax.plot(B[0], B[1], 'ks', markersize=12)
    ax.plot(A[0], A[1], 'ks', markersize=12)
    ax.plot(P1m[0], P1m[1], 'wo', markersize=9, markeredgecolor='blue', markeredgewidth=2)
    ax.plot(P2m[0], P2m[1], 'wo', markersize=12, markeredgecolor='red', markeredgewidth=2.5)
    
    # 标注文字偏移自适应
    offset_scale = max(all_x_max - all_x_min, all_y_max - all_y_min) * 0.03
    ax.text(B[0]-0.6, B[1]-0.8, 'B', fontsize=14, fontweight='bold', ha='right')
    ax.text(A[0]-0.6, A[1]+0.5, 'A', fontsize=14, fontweight='bold', ha='right')
    ax.text(P1m[0]+0.4, P1m[1]+0.5, 'P1', fontsize=13, fontweight='bold', color='blue')
    ax.text(P2m[0]+0.5, P2m[1]+0.6, 'P2', fontsize=13, fontweight='bold', color='red')
    ax.text((B[0]+P1m[0])/2-0.4, (B[1]+P1m[1])/2+0.4, '1', fontsize=11)
    ax.text((P1m[0]+P2m[0])/2+0.3, (P1m[1]+P2m[1])/2+0.3, '2', fontsize=11)
    ax.text((P2m[0]+A[0])/2-0.5, (P2m[1]+A[1])/2+0.3, '3', fontsize=11)
    ax.text((A[0]+B[0])/2-0.3, (A[1]+B[1])/2-0.7, '4', fontsize=11, color='gray')
    
    arc = Arc((B[0], B[1]), 1.2, 1.2, angle=0, theta1=30, theta2=70, color='blue', linewidth=1.5)
    ax.add_patch(arc)
    t3m = np.arctan2(P2m[1]-A[1], P2m[0]-A[0])
    arc2 = Arc((A[0], A[1]), 1.5, 1.5, angle=0,
               theta1=np.degrees(t3m)*0.7-20, theta2=np.degrees(t3m)*0.7+20,
               color='magenta', linewidth=1.5)
    ax.add_patch(arc2)
    
    ax.set_aspect('equal')
    ax.set_xlim(all_x_min, all_x_max)
    ax.set_ylim(all_y_min, all_y_max)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    title_str = f'四连杆机构运动简图及轨迹 (h={h:.1f})\n'
    if K is not None:
        title_str += f'急回特性系数 K={K:.3f}, 极位夹角 θ={theta_deg:.1f}°'
    ax.set_title(title_str, fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'mechanism_h{h:.1f}.png'), dpi=200)
    plt.close()
    return data


def plot_va(data, h, save_dir=None):
    """速度、加速度曲线（每个h单独保存）"""
    if save_dir is None:
        save_dir = out_dir
    t1d = np.degrees(data['theta1'])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(t1d, data['v_P2'], 'b-', linewidth=1.5)
    axes[0].set_xlabel('theta1 (deg)')
    axes[0].set_ylabel('v_P2 (m/s)')
    axes[0].set_title(f'P2 速度大小曲线 (h={h:.1f}, omega1={omega1} rad/s)')
    axes[0].set_xlim(0, 360); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t1d, data['a_P2'], 'r-', linewidth=1.5)
    axes[1].set_xlabel('theta1 (deg)')
    axes[1].set_ylabel('a_P2 (m/s^2)')
    axes[1].set_title(f'P2 加速度大小曲线 (h={h:.1f}, omega1={omega1} rad/s)')
    axes[1].set_xlim(0, 360); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'velocity_accel_h{h:.1f}.png'), dpi=200)
    plt.close()


def plot_pa(data, h, save_dir=None):
    """压力角曲线（每个h单独保存）"""
    if save_dir is None:
        save_dir = out_dir
    t1d = np.degrees(data['theta1'])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t1d, data['pressure_angle'], 'g-', linewidth=1.5)
    ax.set_xlabel('theta1 (deg)')
    ax.set_ylabel('压力角 (deg)')
    ax.set_title(f'压力角变化曲线 (h={h:.1f})')
    ax.set_xlim(0, 360); ax.grid(True, alpha=0.3)
    max_pa = np.nanmax(data['pressure_angle'])
    ax.axhline(y=max_pa, color='r', linestyle='--', alpha=0.5,
               label=f'最大压力角={max_pa:.1f}deg')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pressure_angle_h{h:.1f}.png'), dpi=200)
    plt.close()


def plot_multi_h():
    """图4：多h轨迹"""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(h_values)))
    for i, h in enumerate(h_values):
        data = simulate_cycle(h, n=500)
        ax.plot(data['P2x'], data['P2y'], color=colors[i], linewidth=1.0,
                label=f'h={h:.1f}' if i % 3 == 0 else '')
        ax.plot(0, h, 'o', color=colors[i], markersize=4)
    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.axvline(x=0, color='black', linewidth=1.5)
    ax.fill_between([-2, 14], -1, 0, color='lightgray', alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-3, 14); ax.set_ylim(-2, 28)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('不同 h 值下 P2 的运动轨迹')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_multi_h_trajectories.png'), dpi=200)
    plt.close()


def compute_and_save_K():
    """计算K值表格和曲线"""
    results = []
    for h in h_values:
        K, theta_deg, lim_min, lim_max = find_limit_positions(h)
        data = simulate_cycle(h, n=500)
        t3_min = np.degrees(np.nanmin(data['theta3']))
        t3_max = np.degrees(np.nanmax(data['theta3']))
        results.append({'h': h, 'K': K, 'theta_deg': theta_deg,
                        't3_min': t3_min, 't3_max': t3_max})
    
    csv_path = os.path.join(out_dir, 'quick_return_table.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['h', 'K', 'theta_deg', 't3_min', 't3_max'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    hs = [r['h'] for r in results]
    Ks = [r['K'] if r['K'] is not None else np.nan for r in results]
    ts = [r['theta_deg'] if r['theta_deg'] is not None else np.nan for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('h')
    ax1.set_ylabel('K', color='tab:blue')
    ax1.plot(hs, Ks, 'o-', color='tab:blue', linewidth=1.5, markersize=5)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('急回特性系数 K 随 h 的变化')
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.set_ylabel('极位夹角 theta (deg)', color='tab:red')
    ax2.plot(hs, ts, 's--', color='tab:red', linewidth=1.5, markersize=4)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig5_K_vs_h.png'), dpi=200)
    plt.close()
    return results


# ==================== 主程序 ====================
def main():
    set_style()

    # A点高度固定，只跑 h_default
    h = h_default
    h_str = f'h{h:.1f}'
    h_dir = os.path.join(out_dir, h_str)
    os.makedirs(h_dir, exist_ok=True)

    print(f"正在处理 h={h:.1f} ...")
    K, theta_deg, lim_min, lim_max = find_limit_positions(h)
    data = plot_overview(h, lim_min, lim_max, K, theta_deg, save_dir=h_dir)
    plot_va(data, h, save_dir=h_dir)
    plot_pa(data, h, save_dir=h_dir)

    print(f"\n急回特性 (h={h:.1f}):")
    print("-" * 55)
    Ks = f"{K:.4f}" if K is not None else "N/A"
    ts = f"{theta_deg:.2f}" if theta_deg is not None else "N/A"
    print(f"  K = {Ks}, theta = {ts} deg")
    print("-" * 55)
    print(f"\n结果已保存到: {out_dir}")

    # 同时保存到 CSV (单行)
    csv_path = os.path.join(os.path.dirname(os.path.dirname(out_dir)), "output", "table", "quick_return_table.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['h', 'K', 'theta_deg', 't3_min', 't3_max'])
        t3_min = lim_min['theta3'] if lim_min else 0
        t3_max = lim_max['theta3'] if lim_max else 0
        writer.writerow([h, K if K else '', theta_deg if theta_deg else '',
                         t3_min, t3_max])
    print(f"CSV 已保存到: {csv_path}")


if __name__ == '__main__':
    main()
