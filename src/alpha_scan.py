#!/usr/bin/env python3
"""
仿生蝴蝶翅膀安装角（α）参数扫描
目的：固定攻角机械蝴蝶的最优安装角度分析
输出：净升力、阻力和升阻比随 α 的变化曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from dynamic_analysis import load_geometry, simulate_cycle, AERO_PARAMS

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output'
WEIGHT_mN = AERO_PARAMS['m_total'] * 9.81 * 1000

ALPHA_RANGE = np.arange(5, 86, 2)


def compute_net_lift_drag(sim):
    """返回符号平均（净升力）和平均阻力"""
    net_lift = np.mean(sim['F_lift']) * 1000
    avg_drag = np.mean(np.abs(sim['F_trans_drag'])) * 1000
    return net_lift, avg_drag


def scan_alpha(geo, alphas):
    """扫描安装角 α"""
    front_net, front_drag = [], []
    back_net, back_drag = [], []

    for a in alphas:
        p = AERO_PARAMS.copy()
        p['alpha_deg'] = float(a)
        f_sim = simulate_cycle(geo['Front'], p, n_points=500)
        b_sim = simulate_cycle(geo['Back'], p, n_points=500)

        fn, fd = compute_net_lift_drag(f_sim)
        bn, bd = compute_net_lift_drag(b_sim)
        front_net.append(fn)
        front_drag.append(fd)
        back_net.append(bn)
        back_drag.append(bd)

    return (np.array(front_net), np.array(front_drag),
            np.array(back_net), np.array(back_drag))


def plot_alpha_scan(alphas, front_net, front_drag, back_net, back_drag):
    """绘制安装角扫描图"""
    total_net = 2 * (front_net + back_net)
    total_drag = 2 * (front_drag + back_drag)

    ld_ratio = np.where(total_drag > 0, total_net / total_drag, 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Installation Angle Scan: Net Lift & Drag vs α (f={AERO_PARAMS["f"]}Hz, '
                 f'原始机构角度)',
                 fontsize=14, fontweight='bold')

    colors = {'front': '#1f77b4', 'back': '#2ca02c', 'total': '#d62728'}

    # ====== 左上：净升力（四翅合计）=======
    ax = axes[0, 0]
    ax.plot(alphas, total_net, 'o-', color=colors['total'], lw=2.5, markersize=5,
            label=r'$\Sigma F_\mathrm{net}$ (4 wings)')
    ax.axhline(y=WEIGHT_mN, color='gray', linestyle='--', lw=1.5,
               label=f'Weight = {WEIGHT_mN:.0f} mN')
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    ax.fill_between(alphas, 0, total_net, alpha=0.1, color=colors['total'])
    idx_max = np.argmax(total_net)
    ax.annotate(f'max={total_net[idx_max]:.0f} mN @ {alphas[idx_max]:.0f}°',
                xy=(alphas[idx_max], total_net[idx_max]),
                xytext=(alphas[idx_max]+5, total_net[idx_max]*1.05),
                fontsize=9, color=colors['total'],
                arrowprops=dict(arrowstyle='->', color=colors['total'], lw=1))
    ax.set_ylabel('Net Lift (mN) — signed mean')
    ax.set_xlabel('Installation angle α (°)')
    ax.set_title('Total Net Lift vs α')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ====== 右上：升阻比 ======
    ax = axes[0, 1]
    ax.plot(alphas, ld_ratio, 's-', color='#9467bd', lw=2.5, markersize=5)
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    idx_max_ld = np.argmax(ld_ratio)
    ax.annotate(f'max L/D={ld_ratio[idx_max_ld]:.3f} @ {alphas[idx_max_ld]:.0f}°',
                xy=(alphas[idx_max_ld], ld_ratio[idx_max_ld]),
                xytext=(alphas[idx_max_ld]+8, ld_ratio[idx_max_ld]*0.9),
                fontsize=9, color='#9467bd',
                arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1))
    ax.set_ylabel('L/D ratio')
    ax.set_xlabel('Installation angle α (°)')
    ax.set_title('Lift-to-Drag Ratio vs α')
    ax.grid(True, alpha=0.3)

    # ====== 左下：阻力 ======
    ax = axes[1, 0]
    ax.plot(alphas, total_drag, '^-', color='#ff7f0e', lw=2, markersize=5, label='Total drag')
    ax.plot(alphas, 2*front_drag, '--', color=colors['front'], lw=1.5, alpha=0.7, label='Front')
    ax.plot(alphas, 2*back_drag, '--', color=colors['back'], lw=1.5, alpha=0.7, label='Back')
    ax.set_ylabel('Drag (mN)')
    ax.set_xlabel('Installation angle α (°)')
    ax.set_title('Drag vs α')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ====== 右下：净升力 vs 绝对值平均对比 ======
    ax = axes[1, 1]
    ax.plot(alphas, front_net + back_net, 'o-', color=colors['front'], lw=2, markersize=4,
            label='Front net (per wing)')
    ax.plot(alphas, back_net, 's-', color=colors['back'], lw=2, markersize=4,
            label='Back net (per wing)')
    ax.axhline(y=0, color='k', linestyle='-', lw=0.5)
    ax.set_ylabel('Net Lift per wing (mN)')
    ax.set_xlabel('Installation angle α (°)')
    ax.set_title('Per-Wing Net Lift vs α')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    figures_dir = OUTPUT_DIR / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / 'alpha_scan.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()


def print_table(alphas, front_net, front_drag, back_net, back_drag):
    """打印结果表"""
    total_net = 2 * (front_net + back_net)
    total_drag = 2 * (front_drag + back_drag)
    ld = total_net / np.where(total_drag > 0, total_drag, 1)

    print(f"\n{'α(°)':>6} {'NetLift(mN)':>13} {'Drag(mN)':>11} {'L/D':>8} {'vs W':>6}")
    print("-" * 52)
    for i, a in enumerate(alphas):
        marker = " ★" if ld[i] == max(ld) else (" ●" if total_net[i] == max(total_net) else "  ")
        print(f"{a:6.0f} {total_net[i]:13.1f} {total_drag[i]:11.1f} "
              f"{ld[i]:8.3f} {total_net[i]/WEIGHT_mN:5.1f}x{marker}")

    print(f"\n★ = best L/D ({max(ld):.3f} @ {alphas[np.argmax(ld)]:.0f}°)")
    print(f"● = max net lift ({max(total_net):.0f} mN @ {alphas[np.argmax(total_net)]:.0f}°)")
    print(f"   Weight = {WEIGHT_mN:.0f} mN, 需要 >1x 才能起飞")


def main():
    print("=" * 70)
    print("INSTALLATION ANGLE (α) SCAN")
    print("=" * 70)
    print(f"  频率 = {AERO_PARAMS['f']} Hz")
    print(f"  机构原始角度，不做缩放（负值=下拍，正值=上拍）")
    print(f"  重量 = {WEIGHT_mN:.1f} mN")

    geo = load_geometry()

    print(f"\n[Scanning] α = {ALPHA_RANGE[0]:.0f}° ~ {ALPHA_RANGE[-1]:.0f}°,  "
          f"step={ALPHA_RANGE[1]-ALPHA_RANGE[0]:.0f}°")
    front_net, front_drag, back_net, back_drag = scan_alpha(geo, ALPHA_RANGE)

    print_table(ALPHA_RANGE, front_net, front_drag, back_net, back_drag)

    print(f"\n[Generating] alpha_scan.png ...")
    plot_alpha_scan(ALPHA_RANGE, front_net, front_drag, back_net, back_drag)

    print("\nDone!")


if __name__ == '__main__':
    main()
