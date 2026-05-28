#!/usr/bin/env python3
"""快速绘制 SolidWorks 导出的原始 segment，检查轮廓连续性"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = Path(__file__).parent

# 颜色表
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown']

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, wing_name in zip(axes, ['front', 'back']):
    csv_path = DATA_DIR / f'wing_{wing_name}.csv'
    df = pd.read_csv(csv_path)
    
    # 绘制每个 segment
    for seg_idx in sorted(df['SegmentIndex'].unique()):
        sub = df[df['SegmentIndex'] == seg_idx]
        x = sub['X'].values
        y = sub['Y'].values
        c = colors[seg_idx % len(colors)]
        
        ax.plot(x, y, color=c, lw=2, label=f'Seg {seg_idx} ({len(sub)} pts)')
        ax.scatter(x[0], y[0], color=c, s=80, marker='o', zorder=5, edgecolors='black')
        ax.scatter(x[-1], y[-1], color=c, s=80, marker='s', zorder=5, edgecolors='black')
        
        # 标注端点坐标
        ax.annotate(f'S{seg_idx}s\n({x[0]:.0f},{y[0]:.0f})', 
                    xy=(x[0], y[0]), xytext=(10, 10), textcoords='offset points',
                    fontsize=8, color=c, fontweight='bold')
        ax.annotate(f'S{seg_idx}e\n({x[-1]:.0f},{y[-1]:.0f})', 
                    xy=(x[-1], y[-1]), xytext=(10, -20), textcoords='offset points',
                    fontsize=8, color=c, fontweight='bold')
    
    # 绘制转轴
    axis_df = pd.read_csv(DATA_DIR / 'wing_axis.csv')
    p0 = axis_df[axis_df['Type']==0][['X','Y']].values[0]
    p1 = axis_df[axis_df['Type']==1][['X','Y']].values[0]
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k--', lw=2, label='Axis')
    ax.scatter(*p0, color='black', s=100, marker='*', zorder=6)
    ax.scatter(*p1, color='black', s=100, marker='*', zorder=6)
    ax.annotate('Axis start', xy=p0, xytext=(10, 10), textcoords='offset points', fontsize=9)
    ax.annotate('Axis end', xy=p1, xytext=(10, -20), textcoords='offset points', fontsize=9)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(f'{wing_name.upper()} 原始 Segment 分布')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = DATA_DIR / 'raw_segments_plot.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close()
