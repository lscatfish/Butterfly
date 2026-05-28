#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent

fig, ax = plt.subplots(figsize=(12, 10))

# 绘制前翅
front = pd.read_csv(DATA_DIR / 'wing_front.csv')
ax.plot(front['X'], front['Y'], 'b-', lw=1.5, label='WingFront')
ax.fill(front['X'], front['Y'], alpha=0.2, color='blue')

# 绘制后翅
back = pd.read_csv(DATA_DIR / 'wing_back.csv')
ax.plot(back['X'], back['Y'], 'g-', lw=1.5, label='WingBack')
ax.fill(back['X'], back['Y'], alpha=0.2, color='green')

# 绘制转轴
axis = pd.read_csv(DATA_DIR / 'wing_axis.csv')
p0 = axis[axis['Type'] == 0][['X', 'Y']].values[0]
p1 = axis[axis['Type'] == 1][['X', 'Y']].values[0]
ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r--', lw=2, label='Axis')

ax.set_aspect('equal')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_title('Wing Front & Back (Raw XY from CSV)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR / 'wings_plot.png', dpi=200, bbox_inches='tight')
print('Saved: wings_plot.png')
plt.close()
