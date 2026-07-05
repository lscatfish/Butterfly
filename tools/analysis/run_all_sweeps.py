#!/usr/bin/env python3
"""批量运行所有参数的真正单变量偏离扫描.

每次只改一个参数, 其他参数锁死在 DESIGN_v69 (BASELINE_CONFIG).
运行前先清掉旧的聚合摘要目录, 避免混淆.
"""
import sys, time, shutil
from pathlib import Path
_PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ))
from src.aero.stability_analysis import sweep_parameter, OUT_ROOT

PARAMS = [
    "alpha_front_deg",
    "alpha_back_deg",
    "phase_diff_deg",
    "mech_a",
    "mech_R",
    "phi_offset_deg",
    "k_clap",
]

# 清掉旧的聚合摘要目录 (真正的扫描会重建)
for param in PARAMS:
    old_dir = OUT_ROOT / f"sweep_{param}"
    if old_dir.exists():
        print(f"[clean] removing old {old_dir}")
        shutil.rmtree(old_dir)

t0 = time.time()
for i, param in enumerate(PARAMS):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(PARAMS)}] Sweeping {param} ...")
    print(f"{'='*60}")
    t1 = time.time()
    result = sweep_parameter(param, t_end=5.0, dt=50e-6)
    elapsed = time.time() - t1
    print(f"[{param}] done in {elapsed:.0f}s, {result['_n']} values")

total = time.time() - t0
print(f"\n{'='*60}")
print(f"ALL SWEEPS DONE. Total time: {total/60:.1f} min")
print(f"{'='*60}")
