#!/usr/bin/env python3
"""批量运行所有6个参数的单变量偏离扫描."""
import sys, time
from pathlib import Path
_PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ / "src"))
from stability_analysis import sweep_parameter

PARAMS = [
    "alpha_front_deg",
    "alpha_back_deg",
    "phase_diff_deg",
    "mech_a",
    "mech_R",
    "phi_offset_deg",
]

t0 = time.time()
for i, param in enumerate(PARAMS):
    print(f"\n{'='*60}")
    print(f"[{i+1}/6] Sweeping {param} ...")
    print(f"{'='*60}")
    t1 = time.time()
    result = sweep_parameter(param, t_end=5.0, dt=50e-6)
    elapsed = time.time() - t1
    print(f"[{param}] done in {elapsed:.0f}s, {result['_n']} values")

total = time.time() - t0
print(f"\n{'='*60}")
print(f"ALL SWEEPS DONE. Total time: {total/60:.1f} min")
print(f"{'='*60}")
