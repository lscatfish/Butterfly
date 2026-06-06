#!/usr/bin/env python3
"""v6.4 关键验证: mech 参数 bug 修复后，验证 R1/R2 最佳结果是否一致"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from butterfly_forces import SimulationConfig, ButterflyForceModel, scan_parameters

OUT_DIR = Path(__file__).parent / "v64_optimize"
OUT_DIR.mkdir(exist_ok=True)

# 基于 R1/R2 最佳: α_f=68, α_b=5, phase=-15
# R4 10s 已跑出 L/W=2.069，现在验证：
# 1) 不同 R 是否真的有影响
# 2) 确认 a=7.0 最优

print("=" * 70)
print("验证 mech 参数 (bug修复后, R/b/l 真正生效)")
print("α_f=68, α_b=5, phase=-15")
print("=" * 70)

results = scan_parameters(
    SimulationConfig(alpha_front_deg=68, alpha_back_deg=5,
                     phase_diff_deg=-15, t_end=3.0, dt=50e-6),
    {
        "mech_a": [7.0, 7.5, 7.92],
        "mech_R": [2.0, 2.5],
    },
    t_end=3.0, dt=50e-6, progress=True,
)

with open(OUT_DIR / "verify_mech_fixed.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n验证结果 (R 修复后):")
for r in results:
    mark = "⭐" if r["L/W"] > 1.8 else " "
    print(f"  {mark} a={r['mech_a']} R={r['mech_R']}  "
          f"L/W={r['L/W']:.3f}  peak={r['peak_deg']:.1f}°  n90={r['n90']}")

print("\nDone. Results saved.")
