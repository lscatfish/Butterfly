#!/usr/bin/env python3
"""
v6.4 静态参数精化扫描 — 基于 v6.3 最佳结果细化

v6.3 最佳: α_f=60/α_b=8, L/W=0.943(3s)/1.033(10s)
趋势: α_f↑ + α_b↓ → L/W↑

本扫描扩展:
1. α_f: 58-70 (v6.3 只到 60)
2. α_b: 5-8 (v6.3 最低 8)
3. 相位差: 0, ±15, ±30
4. 机构参数: a ∈ [7.5, 7.92, 8.5]
"""
import sys, json, itertools, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from butterfly_forces import SimulationConfig, ButterflyForceModel, scan_parameters

OUT_DIR = Path(__file__).parent / "v64_optimize"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Round 1: α_f/α_b 精细扫描 (最敏感参数)
# ============================================================
print("=" * 70)
print("Round 1: α_f/α_b 精细扫描")
print("=" * 70)

cfg_base = SimulationConfig(t_end=3.0, dt=50e-6)

# v6.3 完整结果供参考
v63_reference = {
    (60,8):  0.943, (60,10): 0.903, (60,12): 0.859,
    (55,8):  0.859, (55,10): 0.814, (55,12): 0.758,
    (50,8):  0.753, (50,10): 0.703, (48,8):  0.687,
    (45,8):  0.626, (45,10): 0.564,
}

results_r1 = scan_parameters(cfg_base, {
    "alpha_front_deg": [58, 60, 62, 65, 68],
    "alpha_back_deg":  [5, 6, 7, 8, 10],
}, t_end=3.0, dt=50e-6, progress=True)

# Save
with open(OUT_DIR / "round1_alpha_scan.json", "w") as f:
    json.dump(results_r1, f, indent=2, ensure_ascii=False)

print(f"\nRound 1 top results:")
for r in results_r1[:10]:
    marker = "⭐" if r["L/W"] > 0.95 else ("✓" if r["L/W"] > 0.9 else " ")
    prev = v63_reference.get((r["alpha_front_deg"], r["alpha_back_deg"]))
    delta = f" (v6.3: {prev:.3f})" if prev else ""
    print(f"  {marker} α_f={r['alpha_front_deg']}° α_b={r['alpha_back_deg']}°  "
          f"L/W={r['L/W']:.3f}  peak={r['peak_deg']:.1f}°  n90={r['n90']}"
          f"{delta}")

# ============================================================
# Round 2: 相位差扫描 (用 Round 1 最佳)
# ============================================================
best_r1 = results_r1[0]
best_af = best_r1["alpha_front_deg"]
best_ab = best_r1["alpha_back_deg"]
print(f"\n{'='*70}")
print(f"Round 2: 相位差扫描 (α_f={best_af}, α_b={best_ab})")
print("=" * 70)

results_r2 = scan_parameters(
    SimulationConfig(alpha_front_deg=best_af, alpha_back_deg=best_ab, t_end=3.0, dt=50e-6),
    {"phase_diff_deg": [-30, -15, -5, 0, 5, 15, 30, 45, 60, 90, 120, 180]},
    t_end=3.0, dt=50e-6, progress=True,
)

with open(OUT_DIR / "round2_phase_scan.json", "w") as f:
    json.dump(results_r2, f, indent=2, ensure_ascii=False)

print(f"\nRound 2 results:")
for r in results_r2:
    marker = "⭐" if r["L/W"] > 0.95 else ("✓" if r["L/W"] > 0 else "✗")
    print(f"  {marker} phase={r['phase_diff_deg']:>4}°  "
          f"L/W={r['L/W']:.3f}  peak={r['peak_deg']:.1f}°  n90={r['n90']}")

# ============================================================
# Round 3: 机构参数扫描 (影响摆幅和急回比)
# ============================================================
best_r2 = results_r2[0] if results_r2 else results_r1[0]
best_phase = best_r2.get("phase_diff_deg", 0)
print(f"\n{'='*70}")
print(f"Round 3: 机构参数扫描 (α_f={best_af}, α_b={best_ab}, phase={best_phase}°)")
print("=" * 70)

results_r3 = scan_parameters(
    SimulationConfig(alpha_front_deg=best_af, alpha_back_deg=best_ab,
                     phase_diff_deg=best_phase, t_end=3.0, dt=50e-6),
    {"mech_a": [7.0, 7.5, 7.92, 8.5, 9.0],
     "mech_R": [2.0, 2.25, 2.5],
     "mech_b": [6.97, 8.0],
     "mech_l": [7.0, 8.0, 9.0]},
    t_end=3.0, dt=50e-6, progress=True,
)

best_r3 = results_r3[0] if results_r3 else {}

with open(OUT_DIR / "round3_mech_scan.json", "w") as f:
    json.dump(results_r3, f, indent=2, ensure_ascii=False)

print(f"\nRound 3 top results:")
for r in results_r3[:10]:
    marker = "⭐" if r["L/W"] > 1.5 else ("✓" if r["L/W"] > 0 else "✗")
    print(f"  {marker} a={r.get('mech_a','?'):.2f} R={r.get('mech_R','?'):.2f} "
          f"b={r.get('mech_b','?'):.2f} l={r.get('mech_l','?'):.2f}  "
          f"L/W={r['L/W']:.3f}  peak={r['peak_deg']:.1f}°  n90={r['n90']}")

# ============================================================
# Round 4: 最佳参数 10s 长稳验证
# ============================================================
best_overall = best_r3 if best_r3 else results_r1[0]
final_af = best_af if "alpha_front_deg" not in best_overall else best_overall.get("alpha_front_deg", best_af)
final_ab = best_ab if "alpha_back_deg" not in best_overall else best_overall.get("alpha_back_deg", best_ab)
final_phase = best_phase
final_a = best_overall.get("mech_a", 7.0)
final_R = best_overall.get("mech_R", 2.0)
final_b = best_overall.get("mech_b", 6.97)
final_l = best_overall.get("mech_l", 8.0)

print(f"\n{'='*70}")
print(f"Round 4: 10s 长稳验证 (α_f={final_af}, α_b={final_ab}, phase={final_phase}°, a={final_a}, R={final_R}, b={final_b}, l={final_l})")
print("=" * 70)

cfg_final = SimulationConfig(
    alpha_front_deg=final_af, alpha_back_deg=final_ab,
    phase_diff_deg=final_phase,
    mech_a=final_a, mech_R=final_R,
    mech_b=final_b, mech_l=final_l,
    t_end=10.0, dt=50e-6,
)
model = ButterflyForceModel(cfg_final)
out = model.simulate(progress=True)
s = out.summary

print(f"\n  Status: {'✅ STABLE' if s['n_exceed_90']==0 else '❌ DIVERGED'}")
print(f"  L/W (10s, body): {s['L/W']:.3f}")
print(f"  Fz_body: {s['avg_Fz_body_mN']:+.0f} mN")
print(f"  Fz_world: {s['avg_Fz_world_mN']:+.0f} mN")
print(f"  Peak θ: {s['peak_theta_deg']:.1f}°")
print(f"  n90: {s['n_exceed_90']}")

# Save summary
summary = {
    "version": "v6.4",
    "date": time.strftime("%Y-%m-%d %H:%M"),
    "best_v63": {"alpha_front": 60, "alpha_back": 8, "L/W_3s": 0.943, "L/W_10s": 1.033},
    "round1_best": best_r1,
    "round2_best": best_r2,
    "round3_best": best_r3,
    "final_params": {
        "alpha_front_deg": final_af, "alpha_back_deg": final_ab,
        "phase_diff_deg": final_phase,
        "mech_a": final_a, "mech_R": final_R,
        "mech_b": final_b, "mech_l": final_l,
    },
    "final_10s": {
        "L/W": s["L/W"], "Fz_body_mN": s["avg_Fz_body_mN"],
        "Fz_world_mN": s["avg_Fz_world_mN"],
        "peak_deg": s["peak_theta_deg"], "n90": s["n_exceed_90"],
    },
}
with open(OUT_DIR / "v64_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nDone. Results saved to {OUT_DIR}/")
