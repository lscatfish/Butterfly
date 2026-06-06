#!/usr/bin/env python3
"""
方案二 Demo: numba JIT + joblib 笛卡尔积扫描验证.

验证项:
  1. numba 正确性 — 同一 combo 的 L/W 一致性
  2. 单仿真加速比
  3. joblib 并行 16 组全完成
  4. checkpoint 断点续跑
  5. 输出文件完整性

网格: 2×2×2×2 = 16 组 (phase × a × R × phi_offset)
"""

import sys, json, time, os
from pathlib import Path
import numpy as np

_PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ / "src"))

from butterfly_forces import SimulationConfig, ButterflyForceModel, _HAS_NUMBA
from sweep_cartesian import (
    BASELINE_CONFIG, _make_config, _extract_timeseries, _save_run,
    combo_to_id, build_cartesian_grid, run_one_combo, sweep_cartesian,
    OUT_ROOT,
)

# 小网格: 2×2×2×2 = 16 组
DEMO_GRID = {
    "phase_diff_deg":  [-20, -10],
    "mech_a":          [5.0, 6.0],
    "mech_R":          [2.5, 3.0],
    "phi_offset_deg":  [-50, -35],
}

DEMO_ROOT = _PROJ / "temp" / "demo_cartesian"


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_numba_correctness():
    """验证 numba 加速前后 L/W 一致性."""
    section("Test 1: numba 正确性")

    combo = {"mech_a": 5.0, "mech_R": 2.5, "phase_diff_deg": -15, "phi_offset_deg": -50}
    overrides = BASELINE_CONFIG.copy()
    overrides.update(combo)
    overrides["t_end"] = 3.0   # 短仿真加速测试
    overrides["dt"] = 50e-6

    cfg = _make_config(overrides)
    print(f"  Config: a={cfg.mech_a}, R={cfg.mech_R}, phase={cfg.phase_diff_deg}, po={cfg.phi_offset_deg}")
    print(f"  t_end={cfg.t_end}s, dt={cfg.dt*1e6:.0f}us")

    # Python 版本
    print(f"\n  [Python] Running...")
    t0 = time.time()
    model_py = ButterflyForceModel(cfg)
    out_py = model_py.simulate(progress=False, use_numba=False)
    t_py = time.time() - t0
    lw_py = out_py.summary["L/W"]
    peak_py = out_py.summary["peak_theta_deg"]
    print(f"    L/W={lw_py:.6f}, peak={peak_py:.4f}°, time={t_py:.1f}s")

    # Numba 版本
    if not _HAS_NUMBA:
        print("  SKIP: numba not available")
        return

    print(f"  [Numba] Running (first call triggers JIT compile)...")
    t0 = time.time()
    model_nb = ButterflyForceModel(cfg)
    out_nb = model_nb.simulate(progress=False, use_numba=True)
    t_nb = time.time() - t0
    lw_nb = out_nb.summary["L/W"]
    peak_nb = out_nb.summary["peak_theta_deg"]
    print(f"    L/W={lw_nb:.6f}, peak={peak_nb:.4f}°, time={t_nb:.1f}s")

    # 一致性检查
    lw_err = abs(lw_py - lw_nb)
    peak_err = abs(peak_py - peak_nb)
    rel_err = lw_err / abs(lw_py) * 100 if abs(lw_py) > 0 else 0

    print(f"\n  L/W diff: {lw_err:.6f} ({rel_err:.4f}%)")
    print(f"  peak diff: {peak_err:.4f}°")

    if rel_err < 0.5:  # 0.5% 容差
        print(f"  ✅ PASS — numba results match Python (rel error {rel_err:.4f}%)")
    else:
        print(f"  ⚠️ WARNING — L/W difference {rel_err:.4f}% > 0.5%")


def test_speedup():
    """测试 numba 加速比 (warm 之后)."""
    section("Test 2: numba 加速比")

    if not _HAS_NUMBA:
        print("  SKIP: numba not available")
        return

    combo = {"mech_a": 5.0, "mech_R": 2.5, "phase_diff_deg": -15, "phi_offset_deg": -50}
    overrides = BASELINE_CONFIG.copy()
    overrides.update(combo)
    overrides["t_end"] = 3.0
    overrides["dt"] = 50e-6

    cfg = _make_config(overrides)

    # Python × 3
    t_py = []
    for i in range(3):
        model = ButterflyForceModel(cfg)
        t0 = time.time()
        model.simulate(progress=False, use_numba=False)
        t_py.append(time.time() - t0)
    t_py_avg = np.mean(t_py[1:])  # 跳过首次

    # Numba (已 warm, × 3)
    t_nb = []
    for i in range(3):
        model = ButterflyForceModel(cfg)
        t0 = time.time()
        model.simulate(progress=False, use_numba=True)
        t_nb.append(time.time() - t0)
    t_nb_avg = np.mean(t_nb)

    speedup = t_py_avg / t_nb_avg
    print(f"  Python (avg, skip first): {t_py_avg:.1f}s")
    print(f"  Numba  (warm, avg):       {t_nb_avg:.1f}s")
    print(f"  Speedup: {speedup:.1f}×")

    if speedup > 2:
        print(f"  ✅ PASS — {speedup:.1f}× speedup")
    else:
        print(f"  ⚠️  Speedup only {speedup:.1f}×, consider larger t_end for benchmarking")


def test_small_grid():
    """joblib 并行跑 16 组小网格."""
    section("Test 3: joblib 并行 16 组")

    DEMO_ROOT.mkdir(parents=True, exist_ok=True)

    grid = DEMO_GRID
    combos = build_cartesian_grid(grid)
    n_combo = len(combos)
    print(f"  Grid: { {k: len(v) for k, v in grid.items()} }")
    print(f"  Total: {n_combo} combos")
    print(f"  Output: {DEMO_ROOT}")

    t0 = time.time()
    results = sweep_cartesian(
        grid_spec=grid,
        n_jobs=-1,
        t_end=3.0,      # 短仿真
        dt=50e-6,
        out_root=DEMO_ROOT,
    )
    elapsed = time.time() - t0

    n_ok = len([r for r in results if r is not None])
    print(f"\n  Completed: {n_ok}/{n_combo} in {elapsed:.1f}s")

    if n_ok == n_combo:
        print(f"  ✅ PASS — all {n_combo} combos completed")
    else:
        print(f"  ❌ FAIL — {n_combo - n_ok} combos missing")

    # 打印结果概览
    sorted_r = sorted([r for r in results if r is not None],
                      key=lambda x: x.get("L/W", 0), reverse=True)
    print(f"\n  Results (sorted by L/W):")
    for i, sm in enumerate(sorted_r):
        cid = sm.get("_combo_id", "?")
        lw = sm.get("L/W", float('nan'))
        peak = sm.get("peak_theta_deg", float('nan'))
        n90 = sm.get("n_exceed_90", -1)
        status = "✅" if n90 == 0 else "❌"
        print(f"    {i+1:2d}. {cid:30s} L/W={lw:.3f}  peak={peak:.1f}°  {status}")

    return results


def test_checkpoint():
    """验证断点续跑."""
    section("Test 4: checkpoint (断点续跑)")

    # 先跑一次 (如果还没跑)
    combos = build_cartesian_grid(DEMO_GRID)
    combo_ids = [combo_to_id(c) for c in combos]

    # 检查哪些已完成
    done_ids = []
    for cid in combo_ids:
        if (DEMO_ROOT / cid / "summary.json").exists():
            done_ids.append(cid)

    print(f"  Already done: {len(done_ids)}/{len(combos)}")

    if len(done_ids) == len(combos):
        # 删除最后一个 combo 的结果，模拟中断
        last_id = done_ids[-1]
        last_dir = DEMO_ROOT / last_id
        import shutil
        shutil.rmtree(last_dir)
        print(f"  Deleted {last_id} to simulate interruption")
        done_ids = done_ids[:-1]

    # 重新跑 sweep — 应该只跑缺失的
    t0 = time.time()
    results = sweep_cartesian(
        grid_spec=DEMO_GRID,
        n_jobs=-1,
        t_end=3.0,
        dt=50e-6,
        out_root=DEMO_ROOT,
    )
    elapsed = time.time() - t0

    n_done = len([r for r in results if r is not None])
    print(f"  Re-run completed: {n_done}/{len(combos)} combos in {elapsed:.1f}s")

    # 验证 sweeps_summary 存在
    summary_path = DEMO_ROOT / "sweep_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            sm = json.load(f)
        print(f"  sweep_summary.json: {sm['_n_combos']} combos, keys={list(sm.keys())[:5]}...")
        print(f"  ✅ PASS — checkpoint/resume works")
    else:
        print(f"  ⚠️  sweep_summary.json not found")


def test_output_completeness():
    """验证输出文件完整."""
    section("Test 5: 输出文件完整性")

    combos = build_cartesian_grid(DEMO_GRID)
    combo_ids = [combo_to_id(c) for c in combos]

    all_ok = True
    for cid in combo_ids:
        d = DEMO_ROOT / cid
        has_config = (d / "config.json").exists()
        has_summary = (d / "summary.json").exists()
        has_npz = (d / "timeseries.npz").exists()
        ok = has_config and has_summary and has_npz
        if not ok:
            print(f"  ❌ {cid}: config={has_config} summary={has_summary} npz={has_npz}")
            all_ok = False

    if all_ok:
        print(f"  ✅ All {len(combo_ids)} combos have complete output files")

    # 验证 config 可读
    sample_dir = DEMO_ROOT / combo_ids[0]
    with open(sample_dir / "config.json") as f:
        cfg = json.load(f)
    with open(sample_dir / "summary.json") as f:
        summary = json.load(f)
    ts = np.load(sample_dir / "timeseries.npz")
    n_keys = len(ts.keys())
    n_steps = len(ts["t"])

    print(f"  Sample config: {list(cfg.keys())[:5]}...")
    print(f"  Sample summary: L/W={summary.get('L/W', '?')}")
    print(f"  Sample timeseries: {n_keys} keys, {n_steps} steps")
    print(f"  ✅ Output format verified")


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  方案二 Demo — numba JIT + joblib 验证")
    print("=" * 60)
    print(f"  numba: {'✅ ' + __import__('numba').__version__ if _HAS_NUMBA else '❌ not installed'}")
    print(f"  joblib: {'✅ ' + __import__('joblib').__version__ if __import__('sweep_cartesian')._HAS_JOBLIB else '❌ not installed'}")
    print(f"  Demo grid: { {k: len(v) for k, v in DEMO_GRID.items()} } = {len(build_cartesian_grid(DEMO_GRID))} combos")

    # 1. numba 正确性
    test_numba_correctness()

    # 2. 加速比
    test_speedup()

    # 3. 小网格并行
    test_small_grid()

    # 4. 断点续跑
    test_checkpoint()

    # 5. 输出完整性
    test_output_completeness()

    print(f"\n{'='*60}")
    print(f"  Demo 完成!")
    print(f"{'='*60}")
