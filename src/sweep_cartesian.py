#!/usr/bin/env python3
"""
笛卡尔积参数扫描 — joblib 并行 + numba JIT 加速.

与 stability_analysis.py (方案一单变量) 互补:
  - 方案一: 控制变量法, 每参数扫 6-9 个值
  - 方案二 (本模块): 多参数全组合, 粗网格笛卡尔积, 捕获交互效应

输出结构:
  temp/stability/sweep_cartesian/
  ├── checkpoint.json          # {done: [combo_id, ...], pending: [...], current: [...]}
  ├── sweep_summary.json       # 汇总所有 combo 标量指标
  └── <combo_id>/
      ├── config.json
      ├── summary.json
      └── timeseries.npz

使用:
  python src/sweep_cartesian.py --n-jobs 8
"""

import sys, json, time, os, traceback
from pathlib import Path
from itertools import product
import numpy as np

_PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ / "src"))
from butterfly_forces import SimulationConfig, ButterflyForceModel, _HAS_NUMBA

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False
    Parallel = None
    delayed = None

OUT_ROOT = _PROJ / "temp" / "stability" / "sweep_cartesian"

# ============================================================
# v6.5 基线参数 — 与 stability_analysis.BASELINE_CONFIG 一致
# ============================================================
BASELINE_CONFIG = dict(
    alpha_front_deg=68, alpha_back_deg=5,
    phase_diff_deg=-15, mech_a=6.0, mech_R=2.5,
    phi_offset_deg=-50.84, rotation='cw',
    f=15.0, rho=1.225, m_total=0.020, I_yy=3e-5, d_cg=0.015,
    x_front=0.025, x_back=-0.025,
    dt=50e-6, t_end=5.0, theta0_deg=0.0, steady_start=3.0,
    k_3d=0.7, C_rot=1.5, r_rot=0.5, k_clap=1.3, c_damp=5e-4,
)

# ============================================================
# 缺省笛卡尔积网格 — 9 参数粗网格 (2×2×2×4×3×3×3×2×2 = 3456 组)
# 编辑此字典即可自定义扫描范围和分辨率
# ============================================================
DEFAULT_GRID = {
    "alpha_front_deg":  [60, 68],
    "alpha_back_deg":   [3, 5],
    "phase_diff_deg":   [-20, -10],
    "mech_a":           [6, 8, 10, 12],
    "mech_R":           [2.5, 3.0, 3.25],
    "phi_offset_deg":   [-50, -40, -30],
    "f":                [13, 15, 17],
    "c_damp":           [1e-4, 5e-4],
    "rotation":         ["cw", "ccw"],
}

PARAM_SHORT = {
    "alpha_front_deg": "af", "alpha_back_deg": "ab",
    "phase_diff_deg": "ph", "mech_a": "a",
    "mech_R": "R", "phi_offset_deg": "po",
    "f": "f", "c_damp": "cd", "rotation": "rot",
}


# ---- combo_id 编解码 ----
def combo_to_id(combo: dict) -> str:
    """将参数字典编码为紧凑文件夹名 (可逆)."""
    parts = []
    for k in sorted(combo.keys()):
        v = combo[k]
        short = PARAM_SHORT.get(k, k)
        if isinstance(v, float):
            s = f"{v:.4f}".rstrip('0').rstrip('.').replace('.', 'p').replace('-', 'n')
        elif isinstance(v, int):
            s = str(v).replace('-', 'n')
        else:
            s = str(v).replace('.', 'p').replace('-', 'n')
        parts.append(f"{short}{s}")
    return "_".join(parts)


def id_to_combo(combo_id: str, grid_keys: list) -> dict:
    """从文件夹名还原参数字典."""
    combo = {}
    for part in combo_id.split("_"):
        for long, short in PARAM_SHORT.items():
            if part.startswith(short):
                val_str = part[len(short):].replace('p', '.').replace('n', '-')
                try:
                    if '.' in val_str:
                        combo[long] = float(val_str)
                    else:
                        combo[long] = int(val_str)
                except ValueError:
                    combo[long] = val_str
                break
    return combo


# ---- 网格生成 ----
def build_cartesian_grid(grid_spec: dict = None) -> list:
    """从网格定义生成组合列表.

    Args:
        grid_spec: {param_name: [values]}, 默认 DEFAULT_GRID.

    Returns:
        [{param: value}, ...] 所有笛卡尔积组合.
    """
    grid = grid_spec or DEFAULT_GRID
    keys = list(grid.keys())
    values_list = list(grid.values())
    combos = [dict(zip(keys, combo)) for combo in product(*values_list)]
    return combos


# ---- 单组合运行 (joblib worker) ----
def _make_config(overrides: dict) -> SimulationConfig:
    """从 BASELINE_CONFIG + overrides 构建 SimulationConfig."""
    d = BASELINE_CONFIG.copy()
    d.update(overrides)
    for k in ["rotation"]:
        d[k] = str(d[k])
    return SimulationConfig(**{k: v for k, v in d.items()
                                if k in SimulationConfig.__dataclass_fields__})


def _extract_timeseries(out) -> dict:
    """从 SimulationOutput 提取全时程数据 (与 stability_analysis 版本一致)."""
    half = len(out.t) // 2

    Fz_body = np.zeros(len(out.t))
    Fx_body = np.zeros(len(out.t))
    Fz_world = np.zeros(len(out.t))
    for wn in ["FL", "FR", "BL", "BR"]:
        w = out.wings[wn]
        Fz_body += w.force_body[:, 2]
        Fx_body += w.force_body[:, 0]
        Fz_world += w.force_world[:, 2]

    Fz_f_total = out.wings["FL"].force_body[:, 2] + out.wings["FR"].force_body[:, 2]
    Fz_b_total = out.wings["BL"].force_body[:, 2] + out.wings["BR"].force_body[:, 2]
    M_aero = -out.config.x_front * Fz_f_total - out.config.x_back * Fz_b_total
    M_grav = -out.config.m_total * out.config.g * out.config.d_cg * np.sin(out.theta_p)
    M_damp = -out.config.c_damp * out.theta_dot

    return {
        "t": out.t,
        "theta_p": out.theta_p,
        "theta_dot": out.theta_dot,
        "theta_ddot": out.theta_ddot,
        "Fz_body_total": Fz_body,
        "Fx_body_total": Fx_body,
        "Fz_world_total": Fz_world,
        "M_aero": M_aero,
        "M_grav": M_grav,
        "M_damp": M_damp,
        **{f"{wn}_Fz_body": out.wings[wn].force_body[:, 2] for wn in ["FL", "FR", "BL", "BR"]},
        **{f"{wn}_Fx_body": out.wings[wn].force_body[:, 0] for wn in ["FL", "FR", "BL", "BR"]},
        **{f"{wn}_alpha_eff": out.wings[wn].alpha_eff_deg for wn in ["FL", "FR", "BL", "BR"]},
        **{f"{wn}_C_L": out.wings[wn].C_L for wn in ["FL", "FR", "BL", "BR"]},
        **{f"{wn}_C_D": out.wings[wn].C_D for wn in ["FL", "FR", "BL", "BR"]},
        **{f"{wn}_phi": out.wings[wn].phi for wn in ["FL", "FR", "BL", "BR"]},
    }


def _extract_summary(out, ts_dict: dict) -> dict:
    """从 SimulationOutput + timeseries 提取标量摘要."""
    s = out.summary
    half = len(out.t) // 2
    weight_mN = out.config.m_total * out.config.g * 1000

    def _stats(arr, suffix=""):
        return {
            f"mean{suffix}": float(np.mean(arr[half:])),
            f"std{suffix}": float(np.std(arr[half:])),
            f"peak{suffix}": float(np.max(np.abs(arr))),
            f"peak_steady{suffix}": float(np.max(np.abs(arr[half:]))),
        }

    return {
        "L/W": s["L/W"],
        "L/W_body": s.get("L/W_body", float('nan')),
        "peak_theta_deg": s["peak_theta_deg"],
        "n_exceed_90": s["n_exceed_90"],
        "weight_mN": weight_mN,
        **_stats(ts_dict["Fz_body_total"] * 1000, "_Fz_body_mN"),
        **_stats(ts_dict["Fz_world_total"] * 1000, "_Fz_world_mN"),
        **_stats(ts_dict["Fx_body_total"] * 1000, "_Fx_body_mN"),
        **_stats(ts_dict["M_aero"] * 1e6, "_M_aero_uNm"),
        **_stats(ts_dict["M_grav"] * 1e6, "_M_grav_uNm"),
        **_stats(ts_dict["M_damp"] * 1e6, "_M_damp_uNm"),
        **_stats(np.abs(out.theta_dot), "_abs_thetadot_rads"),
        **_stats(np.abs(out.theta_ddot), "_abs_thetaddot_rads2"),
        **_stats(out.wings["FL"].alpha_eff_deg, "_alpha_eff_FL_deg"),
        **_stats(out.wings["BL"].alpha_eff_deg, "_alpha_eff_BL_deg"),
        **_stats(out.wings["FL"].C_L, "_CL_FL"),
        **_stats(out.wings["FL"].C_D, "_CD_FL"),
    }


def _save_run(out_dir: Path, out, ts_dict: dict, summary_extra: dict = None):
    """保存一次运行的数据 (config + summary + timeseries)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = out.config
    config_dict = {f.name: getattr(cfg, f.name)
                   for f in cfg.__dataclass_fields__.values()}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    summary = _extract_summary(out, ts_dict)
    if summary_extra:
        summary.update(summary_extra)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    np.savez_compressed(out_dir / "timeseries.npz", **ts_dict)


def run_one_combo(combo: dict, out_root: Path = None,
                  base_overrides: dict = None,
                  t_end: float = 5.0, dt: float = 50e-6,
                  verbose: bool = False) -> dict:
    """运行单个参数组合, 保存结果到磁盘. (joblib worker)

    Args:
        combo: {param: value} 参数覆盖字典.
        out_root: 输出根目录.
        base_overrides: 额外覆盖基线 (如 t_end, dt).
        t_end, dt: 仿真参数.
        verbose: 是否输出详细信息 (worker 默认关闭).

    Returns:
        summary dict (含 _combo_id, _combo 等元信息).
    """
    out_root = out_root or OUT_ROOT
    combo_id = combo_to_id(combo)
    out_dir = out_root / combo_id

    # 已完成则跳过
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            sm = json.load(f)
        sm["_combo_id"] = combo_id
        sm["_combo"] = combo
        return sm

    overrides = (base_overrides or {}).copy()
    overrides.update(combo)
    overrides.setdefault("t_end", t_end)
    overrides.setdefault("dt", dt)

    # 抑制机构运动学警告 (worker 内部)
    import io, contextlib
    _null = io.StringIO()
    try:
        with contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
            cfg = _make_config(overrides)
            model = ButterflyForceModel(cfg)
            out = model.simulate(progress=False)
    except Exception:
        # 重试: 不抑制输出以便看到错误
        cfg = _make_config(overrides)
        model = ButterflyForceModel(cfg)
        out = model.simulate(progress=False)

    ts = _extract_timeseries(out)

    extra = {"_combo_id": combo_id, "_combo": combo}
    _save_run(out_dir, out, ts, extra)

    sm = _extract_summary(out, ts)
    sm["_combo_id"] = combo_id
    sm["_combo"] = combo
    sm["L/W"] = out.summary["L/W"]
    sm["peak_theta_deg"] = out.summary["peak_theta_deg"]
    sm["n_exceed_90"] = out.summary["n_exceed_90"]
    return sm


# ---- 主入口: 笛卡尔积并行扫描 ----
def sweep_cartesian(grid_spec: dict = None,
                    base_overrides: dict = None,
                    n_jobs: int = -1,
                    t_end: float = 5.0,
                    dt: float = 50e-6,
                    out_root: Path = None) -> list:
    """笛卡尔积参数扫描 — joblib 并行.

    Args:
        grid_spec: {param: [values]} 网格定义, 默认 DEFAULT_GRID.
        base_overrides: 额外基线覆盖.
        n_jobs: joblib 并行数, -1 = 全部核心.
        t_end, dt: 仿真参数.
        out_root: 输出根目录.

    Returns:
        [{summary}] 所有组合的结果.
    """
    out_root = out_root or OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    grid = grid_spec or DEFAULT_GRID
    combos = build_cartesian_grid(grid)
    n_total = len(combos)

    # 单行摘要
    grid_info = ", ".join(f"{k}={len(v)}" for k, v in grid.items())
    nb_tag = "numba" if _HAS_NUMBA else "python"
    print(f"[Cartesian] {grid_info} → {n_total} combos | {nb_tag} × {n_jobs} workers")
    print(f"[Cartesian] Output: {out_root}")

    # Warm-up: 串行跑第一个 combo 触发 numba 缓存
    if _HAS_NUMBA and n_total > 0:
        print(f"[Cartesian] Warm-up (JIT compile)...", end=" ", flush=True)
        t0 = time.time()
        run_one_combo(combos[0], out_root, base_overrides, t_end, dt)
        print(f"done ({time.time()-t0:.1f}s)")

    # 并行执行
    if not _HAS_JOBLIB:
        print("[Cartesian] joblib not installed, running sequentially.")
        results = []
        for i, combo in enumerate(combos):
            sm = run_one_combo(combo, out_root, base_overrides, t_end, dt)
            results.append(sm)
            if sm["n_exceed_90"] == 0:
                print(f"  [{i+1}/{n_total}] {sm['_combo_id']}  L/W={sm['L/W']:.3f} ✅")
        return results

    print(f"[Cartesian] Running {n_total} combos...", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(run_one_combo)(combo, out_root, base_overrides, t_end, dt)
        for combo in combos
    )
    elapsed = time.time() - t0

    # 统计
    n_stable = sum(1 for r in results if r and r.get("n_exceed_90", 1) == 0)
    best = max((r for r in results if r and r.get("n_exceed_90", 1) == 0),
               key=lambda x: x.get("L/W", 0), default=None)

    print(f"[Cartesian] Done: {len(results)} combos in {elapsed:.0f}s "
          f"({elapsed/60:.1f} min) | stable: {n_stable}/{n_total}")
    if best:
        print(f"[Cartesian] Best stable: {best['_combo_id']}  L/W={best['L/W']:.3f}  peak={best['peak_theta_deg']:.1f}°")

    # 汇总
    _build_sweep_summary(results, grid, out_root)
    return results


def _build_sweep_summary(results: list, grid_spec: dict, out_root: Path) -> dict:
    """汇总所有 combo 的标量指标到 sweep_summary.json."""
    keys = ["_combo_id", "L/W", "L/W_body", "peak_theta_deg", "n_exceed_90",
            "mean_Fz_body_mN", "mean_Fz_world_mN", "mean_Fx_body_mN",
            "mean_M_aero_uNm", "peak_M_aero_uNm",
            "mean_M_grav_uNm", "peak_M_grav_uNm",
            "mean_M_damp_uNm", "peak_M_damp_uNm",
            "mean_abs_thetadot_rads", "peak_abs_thetadot_rads",
            "mean_abs_thetaddot_rads2", "peak_abs_thetaddot_rads2",
            "peak_alpha_eff_FL_deg", "peak_alpha_eff_BL_deg",
            "mean_CL_FL", "mean_CD_FL"]

    summary_data = {k: [] for k in keys}
    # 每个 combo 的参数值也存一份
    param_keys = list((grid_spec or DEFAULT_GRID).keys())
    for pk in param_keys:
        summary_data[f"_param_{pk}"] = []

    for sm in results:
        for k in keys:
            summary_data[k].append(sm.get(k, None))
        # 存储参数值
        combo = sm.get("_combo", {})
        for pk in param_keys:
            summary_data[f"_param_{pk}"].append(combo.get(pk, None))

    # 网格元信息
    summary_data["_grid"] = {k: list(v) for k, v in (grid_spec or DEFAULT_GRID).items()}
    summary_data["_n_combos"] = len(results)
    summary_data["_param_keys"] = list(grid_spec.keys())

    with open(out_root / "sweep_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"  Summary saved to {out_root / 'sweep_summary.json'}")
    return summary_data


# ============================================================
# __main__
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="笛卡尔积参数扫描 (方案二)")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="并行 worker 数, -1=全部核心 (default: -1)")
    ap.add_argument("--t-end", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=50e-6)
    ap.add_argument("--grid", type=str, default=None,
                    help="自定义网格 JSON 文件路径")
    ap.add_argument("--list-grid", action="store_true",
                    help="打印默认网格并退出")
    args = ap.parse_args()

    if args.list_grid:
        grid = DEFAULT_GRID
        n = 1
        for v in grid.values():
            n *= len(v)
        print(f"Default grid ({n} combos):")
        for k, v in grid.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    grid_spec = DEFAULT_GRID
    if args.grid:
        with open(args.grid) as f:
            grid_spec = json.load(f)

    results = sweep_cartesian(
        grid_spec=grid_spec,
        n_jobs=args.n_jobs,
        t_end=args.t_end,
        dt=args.dt,
    )

    # 打印 Top 10
    sorted_results = sorted(results, key=lambda x: x.get("L/W", 0), reverse=True)
    print(f"\n=== Top 10 by L/W ===")
    for i, sm in enumerate(sorted_results[:10]):
        cid = sm.get("_combo_id", "?")
        lw = sm.get("L/W", float('nan'))
        peak = sm.get("peak_theta_deg", float('nan'))
        n90 = sm.get("n_exceed_90", -1)
        status = "✅" if n90 == 0 else "❌"
        print(f"  {i+1}. {cid}  L/W={lw:.3f}  peak={peak:.1f}°  {status}")
