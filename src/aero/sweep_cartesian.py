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

_PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJ))
from src.aero.butterfly_forces import SimulationConfig, ButterflyForceModel, _HAS_NUMBA, DESIGN_v69
from src.config import get_sweep_grid

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False
    Parallel = None
    delayed = None

OUT_ROOT = _PROJ / "temp" / "stability" / "sweep_cartesian"

# ============================================================
# v6.9 基线参数 — DESIGN_v69 + 扫参模式数值设定
# ============================================================
BASELINE_CONFIG = {**DESIGN_v69, "dt": 50e-6, "t_end": 5.0, "steady_start": 3.0}

# ============================================================
# v6.9 扫参配置 — 从 config/design_v69.yaml → sweep 读取
# ============================================================
_sw_raw = get_sweep_grid()
if not _sw_raw:
    raise SystemExit("Error: config/design_v69.yaml 中没有 'sweep' 段")

# 分离元配置（_ 前缀）和参数网格
_META_KEYS = {k for k in _sw_raw if k.startswith("_")}
SWEEP_META = {k: _sw_raw[k] for k in _META_KEYS}
SWEEP_GRID = {k: v for k, v in _sw_raw.items() if not k.startswith("_")}

OUT_ROOT = _PROJ / SWEEP_META.get("_out_dir", "temp/stability/sweep_cartesian")
SWEEP_N_JOBS = int(SWEEP_META.get("_n_jobs", -1))

PARAM_SHORT = {
    "alpha_front_deg": "af", "alpha_back_deg": "ab",
    "phase_diff_deg": "ph", "mech_a": "a",
    "mech_R": "R", "phi_offset_deg": "po",
    "f": "f", "c_damp": "cd", "rotation": "rot",
    "k_clap": "kc",
}


# ---- combo_id 编解码 ----
# 仅扫描参数（列表长度 > 1）参与目录名编码
_SCAN_KEYS = [k for k, v in SWEEP_GRID.items()
              if isinstance(v, (list, tuple)) and len(v) > 1]


def combo_to_id(combo: dict) -> str:
    """将参数字典编码为紧凑文件夹名（仅扫描参数，科学计数法无精度丢失）。"""
    parts = []
    for k in _SCAN_KEYS:
        v = combo[k]
        short = PARAM_SHORT.get(k, k)
        if isinstance(v, float):
            mantissa, exp = f"{v:.6e}".split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            s = f"{mantissa}e{exp}"
            s = s.replace("e+", "ep").replace("e-", "en").replace(".", "p").replace("-", "n")
        elif isinstance(v, int):
            s = str(v).replace("-", "n")
        else:
            s = str(v).replace("-", "n")
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
def _normalize_grid(raw: dict) -> dict:
    """将标量值包裹为单元素列表，多值列表原样保留。"""
    out = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = [v]
    return out


def build_cartesian_grid(grid_spec: dict = None) -> list:
    """从网格定义生成组合列表.

    Args:
        grid_spec: {param_name: [values] 或 标量}, 默认 SWEEP_GRID (YAML > sweep).

    Returns:
        [{param: value}, ...] 所有笛卡尔积组合.
    """
    raw = grid_spec or SWEEP_GRID
    grid = _normalize_grid(raw)
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

    Fy_body = np.zeros(len(out.t))
    Fx_body = np.zeros(len(out.t))
    Fy_world = np.zeros(len(out.t))
    for wn in ["FL", "FR", "BL", "BR"]:
        w = out.wings[wn]
        Fy_body += w.force_body[:, 1]
        Fx_body += w.force_body[:, 0]
        Fy_world += w.force_world[:, 1]

    Fy_f_total = out.wings["FL"].force_body[:, 1] + out.wings["FR"].force_body[:, 1]
    Fy_b_total = out.wings["BL"].force_body[:, 1] + out.wings["BR"].force_body[:, 1]
    # 前翅向上力 (Fy>0) 在 z_front>0 产生低头力矩 (负); 后翅在 z_back<0 产生抬头力矩 (正)
    M_aero = -out.config.z_front * Fy_f_total - out.config.z_back * Fy_b_total
    M_damp = -out.config.c_damp * out.theta_dot

    return {
        "t": out.t,
        "theta_p": out.theta_p,
        "theta_dot": out.theta_dot,
        "theta_ddot": out.theta_ddot,
        "Fy_body_total": Fy_body,
        "Fx_body_total": Fx_body,
        "Fy_world_total": Fy_world,
        "M_aero": M_aero,
        "M_damp": M_damp,
        **{f"{wn}_Fy_body": out.wings[wn].force_body[:, 1] for wn in ["FL", "FR", "BL", "BR"]},
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
        **_stats(ts_dict["Fy_body_total"] * 1000, "_Fy_body_mN"),
        **_stats(ts_dict["Fy_world_total"] * 1000, "_Fy_world_mN"),
        **_stats(ts_dict["Fx_body_total"] * 1000, "_Fx_body_mN"),
        **_stats(ts_dict["M_aero"] * 1e6, "_M_aero_uNm"),
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

    grid = grid_spec or SWEEP_GRID
    combos = build_cartesian_grid(grid)
    n_total = len(combos)

    # 单行摘要
    grid_info = ", ".join(f"{k}={len(v) if isinstance(v, (list, tuple)) else 1}"
                         for k, v in grid.items())
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
    """汇总所有 combo 的标量指标到 sweep_summary.json.

    行为:
      - 如果 sweep_summary.json 已存在, 先读取并按 combo_id 合并.
      - 本次新结果会覆盖同 combo_id 的旧条目.
      - 未被覆盖的旧条目保留.
      - 如果文件不存在, 创建新文件.
    """
    keys = ["_combo_id", "L/W", "L/W_body", "peak_theta_deg", "n_exceed_90",
            "mean_Fy_body_mN", "mean_Fy_world_mN", "mean_Fx_body_mN",
            "mean_M_aero_uNm", "peak_M_aero_uNm",
            "mean_M_damp_uNm", "peak_M_damp_uNm",
            "mean_abs_thetadot_rads", "peak_abs_thetadot_rads",
            "mean_abs_thetaddot_rads2", "peak_abs_thetaddot_rads2",
            "peak_alpha_eff_FL_deg", "peak_alpha_eff_BL_deg",
            "mean_CL_FL", "mean_CD_FL"]

    summary_path = out_root / "sweep_summary.json"

    # ---- 读取已有数据（如果存在） ----
    existing_rows = {}
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                old = json.load(f)
            n_old = old.get("_n_combos", 0)
            if n_old > 0:
                for i in range(n_old):
                    cid = old["_combo_id"][i]
                    row = {k: old.get(k, [None] * n_old)[i] for k in keys}
                    row["_combo"] = {}
                    for pk in old.get("_param_keys", []):
                        row["_combo"][pk] = old.get(f"_param_{pk}", [None] * n_old)[i]
                    existing_rows[cid] = row
        except Exception as e:
            print(f"  Warning: 读取已有 sweep_summary.json 失败, 将覆盖: {e}")
            existing_rows = {}

    # ---- 本次新结果 ----
    new_rows = {}
    param_keys = list(grid_spec.keys())
    for sm in results:
        cid = sm.get("_combo_id")
        if cid is None:
            continue
        row = {k: sm.get(k, None) for k in keys}
        row["_combo"] = sm.get("_combo", {})
        new_rows[cid] = row

    # ---- 合并：新结果覆盖旧结果 ----
    merged_rows = {**existing_rows, **new_rows}

    # ---- 重建 summary_data ----
    summary_data = {k: [] for k in keys}
    for pk in param_keys:
        summary_data[f"_param_{pk}"] = []

    for cid in sorted(merged_rows.keys()):
        row = merged_rows[cid]
        summary_data["_combo_id"].append(cid)
        for k in keys:
            if k != "_combo_id":
                summary_data[k].append(row.get(k, None))
        for pk in param_keys:
            summary_data[f"_param_{pk}"].append(row["_combo"].get(pk, None))

    # 网格元信息：取已有和本次 grid 的并集，保留最全的列表
    _grid_meta = {}
    for k in param_keys:
        old_vals = set()
        if summary_path.exists() and isinstance(existing_rows, dict):
            for r in existing_rows.values():
                if k in r["_combo"]:
                    old_vals.add(r["_combo"][k])
        new_vals = set()
        for r in new_rows.values():
            if k in r["_combo"]:
                new_vals.add(r["_combo"][k])
        all_vals = old_vals | new_vals
        v_spec = grid_spec.get(k)
        if isinstance(v_spec, (list, tuple)):
            base = list(v_spec)
        else:
            base = [v_spec]
        # 合并并排序，尽量保持数值顺序
        merged_vals = sorted(all_vals, key=lambda x: (isinstance(x, str), x if isinstance(x, str) else float(x)))
        _grid_meta[k] = merged_vals if len(merged_vals) > 1 else (merged_vals[0] if merged_vals else v_spec)

    summary_data["_grid"] = _grid_meta
    summary_data["_n_combos"] = len(merged_rows)
    summary_data["_param_keys"] = param_keys

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"  Summary saved to {summary_path} (merged {len(existing_rows)} old + {len(new_rows)} new = {len(merged_rows)} combos)")
    return summary_data


# ============================================================
# __main__
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="笛卡尔积参数扫描 (方案二)")
    ap.add_argument("--n-jobs", type=int, default=SWEEP_N_JOBS,
                    help=f"并行 worker 数, -1=全部核心 (default: {SWEEP_N_JOBS})")
    ap.add_argument("--t-end", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=50e-6)
    ap.add_argument("--grid", type=str, default=None,
                    help="自定义网格 JSON 文件路径")
    ap.add_argument("--list-grid", action="store_true",
                    help="打印默认网格并退出")
    args = ap.parse_args()

    if args.list_grid:
        grid = SWEEP_GRID
        n = 1
        for v in grid.values():
            n *= len(v) if isinstance(v, (list, tuple)) else 1
        print(f"Sweep grid from config/design_v69.yaml → sweep ({n} combos):")
        for k, v in grid.items():
            tag = "scan" if isinstance(v, (list, tuple)) and len(v) > 1 else "fix"
            print(f"  [{tag}] {k}: {v}")
        sys.exit(0)

    grid_spec = SWEEP_GRID
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
