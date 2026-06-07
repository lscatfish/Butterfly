#!/usr/bin/env python3
"""
稳定性分析模块 — 单变量偏离扫描 + 基线分析

与 stability_plot.py 通过文件通信:
  temp/stability/
  ├── baseline/
  │   ├── config.json      # 所用参数
  │   ├── summary.json     # 标量指标
  │   └── timeseries.npz   # 全时程
  └── sweep_<param>/
      ├── checkpoint.json  # {done: [...], pending: [...], current: ...}
      ├── sweep_summary.json
      └── <value>/
          ├── config.json / summary.json / timeseries.npz

特性:
  - 每完成一个参数值立即保存 (checkpoint)
  - 支持断点续跑: 读取 checkpoint.json, 跳过已完成的
  - 完全独立于绘图逻辑
"""
import sys, json, time, os, traceback
from pathlib import Path
import numpy as np

_PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ / "src"))
from butterfly_forces import SimulationConfig, ButterflyForceModel

OUT_ROOT = _PROJ / "temp" / "stability"
BASELINE_DIR = OUT_ROOT / "baseline"


# ============================================================
# 基线参数 (能飞的配置, Fz_world > 重量, L/W_world≈2.28)
# α_f=68, α_b=5, phase=-15, a=6.0, R=2.5
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
# 扫描方案: 每个参数向左右偏离
# ============================================================
SWEEP_RANGES = {
    "alpha_front_deg":  [40, 45, 50, 55, 60, 65, 68, 70],
    "alpha_back_deg":   [3, 5, 8, 10, 12, 15, 18],
    "phase_diff_deg":   [-45, -30, -20, -15, -10, -5, 0, 5, 10],
    "mech_a":           [5.0, 6.0, 7.0, 7.5, 7.92, 8.5, 9.5, 11.0, 12.0],
    "mech_R":           [2.0, 2.25, 2.5, 2.75, 3.0, 3.5],
    "phi_offset_deg":   [-65, -60, -55, -50.84, -48, -45, -40, -35, -30],
}

# 数值参数的扫描格式
PARAM_FORMATS = {
    "alpha_front_deg": "af", "alpha_back_deg": "ab",
    "phase_diff_deg": "ph", "mech_a": "a",
    "mech_R": "R", "phi_offset_deg": "po",
}


def _make_config(overrides: dict) -> SimulationConfig:
    d = BASELINE_CONFIG.copy()
    d.update(overrides)
    # 类型修复
    for k in ["rotation"]:
        d[k] = str(d[k])
    return SimulationConfig(**{k: v for k, v in d.items()
                                if k in SimulationConfig.__dataclass_fields__})


def _extract_timeseries(out) -> dict:
    """从 SimulationOutput 提取全时程数据."""
    s = out.summary
    half = len(out.t) // 2

    # 四翅聚合
    Fz_body = np.zeros(len(out.t))
    Fx_body = np.zeros(len(out.t))
    Fz_world = np.zeros(len(out.t))
    for wn in ["FL", "FR", "BL", "BR"]:
        w = out.wings[wn]
        Fz_body += w.force_body[:, 2]
        Fx_body += w.force_body[:, 0]
        Fz_world += w.force_world[:, 2]

    # M_aero: 从前后翅 Fz_body 计算
    # M = 2 * (-x_front * Fz_f - x_back * Fz_b)  其中每翅贡献 = Fz_wing
    # FL+FR 合计 = 2*Fz_f, BL+BR 合计 = 2*Fz_b
    # M_aero = -x_front*(Fz_FL+Fz_FR) - x_back*(Fz_BL+Fz_BR)
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
        # 每翅关键量
        **{f"{wn}_Fz_body": out.wings[wn].force_body[:, 2] for wn in ["FL", "FR", "BL", "BR"]},
        **{f"{wn}_Fx_body": out.wings[wn].force_body[:, 0] for wn in ["FL", "FR", "BL", "BR"]},
        **{f"{wn}_alpha_eff": out.wings[wn].alpha_eff_deg for wn in ["FL", "FR", "BL", "BR"]},
        **{f"{wn}_C_L": out.wings[wn].C_L for wn in ["FL", "FR", "BL", "BR"]},
        **{f"{wn}_C_D": out.wings[wn].C_D for wn in ["FL", "FR", "BL", "BR"]},
        **{f"{wn}_phi": out.wings[wn].phi for wn in ["FL", "FR", "BL", "BR"]},
        # 摇杆主矢: 存 X/Z 分量 (在机构平面 XZ 内, Y=0)
        **{f"{wn}_rocker_pv_x": out.wings[wn].rocker_principal_vec[:, 0] for wn in ["FL", "FR", "BL", "BR"]},
        **{f"{wn}_rocker_pv_z": out.wings[wn].rocker_principal_vec[:, 2] for wn in ["FL", "FR", "BL", "BR"]},
        # 摇杆主矩: Y 分量 (绕 Y 轴的有效扭矩)
        **{f"{wn}_rocker_pm_y": out.wings[wn].rocker_principal_moment[:, 1] for wn in ["FL", "FR", "BL", "BR"]},
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
        # 基础
        "L/W": s["L/W"],
        "peak_theta_deg": s["peak_theta_deg"],
        "n_exceed_90": s["n_exceed_90"],
        "weight_mN": weight_mN,
        # Fz / Fx
        **_stats(ts_dict["Fz_body_total"] * 1000, "_Fz_body_mN"),
        **_stats(ts_dict["Fz_world_total"] * 1000, "_Fz_world_mN"),
        **_stats(ts_dict["Fx_body_total"] * 1000, "_Fx_body_mN"),
        # 力矩
        **_stats(ts_dict["M_aero"] * 1e6, "_M_aero_uNm"),
        **_stats(ts_dict["M_grav"] * 1e6, "_M_grav_uNm"),
        **_stats(ts_dict["M_damp"] * 1e6, "_M_damp_uNm"),
        # 俯仰运动
        **_stats(np.abs(out.theta_dot), "_abs_thetadot_rads"),
        **_stats(np.abs(out.theta_ddot), "_abs_thetaddot_rads2"),
        # 攻角 (前后翅)
        **_stats(out.wings["FL"].alpha_eff_deg, "_alpha_eff_FL_deg"),
        **_stats(out.wings["BL"].alpha_eff_deg, "_alpha_eff_BL_deg"),
        # C_L, C_D
        **_stats(out.wings["FL"].C_L, "_CL_FL"),
        **_stats(out.wings["FL"].C_D, "_CD_FL"),
    }


def _save_run(out_dir: Path, out, ts_dict: dict, summary_extra: dict = None):
    """保存一次运行的完整数据."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # config (从 SimulationConfig dataclass 提取)
    cfg = out.config
    config_dict = {f.name: getattr(cfg, f.name)
                   for f in cfg.__dataclass_fields__.values()}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # summary
    summary = _extract_summary(out, ts_dict)
    if summary_extra:
        summary.update(summary_extra)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # timeseries (npz)
    np.savez_compressed(out_dir / "timeseries.npz", **ts_dict)


def _load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ============================================================
# Public API
# ============================================================

def run_baseline(config_overrides: dict = None,
                  out_dir: Path = None,
                  t_end: float = 5.0, dt: float = 50e-6) -> dict:
    """运行基线分析, 保存到 baseline/.

    Args:
        config_overrides: 覆盖 BASELINE_CONFIG 的参数字典.
        out_dir: 输出目录, 默认 temp/stability/baseline/.
        t_end, dt: 仿真参数.

    Returns:
        summary dict.
    """
    out_dir = out_dir or BASELINE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = config_overrides or {}
    overrides.setdefault("t_end", t_end)
    overrides.setdefault("dt", dt)

    cfg = _make_config(overrides)
    print(f"[baseline] α_f={cfg.alpha_front_deg}, α_b={cfg.alpha_back_deg}, "
          f"phase={cfg.phase_diff_deg}, a={cfg.mech_a}, R={cfg.mech_R}, "
          f"phi_off={cfg.phi_offset_deg}")

    model = ButterflyForceModel(cfg)
    out = model.simulate(progress=True)
    ts = _extract_timeseries(out)
    _save_run(out_dir, out, ts)

    print(f"[baseline] L/W={out.summary['L/W']:.3f}, peak={out.summary['peak_theta_deg']:.1f}°, "
          f"n90={out.summary['n_exceed_90']}")
    return _load_summary(out_dir / "summary.json")


def sweep_parameter(param_name: str,
                     values: list = None,
                     base_overrides: dict = None,
                     t_end: float = 5.0, dt: float = 50e-6) -> dict:
    """单变量偏离扫描.

    从 BASELINE_CONFIG 出发, 每次只改 param_name, 扫描 values 中每个值.
    每完成一个值立即保存到 temp/stability/sweep_<param>/<value>/.
    支持断点续跑.

    Args:
        param_name: 参数名 (必须在 SWEEP_RANGES 中).
        values: 参数值列表, 默认用 SWEEP_RANGES[param_name].
        base_overrides: 额外覆盖基线参数的 dict.
        t_end, dt: 仿真参数.

    Returns:
        sweep_summary dict: {values: [...], L/W: [...], ...}.
    """
    if values is None:
        values = SWEEP_RANGES.get(param_name)
        if values is None:
            raise ValueError(f"Unknown param: {param_name}. Known: {list(SWEEP_RANGES)}")

    sweep_dir = OUT_ROOT / f"sweep_{param_name}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # 读取或初始化 checkpoint
    ckpt_path = sweep_dir / "checkpoint.json"
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        done = set(tuple(v) if isinstance(v, list) else v for v in ckpt.get("done", []))
    else:
        ckpt = {"done": [], "pending": [], "current": None}
        done = set()

    base = (base_overrides or {}).copy()
    base.setdefault("t_end", t_end)
    base.setdefault("dt", dt)

    all_summaries = []
    fmt = PARAM_FORMATS.get(param_name, param_name)

    for i, val in enumerate(values):
        val_key = val if not isinstance(val, float) else round(val, 4)
        if val_key in done:
            print(f"[{fmt}={val}] ({i+1}/{len(values)}) ALREADY DONE, skip")
            # 读取已保存的 summary
            val_dir = sweep_dir / _val_to_str(val)
            sm = _load_summary(val_dir / "summary.json")
            sm["_param"] = param_name
            sm["_value"] = val
            all_summaries.append(sm)
            continue

        print(f"\n[{fmt}={val}] ({i+1}/{len(values)})")
        # 更新 checkpoint
        ckpt["current"] = val
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2)

        overrides = base.copy()
        overrides[param_name] = val

        try:
            cfg = _make_config(overrides)
            model = ButterflyForceModel(cfg)
            out = model.simulate(progress=True)
            ts = _extract_timeseries(out)

            val_dir = sweep_dir / _val_to_str(val)
            _save_run(val_dir, out, ts, {"_param": param_name, "_value": val})

            sm = _load_summary(val_dir / "summary.json")
            sm["_param"] = param_name
            sm["_value"] = val
            all_summaries.append(sm)

            # 标记完成
            done.add(val_key)
            ckpt["done"] = sorted(done, key=lambda x: float(x) if not isinstance(x, str) else 0)
            ckpt["current"] = None
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2)

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            ckpt["current"] = None
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2)
            continue

    # 汇总
    sweep_summary = _build_sweep_summary(all_summaries, param_name)
    with open(sweep_dir / "sweep_summary.json", "w") as f:
        json.dump(sweep_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[{param_name}] sweep done. {len(all_summaries)}/{len(values)} completed.")
    return sweep_summary


def _val_to_str(val) -> str:
    """参数值 → 合法文件夹名."""
    if isinstance(val, float):
        s = f"{val:.4f}".rstrip('0').rstrip('.')
        return s.replace('.', 'p').replace('-', 'n')
    return str(val).replace('.', 'p').replace('-', 'n')


def _build_sweep_summary(summaries: list, param_name: str) -> dict:
    keys = ["_value", "L/W", "peak_theta_deg", "n_exceed_90",
            "mean_Fz_body_mN", "mean_Fz_world_mN", "mean_Fx_body_mN",
            "mean_M_aero_uNm", "peak_M_aero_uNm",
            "mean_M_grav_uNm", "peak_M_grav_uNm",
            "mean_M_damp_uNm", "peak_M_damp_uNm",
            "mean_abs_thetadot_rads", "peak_abs_thetadot_rads",
            "mean_abs_thetaddot_rads2", "peak_abs_thetaddot_rads2",
            "peak_alpha_eff_FL_deg", "peak_alpha_eff_BL_deg",
            "mean_CL_FL", "mean_CD_FL"]
    result = {k: [] for k in keys}
    for sm in summaries:
        for k in keys:
            result[k].append(sm.get(k, None))
    result["_param"] = param_name
    result["_n"] = len(summaries)
    return result


# ============================================================
# __main__ — 调用示例
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="Run baseline analysis")
    ap.add_argument("--sweep", type=str, default=None,
                    help="Parameter name to sweep (e.g. alpha_front_deg)")
    ap.add_argument("--t-end", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=50e-6)
    args = ap.parse_args()

    if args.baseline:
        run_baseline(t_end=args.t_end, dt=args.dt)

    if args.sweep:
        sweep_parameter(args.sweep, t_end=args.t_end, dt=args.dt)

    if not args.baseline and not args.sweep:
        print("Usage: python stability_analysis.py --baseline [--sweep PARAM]")
        print(f"Available sweeps: {list(SWEEP_RANGES)}")
