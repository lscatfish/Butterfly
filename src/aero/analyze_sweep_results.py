#!/usr/bin/env python3
"""
扫参结果分析工具 — 从 sweep_summary.json 生成稳定组合报告.

功能:
  1. 读取 sweep_summary.json 并生成 Markdown 报告.
  2. (--sync) 遍历磁盘上的 combo 子文件夹, 与 json 同步:
     - 磁盘有、JSON 没有  → 追加
     - JSON 有、磁盘没有  → 删除
     - 两者都有但内容不一致 → 以磁盘 summary.json 为准更新

用法:
  # 只读分析（默认目录来自 config/design_v69.yaml 的 _out_dir）
  python src/aero/analyze_sweep_results.py

  # 指定目录并同步 json
  python src/aero/analyze_sweep_results.py "F:/.../sweep_all" --sync

  # 同步 + 生成报告 + 限制 peak theta
  python src/aero/analyze_sweep_results.py --sync --max-peak-theta 60
"""
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np


def _default_sweep_dir() -> Path:
    """从 config/design_v69.yaml 读取默认 _out_dir."""
    proj_root = Path(__file__).resolve().parent.parent.parent
    config_path = proj_root / "config" / "design_v69.yaml"
    default = proj_root / "temp" / "stability" / "sweep_cartesian"
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        sweep = raw.get("sweep", {})
        out_dir = sweep.get("_out_dir", "temp/stability/sweep_cartesian")
        p = Path(out_dir)
        if not p.is_absolute():
            p = proj_root / p
        return p
    except Exception:
        return default


# 报告中使用的标量指标（尽量与 sweep_cartesian.py 的 _build_sweep_summary 一致）
_SUMMARY_SCALAR_KEYS = [
    "L/W", "L/W_body", "peak_theta_deg", "n_exceed_90",
    "mean_Fy_body_mN", "mean_Fy_world_mN", "mean_Fx_body_mN",
    "mean_M_aero_uNm", "peak_M_aero_uNm",
    "mean_M_damp_uNm", "peak_M_damp_uNm",
    "mean_abs_thetadot_rads", "peak_abs_thetadot_rads",
    "mean_abs_thetaddot_rads2", "peak_abs_thetaddot_rads2",
    "peak_alpha_eff_FL_deg", "peak_alpha_eff_BL_deg",
    "mean_CL_FL", "mean_CD_FL",
    "weight_mN",
]


def load_summary(sweep_dir: Path) -> dict:
    """加载 sweep_summary.json, 不存在则返回空结构."""
    path = sweep_dir / "sweep_summary.json"
    if not path.exists():
        return {
            "_combo_id": [], "_combo": [],
            "_param_keys": [], "_grid": {}, "_n_combos": 0,
        }
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_summary(sweep_dir: Path, data: dict):
    """保存 sweep_summary.json."""
    path = sweep_dir / "sweep_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _read_combo_summary(combo_dir: Path) -> dict:
    """读取单个 combo 的 summary.json."""
    path = combo_dir / "summary.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_combo_config(combo_dir: Path) -> dict:
    """读取单个 combo 的 config.json（用于 _combo 字段）."""
    path = combo_dir / "config.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_old_rows(data: dict) -> dict:
    """把旧的 sweep_summary.json 按 combo_id 展开成 dict, 并去重.

    去重规则: 同一个 combo_id 出现多次时保留最后一个(最新)条目.
    """
    rows = {}
    n = data.get("_n_combos", 0)
    if n == 0:
        return rows

    keys = [k for k in data.keys()
            if isinstance(data[k], list) and len(data[k]) == n and not k.startswith("_param_")]
    param_keys = data.get("_param_keys", [])
    for i in range(n):
        cid = data["_combo_id"][i]
        row = {"_combo": {}}
        for k in keys:
            row[k] = data[k][i]
        for pk in param_keys:
            arr_key = f"_param_{pk}"
            if arr_key in data and isinstance(data[arr_key], list):
                row["_combo"][pk] = data[arr_key][i] if i < len(data[arr_key]) else None
        rows[cid] = row  # 重复的 combo_id 会被后面的覆盖
    return rows


def sync_summary(sweep_dir: Path) -> dict:
    """遍历磁盘 combo 文件夹, 与 sweep_summary.json 同步.

    同步规则:
      - 磁盘有、JSON 没有  → 追加
      - JSON 有、磁盘没有  → 删除
      - 两者都有但 scalar 不一致 → 以磁盘 summary.json 更新
      - JSON 内部 combo_id 重复 → 去重保留最后一个
    """
    data = load_summary(sweep_dir)

    # 先对旧 json 内部去重
    old_rows = _load_old_rows(data)
    duplicates_removed = data.get("_n_combos", 0) - len(old_rows)
    if duplicates_removed > 0:
        print(f"[sync] 检测到并移除 {duplicates_removed} 个重复 combo_id")

    param_keys = list(data.get("_param_keys", []))

    # 收集磁盘上所有 combo 文件夹
    disk_combos = {}
    for child in sweep_dir.iterdir():
        if not child.is_dir():
            continue
        combo_id = child.name
        if combo_id.startswith(".") or combo_id in ("checkpoint",):
            continue
        sm = _read_combo_summary(child)
        cfg = _read_combo_config(child)
        if sm is None or cfg is None:
            continue
        disk_combos[combo_id] = {"summary": sm, "config": cfg}

    # 确定参数键：保留 JSON 中的，同时补充磁盘 config 中的新键
    all_keys = set(param_keys)
    for v in disk_combos.values():
        all_keys.update(v["config"].keys())
    # 排序：让扫参参数排在前面，固定参数排在后面
    scan_order = list(data.get("_grid", {}).keys())
    fixed_order = [k for k in sorted(all_keys) if k not in scan_order]
    param_keys = [k for k in scan_order if k in all_keys] + fixed_order

    # 构建新的同步后数据
    new_data = {
        "_combo_id": [],
        "_combo": [],
        "_param_keys": param_keys,
        "_grid": {},
        "_n_combos": 0,
        "_synced_at": datetime.now().isoformat(),
    }

    # _grid 重建
    for k in param_keys:
        vals = set()
        for r in old_rows.values():
            if k in r.get("_combo", {}):
                vals.add(r["_combo"][k])
        for v in disk_combos.values():
            if k in v["config"]:
                vals.add(v["config"][k])
        try:
            vals_list = sorted(vals, key=lambda x: (isinstance(x, str), x if isinstance(x, str) else float(x)))
        except Exception:
            vals_list = sorted(vals, key=str)
        new_data["_grid"][k] = vals_list if len(vals_list) > 1 else (vals_list[0] if vals_list else None)

    # 合并：磁盘优先覆盖旧条目
    merged = {**old_rows}
    for cid, v in disk_combos.items():
        merged[cid] = {
            "_combo": v["config"],
        }
        for k in _SUMMARY_SCALAR_KEYS:
            merged[cid][k] = v["summary"].get(k, None)

    # 填充数组
    for combo_id in sorted(merged.keys()):
        row = merged[combo_id]
        cfg = row.get("_combo", {})
        new_data["_combo_id"].append(combo_id)
        new_data["_combo"].append(cfg)
        for k in param_keys:
            arr_key = f"_param_{k}"
            if arr_key not in new_data:
                new_data[arr_key] = []
            new_data[arr_key].append(cfg.get(k, None))
        for k in _SUMMARY_SCALAR_KEYS:
            if k not in new_data:
                new_data[k] = []
            new_data[k].append(row.get(k, None))

    new_data["_n_combos"] = len(merged)

    save_summary(sweep_dir, new_data)
    return new_data


def extract_stable_combos(data: dict, max_peak_theta: float = None) -> list:
    """提取稳定组合列表, 可选按 peak_theta 过滤."""
    n_exceed_90 = np.array(data.get("n_exceed_90", []))
    if len(n_exceed_90) == 0:
        return []
    stable_mask = n_exceed_90 == 0
    if max_peak_theta is not None:
        peak_theta = np.array(data.get("peak_theta_deg", []))
        stable_mask = stable_mask & (peak_theta <= max_peak_theta)

    indices = np.where(stable_mask)[0]
    combos = []
    param_keys = data.get("_param_keys", [])
    for i in indices:
        combo = {
            "_combo_id": data["_combo_id"][i],
            "L/W": data.get("L/W", [np.nan] * len(data["_combo_id"]))[i],
            "L/W_body": data.get("L/W_body", [np.nan] * len(data["_combo_id"]))[i],
            "peak_theta_deg": data.get("peak_theta_deg", [np.nan] * len(data["_combo_id"]))[i],
            "mean_Fy_world_mN": data.get("mean_Fy_world_mN", [np.nan] * len(data["_combo_id"]))[i],
            "mean_Fy_body_mN": data.get("mean_Fy_body_mN", [np.nan] * len(data["_combo_id"]))[i],
            "mean_M_aero_uNm": data.get("mean_M_aero_uNm", [np.nan] * len(data["_combo_id"]))[i],
            "peak_M_aero_uNm": data.get("peak_M_aero_uNm", [np.nan] * len(data["_combo_id"]))[i],
        }
        for k in param_keys:
            combo[k] = data.get(f"_param_{k}", [None] * len(data["_combo_id"]))[i]
        combos.append(combo)

    combos.sort(key=lambda x: x["L/W"] if x["L/W"] is not None else -np.inf, reverse=True)
    return combos


def param_distribution(combos: list, param_keys: list) -> dict:
    """统计稳定组合在各参数上的分布."""
    dist = {}
    for k in param_keys:
        values = [c[k] for c in combos if c[k] is not None]
        if all(isinstance(v, (int, float)) for v in values):
            counter = Counter(values)
            dist[k] = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
        else:
            counter = Counter(str(v) for v in values)
            dist[k] = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    return dist


def format_value(v) -> str:
    """格式化单个值."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def _html_escape(v) -> str:
    """转义 HTML 特殊字符."""
    s = str(v) if v is not None else "N/A"
    return (s
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def generate_report(data: dict, combos: list, top_n: int, max_peak_theta: float) -> str:
    """生成 Markdown 报告."""
    param_keys = data.get("_param_keys", [])
    grid = data.get("_grid", {})
    scan_keys = [k for k in param_keys
                 if isinstance(grid.get(k, []), list)
                 and len(grid.get(k, [])) > 1]
    fixed_keys = [k for k in param_keys if k not in scan_keys]

    lines = []
    lines.append("# Butterfly MAV 扫参稳定性分析报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if "_synced_at" in data:
        lines.append(f"同步时间: {data['_synced_at']}")
    lines.append("")

    # 总体统计
    lines.append("## 总体统计")
    lines.append("")
    lines.append(f"- 总组合数: **{data.get('_n_combos', 0)}**")
    lines.append(f"- 稳定组合数 (`n_exceed_90 == 0`): **{len(combos)}**")
    if max_peak_theta is not None:
        lines.append(f"- 附加过滤条件: `peak_theta_deg <= {max_peak_theta}°`")
    if combos:
        lws = [c["L/W"] for c in combos if c["L/W"] is not None]
        peaks = [c["peak_theta_deg"] for c in combos if c["peak_theta_deg"] is not None]
        if lws:
            lines.append(f"- 稳定组合 L/W 范围: `{min(lws):.3f} ~ {max(lws):.3f}`")
        if peaks:
            lines.append(f"- 稳定组合 peak θ 范围: `{min(peaks):.1f}° ~ {max(peaks):.1f}°`")
        lines.append(f"- 稳定组合中 L/W > 1 的数量: **{sum(1 for c in combos if c['L/W'] is not None and c['L/W'] > 1)}**")
    lines.append("")

    # 固定参数
    if fixed_keys:
        lines.append("## 固定参数")
        lines.append("")
        lines.append("| 参数 | 值 |")
        lines.append("|---|---|")
        for k in fixed_keys:
            v = combos[0][k] if combos else data.get(f"_param_{k}", ["N/A"])[0]
            lines.append(f"| `{k}` | {format_value(v)} |")
        lines.append("")

    # Top N
    if combos and top_n > 0:
        lines.append(f"## Top {min(top_n, len(combos))} 稳定组合")
        lines.append("")
        display_keys = [k for k in scan_keys if k in combos[0]]
        header = "| 排名 | combo_id | L/W | L/W_body | peak θ | Fy_world | " + " | ".join(display_keys) + " |"
        lines.append(header)
        sep = "|---" * (6 + len(display_keys)) + "|"
        lines.append(sep)
        for rank, c in enumerate(combos[:top_n], start=1):
            row = f"| {rank} | `{c['_combo_id']}` | {c['L/W']:.3f} | {c['L/W_body']:.3f} | {c['peak_theta_deg']:.1f}° | {c['mean_Fy_world_mN']:.1f} |"
            for k in display_keys:
                row += f" {format_value(c[k])} |"
            lines.append(row)
        lines.append("")

    # 参数分布
    dist = param_distribution(combos, scan_keys)
    if dist:
        lines.append("## 稳定组合参数分布")
        lines.append("")
        for k, counts in dist.items():
            lines.append(f"### `{k}`")
            lines.append("")
            lines.append("| 取值 | 出现次数 |")
            lines.append("|---|---|")
            for val, cnt in list(counts.items())[:10]:  # 最多显示前10
                lines.append(f"| {format_value(val)} | {cnt} |")
            lines.append("")

    # 全部稳定组合表（使用 HTML table，避免长 Markdown 表格在某些阅读器里被压成一行）
    if combos:
        lines.append("## 全部稳定组合")
        lines.append("")
        display_keys = [k for k in scan_keys if k in combos[0]]
        col_names = ["combo_id", "L/W", "peak θ", "Fy_world"] + display_keys
        lines.append("<table>")
        lines.append("<thead><tr>" +
                     "".join(f"<th>{_html_escape(c)}</th>" for c in col_names) +
                     "</tr></thead>")
        lines.append("<tbody>")
        for c in combos:
            vals = [
                c["_combo_id"],
                f"{c['L/W']:.3f}",
                f"{c['peak_theta_deg']:.1f}°",
                f"{c['mean_Fy_world_mN']:.1f}",
            ] + [format_value(c[k]) for k in display_keys]
            lines.append("<tr>" +
                         "".join(f"<td>{_html_escape(v)}</td>" for v in vals) +
                         "</tr>")
        lines.append("</tbody></table>")
        lines.append("")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("报告由 `src/aero/analyze_sweep_results.py` 自动生成。")
    return "\n".join(lines) + "\n"


def main():
    default_dir = _default_sweep_dir()

    ap = argparse.ArgumentParser(description="分析扫参结果并生成稳定组合报告")
    ap.add_argument("sweep_dir", type=str, nargs="?", default=str(default_dir),
                    help=f"扫参输出目录 (default: 读取 config/design_v69.yaml 的 _out_dir)")
    ap.add_argument("--sync", action="store_true",
                    help="同步磁盘 combo 文件夹与 sweep_summary.json (会修改 json)")
    ap.add_argument("--top", type=int, default=20,
                    help="Top N 稳定组合 (default: 20)")
    ap.add_argument("--max-peak-theta", type=float, default=None,
                    help="只保留 peak_theta_deg <= 该值的组合")
    ap.add_argument("--out", type=str, default=None,
                    help="报告输出路径 (default: SWEEP_DIR/sweep_report.md)")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        raise FileNotFoundError(f"扫参目录不存在: {sweep_dir}")

    if args.sync:
        print(f"[sync] 正在同步 {sweep_dir} ...")
        data = sync_summary(sweep_dir)
        print(f"[sync] 完成，总组合数: {data['_n_combos']}")
    else:
        data = load_summary(sweep_dir)

    combos = extract_stable_combos(data, max_peak_theta=args.max_peak_theta)
    report = generate_report(data, combos, args.top, args.max_peak_theta)

    out_path = Path(args.out) if args.out else sweep_dir / "sweep_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存: {out_path.resolve()}")
    print(f"稳定组合数: {len(combos)}")


if __name__ == "__main__":
    main()
