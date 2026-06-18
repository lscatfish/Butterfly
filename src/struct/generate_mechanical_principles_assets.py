#!/usr/bin/env python3
"""
机械原理报告配套图和汇总数据的复现入口。

本脚本服务于以下两个文档：

- output/reports/机械原理结构系统分析.md
- output/reports/机械原理曲线图读图说明.md

默认只重新生成图像和 JSON 汇总文件，不覆盖已经手工修改过的
Markdown 报告。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.struct import mechanical_principles_analysis as analysis


def generate_assets(write_report_template: bool = False) -> dict[str, Path]:
    """Regenerate report figures and the mechanical-principles summary JSON."""
    analysis.ensure_dirs()
    results = analysis.compute_results()

    reference_paths = analysis.plot_crank_rocker_reference_crops()
    mechanism_schematic = analysis.plot_mechanism_schematic_reference()

    figure_paths: dict[str, Path] = {
        "crank_ref": reference_paths["mechanism"],
        "crank_fbd": reference_paths["fbd"],
        "rocker_moment": analysis.plot_rocker_moment_vs_crank_angle(results),
        "torque_chain": analysis.plot_torque_chain_vs_crank_angle(results),
        "gear_forces_angle": analysis.plot_gear_mesh_forces_vs_crank_angle(results),
        "output_torque": analysis.plot_output_torque_time(results),
        "system_force_flow": analysis.plot_system_force_flow(results),
    }

    # 可选的受力分析图（依赖已有外部示意图时生成）
    optional_plots = {
        "wing_fbd": analysis.plot_wing_fbd,
        "linkage_force_diagram": analysis.plot_linkage_force_diagram,
        "gear_force_diagram": analysis.plot_gear_force_diagram,
        "annotated_gear_motion_diagram": analysis.plot_annotated_existing_gear_motion_diagram,
    }
    for name, plot_fn in optional_plots.items():
        try:
            figure_paths[name] = plot_fn(results)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
    if mechanism_schematic is not None:
        figure_paths["mechanism_schematic"] = mechanism_schematic

    summary_json = analysis.write_summary_json(results)
    outputs = dict(figure_paths)
    outputs["summary_json"] = summary_json

    if write_report_template:
        # This intentionally overwrites the generated report from the template.
        # Keep this off by default because the report may have been hand edited.
        report_path = analysis.generate_report_v3(results, figure_paths, summary_json)
        outputs["report_template"] = report_path

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the mechanical-principles figures and summary JSON used "
            "by the structure analysis and curve-reading documents."
        )
    )
    parser.add_argument(
        "--write-report-template",
        action="store_true",
        help=(
            "Also regenerate output/reports/机械原理结构系统分析.md from the "
            "script template. This may overwrite hand edits."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_assets(write_report_template=args.write_report_template)

    print("Generated mechanical-principles companion assets:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
