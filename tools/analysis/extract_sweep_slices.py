#!/usr/bin/env python3
"""
从 F: 盘全量笛卡尔积数据中提取单变量扫描切片,
生成 temp/stability/sweep_<param>/sweep_summary.json.

每个 param value 的统计量取全网格中该值所有 combo 的平均值,
反映该参数值在整体设计空间中的平均表现.
"""
import json, re
from pathlib import Path
import numpy as np

SRC_ROOT = Path(r'F:\重大作业考试\26秋\机械原理\全链路气动仿真\temp\stability\sweep_cartesian')
OUT_ROOT = Path('temp/stability')

PARAM_KEYS = [
    'alpha_front_deg', 'alpha_back_deg', 'phase_diff_deg',
    'mech_a', 'mech_R', 'phi_offset_deg', 'k_clap'
]

SUMMARY_KEYS = [
    'L/W', 'peak_theta_deg', 'n_exceed_90',
    'mean_Fz_body_mN', 'mean_Fz_world_mN', 'mean_Fx_body_mN',
    'mean_M_aero_uNm', 'peak_M_aero_uNm',
    'mean_M_grav_uNm', 'peak_M_grav_uNm',
    'mean_M_damp_uNm', 'peak_M_damp_uNm',
    'mean_abs_thetadot_rads', 'peak_abs_thetadot_rads',
    'mean_abs_thetaddot_rads2', 'peak_abs_thetaddot_rads2',
    'peak_alpha_eff_FL_deg', 'peak_alpha_eff_BL_deg',
    'mean_CL_FL', 'mean_CD_FL'
]


def parse_combo_id(cid: str) -> dict:
    return {
        'alpha_back_deg': int(re.search(r'ab(\d+)', cid).group(1)),
        'alpha_front_deg': int(re.search(r'af(\d+)', cid).group(1)),
        'phase_diff_deg': -int(re.search(r'phn(\d+)', cid).group(1)),
        'phi_offset_deg': -int(re.search(r'pon(\d+)', cid).group(1)),
        'mech_a': float(re.search(r'_a(\d+)_', cid).group(1)),
        'mech_R': float(re.search(r'R(\d+p?\d*)_', cid).group(1).replace('p', '.')),
        'f': int(re.search(r'_f(\d+)_', cid).group(1)),
        'k_clap': float(re.search(r'kc(\d+p?\d*)_', cid).group(1).replace('p', '.')),
    }


def main():
    print(f'[extract] scanning {SRC_ROOT} ...')
    combo_dirs = [d for d in SRC_ROOT.iterdir() if d.is_dir()]
    print(f'[extract] found {len(combo_dirs)} combo dirs')

    summaries = []
    for d in combo_dirs:
        cid = d.name
        summary_path = d / 'summary.json'
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            sm = json.load(f)
        params = parse_combo_id(cid)
        sm.update(params)
        sm['_combo_id'] = cid
        summaries.append(sm)

    print(f'[extract] loaded {len(summaries)} summaries')

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for param in PARAM_KEYS:
        sweep_dir = OUT_ROOT / f'sweep_{param}'
        sweep_dir.mkdir(parents=True, exist_ok=True)

        # group by param value
        groups = {}
        for sm in summaries:
            val = sm[param]
            groups.setdefault(val, []).append(sm)

        values = sorted(groups.keys())
        result = {
            '_param': param,
            '_n': len(values),
            '_note': 'statistics are mean across full Cartesian grid for each parameter value',
            '_value': values,
        }
        for k in SUMMARY_KEYS:
            result[k] = []
            for val in values:
                arr = [sm[k] for sm in groups[val] if k in sm and sm[k] is not None]
                if arr:
                    result[k].append(float(np.mean(arr)))
                else:
                    result[k].append(None)

        out_path = sweep_dir / 'sweep_summary.json'
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f'  {param}: {len(values)} values -> {out_path}')


if __name__ == '__main__':
    main()
