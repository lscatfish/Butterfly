#!/usr/bin/env python3
"""
扫描 F:\...\sweep_cartesian 下所有 combo 目录,
重新生成完整的 sweep_summary.json (追加/汇总所有已跑组合).
"""
import json, re, shutil
from pathlib import Path
from datetime import datetime

SRC_ROOT = Path(r'F:\重大作业考试\26秋\机械原理\全链路气动仿真\temp\stability\sweep_cartesian')
SUMMARY_PATH = SRC_ROOT / 'sweep_summary.json'

PARAM_KEYS = [
    'alpha_front_deg', 'alpha_back_deg', 'phase_diff_deg',
    'mech_a', 'mech_R', 'phi_offset_deg', 'f', 'c_damp', 'rotation', 'k_clap'
]

SUMMARY_KEYS = [
    'L/W', 'L/W_body', 'peak_theta_deg', 'n_exceed_90',
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
        'c_damp': float(re.search(r'cd(\d+p\d+)_', cid).group(1).replace('p', '.')),
        'rotation': re.search(r'rot(\w+)', cid).group(1),
    }


def main():
    print(f'[rebuild] scanning {SRC_ROOT} ...')
    combo_dirs = [d for d in SRC_ROOT.iterdir() if d.is_dir()]
    print(f'[rebuild] found {len(combo_dirs)} combo dirs')

    results = []
    for d in combo_dirs:
        cid = d.name
        summary_path = d / 'summary.json'
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            sm = json.load(f)
        params = parse_combo_id(cid)
        sm['_combo_id'] = cid
        for k in PARAM_KEYS:
            sm[f'_param_{k}'] = params[k]
        results.append(sm)

    print(f'[rebuild] loaded {len(results)} summaries')

    # backup old summary
    if SUMMARY_PATH.exists():
        backup = SRC_ROOT / f'sweep_summary.json.bak.{datetime.now():%Y%m%d_%H%M%S}'
        shutil.copy2(SUMMARY_PATH, backup)
        print(f'[rebuild] backed up old summary to {backup}')

    # build summary
    result = {'_combo_id': [r['_combo_id'] for r in results]}
    for k in SUMMARY_KEYS:
        result[k] = [r.get(k) for r in results]
    for k in PARAM_KEYS:
        result[f'_param_{k}'] = [r[f'_param_{k}'] for r in results]

    # grid
    result['_grid'] = {k: sorted({r[f'_param_{k}'] for r in results}) for k in PARAM_KEYS}
    result['_n_combos'] = len(results)
    result['_param_keys'] = PARAM_KEYS
    result['_rebuilt_at'] = datetime.now().isoformat()

    with open(SUMMARY_PATH, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'[rebuild] saved {SUMMARY_PATH} with {len(results)} combos')


if __name__ == '__main__':
    main()
