#!/usr/bin/env python3
"""Parse v6.3 scan output and save as structured JSON + re-run JSON save in scan script"""
import json, re, sys
from pathlib import Path

def parse_scan_output(text):
    """Parse the scan console output into structured data."""
    results = []
    in_data = False
    for line in text.strip().split('\n'):
        # Match data lines: optional flag, then af, ab, L/W, peak, n90, Fz, Fx, td
        m = re.match(
            r'^\s*([+!> ]{2,3})\s+(\d+)\s+(\d+)\s+([-\d.]+)\s+([\d.]+)deg\s+(\d+)\s+([-+\d.]+)\s+([-+\d.]+)\s+([\d.]+)',
            line)
        if m:
            flag = m.group(1).strip()
            results.append({
                "ai_f": int(m.group(2)),
                "ai_b": int(m.group(3)),
                "L/W": float(m.group(4)),
                "peak_deg": float(m.group(5)),
                "n90": int(m.group(6)),
                "Fz_mN": float(m.group(7)),
                "Fx_mN": float(m.group(8)),
                "td_mean": float(m.group(9)),
                "flag": flag,
            })
            in_data = True
        elif in_data and line.strip() == '':
            pass  # skip blank lines mid-data
    return results

# Read the scan output
output_path = Path(r"C:\Users\25619\AppData\Local\Temp\claude\D--code-Butterfly\24b4713d-07e5-4b75-9e78-d89ca42849eb\tasks\bwzp82ruq.output")
text = output_path.read_text(encoding="utf-8")

results = parse_scan_output(text)
print(f"Parsed {len(results)} data rows")

if not results:
    print("WARNING: No data parsed! Trying alternate parse...")
    for line in text.split('\n')[:10]:
        print(repr(line))
    sys.exit(1)

# Compute summary stats
positive = [r for r in results if r["L/W"] > 0 and r["n90"] == 0]
positive.sort(key=lambda x: x["L/W"], reverse=True)

negative = [r for r in results if r["L/W"] <= 0]
n90_nonzero = [r for r in results if r["n90"] > 0]

summary = {
    "version": "v6.3",
    "formula": "LEV/Lee hybrid (Dickinson |a|<=55, smoothstep 55-65, LEV/Lee |a|>=65)",
    "C_D_max": 3.221,
    "C_L_model": "A*sin(2a), A=1.866",
    "C_D_model": "C_D0 + A_D*(1-cos(2a)), C_D0=0.393, A_D=1.414",
    "scan_params": {
        "t_end_s": 3.0,
        "n_steps": 60000,
        "dt_us": 50,
        "alpha_f_range": [28, 32, 35, 38, 40, 42, 45, 48, 50, 55, 60],
        "alpha_b_range": [8, 10, 12, 15, 18, 20, 22, 25, 30],
        "phase": "in-phase",
        "model": "fixed (pitch-only)",
    },
    "total_combos": len(results),
    "positive_lift_count": len(positive),
    "negative_lift_count": len(negative),
    "n90_violations": len(n90_nonzero),
    "best": {
        "ai_f": positive[0]["ai_f"],
        "ai_b": positive[0]["ai_b"],
        "L/W": positive[0]["L/W"],
        "Fz_mN": positive[0]["Fz_mN"],
        "Fx_mN": positive[0]["Fx_mN"],
        "peak_deg": positive[0]["peak_deg"],
        "n90": positive[0]["n90"],
    },
    "top_20_positive": positive[:20],
    "all_results": results,
    "verification_10s": {
        "date": "2026-06-05",
        "configs": [
            {"ai_f": 60, "ai_b": 8,  "L/W": 1.033, "peak_deg": 46.7, "n90": 0, "status": "STABLE", "Fz_mN": 203, "Fx_mN": 96},
            {"ai_f": 60, "ai_b": 10, "L/W": 0.993, "peak_deg": 45.0, "n90": 0, "status": "STABLE", "Fz_mN": 195, "Fx_mN": 104},
            {"ai_f": 60, "ai_b": 12, "L/W": 0.948, "peak_deg": 43.1, "n90": 0, "status": "STABLE", "Fz_mN": 186, "Fx_mN": 112},
        ],
        "best_L/W": 1.033,
        "hover_achieved": True,
    }
}

# Save
out_path = Path(__file__).parent.parent / "data" / "v63_scan_results.json"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Saved {len(results)} results to {out_path}")
print(f"  Positive lift: {len(positive)}, Best: af={positive[0]['ai_f']}/ab={positive[0]['ai_b']} L/W={positive[0]['L/W']:.3f}")

# Also update scan_v6_3.py to auto-save JSON in future
print("\nNow updating scan_v6_3.py to auto-save JSON...")
