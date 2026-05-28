# Butterfly Wing Aerodynamics — Agent Notes

## Project Overview

This project analyzes the aerodynamic performance of bionic butterfly wings designed in **SolidWorks** (`Wings.SLDPRT`).

The workflow is:
1. **SolidWorks**: Design wing planforms (forewing & hindwing) and rotation axis
2. **DXF Export**: Export sketches to DXF for Python processing
3. **Python Analysis**: Parse DXF → compute geometric parameters (area, span, AR, chord distribution, area moments) → quasi-steady aerodynamic estimates

## File Structure

```
Butterfly/
├── Wings.SLDPRT                    # SolidWorks part file (wing geometry)
├── WingFront.DXF                   # Forewing sketch exported from SolidWorks
├── WingBack.DXF                    # Hindwing sketch exported from SolidWorks
├── WingsAxis.DXF                   # Rotation axis (two circles defining hinge line)
├── 仿生蝴蝶翅膀空气动力学分析文献综述.md   # Literature review with formulas
├── AGENTS.md                       # This file
│
├── analyze_dxf.py                  # Main analysis script (read DXF → geometry + aero)
├── plot_wings.py                   # Quick plotting script for raw XY data
├── wing_analysis.png               # Output: analysis plots
├── wing_analysis_results.json      # Output: computed parameters (JSON)
└── chord_distribution.csv          # Output: chord distribution data
```

## SolidWorks Sketch Naming Convention

| Sketch Name | Content | Export File |
|-------------|---------|-------------|
| `草图100` / `WingFront` | Forewing planform (closed contour) | `WingFront.DXF` |
| `草图101` / `WingBack`  | Hindwing planform (closed contour) | `WingBack.DXF` |
| `草图102`              | Rotation axis (two circles)        | `WingsAxis.DXF` |

> **Important**: DXF exports use **sketch-local coordinates** (mm), not global part coordinates. The axis is defined by the two circle centers in the axis sketch.

## DXF Export Procedure

When re-exporting from SolidWorks:

1. Enter sketch edit mode (e.g., `草图100`)
2. `File` → `Save As`
3. Filename: `WingFront.DXF`, Type: `DXF (*.dxf)`
4. Click **Options**:
   - File format: `R2000-2002` or higher
   - Scale output: `1:1`
   - Check **"Output active sketch geometry only"** (exact wording varies by SW version)
5. Repeat for `WingBack.DXF` and `WingsAxis.DXF`

## Python Analysis Pipeline

### Main Script: `analyze_dxf.py`

```bash
python analyze_dxf.py
```

**What it does:**
1. Parses DXF entities (`SPLINE`, `LINE`, `CIRCLE`)
2. Connects disconnected segments by endpoint matching (greedy algorithm)
3. Computes polygon area (shoelace formula)
4. Transforms to local coordinates with rotation axis as reference
5. Computes chord distribution by spanwise binning
6. Calculates area moments: $\hat{r}_1$, $\hat{r}_2^2$
7. Runs quasi-steady aerodynamic estimates (lift, Reynolds number, power)

**Outputs:**
- `wing_analysis.png` — 6-panel figure (global, local, chord distribution, parameter table)
- `wing_analysis_results.json` — all numeric results
- `chord_distribution.csv` — spanwise chord data for both wings

### Quick Plot: `plot_wings.py`

```bash
python plot_wings.py
```

Simple raw XY plot from CSV/DXF data. Useful for visual sanity check.

## Key Parameters & Units

All DXF data is in **millimeters (mm)**. The script converts to meters internally via `MM_TO_M = 1e-3`.

| Parameter | Symbol | Unit | Typical Value (This Design) |
|-----------|--------|------|----------------------------|
| Forewing area | $S_f$ | cm² | ~112 |
| Hindwing area | $S_b$ | cm² | ~17 |
| Forewing span | $R_f$ | mm | ~170 |
| Hindwing span | $R_b$ | mm | ~132 |
| Forewing AR | $AR_f$ | — | ~2.6 |
| Hindwing AR | $AR_b$ | — | ~10.4 |
| Axis length | — | mm | ~86 |

Aerodynamic defaults (editable in `AERO_PARAMS` dict):
- Air density $\rho = 1.225$ kg/m³
- Flapping frequency $f = 10$ Hz
- Flapping amplitude $\Phi_{max} = 80°$
| Angle of attack $\alpha = 45°$
- Total mass $m = 0.0216$ kg

## Known Issues & Lessons Learned

### 1. Do NOT use `GetSketchSegments` for export
The original VBA macro used `sketch.GetSketchSegments`, which:
- Returns segments in random order
- May include reference lines / symmetry axes as regular geometry
- Produces disconnected segments (gaps of 10,000 mm observed)

**Solution**: Use DXF export instead. It preserves exact NURBS curves and outputs contours in correct order.

### 2. `SketchContour` API incompatible
`sketch.GetSketchContours` + `swContour.IsHole` throws **Error 438** on some SolidWorks versions. Avoid this API path.

### 3. Global vs Local Coordinates
`GetSketchSegments` returns **global part coordinates** (can be thousands of mm away from origin). DXF export uses **sketch-local coordinates** (proper wing-scale, ~100-200 mm). Always use DXF for dimensional analysis.

### 4. Integer Overflow in VBA
When computing sample count per segment, use `Long` not `Integer`:
```vba
' BAD:  Integer overflows at 32767
Dim nPts As Integer: nPts = segLen * 2000   ' Error 6 if segLen > 16

' GOOD: Long with reasonable density
Dim nPts As Long: nPts = CLng(segLen * 3)   ' ~3 pts per mm
```

## Dependencies

```
numpy
pandas
scipy
matplotlib
```

No `ezdxf` required — the parser is self-contained in `analyze_dxf.py`.

## To-Do / Extensions

- [ ] Validate DXF parsing against more complex spline types (weights, rational B-splines)
- [ ] Add mass property estimation from wing thickness & material density
- [ ] Export mesh for CFD (Fluent/OpenFOAM) from SolidWorks
- [ ] Couple with flight dynamics simulation using computed $C_L(\alpha)$, $C_D(\alpha)$

## References

See `仿生蝴蝶翅膀空气动力学分析文献综述.md` for full literature review and formula derivations.
