"""v6.9 设计参数加载器 — 从 config/design_v69.yaml 读取，作为所有仿真脚本的单一参数来源。

Usage:
    from src.config import get_design, get_mech_params

    design = get_design()       # 完整 dict（机构 + 气动 + 物理 + 数值）
    mech   = get_mech_params()  # 仅机构参数 dict（给 mechanism.py 用）
"""

from pathlib import Path
import sys

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

# 向上找项目根（src/config.py → 项目根）
_PROJ = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJ / "config" / "design_v69.yaml"

_cache_design = None
_cache_mech = None


def _load_yaml():
    """加载 YAML 配置文件。"""
    global _cache_design, _cache_mech
    if _cache_design is not None:
        return _cache_design

    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {_CONFIG_PATH}\n"
            f"请确认 config/design_v69.yaml 已创建。"
        )

    if _yaml is None:
        raise ImportError("需要 PyYAML 库。请运行: pip install pyyaml")

    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = _yaml.safe_load(fh)

    _cache_design = raw
    return raw


def get_design() -> dict:
    """返回完整设计参数 dict，与旧 DESIGN_v69 dict 格式兼容。

    包含：
      - 翅膀安装 (alpha_front_deg, alpha_back_deg, phase_diff_deg)
      - 机构参数 (mech_a, mech_b, mech_R, mech_c, mech_l, phi_offset_deg, rotation)
      - 物理常数 (f, rho, m_total, I_yy, d_cg, x_front, x_back, g)
      - 数值参数 (dt, t_end, theta0_deg, steady_start)
      - 气动系数 (k_3d, C_rot, r_rot, k_clap, c_damp)
    """
    raw = _load_yaml()
    m = raw["mechanism"]
    a = raw["aero"]
    p = raw["physical"]
    n = raw["numerical"]

    return {
        "alpha_front_deg": float(a["alpha_front_deg"]),
        "alpha_back_deg": float(a["alpha_back_deg"]),
        "phase_diff_deg": float(a["phase_diff_deg"]),
        "mech_a": float(m["a"]),
        "mech_b": float(m["b"]),
        "mech_R": float(m["R"]),
        "mech_c": float(m["c"]),
        "mech_l": float(m["l"]),
        "phi_offset_deg": float(m["phi_offset_deg"]),
        "rotation": str(m["rotation"]),
        "f": float(p["f"]),
        "rho": float(p["rho"]),
        "m_total": float(p["m_total"]),
        "I_yy": float(p["I_yy"]),
        "d_cg": float(p["d_cg"]),
        "x_front": float(p["x_front"]),
        "x_back": float(p["x_back"]),
        "g": float(p["g"]),
        "k_3d": float(a["k_3d"]),
        "C_rot": float(a["C_rot"]),
        "r_rot": float(a["r_rot"]),
        "k_clap": float(a["k_clap"]),
        "c_damp": float(a["c_damp"]),
        "dt": float(n["dt"]),
        "t_end": float(n["t_end"]),
        "theta0_deg": float(n["theta0_deg"]),
        "steady_start": float(n["steady_start"]),
    }


def get_mech_params() -> dict:
    """返回仅机构参数 dict，与 mechanism.py DEFAULT_PARAMS 格式兼容。

    keys: a, b, R, c, l, phi_offset_deg
    """
    global _cache_mech
    if _cache_mech is not None:
        return _cache_mech

    raw = _load_yaml()
    m = raw["mechanism"]
    _cache_mech = {
        "a": float(m["a"]),
        "b": float(m["b"]),
        "R": float(m["R"]),
        "c": float(m["c"]),
        "l": float(m["l"]),
        "phi_offset_deg": float(m["phi_offset_deg"]),
    }
    return _cache_mech


def get_version() -> str:
    """返回版本号字符串，如 "6.9"."""
    raw = _load_yaml()
    return raw.get("version", "?.?")


def get_sweep_grid() -> dict:
    """返回扫参网格定义（笛卡尔积参数扫描用）。

    读 YAML > sweep 段。列表值 = 扫参，标量值 = 固定。
    sweep_cartesian.py 从此读取，不再硬编码 DEFAULT_GRID。

    Returns:
        {param_name: value|list}: 混合格式，标量或列表。
    """
    raw = _load_yaml()
    return raw.get("sweep", {})
