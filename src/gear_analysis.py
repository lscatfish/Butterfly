"""
定轴轮系分析模块
================
两级定轴外啮合圆柱齿轮减速器（等变位传动）

结构:
    轴I  —— 齿轮1 (z1=7,  正变位)
    轴II —— 齿轮2 (z2=40, 负变位) + 齿轮2'(z2'=7, 正变位) [双联固连]
    轴III—— 齿轮3 (z3=40, 负变位)

啮合: 1↔2, 2'↔3 (均为外啮合)
设计: 等变位 x1+x2=0, x2'+x3=0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


# ────────────────────────────────────────────
# 数据类
# ────────────────────────────────────────────

@dataclass
class GearSpec:
    """单个渐开线圆柱齿轮的几何参数。"""

    label: str
    z: int           # 齿数
    x: float         # 变位系数
    m: float         # 模数 mm
    alpha: float      # 压力角 rad
    ha_star: float    # 齿顶高系数
    c_star: float     # 顶隙系数

    # 以下由 _calculate 填充
    d: float = field(init=False)      # 分度圆直径
    db: float = field(init=False)     # 基圆直径
    da: float = field(init=False)     # 齿顶圆直径
    df: float = field(init=False)     # 齿根圆直径
    ha: float = field(init=False)     # 齿顶高
    hf: float = field(init=False)     # 齿根高
    h: float = field(init=False)      # 全齿高
    p: float = field(init=False)      # 齿距
    s: float = field(init=False)      # 分度圆齿厚
    pb: float = field(init=False)     # 基圆齿距
    alpha_a: float = field(init=False) # 齿顶圆压力角 rad

    def __post_init__(self) -> None:
        self._calculate()

    # ---- 内部计算 ----

    @staticmethod
    def _inv(angle: float) -> float:
        """渐开线函数 inv(α) = tan(α) - α"""
        return math.tan(angle) - angle

    def _calculate(self) -> None:
        z, x, m, a = self.z, self.x, self.m, self.alpha

        self.d = m * z
        self.db = self.d * math.cos(a)
        self.ha = m * (self.ha_star + x)
        self.hf = m * (self.ha_star + self.c_star - x)
        self.da = self.d + 2 * self.ha
        self.df = self.d - 2 * self.hf
        self.h = self.ha + self.hf
        self.p = math.pi * m
        self.s = math.pi * m / 2 + 2 * x * m * math.tan(a)
        self.pb = math.pi * m * math.cos(a)

        # 齿顶圆压力角
        if self.da > self.db > 0:
            self.alpha_a = math.acos(self.db / self.da)
        else:
            self.alpha_a = 0.0

    # ---- 便捷输出 ----

    def as_dict(self) -> dict[str, float]:
        return {
            "label": self.label,
            "z": self.z,
            "x": self.x,
            "d": self.d,
            "db": self.db,
            "da": self.da,
            "df": self.df,
            "ha": self.ha,
            "hf": self.hf,
            "h": self.h,
            "p": self.p,
            "s": self.s,
            "pb": self.pb,
            "alpha_a_deg": math.degrees(self.alpha_a),
        }


@dataclass
class MeshPair:
    """一对啮合齿轮的参数。"""

    label: str
    gear_a: GearSpec   # 主动轮
    gear_b: GearSpec   # 从动轮
    m: float
    alpha: float        # 压力角 rad

    # 计算结果
    a_std: float = field(init=False)     # 标准中心距
    a_actual: float = field(init=False)   # 实际中心距
    alpha_prime: float = field(init=False) # 啮合角 rad
    u: float = field(init=False)          # 齿数比
    eps_alpha: float = field(init=False)   # 端面重合度
    i_ratio: float = field(init=False)     # 传动比 (含符号, 外啮合为负)

    def __post_init__(self) -> None:
        self._calculate()

    def _calculate(self) -> None:
        za, zb = self.gear_a.z, self.gear_b.z

        # 标准中心距
        self.a_std = self.m * (za + zb) / 2

        # 等变位传动: a' = a, α' = α
        self.a_actual = self.a_std
        self.alpha_prime = self.alpha

        # 齿数比
        self.u = zb / za

        # 重合度
        ga, gb = self.gear_a, self.gear_b
        eps = (za * (math.tan(ga.alpha_a) - math.tan(self.alpha_prime))
             + zb * (math.tan(gb.alpha_a) - math.tan(self.alpha_prime))) / (2 * math.pi)
        self.eps_alpha = eps

        # 传动比 (外啮合取负)
        self.i_ratio = -zb / za

    def as_dict(self) -> dict[str, float]:
        return {
            "label": self.label,
            "a_std": self.a_std,
            "a_actual": self.a_actual,
            "alpha_prime_deg": math.degrees(self.alpha_prime),
            "u": self.u,
            "eps_alpha": self.eps_alpha,
            "i_ratio": self.i_ratio,
        }


# ────────────────────────────────────────────
# 轮系分析器
# ────────────────────────────────────────────

class FixedAxisGearTrain:
    """两级定轴圆柱齿轮减速器分析。"""

    def __init__(
        self,
        teeth: tuple[int, int, int, int] = (7, 40, 7, 40),
        module: float = 0.3,
        alpha_deg: float = 20.0,
        ha_star: float = 1.0,
        c_star: float = 0.25,
        eta_gear: float = 0.98,
        eta_bearing: float = 0.99,
    ) -> None:
        self.z1, self.z2, self.z2p, self.z3 = teeth
        self.m = module
        self.alpha = math.radians(alpha_deg)
        self.alpha_deg = alpha_deg
        self.ha_star = ha_star
        self.c_star = c_star
        self.eta_gear = eta_gear
        self.eta_bearing = eta_bearing

        # 不根切最小齿数
        self.z_min = 2 * ha_star / (math.sin(self.alpha) ** 2)

        # 等变位系数
        self.x1 = round((self.z_min - self.z1) / self.z_min, 4)
        self.x2 = -self.x1
        self.x2p = round((self.z_min - self.z2p) / self.z_min, 4)
        self.x3 = -self.x2p

        # 创建齿轮对象
        self.gear1 = GearSpec("1", self.z1, self.x1, self.m, self.alpha, ha_star, c_star)
        self.gear2 = GearSpec("2", self.z2, self.x2, self.m, self.alpha, ha_star, c_star)
        self.gear2p = GearSpec("2'", self.z2p, self.x2p, self.m, self.alpha, ha_star, c_star)
        self.gear3 = GearSpec("3", self.z3, self.x3, self.m, self.alpha, ha_star, c_star)

        # 创建啮合对
        self.mesh12 = MeshPair("1-2", self.gear1, self.gear2, self.m, self.alpha)
        self.mesh2p3 = MeshPair("2'-3", self.gear2p, self.gear3, self.m, self.alpha)

        # 总传动比
        self.i_total = self.mesh12.i_ratio * self.mesh2p3.i_ratio

        # 总效率
        self.eta_total = eta_gear ** 2 * eta_bearing ** 3

    # ---- 打印 ----

    def print_summary(self) -> None:
        """在控制台打印关键结果。"""
        print("=" * 60)
        print("定轴轮系分析结果")
        print("=" * 60)

        print(f"\n--- 基本参数 ---")
        print(f"  模数 m = {self.m} mm")
        print(f"  压力角 α = {self.alpha_deg}°")
        print(f"  不根切最小齿数 z_min = {self.z_min:.2f}")

        print(f"\n--- 变位系数 (等变位 x1+x2=0) ---")
        print(f"  x1 = {self.x1:+.4f}  x2 = {self.x2:+.4f}")
        print(f"  x2'= {self.x2p:+.4f}  x3 = {self.x3:+.4f}")

        print(f"\n--- 传动比 ---")
        print(f"  i12   = {self.mesh12.i_ratio:+.4f}")
        print(f"  i2'3  = {self.mesh2p3.i_ratio:+.4f}")
        print(f"  i13   = {self.i_total:+.4f}")
        print(f"  |i13| = {abs(self.i_total):.4f}")

        print(f"\n--- 中心距 ---")
        print(f"  a12   = {self.mesh12.a_actual:.3f} mm")
        print(f"  a2'3  = {self.mesh2p3.a_actual:.3f} mm")

        print(f"\n--- 重合度 ---")
        print(f"  ε12   = {self.mesh12.eps_alpha:.4f}")
        print(f"  ε2'3  = {self.mesh2p3.eps_alpha:.4f}")

        print(f"\n--- 效率 ---")
        print(f"  η_total = {self.eta_total:.4f} ({self.eta_total * 100:.2f}%)")

        print(f"\n--- 齿轮参数明细 ---")
        for g in (self.gear1, self.gear2, self.gear2p, self.gear3):
            d = g.as_dict()
            print(f"  齿轮{d['label']:>2s}  z={d['z']:>2d}  x={d['x']:+.4f}  "
                  f"d={d['d']:.3f}  da={d['da']:.3f}  df={d['df']:.3f}  "
                  f"ha={d['ha']:.3f}  hf={d['hf']:.3f}  s={d['s']:.4f}")

        print("=" * 60)


# ────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────

def main() -> None:
    train = FixedAxisGearTrain(
        teeth=(7, 40, 7, 40),
        module=0.3,
        alpha_deg=20.0,
        ha_star=1.0,
        c_star=0.25,
    )

    train.print_summary()



if __name__ == "__main__":
    main()