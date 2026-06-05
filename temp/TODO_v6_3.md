# TODO: v6.3 LEV/Lee C_L/C_D 公式实施

## 1. 修改核心公式
- [x] 修改 `temp/pitch_dynamics_v6_1.py` 中的 `cl_cd_blended()`
- [x] 创建 `temp/scan_v6_3.py` + 跑完 99 组扫描
- [x] 更新实验记录 `docs/v6_3_CL_CD_formula.md`
- [x] 对 v6.3 top-3 参数跑 10s 验证 + plot ✅ 全部通过 (最佳 L/W=1.033, 已达悬停!)
- [x] 提交 10s 验证结果

## 预计时间
- 公式修改：5 min
- 扫描 11×9=99 组 × 2 min/组 ≈ 3h（可并发）
- 验证：30 min
