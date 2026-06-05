# TODO: v6.3 LEV/Lee C_L/C_D 公式实施

## 1. 修改核心公式
- [x] 修改 `temp/pitch_dynamics_v6_1.py` 中的 `cl_cd_blended()`
  - |alpha| <= 60: Dickinson 经验公式（不变）
  - |alpha| > 60: C_L = 1.866*sin(2*alpha), C_D = 0.393 + 1.414*(1-cos(2*alpha))
  - 60deg 处连续匹配

## 2. 全参数重新扫描
- [ ] 创建 `temp/scan_v6_3.py`
  - alpha_f in [28, 30, 32, 35, 38, 40, 42, 45, 50, 55, 60]
  - alpha_b in [10, 15, 18, 20, 22, 25, 28, 30, 12]
  - 同相 only（反相已知不稳定）
  - t=3s, n=60000, 检查 n90
  - 可以同时跑几个进程加速

## 3. 对比验证
- [ ] 对比 v6.2 混合模型 vs v6.3 最佳参数
  - 升力/推力/pitch 峰值
  - C_L/C_D 使用分布（经验 vs LEV 占比）

## 4. 长时验证
- [ ] 对 v6.3 top-3 参数跑 10s 验证

## 5. 记录 & 提交
- [ ] 更新实验记录 `docs/v6_3_CL_CD_formula.md`
- [ ] 提交 git

## 预计时间
- 公式修改：5 min
- 扫描 11×9=99 组 × 2 min/组 ≈ 3h（可并发）
- 验证：30 min
