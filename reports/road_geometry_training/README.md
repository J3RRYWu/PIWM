# 几何一致性实验复现说明

本目录为探索性第三轮动力学实验，不是独立新数据验证。两条来源和划分与 `reports/controlled_holdout` 相同，使用此前冻结的六个原始 CNN。

- `geometry_euler`：从同一 Hermite 曲线计算形状、曲率及参数速度，Euler 积分。
- `geometry_midpoint`：相同网络，显式 midpoint 积分。
- `body_midpoint`：相同网络与积分器，清空道路信息及初始道路相对量。
- 全部模型从零训练，40 epochs，3 seeds × 2 directions；验证 E100 选 checkpoint。

`protocol.json` 保存实现与输入指纹。`results.json` 保存完整 18 行结果和状态诊断，`curves.npz` 保存逐窗口误差。`audit.json` 已通过完整缓存指标重算及选模检查。`checkpoint_spotcheck.json` 另从每个权重重跑 24 个均匀分布窗口，使用不同 batch size 在浮点容差内核对缓存。

几何推导见 `src/baselines/consistent_road_geometry.py`，训练实现见 `src/baselines/geometric_frenet.py`。单位速度假设检查仅使用验证集，见 `reports/road_geometry/validation_audit.json`。记录的 chart-invalid 比例是 rollout 步状态统计；不能当成所有 midpoint 内部阶段都有效的证明。

主要结果：一致性修正有数学依据，但没有在两个来源上同时胜过标定运动学。body 对照无道路输入，因此其结果不随 CNN 更换而改变。不可将修正代码错误等同于新颖的几何理论。

复核命令（仓库根目录、项目 venv）：

```text
python scripts/check_geometric_frenet.py
python scripts/run_geometric_frenet.py audit
python scripts/verify_saved_rollouts.py geometry
```

模型名称：GPT-6（Codex）；未提供更细型号标识。
