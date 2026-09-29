# 道路分支修改前的无重训诊断

仅训练／验证分区；全部三种子。真值预览替换是特权诊断，不能当作部署结果或因果归因。

| 划分 | 分区 | 模型 | 预测预览 E100 | 真值预览 E100 |
|---|---|---|---:|---:|
| temporal | train | guarded_full | 0.233 ± 0.009 | 0.219 ± 0.004 |
| temporal | train | geometry_midpoint | 0.276 ± 0.005 | 0.250 ± 0.012 |
| temporal | validation | guarded_full | 0.209 ± 0.008 | 0.168 ± 0.005 |
| temporal | validation | geometry_midpoint | 0.260 ± 0.016 | 0.212 ± 0.011 |
| outer0 | train | guarded_full | 0.350 ± 0.019 | 0.275 ± 0.020 |
| outer0 | train | geometry_midpoint | 0.419 ± 0.073 | 0.320 ± 0.053 |
| outer0 | validation | guarded_full | 0.565 ± 0.022 | 0.467 ± 0.030 |
| outer0 | validation | geometry_midpoint | 0.536 ± 0.034 | 0.432 ± 0.041 |
| outer1 | train | guarded_full | 0.255 ± 0.046 | 0.223 ± 0.003 |
| outer1 | train | geometry_midpoint | 0.223 ± 0.063 | 0.167 ± 0.054 |
| outer1 | validation | guarded_full | 0.207 ± 0.055 | 0.184 ± 0.033 |
| outer1 | validation | geometry_midpoint | 0.213 ± 0.039 | 0.181 ± 0.006 |

每窗口位置、速度与角速度误差、训练分位分组、动作支持范围、初始及 midpoint 阶段监测均保存在各 job 的 arrays.npz / summary.json。分母触发保护不等于坐标已失效；局部正分母也不保证全局投影唯一。

模型名称：GPT-6（Codex）。
