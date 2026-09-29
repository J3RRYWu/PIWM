# 共享道路曲线的局部等价性与误差敏感性

日期：2026-09-28。目的：限定方法主张，并解释为什么需要曲率分支对照。下面是标准几何的推导与代码核查，不列为新的 Frenet 理论。

## 1. 适用域

令预测道路为 P(q)，g=|P'(q)|，单位切向 t、单位法向 n、道路航向 theta，曲率 kappa。位置 x=P(q)+d n(q)，车辆航向 alpha=theta(q)+psi。考察曲线分段内的光滑区域，要求 g>0、h=1-d kappa>0，并且局部道路投影唯一；应用当前未触发保护的实现，还要求 g>0.1、h>0.2。

正的 h 只排除了局部雅可比奇异，并不能证明全局投影唯一。结点、切向外推连接、分母 clamp、角度回绕切口及退化曲线必须另行处理。

## 2. 连续时间的坐标抵消

曲线满足 d theta/dq=kappa g，dn/dq=-kappa g t，因此

$$\dot x=g(1-d\kappa)\dot q\,t+\dot d\,n.$$

在没有位姿残差的模型中，代入

$$\dot q=\frac{v\cos\psi}{g(1-d\kappa)},\quad
\dot d=v\sin\psi,\quad
\dot\psi=\omega-\kappa g\dot q$$

得到

$$\dot x=v\cos\psi\,t+v\sin\psi\,n
=v[\cos\alpha,\sin\alpha]^\top,\qquad\dot\alpha=\omega.$$

这是相同几何用于递推和读出时的局部恒等式。它不表示两个有限步长离散网络的预测相同；神经网络输入、位姿残差、数值离散与坐标保护均可破坏该等价条件。当前主模型的独立曲率与形状头也不满足严格共享几何的前提。

若横向与航向残差按连续时间率 r_d、r_psi 表示，则平面位置速度多出 r_d n，航向角速度多出 r_psi。当前实现按固定 1/22 s 步长学习增量，其对应率为增量除以 dt；改变 dt 时不能保持增量不变再声称步长收敛。

**论文含义：** 道路信息的收益应通过学习分支的归纳偏置、可观测性代理、残差方向约束与离散处理来检验，不能把坐标变换本身解释成新增车辆运动定律。当前 d/psi 仍进入网络，屏蔽曲率并不使网络完全独立于道路。

## 3. 固定状态下的局部敏感性

将 g 和 kappa 暂视为局部独立输入，在未触发保护的区域：

$$\frac{\partial\dot q}{\partial\kappa}=\frac{v\cos\psi\,d}{g h^2},\qquad
\frac{\partial\dot q}{\partial g}=-\frac{v\cos\psi}{g^2h},\qquad
\frac{\partial\dot\psi}{\partial\kappa}=-\frac{v\cos\psi}{h^2}.$$

在 |v|≤V、|d|≤D、g≥g_min、h≥delta 的紧致局部区域，相应绝对值上界分别为 VD/(g_min delta²)、V/(g_min² delta)、V/delta²。该结果解释接近坐标奇异处的几何误差放大，但不是最终平面误差的直接下界：上节的共同几何抵消可能消除部分坐标误差。

共享曲线中 g、kappa 实际相关；完整误差分析还须包含读出误差、状态变化、神经分支导数、几何估计相关性及边界切换。可以在规定域内写局部 Lipschitz 递推界，但没有经验证的全域常数时，不声称全局有界误差或稳定性。沿有限样本测得的最大导数也不是全域上界。

## 4. 已完成的代码核验

`reports/road_branch_diagnosis/local_geometry_checks.json`：128 个 float64 状态，使用实际 Hermite 读出 `consistent_pose` 和实际零残差 `GeometricFrenet.field`；在光滑内域通过自动微分计算读出速度。

- 平面速度／航向角速度与上述等价式最大绝对差：1.7763568394002505e-15。
- 三个局部偏导式与自动微分最大绝对差分别为 1.3877787807814457e-17、4.440892098500626e-16、4.440892098500626e-16。
- 此检查不覆盖所有曲线、坐标边界、非零学习残差或闭环部署，不是新的性能实验。

审计脚本：`scripts/audit_road_branch_diagnosis.py`；数据诊断脚本及协议独立冻结，原模型和原结果保留。

模型名称：GPT-6（Codex）。
