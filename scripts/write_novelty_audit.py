from pathlib import Path
import json,numpy as np
R=Path.cwd();out=R/'reports/novelty_audit';out.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads((R/p).read_text(encoding='utf-8'))
t=read('reports/temporal_holdout/results.json');a=read('reports/coupled_road/results.json')
base=np.mean([t['results'][f'temporal/seed{s}/kinematic']['E100'] for s in range(3)])
ref=np.mean([t['results'][f'temporal/seed{s}/guarded_full']['E100'] for s in range(3)])
rows=[]
for d,n,label in [(t,'guarded_full','独立双头 + guarded full'),(t,'geometry_midpoint','共享形状曲线 + 几何 midpoint'),(a,'arc_full','单曲率表示 + 单位弧长圆弧')]:
    v=[d['results'][f'temporal/seed{s}/{n}']['E100'] for s in range(3)];m=float(np.mean(v));diag=[d['diagnostics'][f'temporal/seed{s}/{n}'] for s in range(3)]
    rows.append(dict(name=n,label=label,E100_mean=m,E100_seed_sd=float(np.std(v,ddof=1)),difference_from_guarded_m=m-ref,relative_cost_percent=100*(m/ref-1),reduction_vs_kinematic_percent=100*(1-m/base),max_seed_invalid_chart_percent=100*max(x['invalid_chart_fraction'] for x in diag)))
(out/'performance_tradeoff.json').write_text(json.dumps(dict(scope='Existing exploratory chronological holdout; descriptive comparison, not fresh validation or equivalence testing.',rows=rows),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
text=r'''# 创新点、已有工作与性能取舍核查

日期：2026-09-28。核查对象为当前实际代码、已审计实验及公开论文。此处为针对关键主张的定向检索，不是穷尽性查新；“未发现完全相同方法”不等于证明首创。不同任务的数据集分数不能直接比较。本次不开展新训练，不替换已冻结的模型或结果。

## 1. 结论与建议

当前最可辩护的是**具体结构设计与实证贡献**，尚不是新的基本理论：从短图像历史提取有限道路预览，在预测的道路进度处查询几何，把它与给定动作下的道路相对状态递推、位置读出连接起来；再检验几何一致性、数值保护和误差表现之间的关系。

应保留 guarded full 为当前主模型和性能参照。它是时间实验拟合前指定的模型，不应在看到测试结果后把另一家族包装成“预先选定的最终方法”。统一几何的两个版本是方法补强候选。先修正监督标签，再用冻结的验证规则判断是否提升为主模型。

只有“CNN + Frenet + 残差 + 更低误差”时，新颖性仍偏弱。较好的论文问题是：**如何让视觉估计的道路表示成为可检查的动力学与位置重建接口；这个约束在什么条件下改善预测，又会付出什么代价？** 标准几何公式不是创新，针对实际预测链条的设计、验证和可推广分析才是需要建立的贡献。

## 2. 已有工作覆盖到哪里

| 公开工作／核查来源 | 已经覆盖的内容 | 与当前工作的差异、对主张的约束 |
|---|---|---|
| [Deep Kinematic Models，ICRA 2020](https://arxiv.org/abs/1908.00219) | 将车辆运动学嵌入学习式轨迹预测。 | “神经网络结合车辆运动学”不是新增创新；我们的指定动作递推与视觉道路接口需要单独界定。 |
| [Learning to drive from a world on rails，ICCV 2021](https://arxiv.org/abs/2105.00636) | 将不受自车动作影响的环境模型与紧凑自车前向模型分解，用于策略学习。 | “车辆与环境解耦”作为一般思想已有先例；我们只能突出具体的可解释道路预览与查询机制。 |
| [PRIME，CoRL 2021／PMLR 2022](https://proceedings.mlr.press/v164/song22a.html) | 受约束的模型式轨迹生成与学习式评估相结合。 | Frenet、可行轨迹和学习评估的组合已有。它是多模态行为预测，我们是给定未来动作的车辆递推。 |
| [Frenet-based Domain Normalization，ICRA 2023](https://arxiv.org/abs/2305.17965) | 借助 Frenet 表示缓解场景几何带来的域差异。 | 不能声称首创 Frenet 学习表示或泛化作用；我们的来源外推结果也不能支撑比它更强的泛化主张。 |
| [Stay on Track，2023／2024 版本](https://arxiv.org/html/2306.00605v2) | 用车道中心线定义的 Frenet 坐标包装预测器，研究道路约束和预测效果。 | “道路坐标带来归纳偏置”及准确率与泛化的权衡已有讨论。我们的区别在视觉估计道路、动作条件递推以及预测曲线与动力学的一致性。 |
| [PERL，2023／2024 版本](https://arxiv.org/abs/2309.15284) | 物理跟驰模型加学习残差。 | “物理模型 + 神经残差”本身不新。本文不同的几何结构必须有独立证据。 |
| [End-to-end Lane Detection through Differentiable Least-Squares Fitting，ICCVW 2019](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVRSUAD/Van_Gansbeke_End-to-end_Lane_Detection_through_Differentiable_Least-Squares_Fitting_ICCVW_2019_paper.pdf) | 视觉网络与可微几何拟合共同训练，直接监督车道参数。 | 可微道路几何、显式几何监督、可解释中间量不是空白；它不直接等同于我们的动作条件递推实验。 |
| [Learning to Drive from a World Model，2025](https://arxiv.org/html/2504.19077v1) | 学习式世界模型／重投影模拟支持驾驶策略训练。 | 不能把视觉驾驶世界模型作为一般性首创。我们并未验证其闭环策略训练任务。 |
| [GeoWAM，2026-08 预印本](https://arxiv.org/abs/2608.23486) | 以视觉几何表示场景并关联自车轨迹预测。 | “从像素转向几何世界状态”的大方向已有近期工作。我们的低维道路几何及显式递推是较窄的研究对象，不能直接比较论文分数。 |
| [我们已有 PIWM，arXiv v6，2026-04](https://arxiv.org/html/2412.12870v6) | 物理变量对齐、部分已知动力学、弱分布监督、量化／视觉编码等。 | 这些属于既有基础，不能重新列为期刊新增贡献。当前道路感知采用直接地图标签，也不能说已新增验证弱监督道路学习。 |

核查深度：对当前代码及本地实验逐项核对；PIWM v6、Stay on Track、Frenet-domain-normalization 与 2025 driving-world-model 的正文可访问，其余以作者论文页、出版方摘要或可访问论文片段确认上述核心内容。没有复现这些外部方法；不能声称已经逐一排除所有实现层面的重合。检索中也遇到部分 CVF 页面 403，采用作者 arXiv 版本补充可获得内容。

## 3. 当前可以写成哪三条贡献

### C1：面向给定动作预测的视觉道路状态接口——主设计贡献，增量性

15 帧图像生成有限局部道路预览；道路几何与车辆动态量显式分开；递推使用预测进度查询预览；输出从预测几何重建，而不是每步查询测绘中心线。可检验的差异在这条具体链路，而不是单个 CNN 或坐标公式。

必须同时交代：初始 d、航向误差仍有特权信息，未来动作已知，道路标签来自地图。不能简写成完整 map-free／纯视觉自主驾驶，也不能说动态障碍或视野外道路已被建模。

### C2：道路表示、动力学查询与位置读出的一致连接——最值得补强的方法贡献

当前双头模型允许曲率与道路形状不同步。共享曲线版本从 P(q) 同时求参数速度 g 和几何曲率 κ；圆弧版本由同一曲率表示生成道路位置与切向。这样可以明确检查预测链中到底用了哪条曲线。

贡献边界：g 修正、Frenet 运动学、圆弧积分都是标准数学。数值下限被触发后是保护性延拓，学习残差也允许偏离理想运动学，所以不能宣称所有预测都严格物理可行。创新强度取决于一致连接解决了怎样的实际学习问题及其对照证据，不能只靠重新命名模块提升。

### C3：收益与失效条件的受控验证——实证贡献，不冒充新算法

同输入对照、从零训练消融、源记录与时间划分、几何诊断、完整种子、权重复算，共同回答“几何是否有用、收益依赖什么条件”。时间留出已有明显收益，来源转移没有一致优势；两者构成适用边界。

审计首先是研究严谨性的要求。只有提炼出超越单个代码 bug 的可复查规律，并经过相应对照，才应进一步作为可推广方法论贡献。

## 4. 实测性能与一致性取舍

以下仅比较已有两条记录的同一时间留出窗口，E100 为米。± 是三训练种子样本标准差，不是独立场景置信区间。圆弧模型使用新编码器，不能把它与 guarded full 的差异全部归因于一个约束。

| 版本 | E100 | 相对 guarded 的误差增加 | 相对运动学的误差降低 | 最大种子完整步无效图比例 |
|---|---:|---:|---:|---:|
'''
for r in rows:text+=f"| {r['label']} | {r['E100_mean']:.3f} ± {r['E100_seed_sd']:.3f} | {r['difference_from_guarded_m']:.3f} m（{r['relative_cost_percent']:.1f}%） | {r['reduction_vs_kinematic_percent']:.1f}% | {r['max_seed_invalid_chart_percent']:.4f}% |\n"
text+=r'''
- guarded full：当前准确率最强，但两个道路头不保证一致。
- geometry midpoint：误差代价约 2.9 cm，是有价值的折中候选；无效图比例较高，不能说它在所有维度都更好。
- arc full：误差代价约 5.7 cm，生成曲线具单位参数速度，完整步监测中没有出现非正 Frenet 分母；这只是观测结果，不是安全证明，midpoint 内部阶段未被完整监测。

同组圆弧对照中，新耦合编码器的 0.400 m 优于原独立编码器配相同圆弧解码的 0.468 m，约降低 14.5%。它支持耦合监督在本实验里的作用；仍不能据此声称圆弧方法整体胜过 0.343 m 的双头方法。

这些都是开发过程中的探索性结果。不能因为结果接近就宣称统计等价，也不能自行认定 5.7 cm 在车辆任务中无关紧要。总体来源外推结果仍须保留；新颖性不能根据表里的误差排名打分。

## 5. 推荐的平衡路线

1. 保留当前 guarded full 为已建立的性能参照，主文对它的优点如实报告；不要为了让标题更新颖而隐藏表现更好的版本。
2. 先统一几何标签：平滑后重新按弧长参数化，曲率、形状、进度和初始道路状态来自同一几何定义。旧的 10.450 m 名义周长与约 9.743 m 平滑后折线长度不应被当成精确一致。
3. 用相同划分、预算、种子，复核当前双头、共享曲线和单曲率三条路线及关键同输入对照；所有道路模型用同一修正监督，原始 xy 评价真值不变。
4. 主模型选择依据预先固定的验证规则，结合 E100、表示一致性误差、无效图／近奇异比例和计算成本。容忍的精度损失应有任务依据，不能看完测试再定门槛。测试完整报告，旧测试已使用的事实不改变。
5. 若严格单一几何仍损失较大，可只增加一个“软一致性双头”候选：在保留双头自由度的同时，惩罚曲率头与形状导出曲率的偏差。曲线若用一般参数 q，应显式使用 g，而不是无条件强迫 g=1。损失权重仅按验证集选择。这是尚未实现的候选，不是已证实的新发明；可微几何与一致性正则也有大量先例。

目前不建议扩展 Transformer、大世界模型或额外任务来装饰创新点。也不建议为了保留“弱监督”标题，再把一个尚未验证的监督设定硬加进来。优先让道路表示这一项贡献有清楚的数学接口、可信对照及可重复收益。

## 6. 最能区分新贡献的对照

- 静态预览拼接的 Cartesian 控制已有；还应在相同容量和保护下，比较同一结构的“按预测进度查询”“固定起点几何”“与进度无关的摘要”，并各自从零训练。这样才能将收益归因于道路查询机制，单纯测试时置零不足以替代。
- 双头／共享曲线／单曲率需在一致标签与共同训练条件下比较，同时报告道路重建误差、几何一致性误差及轨迹误差。只改善几何却损失预测精度也必须报告。
- 检查数值保护是否解释了相当部分收益：同输入 Cartesian 应有可比的增量约束。若强对照缩小差距，更新贡献表述。

这是补强路线，不是本次已经完成的实验。两来源都曾被研究，再加一轮不会获得新的独立验证集。

## 7. 可用于引言的贡献表述草案

> We study a compact visual road-context representation for action-conditioned vehicle prediction. A finite preview inferred from image history is queried at predicted progress and coupled to road-relative state updates and local position reconstruction. We compare independent and geometrically coupled representations to expose the interaction between prediction accuracy, geometric consistency, and coordinate validity. Controlled retrospective evaluations quantify the benefits and limitations under matched information conditions and distinguish chronological prediction from transfer between recordings.

这里使用 study / compare / quantify，避免尚未被查新与实验支持的 first、guaranteed、universally superior。待完成统一标签复核后，再决定是否把“geometrically coupled representation”提升为主方法标题。

当前建议标题方向：**Visual Road Context for Structured Vehicle Prediction: Accuracy and Geometric Consistency**。这是定位建议，不自动替换主稿。

## 8. 本地证据与状态

- 当前实现：`src/baselines/frenet_diagnostic.py`、`geometric_frenet.py`、`unit_arc_geometry.py`。
- 性能来源：`reports/temporal_holdout/results.json`、`reports/coupled_road/results.json`；比较数据另存 `reports/novelty_audit/performance_tradeoff.json`。
- 来源外推限制及原评分：`Jounral_PIWM/RAS_revision_audit_2026-09-27.md`。
- 本次查新没有证明强理论原创性，也没有理由只因定位改变就提高原先 3/5 的模拟 Overall 评分。它给出的价值是更明确、可执行的贡献边界与模型选择路线。
- 本次未修改论文正文、冻结训练源码或 checkpoint；未把建议实验写成完成结果。可审核后将本核查文档纳入 commit。

模型名称：GPT-6（Codex）；当前会话未提供更细型号标识。
'''
p=R/'Jounral_PIWM/novelty_performance_audit_2026-09-28.md';p.write_text(text,encoding='utf-8');print(p);print(json.dumps(rows,ensure_ascii=False,indent=2))
