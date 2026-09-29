"""Consolidated review from complete audited results, separate from initial review."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text(encoding='utf-8'))
def link(label,p):return f'[{label}](<{(R/p).as_posix()}>)'
def mean_sd(data,key,metric='E100'):
    v=[data[key.format(s=s)][metric] for s in range(3)]
    return f'{np.mean(v):.3f} ± {np.std(v,ddof=1):.3f}'
for name in ['controlled_holdout','frenet_diagnostic','road_geometry_training','conditioned_dynamics','temporal_holdout','coupled_road']:
    assert read(f'reports/{name}/audit.json')['status']=='PASS'
for name in ['road_geometry_training','conditioned_dynamics','temporal_holdout','coupled_road']:
    assert read(f'reports/{name}/checkpoint_spotcheck.json')['status']=='PASS'
t=read('reports/temporal_holdout/results.json')['results'];a=read('reports/coupled_road/results.json')['results'];original=read('reports/controlled_holdout/test_results.json')['results'];cond=read('reports/conditioned_dynamics/results.json')['results']
lines=['# RAS 修订后严格复审与投稿判断','',
'日期：2026-09-27。对象：完成六轮受控／探索性实验后的修订稿；本报告不覆盖原始审稿记录。',
'**最终模拟评分：3/5，Major Revision（大修后再审）。** 原始稿为 2/5 Reject；升分来自新实验与审计证据，不是文字润色。当前已具备按有限范围实证研究讨论送审的基础，但尚不能判断为“有较强概率被 RAS 接收”。作者信息未完成，文件也不是可直接上传的终稿。',
'', '未获得 RAS Editorial Manager 内部审稿表，作者也没有该表。因此下列栏目参照 [Elsevier 公开审稿指南](https://www.elsevier.com/reviewer/how-to-review)，评分为本报告自拟，不能冒充官方评分。1=强烈拒稿，2=拒稿，3=大修后再审，4=小修，5=接收；Overall 是综合判断，不机械平均。','',
'## 1. 逐项复审','',
'| 项目 | 评分／判断 | 依据 |','|---|---|---|',
'| Title / Abstract | 基本准确 | 已限定为已知初始状态、给定未来动作的离线预测，同时报告时间留出收益与来源外推限制。 |',
'| Originality 创新性 | 2/5 | 视觉道路预览、Frenet 坐标、学习残差的组合有工程价值；Frenet 公式、曲率积分和弧长修正本身是标准几何。尚不足以把它们包装为新的基本原理。 |',
'| Significance 贡献 | 3/5 | 明确状态表示、几何一致性诊断、匹配对照和可复算收益形成可讨论的贡献；适用范围仍窄。 |',
'| Methods 技术严谨性 | 3/5 | 已区分信息条件，修复不受支持的 Markov／有界性主张，补数值和几何核查；标签仍为近似几何，受保护更新仍可能越出有效坐标图。 |',
'| Results 实验充分性 | 3/5 | 完整保存所有种子和对照，验证选模、时间隔离、相同原始位置真值；两个已研究过的记录不能提供独立多场景验证。 |',
'| Discussion 解释 | 4/5 | 正面结果与不利外推结果并列，不把有限误差、优化稳定或编码器改善等同于普遍预测优势。 |',
'| References 相关工作 | 3/5 | 补充 Frenet 与几何预测相关工作并收窄优先权主张；不是穷尽式文献系统综述，已有书目仍有少量页码字段待核实。 |',
'| Clarity 清晰性 | 4/5 | 正文已按实际 CNN、训练、初始化和读出重写；43 页中保留较多会议／历史材料，定稿时可以进一步压缩。 |',
'| Reproducibility 可复现性 | 4/5 | 保存协议、代码与数据指纹、权重、历史、逐窗口误差，新增实验已通过审计；外部发布与完整独立复现尚待作者落实。 |',
'| Scope 期刊匹配 | 4/5 | 机器人运动预测与结构化建模方向合适；最终录用取决于编辑对增量与实证范围的评价。 |',
'| **Overall** | **3/5：Major Revision** | **已有实质改善，不能给 Accept 或高概率接收承诺。** |','',
'## 2. 最重要的新证据','',
'两来源分别按时间划分并在边界留出至少 115 帧。39 段训练、13 段验证、15 段指定测试、4 段隔离；其中 14 个测试片段长度足够，共 527 个重叠预测窗口。三种子均保留，CNN 25 epochs、动力学 40 epochs，按验证指标选 checkpoint。表中为 E100（米），均值 ± 种子样本标准差。','',
'| 时间留出方法 | E100 |','|---|---:|']
for name,label in [('kinematic','标定运动学'),('cartesian_no_road','Cartesian，无道路'),('cartesian_road','Cartesian，同道路输入'),('guarded_full','预先指定主模型：guarded full'),('guarded_shape','shape-only'),('geometry_midpoint','形状导出几何 + midpoint')]:lines.append(f'| {label} | {mean_sd(t,"temporal/seed{s}/"+name)} |')
lines+=['',
'完整模型相对标定运动学降低 E100 约 **61.8%**，相对同输入 Cartesian 降低约 **68.6%**。两个来源分别报告时同样改善；先平均种子、再按片段聚合后，对三个主要基线均为 **14/14** 片段改善。相对 shape-only 的平均降低约 **16.7%**，只有 **10/14** 片段改善，不能称为处处占优。也不能把 E100 优势写成所有短期指标均最优。',
'这些是描述性配对结果。窗口与片段有关联，种子标准差不是独立采集记录的置信区间；没有把 527 个窗口当成 527 次独立试验。训练仅有两档油门，时间划分也不是严格同分布测试。',
'',link('时间留出完整结果','reports/temporal_holdout/RESULTS.md')+'；'+link('配对与数值审计','reports/temporal_holdout/PAIRED_AUDIT.md')+'。','',
'单位弧长耦合模型进一步把曲率和位置绑定到同一条分段圆弧：','',
'| 最后一轮方法 | E100 |','|---|---:|']
for name,label in [('arc_full','耦合道路完整模型'),('arc_shape','耦合道路 shape-only'),('arc_independent_encoder','原独立编码器 + 同圆弧解码'),('cartesian_road','Cartesian，同耦合道路输入')]:lines.append(f'| {label} | {mean_sd(a,"temporal/seed{s}/"+name)} |')
lines+=['',
'耦合完整模型相对独立编码器圆弧对照降低约 14.5%，但没有优于 guarded full 的 0.343 m。因此它可以作为一致性方法及诊断结果，不能写成整体最优模型，也不能把几何一致性等同于准确率保证。',
'',link('耦合道路完整结果','reports/coupled_road/RESULTS.md')+'；'+link('保存权重复算','reports/coupled_road/checkpoint_spotcheck.json')+'。','',
'## 3. 必须保留的反证和范围','',
'| 来源留出：E100（m） | 第一条记录作测试 | 第二条记录作测试 |','|---|---:|---:|']
for name,label in [('kinematic','标定运动学'),('frenet','初始 Frenet')]:lines.append('| '+label+' | '+' | '.join(mean_sd(original,f'outer{f}/seed{{s}}/{name}/normal') for f in range(2))+' |')
for name,label in [('guarded_full','统一感知修复后的 guarded full'),('geometry_midpoint','统一感知修复后的几何 midpoint')]:lines.append('| '+label+' | '+' | '.join(mean_sd(cond,f'outer{f}/seed{{s}}/{name}') for f in range(2))+' |')
lines+=['',
'跨来源性能没有恢复为一致优于简单标定模型。两来源速度和油门分布不同，第一条记录油门恒定，使驱动力系数与常量偏置无法单独辨识；这解释了为何需要报告工况，但不能证明所有泛化失败都由工况引起。后来的时间划分不能替代前面的压力测试。',
'所有记录都曾参与研究。新一轮内部训练／验证／测试隔离可以消除本轮选 epoch 泄漏，不能抹掉整个开发过程看过旧数据的事实。',
'推理读出不查真实地图，不等于整个系统不依赖地图：训练道路标签和初始道路相对状态仍来自地图／位姿。未来动作已知，不能声称已经验证自主闭环控制。',
'',
'## 4. 道路标签审计说明了什么','',
'旧地图名义长度为 10.450 m，平滑后折线长度约 9.743 m；原生网格参数速度相对 1 的绝对偏差中位数为 0.0512。旧曲率另外经过平滑。按圆弧解码积分旧曲率，与旧形状在验证窗口上的坐标 RMSE 为 0.207 m。',
'这说明它们不是精确一致的几何真值，包含平滑、未重新弧长参数化以及离散近似的共同影响。它不证明全部预测误差由标签造成，也不是丢弃不利模型的理由。主稿已将它写成近似结构化模型的限制；不能继续保留“严格物理一致／由此保证误差有界”的表述。',
'',link('标签审计报告','reports/road_geometry/NATIVE_LABEL_AUDIT.md')+'。','',
'## 5. 原来的结论哪些还能用','',
'- 可以保留：显式物理变量便于解释和诊断；在本次已说明条件的时间留出实验中，结构化道路模型有明显终点误差收益。',
'- 需要改写：把“普遍更准”改成“时间留出收益明显、来源转移受限”；把“map-free 系统”改成“预测读出不查询真实地图”；把“解决部分可观测性”改成“引入有限视觉道路上下文”。',
'- 不能继续作为已证实结论：状态必然 Markov、误差不累积或有界、道路本身使相同完整状态与动作产生不同车辆运动、未知赛道泛化、纯视觉闭环部署。',
'- 原来的 0.463/0.516 m 数字可以在历史实验中保留，但必须附带原有选模和信息条件，不能移到新协议主表中当作独立测试成绩。','',
'## 6. 现在究竟还要实验，还是改文字','',
'**不是靠改文字完成的：这一轮已实际补做六组实验方案，修复和审计训练、感知、几何、分割与评估，并把有利和不利结果同时写入正文。** 当前最有力的主线是“有限信息条件下的同赛道预测与可审计几何表示”。',
'若按这个收窄后的主线尝试送审，当前数据可支撑一个受限实证研究，不必为了得到漂亮结果无限追加消融。定稿需要明确会议扩展增量、压缩历史材料，并由作者确认署名、CRediT、利益冲突、基金、数据／代码发布和 AI 使用说明。',
'若目标仍是“较强把握被 RAS 接收”，当前还不能宣布达成。最有价值的技术补强是：在修正并重新弧长参数化的标签下，以冻结方案复核同一主模型及同输入对照；检验初始化误差和预览外推对结论的影响。它们可利用现有原始数据开展，但仍属于探索性补强，不会变成独立新场景验证，也不能自动增加理论新颖性。',
'新增记录目前不可用，因此不再把“必须提供新原始数据”作为继续写作的前提。跨赛道泛化或前瞻性确认仍缺证据；这些主张应留到确有数据时再讨论。',
'**停止标准不是某次测试变好，也不是训练次数足够：应是主张与证据一致、关键对照完整、流程可复算、作者信息齐全。当前已完成前三项的受限版本；较高接收把握仍无法从这些材料可靠推断。**','',
'## 7. 文件与复核范围','',
'- '+link('修订稿 LaTeX','Jounral_PIWM/elsarticle-template-num.tex')+'；PDF 已编译并检查 43 页。',
'- '+link('初始严格审稿','Jounral_PIWM/RAS_strict_review_2026-09-27.md')+'保留，避免把新实验倒填成初审时已经完成。',
'- '+link('完整实验序列','reports/EXPERIMENT_LEDGER.md')+'列出每轮动机和证据边界。',
'- 六组结果审计均 PASS；后四组逐 checkpoint 抽取 24 个均匀分布窗口复算通过。抽查不等于所有代码由独立研究者完整复现。',
'- 时间／耦合感知的训练均值、尺度、验证损失均从保存权重再次核对；所有指定种子、失败诊断和不利结果保留。',
'- 原稿、旧图和旧结果保存在归档中。没有提交论文、公开数据或向他人发送内容。',
'- 建议作者审核后 commit；checkpoints、部分日志和原始数据被 Git 忽略，需单独备份。','',
'模型名称：GPT-6（Codex）；当前会话未提供更细的型号标识。']
path=R/'Jounral_PIWM/RAS_revision_audit_2026-09-27.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(path)
