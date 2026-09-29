from pathlib import Path
import json,hashlib
R=Path.cwd();review=R/'Jounral_PIWM/RAS_revision_audit_2026-09-27.md'
s=review.read_text(encoding='utf-8');footer='模型名称：GPT-6（Codex）；当前会话未提供更细的型号标识。'
plan='''## 8. 下一步修改的优先顺序

### P1：统一几何标签，再做一次固定方案复核

现有中心线平滑后未重新按真实弧长参数化，曲率与形状又分别处理。应由同一条平滑中心线重新生成弧长、切向、曲率、局部形状及初始道路状态，并以解析圆／直线和数值导数核对它们的关系。不能只改曲率、继续沿用旧的进度与形状标签。

复核保持原时间分割、原始 xy 真值、训练预算、三种子和验证选模不变；所有使用道路的主模型与对照都用同一套修正标签重新训练，无道路运动学仅在输入不变时复用。修正前后完整列出，不按修正后的测试分数决定是否保留。完成标准是几何审计通过，并清楚报告主要性能收益是否仍成立；不能预设修正必然降低误差。

### P2：用当前新协议做初始化敏感性测试

对 d、航向误差、速度和角速度分别及联合扰动，报告完整模型与同输入基线在相同噪声实现下的 E100/ADE 变化和失效比例。没有实测传感器误差分布时，只能把预先固定幅度称为敏感性情景，不声称模拟真实传感器。不训练到某个噪声结果最好后再宣称鲁棒。

完成标准是明确当前优势允许多大初始化偏差，以及哪个输入最敏感。该实验检验已知初始状态假设的脆弱性，不把系统变成纯视觉定位或闭环控制。

### P3：加强一个最关键的同输入对照

让 Cartesian 残差控制使用与主模型可比的增量约束，并核对参数规模和验证学习曲线；若调整训练时长或学习率，两个模型都使用预先固定、对称的验证搜索预算。不要为了增加数量堆很多弱基线。

完成标准是排除“主要收益来自增量限幅、优化预算或基线没训好”的解释。若强对照明显缩小差距，应如实更新贡献，不能删掉强对照。

### 论文同步收束

主线围绕：显式道路预览 → 受保护的结构化预测 → 时间留出收益 → 来源转移与几何限制。把会议复述、历史版本和较长失败诊断压缩或移入补充材料，同时在正文保留影响结论的关键不利比较。创新表述突出具体设计与已验证作用，不把标准 Frenet 公式称为新理论。

完成 P1 后先重新审稿，再决定是否需要全部 P2/P3；避免无止境改方案直到测试数字好看。没有新增采集并不妨碍这些修订，但它们都不能替代独立新环境验证。

'''
assert s.endswith(footer+'\n');review.write_text(s[:-len(footer+'\n')]+plan+footer+'\n',encoding='utf-8')
notice='''## 2026-09-27：严格审计修订状态（优先于下方历史记录）

当前主稿由 `scripts/revise_audited_manuscript.py` 生成，原稿备份在 `_archive/journal/audited_revision_before/`。本轮六组受控／探索性实验已全部完成，结果与审计分别在 `reports/controlled_holdout`、`frenet_diagnostic`、`road_geometry_training`、`conditioned_dynamics`、`temporal_holdout`、`coupled_road`。旧五折数字在主稿历史附录保留，不能作为新协议测试结果使用。

时间留出 guarded full 的 pooled E100 为 0.343±0.010 m，对照运动学为 0.898±0.006 m；耦合圆弧模型为 0.400±0.005 m。来源留出没有一致优于运动学。只有两条已开发记录、已知初始状态及未来动作；不可声称独立新场景、纯视觉部署或误差有界。

最终模拟复审与后续优先级见 `Jounral_PIWM/RAS_revision_audit_2026-09-27.md`，3/5 Major Revision，非 RAS 官方评分。当前不是可直接上传的终稿：作者信息与声明待确认。新增实质发现：旧几何标签的名义弧长与平滑中心线不精确一致；下一步优先统一标签再做受控复核。

论文已编译为 43 页，当前 PDF 全页缩略图及重点图表检查通过；没有未定义引用或 overfull 警告。BibTeX 仍有 5 项缺页码提示，不能称为所有书目元数据完整。所有本轮训练任务已结束；没有自动继续运行的训练或定时任务。

注意：生成器从归档原稿及结果重建正文，会覆盖手工编辑；作者开始手改后不要直接重跑。`checkpoints/`、部分日志和数据未纳入 Git，需另行备份。此前 HANDOFF 中“可部署”“CV 消除非单调”等建议不自动适用于当前审计稿。

---

'''
for name in ['HANDOFF.md','Jounral_PIWM/HANDOFF.md']:
    p=R/name;p.write_text(notice+p.read_text(encoding='utf-8'),encoding='utf-8')
p=R/'Jounral_PIWM/README_BUILD.md';s=p.read_text(encoding='utf-8');s=s.replace('The main file is `elsarticle-template-num.tex`. All eight external figures are\nin `imgs/`; three method figures are editable TikZ directly in the main file.','The main file is `elsarticle-template-num.tex`. External figures are in `imgs/`;\nthe retained architecture diagram is editable TikZ in the main file.\nThe audited revision currently builds to 43 pages. Author metadata and\ndeclarations remain unfinished; compilation does not establish submission readiness.');s+='\n## Audited experiment update (2026-09-27)\n\nThe `reproduce/` bundle describes the earlier figure revision only. It does not\nreproduce the six new experimental suites. Their protocols, code hashes, results\nand audits are under the repository-level `reports/`; training scripts are under\n`scripts/` and saved models under `checkpoints/`. See `reports/EXPERIMENT_LEDGER.md`.\nNew plots: `scripts/plot_conditioned_dynamics.py` and\n`scripts/summarize_temporal_evidence.py`.\n\n`scripts/revise_audited_manuscript.py` rebuilds the manuscript from the preserved\noriginal and audited result files. It overwrites the current main TeX file; do\nnot run it after manual author edits unless those edits are first preserved.\n\nThe final build has no undefined references or overfull warnings. Five existing\nBibTeX entries lack page fields; author details and disclosures still need review.\n';p.write_text(s,encoding='utf-8')
p=R/'Jounral_PIWM/controlled_holdout_rewrite.md';s=p.read_text(encoding='utf-8');p.write_text('> Historical intermediate rewrite. The current main TeX and RAS_revision_audit_2026-09-27.md supersede this partial draft.\n\n'+s,encoding='utf-8')
p=R/'reports/manuscript_audit/render_manifest.json';obj=json.loads(p.read_text());obj.update(visual_review='PASS: all 43 pages inspected as contact sheets; full-size architecture, tables, and temporal plot inspected; no clipping/overlap found.',latex_undefined_references=0,latex_overfull_warnings=0,bibtex_missing_page_warnings=5,submission_metadata='incomplete; author confirmation needed');p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
print('Review plan, handoffs, build notes and visual QA updated.')
