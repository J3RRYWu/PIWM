"""Write current strict-review findings and synchronize the experiment handoff."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[1];O=R/'reports/curvature_branches';J=R/'Jounral_PIWM'
def read(p):return json.loads((R/p).read_text(encoding='utf-8-sig'))
def main():
    for p in ['curvature_branches/audit.json','road_branch_diagnosis/audit.json','road_branch_transfer/audit.json','road_branch_diagnosis/local_geometry_checks.json']:
        assert read('reports/'+p)['status']=='PASS'
    data=read('reports/curvature_branches/results.json')['results'];pair=read('reports/curvature_branches/paired_comparisons.json');diagnosis=read('reports/road_branch_transfer/summary.json')
    def cell(name,fold):
        a=[data[f'{fold}/seed{i}/{name}']['E100'] for i in range(3)];return f'{np.mean(a):.3f} ± {np.std(a,ddof=1):.3f}'
    rows=[]
    for name,label in [('guarded_full','完整模型（既有）'),('curvature_response_off','仅位姿残差接收曲率'),('curvature_pose_off','仅速度／角速度响应接收曲率'),('curvature_both_off','两个学习分支均不接收曲率'),('kinematic','运动学基线（既有）')]:
        rows.append('| '+label+' | '+' | '.join(cell(name,f) for f in ['temporal','outer0','outer1'])+' |')
    text='''# RAS 严格复审：曲率分支、转移诊断与局部几何

日期：2026-09-28。此报告接续同日的几何修正审计，反映本轮新增实验后的当前判断。

**最终模拟评分：Overall 3/5，Major Revision。技术证据有所加强，但尚不支持升为 weak accept，也没有可校准的高录用概率。** 本轮的成果是更可靠的机制判断及边界，不是找到了一个全面更好的模型。

沿用此前自拟量表：1=强烈拒稿，2=拒稿，3=大修，4=小修／倾向接收，5=接收。未取得 RAS 内部表格，这不是期刊官方评分。

## 1. 实际完成

- 新增 27 个动力学训练：3 个曲率分支变体 × 3 划分 × 3 种子；每个 40 epochs，验证 E100 选 checkpoint。没有新增 CNN。
- 完成完整 2×2 对照：曲率是否进入速度／角速度响应分支、是否进入位姿残差分支。所有变体的几何递推继续使用动态曲率，参数量和训练规则相同。
- 完整分支实现与原模型的训练窗口输出和参数梯度逐位一致；分别验证屏蔽分支对特征变化不敏感，而几何仍对曲率变化敏感。
- 完成 72 组训练／验证无重训诊断和 36 组测试分区诊断。后者在全部新模型训练结束后运行，分组来自训练分区；同一权重的预测道路与真值道路配套替换。
- 补查共享曲线的每个 midpoint 内部阶段，区分保护触发、非正分母、曲线退化和预览外推。
- 在实际代码上验证局部坐标等价关系及敏感性公式；完成全部新权重、验证选模、逐窗口指标和地图缓冲区独立性的审计。
- 将方法、全部分支结果、真值预览诊断及关键限制写入当前论文；旧稿备份于 `_archive/journal/curvature_branches_before/`。

## 2. 关键结果

E100 单位 m，± 为三个训练种子的样本标准差，不是独立环境置信区间。

| 模型 | 时间划分 | 来源 A | 来源 B |
|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

完整模型相对“两学习分支均不接收曲率”的时间端点误差只降低约 **3.3%**，在 **2/3 种子、8/14 片段**中更好。原先相对“固定起点曲率”的 **9.9%** 差距仍真实存在，但衡量的是不同对照。不能据此把全部相对 Cartesian 的大幅收益归给学习分支中的动态曲率，也不能声称没有该输入模型就失效。

去掉位姿分支曲率后，来源 A 的三个种子均改善，平均减少约 0.077 m；去掉响应分支曲率后，来源 B 的三个种子均改善，平均减少约 0.075 m。两种操作在时间划分的三个种子上都变差。两个学习分支均去掉曲率，来源 A 的三个种子改善，来源 B 只有两个种子改善。没有统一的全面占优方案，也没有新变体在任一来源方向的平均 E100 超过运动学基线。

这些结果支持“曲率接入位置与数据条件之间存在取舍”。它们不证明曲率对车辆物理响应具有因果影响；d/psi 依然进入所有网络，所谓两分支不接收曲率也不是完全不使用道路信息。

[完整分支报告](../reports/curvature_branches/RESULTS.md)；[成对种子／分片段比较](../reports/curvature_branches/paired_comparisons.json)。

## 3. 来源转移失败定位到了什么程度

主模型预测预览／真值预览的平均 E100：时间 **0.359/0.305 m**，来源 A **1.038/0.993 m**，来源 B **1.327/1.222 m**。即使输入精确道路，两方向仍高于运动学的 **0.936/1.158 m**。因此，直接改善道路预览并不足以消除当前差距；替换输入也可能引起分布变化，不能把差值当作严格误差因果分解。

两个来源留出方向的 **全部测试窗口**，都含有至少一个超出训练集逐维转向／油门范围的动作。来源变化和操作范围外推同时发生，不能将该实验解释成纯视觉域迁移。时间测试也有约 72.1% 窗口包含这种动作；这不是所有帧都范围外，也不意味着范围内就有充分联合分布覆盖。

这为失败给出更具体、可审核的条件，但尚未解决外推。现有记录中动作变化的覆盖有限，继续单凭这两条记录调神经网络，不能识别所有新工况下的车辆响应。不能用“数据有问题”把来源表删除。

[真值道路诊断](../reports/road_branch_transfer/RESULTS.md)；逐窗口分组及动作范围均在该目录保存。

## 4. 几何细节补到了哪里

- 实际共享曲线场与位置读出的局部连续等价关系，在 128 个 float64 光滑内域状态上通过，最大绝对差约 1.8e-15。三个敏感性偏导与自动微分误差均小于 4.5e-16。
- 这证明检查涉及的代码符合标准几何恒等式，不构成新理论或全局稳定性证据。连续、无残差、共享几何的 Frenet 模型转换回平面坐标后，满足普通车辆运动学；坐标变换本身不增加车辆物理规律。
- 共享曲线预测预览的 midpoint 中间状态非正分母比例，取三种子最大值，时间约 **0.4364%**，来源 A **0.5273%**，来源 B **1.2880%**。此前全步数字没有覆盖这些内部阶段，现已分别保留。触发保护与非正分母分开统计；没有通过丢弃窗口降低误差。
- 0.5 m 曲率离散表示的近似误差仍在；本轮没有冒称已修复它。固定步长残差的单位也已说明，不能直接减小 dt 后声称得到积分收敛证明。

[局部几何说明及核验](../reports/road_branch_diagnosis/LOCAL_GEOMETRY_ANALYSIS.md)。

## 5. 模拟审稿判断

| 栏目 | 评分 | 当前判断 |
|---|---:|---|
| 创新性 | 3/5 | 具体视觉道路接口与接入位置的实证判断更清楚；标准坐标公式、残差学习和屏蔽特征本身不新。 |
| 贡献价值 | 3/5 | 时间收益及其适用边界可复核；动态曲率在学习分支中的独立增益较小。 |
| 技术严谨性 | 4/5 | 干净分支对照、局部代码恒等式和内部阶段监测补强；坐标有效性与离散表示仍有限。 |
| 实验证据 | 3/5 | 多了机制证据，仍无新增独立记录或不同环境验证。 |
| 讨论与结论 | 4/5 | 区分感知误差、操作外推、坐标处理；保留所有不利结果。 |
| 可复现性 | 4/5 | 协议、源码指纹、权重、逐窗口误差、内部复算完整；不是外部复现。 |
| **Overall** | **3/5** | **Major Revision，尚未到 weak accept。** |

原研究方向并未因这些结果失去价值，但贡献主张需要调整：应强调有限视觉道路接口的结构设计、受控实证及适用边界，而不是把动态神经曲率包装成关键性能突破。

## 6. 本轮停止条件与剩余工作

27 个训练及诊断、复算、论文更新均完成，没有后台训练或自动追加实验。保留此前预选完整模型；本轮没有按测试结果挑选新主模型，也没有随后追加参数搜索。

目前不据此再加一轮“屏蔽更多道路特征”的调参。原因是本轮没有给出统一改进，真值预览也不能消除来源外推差距；继续在同一测试记录上重试，不能补足证据独立性。前一方案中的几何候选训练和独立生成条件验证尚未执行，不能把它们算作完成。

若继续以明显提高严格审稿把握为目标，下一步应是独立生成、带模型失配的受控条件验证，检查几何与动作变化各自的作用，并补上有明确触发条件的坐标后备处理。它可以检验机制，不能替代真实新赛道实验；也不能预先承诺其结果达到接收门槛。当前已有材料适合形成范围收窄的实证稿，但尚不能称为有较强接收把握的 RAS 终稿。

作者信息及声明仍需作者最终确认；本轮未投稿、公开数据、发邮件或推送代码。可审阅后 commit，并单独备份被 Git 忽略的权重和数据。

模型名称：GPT-6（Codex）；当前会话未提供更细型号标识。
'''
    (J/'RAS_curvature_branch_review_2026-09-28.md').write_text(text,encoding='utf-8')
    manifest=read('reports/curvature_branches/manuscript_render_manifest.json')
    assert manifest['pdf_sha256']==hashlib.sha256((J/'elsarticle-template-num.pdf').read_bytes()).hexdigest()
    manifest.update(visual_review='PASS: all-page contact sheets and full-resolution equations/new tables inspected; table paragraph breaks repaired and re-rendered',inspected_detail_pages=[10,15,16])
    (O/'manuscript_render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    notice='''## 2026-09-28 最新：曲率分支审计完成

此段优先于下面更早的进度。当前论文 40 页；完整模型仍是预先固定的性能参照。新增 27 个动力学训练，全部完成；累计新协议训练为 36 个 CNN、231 个动力学模型（不计历史开发模型），原始记录仍为同一赛道两条。

- 分支对照：`reports/curvature_branches/`；脚本 `scripts/run_curvature_branches.py`；新增模型 `src/baselines/branch_query.py`。主报告序列化的 NumPy 整数问题由独立 `scripts/summarize_curvature_branches.py` 适配，冻结源码与结果未改。
- 无重训诊断：`reports/road_branch_diagnosis/`（72 组训练／验证）及 `reports/road_branch_transfer/`（36 组测试）。两个来源测试均为动作范围外推；完整模型的精确道路替换仍未超过运动学的来源 E100。
- 时间 E100：full .359；response-off .378；pose-off .368；both-off .371。完整模型相对 both-off 约 3.3% 收益，不能把相对静态曲率的 9.9% 当成同一对照。新变体未在任一来源方向的平均 E100 超过运动学。
- 最新复审：`Jounral_PIWM/RAS_curvature_branch_review_2026-09-28.md`，Overall 3/5 Major Revision，尚非 weak accept。全部不利结果保留。
- 主稿更新由 `scripts/revise_branch_manuscript.py` 一次性执行；再次运行会拒绝覆盖。上一稿备份 `_archive/journal/curvature_branches_before/`。旧生成器可能回退当前稿，不要盲目重跑。
- 当前 PDF SHA256: `'''+manifest['pdf_sha256']+'''`；40 页，编译及渲染核查通过。无新增训练进程、自动任务、提交或推送。几何后备方法训练和独立生成条件验证尚未做。
- 审阅后 commit；权重／数据等被忽略资产需单独备份。

'''
    for rel in ['HANDOFF.md','Jounral_PIWM/HANDOFF.md','Jounral_PIWM/README_BUILD.md']:
        p=R/rel;s=p.read_text(encoding='utf-8')
        if '最新：曲率分支审计完成' not in s:p.write_text(notice+s,encoding='utf-8')
    p=R/'reports/EXPERIMENT_LEDGER.md';s=p.read_text(encoding='utf-8')
    if '最新：曲率分支审计完成' not in s:p.write_text(notice+s,encoding='utf-8')
    audit=read('reports/curvature_branches/audit.json')
    record=dict(status='PASS',fits=27,training_validation_diagnostic_rows=72,test_diagnostic_rows=36,
        manuscript_sha256=manifest['pdf_sha256'],manuscript_pages=40,overall_simulated=3,
        protocol_sha256=hashlib.sha256((O/'protocol.json').read_bytes()).hexdigest(),
        postprocessing_sources={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'scripts/revise_branch_manuscript.py',R/'scripts/render_branch_manuscript.py',R/'scripts/summarize_curvature_branches.py']})
    (O/'DELIVERY.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Review, handoff, ledger and delivery manifest written.',manifest['pdf_sha256'])
if __name__=='__main__':main()
