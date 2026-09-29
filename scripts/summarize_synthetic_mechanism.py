"""Summarize the finite pilot without extending its experimental grid."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_synthetic_mechanism as ex
b=ex.b;OUT=ex.OUT;R=b.ROOT
NAMES={'kinematic':'Kinematics','cartesian_road':'Cartesian + road','cartesian_blind':'Cartesian, no road','guarded_full':'Full Frenet','curvature_both_off':'No neural curvature','geometry_midpoint':'Shared curve'}

def main():
    cfg=b.read_json(OUT/'protocol.json');ex.verify_dev(cfg);ex.completed(cfg)
    audit=b.read_json(OUT/'audit.json');assert audit['status']=='PASS'
    decision=b.read_json(OUT/'decision.json');rows=b.read_json(OUT/'results.json')['results'];limits=np.array(b.read_json(OUT/'limits.json')['limits'])
    summary={}
    for model in ex.MODELS:
        for plant in ['nominal','shifted']:
            for regime in ['within','outside']:
                for noise in ex.NOISES:
                    key=f'{model}/{plant}/{regime}/{noise}';summary[key]={}
                    for metric in ['E25','E50','E100','ADE','p95_E100','speed_E100','yaw_rate_E100']:
                        vals=np.array([rows[f'seed{s}/{model}/{plant}/{regime}/{noise}'][metric] for s in [0,1,2]])
                        summary[key][metric]=dict(mean=float(vals.mean()),sd=float(vals.std(ddof=1)),seeds=vals.tolist())
    b.write_json(OUT/'summary.json',summary)
    # Post-hoc descriptive audit only, not a new candidate, refit, or gate change.
    guards={}
    for plant in ['nominal','shifted']:
        for regime in ['within','outside']:
            data=np.load(OUT/f'test_{plant}_{regime}.npz');truth=data['truth'];inc=np.abs(np.diff(truth[:,:,3:],axis=1))
            fraction=(inc>limits+1e-7).mean(axis=(0,1))
            t=np.arange(101)[None,:];v0=truth[:,0:1,3]
            gap=np.maximum(0,np.maximum(truth[:,:,3]-v0-t*limits[0],v0-t*limits[0]-truth[:,:,3]))
            guards[plant+'/'+regime]=dict(speed_increment_exceeds_cap_fraction=float(fraction[0]),yaw_increment_exceeds_cap_fraction=float(fraction[1]),
                max_speed_reachability_gap_m_s=float(gap.max()),episodes_with_unreachable_true_speed_fraction=float((gap.max(1)>1e-6).mean()),
                maximum_true_path_length_first25_m=float((truth[:,:25,3].sum(1)/22).max()))
    b.write_json(OUT/'posthoc_guard_diagnostic.json',dict(status='descriptive post-hoc diagnostic; not used to alter frozen decision',limits=limits.tolist(),results=guards,
        interpretation='Some true increments and speeds lie outside the fixed neural update envelope. This does not identify the full endpoint-error cause, and kinematics has no such cap but also degrades.'))
    # Scientific artifact: all six models, both plants, all three main contrasts.
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True,constrained_layout=True)
    colors=['#275DAD','#D18732','#55916E'];conditions=[('within','clean','Within / clean'),('within','strong','Within / smooth error'),('outside','clean','Outside / clean')]
    x=np.arange(len(ex.MODELS));width=.24
    for ax,plant in zip(axes,['nominal','shifted']):
        for j,(regime,noise,label) in enumerate(conditions):
            vals=[summary[f'{name}/{plant}/{regime}/{noise}']['E100'] for name in ex.MODELS]
            ax.bar(x+(j-1)*width,[v['mean'] for v in vals],width,yerr=[v['sd'] for v in vals],capsize=2,label=label,color=colors[j],alpha=.95)
        ax.set_title('Nominal plant' if plant=='nominal' else 'Changed vehicle parameters')
        ax.set_xticks(x,[NAMES[n] for n in ex.MODELS],rotation=32,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Endpoint error at 100 steps (m)');axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Synthetic dynamics-interface pilot: 128 paired episodes, mean ± training-seed SD',fontsize=12)
    fig.savefig(OUT/'pilot_comparison.png',dpi=180);plt.close(fig)
    def cell(model,plant,regime,noise):
        v=summary[f'{model}/{plant}/{regime}/{noise}']['E100'];return f"{v['mean']:.3f} ± {v['sd']:.3f}"
    lines=['# 有限仿真机制实验：是否值得深入','','日期：2026-09-28。**冻结判据结果：MECHANISM_ONLY（仅支持机制判断），GO=false。** 本轮不启动下一轮结构搜索。',
        '', '**判断：当前证据不支持继续扩展“屏蔽曲率／共享曲线”这两条结构路线；它支持将动作范围外推作为更具体的后续研究问题。尚未得到可支持 weak accept 的新方法。**',
        '', '## 1. 实验做了什么', '',
        '- 6 类模型 × 3 个训练种子，共 **18 次从零训练**，每次 40 epochs。384 个训练场景、128 个验证场景；仅验证 E100 选权重。没有新增 CNN 或真实车辆记录。',
        '- 测试为另一生成种子的 128 个场景，与训练、验证场景参数无重复；2 套车辆参数 × 2 个动作范围 × 3 个道路扰动级别，全部 **216 个模型／种子／条件结果**保留。',
        '- 同一测试场景在各条件间配对：改变油门范围时保持初始状态、道路和转向不变；改变道路扰动时保持动作与真实轨迹完全不变；改变车辆参数时保持动作及初始状态不变。',
        '- 真值使用独立的七状态平面动力学：侧向速度、饱和轮胎力、转向执行器滞后与二次阻力，RK4 每帧 16 子步。被测模型仍为五状态递推，包含模型失配。参数是透明的示例设定，**不是经过标定的 DonkeyCar，也不是 CommonRoad 的复现或高保真车辆保证**。',
        '- 道路先于轨迹生成；车辆动力学方程不读取道路。转向命令与道路曲率相关，但道路不会直接改变轮胎力。完整未来动作按原任务供应给所有预测器，仍是离线动作条件预测。',
        '- 这是动力学／道路接口实验，没有渲染相机图像或评估视觉编码器。独立生成不等于外部独立复现，更不等于真实新赛道验证。',
        '', '模型层次设计参考了 [CommonRoad 作者的车辆模型说明](https://commonroad.in.tum.de/static/media/LA17.8f9d9dc3.pdf)：运动学单轨与考虑滑移的动力学单轨有不同假设。本轮方程、参数及实现由本项目单独定义，没有采用其发布分数或声称实现等价。',
        '', '## 2. 全部条件的主要结果', '', 'E100 单位 m；± 是三个训练种子的样本标准差。它不是不同真实环境的不确定度。', '', '![有限实验比较](../reports/synthetic_mechanism/pilot_comparison.png)', '']
    for plant in ['nominal','shifted']:
        lines += [f'### {plant}', '', '| 模型 | 范围内／无误差 | 范围内／轻扰动 | 范围内／较强扰动 | 范围外／无误差 | 范围外／轻扰动 | 范围外／较强扰动 |','|---|---:|---:|---:|---:|---:|---:|']
        for name in ex.MODELS:
            lines.append('| '+NAMES[name]+' | '+' | '.join(cell(name,plant,regime,noise) for regime in ['within','outside'] for noise in ex.NOISES)+' |')
        lines.append('')
    lines += ['## 3. 按事先固定的门槛判断', '',
        'GO 要求预列候选在同一目标情景中，对两套车辆参数都比 full 降低至少 20% 的 E100，三个配对模型种子均改善、配对场景 bootstrap 改善区间下界为正；同时干净范围内代价不超过 10%，且目标误差不能比最强简单对照高出超过 10%。这些是本次研究决策门槛，不是 RAS 审稿标准。', '',
        '两个候选（不向学习分支输入曲率、共享曲线）× 两个目标（范围内较强道路扰动、范围外无道路误差），**四项判断全部未通过**。没有根据结果选择不同阈值、提高噪声、增加种子或重新训练。', '',
        '| full 的配对条件变化 | nominal 平均增量与条件 bootstrap 95% 区间（m） | shifted 平均增量与区间（m） | 是否通过预定机制门槛 |','|---|---:|---:|---|']
    for factor,label in [('road','范围内：较强道路误差 − 无误差'),('action','无道路误差：范围外 − 范围内')]:
        item=decision['mechanisms'][factor];cells=[]
        for plant in ['nominal','shifted']:
            v=item['plants'][plant];cells.append(f"{v['mean_m']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
        lines.append('| '+label+' | '+' | '.join(cells)+' | '+('是' if item['pass_both'] else '否')+' |')
    lines += ['', '区间以三个模型种子平均后的场景配对差做 2000 次 bootstrap，只描述本生成分布下的探索性不确定度；没有校正多重候选比较，不能当作真实环境的确证性检验。', '',
        '## 4. 结论及解释边界', '',
        '**本仿真网格内，动作范围变化的误差增量远大于预设平滑道路扰动；简单运动学模型在所有这些条件的平均 E100 都优于所比较的神经模型。** 两套车辆参数下方向一致，不是一个训练种子的偶然胜负。它支持研究动作外推，未支持当前两种结构改法。', '',
        '但这轮不能推出“真实道路感知不重要”：轻／较强扰动的道路坐标 RMSE 实际约为 0.00945/0.03780 m，曲率 RMSE 约为 0.00649/0.02601 m^-1；0.10 m 是形变函数的系数，不是道路 RMSE。扰动经过共同曲线构造并保持初始位置和切向，因此未覆盖真实独立预测头的不一致、初始状态错误或更大的曲率误差。', '',
        '改变油门会使速度和行驶距离变化，属于动作干预的后果。范围外 100 步真值端点超出 4.5 m 预览半径的比例为 nominal 75.0%、shifted 80.5%，所以最终动作效应不能全部解释成纯车辆响应网络失配。模型预览外推与内部状态监测均保留；没有丢弃这些场景。', '',
        '## 5. 补充诊断：限幅的适用范围', '',
        '以下为结果出来后的描述性核查，**没有用于改变冻结门槛**：神经模型两种单步运动增量的训练限幅均约 0.01。范围外真值速度增量超过该限幅的比例，nominal 为 21.94%，shifted 为 21.73%；范围内仅约 0.031%。这指出训练分位数限幅不能自动解释为物理硬件极限。', '',
        '该现象只说明部分真实变化超出了当前神经更新的表示范围，不能证明它解释全部端点误差。没有此限幅的运动学基线也在范围外变差。初始执行器状态、未观测侧向运动、窄动作覆盖和长时误差累积同样存在；本轮没有逐项因果分解。', '',
        '若下一轮获得授权，值得预先检验的单一假设是：**将可外推的车辆响应与受限学习残差分开，并使保护范围依据明确的物理条件定义，能否在保留范围内精度的同时改善动作外推？** 这只是后续假设，不能列为当前已经提出并验证的新方法。直接放宽限幅或增加一个门控，也不能预先视为创新。', '',
        '## 6. 审计与交付', '',
        '- 18 份选中权重的验证误差、40-epoch 最小值和参数量复核；216 条结果每条重算 24 个固定场景，指标从逐场景数组重新汇总，均通过。',
        '- 重算确认道路盲模型对三个道路扰动等级输出逐位相同；地图缓冲区篡改不改变 Frenet 预测。生成器镜像、直行、圆弧与步长精度核验通过。',
        '- 仿真数据、参数、协议、源码指纹、训练历史、权重、全部情景误差和判定保存在 `reports/synthetic_mechanism/` 与 `checkpoints/synthetic_mechanism/`。训练只访问训练／验证数据，测试生成在全部拟合完成后执行。',
        '- 当前 40 页论文及其 3/5、Major Revision 评审保持不变。本轮没有足够依据提高到 weak accept，也未把试验性仿真自动混入真实车辆结果。',
        '- 本轮有限实验到此结束，无下一轮训练、后台任务、投稿或推送。建议审阅后 commit，并单独备份被 Git 忽略的权重。', '',
        '模型名称：GPT-6（Codex）；当前会话未提供更细型号标识。']
    report='\n'.join(lines)+'\n';(R/'Jounral_PIWM/synthetic_mechanism_pilot_2026-09-28.md').write_text(report,encoding='utf-8')
    # Same text with the image location adjusted for the experiment folder.
    (OUT/'RESULTS.md').write_text(report.replace('../reports/synthetic_mechanism/pilot_comparison.png','pilot_comparison.png'),encoding='utf-8')
    b.write_json(OUT/'DELIVERY.json',dict(status='PASS',decision=decision['decision'],fits=18,evaluation_rows=216,
        report_sha256=b.digest(R/'Jounral_PIWM/synthetic_mechanism_pilot_2026-09-28.md'),protocol_sha256=b.digest(OUT/'protocol.json'),figure_sha256=b.digest(OUT/'pilot_comparison.png'),
        manuscript_unchanged=True,manuscript_sha256=b.digest(R/'Jounral_PIWM/elsarticle-template-num.pdf'),summary_script_sha256=b.digest(__file__)))
    notice='''## 2026-09-28 最新：有限仿真试验完成，未扩大实验

本次新增 18 个仿真动力学模型（与此前 231 个真实日志协议模型分列），6 类 × 3 种子，全部 216 个评估情景及审计完成。未新增真实记录或 CNN。结果目录 `reports/synthetic_mechanism/`，权重 `checkpoints/synthetic_mechanism/`；最新报告 `Jounral_PIWM/synthetic_mechanism_pilot_2026-09-28.md`。

冻结判据为 MECHANISM_ONLY，GO=false：动作范围外推效应得到本生成条件下的支持，两个结构候选均未通过继续深入门槛，运动学基线更强。平滑耦合道路扰动较小，不能据此否定真实感知作用。训练分位数限幅与范围外真实增量不相容的补充核查仅属事后描述。

当前主论文仍为此前 40 页，SHA256 `69491cbf6c9edaf08437abe33ed29fd805f7bfa7a6154508fad38614eb24a4f2`，没有混入试验性仿真；模拟评分仍 3/5 Major Revision。无追加训练、自动任务、提交、推送或投稿。可审阅后 commit 并备份忽略资产。

'''
    for rel in ['HANDOFF.md','Jounral_PIWM/HANDOFF.md','reports/EXPERIMENT_LEDGER.md']:
        path=R/rel;old=path.read_text(encoding='utf-8')
        if '最新：有限仿真试验完成' not in old:path.write_text(notice+old,encoding='utf-8')
    print('Report and figure saved; frozen decision',decision['decision'])
if __name__=='__main__':main()
