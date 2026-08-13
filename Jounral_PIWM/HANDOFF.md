# PIWM 会议 → 期刊 扩展 · 交接文档 (HANDOFF)

> ## 🛑 2026-08-08:正文里的主表数字目前**不可复现**,改稿前必读
>
> 详细证据在 **`piwm/HANDOFF.md` §0**。这里只说对写作的影响:
>
> | 正文现在写的 | 实际情况 |
> |---|---|
> | ours `E_xy@100 = 0.267 m` | 受控重训 3 seed = **0.412 ± 0.080**;论文那个 checkpoint 比三次里最好的还好 0.079 |
> | 最强基线 `0.383 m`(Vid2Param) | V2P 是唯一被允许看图像的基线,且它的 θ 从含 20 维车道路点的编码器输出推出,**能编码道路** |
> | 「我们赢在结构不在信息」 | 我方状态是**已知中心线上的 (s,d)**,坐标系本身就是特权信息 |
>
> **效应量(0.383−0.267 = 0.116)比我方单侧训练噪声(±0.080)还小。主表必须改成 mean ± range。**
>
> ✅ **同一轮也有好消息**:模型改进后 map-free 版本几乎不要钱(3 seed 配对差 **+0.004 m**),
> 「pure world model」的说法从站不住变成**站得住**,而且不再需要「基线拿到 GT 车道路点作补偿」
> 那种别扭辩护。这条**不依赖任何基线数字**,可以先写。
>
> ✅ **2026-08-12:基线 9 次重训跑完,带误差棒的主表出来了**(`piwm/HANDOFF.md` §0.9,
> 脚本 `scripts/eval_seeds_table.py`)。对写作的影响有三条,**比上面那版温和**:
>
> 1. **论文用的基线 checkpoint 也是幸运抽样** —— V2P 重训均值 0.510 ± 0.024(正文写 0.383),
>    GOKU 0.631(写 0.478),DVBF 0.999(写 0.610)。所以"只有 ours 那格不可复现"的说法不对,
>    是**整张表都是单次抽样**。两边都改成 3-seed 均值后:**0.412 ± 0.080 vs 0.510 ± 0.024**,
>    领先 1.24×(正文现在写的是 1.4×)。
> 2. **主论点要从"领先 1.4×"改成配对结论**:逐窗口配对 + bootstrap CI,对最强基线 V2P 是
>    **2/3 seed 显著赢、1/3 打平、0/3 输**;对 GOKU/DVBF 是 3/3 显著赢。这个说法有 CI 撑着。
> 3. **交叉点那张图反而更强了**:V2P 在 @25/@50 大幅领先且误差棒极小(0.036±0.003 / 0.088±0.015
>    vs 我方 0.121±0.005 / 0.234±0.022),不是噪声。"短程输、长程赢(有界 vs 发散)"可以放心写。
>
> ⚠️ 改稿前注意:`DVBF_s2` 是离群点(1.621 m,另两个 ~0.69),DVBF 的 ±0.470 几乎全来自它;
> V2P 的 θ 用共享 v6 编码器(没跟着 seed 重训),它的方差可能被低估。两点都该在正文说明。
>
> ### 🟢 2026-08-13:5-fold CV 也跑完了(`piwm/HANDOFF.md` §0.10)—— **以这一版为准改稿**
>
> 70 个任务 0 失败。对写作的影响,比 08-12 那版又变了,而且**总体是好消息**:
>
> 1. **主论点升级**。不要再写"误差比基线小 1.4×",改写成"**对数据划分鲁棒得多**":
>    换验证集时我方 0.412 → 0.463(几乎不动),V2P 0.510 → 0.720,DVBF 0.999 → 1.287。
>    跨划分离散度 **±0.046 vs ±0.194 vs ±0.850**。这个论点比原来的强,而且有 5 折撑着。
> 2. **战绩从"2 赢 1 平"变成"5/5 全赢"**。fold 内逐窗口配对,对 V2P 平均 −0.257 m,
>    五折全赢;对 GOKU/DVBF 也是 5/5。
> 3. **δ 那一节要反过来写,这是最大的改动**。正文现在在 Limitations 里说
>    "δ=5% > δ=10% 是单次训练方差" —— **这个解释是错的**。5 折上我方随 δ 单调**变好**
>    (0.463 → 0.452 → 0.319),基线单调**变差**(V2P 0.720 → 0.992 → 1.015)。
>    这是可复现的真实效应:弱监督标签噪声对物理结构化模型起正则化作用、对纯学习基线是损害。
>    **应该从"局限"搬到"结果",它正好是"结构带来鲁棒性"主线的直接证据。**
> 4. **"map-free 几乎免费"要 softening**。5 折配对差 **+0.053 m**(不是之前单 split 的
>    +0.004)。仍然小,但别写 "free",写"约 5 cm 的代价"。
>
> ✅ **v6 编码器泄漏:已量化,判定为空**(`piwm/HANDOFF.md` §0.10 缺陷 1)。V2P 的 θ 用的
> 是跨折共享、未按折重训的 v6 图像编码器。fold 0 的留出集恰好拆成"v6 从没见过的 7 个"和
> "v6 训练过的 7 个",实测 V2P 相对三个无编码器模型的优势在后者上反而**低 0.086 m**。
> **不用重训,但正文要如实披露并附这个数字**——有数字的披露比"承认瑕疵"强得多。建议措辞:
> "Vid2Param 的参数编码器跨折共享、未按折重训;我们量化了这一点:在该编码器训练过的
> 留出 episode 上,其相对无编码器基线的优势反而低 0.086 m,即曝光未带来可测量的优势。"
>
> ⚠️ **仍未解决的实质问题**(比上面那条重要):§0.3 —— 会议版把 GokuNet 和 Vid2Param **都**
> 当作吃观测的方法,本仓库却删了 GOKU 的观测通路、只留 V2P 的,**这才是 V2P 成为最强基线的
> 原因**。`goku_obs` 已实现(参数量与 V2P 完全相同)但 5-fold 没跑。要么补跑(每折 1 个,
> 共 5 个任务),要么在正文披露这个不对称。
>
> 数据现在齐了(单 split 3-seed + 5-fold 两套都有),可以动正文了。


> 用途：换机器 / 换会话后，读这一份即可恢复全部上下文，继续写这篇期刊。
> 最后更新：2026-07-28。**本轮把 Frenet 结果搬进了正文，并修完了一批图与格式问题。**
> 当前状态：`pdflatex → bibtex → pdflatex ×2` **0 error / 0 undefined / 42 页**（本机实测）。

配套文档：`piwm/HANDOFF.md`（代码与实验线）。**两份要一起读**，见 §9。

---

## 0. 一句话

把 PIWM 会议论文改写成**期刊版**。核心是从"全局可观测基准"扩展到"**局部可观测驾驶**"：
新增局部可观测案例，并对**道路环境**做物理可解释的**编码**与**预测**（会议版只建模了车本身）。
定量主结果 = **真实 DonkeyCar 上的 Frenet 世界模型**。

---

## 1. 目标与三个扩展点（用户定的框架，务必遵守）

**会议版** = *Physically Interpretable World Models via Weakly Supervised Representation Learning*
三案例（CartPole / Lunar Lander / DonkeyCar），全局可观测；架构 = 视觉编码器 + Transformer 物理编码器 +
可学习结构化动力学；弱监督（分布式、mean-proximal）。结论：**extrinsic + 离散(VQ) 最强**，能恢复真实物理参数。

**期刊三个扩展点：**
1. **局部可观测案例**：`CarRacing`（仿真、俯视）+ `DonkeyCar`（**真实物理小车**）。
   ⚠️ 见 §4：本版**定量结果只有 DonkeyCar**，CarRacing 的定量对比已注释掉。
2. **道路环境 ENCODING**：可解释状态**因子化** = 车辆相对状态 `z^v=(x_rel,y_rel,ψ_rel)` +
   道路上下文 `z^r=(e_CTE, e_ψ, κ)`；从前视图像时间窗推断前方 **4.5 m 曲率剖面**（10 个采样点）。
3. **道路环境 PREDICTION**：split-dynamics = 结构化车辆运动学 + **几何驱动的道路转移**——
   道路不被"预测"，而是被**按已走弧长重新索引**（advection），所以长程不漂。

**硬性写作要求**：contributions **之前**先讲 challenge / motivation
（driving 局部可观测 → 车辆状态非马尔科夫 → 既要编码又要预测道路）。已落实在 Intro 的
"The challenge" / "Our approach" 小节（`.tex` 约 143 / 196 / 236 行）。

---

## 2. 关键文件与位置

当前机器仓库根：**`C:\Users\suian\OneDrive\桌面\PIWM\`**（不是旧文档写的 `E:\Desktop\PIWM\`，
E: 盘在这台机器上不存在）。git 根 = `<仓库根>\piwm\`，`Data_Donkeycar*` 在**仓库根**一层。

### 期刊主稿（`piwm/Jounral_PIWM/`）

| 文件 | 说明 |
|---|---|
| **`elsarticle-template-num.tex`** | **期刊正文——唯一要改的主文件**（~105 KB / 约 1900 行） |
| `elsarticle-template-num.pdf` | 成稿，42 页 |
| `elsarticle-template-num.bbl` | 生成的参考文献（投稿用），81 条，0 undefined |
| `sample-base.bib` | 参考文献库（121 条，cite key 已齐） |
| `elsarticle-num.bst` | bib 样式 |
| `imgs/` | **所有图，本文件夹现在自包含**（见下） |
| `backup.tex` | 早期草稿。⚠️ 它的模型是 VAE+LSTM，且自称数据来自 **DonkeyCar 模拟器**；旧的真车数字出自这里，本轮已从正文撤下（见 §4） |
| `donkeycarbestlap.png` | 旧最佳圈图（已不在正文引用，见 §4） |
| `HANDOFF.md` | 本文件 |

改稿前的 `.tex` 备份在 `piwm/_archive/journal/elsarticle-template-num_pre-frenet.tex`。

### 图（**已全部收进 `imgs/`，`.tex` 用裸文件名 + `\graphicspath{{imgs/}}`**）

这解决了旧版最大的坑：图用相对路径引到文件夹外，一次仓库重构就全断，而且
Elsevier Editorial Manager 上传时会把文件拍平，外部路径必然失效。

正文**当前实际使用**的图：

| 图 | 来源 | 说明 |
|---|---|---|
| `fig:partial_obs` | **TikZ（写在 .tex 里）** | 两车同状态 / 前方一直一弯，含相机预览楔形。替换了原来的 64×64 占位相机帧 |
| `fig:arch` | **TikZ** | 编码链 + split dynamics + privileged 监督 + 自回归回路 |
| `fig:encoding` | **TikZ** | `z^r` 的几何定义 + 相机预览范围 + 编码器输出的 10 点曲率剖面 |
| `fig:prediction` | **TikZ** | 曲率剖面只感知一次、按已走弧长重新索引（advection） |
| `fig:donkey_main` | `imgs/fig_main.pdf` | Frenet(感知) + oracle + 基线 + SINDYc，**矢量**，与主表完全一致 |
| `fig:delta` | `imgs/fig_delta_supervision.pdf` | δ 弱监督**标签**噪声 |
| `fig:noise` | `imgs/fig_noise.pdf` | **测试时初始状态**高斯噪声（期刊新增，和 δ 不是一回事） |

`imgs/` 里还放了几张**目前没被引用**的图，别误用：
- `fig_frenet_vs_baselines.png` —— `eval_frenet_vs_baselines.py` 的工作图，内容与 `fig_main.pdf`
  相同但是位图。正文用矢量的 `fig_main.pdf`。
- `fig_stability.pdf` —— log 纵轴稳定性图（含 SINDYc 发散）。数据是对的，只是正文没放，
  想强化"有界 vs 发散"可以加。
- `architecture.pdf` / `lane_waypoints.pdf` / `lane_resample.pdf` —— 旧的 20 维路点 + R^29 +
  动态自行车那套，与正文叙述不符，已被 TikZ 图取代。
- `final_state_mse.png` / `fig_rollout_trajectory.png` / `fig_lane_overlay.png` /
  `donkeycarbestlap.png` / `traj1_t0.png` —— 见 §4 的撤下理由。

### 参考资料
- `piwm/papers/PIWMconference version.pdf` —— 会议版原文（注意在 `papers/`，且 `papers/` 被 gitignore）。
- 范例 conf→journal 对（`exampleConf.pdf` / `exampleJounral.pdf`）：旧文档说在
  `C:\Users\suian\Documents\WeChat Files\...\2026-06\`，**这台机器上该路径不存在**，需另找。

---

## 3. 正文当前结构

1. **Introduction** — challenge/motivation 在前，contributions 在后。
2. **Related Work** — 5 子线。
3. **Preliminaries** — CPS、世界模型、物理可解释性、弱监督。
4. **Partial Observability in Driving**〔新〕— Def + 因子化状态 `z^v/z^r`。
5. **PIWM Architecture** — 表示学习；**5.2 编码道路**〔新〕；**5.3 预测道路**〔新〕
   （含 Frenet 方程 `eq:frenet_s/d/psi/kappa` 与学习的执行器映射 `eq:actuation`）；训练三阶段 + Alg.1。
6. **Experimental Evaluation** — 三 Tier；baselines；metrics（`E_xy(k)`，米制距离）。
7. **Results** — Tier-1 定性沿用会议；**7.2 CarRacing 只有定性说明**；**7.3 真车 Frenet 主结果**
   （Table 1 主表 / Table 2 κ 来源消融 / Table 3 δ 噪声 / Fig 5 / Fig 6）。
8. **Discussion** — 交叉点意味着什么、结构而非信息、可解释性、privileged training、limitations、future。
9. **Conclusion** + **Competing Interest / CRediT / Data Availability** + **Acknowledgements** + **Appendix A**。

---

## 4. 数字与出处（接手必读）

### ✅ 正文当前使用的数字

真车、201 个 val 窗口、100 步 rollout、真实米、GT 初始化。
**2026-07-28 全部经实跑逐格验证**（不是抄文档，是重新跑出来对的）：

| 表/图 | 生成脚本 | 验证状态 |
|---|---|---|
| Table 1 主表 | `src/eval_frenet_vs_baselines.py` | ✅ 实跑，5 个模型 × 3 个 horizon 全对 |
| Table 2 κ 消融 | `scripts/compare_kappa_rollout.py` | ✅ 实跑，5 行全对；第 6 行(VQ+Transformer)为本轮新增 |
| Table 3 / Fig 6 δ 噪声 | `scripts/fig_delta_supervision.py` | ✅ 实跑，6 格全对 |
| Fig 5 / Fig 7 | `src/paper_figures.py` | ✅ 我重生成，且与 Table 1 同源 |

> 复现命令（从 `piwm/` 运行，先 `export PYTHONUTF8=1`）：
> ```
> .venv/Scripts/python.exe src/eval_frenet_vs_baselines.py
> .venv/Scripts/python.exe scripts/compare_kappa_rollout.py
> .venv/Scripts/python.exe scripts/fig_delta_supervision.py
> .venv/Scripts/python.exe src/paper_figures.py
> ```

**Table 1 主表 `tab:donkey_main`** — `E_xy` (m)，`src/eval_frenet_vs_baselines.py`

| 模型 | @25 | @50 | @100 |
|---|---|---|---|
| Vid2Param | **0.036** | **0.076** | 0.383 |
| GokuNet | 0.061 | 0.130 | 0.478 |
| DVBF | 0.071 | 0.167 | 0.610 |
| SindyC | 发散 | | ~1e17 |
| **PIWM-Frenet (ours, 感知 κ)** | 0.090 | 0.158 | **0.267** |
| *PIWM-Frenet (oracle κ, 特权)* | *0.085* | *0.148* | *0.246* |

**Table 2 κ 来源消融 `tab:kappa_source`**（`scripts/compare_kappa_rollout.py`）：
地图 0.246 | GT 剖面 0.263 | **从零训 encoder 0.267**（κ-RMSE 0.1193 1/m）| warm-start 0.274 | 冻结 0.290

**Table 3 δ 弱监督噪声 `tab:delta`**：@50 = 0.158 / 0.197 / 0.162，@100 = 0.267 / 0.334 / 0.253（δ=0/5/10%）

**正文里的三个核心论点**（都已写进 Discussion）：
1. 交叉点（约 65 步）之前基线更好，之后我方大幅领先 —— 有界 vs 发散。
2. **结构而非信息**：oracle 只比感知好 ~2 cm；而基线在 t0 还额外拿到 GT 车道航点。
3. 4.5 m 预览覆盖整个 100 步 rollout（实测 mean 3.3 m / p90 4.45 m），所以道路是**观测到的**。

### ❌ 本轮从正文撤下的数字（已在 `.tex` 里注释保留，未删除）

**(a) CarRacing Table 1**（旧 `tab:carracing`，0.002/0.026/0.074/0.306）
标成 "PIWM (ours, full)" 的那一行，逐位等于 `_archive/logs/overnight.log` 第 3 列
`PIWM-bicycle-v4`——而 `src/models/dynamics_bicycle_v4.py` 自述是"动态自行车 + 可学轮胎刚度，
消融变体 #3"，**不含任何道路上下文**。同一次运行里真正带道路的 `PIWM-lane-v5`
（0.0013/0.0333/0.2671/**1.5950**）被从表里删掉了，而且它输。
另外表里是 **MSE**，正文 Metrics 声明的是 RMSE。
→ **要恢复**：用带道路上下文的模型重跑 CarRacing，然后取消注释、按实际 checkpoint 重新标行名、
写明指标、逐句复核分析段。

**(b) 旧真车数字**（单步 0.075 m、最佳圈 0.0238 m、长程 0.05–0.07 m）
出处 `backup.tex`，其模型是 VAE + LSTM，无因子化、无道路上下文、无结构化运动学；
且该文档自称数据来自 **DonkeyCar 模拟器**（`backup.tex:168,185`），与"真车"表述冲突。
另外单步 0.075 与长程 0.05–0.07 是**两个不同的量共用一个符号**（相对窗口位移 vs 整圈全局轨迹），
所以才会出现"100 步误差比 1 步还小"。
最佳圈图 `donkeycarbestlap.png` 是**照片查看器的截图**（含 Windows 工具栏、matplotlib 窗口边框），
且图例写着 `Pred (every stride, true yaw)`——**喂了真值 yaw**，正文没说明。
→ 要恢复：用当前模型重新算、写清每个数字是什么量、从源码重出图、如仍用真值 yaw 必须写明。

---

## 5. 方法叙述 vs 实现（这笔账本轮基本还清了）

旧版正文按"理想模型"写，而驱动代码落地的是 20 维车道路点 + CNN + 单轴距运动学——**两套东西**。
本轮把 Tier-3 换成 Frenet 后，叙述与实现基本对上了：

| 正文 | 实现 |
|---|---|
| `z^r = (e_CTE, e_ψ, κ)` | `frenet_dynamics.py` 状态 `[s, d, psi_e, v, omega]`，`d`=CTE，`psi_e`=航向误差 ✓ |
| 从图像时间窗编码道路 | `road_perception.py` `RoadContextEncoder`：15 帧图像栈 → 前方 10 点曲率剖面 ✓ |
| 道路按弧长重新索引，不外推 | `rollout_perceived()` ✓ |
| privileged 监督，测试withhold | 训练用地图 κ，测试用相机 ✓ |
| 解析运动学 + 少量学习项 | 位姿方程解析，只学 `dv_net`/`dom_net` + 小残差 ✓ |

✅ **§5.1 那处也在 2026-07-28 解决了 —— 而且是用实验解决的，不是改措辞。**

旧版正文声称 driving 用会议版最强配置（extrinsic **VQ-VAE 512 码本 + Transformer 物理编码器**），
而实际编码器是 CNN + 线性 κ 头。我们**把声称的那套实现出来并真的训了**
（`src/models/road_perception_vqformer.py`，逐帧 CNN → VQ-512 → 3 层 Transformer）：

| | κ-RMSE (1/m) | E_xy@100 (m) | 参数 | epoch |
|---|---|---|---|---|
| **CNN + 线性头（论文采用）** | **0.1193** | **0.267** | 1.76M | 25 |
| VQ + Transformer（作为 baseline 写进 Table 2） | 0.1897 | 0.275 | 1.59M | 100 |

同等容量、**4 倍训练预算**、码本健康（~270/512 有效码）、已收敛（最好值在 ep43，之后平台期），
两个指标都更差。所以正文现在的说法是：**会议版的 VQ 结论是在全局可观测基准上得到的，
外推到 driving 不成立，我们实测了**。checkpoint 在 `checkpoints/frenet/kappa_vqformer.tar`。

> ⚠️ **第一次跑这个实验的结果是废的,别引用**:朴素 VQ（`uniform ±1/512` 初始化、无死码复活）
> 码本塌到 12/512、aux loss 前期暴涨 30 倍、25 epoch 远未收敛,得到 κ-RMSE 0.3487。
> 修法是**数据相关初始化 + 死码复活**(见 `road_perception_vqformer.py` 里 `VectorQuantizer` 的注释),
> 修完 perplexity 12→273。拿塌掉的码本去下结论，等于我们这轮一直在清除的那种不公平比较。

---

## 6. 环境与编译

- **本机已装 MiKTeX 25.12**（用户级）：`C:\Users\suian\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`
  未加进 PATH，用之前先 prepend。已开 `[MPM]AutoInstall=1`。
- **编译**（在 `piwm/Jounral_PIWM/`）：`pdflatex` → `bibtex elsarticle-template-num` → `pdflatex` ×2。
  验收：**0 error / 0 undefined / 42 页**。目前只剩 1 处 17 pt overfull（Definition 1，旧有，无害）。
- **看渲染效果**：MiKTeX 自带 Ghostscript，
  `mgs -dNOPAUSE -dBATCH -dQUIET -dFirstPage=N -dLastPage=N -sDEVICE=png16m -r95 -sOutputFile=pgN.png x.pdf`
- **Python 环境（2026-07-28 已建好）**：见 `piwm/README.md` / `piwm/requirements*.txt`。
  - `piwm/.venv` —— Python 3.11.9 + **torch 2.13.0+cu130**，本机是 **RTX 5080（Blackwell, sm_120）**，
    实测 `cuda.is_available()=True`。训练/评估/出图都用它。
  - `piwm/.venv-sindy` —— Python 3.11.9 + **CPU** torch + pysindy，只用来跑 SINDYc（见 README）。
  - ⚠️ 仓库路径含中文（`桌面`），跑 Python 前必须 `export PYTHONUTF8=1`，否则 cp1252 编码报错。

---

## 7. 待办 / 下一步

- [ ] **填作者块**。`.tex` 里现在是显眼的占位符 `[AUTHOR LIST --- TO BE COMPLETED BEFORE SUBMISSION]`，
      上方注释里有会议版作者名可参考。同时填 CRediT、确认 Competing Interest、Data Availability 的仓库地址。
- [x] ~~§5.1 编码器叙述对账~~ **已完成 2026-07-28**，用实验解决（见 §5）。
- [x] ~~用 fresh 基线重跑 `src/paper_figures.py`~~ **已完成 2026-07-28**：
      三张图全部用 fair 基线重生成，`fig_main.pdf` 和 `fig_noise.pdf` 已进正文（Fig 5 / Fig 7）。
      同时修了 `paper_figures.py` 的一个实质 bug —— 它原来画的是 **oracle（查地图 κ，特权）** 模型
      却标成 "PIWM-Frenet (ours)"。现在画的是感知 κ 的可部署模型，oracle 作为虚线参考曲线，
      数字与主表逐格一致（0.090/0.157/0.267 vs 0.085/0.148/0.246）。
      批量实现与 `FrenetDynamics.rollout_perceived` 有断言对拍（max|diff| 4.5e-07）。
- [ ] 5-fold CV：消除 δ=5% > δ=10% 的非单调（已在 Limitations 里如实说明是单次训练方差）。
- [ ] CarRacing：要么重跑补回定量对比（见 §4a），要么保持现在的定性处理。
- [ ] 定 journal 目标（现 `\journal{Robotics and Autonomous Systems}`，占位可改）。
- [ ] 投稿前把主文件名从 `elsarticle-template-num.tex` 改掉（"template" 不该出现在投稿件里）。

---

## 8. 换机器清单

1. 拷**整棵** `<仓库根>\piwm\` 树（含 `Jounral_PIWM/`（含 `imgs/`）、`figures/`、`reports/`、
   `papers/`、`checkpoints/`、`_archive/`、代码、logs）。
2. 拷 `<仓库根>\frame_samples\`（现在正文不再引用，但 `imgs/traj1_t0.png` 的来源在这）。
3. 拷 `<仓库根>\Data_Donkeycar\`、`Data_Donkeycar_prep\`、**`Data_Donkeycar_frenet\`**（第三个旧清单漏了）。
4. **注意 git 里没有的东西**：`checkpoints/`、`*.tar`、`*.log`（含 `overnight.log`）、`vis/`（含
   `_archive/vis/`）、`papers/`（含会议版 PDF）、`figures/*.png`。这些必须手拷。
5. 新机装 **MiKTeX** + **Python 3.11 CUDA torch**。
6. 首次编译验证 **42 页 / 0 undefined**。

---

## 9. 和 `piwm/HANDOFF.md` 的关系

两份文档管两条线，**本轮已经打通**：代码线 §5 的 Frenet 结果就是本文档 §4 正文里的主表。

- `piwm/HANDOFF.md` = 实验/训练线（补 δ=10% 基线、5-fold CV、重出图）。
- 本文件 = 写作线。
- **谁是准的**：数字以 `piwm/HANDOFF.md` §5 和 `src/eval_frenet_vs_baselines.py` 为准；
  正文里的数字必须能追溯到那里。
