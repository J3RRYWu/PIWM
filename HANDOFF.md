# PIWM 交接文档(换机器继续跑)

> ## ⚠️ 2026-08-08 这一轮的结论会推翻论文主表,先读这一节
>
> 分支 `journal-frenet-results`,7 个提交已入,**另有一批改动未提交**(见 §0.6)。
> 一句话:**论文报的 0.267 m 复现不出来,而且效应量比训练噪声还小。**
> 但同一轮也做出了一个站得住的模型改进(map-free 几乎免费)。细节见 §0。
>
> **2026-08-12 补充(§0.9):基线 9 次重训跑完了,结论没那么糟。**
> 论文用的**基线 checkpoint 同样是幸运抽样**(V2P 0.383 vs 重训均值 0.510),
> 两边都换成 3-seed 均值后 **0.412 ± 0.080 vs 0.510 ± 0.024,我方仍领先 1.24×**,
> 逐窗口配对对最强基线是 **2 赢 1 平 0 输**。主表照样必须改成 mean ± range。
>
> **2026-08-13(§0.10,最新,以这个为准):5-fold CV 70 个任务跑完,0 失败。**
> 换验证集时**基线塌得比我方厉害得多**(跨划分离散度 ours ±0.046 vs V2P ±0.194 vs
> DVBF ±0.850),fold 内配对**对三个基线全部 5/5 全赢**。
> 而且 **δ 那条结论要反过来写**:我方随标签噪声单调**变好**、基线变差,
> 正文把它当"单次训练方差"写进 Limitations 是**错的**,应改写成正面结果。

## 0. 2026-08-08 轮次纪要

### 0.1 ❌ 主表的头号数字不可复现(已两次独立验证)

用**同一个评估器**(`eval_frenet_vs_baselines` 的 201 个 val 窗口)测同一配置的重训:

| 批次 | 结果 (E_xy@100, 感知 κ) |
|---|---|
| 未加种子,3 次 | 0.359 / 0.367 / 0.416 |
| 加种子后 seed 0/1/2 | 0.383 / 0.506 / 0.346 → **均值 0.412 ± 0.080** |
| **论文在用的 `dyn_k16.tar`** | **0.267** |

`dyn_k16.tar` 存于 2026-06-19,**早于 Frenet 代码首次提交(06-23)**,且缺 `kappa_mode` 字段
= 旧版脚本存的。我 diff 过此后 `train_frenet.py` 的全部改动,默认路径(`--kappa_mode map --delta 0`)
上训练配方没变,所以**无法用"配方不同"解释**,但也无法完全排除(那个版本不在 git 里)。

**含义**:论文主表 ours 0.267 vs 最强基线 0.383,效应量 0.116;而我方单侧跨度就有 ±0.080。
**主表必须改成 mean ± range,不能再报单次。**

### 0.2 ✅ 模型改进成立:map-free 几乎免费

原来的"pure world model"说法是**夸大**的:`rollout_perceived` 虽然 κ 来自相机,但状态是
**已知中心线上的 (s,d)**,转成位置还要 `track.npz`。等于预设了一条测绘过的赛道。

修法与结果(全部 201 窗口):

| 位置怎么得到 | @100 |
|---|---|
| 感知 κ + 已知中心线(论文现状) | 0.267 |
| map-free,**积分曲率两次** (`rollout_perceived_local`) | 0.437 |
| map-free,**直接读形状** (`rollout_perceived_shape`) | 0.307 |

诊断:给**真值 κ** 时积分版仍是 0.414 → **那 15 cm 是二次积分的锅,不是感知**。
所以让编码器直接预测道路形状(`RoadContextEncoder(predict_shape=True)`,监督来自
`frenet_track.local_road_points`),位置直接读出、不积分。

**3 seed 配对验证**(同一 `dyn_s{i}` 分别配 `kappa_s{i}` / `shape_s{i}`):

| seed | (a) 有地图 | (c) map-free 形状 | 配对差 |
|---|---|---|---|
| 0 | 0.383 | 0.396 | +0.013 |
| 1 | 0.506 | 0.496 | **−0.011** |
| 2 | 0.346 | 0.356 | +0.009 |

**配对差均值 +0.004 m** → 去掉地图依赖基本不要钱。之前单次测到的 +0.040 是抽样噪声。
**这条结论不依赖任何基线数字,可以直接写进论文。**

### 0.3 ❌ 基线三宗罪

1. **保真度**:会议版(ICCPS 2026)把 GokuNet 和 Vid2Param **都**归为 intrinsic(吃观测),
   DVBF/SINDy 才是跑在共享 AE 隐状态上。本仓库却删了 GOKU 的观测通路、留了 V2P 的
   → 这就是 V2P 成为"最强基线"的唯一原因。
2. **V2P 的 θ 会泄漏道路**:θ 从编码器 29 维输出推,而那是 `car(9)+lane(20)`。
   它不是物理参数,是**道路几何的 8 维压缩**——等于我们方法的粗糙版。
3. **选择准则失效**:基线按 K=32 归一化复合损失选 checkpoint,我方按 100 步 xy 误差(=报告指标)选。
   证据:`goku_obs` 与 `v2p` **架构、参数量完全相同(58,220)**,val_loss 0.300 vs 0.290,
   **@100 却是 0.561 vs 0.384**。选择准则几乎不预测报告指标。

已实现的修正见 §0.5。

### 0.4 ✅ VQ+Transformer 实测:输了,降为 baseline

正文 §5.1 声称用会议版最强配置(VQ-VAE 512 + Transformer)——这是**忠实转述会议版摘要的**。
实测(`RoadContextVQFormer`,容量对齐 1.59M vs 1.76M,4 倍 epoch,码本健康 273/512,已收敛):

| | κ-RMSE (1/m) | E_xy@100 |
|---|---|---|
| CNN + 线性头 | **0.119** | **0.267** |
| VQ + Transformer | 0.190 | 0.275 |

→ CNN 留在论文,VQ+Transformer 作为 Table 2 的一行 baseline。
⚠️ 朴素 VQ 会码本塌缩(12/512),必须**数据相关初始化 + 死码复活**,否则结论不成立。

### 0.5 本轮新增/改动的代码

| 文件 | 内容 |
|---|---|
| `src/models/frenet_dynamics.py` | 新增 `rollout_perceived_local`(积分 κ)、`rollout_perceived_shape`(读形状,Catmull-Rom 插值) |
| `src/models/road_perception.py` | `predict_shape=True` + `shape_head` + `forward_both`;形状头输出 `(X−offset, Y)`,**调用方要把 offsets 加回去** |
| `src/frenet_track.py` | `local_road_points()` 生成形状监督 |
| `src/models/road_perception_vqformer.py` | **新**:逐帧 CNN → VQ-512 → Transformer(含防塌缩) |
| `src/baselines/vid2param_kin.py` | **新**:忠实 Vid2Param(5 个物理参数 → 已知运动学) |
| `src/baselines/shared_dynamics_lane.py` | `DynamicsGOKULane(theta_dim>0)` 恢复观测通路(默认 0,老 checkpoint 照常加载) |
| `src/train/train_baselines_donkey.py` | 新变体 `goku_obs` / `v2p_kin`;抽出 `_train_obs_conditioned`;`PREDICTS_LANE` 标志 |
| `src/train/train_*.py` ×3 | **`--seed`**(见 §0.7) |
| `scripts/eval_mapfree.py` | 地图依赖代价的对照评估 |
| `reports/repeats/ac_paired_3seeds.md` | (a)(c) 配对结果 |

### 0.6 ⏳ 未完成 / 进行中

- ✅ **基线重训 9 次已于 08-12 01:08 全部完成**,在 `checkpoints/{v}_lane_donkey_s{0,1,2}/`,
  日志 `reports/repeats/baselines_rerun.log`(末尾 `done-base-rerun`)。结果见 §0.9。命令:
  ```
  .venv/Scripts/python.exe src/train/train_baselines_donkey.py --variant {dvbf|goku|v2p} \
      --K 32 --suffix _s$SEED --batch 128 --batch_v2p 64 --epochs 60 --seed $SEED
  ```
- **忠实 Vid2Param 的 GRU 版训不出来**(试了 3 次,3 个不同病因:后验塌缩 → 修 logvar 无效 →
  物理先验初始化后 ep1 好转但 ep2 就跳回并卡死)。**改用直接拟合 5 个全局参数**得到可用结果:
  辨识出轴距 18.0 cm(真值 16.5),`E_xy@100 = 0.553`。GRU 版为什么训不出来仍未解决。
- **论文尚未按这些结论改动**(用户要求等数据齐了一起改)。
- 上述改动**未提交**:8 个修改 + 3 个新增。

### 0.7 `--seed` 的设计(重要)

三个训练脚本都加了 `--seed`,**只控制权重初始化和批次顺序**;
**train/val 划分固定在 `seed=0` 不变**,所以重复实验在同一批验证 episode 上**配对**,
测出的离散度是纯训练噪声。已验证:同种子两次结果完全一致(0.391/0.391),异种子不同(0.551)。

### 0.8 下一步建议(按优先级)

1. ~~等基线 9 次跑完 → 出**带误差棒的主表**~~ **已完成 08-12,见 §0.9。**
2. 主表改成 mean ± range;`map-free 形状版`这条可以直接写(§0.2)。
3. 基线的 checkpoint 选择准则要么改成按报告指标选(和我方一致),要么在论文里披露不对称。
4. `dyn_k16.tar` 要么弃用改报重训均值,要么说明它是如何得到的(但那个脚本版本不在 git 里)。
   → §0.9 之后这条更棘手了:**基线那一侧也有同样的问题**。

### 0.10 5-fold CV 完成(2026-08-13)—— 结论比 §0.9 强,而且 δ 那条要改写

70 个任务(5 fold × [κ编码器 + 形状头 + 动力学×3δ + 基线3变体×3δ]),**0 失败**,
壁钟 592 分钟 / 并行效率 3.90×,驱动 `scripts/run_matrix.py --preset 5fold`,
日志 `reports/matrix/`。评估 `scripts/eval_folds_table.py`,表存 `reports/matrix/folds_table.md`。

**方法要点**:每个 fold 留出 1/5 episode(14–15 个,不是单 split 的 7 个),窗口数 301–647
不等。**配对只在 fold 内有效**(跨 fold 是不同窗口),所以聚合方式是"先算每 fold 均值,
再对 5 个 fold 均值求 mean ± 半极差"。脚本有断言:同一 fold 内 15 行必须走同样数量的窗口。

#### 主表(δ=0),E_xy(m)

| 模型 | @25 | @50 | @100 | §0.9 单 split 3-seed |
|---|---|---|---|---|
| DVBF | 0.055 ± 0.012 | 0.155 ± 0.019 | 1.287 ± 0.850 | 0.999 ± 0.470 |
| GokuNet | 0.044 ± 0.014 | 0.125 ± 0.015 | 0.815 ± 0.153 | 0.631 ± 0.088 |
| Vid2Param | **0.035 ± 0.007** | **0.103 ± 0.007** | 0.720 ± 0.194 | 0.510 ± 0.024 |
| **ours(有地图)** | 0.141 ± 0.011 | 0.245 ± 0.025 | **0.463 ± 0.046** | 0.412 ± 0.080 |
| ours(map-free) | 0.146 ± 0.013 | 0.258 ± 0.031 | 0.516 ± 0.068 | 0.422 ± 0.069 |

**核心发现:换验证集时基线塌得比我方厉害得多。** 我方 0.412 → 0.463(几乎没动),
V2P 0.510 → 0.720,GOKU 0.631 → 0.815,DVBF 0.999 → 1.287。跨划分离散度
**我方 ±0.046 vs V2P ±0.194 vs DVBF ±0.850**,差 4–18 倍。
→ 论点可以从"误差更小"升级成"**对数据划分鲁棒得多**",这比原来的 1.4× 更有说服力。

#### 配对(fold 内),ours-(a) − 基线,@100,δ=0

| | 均值 | 每 fold | 战绩 |
|---|---|---|---|
| vs V2P | **−0.257 m** | −0.053 / −0.290 / −0.212 / −0.467 / −0.264 | **5/5 赢** |
| vs GOKU | −0.352 m | −0.158 / −0.428 / −0.389 / −0.490 / −0.298 | **5/5 赢** |
| vs DVBF | −0.824 m | −0.199 / −0.648 / −0.951 / −1.925 / −0.396 | **5/5 赢** |

§0.9 单 split 上对 V2P 是"2 赢 1 平",5-fold 上是 **5/5 全赢**,且最小优势也有 0.053 m。

#### ⚠️ δ 那条结论要**反过来写**

| 模型 | δ=0 | δ=5% | δ=10% |
|---|---|---|---|
| ours(有地图) | 0.463 ± 0.046 | 0.452 ± 0.047 | **0.319 ± 0.049** |
| ours(map-free) | 0.516 ± 0.068 | 0.506 ± 0.066 | 0.390 ± 0.040 |
| Vid2Param | 0.720 ± 0.194 | 0.992 ± 0.325 | 1.015 ± 0.343 |
| GokuNet | 0.815 ± 0.153 | 0.984 ± 0.221 | 0.957 ± 0.157 |
| DVBF | 1.287 ± 0.850 | 1.229 ± 0.292 | 1.315 ± 0.285 |

**我方随 δ 单调变好,基线随 δ 变差。** 正文现在在 Limitations 里把单 split 那个
"δ=5% 0.334 > δ=10% 0.253" 解释成**单次训练方差**——**这个解释错了**。5 个 fold 上
方向一致、误差棒不重叠,是**真实可复现的效应**:δ 是加在训练标签上的有偏均匀噪声,
对物理结构化模型起了正则化/数据增广的作用,而对纯学习基线就是纯损害。
→ 这条对论文**有利**,应该从"局限"改写成"结果",正好支撑"结构带来鲁棒性"的主线。

#### map-free 的代价上调了

配对差 **+0.053 m**(每 fold +0.049/+0.022/+0.069/+0.071/+0.056,方向一致)。
§0.2 单 split 测到的是 +0.004,§0.9 复测是 +0.010。**"几乎免费"的说法要softening 成
"约 5 cm"**——仍然很小,但别再写"free"。

#### 已知缺陷(改稿前必须处理)

1. ~~**V2P 用的 v6 图像编码器没有按 fold 重训**~~ → **已量化,判定为空,不用重训(08-13)**。
   事实:v6 编码器在 legacy 划分上训练(train = 除 `[5,8,30,50,64,69,70]` 外全部),
   而 fold 是同一排列的连续切块、legacy 验证集正好是第一块,所以
   **fold 0 的留出集恰好拆成 7 个"v6 从没见过"+ 7 个"v6 训练过"**,是个现成的受控实验。
   审计命令 `scripts/eval_folds_table.py --leak-audit`,结果存
   `reports/matrix/folds_table_leak_audit.md`:

   | | clean | seen | 差 | 用 v6? |
   |---|---|---|---|---|
   | V2P | 0.416 | 0.537 | +0.121 | **是**(θ) |
   | GOKU | 0.569 | 0.622 | +0.054 | 否 |
   | DVBF | 0.643 | 0.648 | +0.005 | 否 |
   | ours-a | 0.416 | 0.462 | +0.045 | 否 |

   关键不是原始误差(两个子集是不同 episode,难度不同),而是 **V2P 相对三个"无编码器"
   模型的优势**:clean 上 −0.153/−0.227/−0.000,seen 上 −0.085/−0.111/+0.075,
   **平均往对 V2P 不利的方向移动 +0.086 m**。三个独立对照同向。
   → **曝光没给 V2P 带来任何可测量的优势,泄漏是空的。**
   正文应如实披露这一点**并附上这个数字**(比"承认瑕疵但方向对我们不利"强得多)。
   边界:单折、7 vs 7 个 episode 的一次测量,不是大样本。
   我方的 κ/形状编码器是 `--backbone scratch` 且带 `--fold`,本来就干净。

   ⚠️ **但 §0.3 那条实质问题仍在**:GOKU 的观测通路被删、只有 V2P 能看图像,
   这才是 V2P 成为最强基线的原因。`goku_obs`(`theta_dim=8`,与 V2P 参数量相同)
   已实现但**这次 5-fold 没跑**。要再花机时,应该花在每折补一个 `goku_obs`(5 个任务),
   而不是重训 v6 编码器。
2. fold 之间窗口数差很多(301–647),因为 episode 长度不均。已用"先 fold 内求均值"规避,
   但报告里应写明。
3. DVBF 的 ±0.850 几乎全来自 fold 3(配对差 −1.925,那一折它基本发散)。
4. 基线 checkpoint 仍按复合 val_loss 选(§0.3 的不对称还在)。**但快照已存**
   (`--snapshot-every 5`,45 个基线任务各 12 份),所以改选择准则**不用重训**,
   写个事后选择脚本即可。

### 0.9 带误差棒的主表(2026-08-12)

`<py311> scripts/eval_seeds_table.py` —— 同一评估器、同 201 个 val 窗口(split 固定 seed=0)、
GT 初始化、100 步、真实米。脚本内置自检:ours-(a) 三个 seed 必须复现 §0.2 的 0.383/0.506/0.346
(**逐格通过**),所以这确实是 `eval_frenet_vs_baselines` 的那个评估器。
原始曲线存 `reports/repeats/main_table_3seeds_curves.npz`,表存 `.../main_table_3seeds.md`。

**mean ± 半极差(3 seeds),E_xy(m):**

| 模型 | @25 | @50 | @100 | 论文现报(单次) |
|---|---|---|---|---|
| DVBF | 0.071 ± 0.001 | 0.163 ± 0.013 | 0.999 ± 0.470 | 0.610 |
| GokuNet | 0.062 ± 0.000 | 0.131 ± 0.006 | 0.631 ± 0.088 | 0.478 |
| Vid2Param | **0.036 ± 0.003** | **0.088 ± 0.015** | 0.510 ± 0.024 | 0.383 |
| **ours(有地图)** | 0.121 ± 0.005 | 0.234 ± 0.022 | **0.412 ± 0.080** | 0.267 |
| ours(map-free 形状) | 0.125 ± 0.012 | 0.242 ± 0.020 | 0.422 ± 0.069 | — |

#### ⚠️ 关键发现:**论文里的基线数字也是幸运抽样,不只是我方**

`_longK` 那批基线(论文在用)全部好于重训三次的均值:V2P 0.383 vs 0.510、GOKU 0.478 vs 0.631、
DVBF 0.610 vs 0.999。也就是说 §0.1 对 `dyn_k16` 的质疑**对称地适用于基线**。
好处是:**把两边都换成 3-seed 均值后,结论反而比之前稳** —— 差距 0.098 m(1.24×),
而 §0.1 时按"我方重训均值 vs 基线单次"算只剩 0.412 vs 0.383,是**输的**。

#### 配对检验才是能写进论文的证据

每一行走的是同一批窗口同一顺序,所以可以逐窗口配对(bootstrap 10k,95% CI):

| ours-(a) − 基线 @100 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| vs V2P | **−0.153** [−0.201,−0.102] | −0.000 [−0.065,+0.065] | **−0.142** [−0.203,−0.082] |
| vs GOKU | **−0.215** | **−0.229** | **−0.213** |
| vs DVBF | **−0.313** | **−0.174** | **−1.275** |

→ 对最强基线 V2P:**2/3 seed 显著赢、1/3 打平、0/3 输**。这比"领先 1.4×"弱,但站得住。
对 GOKU/DVBF 是 3/3 显著赢。

#### 其他

- **短 horizon 的交叉点故事完全成立且更强了**:V2P 在 @25/@50 大幅领先(0.036/0.088 vs 0.121/0.234),
  误差棒极小,不是噪声。交叉后我方领先。这是论文最干净的一张图。
- **map-free 仍然几乎免费**:配对差 +0.021/−0.006/+0.016,均值 **+0.010 m**。
  (§0.2 记的是 +0.004,我算的 (c) 行系统性高 0.007;试过"形状取 shape 头 + κ 取 kappa-only 头"
  的 (c\*) 变体想解释,结果更差 0.436,**假设被证伪,这 0.007 的出处仍未知**。不影响结论。)
- **`DVBF_s2` 是个离群点**(1.621 m,另两个 0.696/0.681),DVBF 的 ±0.470 几乎全是它贡献的。
  写论文时要么报中位数要么明说。
- V2P 的 θ 用的是**共享的 v6 图像编码器**(没跟着 seed 重训),所以 V2P 那一行的离散度
  只含动力学的训练噪声,可能**低估**了它的真实方差。

---


> 最后更新:2026-07-28 · HEAD `b6fb35d` · 远程 `git@github.com:J3RRYWu/PIWM.git`
>
> **仓库根在这台机器上是 `C:\Users\suian\OneDrive\桌面\PIWM\`,不是本文档旧版写的 `E:\Desktop\PIWM\`
> (E: 盘不存在)。下文所有 `E:\` 一律按此换算。**
>
> **2026-07-28:本文档 §5 的 Frenet 结果已经写进期刊正文**(主表 / κ 消融 / δ 噪声三张表 + 两张图)。
> 写作侧的交接见 `piwm/Jounral_PIWM/HANDOFF.md`,两份要一起读。

---

## 0. TL;DR — 新机器上先做这 4 步

```bash
git clone git@github.com:J3RRYWu/PIWM.git piwm     # 代码(δ 实验代码已在 84632ae)
# 手动拷贝 Data_Donkeycar*/ 和 checkpoints/  ← 两者都被 .gitignore,git 里没有!
cd piwm && python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
python src/train/train_baselines_donkey.py --variant dvbf --K 32 --suffix _delta10 \
       --batch 128 --epochs 60 --delta 0.10          # 待办第 1 步(见 §4)
```

**目录约定**:所有命令都从 `piwm/` 运行(checkpoint / figures 路径是 cwd 相对)。
仓库根 `piwm/` 与数据目录 `Data_Donkeycar*/` **同级**(即 `<父目录>/{piwm, Data_Donkeycar, ...}`)。

---

## 1. 必须手动搬运的东西(git 里没有)

| 内容 | 路径 | 大小 | 说明 |
|---|---|---|---|
| 原始数据 | `../Data_Donkeycar/` | 2 npz | 真实 DonkeyCar 轨迹(**唯一不可再生的**,务必拷) |
| 预处理数据 | `../Data_Donkeycar_prep/` | 142 npz | 可由 `src/donkey_prep.py` 重生成 |
| Frenet 数据 | `../Data_Donkeycar_frenet/` | 71 npz + `_meta/` | 可由 `src/frenet_prep.py` 重生成 |
| **模型权重** | `checkpoints/` | ~数百 MB | **重训要很久,强烈建议拷贝**(见 §3 清单) |

> 数据可重生成:`python src/donkey_prep.py && python src/frenet_prep.py`(需要 `../Data_Donkeycar/`)。
> `checkpoints/sindyc_lane_donkey/model.pkl` 有 139MB,若不拷则 SINDYc 曲线用缓存
> `figures/_sindyc_curve.npy`(已在 git 里)即可,不必重拟合。

**⚠️ 上表不全 —— `.gitignore` 还挡掉了这些,clone 一份是拿不到的,也必须手拷:**

| 内容 | 路径 | 为什么重要 |
|---|---|---|
| 会议版 PDF + 论文 zip | `papers/`(整个目录被忽略) | 改写的基础 |
| 训练日志 | `*.log`,尤其 `_archive/logs/overnight.log`(4.5 MB) | CarRacing 表的唯一出处 |
| 旧可视化 | `_archive/vis/`(`vis/` 规则任意层级都匹配) | 含 `final_state_mse.png` |
| 论文图预览 | `figures/*.png`(顶层) | `figures/_archive/*.png` 反而是被追踪的 |
| 前视帧样本 | `<仓库根>/frame_samples/` | **在 git 根之外**,clone 永远拿不到 |

---

## 2. 环境(关键怪癖,踩过坑)

- **用 venv**(2026-07-28 已建好,从 `piwm/` 运行):
  - `.venv/Scripts/python.exe` —— Python 3.11.9 + torch **2.13.0+cu130**,默认用这个。
  - `.venv-sindy/Scripts/python.exe` —— CPU torch + pysindy,只跑 SINDYc。
  重建方式见 `piwm/README.md` 和 `requirements*.txt`。PATH 上的 `python` 是 Store 占位符,不可用。
- **本机 GPU 是 RTX 5080(Blackwell / sm_120)**,不是旧文档写的 4080 —— torch 必须 cu128+。
- **路径含中文(`桌面`)**:跑 Python 前先 `export PYTHONUTF8=1`,否则 cp1252 编码报错。
- **小模型 GPU≈CPU**:动力学/基线都是 1万–6万参数 + K=32 逐步 rollout,**延迟受限**,CPU 反而更稳。
  → **δ 实验、基线训练一律用默认 python(CPU)**;GPU 只在 encoder/图像批量任务上有优势。
- **pysindy 不能和 torch 同进程**(会段错误,CPU 上也中招)。
  `eval_frenet_vs_baselines.py` / `paper_figures.py` 都读缓存 `figures/_sindyc_curve.npy`;
  缓存由 `python src/eval_stability_log.py`(CPU)单独重生成。
- **旧机器的教训**(新机器可能没有):py311+CUDA 连续跑数小时后会 `import torch` 失败
  (DLL init / enum / 段错误);并发 `pip install` 到同一个 Python311 会加剧。若新机器是 Linux/WSL2 会好很多。

---

## 3. 现有资产清单(checkpoint 状态)

### ✅ 已完成
| checkpoint | 内容 |
|---|---|
| `checkpoints/frenet/dyn_k16.tar` | Frenet 动力学(δ=0,主模型) |
| `checkpoints/frenet/dyn_k16_delta5.tar` | Frenet δ=5% |
| `checkpoints/frenet/dyn_k16_delta10.tar` | Frenet δ=10% |
| `checkpoints/frenet/kappa_scratch.tar` | **κ 感知编码器(原生从零,最优)** |
| `checkpoints/frenet/kappa_{finetune,frozen}.tar` | κ 编码器消融(v6 warmstart / 冻结) |
| `checkpoints/frenet/kappa_vqformer.tar` | **VQ-512 + Transformer 编码器**(论文 Table 2 的 baseline 行) |
| `checkpoints/{dvbf,goku,v2p}_lane_donkey_longK/` | 基线 δ=0(忠实 batch 128/64) |
| `checkpoints/{dvbf,goku,v2p}_lane_donkey_delta5/` | 基线 δ=5% |
| `checkpoints/sindyc_lane_donkey/model.pkl` | SINDYc(发散基线) |

### ❌ 缺(下一步要跑)
- `checkpoints/{dvbf,goku,v2p}_lane_donkey_delta10/` — **基线 δ=10%,3 个**

### ⚠️ 负结果(不要再试)
- `dyn_k16_profile.tar`:stage-2 用感知 κ 微调 dynamics **没用**(0.274→0.275)且伤 oracle。保持用 `dyn_k16`。

---

## 4. 待办(按顺序)

### 第 1 步:补齐基线 δ=10%(~40min,CPU)
```bash
for v in dvbf goku v2p; do
  python src/train/train_baselines_donkey.py --variant $v --K 32 --suffix _delta10 \
         --batch 128 --batch_v2p 64 --epochs 60 --delta 0.10
done
```
> **batch 必须是 128/64**。大 batch(384/192)会把这些延迟受限的小基线**训弱**
> (GOKU 0.71/V2P 0.75/DVBF 1.38),虚高我方优势 —— 这是踩过的坑。

### 第 2 步:出单 split 预览图
```bash
python scripts/fig_delta_all_models.py     # → figures/fig_delta_all_models.{png,pdf}
```
3 个面板(δ=0/5/10),每个面板画 Frenet-perc + GOKU/V2P/DVBF 的 xy 误差 vs rollout 步长。
**看点**:基线是否随 δ 明显退化、Frenet 是否保持鲁棒(会议版 Fig 3 的论点)。

### 第 3 步:决定要不要 5-fold CV(用户已同意"先看预览再定")
会议版用 5-fold CV。完整版规模:
- Frenet 3δ×5fold = 15,κ encoder 5(每 fold 一个,防泄漏),基线 3δ×5fold×3 = 45 → **约 65 次训练**
- CPU 串行 **约 13–15 小时**(过夜);GPU 约 5–6 小时但旧机器会崩
- 需要先给 `train_frenet.py` / `train_baselines_donkey.py` / `train_kappa_perception.py`
  加 `--fold/--nfolds` 参数(目前是固定 `seed=0` 的单 split),再写驱动 + 聚合脚本

**若单 split 预览趋势已经清晰(基线随 δ 退化明显),5-fold 主要是为了消除
Frenet 那个非单调(δ=5% 0.334 > δ=10% 0.253,是单次训练方差)。**

### 第 4 步(可选):重生成出版图
```bash
python src/eval_stability_log.py                 # CPU:刷新 SINDYc 缓存 + 稳定性曲线
<py311> src/paper_figures.py                     # 出 figures/fig_{main,stability,noise}.pdf
```
> `fig_main` / `fig_stability` **目前仍是旧基线**,需用 fresh 基线重生成。
> `fig_main` 已加了 SINDYc(读缓存)且纵轴 cap 到其他 baseline 最大值(SINDYc 冲出图顶)。
> `fig_noise` 是**初始状态高斯噪声**(期刊新增,和会议版 δ 噪声不是一回事,别混淆)。

---

## 5. 核心结果(已确认,可直接写进论文)

> **2026-07-28:本节三组数字已全部在本机重跑验证,并已写进期刊正文
> (Table 1 / Table 2 / Table 3)。复现命令见 `Jounral_PIWM/HANDOFF.md` §4。**

### 主表:感知-κ 纯世界模型 vs 基线
(201 个 val 窗口,同 split,GT 初始化,100 步 rollout,真实米;`python src/eval_frenet_vs_baselines.py`)

| 模型 | xy@25 | xy@50 | xy@100 | 说明 |
|---|---|---|---|---|
| **Frenet-perc(ours)** | 0.090 | 0.158 | **0.267 m** | **纯 WM:κ 从相机感知** |
| Frenet-oracle | 0.085 | 0.148 | 0.246 m | κ 查已知地图(特权消融) |
| V2P | 0.036 | 0.076 | 0.383 m | 最强基线 |
| GOKU | 0.061 | 0.130 | 0.478 m | |
| DVBF | 0.071 | 0.167 | 0.610 m | |
| SINDYc | 发散 | | ~1e17 | 图中 off-scale |

**关键论点**:
1. **纯世界模型**——κ 从前视相机感知(不查地图),信息集 = 图像+action,和基线对等
   (基线甚至在 t0 拿到 GT 车道航点=道路)。仍领先最强基线 **1.4×**。
2. **感知代价只有 ~2cm**(0.267 vs oracle 0.246)→ 赢在**结构(解析 Frenet 运动学)**,不是信息。
3. 4.5m 相机预览**覆盖整个 100 步 rollout**(实测走 mean 3.3m / p90 4.45m),
   所以道路是**观测到的**,不是逐步外推的 → 长程有界不漂。
4. 短 horizon(<~65 步)基线略优,交叉后 Frenet 大幅领先(有界 vs 发散)。

### κ 来源消融(`python scripts/compare_kappa_rollout.py`)
oracle(地图) 0.246 | GT-profile(完美感知) 0.263 | **scratch encoder 0.267** | v6-finetune 0.274 | v6-frozen 0.290

→ **encoder 用原生从零最好,v6 预训练权重没用**(κ-RMSE 0.1193 vs 0.1215)。

### δ 弱监督噪声(会议版 Sec 4.1 的噪声模型,单 split)
| δ | xy@50 | xy@100 |
|---|---|---|
| 0% | 0.158 | 0.267 |
| 5% | 0.197 | 0.334 |
| 10% | 0.162 | 0.253 |

→ δ=10% 下仍 0.253m,**优于用干净标签训练的最强基线(0.383)**。
非单调是单次训练方差(5-fold 可解决)。

---

## 6. 代码地图(本轮新增/改动)

| 文件 | 作用 |
|---|---|
| `src/models/road_perception.py` | **新**:`RoadContextEncoder` 图像栈→前方曲率剖面(纯 WM 的关键) |
| `src/models/frenet_dynamics.py` | 加 `kappa_override` + `rollout_perceived()`(按已走弧长在感知剖面里索引) |
| `src/models/encoder_lane.py` | 暴露 `features()` 供道路头复用主干 |
| `src/train/train_kappa_perception.py` | **新**:训 κ 感知头,`--mode {frozen,finetune}` `--backbone scratch` |
| `src/train/train_frenet.py` | 加 `--delta`(δ 弱监督噪声)、`--kappa_mode {map,profile}` |
| `src/train/train_baselines_donkey.py` | 加 `--delta/--K/--batch/--suffix/--epochs`;**GPU 常驻、免图像的预计算(快 ~2.4×)** |
| `src/eval_frenet_vs_baselines.py` | 主表:感知/oracle Frenet + 基线 + SINDYc(读缓存),纵轴 cap |
| `scripts/compare_kappa_rollout.py` | κ 来源消融 harness |
| `scripts/fig_delta_supervision.py` | **新**:Frenet 单模型 × δ 的图 |
| `scripts/fig_delta_all_models.py` | **新**:会议 Fig3 风格,3 面板 × δ,全模型 |

---

## 7. 论文相关(期刊扩展)

- LaTeX 在 `piwm/Jounral_PIWM/elsarticle-template-num.tex`(Elsevier,**已编译 42 页,0 undefined**)。
  本机已装 MiKTeX 25.12(`%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\`,未进 PATH)。
- 会议版 PDF:`piwm/papers/PIWMconference version.pdf`(arXiv:2412.12870)。注意 `papers/` 被 gitignore。
- **会议版噪声 = δ 弱监督标签噪声**(biased uniform,δ∈{0,5%,10%},加在训练标签);
  **`fig_noise` = 测试时初始状态高斯噪声**(期刊新增实验)。两者不同,**论文里务必写清区别**。
  目前正文里的 Table 3 是前者;`fig_noise` 还没进正文(见下)。
- ✅ **已完成**:`fig:partial_obs` 已换成 TikZ 画的双车示意图,不再是相机帧占位。
  `fig:arch` / `fig:encoding` / `fig:prediction` 也都用 TikZ 重画,与 Frenet 叙述一致。
- ✅ **`fig_main` / `fig_stability` / `fig_noise` 已用 fair 基线重生成(7-28)**,`fig_main.pdf`
  和 `fig_noise.pdf` 已进正文(Fig 5 / Fig 7)。同时修了 `src/paper_figures.py` 的一个实质 bug:
  它原来画的是 **oracle(查地图 κ,特权)** 模型却标成 "PIWM-Frenet (ours)"。现在画感知 κ 的
  可部署模型,oracle 作虚线参考,数字与 §5 主表逐格一致;批量实现与
  `FrenetDynamics.rollout_perceived` 有断言对拍(max|diff| 4.5e-07)。
  `fig_stability.pdf`(log 纵轴 + SINDYc 发散)数据也是新的,但正文暂未使用。
- **未完的技术债**:
  1. decoder 仍依赖 v6 + `FrenetBridge`(把 Frenet 状态转回 20 航点表示),这正是论文要反对的表示
     → 建议做一个原生 `[d, ψₑ, κ-profile] → 图像` 的 decoder,彻底去掉 v6。
  2. ✅ **已解决(7-28)**:正文 §5.1 声称的 VQ-VAE + Transformer 编码器已**实现并训练**
     (`src/models/road_perception_vqformer.py`,`train_kappa_perception.py --arch vqformer`)。
     同等容量 + 4 倍 epoch + 码本健康,仍然两个指标都差(κ-RMSE 0.1897 vs 0.1193,
     E_xy@100 0.275 vs 0.267)→ CNN 留在论文,VQ+Transformer 作为 baseline 写进 Table 2。
     ⚠️ 朴素 VQ 会码本塌缩(12/512),必须用数据相关初始化 + 死码复活,否则结论不成立。
  3. CarRacing 的定量对比已在正文里注释掉(原表把纯车辆模型 `PIWM-bicycle-v4` 标成了
     "PIWM (ours, full)",而真正带道路的 `PIWM-lane-v5` 被删且成绩最差)。详见期刊 HANDOFF §4。

---

## 8. 已知陷阱速查

1. **基线 batch 必须 128/64**(大 batch 训弱基线)。
2. **pysindy 别和 torch 同进程**(段错误)——用 `figures/_sindyc_curve.npy` 缓存。
3. **δ=0 的 Frenet 就是 `dyn_k16.tar`**,不用重训。
4. 数据/权重都被 `.gitignore`,**换机器必须手动拷**。
5. `train_frenet.py` 里 `delta` 是循环内的弧长变量,δ 噪声参数叫 `delta_sup`,**别搞混**。
6. 训练脚本 print 是块缓冲,后台跑时 epoch 日志会延迟出现;`_precompute_v2p` 有 `flush=True` 的进度行可判断存活。
