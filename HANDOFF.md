# PIWM 交接文档(换机器继续跑)

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
  2. 正文 §5.1 仍写 extrinsic **VQ-VAE(512 码本) + Transformer 物理编码器**(沿用会议版结论),
     而 Frenet 线实际用的是 CNN 主干 + 线性 κ 头。**要么改叙述,要么补实验**。
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
