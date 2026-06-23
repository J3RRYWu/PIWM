# PIWM — Physically Interpretable World Model

会议版(CarRacing 仿真)→ 期刊版(新增 **DonkeyCar 真车**)的世界模型代码。
**当前活代码 = DonkeyCar 上的 Frenet 世界模型**;CarRacing 会议代码已归档到 `_archive/carracing_conf/`。

---

## 一句话核心思想
前视相机 + **固定赛道**下,把会漂移的「20 个自由车道航点」换成**物理 Frenet 状态**
`z = [s, d, ψₑ, v, ω]`(弧长 / 横偏=CTE / 朝向误差 / 速度 / 角速度)**+ 已知赛道曲率 κ(s) 查表**。
道路形状不再是要预测的自由量(查表得到)→ **长 horizon rollout 不会漂** → 100 步位置预测比
GOKU/V2P 稳约 **2×**(0.25 m vs 0.45 m),且每一维都物理可解释、仅 **14k 参数**。

---

## 环境(关键)
| | |
|---|---|
| **GPU 任务** | `C:\Users\suian\AppData\Local\Programs\Python\Python311\python.exe`(CUDA torch / RTX 4080) |
| 默认 `python`(3.13) | **CPU-only** torch,只用于不吃 GPU 的小活 |
| 注意 | 小模型逐步 rollout 是延迟受限,**先批量化(堆 batch)GPU 才有意义** |

一键出全部论文图:`<py311> src/paper_figures.py` → `figures/fig_{main,noise,stability}.{pdf,png}`(论文用 `.pdf`)。
**所有命令从 `piwm/` 运行**(checkpoint/figures 是 cwd 相对;源码在 `src/`)。

---

## 复现流程(数据 → 训练 → 图)
```
1. 数据预处理(原始 npz → 训练用)
     python src/donkey_prep.py        # → ../Data_Donkeycar_prep/    (31-dim 基线用)
     python src/frenet_prep.py        # → ../Data_Donkeycar_frenet/  (Frenet 5-dim 状态 + κ 表)

2. 训练
     <py311> src/train/train_frenet.py             # Frenet 动力学 (我们的)
     <py311> src/train/train_baselines_donkey.py   # GOKU/V2P/DVBF/SINDYc 基线 (K=8)
     # K=32 长horizon基线已有 checkpoint(脚本在 _archive/train/train_all_longK_donkey.py)

3. 评估 + 出图
     # SINDYc 涉及 pysindy,在 CPU(3.13)跑;其余用 py311 GPU
     python src/eval_stability_log.py              # log稳定性(含SINDYc)+ 缓存 _sindyc_curve.npy
     <py311> src/paper_figures.py                  # 三张论文图(读 SINDYc 缓存,不碰 pysindy)
     <py311> src/eval_frenet_vs_baselines.py       # 100步位置误差对比表
     <py311> src/noise_robustness.py               # 噪声鲁棒性(跨模型 + 逐物理量)
     <py311> src/eval_frenet_pipeline.py           # 完整前视流程 + 图像 MSE + filmstrip
```

---

## 目录
```
piwm/                      ← 仓库根 + 运行目录(从这里跑所有命令)
├── src/                   ← 全部源码
│   ├── 核心共享    config / lane_utils / relative_coords / utils / track_utils .py
│   ├── DonkeyCar   donkey_{config,prep,dataset,track}.py     数据/坐标/glitch过滤
│   ├── Frenet(贡献) frenet_{track,prep,bridge}.py           κ表 / 状态预处理 / ↔29维桥
│   ├── 评估+图     eval_frenet_{vs_baselines,pipeline}.py  eval_stability_log.py
│   │               noise_robustness.py  paper_figures.py
│   ├── models/     frenet_dynamics(我们的) · encoder_lane · decoder_lane
│   │               dynamics_lane_v6_kin · dynamics_bicycle_kin (训encoder时用)
│   │               dynamics_lane_v5 · dynamics_bicycle_v4 (SeqLaneDataset 传递依赖)
│   ├── baselines/  shared_dynamics_lane.py = GOKU / V2P / DVBF
│   └── train/      train_frenet · train_baselines_donkey · train_piwm_lane_v6_donkey
│                   train_piwm_lane_v5 (含共享 SeqLaneDataset)
├── checkpoints/    权重(gitignored):frenet/ · {goku,v2p,dvbf}_lane_donkey_longK/ · ...
├── figures/        当前论文图 fig_{main,noise,stability}.{pdf,png}
├── papers/         会议 PDF + 期刊 zip + report_*.pdf
├── Jounral_PIWM/   期刊 LaTeX 源
└── _archive/       归档(可逆,不进活代码):
                    carracing_conf/  CarRacing 会议全套(models/train/eval/viz + 旧README)
                    models/ train/ scripts/  早期 donkey 失败迭代
                    logs/ vis/        训练日志 + 旧输出图
仓库根 (../):
    Data_Donkeycar/        原始真车数据(只读)
    Data_Donkeycar_prep/   31-dim 预处理(可重生成)
    Data_Donkeycar_frenet/ Frenet 状态(可重生成)
```

---

## 关键文件速查
| 文件 | 作用 |
|---|---|
| `frenet_track.py` | 把无序赛道点云排序+重采样→均匀弧长网格,提供 κ(s)、xy↔(s,d) |
| `frenet_prep.py` | 原始 npz → Frenet 状态序列 `[s,d,ψₑ,v,ω]` + 前方曲率剖面 |
| `models/frenet_dynamics.py` | **核心**:解析 Frenet 传播(s,d,ψₑ)+ 小 MLP 学(v̇,ω̇),可微 κ 查表 |
| `frenet_bridge.py` | Frenet 5维 ↔ 旧 29维观测,让现有 encoder/decoder 直接复用 |
| `baselines/shared_dynamics_lane.py` | 三个基线动力学:GOKU(已知运动学+MLP)/V2P(θ系统辨识)/DVBF(局部线性) |
| `paper_figures.py` | 一键生成三张出版级图(统一样式 + PDF) |
| `eval_frenet_vs_baselines.py` | 黄金标准对比:同窗口、GT 初始化、100 步、真实米;含 `build_state31`/`sd2xy` 复用 |

---

## 结果一览(DonkeyCar val,100 步 rollout,真实米,越低越好)
| 模型 | 位置误差@100 | 参数 |
|---|---|---|
| **Frenet(ours)** | **0.25 m** | **14k** |
| V2P | 0.44 m | 58k |
| GOKU | 0.47 m | 37k |
| DVBF | 0.80 m | 10k |
| SINDYc | 发散(~1e17) | — |

短 horizon(≤50 步)基线略优,**交叉点 ~65 步后 Frenet 大幅领先**(有界 vs 超线性发散)。

---

## 命名约定(教训)
迭代用**语义名**(`_damped`、`_frenet`),**别用纯数字后缀** `_v2/v3/v4` —— 会和 PIWM 版本号(v5/v6)撞车,极易误读为"版本退回"。详见 `../CLAUDE.md`。
