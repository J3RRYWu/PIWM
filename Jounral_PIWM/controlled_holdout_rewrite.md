> Historical intermediate rewrite. The current main TeX and RAS_revision_audit_2026-09-27.md supersede this partial draft.

# 期刊正文修订草稿：问题定义与实验边界

以下段落可替换当前稿件相应论述。补充实验结果以 `reports/controlled_holdout/RESULTS.md` 为准；此文件先修正不依赖数值结果的科学表述，不承诺待验证的优势。

## 问题定义：替换 Introduction 与 Section 4.1 的核心动机

We study offline, action-conditioned trajectory prediction on a fixed driving track. Given a known initial vehicle state, a short history of forward-facing images, and a future action sequence, the predictor estimates the vehicle's subsequent motion. Road geometry does not, by itself, change planar vehicle motion when the complete vehicle state, actuation sequence, and physical conditions are fixed. It does affect the controller's choice of actions, the interpretation of road-relative coordinates, and the surrounding scene that a visual model observes. Our objective is therefore to investigate whether an explicit, camera-derived representation of this geometry provides a useful inductive bias for finite-horizon prediction.

The proposed representation contains the vehicle's speed and yaw rate, its progress and displacement relative to the road, and a finite curvature and centerline preview inferred from the observation history. This is a structured approximation to the information needed for prediction. We do not claim that a finite image history resolves all partial observability, or that these variables are a sufficient Markov state for arbitrary driving environments.

中文说明：将“道路几何使给定动作的车辆动力学必然非 Markov”改为“显式道路表示是否改善有限时域动作条件预测”的可检验命题。

## 测试信息条件：在摘要／实验设置中明确

The experiments use ground-truth initialization and recorded future actions. A camera encoder supplies the road preview, but the initial lateral and heading errors are obtained from the annotated vehicle pose and track geometry. The evaluation therefore measures prediction with privileged initialization; it does not demonstrate a complete camera-only state-estimation and driving system.

In the controlled map-free comparison, predicted positions are reconstructed from the camera-derived road geometry, without querying the surveyed centerline during rollout. The surveyed track is used to construct training labels and the initial road-relative state. Position error is measured against the original logged pose rather than a position reconstructed from projected Frenet labels.

中文说明：区分训练标签、初始状态特权、预测阶段地图查询、评价真值四种不同用途。

## 道路预览与稳定性：替换“不会累积／保证有界”的表述

The road preview remains fixed during rollout and is queried at the model's predicted progress. This removes a recurrent update for road geometry, but it does not eliminate error propagation. Errors in progress can shift the curvature query, and errors in actuation, heading, and road geometry can still influence future states. The denominator of the Frenet coordinate transformation also restricts the useful operating region. We therefore assess stability empirically over the evaluated horizons and do not assert a general bounded-error guarantee.

Outside the observed preview, the controlled experiment extends the reconstructed road along its endpoint tangent and holds curvature at its endpoint value. A separate diagnostic retains the legacy endpoint-position clamp. These alternatives are reported explicitly because clamping the reconstructed position can affect endpoint error without establishing accurate long-horizon dynamics.

中文说明：不能把曲率不递推，写成位置误差不会累积；不能把端点位置截断，写成模型保证稳定。

## 训练与测试隔离：替换目前五折协议的主张

The additional evaluation uses a fixed, source-held-out protocol. Each of the two original recordings is held out in turn. Within the remaining recording, the final twenty percent of contiguous segments form the validation set, and a temporal gap of at least 115 frames separates training and validation. All learned components and normalization statistics are fitted on training data only. The road encoder is selected using validation supervision loss, and each dynamics model is selected using validation position error at 100 steps. The test sets are evaluated only after the prescribed training runs have completed.

This is a retrospective reanalysis of records already used during method development. It provides separation between fitting, checkpoint selection, and final evaluation for the present runs, but it is not a prospective evaluation on previously unexamined recordings. The two records also share a track, so the experiment does not establish generalization to unseen road layouts. We retain the existing surveyed-track preprocessing, whose orientation was determined from the first recording and whose grid configuration had previously been explored during development. The present isolation therefore applies to learned-parameter fitting and checkpoint selection, rather than to a prospectively untouched end-to-end preprocessing pipeline.

## 匹配对照：用于新增实验段落

We compare Frenet prediction with an action-conditioned kinematic model, a Cartesian residual predictor with the same frozen visual road preview, and a Cartesian residual predictor without road inputs. The two Cartesian models have the same architecture, with road inputs zeroed in the no-road variant. The trainable predictors use the same rollout windows, physical-output loss, training horizon, epoch budget, and validation selection metric. A constant-motion predictor is included as an action-independent diagnostic. These are controlled implementations designed to isolate modeling choices, rather than claimed reproductions of DVBF, GOKU-net, or Vid2Param.

The preview interventions distinguish two information pathways. Joint interventions replace both curvature and centerline shape with a straight road, the training-set mean, or a shuffled preview. Curvature-only interventions preserve the predicted shape while changing curvature. Their interpretation is limited to sensitivity under inference-time interventions; the independently trained no-road model provides the complementary comparison without road information during training.

## 弱监督贡献边界：现有实验不足时采用

The main road encoder is trained with exact geometric labels. The previously reported noisy-supervision experiment perturbs the dynamics state labels and initialization; it does not perturb the image-to-curvature training pairs. Accordingly, that experiment evaluates robustness to state supervision noise, rather than weakly supervised learning of the visual road representation. Establishing the latter requires an additional experiment that corrupts the curvature and road-shape labels during encoder training.

## 应删除或收缩的具体说法

- 删除“相同车辆状态和相同控制动作，因车道曲率不同必然产生不同世界坐标轨迹”。
- 删除“固定预览使误差不会累积”和无条件“rollout bounded”。
- 不再用地图辅助的 0.463 m 代表完整无地图、无特权初始化的部署性能。
- 不将 CarRacing 记作已经完成当前道路模型定量验证的新增场景。
- 不再将删减的基线直接称为原论文完整方法。
- 将当前噪声实验称为状态监督扰动；若标题保留弱监督道路感知，必须另补相应训练实验。
- 统一正文算法与实际使用的 CNN、曲率／形状头、动态模型和位置恢复过程。

后续应根据完整测试结果撰写 Results 与 Conclusion；不应先写“领先”“稳定”或“道路不可缺少”，再从数据中挑选支持它们的指标。

## 完成测试后的 Results 草稿

Under the source-held-out protocol, the Frenet predictor achieved mean endpoint errors at 100 steps of 1.086 ± 0.150 m and 1.463 ± 0.037 m on the two held-out recordings, respectively. The corresponding errors were 0.936 ± 0.005 m and 1.158 ± 0.006 m for the calibrated kinematic control, 1.230 ± 0.128 m and 1.466 ± 0.156 m for the Cartesian predictor without road information, and 1.344 ± 0.223 m and 1.653 ± 0.035 m for the Cartesian predictor with the same road preview. Uncertainty values denote sample standard deviations over three training seeds, not confidence intervals over independent recording sessions. These results show lower mean endpoint errors for the Frenet model than for the road-conditioned Cartesian control under the prescribed budget, but do not establish an advantage over the calibrated kinematic control.

Replacing both curvature and road shape with a straight-road preview increased the Frenet endpoint error to 2.129 ± 0.546 m and 2.415 ± 0.071 m. However, setting curvature alone to zero while retaining predicted shape gave 0.994 ± 0.126 m and 1.517 ± 0.069 m. Replacing curvature alone with its training-set mean gave 1.156 ± 0.187 m and 1.411 ± 0.037 m. Thus the joint ablation cannot be attributed solely to the curvature-dependent dynamics, and the benefit of sample-specific curvature is not consistent across recordings. These inference-time interventions assess model sensitivity and do not replace comparisons with separately trained ablated models.

Two of the three Frenet training runs in the first held-out direction deteriorated substantially after their validation-optimal epochs. All test results use checkpoints selected solely by validation error. Finite test predictions from these selected checkpoints should not be interpreted as a guarantee of stable optimization or bounded rollout error.

中文结论：本轮只能支持有限的结构对照结果，不能写全面优于物理基线、曲率贡献已被充分隔离，或稳定性问题已解决。所有数值来自 reports/controlled_holdout/test_results.json；训练恶化来自各模型 frenet_history.json。

## 第二轮追加：数值保护与从零重训消融

Following the initial retrospective analysis, we conducted an exploratory numerical-stability study using the same source partitions and frozen perception models. All dynamics variants were trained from scratch for 40 epochs with the same physical-output loss and validation-based checkpoint selection. The guarded variant replaces the signed denominator protection with a positive floor of 0.2, wraps the road-relative heading, and bounds learned increments using hyperbolic tangent functions. Speed and yaw-rate increment limits are set from the 99.5th percentiles of absolute one-frame changes in the training data, with a minimum of 0.01 in the corresponding units. Lateral and heading residuals are bounded at 0.1 m and 0.1 rad per step, respectively. These are numerical safeguards, including an extension outside the valid positive Frenet coordinate chart, rather than guarantees of physically valid or bounded-error trajectories.

The original implementation was reproduced exactly across all six fold–seed combinations. In two runs, its final validation endpoint error rose to 99.875 m and 21.681 m, accompanied by large pre-clipping gradients and frequent non-positive values of the raw Frenet denominator. In the corresponding guarded runs, final validation errors were 0.557 m and 0.543 m. However, the guarded model still occasionally left the valid coordinate chart. Because denominator handling, angular representation, and residual bounds were changed together, this experiment does not isolate the contribution of each safeguard.

The guarded full-preview model obtained retrospective endpoint errors of 1.022 ± 0.154 m and 1.432 ± 0.019 m on the two previously examined source recordings. A separately trained variant that retained the same predicted road shape but set curvature to zero throughout training and evaluation obtained 0.846 ± 0.048 m and 1.471 ± 0.062 m. Thus, numerical stabilization did not establish a consistent benefit from curvature-dependent dynamics. The corresponding no-road variant obtained 1.029 ± 0.078 m and 1.579 ± 0.161 m, but this comparison also removes the initial road-relative displacement and heading and therefore cannot isolate shape alone.

A privileged reference trained and evaluated using ground-truth curvature and shape previews obtained 0.933 ± 0.068 m and 1.285 ± 0.039 m. The calibrated kinematic reference obtained 0.936 ± 0.005 m and 1.158 ± 0.006 m. These comparisons suggest that the performance gap cannot be attributed entirely to road-perception error. They do not establish a theoretical performance bound or a single dominant source of error. All uncertainty values denote sample standard deviations over training seeds. Because these recordings had already informed prior analysis, the follow-up is exploratory and requires confirmation on newly collected data.

中文使用说明：建议作为独立的 exploratory stability and ablation study，不能替换成预先未接触的新测试证据。“仅形状”只隔离动力学曲率通路，CNN 仍受曲率与形状联合监督。以上数据见 reports/frenet_diagnostic/results.json 和对应训练历史。

---
模型名称：GPT-6（Codex）；当前会话未提供更细的型号标识。
