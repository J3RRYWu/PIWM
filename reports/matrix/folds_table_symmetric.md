# 5-fold CV tables (symmetric checkpoint selection)

```
SYMMETRIC SELECTION: baselines take the epoch chosen by E_xy@100 (reports\matrix\selected_checkpoints.json, 45 runs). Ours already selects this way; both sides are therefore selected on the evaluation windows.
fold 0: fold 0/5: 57 train / 14 val eps, val=[4, 5, 8, 16, 19, 30, 35, 36, 50, 53, 64, 66, 69, 70]
  [sel] dvbf_lane_donkey_f0 -> best.tar
  [sel] goku_lane_donkey_f0 -> best.tar
  [sel] v2p_lane_donkey_f0 -> best.tar
  [sel] dvbf_lane_donkey_f0_d5 -> ep15.tar
  [sel] goku_lane_donkey_f0_d5 -> ep15.tar
  [sel] v2p_lane_donkey_f0_d5 -> ep15.tar
  [sel] dvbf_lane_donkey_f0_d10 -> ep10.tar
  [sel] goku_lane_donkey_f0_d10 -> ep10.tar
  [sel] v2p_lane_donkey_f0_d10 -> ep40.tar
  17 rows x 647 windows
fold 1: fold 1/5: 57 train / 14 val eps, val=[2, 10, 11, 17, 20, 23, 27, 28, 43, 49, 52, 55, 62, 65]
  [sel] dvbf_lane_donkey_f1 -> ep10.tar
  [sel] goku_lane_donkey_f1 -> ep25.tar
  [sel] v2p_lane_donkey_f1 -> ep30.tar
  [sel] dvbf_lane_donkey_f1_d5 -> ep10.tar
  [sel] goku_lane_donkey_f1_d5 -> ep20.tar
  [sel] v2p_lane_donkey_f1_d5 -> ep15.tar
  [sel] dvbf_lane_donkey_f1_d10 -> ep15.tar
  [sel] goku_lane_donkey_f1_d10 -> ep5.tar
  [sel] v2p_lane_donkey_f1_d10 -> ep25.tar
  17 rows x 301 windows
fold 2: fold 2/5: 56 train / 15 val eps, val=[0, 1, 3, 6, 18, 21, 22, 24, 26, 34, 37, 42, 44, 67, 68]
  [sel] dvbf_lane_donkey_f2 -> ep25.tar
  [sel] goku_lane_donkey_f2 -> ep50.tar
  [sel] v2p_lane_donkey_f2 -> ep35.tar
  [sel] dvbf_lane_donkey_f2_d5 -> ep60.tar
  [sel] goku_lane_donkey_f2_d5 -> ep5.tar
  [sel] v2p_lane_donkey_f2_d5 -> ep15.tar
  [sel] dvbf_lane_donkey_f2_d10 -> ep10.tar
  [sel] goku_lane_donkey_f2_d10 -> ep10.tar
  [sel] v2p_lane_donkey_f2_d10 -> ep15.tar
  17 rows x 474 windows
fold 3: fold 3/5: 57 train / 14 val eps, val=[9, 14, 15, 25, 31, 32, 38, 46, 47, 51, 57, 58, 60, 61]
  [sel] dvbf_lane_donkey_f3 -> ep25.tar
  [sel] goku_lane_donkey_f3 -> ep25.tar
  [sel] v2p_lane_donkey_f3 -> ep50.tar
  [sel] dvbf_lane_donkey_f3_d5 -> ep15.tar
  [sel] goku_lane_donkey_f3_d5 -> ep30.tar
  [sel] v2p_lane_donkey_f3_d5 -> ep20.tar
  [sel] dvbf_lane_donkey_f3_d10 -> ep10.tar
  [sel] goku_lane_donkey_f3_d10 -> ep10.tar
  [sel] v2p_lane_donkey_f3_d10 -> ep5.tar
  17 rows x 554 windows
fold 4: fold 4/5: 57 train / 14 val eps, val=[7, 12, 13, 29, 33, 39, 40, 41, 45, 48, 54, 56, 59, 63]
  [sel] dvbf_lane_donkey_f4 -> ep50.tar
  [sel] goku_lane_donkey_f4 -> ep45.tar
  [sel] v2p_lane_donkey_f4 -> ep45.tar
  [sel] dvbf_lane_donkey_f4_d5 -> ep20.tar
  [sel] goku_lane_donkey_f4_d5 -> ep10.tar
  [sel] v2p_lane_donkey_f4_d5 -> ep40.tar
  [sel] dvbf_lane_donkey_f4_d10 -> ep25.tar
  [sel] goku_lane_donkey_f4_d10 -> ep30.tar
  [sel] v2p_lane_donkey_f4_d10 -> ep5.tar
  17 rows x 507 windows

folds with results: [0, 1, 2, 3, 4]

==========================================================================
MAIN TABLE  E_xy (m), mean +/- std over 5 folds
==========================================================================
model                           @25              @50             @100   k
--------------------------------------------------------------------------
DVBF                  0.061+/-0.011    0.173+/-0.031    0.837+/-0.131   5
GokuNet               0.046+/-0.012    0.131+/-0.013    0.725+/-0.107   5
GokuNet (obs)         0.035+/-0.006    0.103+/-0.006    0.720+/-0.141   5
Vid2Param             0.036+/-0.008    0.107+/-0.008    0.703+/-0.142   5
ours (map)            0.141+/-0.009    0.245+/-0.021    0.463+/-0.045   5
ours (map-free)       0.146+/-0.012    0.258+/-0.023    0.516+/-0.056   5

==========================================================================
DELTA WEAK-SUPERVISION NOISE  E_xy@100 (m), mean +/- std over folds
(the single-split table is non-monotonic -- 5% worse than 10% -- and the
 paper attributes that to single-run variance; these error bars test it)
==========================================================================
model                          d=0%             d=5%            d=10%
--------------------------------------------------------------------------
ours (map)            0.463+/-0.045    0.452+/-0.035    0.319+/-0.044
ours (map-free)       0.516+/-0.056    0.506+/-0.049    0.390+/-0.036
Vid2Param             0.703+/-0.142    0.919+/-0.189    0.938+/-0.211
GokuNet               0.725+/-0.107    0.896+/-0.152    0.864+/-0.142
GokuNet (obs)         0.720+/-0.141               --               --
DVBF                  0.837+/-0.131    1.067+/-0.188    1.075+/-0.177

==========================================================================
PAIRED within each fold: ours-(a) minus baseline, E_xy@100, delta=0
(pairing is only valid inside a fold -- different folds are different windows)
==========================================================================
  vs DVBF  mean -0.374m  per fold [-0.199, -0.542, -0.428, -0.353, -0.348]  ours better in 5/5
  vs GOKU  mean -0.262m  per fold [-0.158, -0.195, -0.337, -0.357, -0.262]  ours better in 5/5
  vs V2P   mean -0.240m  per fold [-0.052, -0.224, -0.206, -0.462, -0.256]  ours better in 5/5
  vs goku_obs mean -0.257m  per fold [-0.052, -0.290, -0.212, -0.467, -0.265]  ours better in 5/5

==========================================================================
IS THE DELTA GAIN ABOUT WEAK SUPERVISION, OR JUST NOISE?
same dynamics, same per-dim noise MAGNITUDE, delta=10%:
==========================================================================
  no noise (delta=0)             0.463 +/- 0.045 m   (k=5)
  biased-uniform (conference)    0.319 +/- 0.044 m   (k=5)
  matched Gaussian (control)     0.313 +/- 0.041 m   (k=5)

  paired gauss - biased-uniform: -0.005m  per fold [-0.024, -0.018, -0.003, +0.011, +0.007]
  -> a gain of the same size under both means the effect is ordinary regularisation
     and must be described that way; only a clear advantage for the biased-uniform
     form makes it a statement about weak supervision.

paired (a) vs (c): cost of dropping the known centreline, delta=0
  mean +0.053m  per fold [+0.049, +0.022, +0.069, +0.071, +0.056]
```
