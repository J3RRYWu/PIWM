# 5-fold CV tables

```
fold 0: fold 0/5: 57 train / 14 val eps, val=[4, 5, 8, 16, 19, 30, 35, 36, 50, 53, 64, 66, 69, 70]
  16 rows x 647 windows
fold 1: fold 1/5: 57 train / 14 val eps, val=[2, 10, 11, 17, 20, 23, 27, 28, 43, 49, 52, 55, 62, 65]
  16 rows x 301 windows
fold 2: fold 2/5: 56 train / 15 val eps, val=[0, 1, 3, 6, 18, 21, 22, 24, 26, 34, 37, 42, 44, 67, 68]
  16 rows x 474 windows
fold 3: fold 3/5: 57 train / 14 val eps, val=[9, 14, 15, 25, 31, 32, 38, 46, 47, 51, 57, 58, 60, 61]
  16 rows x 554 windows
fold 4: fold 4/5: 57 train / 14 val eps, val=[7, 12, 13, 29, 33, 39, 40, 41, 45, 48, 54, 56, 59, 63]
  16 rows x 507 windows

folds with results: [0, 1, 2, 3, 4]

==========================================================================
MAIN TABLE  E_xy (m), mean +/- half-range over 5 folds
==========================================================================
model                           @25              @50             @100   k
--------------------------------------------------------------------------
DVBF                  0.055+/-0.012    0.155+/-0.019    1.287+/-0.850   5
GokuNet               0.044+/-0.014    0.125+/-0.015    0.815+/-0.153   5
GokuNet (obs)         0.035+/-0.007    0.103+/-0.007    0.721+/-0.194   5
Vid2Param             0.035+/-0.007    0.103+/-0.007    0.721+/-0.195   5
ours (map)            0.141+/-0.011    0.245+/-0.025    0.463+/-0.046   5
ours (map-free)       0.146+/-0.013    0.258+/-0.031    0.516+/-0.068   5

==========================================================================
DELTA WEAK-SUPERVISION NOISE  E_xy@100 (m), mean +/- half-range over folds
(the single-split table is non-monotonic -- 5% worse than 10% -- and the
 paper attributes that to single-run variance; these error bars test it)
==========================================================================
model                          d=0%             d=5%            d=10%
--------------------------------------------------------------------------
ours (map)            0.463+/-0.046    0.452+/-0.047    0.319+/-0.049
ours (map-free)       0.516+/-0.068    0.506+/-0.066    0.390+/-0.040
Vid2Param             0.721+/-0.195    0.993+/-0.326    1.015+/-0.343
GokuNet               0.815+/-0.153    0.984+/-0.221    0.957+/-0.157
GokuNet (obs)         0.721+/-0.194               --               --
DVBF                  1.287+/-0.850    1.229+/-0.292    1.315+/-0.285

==========================================================================
PAIRED within each fold: ours-(a) minus baseline, E_xy@100, delta=0
(pairing is only valid inside a fold -- different folds are different windows)
==========================================================================
  vs DVBF  mean -0.824m  per fold [-0.199, -0.648, -0.951, -1.925, -0.396]  ours better in 5/5
  vs GOKU  mean -0.352m  per fold [-0.158, -0.428, -0.389, -0.490, -0.298]  ours better in 5/5
  vs V2P   mean -0.257m  per fold [-0.053, -0.289, -0.212, -0.468, -0.265]  ours better in 5/5
  vs goku_obs mean -0.258m  per fold [-0.053, -0.291, -0.213, -0.467, -0.264]  ours better in 5/5

paired (a) vs (c): cost of dropping the known centreline, delta=0
  mean +0.053m  per fold [+0.049, +0.022, +0.069, +0.071, +0.056]
```
