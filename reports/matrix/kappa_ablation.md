# kappa-source ablation, 5-fold

```
fold 0: 647 windows, 12 rows (192s)
fold 1: 301 windows, 12 rows (283s)
fold 2: 474 windows, 12 rows (442s)
fold 3: 554 windows, 12 rows (622s)
fold 4: 507 windows, 12 rows (793s)

==============================================================================
KAPPA SOURCE ABLATION  E_xy (m), mean +/- std over 5 folds
==============================================================================
curvature source                      RMSE_k           @25           @50          @100
------------------------------------------------------------------------------
map lookup (privileged)                   -- 0.144+/-0.011 0.266+/-0.022 0.505+/-0.043
true preview (perfect perception)         -- 0.141+/-0.009 0.245+/-0.022 0.466+/-0.046
camera, scratch                       0.1535 0.141+/-0.009 0.245+/-0.021 0.463+/-0.045
camera, v6 warm-start                 0.1475 0.141+/-0.009 0.246+/-0.019 0.462+/-0.052
camera, v6 frozen                     0.2469 0.141+/-0.010 0.243+/-0.021 0.461+/-0.053
camera, VQ+Transformer                0.2191 0.142+/-0.009 0.246+/-0.019 0.463+/-0.051
camera, extrinsic (conf.)             0.1934 0.140+/-0.008 0.240+/-0.017 0.454+/-0.048
camera, intrinsic (lam 1e3)           0.1391 0.140+/-0.009 0.244+/-0.022 0.464+/-0.048
camera, intrinsic (lam 1e4)           0.1358 0.141+/-0.010 0.242+/-0.020 0.460+/-0.045
camera, intrinsic (lam 1e5)           0.1346 0.140+/-0.009 0.243+/-0.021 0.463+/-0.047
camera, LSTM                          0.8979 0.175+/-0.014 0.270+/-0.006 0.456+/-0.089
camera, Transformer                   0.1223 0.141+/-0.010 0.248+/-0.023 0.471+/-0.047

RMSE_k is the encoder's curvature error in 1/m, averaged over the preview offsets;
the privileged rows have none because they do not estimate curvature.

PAIRED within fold, @100, relative to the scratch camera encoder:
  map lookup (privileged)            +0.042m   per fold [+0.010, +0.038, +0.032, +0.090, +0.038]
  true preview (perfect perception)  +0.003m   per fold [+0.005, +0.003, +0.008, +0.001, -0.000]
  camera, v6 warm-start              -0.001m   per fold [-0.008, -0.008, +0.015, -0.003, -0.003]
  camera, v6 frozen                  -0.002m   per fold [-0.010, -0.015, +0.021, +0.006, -0.010]
  camera, VQ+Transformer             -0.000m   per fold [-0.009, +0.002, +0.014, -0.006, -0.002]
  camera, extrinsic (conf.)          -0.009m   per fold [-0.007, +0.005, +0.008, -0.023, -0.028]
  camera, intrinsic (lam 1e3)        +0.001m   per fold [+0.016, -0.001, +0.005, -0.012, -0.005]
  camera, intrinsic (lam 1e4)        -0.003m   per fold [+0.007, -0.001, -0.007, -0.010, -0.003]
  camera, intrinsic (lam 1e5)        -0.000m   per fold [+0.010, -0.002, +0.001, -0.008, -0.002]
  camera, LSTM                       -0.008m   per fold [+0.032, +0.128, -0.002, -0.094, -0.101]
  camera, Transformer                +0.008m   per fold [+0.006, -0.001, +0.014, +0.016, +0.005]

CAVEATS. The v6 rows warm-start from an encoder trained on the legacy split, which
saw episodes these folds hold out: that leak favours them. The VQ+Transformer row is
the conference version's strongest configuration, at matched capacity and 4x the epochs.
```
