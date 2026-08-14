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

folds with results: [0]

==========================================================================
MAIN TABLE  E_xy (m), mean +/- std over 1 folds
==========================================================================
model                           @25              @50             @100   k
--------------------------------------------------------------------------
DVBF                  0.070+/-0.000    0.159+/-0.000    0.646+/-0.000   1
GokuNet               0.063+/-0.000    0.144+/-0.000    0.606+/-0.000   1
GokuNet (obs)         0.040+/-0.000    0.095+/-0.000    0.500+/-0.000   1
Vid2Param             0.040+/-0.000    0.095+/-0.000    0.500+/-0.000   1
ours (map)            0.147+/-0.000    0.262+/-0.000    0.448+/-0.000   1
ours (map-free)       0.159+/-0.000    0.282+/-0.000    0.497+/-0.000   1

==========================================================================
DELTA WEAK-SUPERVISION NOISE  E_xy@100 (m), mean +/- std over folds
(the single-split table is non-monotonic -- 5% worse than 10% -- and the
 paper attributes that to single-run variance; these error bars test it)
==========================================================================
model                          d=0%             d=5%            d=10%
--------------------------------------------------------------------------
ours (map)            0.448+/-0.000    0.415+/-0.000    0.370+/-0.000
ours (map-free)       0.497+/-0.000    0.447+/-0.000    0.421+/-0.000
Vid2Param             0.500+/-0.000    0.635+/-0.000    0.611+/-0.000
GokuNet               0.606+/-0.000    0.640+/-0.000    0.630+/-0.000
GokuNet (obs)         0.500+/-0.000               --               --
DVBF                  0.646+/-0.000    0.786+/-0.000    0.781+/-0.000

==========================================================================
PAIRED within each fold: ours-(a) minus baseline, E_xy@100, delta=0
(pairing is only valid inside a fold -- different folds are different windows)
==========================================================================
  vs DVBF  mean -0.199m  per fold [-0.199]  ours better in 1/1
  vs GOKU  mean -0.158m  per fold [-0.158]  ours better in 1/1
  vs V2P   mean -0.052m  per fold [-0.052]  ours better in 1/1
  vs goku_obs mean -0.052m  per fold [-0.052]  ours better in 1/1

==========================================================================
IS THE DELTA GAIN ABOUT WEAK SUPERVISION, OR JUST NOISE?
same dynamics, same per-dim noise MAGNITUDE, delta=10%:
==========================================================================
  no noise (delta=0)             0.448 +/- 0.000 m   (k=1)
  biased-uniform (conference)    0.370 +/- 0.000 m   (k=1)
  matched Gaussian (control)     0.347 +/- 0.000 m   (k=1)

  paired gauss - biased-uniform: -0.024m  per fold [-0.024]
  -> a gain of the same size under both means the effect is ordinary regularisation
     and must be described that way; only a clear advantage for the biased-uniform
     form makes it a statement about weak supervision.

paired (a) vs (c): cost of dropping the known centreline, delta=0
  mean +0.049m  per fold [+0.049]
```
