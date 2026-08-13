# 主表 3 seeds (mean +/- range)

```
n windows = 201   (val split seed=0, stride 4, 100-step rollout, GT init)

row                 xy@25    xy@50    xy@100
--------------------------------------------
DVBF_s0            0.070m   0.145m    0.696m
DVBF_s1            0.072m   0.172m    0.681m
DVBF_s2            0.072m   0.171m    1.621m
GOKU_s0            0.063m   0.134m    0.598m
GOKU_s1            0.063m   0.123m    0.736m
GOKU_s2            0.062m   0.135m    0.560m
V2P_s0             0.036m   0.087m    0.536m
V2P_s1             0.033m   0.073m    0.507m
V2P_s2             0.039m   0.104m    0.489m
ours-a_s0          0.116m   0.212m    0.383m
ours-a_s1          0.121m   0.255m    0.506m
ours-a_s2          0.126m   0.235m    0.346m
ours-c_s0          0.114m   0.218m    0.403m
ours-c_s1          0.124m   0.257m    0.500m
ours-c_s2          0.138m   0.250m    0.363m
ours-c*_s0         0.118m   0.228m    0.407m
ours-c*_s1         0.141m   0.282m    0.529m
ours-c*_s2         0.132m   0.251m    0.371m
paper(dyn_k16)     0.090m   0.158m    0.267m

mean +/- half-range over 3 seeds (E_xy, m)
model                         @25              @50             @100
--------------------------------------------------------------------
DVBF                0.071+/-0.001    0.163+/-0.013    0.999+/-0.470
GokuNet             0.062+/-0.000    0.131+/-0.006    0.631+/-0.088
Vid2Param           0.036+/-0.003    0.088+/-0.015    0.510+/-0.024
ours (map)          0.121+/-0.005    0.234+/-0.022    0.412+/-0.080
ours (map-free)     0.125+/-0.012    0.242+/-0.020    0.422+/-0.069
ours (mf, k*)       0.130+/-0.012    0.254+/-0.027    0.436+/-0.079
paper row                  0.090           0.158           0.267

paired gap on xy@100, ours-(a) minus baseline, per seed (negative = ours better; 95% CI, 10k-resample bootstrap over windows)
------------------------------------------------------------------------------
  ours-a_s0 - V2P_s0   = -0.153m  [-0.201,-0.102]
  ours-a_s1 - V2P_s1   = -0.000m  [-0.065,+0.065]
  ours-a_s2 - V2P_s2   = -0.142m  [-0.203,-0.082]
  ours-a_s0 - GOKU_s0   = -0.215m  [-0.272,-0.159]
  ours-a_s1 - GOKU_s1   = -0.229m  [-0.307,-0.154]
  ours-a_s2 - GOKU_s2   = -0.213m  [-0.272,-0.153]
  ours-a_s0 - DVBF_s0   = -0.313m  [-0.378,-0.245]
  ours-a_s1 - DVBF_s1   = -0.174m  [-0.272,-0.082]
  ours-a_s2 - DVBF_s2   = -1.275m  [-1.554,-1.015]

paired (a) vs (c): what dropping the known centreline costs
  (c) seed 0: +0.021m
  (c) seed 1: -0.006m
  (c) seed 2: +0.016m
  (c*) seed 0: +0.024m
  (c*) seed 1: +0.023m
  (c*) seed 2: +0.025m

self-check vs reports/repeats/ac_paired_3seeds.md (ours-a @100)
  seed 0: got 0.383  expected 0.383  OK
  seed 1: got 0.506  expected 0.506  OK
  seed 2: got 0.346  expected 0.346  OK
  -> evaluator matches the 08-08 harness
```
