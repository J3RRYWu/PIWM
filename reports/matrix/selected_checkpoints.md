# post-hoc checkpoint selection

```
run                         val_loss pick   metric pick  epoch     gain
------------------------------------------------------------------------
dvbf_lane_donkey_f0                0.647m        0.647m   best  +0.000m
dvbf_lane_donkey_f0_d5             0.825m        0.791m     15  -0.034m
dvbf_lane_donkey_f0_d10            0.874m        0.780m     10  -0.094m
goku_lane_donkey_f0                0.605m        0.605m   best  +0.000m
goku_lane_donkey_f0_d5             0.660m        0.635m     15  -0.026m
goku_lane_donkey_f0_d10            0.789m        0.628m     10  -0.161m
v2p_lane_donkey_f0                 0.497m        0.497m   best  +0.000m
v2p_lane_donkey_f0_d5              0.657m        0.634m     15  -0.022m
v2p_lane_donkey_f0_d10             0.630m        0.607m     40  -0.023m
dvbf_lane_donkey_f1                1.069m        0.960m     10  -0.108m
dvbf_lane_donkey_f1_d5             1.219m        0.964m     10  -0.255m
dvbf_lane_donkey_f1_d10            1.373m        1.064m     15  -0.310m
goku_lane_donkey_f1                0.866m        0.632m     25  -0.234m
goku_lane_donkey_f1_d5             1.079m        0.906m     20  -0.173m
goku_lane_donkey_f1_d10            0.976m        0.884m      5  -0.092m
v2p_lane_donkey_f1                 0.708m        0.636m     30  -0.072m
v2p_lane_donkey_f1_d5              0.926m        0.884m     15  -0.042m
v2p_lane_donkey_f1_d10             0.909m        0.886m     25  -0.023m
dvbf_lane_donkey_f2                1.476m        0.952m     25  -0.524m
dvbf_lane_donkey_f2_d5             1.292m        1.278m     60  -0.014m
dvbf_lane_donkey_f2_d10            1.414m        1.097m     10  -0.317m
goku_lane_donkey_f2                0.896m        0.845m     50  -0.050m
goku_lane_donkey_f2_d5             1.104m        0.910m      5  -0.195m
goku_lane_donkey_f2_d10            0.913m        0.846m     10  -0.066m
v2p_lane_donkey_f2                 0.727m        0.719m     35  -0.007m
v2p_lane_donkey_f2_d5              0.973m        0.896m     15  -0.077m
v2p_lane_donkey_f2_d10             0.993m        0.953m     15  -0.039m
dvbf_lane_donkey_f3                2.307m        0.775m     25  -1.531m
dvbf_lane_donkey_f3_d5             1.398m        1.151m     15  -0.248m
dvbf_lane_donkey_f3_d10            1.455m        1.157m     10  -0.298m
goku_lane_donkey_f3                0.906m        0.774m     25  -0.131m
goku_lane_donkey_f3_d5             1.074m        1.029m     30  -0.045m
goku_lane_donkey_f3_d10            1.104m        0.999m     10  -0.105m
v2p_lane_donkey_f3                 0.881m        0.874m     50  -0.006m
v2p_lane_donkey_f3_d5              1.300m        1.133m     20  -0.168m
v2p_lane_donkey_f3_d10             1.308m        1.172m      5  -0.135m
dvbf_lane_donkey_f4                0.908m        0.865m     50  -0.043m
dvbf_lane_donkey_f4_d5             1.397m        1.159m     20  -0.238m
dvbf_lane_donkey_f4_d10            1.374m        1.241m     25  -0.133m
goku_lane_donkey_f4                0.802m        0.763m     45  -0.039m
goku_lane_donkey_f4_d5             0.970m        0.963m     10  -0.007m
goku_lane_donkey_f4_d10            0.996m        0.936m     30  -0.060m
v2p_lane_donkey_f4                 0.777m        0.767m     45  -0.010m
v2p_lane_donkey_f4_d5              1.081m        1.026m     40  -0.055m
v2p_lane_donkey_f4_d10             1.204m        1.055m      5  -0.149m

========================================================================
effect on the table: mean E_xy@100 over folds, by selection rule
========================================================================
model          val_loss (current)    reported metric     shift
------------------------------------------------------------------------
dvbf                       1.269m             0.992m   -0.277m
goku                       0.916m             0.824m   -0.092m
v2p                        0.905m             0.849m   -0.055m

(selection stride 8, so these are not the table's numbers -- rerun eval_folds_table.py on the chosen files for those)
SYMMETRY, NOT CLEANLINESS: ours already picks its epoch this way. Applying it to the baselines removes the asymmetry but leaves BOTH sides selected on the evaluation windows, which the paper must state.
```
