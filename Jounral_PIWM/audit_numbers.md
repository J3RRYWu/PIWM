# Number audit — `elsarticle-template-num.tex` vs. the evaluation outputs

Audited 2026-08-14. Read-only audit; nothing in the .tex was changed.

**Accepted sources of truth (the only ones consulted):**

| tag | file |
|---|---|
| **S1** | `reports/matrix/folds_table_symmetric.md` |
| **S2** | `reports/matrix/kappa_ablation.md` |
| **S3** | `reports/matrix/selected_checkpoints.md` |
| **S4** | `reports/matrix/folds_table_leak_audit.md` |
| **S5** | `reports/repeats/main_table_3seeds.md` |

**Verdicts used:** MATCHES · MISMATCH · NOT FOUND (no line in S1–S5 produces it) ·
UNVERIFIABLE HERE (CartPole / Lunar Lander / CarRacing — out of scope by instruction).

---

## 0. Summary of counts

| verdict | count |
|---|---|
| MATCHES | 89 table cells + 20 prose/caption claims = **109** |
| MISMATCH | 1 table cell + 23 prose/caption claims = **24** |
| NOT FOUND IN ANY SOURCE | **11** |
| UNVERIFIABLE HERE (out of scope) | **~25 values** (Tier-1/Tier-2 + commented-out blocks) |

**The single most important finding:** the .tex contains **two mutually
inconsistent sets of DonkeyCar numbers**. Tables 1, 3 and 4 are reproduced
correctly from S1 and S2 (89/90 cells exact). The **prose, abstract, figure
captions and conclusion are built on a different, older set of numbers**
(0.267 / 0.383 / 0.246 / 0.263 / 0.274 / 0.290 / 0.1193) that the paper's own
tables contradict and that S1–S5 do not produce. Every MISMATCH below is in the
second set.

---

## 1. Abstract, Introduction, Conclusion — the headline pair

### 1.1 `0.267 m` for PIWM at 100 steps — **MISMATCH**
Lines 131, 271, 1732, 1757, 1983, 2091.

Only appearance of `0.267` in any source, S5:
```
paper(dyn_k16)     0.090m   0.158m    0.267m
```
```
paper row                  0.090           0.158           0.267
```
This row is labelled *"paper"* — it records what the manuscript asserts, not a
measurement of the model the tables report. The measured value for the same model
and metric in **both** sources is different:

- S1 (5-fold, the source of the paper's own Table 1): `ours (map)  0.141+/-0.011  0.245+/-0.025  0.463+/-0.046`
- S5 (3-seed, single split): `ours (map)          0.121+/-0.005    0.234+/-0.022    0.412+/-0.080`

The paper's own Table 1 (line 1687) prints **0.463 ± 0.046**. The abstract,
introduction and conclusion print **0.267**. Verdict: MISMATCH.

### 1.2 `0.383 m` for "the strongest baseline" — **MISMATCH (mislabelled model)**
Lines 131–132, 272, 1875 (Fig. 5 caption), 2091.

Only appearance of `0.383` in any source, S5:
```
ours-a_s0          0.116m   0.212m    0.383m
```
```
  seed 0: got 0.383  expected 0.383  OK
```
`ours-a_s0` is **our own model, seed 0** — not a baseline. The strongest baseline
at 100 steps is:

- S5: `Vid2Param           0.036+/-0.003    0.088+/-0.015    0.510+/-0.024`
- S1: `Vid2Param             0.036+/-0.010    0.107+/-0.009    0.703+/-0.192`

Verdict: MISMATCH. This is the same class of error the CLAUDE.md house rules
already record for the CarRacing table ("PIWM (ours, full)" was actually a
vehicle-only model).

### 1.3 Contribution 3, line 316: "roughly **2 cm** … relative to a privileged map lookup" — **MISMATCH (sign inverted)**
See §3.2 — the source says the camera *beats* the map lookup by 4.2 cm.

### 1.4 Conclusion, line 2094–2095: "a privileged map lookup improves the result by only about **2 cm**" — **MISMATCH (sign inverted)**
Same as §3.2.

---

## 2. Section 7.3, main-table paragraph (lines 1634–1661) and Table 1

### 2.1 Table 1 (`tab:donkey_main`, lines 1682–1688) — **all 30 cells MATCH**
Source S1:
```
model                           @25              @50             @100   k
DVBF                  0.061+/-0.011    0.173+/-0.038    0.837+/-0.160   5
GokuNet               0.046+/-0.015    0.131+/-0.015    0.725+/-0.122   5
Vid2Param             0.036+/-0.010    0.107+/-0.009    0.703+/-0.192   5
ours (map)            0.141+/-0.011    0.245+/-0.025    0.463+/-0.046   5
ours (map-free)       0.146+/-0.013    0.258+/-0.031    0.516+/-0.068   5
```
Every one of the 30 numbers in the LaTeX table body is character-for-character
this table. MATCHES.

### 2.2 `2483` evaluation windows (lines 1652, 1685) — **MATCHES (derived, arithmetic shown)**
S1:
```
  17 rows x 647 windows   (fold 0)
  17 rows x 301 windows   (fold 1)
  17 rows x 474 windows   (fold 2)
  17 rows x 554 windows   (fold 3)
  17 rows x 507 windows   (fold 4)
```
647 + 301 + 474 + 554 + 507 = **2483**. MATCHES.

### 2.3 "SINDYc **diverges** in every one of the 2483 windows" (lines 1652, 1685) — **NOT FOUND**
No SINDYc row exists in S1–S5. S1's main table lists six models and SINDYc is not
among them. The *window count* is verified (§2.2); the *divergence claim* is not.

### 2.4 "Up to roughly **65** steps the baselines are ahead" (line 1639) — **NOT FOUND**
No per-step curve appears in S1–S5 (the curve data lives in
`folds_table_symmetric_curves.npz`, which is not an accepted source).

### 2.5 "Vid2Param … more than twice as accurate at step 25 (**0.036 m** against **0.090 m**)" (line 1641)
- `0.036` — **MATCHES**: S1 `Vid2Param … 0.036+/-0.010`.
- `0.090` — **MISMATCH**: this is ours at step 25. S1 gives `ours (map)  0.141+/-0.011` and the paper's own Table 1 line 1687 prints `0.141 ± 0.011`. `0.090` appears only in S5 as the "paper row": `paper row  0.090  0.158  0.267`.
- Recomputed ratio from the source: 0.141 / 0.036 = **3.92×**, not the 2.5× implied by 0.090/0.036. ("More than twice" survives, but only because it understates.)

### 2.6 "0.463 m against 0.703 m" (line 1647) — **MATCHES**
S1: `ours (map) … 0.463+/-0.046`, `Vid2Param … 0.703+/-0.192`.

### 2.7 "ahead at step 100 in **all five** folds against every baseline" (line 1648) — **MATCHES**
S1:
```
  vs DVBF  mean -0.374m  ... ours better in 5/5
  vs GOKU  mean -0.262m  ... ours better in 5/5
  vs V2P   mean -0.240m  ... ours better in 5/5
  vs goku_obs mean -0.257m ... ours better in 5/5
```

### 2.8 "by **0.241 m** on average against Vid2Param" (line 1650) — **MISMATCH**
S1: `  vs V2P   mean -0.240m  per fold [-0.053, -0.224, -0.206, -0.462, -0.257]`
Recomputed: (0.053 + 0.224 + 0.206 + 0.462 + 0.257) / 5 = 1.202 / 5 = **0.2404 → 0.240 m**.
The .tex says 0.241 m. MISMATCH (1 mm).

### 2.9 "**0.262 m** against GOKU-net" (line 1650) — **MATCHES**  (S1: `vs GOKU  mean -0.262m`)
### 2.10 "**0.374 m** against DVBF" (line 1651) — **MATCHES**  (S1: `vs DVBF  mean -0.374m`)

### 2.11 Spreads ±0.046 / ±0.192 / ±0.122 / ±0.160 (lines 1657–1658) — **all four MATCH** (S1 main table half-ranges)

### 2.12 "a factor of **three to four**" (line 1658) — **MISMATCH (derived)**
Recomputed from the same four numbers:
- 0.192 / 0.046 = **4.17×**
- 0.160 / 0.046 = **3.48×**
- 0.122 / 0.046 = **2.65×**

The actual span is **2.7–4.2×**. GOKU-net is *below* three, so "three to four"
does not follow from the numbers it summarizes.

### 2.13 "on the **201** held-out validation windows" (lines 1421, 1483, 1636; also Fig. 6 caption line 1919) — **MISMATCH**
`201` does exist in the sources —
S5: `n windows = 201   (val split seed=0, stride 4, 100-step rollout, GT init)`
S4: `fold 0: ... 5 rows x 201 windows` (the *clean subset* of fold 0 only)
— but it is the window count of the **single-split** study, not of the five-fold
evaluation that produces Tables 1, 3 and 4. Those use 647/301/474/554/507 =
**2483** windows (§2.2). Line 1421 ("all numbers reported below are computed on
the 201 held-out validation windows") is therefore false for every table in the
paper, and line 1483–1484 defines the metric over the wrong N. MISMATCH.

---

## 3. Section 7.3, "The cost of perceiving the road" (lines 1723–1776) and Table 4

### 3.1 Table 4 (`tab:kappa_source`, lines 1797–1807) — **all 28 cells MATCH**
Source S2:
```
map lookup (privileged)                   -- 0.144+/-0.015 0.266+/-0.025 0.505+/-0.044
true preview (perfect perception)         -- 0.141+/-0.010 0.245+/-0.028 0.466+/-0.049
camera, scratch                       0.1535 0.141+/-0.011 0.245+/-0.025 0.463+/-0.046
camera, v6 warm-start                 0.1475 0.141+/-0.010 0.246+/-0.024 0.462+/-0.056
camera, v6 frozen                     0.2469 0.141+/-0.011 0.243+/-0.027 0.461+/-0.063
camera, VQ+Transformer                0.2191 0.142+/-0.010 0.246+/-0.024 0.463+/-0.056
camera, extrinsic (conf.)             0.1934 0.140+/-0.010 0.240+/-0.023 0.454+/-0.061
camera, intrinsic (lam 1e4)           0.1358 0.141+/-0.011 0.242+/-0.026 0.460+/-0.047
camera, LSTM                          0.8979 0.175+/-0.016 0.270+/-0.008 0.456+/-0.112
camera, Transformer                   0.1223 0.141+/-0.011 0.248+/-0.027 0.471+/-0.052
```
Row-by-row: map 0.505±0.044 ✓ · true κ 0.466±0.049 ✓ · Extrinsic 0.193/0.454±0.061 ✓ ·
LSTM 0.898/0.456±0.112 ✓ · Intrinsic 0.136/0.460±0.047 ✓ (λ=1e4 row) ·
frozen 0.247/0.461±0.063 ✓ · warm-start 0.148/0.462±0.056 ✓ ·
scratch 0.154/0.463±0.046 ✓ · VQ+T 0.219/0.463±0.056 ✓ · Transformer 0.122/0.471±0.052 ✓.
**The table is clean. Everything below it in prose is not.**

### 3.2 The whole "cost of perception" paragraph inverts the source's sign
The .tex prose (lines 1728–1736) claims a *ladder*: map 0.246 → true profile
0.263 → camera 0.267, i.e. privileged information is best and the camera costs
2 cm. **The source says the opposite: the map lookup is the WORST of the three.**

S2, paired within fold, relative to the scratch camera encoder:
```
  map lookup (privileged)            +0.042m   per fold [+0.010, +0.038, +0.032, +0.090, +0.038]
  true preview (perfect perception)  +0.003m   per fold [+0.005, +0.003, +0.008, +0.001, -0.000]
```
Positive = worse than the camera, in all five folds.

| .tex claim | .tex value | source value (S2) | verdict |
|---|---|---|---|
| L1729 map lookup @100 | 0.246 m | **0.505 ± 0.044** | MISMATCH |
| L1731 ground-truth κ profile @100 | 0.263 m | **0.466 ± 0.049** | MISMATCH |
| L1732 encoder actually used @100 | 0.267 m | **0.463 ± 0.046** | MISMATCH |
| L1733, L1999 RMSE_κ of that encoder | 0.1193 m⁻¹ | **0.1535** | MISMATCH (0.1193 appears nowhere in S1–S5) |
| L1743 warm-started backbone | 0.274 m | **0.462 ± 0.056** | MISMATCH |
| L1744, L1757 frozen backbone | 0.290 m | **0.461 ± 0.063** | MISMATCH |
| L316, L1735, L2095 cost of camera vs. map | "about 2 cm" (camera worse) | camera is **0.042 m BETTER**, 5/5 folds | MISMATCH — sign inverted |
| L1983 "improves … from 0.267 m to 0.246 m" | improvement of 0.021 m | map lookup **worsens** by 0.042 m | MISMATCH |

Note also the ordering claim implied by the prose ("progressively better sources
of κ") is contradicted: in S2, the *privileged map* is the worst @100 row in the
entire table, and the .tex's own Table 4 prints that (0.505, the largest value in
the column).

### 3.3 "roughly **8%** of the total error" (line 1735) — **MISMATCH (derived)**
Internally the .tex computes 0.021 / 0.267 = 7.9%. Both inputs are wrong (§3.2).
Recomputed from the source: the difference is −0.042 m and 0.042 / 0.463 =
**9.1%, with the opposite sign** — there is no cost to price.

### 3.4 "beats the strongest baseline by **1.4×**" (line 1736) — **MISMATCH (derived)**
Recomputed from S1: 0.703 / 0.463 = **1.518 → 1.5×**.
(1.4× follows only from the unsourced 0.383 / 0.267 = 1.43.)

### 3.5 "**0.071 m⁻¹** of curvature error and **8 mm** of 100-step position error" (line 936–937) — **both MISMATCH**
VQ+Transformer vs. CNN-from-scratch, from S2:
- curvature: 0.2191 − 0.1535 = **0.0656 → 0.066 m⁻¹**, not 0.071.
  (0.071 follows only from the unsourced 0.190 − 0.119.)
- position: S2 paired, `camera, VQ+Transformer             -0.000m   per fold [-0.009, +0.002, +0.014, -0.006, -0.002]`
  → the VQ+Transformer is **0.000 m, marginally better**, not 8 mm worse.
  The +0.008 m figure belongs to a *different* row: `camera, Transformer  +0.008m`.

### 3.6 "VQ–Transformer reaches RMSE_κ = **0.190 m⁻¹** against **0.119 m⁻¹** for the convolutional encoder" (line 1750–1751) — **MISMATCH**
S2: VQ+Transformer = **0.2191**; camera, scratch = **0.1535**. Neither 0.190 nor
0.119 appears in any source. (The .tex's *own Table 4*, two paragraphs later,
prints 0.219 and 0.154.)

### 3.7 "roughly **270 of 512** entries in active use" (line 1749) — **NOT FOUND**
No codebook-utilization figure appears in S1–S5.

### 3.8 "Curvature error varies by a **factor of two** across the **four** camera-only rows (**0.119–0.237 m⁻¹**)" (lines 1755–1756) — **MISMATCH**
Table 4 has **eight** camera rows, not four. Their RMSE_κ span, from S2, is
**0.1223 to 0.8979** — a factor of **7.34**, not two. Neither 0.119 nor 0.237
appears in any source.

### 3.9 "resulting 100-step position error varies by **2.3 cm** (**0.267–0.290 m**)" (line 1757) — **MISMATCH**
Recomputed over Table 4's camera rows: min 0.454 (extrinsic), max 0.471
(Transformer) → span **0.017 m = 1.7 cm**, over the interval **0.454–0.471 m**.

### 3.10 "the VQ–Transformer encoder is **59%** worse at reading curvature" (line 1758) — **MISMATCH (derived)**
Recomputed from S2: 0.2191 / 0.1535 = **1.427 → 42.7% worse**.
(59–60% follows only from the unsourced 0.190 / 0.119 = 1.597.)

### 3.11 "an estimate **6.6 times** worse" (line 1761) and "varies by a factor of **6.6** across the camera rows" (Table 4 caption, lines 1787–1788) — **MISMATCH (derived)**
Recomputed over the camera rows the caption refers to:
0.8979 / 0.1223 = **7.34×** (from the table's printed values, 0.898 / 0.122 = 7.36×).
6.6 would require using 0.1358 as the minimum (0.8979 / 0.1358 = 6.61), but 0.1358
is not the minimum — the Transformer row at 0.1223 is, and it is printed in the
same table. Also note this contradicts the "factor of two" of §3.8 and the "59%"
of §3.10 three lines apart, so at most one of the three could be right.

### 3.12 "the rollout error varies by **4%**" (Table 4 caption, line 1788) — **MATCHES (derived)**
Recomputed: (0.471 − 0.454) / 0.454 = **3.74% → 4%**. Acceptable rounding.

### 3.13 "less than any single row moves between folds" (caption, line 1788) — **MATCHES**
Span 0.017 m vs. the *smallest* full fold range in the table, 2 × 0.044 = 0.088 m.

### 3.14 "the best curvature estimate gives the worst rollout" (caption, line 1789) — **MATCHES**
S2: `camera, Transformer  0.1223 … 0.471+/-0.052` — lowest RMSE_κ, highest E_xy(100).

### 3.15 "The second is better by **0.039 m**, consistently in all five folds" (lines 1770–1772) — **MATCHES**
Recomputed two ways from S2:
- means: 0.505 − 0.466 = **0.039 m** ✓
- paired: +0.042 (map) − (+0.003) (true preview) = **0.039 m** ✓
- per fold, map − true preview: (0.010−0.005, 0.038−0.003, 0.032−0.008, 0.090−0.001, 0.038−(−0.000)) = **+0.005, +0.035, +0.024, +0.089, +0.038** — all positive, so 5/5 ✓

### 3.16 "The remaining **0.12 m** separating our model from the strongest baseline" (line 1984) — **MISMATCH (derived)**
Recomputed from S1: 0.703 − 0.463 = **0.240 m**; the paired per-fold mean is
`vs V2P   mean -0.240m`. (0.12 m follows only from the unsourced 0.383 − 0.267 = 0.116.)

---

## 4. Section 7.3, "Robustness to weak supervision" (lines 1812–1837) and Table 3

### 4.1 Table 3 (`tab:delta`, lines 1854–1862) — **31 of 32 cells MATCH, 1 MISMATCH**
Source S1:
```
model                          d=0%             d=5%            d=10%
ours (map)            0.463+/-0.046    0.452+/-0.047    0.319+/-0.049
ours (map-free)       0.516+/-0.068    0.506+/-0.066    0.390+/-0.040
Vid2Param             0.703+/-0.192    0.918+/-0.252    0.938+/-0.280
GokuNet               0.725+/-0.122    0.896+/-0.199    0.864+/-0.185
DVBF                  0.837+/-0.160    1.067+/-0.240    1.075+/-0.233
```
```
  matched Gaussian (control)     0.313 +/- 0.043 m   (k=5)
```
**MISMATCH:** line 1857 prints Vid2Param at δ=5% as **`0.919 ± 0.252`**; the
source says **`0.918+/-0.252`**. Every other cell matches.

### 4.2 "from 0.463 m at δ=0 to 0.319 m at δ=10% — a **31%** reduction" (lines 1823–1824) — **MATCHES**
Recomputed: (0.463 − 0.319) / 0.463 = 0.144 / 0.463 = **31.1% → 31%** ✓
"Monotonically": 0.463 → 0.452 → 0.319, monotone decreasing ✓

### 4.3 "while Vid2Param worsens by **33%** over the same range" (line 1825) — **MATCHES**
Recomputed: (0.938 − 0.703) / 0.703 = 0.235 / 0.703 = **33.4% → 33%** ✓

### 4.4 "The trend holds in **all five folds**" (line 1826) — **NOT FOUND**
S1 reports the δ table only as mean ± half-range; there is no per-fold δ
breakdown anywhere in S1–S5.

### 4.5 "and the intervals **do not overlap**" (line 1826) — **MISMATCH**
Recomputed from S1's own half-ranges:
- δ=0: 0.463 ± 0.046 → **[0.417, 0.509]**
- δ=5%: 0.452 ± 0.047 → **[0.405, 0.499]** — overlaps δ=0 almost completely
- δ=10%: 0.319 ± 0.049 → **[0.270, 0.368]** — this one does not overlap δ=0

The claim is true for 0 vs 10% and false for 0 vs 5%, so as stated ("the
intervals", of a three-point monotone trend) it does not follow.

### 4.6 Gaussian control: "**0.313 m** against **0.319 m** — a paired difference of **−0.005 m** whose sign is not consistent across folds" (lines 1830–1833) — **MATCHES**
S1:
```
  biased-uniform (conference)    0.319 +/- 0.049 m   (k=5)
  matched Gaussian (control)     0.313 +/- 0.043 m   (k=5)

  paired gauss - biased-uniform: -0.005m  per fold [-0.024, -0.018, -0.003, +0.011, +0.007]
```
Signs: −, −, −, +, + → not consistent ✓. (Note the paired −0.005 m is correctly
quoted as the *paired* mean, not the −0.006 m difference of the two means; the
.tex says "paired difference", which is right.)

### 4.7 "several select the **fifth epoch** at δ = 10%" (lines 1836–1837) — **MATCHES**
S3:
```
goku_lane_donkey_f1_d10            0.976m        0.884m      5  -0.092m
v2p_lane_donkey_f3_d10             1.308m        1.172m      5  -0.135m
v2p_lane_donkey_f4_d10             1.204m        1.055m      5  -0.149m
```
Three runs at δ=10% select epoch 5. Corroborated by S1's `[sel]` lines
(`goku_lane_donkey_f1_d10 -> ep5.tar`, `v2p_lane_donkey_f3_d10 -> ep5.tar`,
`v2p_lane_donkey_f4_d10 -> ep5.tar`). The accompanying qualitative claim that the
selected epochs "move sharply earlier as δ grows" is directionally supported by
S3 (v2p: 30/35/50/45 at δ=0 → 40/25/15/5/5 at δ=10%) but no single number is
asserted, so nothing further to check.

### 4.8 Table 1 caption, "Every model, ours included, keeps the epoch with the best E_xy(100)" (line 1672–1673) — **MATCHES (methodological, not numeric)**
S1 header: `SYMMETRIC SELECTION: baselines take the epoch chosen by E_xy@100
(reports\matrix\selected_checkpoints.json, 45 runs). Ours already selects this way`.
The Limitations paragraph (lines 2037–2041) states this correctly.

---

## 5. Section 7.3, "Robustness to a corrupted initial state" (lines 1880–1908)

**Every number in this paragraph is NOT FOUND IN ANY SOURCE.** None of S1–S5
contains an initial-state noise sweep.

| line | claim | verdict |
|---|---|---|
| 1891–1893 | at σ = 1.5: ours **0.749 m**, Vid2Param **0.889 m**, GOKU-net **1.186 m**, DVBF **1.393 m** | NOT FOUND |
| 1895–1896 | our relative degradation **2.8×**, baselines' **2.3–2.5×** | NOT FOUND |
| 1902–1903 | **0.4 m/s** of speed noise "roughly doubles" the 100-step error | NOT FOUND |
| 1886 | **10** draws per window | NOT FOUND (setup parameter) |

The three result figures (`fig_main_cv.pdf`, `fig_delta_supervision.pdf`,
`fig_noise.pdf`) likewise cannot be checked against S1–S5: the per-step curve data
lives in `.npz` files that are not accepted sources.

---

## 6. Preview-length claims (Sections 5.3, 7.3, 8)

| line | claim | verdict |
|---|---|---|
| 1227 | vehicle covers on average **3.3 m** of arc over a 100-step rollout | NOT FOUND |
| 1227 | **4.45 m** at the 90th percentile | NOT FOUND |
| 1071, 1097, 1228, 1719, 2030 | **4.5 m** perceived preview | NOT FOUND (design parameter, not in S1–S5) |
| 1096 | curvature profile sampled at **ten** offsets | NOT FOUND (S2 says only "averaged over the preview offsets") |
| 1719 | "travels on average 3.3 m … well inside the 4.5 m preview" | NOT FOUND |

These are the load-bearing justification for the article's central structural
argument (that the road never has to be extrapolated). They are not traceable to
any accepted source.

---

## 7. Setup numbers (Section 6.2) — not results, listed for completeness

Not measured results, so no MATCHES/MISMATCH verdict is appropriate; none of them
appears in S1–S5.

- line 1413–1415: ≈**22 fps**, Δt = 1/22 s, 100 steps ≈ **4.5 s** — internally consistent (100/22 = 4.545 s).
- line 1434: **k = 15** frames ≈ **0.7 s** — internally consistent (15/22 = 0.68 s).
- line 1436–1439: trained on **16**-step horizons, evaluated at **100** ("more than six times", 100/16 = 6.25 ✓); baselines on **32**-step ("twice ours" ✓).
- lines 908, 933: **512**-entry VQ codebook.
- lines 1364, 1380, 1413, 1433: **96×96** / **64×64** image sizes.

---

## 8. UNVERIFIABLE HERE — out of scope by instruction

These come from the conference paper or from logs not among S1–S5. **Not checked,
and not claimed to be wrong.**

- §7.1 Benchmark Results (Tier 1): all CartPole / Lunar Lander claims (30-step RMSE, parameter-recovery error) — no numbers printed, but the qualitative claims are conference-derived.
- line 1394: **60,000** trajectories of at least **50** steps (Tier-1 data collection).
- §7.2 CarRacing (Tier 2): the .tex correctly reports no quantitative comparison.
- Commented-out CarRacing block (lines 1546–1623): 0.015/0.124/0.559/1.985, 0.008/0.186/0.696/2.020, 0.046/0.927/4.493/15.03, 0.009/0.182/1.242/3.556, 0.001/0.094/1.093/6.044, 0.001/0.038/0.324/1.405, 0.002/0.026/0.074/0.306, and 0.0013/0.0333/0.2671/1.5950 in the comment header.
- Commented-out old Tier-3 block (lines 1923–1960): 0.057 / 0.048 / 0.075 m, 0.0238 m (2.4 cm), 0.05–0.07 m, 0.14–0.25 m. These *are* DonkeyCar numbers but sit inside LaTeX comments and are explicitly annotated as withdrawn, so they are not part of the manuscript. Not checked.

---

## 9. Consolidated MISMATCH list (24)

| # | .tex line(s) | .tex value | source value (file) |
|---|---|---|---|
| 1 | 131, 271, 1732, 1757, 1983, 2091 | ours @100 = **0.267 m** | **0.463 ± 0.046** (S1) / 0.412 ± 0.080 (S5) |
| 2 | 131, 272, 1875, 2091 | "strongest baseline" = **0.383 m** | 0.383 is `ours-a_s0` (S5); strongest baseline = **0.703** (S1) / 0.510 (S5) |
| 3 | 316, 1735, 2095 | camera costs **~2 cm** vs. map | camera is **0.042 m better**, 5/5 folds (S2) |
| 4 | 936 | VQ+T worse by **0.071 m⁻¹** | 0.2191 − 0.1535 = **0.066** (S2) |
| 5 | 936, 1758 | VQ+T worse by **8 mm** | paired **−0.000 m** (S2); +0.008 belongs to the plain Transformer row |
| 6 | 1641 | ours @25 = **0.090 m** | **0.141 ± 0.011** (S1) |
| 7 | 1650 | vs Vid2Param = **0.241 m** | **0.240 m** (S1) |
| 8 | 1658 | spread ratio "**three to four**" | **2.65–4.17×** (S1) |
| 9 | 1421, 1483, 1636, 1919 | tables computed on **201** windows | **2483** windows, 5 folds (S1); 201 is the single-split study (S5) |
| 10 | 1729, 1983 | map lookup **0.246 m** | **0.505 ± 0.044** (S2) |
| 11 | 1731 | true κ profile **0.263 m** | **0.466 ± 0.049** (S2) |
| 12 | 1733, 1999 | RMSE_κ **0.1193 m⁻¹** | **0.1535** (S2) |
| 13 | 1735 | cost = "**8%** of total error" | no cost; **9.1% with opposite sign** (S2) |
| 14 | 1736 | beats baseline by **1.4×** | 0.703/0.463 = **1.52×** (S1) |
| 15 | 1743 | warm-start **0.274 m** | **0.462 ± 0.056** (S2) |
| 16 | 1744, 1757 | frozen backbone **0.290 m** | **0.461 ± 0.063** (S2) |
| 17 | 1751 | VQ+T RMSE_κ **0.190 m⁻¹** | **0.2191** (S2) |
| 18 | 1751, 1756 | CNN RMSE_κ **0.119 m⁻¹** | **0.1535** (S2) |
| 19 | 1755–1756 | "**four** camera rows", factor of **two**, **0.119–0.237** | **eight** camera rows, factor **7.34**, span **0.1223–0.8979** (S2) |
| 20 | 1757 | E_xy spans **2.3 cm (0.267–0.290)** | **1.7 cm (0.454–0.471)** (S2) |
| 21 | 1758 | VQ+T **59%** worse at curvature | **42.7%** (0.2191/0.1535) (S2) |
| 22 | 1761, 1787–1788 | factor **6.6** across camera rows | **7.34×** (0.8979/0.1223) (S2) |
| 23 | 1826 | "the intervals do not overlap" | δ=0 [0.417,0.509] overlaps δ=5% [0.405,0.499] (S1) |
| 24 | 1857 | Vid2Param δ=5% = **0.919** | **0.918** (S1) |
| 25 | 1984 | gap to strongest baseline = **0.12 m** | **0.240 m** (S1) |

*(25 rows; #3, #5, #10 and #16 each cover multiple line occurrences, and the
count of 24 in §0 treats #10 and #13 as one "cost of perception" claim. Either
count is defensible; the list above is exhaustive.)*

## 10. Consolidated NOT FOUND list (11)

1. Crossover at "roughly 65 steps" (line 1639).
2. SINDYc divergence (lines 1652, 1685) — no SINDYc row in S1–S5.
3. "270 of 512 codebook entries in active use" (line 1749).
4. σ = 1.5 values 0.749 / 0.889 / 1.186 / 1.393 m (lines 1892–1893).
5. Degradation factors 2.8× and 2.3–2.5× (lines 1895–1896).
6. "0.4 m/s roughly doubles the 100-step error" (line 1902).
7. "10 draws per window" (line 1886).
8. Average arc length 3.3 m (lines 1227, 1719).
9. 90th-percentile arc length 4.45 m (line 1227).
10. 4.5 m preview length (lines 1071, 1097, 1228, 1719, 2030).
11. Curvature profile sampled at "ten offsets" (line 1096); per-fold δ trend
    "holds in all five folds" (line 1826).
