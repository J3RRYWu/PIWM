# PIWM Journal Paper — Terminology Reference Sheet

Reference for `elsarticle-template-num.tex` (Elsevier `elsarticle`, target: *Robotics and Autonomous Systems*).
Compiled 2026-08-13. Every bibliographic claim below is tagged **[VERIFIED]** (checked against a primary
source: publisher page, Crossref DOI record, arXiv, or dblp) or **[UNVERIFIED]** (could not reach a primary
source — do not put it in the paper without checking).

---

## 1. Baseline methods — canonical names and authors' own terminology

### 1.1 DVBF — `karl2016deep`

| Field | Value | Status |
|---|---|---|
| **Canonical name** | **DVBF**, expanded as **Deep Variational Bayes Filters** (plural "Filters") | [VERIFIED] |
| Variant name | **DVBF-LL** (the locally-linear transition variant) | [VERIFIED] |
| Title | *Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data* | [VERIFIED] |
| Authors | Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt | [VERIFIED] |
| Venue | **ICLR 2017** (conference track). Also on arXiv as `arXiv:1605.06432` (2016). | [VERIFIED — dblp `conf/iclr/KarlSBS17`; OpenReview `HyTqHL5xg`] |
| Author-name spelling | Publications use "Soelch" (ASCII) and "Sölch" (Semantic Scholar). The van-particle is written **"Patrick van der Smagt"** — lowercase *van der*. | [VERIFIED] |

**Authors' own component terminology** (exact terms, from the paper body):

| Component | Authors' term |
|---|---|
| Latent state | "**latent state**"; the "latent sequence $z_{1:T}$", $z_t \in \mathcal{Z} \subset \mathbb{R}^{n_z}$ |
| Inferred quantity (what the recogniser actually outputs) | "**transition parameters** $\beta_t$", also "**stochastic parameters**" — *not* the latent state |
| Split of $\beta_t$ | "we split $\beta_t = (w_t, v_t)$": $w_t$ is "a **sample-specific process noise**", $v_t$ are "**universal transition parameters**" |
| Transition/dynamics model | "**transition model**" / "**state transition**"; $p(z_{1:T} \mid \beta_{1:T}, u_{1:T})$ |
| Inference pathway | "**recognition model**" $q_\phi(\beta_{1:T} \mid x_{1:T}, u_{1:T})$ |
| Locally-linear variant | state- and control-**dependent** linear combination $A_t = \sum_{i=1}^{M}\alpha_t^{(i)} A^{(i)}$ over $M$ "triplets of matrices", each a data-**independent** but learned "globally linear system"; the $\alpha_t^{(i)}$ are the mixing weights |
| Multi-step prediction | "**generative sampling**", "(realistic / plausible) **long-term prediction**" |

**What it does (authors' framing):** DVBF performs unsupervised identification of latent Markovian state-space
models from raw data by amortised stochastic-gradient variational inference. Its central design choice is to
have the recognition model emit *transition parameters* rather than states, so gradients flow **through** the
transitions over time and the transition function — not the recognition model — shapes the latent space.

> ⚠️ **Bib discrepancy:** `sample-base.bib` currently has `karl2016deep` as `@article{... journal={arXiv preprint arXiv:1605.06432}, year={2016}}`. The paper was **published at ICLR 2017**. For a journal submission, cite the ICLR version (`@inproceedings`, `booktitle={International Conference on Learning Representations (ICLR)}, year={2017}`) — reviewers notice preprint-only citations of published work. Also, the current bib writes the fourth author as `Van der Smagt, Patrick`; it should be `van der Smagt, Patrick`, and to stop BibTeX mangling the particle, brace it: `{van der Smagt}, Patrick`.

---

### 1.2 GOKU-net — `linial_generative_2021`

| Field | Value | Status |
|---|---|---|
| **Canonical name** | **GOKU-net** (singular) / **GOKU-nets** (plural). Expansion: **G**enerative **O**DE modeling with **K**nown **U**nknowns. | [VERIFIED — abstract: "called **GOKU-net** for Generative ODE modeling with Known Unknowns"] |
| **NOT** | ~~GokuNet~~, ~~Goku-Net~~, ~~GOKU-Net~~, ~~GOKU Net~~. Capital `GOKU`, hyphen, **lowercase** `net`. | [VERIFIED] |
| Title | *Generative ODE Modeling with Known Unknowns* (arXiv casing) / *Generative ODE modeling with known unknowns* (ACM DL sentence casing) | [VERIFIED] |
| Authors | Ori Linial, Neta Ravid, Danny Eytan, Uri Shalit | [VERIFIED] |
| Venue | Proceedings of the Conference on Health, Inference, and Learning (**ACM CHIL '21**), ACM, New York, NY, USA, **pp. 79–94**, 8 April 2021. DOI `10.1145/3450439.3451866`. arXiv `2003.10775`. | [VERIFIED — Crossref] |

**Authors' own component terminology:**

| Component | Authors' term |
|---|---|
| Latent state under the ODE | "**latent trajectories**" $Z$ / $z(t)$; "the dynamics of the latent variables $Z^i$ are governed by an ODE" |
| Inferred parameters | "**known-unknowns**" (the framing term) and "**ODE parameters** $\theta_f$"; they "**estimate the static parameters** $\theta_f$" |
| Dynamics module | "the **known ODE function** $f$" / "the **known ODE functional form** $f$" (given, not learned) |
| Inference pathway | **two separate encoders**: $\phi^{\mathrm{enc}}_{\tilde z_0}$ — "an RNN which goes over the observed $X$ backwards" (initial latent state); and $\phi^{\mathrm{enc}}_{\tilde\theta_f}$ — "a bi-directional LSTM" (ODE parameters) |
| Decoder | "**emission function**" $\hat g$ ("learned emission function") |

**What it does (authors' framing):** GOKU-net is a variational autoencoder whose latent dynamics are *given*
by a known ODE functional form, with the ODE's unobserved variables and static parameters — the
"known-unknowns" — inferred from the observed time series. The authors argue this recovers meaningful
unobserved system parameters, extrapolates much better, and trains from far smaller datasets than LSTM or
Latent-ODE baselines.

> ⚠️ **Paper discrepancy:** the .tex writes **`GokuNet` 12×** (lines 1450, 1466–7, 1614, 1674, 1857, 2021 …)
> and **`GOKU-Net` 1×** (line 412). Both are wrong. Global replace → **`GOKU-net`**.
>
> ⚠️ **Factual issue, line 412:** "GOKU-Net restricts latent variables to plausible physical ranges without
> binding them to specific quantities." This does not match the authors' framing. GOKU-net does *not*
> range-restrict latents; it constrains latent evolution to a **known ODE functional form** and infers that
> ODE's parameters, which *are* specific named physical quantities in their experiments (pendulum length,
> cardiac contractility, vascular resistance). Reword — e.g. "GOKU-net constrains latent dynamics to a known
> ODE and infers its parameters, but the ODE form must be specified a priori and the latents are tied to that
> ODE rather than to quantities measurable from the observation."

---

### 1.3 Vid2Param — `asenov2019vid2param`

| Field | Value | Status |
|---|---|---|
| **Canonical name** | **Vid2Param** — capital `V`, capital `P`, no space, no hyphen. | [VERIFIED] |
| **NOT** | ~~vid2param~~, ~~Vid2param~~, ~~VID2PARAM~~. | [VERIFIED] |
| Title (authors' / arXiv / Edinburgh) | *Vid2Param: Modelling of Dynamics Parameters from Video* (British "**Modelling**") | [VERIFIED — arXiv 1907.06422; Univ. of Edinburgh Research Explorer] |
| Title (IEEE published version) | *Vid2Param: Modeling of Dynamics Parameters From Video* (IEEE Americanises to "**Modeling**" and capitalises "From") | [VERIFIED — Crossref record for DOI `10.1109/LRA.2019.2959476`] |
| Authors | Martin Asenov, Michael Burke, Daniel Angelov, Todor Davchev, Kartic Subr, Subramanian Ramamoorthy | [VERIFIED] |
| Venue | IEEE Robotics and Automation Letters, **vol. 5, no. 2, pp. 414–421, April 2020**. DOI `10.1109/LRA.2019.2959476`. | [VERIFIED — Crossref] |
| Erratum | *Correction to "Vid2Param: Modelling of Dynamics Parameters From Video"*, IEEE RA-L **5(2), p. 2872, April 2020**, DOI `10.1109/LRA.2020.2973022`. | [VERIFIED — Crossref] — *content* of the correction [UNVERIFIED, IEEE Xplore blocks automated fetch] |

**Authors' own component terminology:**

| Component | Authors' term |
|---|---|
| Architecture | "**Variational Recurrent Neural Network (VRNN)**", extended "with additional constraints" |
| Latent split | the latent is decomposed into "**dynamics parameters of interest**" and "a remaining $z'$" used for image reconstruction |
| Inferred parameters | "**dynamics parameters**" / "**parameters of interest**" $\theta$ — e.g. "position, velocity, restitution, air drag and other physical properties" |
| Inference pathway | "**encoder**" ($\phi^{\mathrm{enc}}_\tau$) |
| Training regime | "trained **entirely in simulation**, in an **end-to-end** manner with **domain randomization**" |
| Task framing | "**online system identification**"; "probabilistic **forward predictions** of parameters of interest" |

**What it does (authors' framing):** Vid2Param integrates a physically based dynamics model with a recurrent
variational autoencoder, adding a loss that forces part of the latent to *be* the physical dynamics
parameters. Trained purely in simulation with domain randomisation, it performs online system identification
directly from video and makes probabilistic forward predictions — demonstrated by a PR2 intercepting an
unknown bouncing ball under partial occlusion.

> ⚠️ **Bib discrepancies:** current entry has `title={Vid2param: Modeling of dynamics parameters from video}`
> and `year={2019}`.
> - Casing: `Vid2param` → **`Vid2Param`**, and brace-protect it (`title={{Vid2Param}: ...}`) so the `num` style
>   cannot lowercase it.
> - Year: the issue is **2020** (5(2), Apr 2020); 2019 is only the IEEE Xplore early-access date. Since the
>   cite key is `asenov2019vid2param`, either keep the key and set `year={2020}` (keys need not match years),
>   or change both. **Recommend `year={2020}`, key unchanged** — changing the key means touching 15 call sites.
> - Spelling: pick one. Since you are citing the *IEEE published* record (vol/no/pages), use IEEE's
>   "**Modeling ... From Video**"; if you prefer the authors' own "Modelling", keep it but be consistent.
> - Also fix the 3 lowercase `vid2param` occurrences in the .tex prose.

---

### 1.4 SINDYc — `BRUNTON2016710`

| Field | Value | Status |
|---|---|---|
| **Canonical name** | **SINDYc** — `SINDY` in caps, **lowercase trailing `c`** (for "with control"). Used this way throughout the body text. | [VERIFIED — arXiv 1605.06682 body text] |
| **NOT** | ~~SindyC~~, ~~SINDYC~~, ~~SINDy-c~~, ~~SINDYC~~. | [VERIFIED] |
| Base method spelling **in this paper** | "**SINDY**" (all caps) — note this differs from the more common community spelling "SINDy" used in the PNAS paper and in `pysindy`. Inside this paper, follow "SINDYc". | [VERIFIED] |
| Title | *Sparse Identification of Nonlinear Dynamics with Control (SINDYc)* | [VERIFIED — Crossref DOI `10.1016/j.ifacol.2016.10.249`] |
| Authors | Steven L. Brunton, Joshua L. Proctor, J. Nathan Kutz | [VERIFIED] |
| Venue | **IFAC-PapersOnLine, vol. 49, no. 18, pp. 710–715, 2016** — 10th IFAC Symposium on Nonlinear Control Systems (NOLCOS 2016). | [VERIFIED — Crossref] |
| **Is p. 710 of IFAC-PapersOnLine 49(18) the with-control paper?** | **YES.** Crossref returns exactly this title/author/volume/issue/page tuple for the DOI. Confirmed it is *not* plain SINDy. | [VERIFIED] |

**Distinguish from plain SINDy** (do **not** conflate; different paper, same three authors, same year):

> S. L. Brunton, J. L. Proctor, J. N. Kutz, "Discovering governing equations from data by sparse
> identification of nonlinear dynamical systems," *PNAS* **113**(15):3932–3937, 12 Apr 2016,
> DOI `10.1073/pnas.1517384113`. [VERIFIED — Crossref]

Plain SINDy has **no control input**. SINDYc extends it to systems "with inputs and forcing" / "feedback
control". Since PIWM's baseline is action-conditioned, **SINDYc is the correct citation** — the current bib
entry is right on this point.

**Authors' own component terminology:**

| Component | Authors' term |
|---|---|
| Feature library | "**library of candidate nonlinear functions**" $\Theta(x, u)$; data-matrix form $\Theta^{T}(X, \Upsilon)$; elements are "**candidate functions**" |
| Coefficients | the sparse coefficient matrix $\Xi$ — "the coefficients $\Xi$ in this library are **sparse** for most dynamical systems" |
| Fitting procedure | "**sparse regression**" ("we employ sparse regression to identify a sparse $\Xi$"); LASSO is given explicitly |
| State / input | state $x$ (data matrix $X$); control $u$ (history matrix $\Upsilon$). Input words used interchangeably: "**external inputs**", "**forcing**", "**actuation**", "**feedback control**" |
| Result | "the **model**" / "identifying the **model structure**" |
| Related theory | connections to **dynamic mode decomposition (DMD)** and **Koopman operator** theory |

**What it does (authors' framing):** SINDYc identifies the governing equations of a nonlinear dynamical system
*with actuation* directly from measurement data, by sparse regression of the state derivative onto a library
of candidate nonlinear functions of state **and** control. The result is a parsimonious, human-readable
closed-form model, demonstrated on the forced Lotka–Volterra and Lorenz systems.

> ⚠️ **Paper discrepancy:** the .tex writes **`SindyC` 6×** (lines 1452, 1650, 1676, 1693, and in commented
> blocks). Global replace → **`SINDYc`**.
>
> ℹ️ **Bib note:** the Crossref title carries a trailing funding footnote
> ("**SLB acknowledges support from …**") glued onto the title string. Do **not** copy that in; the clean
> title is *Sparse Identification of Nonlinear Dynamics with Control (SINDYc)*. The current bib entry
> additionally has a stray trailing period after `(SINDYc).` — harmless, but drop it. Brace-protect the
> acronym: `title = {Sparse Identification of Nonlinear Dynamics with Control ({SINDYc})}`.

---

## 2. Summary of baseline-name corrections needed in `elsarticle-template-num.tex`

| Currently in .tex | Count | Correct form | Action |
|---|---|---|---|
| `GokuNet` | 12 | **`GOKU-net`** | global replace |
| `GOKU-Net` (line 412) | 1 | **`GOKU-net`** | replace |
| `SindyC` | 6 | **`SINDYc`** | global replace |
| `vid2param` (lowercase, prose) | 3 | **`Vid2Param`** | replace |
| `Vid2Param` | 15 | ✅ correct | none |
| `DVBF` | 12 | ✅ correct | none |

(Counts are from the .tex as of 2026-08-13 and include commented-out blocks; fix those too, since they are
marked "DO NOT DELETE / TO RESTORE".)

---

## 3. Style exemplars — recent RAS / T-RO / RA-L papers on learned dynamics & world models

For reference, the paper's current experimental structure is:

```
4  Experimental Evaluation      5  Results
   4.1 Environments and Evaluation Tiers    5.1 Benchmark Results (Tier 1)
   4.2 Experimental Setup                   5.2 CarRacing (Tier 2)
   4.3 Baselines                            5.3 Real DonkeyCar Results (Tier 3)
   4.4 Metrics
```

**Provenance note.** All bibliographic records below were checked against the Crossref DOI record
[VERIFIED]. Results-section content was read from the arXiv full text (ar5iv/arXiv HTML) [VERIFIED] —
except the RAS entry (§3.4), whose full text ScienceDirect blocks.

> ⚠️ **Read §3.5 before using these.** On the specific question of *how to word a comparison you partly
> lose*, two of the three exemplars are of no use, because they never concede a metric to a baseline.

### 3.1 Deep Dynamics (2024) — closest topical match; metric vocabulary only

**Citation** [VERIFIED — Crossref]: J. Chrosniak, J. Ning, M. Behl, "Deep Dynamics: Vehicle Dynamics Modeling
With a Physics-Constrained Neural Network for Autonomous Racing," *IEEE Robotics and Automation Letters*,
vol. 9, no. 6, pp. 5292–5297, June 2024. DOI `10.1109/LRA.2024.3388847`.
**URLs:** <https://doi.org/10.1109/LRA.2024.3388847> · full text <https://arxiv.org/abs/2312.04374>
*Caution:* arXiv titles it "Physics-**Informed**"; the published RA-L title is "Physics-**Constrained**". Cite the published form.

**Section ordering** [VERIFIED]: V-A Training and Testing Datasets → V-B Nominal Model Coefficient Ranges →
V-C Evaluation Metrics → V-D Open-Loop Testing → V-E Closed-Loop Testing → V-F Hyperparameter Tuning.

**Cross-validation / error bars** [VERIFIED]: **none.** No mean/std, no seeds, no error bars, no CV — single
point estimates in tables with no uncertainty quantification. Useful mainly as evidence that our 5-fold CV
already exceeds the statistical norm of this literature.

**Metrics** [VERIFIED]: RMSE, maximum error $\epsilon_{\max}$, ADE, FDE (open-loop); lap time, average speed,
track violations (closed-loop). Good precedent for our one-step-vs-horizon metric split.

**Baseline wording** [VERIFIED, verbatim]: "outperforms all variants of the DPM by over 3 orders of
magnitude"; "surpasses the DPM and its variants in terms of RMSE and $\epsilon_{\max}$ by over 8%".
Note the pattern: **percentage deltas reported per state variable** (8% for $v_x$, 56% for $v_y$, 53% for
$\omega$), not as one aggregate — directly transferable to our Frenet state.

**Partial-loss wording: NOT AVAILABLE** [VERIFIED] — the paper identifies no metric on which a baseline beats it.

### 3.2 Terrain-Aware Kinodynamic Model (2023) — the error-vs-horizon figure to imitate

**Citation** [VERIFIED — Crossref]: H. Lee, T. Kim, J. Mun, W. Lee, "Learning Terrain-Aware Kinodynamic Model
for Autonomous Off-Road Rally Driving With Model Predictive Path Integral Control," *IEEE Robotics and
Automation Letters*, vol. 8, no. 11, pp. 7663–7670, Nov. 2023. DOI `10.1109/LRA.2023.3318190`.
**URLs:** <https://doi.org/10.1109/LRA.2023.3318190> · full text <https://ar5iv.labs.arxiv.org/html/2305.00676>

**Section ordering** [VERIFIED]: V-A Experimental Setup → V-B Training Procedure → V-C Analysis on The
Kinodynamic Model (open-loop accuracy) → V-D Autonomous Rally Driving Task (closed-loop). The
open-loop-then-closed-loop split is a clean frame for our Tier-2/Tier-3 story.

**The figure to imitate** [VERIFIED]. Fig. 3 plots multi-step prediction error against horizon, one panel per
predicted variable, mean line plus shaded band. Caption verbatim:

> "Multi-step prediction errors for the values (a) $z$, (b) $\phi$, and (c) $\theta$. Each line represents the
> mean, and the shaded region represents 1/5 of the standard deviation for visual clarity."

Two things to steal: (i) it is **explicit that the band is a scaled std**, not a true 1σ — precisely the
disclosure that protects you at review; (ii) it **panels per variable** rather than collapsing to one scalar,
which suits our factorized Frenet state. (Its panels are terrain variables — height, roll, pitch — not planar
vehicle state, so the analogy is to the *format*, not the content.)

**Repeated-run reporting** [VERIFIED]: closed-loop lap time and peak vertical force as **mean ± std over 10
laps** per controller; 200 min training / 16 min validation data, stated plainly.

**Partial-loss wording: NOT AVAILABLE** [VERIFIED] — no scenario is reported where a simpler baseline is
competitive at any horizon, and no computational trade-off favouring baselines is conceded.

### 3.3 CADY (2025) — statistical rigor, caption discipline, and the only partial-loss precedent

**Citation** [VERIFIED — Crossref]: A. Murillo-González, J. Xu, L. Liu, "Learning Causal Structure
Distributions for Robust Planning," *IEEE Robotics and Automation Letters*, vol. 10, no. 10, pp. 9916–9923,
Oct. 2025. DOI `10.1109/LRA.2025.3598663`.
**URLs:** <https://doi.org/10.1109/LRA.2025.3598663> · full text <https://arxiv.org/html/2508.06742v1>

A latent-space probabilistic dynamics model evaluated on Cartpole, Pusher and a Jackal ground vehicle with
real field trials — structurally our closest analogue (latent world model + physical vehicle).

**Section ordering** [VERIFIED]: V-A Evaluation Setup → V-B Learned Models → V-C Lower Computational
Requirements → V-D Increased Robustness to Missing and Noisy Inputs → V-E Effects of Unmodeled Interventions
→ V-F Field Demonstrations. Organised **by claim, not by apparatus** — each title asserts something the
results then substantiate. Worth considering against our current apparatus-ordered
"Benchmark / CarRacing / Real DonkeyCar".

**Cross-validated reporting — the exemplar** [VERIFIED, verbatim captions]:
- Fig. 4: "Statistics (mean ± std) computed over five runs"
- with the unequal count disclosed rather than hidden: "three runs for CMI and Reg"
- Fig. 9 and Table II: "Mean ± std. dev. over 10 trials"

Every aggregate carries an explicit $n$, and where $n$ differs between methods the paper **says so**. That is
the discipline to copy into our table and figure captions, together with an explicit statement of our 5-fold
CV protocol.

**Baseline wording** [VERIFIED, verbatim]: "CADY outperforms the baseline, exhibiting a 1.6% lower
degradation" — comparator value always given in parentheses.

**Hedged attribution** [VERIFIED, verbatim]: "we hypothesize that this discrepancy arises from weaker
evidence supporting..." — *"We hypothesize"* rather than asserted causation. The right register for our claim
about *why* the physics-grounded latent beats DVBF, and for the Vid2Param-ordering reversal at L2020–2024.

**Partial-loss wording — the one usable precedent** [VERIFIED, verbatim]:
- conceding parity to a baseline (§V-B, PETS on Pusher): *"As more data becomes available, it maintains
  comparable performance or even surpasses it."*
- conceding a weak result while keeping the claim (§V-D, Jackal Mission 2): *"Although we see lower results
  in this hard task, we still achieved 27% success rate."*

The second is the construction we need: **"Although ⟨concession⟩, we still ⟨retained claim⟩."**

### 3.4 RAS venue-match paper — metadata only

**Citation** [VERIFIED — Crossref]: W. Gu, S. Primatesta, A. Rizzo, "Physics-informed Neural Network for
Quadrotor Dynamical Modeling," *Robotics and Autonomous Systems*, vol. 171, art. 104569, Jan. 2024.
DOI `10.1016/j.robot.2023.104569`. URL <https://doi.org/10.1016/j.robot.2023.104569>

**Results-section content: [UNVERIFIED]** — ScienceDirect returns HTTP 403 to automated fetching and no
preprint was located. Included only because it is the *target venue* with a closely related topic; open it
manually to calibrate RAS house style (section depth, figure density, how much derivation RAS tolerates
before the experiments). **Do not cite any claim about its contents from this sheet.**

### 3.5 How to use these — and where they fall short

| Need | Follow | Status |
|---|---|---|
| Section skeleton (claim-driven titles) | CADY §3.3 | verified |
| Open-loop before closed-loop split | Lee et al. V-C → V-D (§3.2) | verified |
| One-step vs horizon metric split | Deep Dynamics V-C/V-D (§3.1) | verified |
| $E_{xy}$-vs-horizon figure format | Lee et al. Fig. 3 — per-variable panels, mean line, **band definition disclosed** (§3.2) | verified |
| Table/figure captions with explicit $n$ | CADY (§3.3) | verified |
| Per-variable percentage deltas | Deep Dynamics (§3.1) | verified |
| "We hypothesize" attribution register | CADY (§3.3) | verified |
| **Wording a comparison you partly lose** | **CADY only** (§3.3) — the other two concede nothing | verified |

> ⚠️ **Honest gap, relevant to our situation.** Our results have baselines winning at short horizons
> (Vid2Param is bold at $k=25$ and $k=50$ in the table at L1673–1680) and PIWM winning at long horizons.
> **None of these three exemplars models that rhetoric well.** Deep Dynamics and Lee et al. report clean
> sweeps and concede nothing; CADY concedes only twice, and never on a headline metric. So there is no
> ready-made template here to copy — the framing has to be built rather than borrowed. What the verified
> material does support:
> 1. Use CADY's **"Although ⟨concession⟩, we still ⟨claim⟩"** construction verbatim as the sentence shape.
> 2. Use Deep Dynamics' **per-variable / per-horizon percentage deltas** so the crossover is reported as a
>    *structured finding* — where each method wins — rather than as one aggregate that hides it.
> 3. State the crossover as the paper's own claim in the section title (CADY's claim-driven headings), e.g.
>    "Long-horizon stability", so the short-horizon deficit reads as scope rather than as a loss.
> 4. Pair the concession with a **mechanism**, as the paper already does at L1661–1663 (Vid2Param is the only
>    baseline with a visual pathway) — that existing explanation is the strongest asset here and should be
>    stated *before* the table, not after.
>
> A dedicated search for RA-L/T-RO papers that explicitly report metric crossovers would be worthwhile;
> it was not part of this pass and is **[NOT DONE]**.

---

## 4. Recommended canonical terms for this paper

Use these forms consistently. "First use" gives the recommended full form on first mention; thereafter use the
short form. Rationale flags where the current .tex deviates.

### 4.1 Baseline method names

| Concept | **Use** | Never write | First use in text |
|---|---|---|---|
| Deep Variational Bayes Filters | **DVBF** | Deep VBF, dvbf | "Deep Variational Bayes Filters (DVBF)~\cite{karl2016deep}" |
| Generative ODE modeling with Known Unknowns | **GOKU-net** | GokuNet, GOKU-Net, Goku-net | "GOKU-net~\cite{linial_generative_2021}" |
| Vid2Param | **Vid2Param** | vid2param, Vid2param | "Vid2Param~\cite{asenov2019vid2param}" |
| Sparse Identification of Nonlinear Dynamics with control | **SINDYc** | SindyC, SINDYC, SINDy-c | "SINDYc~\cite{BRUNTON2016710}" |
| our method | **PIWM** | piwm, PiWM | "Physically Interpretable World Model (PIWM)" |
| our driving variant | **PIWM-Frenet** | PIWM-frenet, PIWM Frenet | as used in Table (line 1678) ✅ |

### 4.2 Modelling concepts

| Concept | **Use** | Avoid | Note |
|---|---|---|---|
| The overall learned model class | **world model** (lowercase in running prose) | "World Model" mid-sentence | Capitalise only in the title, section headings, and the `\begin{definition}[World Model]`. Current .tex is already consistent this way ✅ |
| Autoregressive multi-step simulation | **rollout** (noun, one word, no hyphen) | roll-out, roll out (as noun) | "roll out" is fine as a *verb*: "we roll out the model for 100 steps". Current .tex: 42 noun uses ✅, 2 verb uses ✅ |
| The length of a rollout | **rollout horizon** *or* **prediction horizon** — **pick one** | mixing both | Currently .tex uses "rollout horizons" (L1658) and "prediction horizon" (L1572, commented). **Recommend "rollout horizon"** throughout, since the metric is defined per rollout step $k$ |
| Index along the rollout | **rollout step $k$** | timestep $k$, horizon $k$ | matches `eq:metric_exy` ✅ |
| Unconstrained learned latent | **latent state** | "hidden state", "embedding" (inconsistently) | reserve "hidden state" for RNN internals |
| Interpretable, physically-grounded latent | **physical state** | "physics state", "interpretable state" | current .tex uses "physical state" 10× ✅ |
| Road-descriptor part of the physical state | **road context** | "lane context", "road state" | current .tex uses "road context" / "road-context encoder" ✅ |
| Training signal on the physical state | **weak supervision** (noun) / **weakly supervised** (adjective) | "weak-supervision", "weakly-supervised" (hyphenated adj. before noun is acceptable: "weakly supervised training" — Elsevier prefers no hyphen after an *-ly* adverb) | **Never hyphenate after `-ly`**: write "weakly supervised model", not "weakly-supervised model" |
| The specific PIWM flavour of it | **distribution-based weak supervision** | — | as introduced at L171 ✅ |

### 4.3 Geometry / vehicle terms

| Concept | **Use** | Avoid | Note |
|---|---|---|---|
| Road-aligned coordinate frame | **Frenet frame** (capital F, it is a proper name — Jean Frédéric Frenet) | frenet frame, Frenet-frame, "Frenet Frame" | Lowercase `frenet` currently appears only inside `\label{eq:frenet_*}` — that is fine, labels are not typeset. Prose is correct (3×) ✅ |
| First mention | **road-aligned (Frenet) frame** | — | exactly as at L1195 ✅ |
| Local road curvature | **curvature** $\kappa$; the sequence ahead is the **curvature profile** | "curvature sequence", "curvature horizon" | current .tex uses "curvature profile" 7× ✅ |
| Signed lateral offset from lane centre | **cross-track error**, symbol $e_{\mathrm{CTE}}$ | "lateral error", "CTE" unexpanded, "cross track error" (no hyphen) | Expand on first use: "the cross-track error (signed lateral distance from the lane centre)" ✅ L737 |
| Angle between vehicle heading and road tangent | **heading error**, symbol $e_{\psi}$ | "yaw error", "orientation error", "heading offset" | ✅ consistent at L738 |
| Spelling of centre/center | **pick one** | mixing | ⚠️ **Real inconsistency:** the .tex has `center*` **19×** and `centre*` **14×**, including "lane center" at L245/616/738 vs "lane centre" at L1052/1096/1197, plus "centreline" at L1043. Elsevier/RAS accepts either variety but requires internal consistency. **Recommend US "center"/"centerline"** to match "modeling", "quantized", "regularization", "behavior" already used elsewhere |

### 4.4 Evaluation terms

| Concept | **Use** | Avoid | Note |
|---|---|---|---|
| Baseline with information the model must otherwise infer | **privileged** (adjective) — "a privileged variant", "privileged $\kappa$" | "cheating", "upper bound" | ✅ used 33× |
| The specific privileged row | **oracle** (noun/adjective), always paired on first use: "a privileged **oracle** variant in which $\kappa$ is read from the known track map" | using "oracle" alone with no explanation | ✅ L1475 defines it correctly; keep "privileged" as the *property* and "oracle" as the *name of that row* |
| **Primary driving metric** | **position error at rollout step $k$**, symbol $E_{xy}(k)$, units metres | calling it "RMSE", "ADE", "L2 error" | ⚠️ **Important.** $E_{xy}$ is a **mean Euclidean distance**, not a root-mean-square error. The paper already says so explicitly ("Because $E_{xy}$ is a distance and not a squared error…"). Do not let the word RMSE leak into the $E_{xy}$ discussion |
| Note on ADE | $E_{xy}(k)$ at a fixed $k$ is a **final-displacement**-style error, not ADE | do not call it ADE/FDE | The trajectory-forecasting community reserves ADE/FDE for specific averaging conventions; you average over *windows* at a fixed $k$, not over $k$. Say so, or a reviewer will ask |
| Curvature-encoder metric | **$\mathrm{RMSE}_\kappa$**, units $\mathrm{m}^{-1}$ | "curvature error" alone | ✅ consistent |
| Tier-1 benchmark metric | **state RMSE** at a 30-step horizon | "prediction error" | ✅ |
| Held-out set | **held-out validation windows** ($N = 201$) | "test set" (you never call it that) | ✅ consistent |
| Uncertainty on a reported number | **mean $\pm$ standard error over the 201 held-out validation windows** — state the *population being averaged over* every time | bare "$\pm$" with no stated population | ✅ done correctly in the Fig. caption at L1689. ⚠️ But see note below — this is **window** variance, not **seed** variance |

> ⚠️ **Variance-reporting gap (likely reviewer objection).** The paper's only spread statistic is
> **standard error across the 201 evaluation windows** (L1689). There is no **across-seed** variance:
> nothing in the .tex reports training repeats, seed counts, or run-to-run spread for PIWM or for any
> baseline. Robotics venues increasingly expect "mean $\pm$ std over $n$ seeds" for learned models. Either
> add seeds, or say explicitly in the Metrics subsection that the reported $\pm$ is *across evaluation
> windows for a single trained model*, and list it as a limitation. Do not let the two kinds of variance be
> confused by a reader — they answer different questions.

> ⚠️ **Known metric bug, flagged in the .tex itself (L1553):** the commented-out CarRacing table reports
> **MSE**, not the RMSE declared in the Metrics subsection. If that table is ever restored, the metric must be
> restated explicitly. Already documented in the comment block — do not lose it.

---

## 5. Open items / things I could not verify

- **[UNVERIFIED]** Content of the Vid2Param *Correction* (RA-L 5(2):2872). IEEE Xplore blocks automated
  fetching. Its existence, title, page and DOI are verified via Crossref, but *what* was corrected is not.
  If any Vid2Param number is quoted in the paper, check the correction manually first.
- **[UNVERIFIED]** Whether ICLR 2017 has an official page-numbered proceedings citation. ICLR does not issue
  page numbers; `@inproceedings` with `booktitle` + `year` is the accepted form.
- ~~ACM CHIL '21 page range for GOKU-net~~ — **[VERIFIED]** as pp. 79–94 via Crossref, and the bib entry
  already carries `pages = {79--94}` correctly. ✅ No change needed.
- **Bib hygiene (cosmetic):** `linial_generative_2021` carries a Zotero `file = {Full Text PDF:/home/ivan/Dropbox/...}`
  field pointing at someone else's machine, plus a full `abstract`. Neither is rendered by `elsarticle-num.bst`,
  but both are worth stripping before submitting the `.bib` as a source file.
- The `sample-base.bib` entry for `linial_generative_2021` uses sentence-case ACM casing
  ("Generative {ODE} modeling with known unknowns"). That is correct for the CHIL version; the arXiv version
  is title-cased. Either is defensible; the braces around `{ODE}` are already right.
