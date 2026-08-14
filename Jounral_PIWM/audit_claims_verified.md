# Citation Claim Verification (source-checked)

**Date:** 2026-08-14 · **Input:** `audit_citations.md` Part 2 (26 NEEDS-SOURCE-CHECK claims + the
hussein2017imitation item) · **Target:** `elsarticle-template-num.tex`

**Method.** Every verdict below is backed by a fetch performed in this session (arXiv/ar5iv
abstract or full text, publisher page, open repository PDF, or — for Ammoun 2009 — the HAL PDF
downloaded through a browser session and text-extracted with MiKTeX `pdftotext`). Nothing is
marked SUPPORTED from memory. Where a claim has several clauses, each clause is judged; the
headline verdict is the worst load-bearing clause. Quotes are ≤15 words, verbatim from the
fetched source.

**Tally: 17 SUPPORTED · 4 CONTRADICTED · 6 NOT-IN-SOURCE · 0 NO-ACCESS** (27 items).

---

## The four baselines

### NSC-1 — GOKU-net, §2.2 line 406 — **SUPPORTED**
Claim: constrains latent dynamics to a known ODE; infers its parameters; form fixed a priori; latents tied to it.
Source: arXiv 2003.10775 (full text via ar5iv).
- "governed by an ODE with known functional form f" — known-ODE constraint: supported.
- Inference network "infers the ODE parameters" from the observation sequence: supported.
- Functional form is a required input, not learned: supported.
- Latent trajectory is the ODE state trajectory: supported. Training is unsupervised (no
  parameter labels) — consistent with the manuscript's contrast.

### NSC-2 — GOKU-net, §5.3 line 1466 — **SUPPORTED**
Claim: GOKU-net has an ODE-parameter-inference-from-observations pathway (which the paper's ablation disables).
Source: arXiv 2003.10775. The inference function has "two components: The first infers the ODE
parameters", from the observed sequence. GOKU-net does possess exactly the pathway the
manuscript says was removed. (Whether the ablation was actually run that way is a checkpoint
question, outside citation scope.)

### NSC-3 — DVBF, §2.2 line 411 — **CONTRADICTED**
Claim: DVBF adds latent state-space dynamics to VAEs "but, absent physical supervision, do not recover interpretable variables."
Source: arXiv 1605.06432 (full text via ar5iv).
- First half supported: "unsupervised learning and identification of latent Markovian state
  space models" leveraging "Stochastic Gradient Variational Bayes".
- Second half contradicted: the DVBF authors claim the opposite, without any supervision —
  "latent states can be recovered which identify the underlying physical quantities"
  (their Conclusion); pendulum experiments recover position and velocity from pixels.
  The manuscript may still argue DVBF's recovery is incidental/unlabeled rather than
  *aligned to named* variables, but as written the sentence contradicts the source's own claim.
  Suggested fix: rephrase to "do not *label or align* latents to designated physical variables".

### NSC-4 — DVBF "without its filtering step", §5.3 line 1466 — **SUPPORTED**
Source: arXiv 1605.06432. The paper describes "the recognition model as a filter" and a
"filtering mode" in which "reconstruction sampling has access to observation sequence and
performs filtering". DVBF has a filtering step in the authors' own terms.

### NSC-5 — Vid2Param, §2.3 line 443 — **SUPPORTED** (second clause: see caveat)
Claim: "requires full supervision and provides no explicit world model for long-horizon rollout."
Source: arXiv 1907.06422 (full text via ar5iv).
- Full supervision: supported — trained on simulator-generated videos with specified
  ground-truth parameters ("10000 training and 100 test videos", parameter ranges given).
- No explicit world model: supported in the intended sense — prediction uses a *known*
  analytic physics model, "an analytic simulator with a recurrent latent model", not a learned
  world model. Caveat: the paper *does* "make probabilistic forward predictions", so phrase as
  "no learned world model" to be unattackable.

### NSC-6 — Vid2Param infers latent parameter vector from image window, §5.3 line 1462 — **SUPPORTED**
Source: arXiv 1907.06422. "perform online system identification" and "probabilistic inference
of parameters" from video. Inferring physical parameters from video is indeed the method's core.

### NSC-7 — SINDYc "classical dynamics-discovery baseline", §5.3 line 1452 — **SUPPORTED**
Source: arXiv 1605.06682 (SINDYc). "identification of nonlinear dynamical systems with inputs
and forcing using regression methods"; operates on state measurements (Lotka-Volterra, Lorenz),
not images. ("Operating on the interpretable latent" is this paper's setup, not a claim about Brunton.)

### NSC-8 — §2.4 line 454, SINDy/differentiable-physics/end-to-end sysID recover equations or parameters — **SUPPORTED**
- `BRUNTON2016710`: recovers governing equations (above).
- `de2018end` (NeurIPS 2018 abstract page): "a differentiable physics engine ... for
  end-to-end learning"; can "learn physical parameters from data".
- `yao2024marrying` (arXiv 2405.13888): identifiable models that "isolate the
  trajectory-specific parameters"; parameter identification is the point.

### NSC-9 — "DVBF, GOKU-net and Vid2Param are physics-aware latent sequence models", §5.3 line 1450 — **NOT-IN-SOURCE** (for DVBF)
- GOKU-net: physics-aware — dynamics constrained to a known ODE. Supported.
- Vid2Param: "physically based dynamics model" integrated with a recurrent VAE. Supported.
- DVBF: **nothing in the DVBF paper incorporates physical priors or known physics** — it is
  general unsupervised state-space learning. Calling it "physics-aware" is not supported by its
  paper. Suggested fix: "latent sequence models" or single out DVBF as the physics-free one
  (which §5.3 arguably wants anyway, since it is the structural ablation).

---

## Other works

### NSC-10 — SPARTAN, §2.2 line 409 — **SUPPORTED**
Claim: "produces semantically structured outputs but incorporates neither known dynamics nor action-conditioned prediction."
Source: arXiv 2411.06890 (full text via ar5iv).
- Structured outputs: "learns sparse, context-dependent interaction graphs"; "highly
  interpretable world model". Supported.
- No known dynamics: dynamics are purely learned (Transformer); no physics equations. Supported.
- Not action-conditioned: the transition is "p(s^(t+1)|s^t)" with no action input anywhere in
  the paper (interventions ≠ agent actions). Supported.

### NSC-11 — Neural ODEs "learn smooth but uninterpretable latent dynamics", §2.2 line 413 — **SUPPORTED** (with note)
Source: arXiv 1806.07366 (full text via ar5iv). Continuous/smooth: supported —
"continuous-time latent variable models"; "Extrapolating this latent trajectory" arbitrarily in
time. "Uninterpretable": the paper makes **no interpretability claim** for its latents — the
manuscript's negative characterization is consistent with the source's silence, but it is the
manuscript's own assessment, not a statement found in the source.

### NSC-12 — VQ-VAE / VQ-VAE-2 "regularize the latent through a learned codebook", §2.2 line 399 — **SUPPORTED**
- arXiv 1711.00937: encoder "outputs discrete, rather than continuous, codes; and the prior is learnt".
- arXiv 1906.00446: vector-quantized discrete latent codes for large-scale generation.

### NSC-13 — tomar2021, "representation learning specialized for pixel-based control improves downstream performance", §2.2 line 416 — **CONTRADICTED**
Source: arXiv 2111.07775. The paper's finding is the opposite of an endorsement: a simple
baseline "with no metric-based learning, no data augmentations, no world-model learning, and no
contrastive learning" matches specialized methods, and those methods "reduce to the same
performance as the baseline" under distractors. Citing it as evidence that specialized
representation learning *improves* downstream performance misrepresents it. Either invert the
sentence ("...has been critically examined") or swap in a citation that actually shows gains
(e.g. CURL/RAD/DrQ line).

### NSC-14 — Trajectron++, §2.1 line 366 — **SUPPORTED** (two wording caveats)
Source: arXiv 2001.03093 (abstract). "graph-structured recurrent model" incorporating
"heterogeneous data (e.g., semantic maps)", "outperforming a wide array of state-of-the-art"
methods. Caveats: the paper says **semantic** maps, not "high-definition maps", and
Trajectron++ itself is recurrent/graph-based — the "attention-based" half of the class label
needs a different exemplar or no exemplar.

### NSC-15 — KalmanNet, "hybrid schemes embed classical filters inside learned models", §2.1 line 368 — **SUPPORTED**
Source: arXiv 2107.10043. "incorporating the structural SS model with a dedicated recurrent
neural network module in the flow of the KF"; title is literally "Neural Network Aided Kalman
Filtering for Partially Known Dynamics".

### NSC-16 — ammoun2009real, "constant-velocity and Kalman-filter formulations", §2.1 line 356 — **NOT-IN-SOURCE** (constant-velocity half)
Source: HAL inria-00438624 full PDF (downloaded via browser session, text-extracted).
- Kalman filter: supported — "The path prediction is performed using a linear Kalman filter."
- Constant-velocity: **not in the paper.** The word "constant" does not occur; the motion model
  is "the bicycle dynamic model" with position, speed and acceleration as inputs. If the
  sentence keeps "constant-velocity", it needs a different citation (or drop that half here).

### NSC-17 — HJ reachability trio, §2.1 line 356 — **SUPPORTED**
- `li2021prediction` (arXiv 2011.12406): "Hamilton-Jacobi (HJ) Reachability is a formal method
  that verifies safety".
- `nakamura2023online` (arXiv 2210.01199): "a Hamilton-Jacobi (HJ) reachability-based approach";
  "parameter-conditioned forward reachable tube".
- `muthali2023multi` (arXiv 2304.00432, full text): "we turn to Hamilton-Jacobi (HJ)
  reachability analysis"; forward reachable tubes with probabilistic guarantees.

### NSC-18 — Dreamer/PlaNet line "learns recurrent latent state-space models from pixels", §2.3 line 427 — **SUPPORTED**
- `hafner2019learning` (arXiv 1811.04551): "learns the environment dynamics from images",
  "latent dynamics model with both deterministic and stochastic transition components".
- `hafner2020mastering` (arXiv 2010.02193): behaviors learned "in the compact latent space of a
  powerful world model" (ICLR 2021 — note: bib still lists it as arXiv preprint).
- `hafner2023mastering` (arXiv 2301.04104): "learns a model of the environment and improves its
  behavior by imagining future scenarios."

### NSC-19 — "transformer- and masked-prediction world models improve sample efficiency and long-horizon fidelity", §2.3 line 429 — **NOT-IN-SOURCE** (daydreamer placement)
- `micheli2022transformers` (arXiv 2209.00588): "a discrete autoencoder and an autoregressive
  Transformer", "data-efficient agent" on Atari 100k. Supported.
- `seo2023masked` (arXiv 2206.14244): ViT autoencoder reconstructing "masked convolutional
  features"; sample-efficient visual robot learning. Supported.
- `wu2023daydreamer` (arXiv 2206.14176): DayDreamer is a **Dreamer/RSSM** system — neither
  transformer-based nor masked-prediction. It supports "sample efficiency" ("in only 1 hour")
  but does not belong under this descriptor; move it to the Dreamer sentence (line 427).
- "Long-horizon fidelity": none of the three *claims* improved long-horizon fidelity; IRIS only
  notes the world model "has to be accurate over extended periods of time". Soften or cite
  TECO/长horizon-specific work.

### NSC-20 — occupancy/rendered-scene world models "...at substantial annotation and compute cost", §2.3 line 438 — **NOT-IN-SOURCE** (cost clause)
- Structured 3D futures: supported for all four — OccWorld (arXiv 2311.16038) "decode the
  future occupancy and ego trajectory"; UniWorld (arXiv 2308.07234) "predicting 4D geometric
  occupancy"; RenderWorld (arXiv 2409.11356) occupancy + Gaussian-splat rendering;
  GaussianWorld (arXiv 2412.10373) 4D occupancy forecasting.
- No physical-variable alignment: consistent — none of the four discusses aligning internal
  state to speed/heading/curvature.
- "Substantial annotation and compute cost": **the papers claim the opposite direction** —
  OccWorld: occupancy is "more economical to obtain"; UniWorld: "25% reduction in 3D training
  annotation costs"; RenderWorld: self-supervised 3D labels, "reduces GPU memory consumption";
  GaussianWorld: "without introducing additional computations". The clause may be true of the
  occupancy ecosystem at large, but it is not supported by (and sits awkwardly against) the
  cited papers' self-descriptions. Rephrase or cite a survey for the cost point.

### NSC-21 — neuro-symbolic world models "require predefined symbolic inputs unavailable from raw sensors", §2.3 line 441 — **CONTRADICTED** (for liang2024visualpredicator)
- `balloch2023neuro` (arXiv 2301.06294, full text): supported — WorldCloner "uses symbolic
  features from MiniGrid", i.e. predefined object-level state, not raw pixels.
- `liang2024visualpredicator` (arXiv 2410.23156): **contradicted** — VisualPredicator's whole
  contribution is an "online algorithm for inventing such predicates" (neuro-symbolic
  predicates over raw visual input); it exists to remove the predefined-symbols requirement.
  Cite it as the counterexample/progress, or drop it from this sentence.

### NSC-22 — Phy-Taylor "Taylor-monomial-structured latent dynamics", §2.4 line 462 — **SUPPORTED**
Source: arXiv 2209.13511. Augments layers with "monomials of Taylor series expansion of
nonlinear functions capturing physical knowledge".

### NSC-23 — §2.4 lines 459-461, three-mechanism sentence — **CONTRADICTED** (djeumou clause)
- Bounds on states/actions — `tumu2023physics` (arXiv 2302.01060): "a surrogate dynamical model
  to ensure that predicted trajectories are dynamically feasible" — supported (feasibility
  constraints). `sridhar2023guaranteed` (arXiv 2212.01346): "compute bounds that should be
  respected by the neural network in each subset" — supported.
- "Physics-aware losses~\cite{djeumou2023learn}": **contradicted** — Djeumou et al. encode
  physics in the *architecture*, not the loss: "construct the drift term to leverage a priori
  physics knowledge as inductive bias". Recategorize as physics-structured dynamics/inductive
  bias, or cite a genuine physics-loss paper (e.g. PINN line) for that clause.
- Kinematics-inspired layers — `cui2020deep` (arXiv 1908.00219, full text): "a new kinematic
  layer between the final hidden layers", "explicitly embeds the kinematics" — supported.

### NSC-24 — chen2020learning teacher-student/privileged, §4.3 line 1016 — **SUPPORTED**
Source: arXiv 1912.12294 (Learning by Cheating). "This privileged agent cheats by observing the
ground-truth layout"; "the privileged agent acts as a teacher that trains a purely vision-based
sensorimotor agent." The cited setup matches the manuscript's description.

### NSC-25 — kong2015kinematic, §2.5 line 501 — **SUPPORTED**
Source: full PDF (mirror of the IV 2015 paper, verified title/authors). Kinematic and dynamic
bicycle models used for MPC on vehicle state (position, heading, speed); entirely state-based,
no learned image representations — matching "low-dimensional state vectors rather than learned
image representations".

### NSC-26 — CarRacing facts cited to brockman2016openai, §5.1 line 1361 — **NOT-IN-SOURCE**
- The facts are correct: Gymnasium documentation confirms "A top-down 96x96 RGB image of the
  car and race track" and "The generated track is random every episode."
- But the cited Gym paper (arXiv 1606.01540, full text) **never mentions CarRacing**, 96x96, or
  any racing environment; Box2D appears only as a post-release aside. The citation is the
  conventional one for Gym, yet the specific claims are not in it. Optionally add a footnote to
  the Gymnasium/Gym documentation or the environment source for the resolution claim.

### Hussein/Viitala item — §2.5 line 503 — **NOT-IN-SOURCE** (hussein2017imitation)
Claim: "Learning-based control has also been demonstrated on RC-scale physical cars ... platforms akin to the DonkeyCar~\cite{viitala2021learning, hussein2017imitation}".
- `viitala2021learning` (arXiv 2008.00715): **supported**, and strongly — "an RL agent has to
  learn to drive a Donkey car around three miniature tracks"; also tests imitation learning.
- `hussein2017imitation` (open-access CSUR PDF, full text searched): a methods survey with **no
  RC-scale car content** — its driving material is TORCS/simulators and full-scale road
  vehicles (Pomerleau etc.); the authors demonstrate nothing themselves. The citation is
  misplaced for "demonstrated on RC-scale physical cars". Move it to a generic
  imitation-learning sentence or delete it here; `viitala2021learning` alone carries the claim.

---

## Flag list for editing (all CONTRADICTED / NOT-IN-SOURCE)

| Line | Key | Problem | Minimal fix |
|---|---|---|---|
| 411 | karl2016deep | DVBF authors claim unsupervised recovery of physical quantities | rephrase: "do not label/align latents to designated physical variables" |
| 416 | tomar2021... | paper argues specialized representation learning does NOT beat a simple baseline | invert sentence or replace citation |
| 441 | liang2024visualpredicator | VisualPredicator invents predicates from raw visual input | drop from "require predefined symbolic inputs" or cite as counterexample |
| 459-461 | djeumou2023learn | physics enters via drift-term inductive bias, not a loss | move to "physics-structured dynamics" clause |
| 1450 | karl2016deep | DVBF is not physics-aware | relabel the trio |
| 356 | ammoun2009real | no constant-velocity model in the paper (bicycle model + Kalman) | keep for Kalman only; new citation for CV |
| 429 | wu2023daydreamer | DayDreamer is RSSM, not transformer/masked; no long-horizon-fidelity claim in any of the three | move DayDreamer to line 427; soften "long-horizon fidelity" |
| 438 | four occupancy papers | they advertise *reduced* annotation/compute cost | rephrase cost clause or cite a survey |
| 1361 | brockman2016openai | Gym paper never mentions CarRacing/96x96 (facts verified via Gymnasium docs) | optionally add docs footnote |
| 503 | hussein2017imitation | survey, no RC-car demonstrations | remove here; viitala alone suffices |

Secondary notes (SUPPORTED items with wording caveats): line 366 "high-definition maps" →
Trajectron++ says *semantic* maps, and it is not attention-based; line 443 "no explicit world
model" → say "no *learned* world model"; line 427 `hafner2020mastering` bib entry should be
updated to ICLR 2021 (stated on its arXiv page).
