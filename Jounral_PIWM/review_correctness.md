# Correctness review — `elsarticle-template-num.tex`

Reviewer pass: internal consistency, cross-reference integrity, in-document terminology,
claim-vs-evidence, LaTeX hygiene. Read-only; no source file was modified.

**Explicitly excluded by the brief** (not reported below): the numeric values in Section 7.3 and
the results tables (0.267, 0.383, 1.4×, …), which are known stale; writing style/tone; anything
needing an external source (moved to the QUESTIONS section).

**Line numbers verified against the file as of this pass (2145 lines).**

Counts: **1 blocker, 10 major, 15 minor**, plus 4 UNCERTAIN and 4 QUESTIONS.

---

## 1. Internal inconsistency (symbols, quantities, units)

| Line(s) | Severity | What is wrong | Exact current text | Exact proposed replacement | LOCALLY-VERIFIABLE |
|---|---|---|---|---|---|
| 749 (with 1176–1178, 1198, 1204–1214, 1869–1872) | blocker | The factorized state is declared to be 6-dimensional and to contain only $(x_{\mathrm{rel}},y_{\mathrm{rel}},\psi_{\mathrm{rel}},e_{\mathrm{CTE}},e_\psi,\kappa)$, but the dynamics propagate $v$, $\omega$ and $s$ (eqs. 16–19), and Fig. 7(b) perturbs "the speed $v$" and "the yaw rate $\omega$" as *state variables*. None of $v,\omega,s$ is in $\zstar$. | `    \zstar_t = \left[ \zv_t,\; \zr_t \right] \in \mathbb{R}^6.` | `    \zstar_t = \left[ \zv_t,\; \zr_t \right] \in \mathbb{R}^8,` — and extend eq. (3) / eq. (4) so that $\zv_t=(x_{\mathrm{rel}},y_{\mathrm{rel}},\psi_{\mathrm{rel}},v,\omega)$ and $\zr_t=(e_{\mathrm{CTE}},e_\psi,\kappa)$, or state explicitly that $(s,v,\omega)$ are auxiliary variables carried alongside $\zstar$ | yes |
| 752–755 (with 1213, 1224–1226) | major | The pair $(\zv_t,\zr_t)$ is asserted Markov, but the paper's own road transition (20) computes $\kappa_{t+1}$ from the *whole* perceived profile $\hat\kappa(\cdot)$ and from the arc length $s_{t+1}$ — neither of which is in the state. The 6-D state is therefore not Markov by the model's own equations. | `The pair $(\zv_t, \zr_t)$ satisfies the Markov property under the true augmented`<br>`dynamics~\eqref{eq:augmented_dyn}: knowing the current factorized state and the`<br>`action is sufficient to determine the next state, up to the unknown parameters`<br>`$\theta^*$.` | `The pair $(\zv_t, \zr_t)$, together with the perceived curvature profile`<br>`$\hat{\kappa}(\cdot)$ and the arc length $s_t$ carried alongside it, satisfies the`<br>`Markov property under the true augmented dynamics~\eqref{eq:augmented_dyn}:`<br>`knowing these and the action is sufficient to determine the next state, up to`<br>`the unknown parameters $\theta^*$.` | yes |
| 1130 vs 1176–1178, and 872 | major | Eq. (11) declares the vehicle transition as $\phi_\theta(\zv_t, a_t)$ and Fig. 2's caption says the branches' "only coupling" is $\hat\zv_{t+1}$, but the learned actuation map inside $\phi_\theta$ (eq. 16) consumes $\zr_t$. The road therefore feeds the vehicle branch, contradicting both. | `    \hat{\zv}_{t+1} &= \phi_\theta(\zv_t,\; a_t),` | `    \hat{\zv}_{t+1} &= \phi_\theta(\zv_t,\; \zr_t,\; a_t),` (and amend the Fig. 2 caption, which currently claims a single coupling arrow) | yes |
| 1163–1164 vs 1174–1175 and 1220–1221 | major | Three mutually inconsistent statements about what is learned: $l_f,l_r$ "unknown and learned"; then "we learn **only** a compact actuation map"; then "**only** the residual and the actuation map are learned". | `axles (unknown and learned during training), $\delta_{f,t}$ is the steering` | `axles (measured on the platform and held fixed), $\delta_{f,t}$ is the steering` — or, if they really are learned, change L1175 to `We therefore learn the axle geometry $(l_f,l_r)$ together with a compact actuation map` and L1220–1221 accordingly | yes |
| 196, 610 vs 1154–1158 | major | $x_t$ denotes the full vehicle state $(x,y,\psi,v)$ in Sections 1/4, and the scalar longitudinal coordinate in eqs. (13)–(15). Same symbol, two meanings, both in running text. | `The vehicle's physical state $x_t = (x, y, \psi, v)$ does not capture this` | Keep $x_t$ for the state and rename the scalar, e.g. `    p^x_{t+1}    &= p^x_t    + v_t\cos(\psi_t + \beta_t)\,\Delta t,` in eq. (13) (and correspondingly in eq. 14) | yes |
| 997, 1321, 1435 vs 1483, 1488–1496 | major | $k$ is defined as the context-window length ($k=15$) and also used as the rollout step index ($E_{xy}(k)$, $k\in\{25,50,100\}$, "at $k=0$"). It is additionally the codebook index $\mathbf{e}_k$ at L902. | `where $k$ is the context-window length.` | `where $c$ is the context-window length.` (rename the window length to $c$ everywhere: L817, L993, L997, L1321, L1326, L1435; keep $k$ for the rollout step) | yes |
| 890–892 vs 67–68, 768 | major | Notation collision: $z_v^*$ is the *visual* residual component here, while $\zv=z^{v}$ is the *vehicle-relative* state; and $\mathcal{E}_v$ is the vision encoder (L910) while $\mathcal{E}^{v}$ is the vehicle-relative encoder (L768). Sub- vs superscript is the only thing distinguishing them. | `interpretable latent, partitioned as $\zstar = [z_p^*, z_v^*]$, where $z_p^*$ is` | `interpretable latent, partitioned as $\zstar = [z_p^*, z_{\mathrm{img}}^*]$, where $z_p^*$ is` (rename the residual visual component throughout L890–899, and rename the encoders in Problem 1 at L768 to $\mathcal{E}_p^{v}$ / $\mathcal{E}_p^{r}$) | yes |
| 1728, 1972 vs 1746, 1751, 1785 | minor | The same quantity, the deployed encoder's curvature error, is printed at two precisions in the same document. | `\SI{0.1193}{m^{-1}}$.` | `\SI{0.119}{m^{-1}}$.` (also at L1972) | yes |
| 1420–1421 | minor | Two unit-typesetting conventions in one sentence: siunitx for the frame rate, hand-set `\mathrm` for the interval. | `(steering, throttle), and vehicle pose, at approximately \SI{22}{fps}, so that`<br>`$\Delta t = 1/22\,\mathrm{s}$ and the 100-step evaluation horizon corresponds to` | `(steering, throttle), and vehicle pose, at approximately \SI{22}{fps}, so that`<br>`$\Delta t = \SI{1/22}{s}$ and the 100-step evaluation horizon corresponds to` | yes |
| 1444–1447 | minor | "identical conditions" is contradicted inside the same sentence (per-baseline batch sizes) and by the preceding sentence (our 16-step vs baselines' 32-step training horizon). | `All baselines are trained under identical conditions with the same data, the same`<br>`optimizer and the same evaluation protocol, and with the batch sizes at which each`<br>`baseline was tuned;` | `All baselines are trained on the same data with the same optimizer and the same`<br>`evaluation protocol, each at the batch size at which it was tuned;` | yes |
| 1652–1655, 1678 | minor | The text says the error value is not quoted, immediately after quoting two of them; the table row quotes one as well. | `whose lap is a few metres across. Our evaluation clamps it at \SI{100}{m}, so we`<br>`report it as divergent rather than quoting an error value.` | `whose lap is a few metres across. Our evaluation clamps it at \SI{100}{m}, so the`<br>`step-50 and step-100 entries are reported as divergent rather than as error values.` | yes |
| 1431–1433 vs 928–930, 1741–1743 | minor | "the strongest configuration" from the conference paper was extrinsic **+ discrete VQ**; this article keeps the extrinsic half and explicitly rejects the discrete half, but L1431–1433 cites the endorsement without that caveat. | `by a physical encoder producing the factorized state), which the conference paper`<br>`showed to be the strongest configuration.` | `by a physical encoder producing the factorized state), whose extrinsic structure the`<br>`conference paper showed to be strongest; the discrete (VQ) latent it also endorsed is`<br>`not carried over, for the reason given in Section~\ref{sec:repr_learning}.` | yes |

## 2. Cross-reference integrity

All `\ref`/`\eqref` targets resolve to an existing `\label` (89 references checked against 58 labels;
commented-out references point only at commented-out labels). All 81 distinct `\cite` keys exist in
`sample-base.bib`. Every figure (7), table (3) and the algorithm are referenced at least once in the
body. One genuine mis-target:

| Line(s) | Severity | What is wrong | Exact current text | Exact proposed replacement | LOCALLY-VERIFIABLE |
|---|---|---|---|---|---|
| 1664 (target 1449) | major | The reduction of DVBF/GOKU-net to dynamics-only models is stated in §6.3 *Baselines*, not in §6.2 *Experimental Setup*. `sec:setup` resolves to **6.2** (see `elsarticle-template-num.aux`), so this points a reader at the wrong subsection. `\subsection{Baselines}` carries no label. | `and GOKU-net are run as dynamics-only models (Section~\ref{sec:setup}), which is` | Add `\label{sec:baselines}` to `\subsection{Baselines}` on L1449, then: `and GOKU-net are run as dynamics-only models (Section~\ref{sec:baselines}), which is` | yes |

Non-findings confirmed while checking, recorded so they are not re-investigated: `\ref{sec:interp_loss_def}`
at L1404/L1800/L1818 renders as "Appendix A" (aux: `\newlabel{sec:interp_loss_def}{{Appendix~A}...}`),
so the bare `\ref` without a preceding "Appendix" is correct, not a bug. No `\ref`/`\cite` anywhere is
preceded by an ordinary space instead of `~`.

## 3. Terminology inconsistency within the document

| Line(s) | Severity | What is wrong | Exact current text | Exact proposed replacement | LOCALLY-VERIFIABLE |
|---|---|---|---|---|---|
| 245, 618, 740 vs 1054, 1098, 1199 | major | The identical phrase is spelled both ways, three times each. Also `centreline` (1425), `centre of mass` (1163) against `center` elsewhere. | `respect to the lane centre: the cross-track error is the signed lateral` (L1098) | `respect to the lane center: the cross-track error is the signed lateral` — and globally: L1054 `{lane center}`, L1199 `lane center,`, L1163 `centre of mass` → `center of mass`, L1425 `centreline` → `centerline`. (US is the better target: `modeling`, `optimizing`, `quantized`, `organize`, `behavior`, `visualized`, `artifacts` are already US.) | yes |
| 1428, 1484, 1654, 1659, 1770, 1816; 1500, 2013; 872; 1226, 1286; 1440 | minor | Beyond center/centre, the file mixes UK forms — `metres` (6×), `centimetres` (2×), `labelled`, `travelled` (2×), `favours` — with the US forms listed above. No `meters` or `traveled` appears, so the UK spellings are self-consistent but clash with the US majority elsewhere. | `Euclidean distance, in metres, between the predicted and the true vehicle` | `Euclidean distance, in meters, between the predicted and the true vehicle` (and the parallel changes at the other lines listed) | yes |
| 264, 267, 1386 vs 117, 212, 693, 983, 1100, 1388, 1544 | minor | The same camera is called both `front-facing` (3×) and `forward-facing` (7×) — including in adjacent sentences at L1386 and L1388. | `The car observes its environment through a front-facing camera (the test-time` | `The car observes its environment through a forward-facing camera (the test-time` (also L264, L267) | yes |
| 274 vs 1453, 1641, 1522 | minor | The same baseline set is called `data-driven or physics-informed`, `physics-aware latent sequence models`, `physics-aware baselines`, and `data-driven and physics-aware baselines`. | `against \SI{0.383}{m} for the strongest data-driven or physics-informed baseline,` | `against \SI{0.383}{m} for the strongest physics-aware baseline,` | yes |
| 997, 1321, 1435 | minor | Three names for one hyperparameter: `context-window length`, `context length`, `context window`. | `         $\lambda_v, \lambda_r, \lambda_{\mathrm{latent}}$, context length $k$` | `         $\lambda_v, \lambda_r, \lambda_{\mathrm{latent}}$, context-window length $k$` | yes |
| 121, 287 vs 260, 263, 1361, 1368, 1384 | minor | Environment names are set in `\emph` in the abstract/contributions and in `\textbf` in the introduction and Section 6.1. | `In \textbf{CarRacing}, a top-down simulated racing environment, the road geometry` | `In \emph{CarRacing}, a top-down simulated racing environment, the road geometry` (pick one convention and apply to CarRacing, DonkeyCar, CartPole, Lunar Lander) | yes |
| 1198–1199 | minor | $e_{\mathrm{CTE}}$ and $e_\psi$ are defined as the **road-context** state in §4.2 (eq. 4) but are called part of "the vehicle state" here. | `Concretely, we work in a road-aligned (Frenet) frame in which the vehicle state`<br>`is $(s, e_{\mathrm{CTE}}, e_\psi, v, \omega)$, with $s$ the arc length along the` | `Concretely, we work in a road-aligned (Frenet) frame in which the combined`<br>`vehicle--road state is $(s, e_{\mathrm{CTE}}, e_\psi, v, \omega)$, with $s$ the arc length along the` | yes |
| 1774 | minor | "privileged upper bounds" is ambiguous: these rows carry the *lowest* $E_{xy}$ in the table, so "upper bound" reads as an upper bound on error unless the reader supplies the intended sense. | `privileged upper bounds, not deployable models; the remaining four read the front` | `privileged best cases, not deployable models; the remaining four read the front` | yes |

## 4. Claims that do not follow from the document's own evidence

| Line(s) | Severity | What is wrong | Exact current text | Exact proposed replacement | LOCALLY-VERIFIABLE |
|---|---|---|---|---|---|
| 1713–1716, 2005 (against 1233–1235) | major | The paper's own statistic is a 90th percentile of \SI{4.45}{m} against a \SI{4.5}{m} preview — i.e. roughly 10% of windows come within \SI{5}{cm} of leaving the preview, and some plausibly leave it. "well inside", "for the entire horizon", "cannot accumulate" and "with margin" all overstate that. | `average \SI{3.3}{m} over the full rollout, well inside the \SI{4.5}{m} preview,`<br>`the road is \emph{observed} for the entire horizon and error in it cannot`<br>`accumulate.` | `average \SI{3.3}{m} over the full rollout (\SI{4.45}{m} at the $90$th percentile),`<br>`the road is \emph{observed} for essentially the entire horizon in the large`<br>`majority of windows, so error in it does not accumulate.` — and at L2005 replace `covers the full 100-step rollout with margin` with `covers the full 100-step rollout for all but the longest-travelling windows` | yes |
| 258, 291–293, 1413–1415 (against 1538–1545, 2016–2020) | major | The introduction says the formulation is *validated* in two environments and Contribution 1 says the two case studies *demonstrate the benefit*; §7.2 withdraws the CarRacing comparison entirely and Limitation "Sixth" states the article claims no simulated-domain result. §6.2 likewise promises an evaluation ("evaluate whether the factorized-state formulation improves long-horizon prediction") that is never reported. | `We validate the extended formulation in two new partially observable driving`<br>`environments.` | `We develop the extended formulation in two new partially observable driving`<br>`environments and validate it quantitatively on the physical one.` — and at L1413–1415 replace `and`<br>`evaluate whether the factorized-state formulation improves long-horizon prediction`<br>`relative to baselines that use only vehicle-state labels.` with a statement that CarRacing is used for development only, per Section~\ref{sec:carracing_results} | yes |
| 1504–1507 (against 1514–1530) | major | The Metrics subsection commits to reporting Tier-1 state RMSE and relative parameter-recovery error; Section 7.1 contains no numbers, no table and no figure — only prose summary. | `For benchmark tasks (Tier~1) we follow the conference protocol and report state`<br>`RMSE over all state dimensions at a 30-step horizon, together with the relative`<br>`parameter-recovery error for the environments whose ground-truth physical`<br>`parameters are known (CartPole, Lunar Lander).` | `For benchmark tasks (Tier~1) the conference protocol reports state RMSE over all`<br>`state dimensions at a 30-step horizon, together with the relative`<br>`parameter-recovery error for the environments whose ground-truth physical`<br>`parameters are known (CartPole, Lunar Lander); we summarize those results`<br>`qualitatively in Section~\ref{sec:conf_results} and refer to~\cite{piwm_conf} for the tables.` | yes |
| 1496–1498 vs 1661–1662, 1782–1788 | major | "all rollouts are initialized from the ground-truth state at $k=0$" cannot hold for the camera-only rows of Table 2, whose whole point is that $\kappa$ — a component of the state — comes from the encoder and differs per row. As written, all four camera rows would be identical to the "Ground-truth profile" row. | `$k=0$ so that the comparison isolates the dynamics rather than the encoder.` | `$k=0$, and (except where Table~\ref{tab:kappa_source} states otherwise) from the`<br>`ground-truth road context, so that the comparison isolates the dynamics rather than the encoder.` | yes |
| 1152 (against eqs. 13–15) | minor | Eqs. (13)–(15) update global/inertial position and heading with $\cos(\psi_t+\beta_t)$; that is not a body-frame form (in a body frame the position update would be trivial). The contrast the paragraph then draws is global vs road-aligned, not body vs road-aligned. | `In the body frame this is the classical \emph{bicycle kinematic model}` | `In the inertial frame this is the classical \emph{bicycle kinematic model}` | yes |
| 176–177 (against 1518–1520) | minor | "DonkeyCar" in the introduction is the *simulated* conference platform, which the paper elsewhere goes out of its way to distinguish from the physical car of this article; the intro mention is unqualified. | `Experiments on CartPole, Lunar Lander, and DonkeyCar demonstrated that` | `Experiments on CartPole, Lunar Lander, and a simulated DonkeyCar demonstrated that` | yes |

## 5. LaTeX hygiene

| Line(s) | Severity | What is wrong | Exact current text | Exact proposed replacement | LOCALLY-VERIFIABLE |
|---|---|---|---|---|---|
| 227 | minor | A stray lone backslash sits between two paragraphs. It compiles (no error in `elsarticle-template-num.log`) but is not intentional, and it starts the paragraph with a spurious control token. | `\` | *(delete the line)* | yes |
| 318, 1751, 1752 | minor | `\SI{...}{}` with an empty unit argument is a misuse of siunitx; ranges should use `\SIrange`/`\numrange`. | `varies by \SI{2.3}{cm} (\SI{0.267}{}--\SI{0.290}{m}); the VQ--Transformer encoder` | `varies by \SI{2.3}{cm} (\SIrange{0.267}{0.290}{m}); the VQ--Transformer encoder` — likewise `\SIrange{0.119}{0.237}{m^{-1}}` at L1751 and `\num{100}-step` at L318 | yes |
| 566–568 | minor | "Property~1" / "Property~2" are hard-coded numbers pointing at an unlabelled `enumerate`; they silently go stale if the list is reordered. | `Property~1 requires alignment between the latent and the physical state, which` | Add `\label{prop:semantics}` / `\label{prop:temporal}` to the two `\item`s (L561, L563) and write `Property~\ref{prop:semantics} requires alignment between the latent and the physical state, which` | yes |

Checked and clean: no undefined control sequences or errors in the compile log (only 2 overfull and 5
underfull hboxes); no straight `"` quotation marks anywhere in typeset text (the 7 occurrences are all
inside `%%` comments); the three convenience macros `\zv`, `\zr`, `\zstar` are used consistently — the
literal forms `z^{v}`, `z^{r}`, `z^{*}` appear only in the `\newcommand` definitions; no `\textbf` is
used where one of those macros belongs. Baseline names are now internally consistent
(`GOKU-net` 13×, `SINDYc` 6×, `Vid2Param` 15×, `DVBF` 11×) — the variants flagged in `TERMINOLOGY.md`
(`GokuNet`, `GOKU-Net`, `SindyC`) no longer occur.

## UNCERTAIN

- **L1641 vs Table 1 (L1675–1682).** "Up to roughly $65$ steps the physics-aware baselines are
  \emph{ahead} of our model" does not match the table's own step-50 column, where DVBF (0.167) is
  already behind our model (0.158). I am not reporting this as a finding because the brief excludes
  the Section 7.3 values; flagging it only so that whoever rewrites those numbers re-checks the
  "roughly 65 steps" claim and the word "baselines" (plural) against whatever the new table says.
- **L1078, L1103–1104, L1235.** The preview length (\SI{4.5}{m}) and the rollout duration
  ($\approx$\SI{4.5}{s}) are numerically identical but unrelated quantities, and they appear close
  together (L272, L1234, L1638). I could not tell whether this is coincidence or a transcription
  slip; if coincidence, it is worth a clarifying word, but I have no evidence it is an error.
- **L1387 vs L1419–1420.** §6.1 says the car "logs wheel-encoder speed, steering, and pose" while
  §6.2 lists "control actions (steering, throttle), and vehicle pose" without speed — even though
  $v$ is propagated by eq. (16) and perturbed in Fig. 7(b). This may just be differing granularity
  rather than an inconsistency.
- **L1420, `\SI{22}{fps}`.** `fps` is not a siunitx-defined unit. The compile log shows no error or
  warning, so it appears to typeset as a literal; I did not verify what siunitx version is in use or
  whether a stricter version would complain.

## QUESTIONS for the human (need a source outside this repo)

1. **L1152 / eqs. (13)–(15).** I read the given equations as inertial-frame, not body-frame (see the
   finding above). Confirm against whichever reference the bicycle model was taken from before
   changing the wording.
2. **L412–415** characterizes GOKU-net as constraining "the latent dynamics to a known ODE" whose
   parameters are inferred, with "the latents ... tied to it rather than to quantities that can be
   measured from the observation". `TERMINOLOGY.md` §1.2 says an earlier, different characterization
   was factually wrong; the current text appears to be the corrected version, but I cannot verify it
   against the source.
3. **L1468–1471** claims DVBF's "filtering step" and GOKU-net's "inference of ODE parameters from the
   observation sequence" were removed. Whether disabling exactly those components is a fair reduction
   of each published method needs the papers.
4. **L2112–2113** cites NSF Grant Numbers CNS~2513076 and CCF~2403616. Not checkable here.

Bibliography items previously flagged in `TERMINOLOGY.md` (the `karl2016deep` preprint-vs-ICLR issue,
the `asenov2019vid2param` year and casing) are **already fixed** in `sample-base.bib` — `karl2016deep`
is now `@inproceedings` / ICLR 2017 with `{van der Smagt}` braced, and `asenov2019vid2param` now has
`year={2020}` and `{Vid2Param}` brace-protected. No action needed.
