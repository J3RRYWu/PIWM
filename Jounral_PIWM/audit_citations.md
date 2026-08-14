# Citation and Terminology Audit

**Target:** `elsarticle-template-num.tex` (2172 lines) · **Bib:** `sample-base.bib` (121 entries)
**Date:** 2026-08-14 · **Mode:** read-only; nothing in the manuscript or bib was changed.

## What I actually checked, and what I did not

**Checked mechanically, inside this repository only:**

- Every `\cite{...}` key in the .tex (commented-out blocks excluded from the "active"
  set, included in the "unused" test) against the `@type{key,` list in `sample-base.bib`.
- Required fields per entry type against what `elsarticle-num.bst` demands, cross-read
  against the actual BibTeX run log `elsarticle-template-num.blg`.
- Duplicate bib keys, and duplicate *works* by normalized-title collision.
- Which cited entries carry `arXiv preprint` as their venue.
- String-level terminology, spelling and symbol scans over the .tex.

**NOT checked, and deliberately not guessed at:**

- I did **not** open, fetch, download or search for a single cited paper. No web access
  was used at any point in this audit.
- I therefore make **no claim** about whether any sentence in the manuscript accurately
  describes what a cited paper does. Every such sentence is listed in Part 2 as
  NEEDS-SOURCE-CHECK and left unresolved. Where I appear to assert something about a
  cited work below, it is read off the bib entry's own `title`/`journal` field and I say
  so explicitly.
- I did **not** verify any DOI, and I have not quoted any cited paper. There are no
  Crossref lookups in this report because I performed none.
- Whether a given arXiv-only entry has since appeared at a venue is **unknown to me**
  and is listed as NEEDS-SOURCE-CHECK, not asserted.

---

## PART 1 — Citation integrity

### 1.1 Unresolved citations: **none**

101 `\cite` instances resolve to 81 unique keys in the active (uncommented) text. All 81
exist in `sample-base.bib`. Including the commented-out CarRacing and old-Tier-3 blocks
adds no further keys. BibTeX confirms: `You've used 81 entries`, zero
"I didn't find a database entry" errors in `elsarticle-template-num.blg`.

### 1.2 Missing fields for `elsarticle-num`

The BibTeX run itself emits exactly 7 warnings; all are on cited entries:

| Entry | Warning |
|---|---|
| `alshiekh2018safe` | can't use both `volume` and `number`; empty `pages` |
| `piwm_conf` | empty `pages` |
| `li2021prediction` | empty `pages` |
| `alahi2016social` | empty `pages` |
| `gupta2018social` | empty `pages` |
| `karl2016deep` | empty `pages` |

No cited entry is missing a mandatory field for its type (author/title/journal/booktitle/
year all present across all 81). The `pages` warnings are style-nag, not fatal; the
`alshiekh2018safe` volume+number clash is the AAAI-proceedings-as-volume idiom and will
silently drop one field.

### 1.3 Malformed DOI fields (verified in the rendered `.bbl`)

Two cited entries store a full URL in the `doi` field:

- `BRUNTON2016710` — `doi = {https://doi.org/10.1016/j.ifacol.2016.10.249}`
- `ZHONG2023115664` — `doi = {https://doi.org/10.1016/j.cma.2022.115664}`

This is not cosmetic. `elsarticle-template-num.bbl` lines 320–321 render it as:

```
\href {https://doi.org/https://doi.org/10.1016/j.ifacol.2016.10.249}
  {\path{doi:https://doi.org/10.1016/j.ifacol.2016.10.249}}
```

i.e. a doubled `https://doi.org/` prefix and a printed string `doi:https://doi.org/...`.
The SINDYc reference — one of the four comparison methods — currently prints a broken DOI.
Fix: strip to the bare `10.1016/...`.

### 1.4 Duplicate entries for the same work

- **`brockman2016openai` and `1606.01540` are the same work** (OpenAI Gym, Brockman et al.
  2016). Only `brockman2016openai` is cited; `1606.01540` is dead weight but is a genuine
  duplicate key for one work and should be deleted.
- `goodfellow2016deep` / `lecun2015deep` collide on the normalized title "Deep learning"
  but are a book and a *Nature* article respectively — **false positive**, not a duplicate.
  Both are uncited.

No duplicate bib *keys* (121 entries, 121 unique keys).

### 1.5 arXiv preprints among cited entries

18 cited entries give an arXiv identifier as their venue. **None of them contains a
`journal`/`booktitle` field naming a real venue**, so the specific condition you asked
about — an entry that itself shows the work was published somewhere yet still cites the
preprint — is **not satisfied by any entry in this bib**. That is the only thing I can
determine without leaving the repository.

Whether these works have since been published elsewhere is **NEEDS-SOURCE-CHECK**; I did
not look. The list, so someone with source access can triage it:

`ha2018world`, `kingma2013auto`, `hafner2020mastering`, `hafner2023mastering`,
`micheli2022transformers`, `chung2014empirical`, `bai2018empirical`, `brockman2016openai`,
`balloch2023neuro`, `liang2024visualpredicator`, `zuo2024gaussianworld`, `min2023uniworld`,
`yan2024renderworld`, `mao2025safe`, `djeumou2023learn`, `lei2024spartan`,
`tomar2021learningrepresentationspixelbasedcontrol`, `mosbach2025soldslotobjectcentriclatent`.

Two of these are load-bearing for the argument and worth checking first:
`lei2024spartan` (SPARTAN, a claim is made about it in §2.2) and `hafner2023mastering`
(cited twice as the Dreamer line).

**`piwm_conf` is the reverse case and is fine:** it has `booktitle = {ICCPS}`, `year=2026`
*and* `note = {arXiv:2412.12870}`. The note is supplementary, not a substitute.

### 1.6 Key/year mismatches (cosmetic, but they mislead readers of the source)

- `asenov2019vid2param` — key says 2019; the entry's own `year = {2020}` (IEEE RA-L 5(2)).
- `karl2016deep` — key says 2016; the entry's own `year = {2017}` (ICLR).

Both render correctly (the numeric style prints the year field, not the key); this only
matters when reading the `.tex`/`.bib` source, where "2016 DVBF" and "2019 Vid2Param" are
what a reader will carry away.

### 1.7 Bib hygiene: personal paths leak into the submitted file

Two cited entries carry Zotero `file` fields pointing at a named individual's home
directory:

- `mao_phy-taylor_2025` — `file = {IEEE Xplore Abstract Record:/home/ivan/Dropbox/configs/zotero_storage/storage/N62G7CXS/10297119.html:text/html}`
- `linial_generative_2021` — `file = {Full Text PDF:/home/ivan/Dropbox/configs/zotero_storage/storage/RAVINAXX/Linial et al. - 2021 ...pdf:application/pdf}`

BibTeX ignores these, but Elsevier's Editorial Manager takes the `.bib` as a source file.
Ten entries also carry full `abstract` blocks. Neither breaks the build; both are worth
stripping before submission.

### 1.8 Unused entries

40 of 121 bib entries are cited nowhere (including the commented-out blocks). Harmless for
the build. Listed for pruning: `1606.01540`, `Bengio+chapter2007`, `Gauss1857`,
`Lagrange1788`, `NEURIPS2018_2de5d166`, `acharya2023learning`, `agrawallearning`,
`balle2016end`, `cabannnes2021disambiguation`, `chen2022automated`, `extended`,
`ferraro2023focus`, `fu2021learning`, `gatys2016image`, `geng2024bridging`,
`gershman2014amortized`, `goodfellow2016deep`, `goodfellow2020generative`, `he2016deep`,
`konidaris2018skills`, `le2025pixie`, `lecun1998gradient`, `lecun2015deep`,
`lew2022simple`, `lillicrap2015continuous`, `mao2024language`, `mao2024zero`,
`mnih2015human`, `mukhoti2023deep`, `peng2024human`,
`peng2024learninglanguageguidedstateabstractions`, `peper2025four`, `sam2024bayesian`,
`sutton1999between`, `vaswani2017attention`, `wen2023any`, `xie2016unsupervised`,
`xue2019supervised`, `yu2019review`, `zhang2020invariant`.

---

## PART 2 — Citation content: claims requiring source verification

**26 NEEDS-SOURCE-CHECK claims.** Each is a place where the manuscript asserts a property,
method or capability of a cited work rather than merely citing it as related. I have not
verified any of them and give no opinion on whether they are correct. They are ordered
with the four baseline papers first, since those are the comparison methods.

### The four baselines (highest stakes)

**NSC-1 — `linial_generative_2021` (GOKU-net) — §2.2, line 406**
> "GOKU-net constrains the latent dynamics to a known ODE and infers that ODE's
> parameters~\cite{linial_generative_2021}, but the functional form must be fixed a priori
> and the latents are tied to it rather than to quantities that can be measured from the
> observation"

Three separable assertions (constrains to a known ODE; infers its parameters; form must be
fixed a priori) plus a negative characterization. NEEDS-SOURCE-CHECK.

**NSC-2 — `linial_generative_2021` (GOKU-net) — §5.3 Baselines, line 1466**
> "DVBF and GOKU-net are run without their own observation pathways---DVBF without its
> filtering step and GOKU-net without its inference of ODE parameters from the observation
> sequence"

This asserts that GOKU-net *has* an ODE-parameter-inference-from-observations pathway and
that it was disabled. NEEDS-SOURCE-CHECK. **This is the most consequential claim in the
paper about a baseline**, because §7 (Limitation Seventh, line 2047) rests the entire
DVBF/GOKU-net ordering on it.

**NSC-3 — `karl2016deep` (DVBF) — §2.2, line 411**
> "Deep Variational Bayes Filters add latent state-space dynamics to VAEs~\cite{karl2016deep}
> but, absent physical supervision, do not recover interpretable variables"

NEEDS-SOURCE-CHECK.

**NSC-4 — `karl2016deep` (DVBF) — §5.3 Baselines, line 1466**
> "DVBF without its filtering step"

Asserts DVBF has a filtering step that was removed. NEEDS-SOURCE-CHECK.

**NSC-5 — `asenov2019vid2param` (Vid2Param) — §2.3, line 443**
> "The closest prior work, Vid2Param, recovers physical parameters from
> video~\cite{asenov2019vid2param} but requires full supervision and provides no explicit
> world model for long-horizon rollout."

Two negative claims ("requires full supervision", "no explicit world model"). NEEDS-SOURCE-CHECK.

**NSC-6 — `asenov2019vid2param` (Vid2Param) — §5.3, line 1462**
> "Vid2Param additionally infers a latent parameter vector from the image window, since
> inferring physical parameters from video is the whole of that method"

NEEDS-SOURCE-CHECK.

**NSC-7 — `BRUNTON2016710` (SINDYc) — §5.3, line 1452**
> "\textbf{SINDYc}~\cite{BRUNTON2016710} is a classical dynamics-discovery baseline
> operating on the interpretable latent."

The "operating on the interpretable latent" half is a statement about *this paper's* setup,
not about Brunton et al.; the "classical dynamics-discovery" half is a characterization of
the cited work. NEEDS-SOURCE-CHECK (low risk, but listed for completeness).

**NSC-8 — `BRUNTON2016710` — §2.4, line 454**
> "Sparse identification of nonlinear dynamics and differentiable-physics or end-to-end
> system identification recover governing equations or parameters~\cite{BRUNTON2016710,
> de2018end, yao2024marrying}"

NEEDS-SOURCE-CHECK (also covers `de2018end`, `yao2024marrying`).

**NSC-9 — all four baselines jointly — §5.3, line 1450**
> "\textbf{DVBF}~\cite{karl2016deep}, \textbf{GOKU-net}~\cite{linial_generative_2021} and
> \textbf{Vid2Param}~\cite{asenov2019vid2param} are physics-aware latent sequence models"

The label "physics-aware" is applied to all three at once. NEEDS-SOURCE-CHECK.

### Other works the manuscript makes claims about

**NSC-10 — `lei2024spartan` — §2.2, line 409**
> "SPARTAN produces semantically structured outputs but incorporates neither known dynamics
> nor action-conditioned prediction"

Two negative claims about a method. NEEDS-SOURCE-CHECK.

**NSC-11 — `chen2018neural` — §2.2, line 413**
> "continuous-time models such as Neural ODEs learn smooth but uninterpretable latent dynamics"

**NSC-12 — `van2017neural`, `razavi2019generating` — §2.2, line 399**
> "Discrete VQ-VAEs regularize the latent through a learned codebook"

**NSC-13 — `tomar2021learningrepresentationspixelbasedcontrol` — §2.2, line 416**
> "Representation learning specialized for pixel-based control improves downstream performance"

**NSC-14 — `salzmann2020trajectron++` — §2.1, line 366**
> "graph- and attention-based models such as Trajectron++ achieve strong forecasting by
> conditioning on scene graphs and high-definition maps"

**NSC-15 — `revach2022kalmannet` — §2.1, line 368**
> "Hybrid schemes embed classical filters inside learned models, as in learned Kalman filtering"

**NSC-16 — `ammoun2009real` — §2.1, line 356**
> "constant-velocity and Kalman-filter formulations for vehicle motion"

**NSC-17 — `li2021prediction`, `nakamura2023online`, `muthali2023multi` — §2.1, line 356**
> "Hamilton--Jacobi reachability for sets of reachable states with formal guarantees"

**NSC-18 — `hafner2019learning`, `hafner2020mastering`, `hafner2023mastering` — §2.3, line 427**
> "The Dreamer and PlaNet line learns recurrent latent state-space models from pixels"

**NSC-19 — `micheli2022transformers`, `seo2023masked`, `wu2023daydreamer` — §2.3, line 429**
> "transformer- and masked-prediction world models improve sample efficiency and long-horizon fidelity"

**NSC-20 — `zheng2024occworld`, `min2023uniworld`, `yan2024renderworld`, `zuo2024gaussianworld` — §2.3, line 438**
> "recent world models predict rich structured futures such as 3D occupancy grids or rendered
> scenes ... improving forecasting realism but still without aligning the internal state to
> physical variables, and at substantial annotation and compute cost"

The "substantial annotation and compute cost" clause is a quantitative claim about four
papers at once. NEEDS-SOURCE-CHECK.

**NSC-21 — `balloch2023neuro`, `liang2024visualpredicator` — §2.3, line 441**
> "Neuro-symbolic world models improve generalization but require predefined symbolic inputs
> unavailable from raw sensors"

**NSC-22 — `mao_phy-taylor_2025` — §2.4, line 462**
> "Taylor-monomial-structured latent dynamics (Phy-Taylor)"

**NSC-23 — `cui2020deep` / `djeumou2023learn` / `tumu2023physics`, `sridhar2023guaranteed` — §2.4, line 459–461**
> "physical priors have been added as bounds on states and actions~\cite{tumu2023physics,
> sridhar2023guaranteed}, physics-aware losses~\cite{djeumou2023learn}, kinematics-inspired
> layers~\cite{cui2020deep}"

Three distinct method characterizations in one sentence. NEEDS-SOURCE-CHECK.

**NSC-24 — `chen2020learning` — §4.3, line 1016**
> "This is precisely a teacher--student / learning-from-privileged-information
> setup~\cite{chen2020learning}."

"Precisely" is a strong equivalence claim about this paper's method matching that one's.
NEEDS-SOURCE-CHECK.

**NSC-25 — `kong2015kinematic` — §2.5, line 501**
> "physics-based predictors apply kinematic or dynamic bicycle models to low-dimensional
> state vectors~\cite{kong2015kinematic} rather than to learned image representations"

**NSC-26 — `brockman2016openai` — §5.1, line 1361**
> "\textbf{CarRacing}~\cite{brockman2016openai} is a top-down simulated racing environment in
> which the vehicle navigates a procedurally generated track. The observation is a
> bird's-eye-view RGB image (native $96 \times 96$, resized to $64 \times 64$ ...)"

The native resolution is a factual claim about the cited environment. NEEDS-SOURCE-CHECK.

### One claim I can partly evaluate from the bib entry alone

**§2.5, line 503** (`viitala2021learning`, `hussein2017imitation`):

> "Learning-based control has also been demonstrated on RC-scale physical cars, including
> imitation and reinforcement learning on platforms akin to the
> DonkeyCar~\cite{viitala2021learning, hussein2017imitation}, but without an interpretable
> predictive world model."

`viitala2021learning`'s own title is *"Learning to drive (L2D) as a low-cost benchmark for
real-world reinforcement learning"* (ICAR 2021) — plausibly on point.
`hussein2017imitation`'s own title is *"Imitation learning: A survey of learning methods"*
(**ACM Computing Surveys 50(2)**). A survey article is not a demonstration on an RC-scale
physical car. **Reading the bib entry's own title field, this citation appears misplaced for
the sentence it supports.** I am asserting only what the bib entry says about itself; whether
the survey happens to contain an RC-car demonstration is NEEDS-SOURCE-CHECK.

### One uncited claim

**§4.4, line 1145** — "In the body frame this is the classical \emph{bicycle kinematic model}"
introduces Eqs. (10)–(12) with **no citation**, although `kong2015kinematic` (kinematic and
dynamic vehicle models) is in the bib and cited in §2.5. A standard model still normally
takes a citation at the point of use in a journal article.

---

## PART 3 — Terminology consistency (internal only)

### 3.1 Method names — one real variant, plus two naming splits

| Name | Occurrences | Verdict |
|---|---|---|
| `GOKU-net` | 12, all identical | consistent |
| `SINDYc` | 4, all identical | consistent |
| `DVBF` | 11 + one expansion "Deep Variational Bayes Filters" (line 411) | consistent (expansion on first use is correct, though it is at line 411 while the abbreviation `DVBF` first appears at line 1450 — the expansion precedes it, so this is fine) |
| `Vid2Param` | 16 in prose; the 3 lowercase `vid2param` hits are all inside the cite key `asenov2019vid2param` | consistent — **not flagged**, per instruction |
| `PIWM` | 34 | see below |
| `PIWM-Frenet` | 4 (lines 1687, 1688, 1854, 1855) | see below |

**T-1 (real inconsistency) — `PIWM` vs `PIWM-Frenet`.** The proposed method is called plain
"PIWM" everywhere in the abstract, introduction, architecture and conclusion, but the two
results tables label its rows **`PIWM-Frenet (ours, perceived $\kappa$)`** and
**`PIWM-Frenet (map-free)`** (Tables 1 and 3). "PIWM-Frenet" is introduced nowhere in the
text — the string appears for the first time inside a table body, is never defined, and the
Discussion and Conclusion revert to "PIWM". A reader cannot tell whether `PIWM-Frenet` is the
method the paper describes or a variant of it. Either define it at first use in
§4.5 or relabel the table rows.

**T-2 (real inconsistency) — `VQ--Transformer` vs `VQ\,$+$\,Transformer`.** The prose calls
the ablation **`VQ--Transformer`** (lines 940, 1750, 1757) while the Table 2 row is
**`VQ\,$+$\,Transformer`** (line 1806). Same configuration, two renderings.

**T-3 (minor) — `map-free` is undefined at first use.** The row `PIWM-Frenet (map-free)`
appears in Table 1; the term is explained only in the Table 1 caption's last sentence and
never in the running text, and it is not the same thing as the "Known track map" row of
Table 2. Two different map-related contrasts sit in adjacent tables under similar names.

### 3.2 US/UK spelling — clean in prose

- `-ize`/`-ise`: 50 hits, all of them legitimate words (`noise`, `precise`, `supervise`,
  `comprises`, `arising`, `otherwise`). **No `-ise`/`-ize` mixing.**
- `behavior` (line 380) — US, single occurrence, no `behaviour` anywhere. Consistent.
- `color` — 36 hits, **all inside TikZ style definitions** (`color=black!10` etc.), i.e.
  package syntax, not prose. Not a spelling choice.
- **`centre` vs `center`: 7 `centre` hits at lines 646, 656, 666, 1028, 1046, 1239, 1266 — every
  one is the TikZ style name `centre/.style={...}` and its uses.** Prose consistently uses
  `center` / `lane center` / `centerline`. This is **not** a prose spelling inconsistency, but
  the source now mixes both spellings of the same word within the same figure block, and a
  copy-editor grepping the file will flag it. Cosmetic; renaming the style to `centreline`→
  `centerline` would silence it.

**Verdict: no genuine US/UK mixing in the manuscript text.**

### 3.3 One concept, two names / one term, two concepts

**T-4 — the vehicle state is described two incompatible ways.**
- Eq. (4)–(6) define $\zv$ as a **3-vector** $(x_{\mathrm{rel}}, y_{\mathrm{rel}},
  \psi_{\mathrm{rel}})$, and Eq. (7) declares $\zstar = [\zv, \zr] \in \mathbb{R}^6$.
- But Eq. (13) (line 1169) propagates $v$ and $\omega$ as part of the vehicle branch, and
  line 1191 states the vehicle state is $(s, e_{\mathrm{CTE}}, e_\psi, v, \omega)$ — a
  **5-vector in a different frame**, with $s$ replacing $(x_{\mathrm{rel}}, y_{\mathrm{rel}})$
  and $e_{\mathrm{CTE}}, e_\psi$ shared with $\zr$.
- Figure 6(b) / line 1903 then treats $v$, $\omega$, $e_{\mathrm{CTE}}$, $e_\psi$ as "each
  physical state variable", consistent with the 5-vector and not with $\mathbb{R}^6$.

So "the interpretable state" names two different objects. The $\mathbb{R}^6$ declaration in
Eq. (7) is never reconciled with the Frenet state, and $v$, $\omega$ are in no component of
$\zstar$ as defined.

**T-5 — "body frame" vs the equations that follow.** Line 1145 says "In the body frame this
is the classical bicycle kinematic model", but Eqs. (10)–(12) update **global** $x_{t+1},
y_{t+1}$ using the **absolute** heading $\psi_t$ — that is the inertial/global frame, not the
body frame. The frame label contradicts the equations under it.

**T-6 — `Frenet` vs `road-aligned`.** Both name the same frame: "road-aligned (Frenet) frame"
(line 1190), then "road-aligned" alone (lines 1162, 1195, 1990), then "Frenet" alone in the
table row names. One concept, two interchangeable names, plus a third implicit one via the
model name `PIWM-Frenet`.

**T-7 — the heading error has three names.** "vehicle--road heading error" (abstract line 125,
contribution line 299), "heading error" (7×), and "the vehicle's heading relative to the road
tangent" / "angular relationship to it" (line 198). All are $e_\psi$. The abstract-only form
"vehicle--road heading error" never reappears after line 299.

**T-8 — hyphenation of `road context` / `split dynamics` is inconsistent but grammatically
defensible.** `road-context` 48× vs `road context` 40×; `split-dynamics` 7× vs `split
dynamics` 4×. Spot-checking, the split tracks attributive-vs-noun use correctly in the cases
I read (e.g. "road-context encoder" vs "the road context is computed"). Not flagged as an
error; noted so that a copy-editor does not "fix" it in the wrong direction.

### 3.4 Symbols: used before definition, or defined twice

**T-9 (worst symbol collision) — $z_v^*$ vs $\zv$.**
The preamble macro (line 68) defines `\zv` as `z^{v}` = **vehicle**-relative state. But §4.1
line 884 partitions the intrinsic latent as $\zstar = [z_p^*, z_v^*]$ where $z_v^*$ "captures
the residual **visual** information needed for reconstruction". Two different meanings of a
$v$-marked $z$, in the same document, both attached to $\zstar$. Compounded by
$\mathcal{E}_v$ = vision encoder alongside $\zv$ = vehicle state and $\lambda_v$ = the
**vehicle** loss weight (Eq. 9).

**T-10 — $k$ carries three meanings.**
1. codebook entry index, $\mathbf{e}_k$ (line 896);
2. context-window length, $\mathcal{E}_p(z_{t-k+1:t})$, "$k = 15$ frames" (lines 991, 1433);
3. rollout step index in the primary metric, $E_{xy}(k)$, "$k \in \{25,50,100\}$" (lines 1481–1495).

Meanings 2 and 3 are both live in the results discussion, where "$k$" could be 15 or 100.

**T-11 — $N$ carries two meanings.** Dataset size $\{\cdot\}_{1:N}$ = number of trajectories
(Definition 3, line 566; Problem 1, line 759) and $N = 201$ held-out **validation windows**
in Eq. (14) (line 1483).

**T-12 — $\delta$ carries two meanings.** Relative noise width $\delta \in \{0\%,5\%,10\%\}$
(lines 579, 1396, 1820, appendix) and the **steering angle** $\delta_{f,t}$ in the slip-angle
definition (line 1155). Subscripted differently, but the same glyph for a noise scale and an
actuator command.

**T-13 — $\Delta$ carries four meanings.** Sampling interval $\Delta t$; preview offset
$\Delta \in [0, 4.5\,\mathrm{m}]$ (lines 1071, 1077); traveled arc length $\Delta s$ (line 1279);
label-noise shift $\Delta_i$ (appendix line 2152).

**T-14 — $\mathcal{D}_p$ used without definition.** Eq. (3) (line 916) contains
$\mathcal{L}_{\mathrm{recon}}(z, \mathcal{D}_p(\zstar))$. $\mathcal{D}$, $\mathcal{D}_v$,
$\mathcal{E}_p$ and $\mathcal{E}_v$ are all defined; $\mathcal{D}_p$ is not, anywhere.

**T-15 — residual superscript $d$ does not match the variable it corrects.** Eqs. (18)–(19)
introduce $\varepsilon^d_{\theta_r}$ as the residual on $e_{\mathrm{CTE}}$ — superscript $d$
(for "distance"?) where every other reference to that quantity is CTE. $\varepsilon^\psi$
matches its variable; $\varepsilon^d$ does not.

**T-16 — $\omega$ used before definition.** $\omega_{t+1}$ appears in Eq. (13) (line 1171) with
no gloss; "$\omega$ the yaw rate" is stated only at line 1192, one paragraph later.
Similarly $s$ appears in Figure 4 (line 1066, "arc length $s$") before its definition at
line 1191.

**T-17 — $\eta_t$ is called a "Jacobian".** Line 1194: "$\eta_t = 1 - e_{\mathrm{CTE},t}\kappa_t$
for the Jacobian of the road-aligned parameterization". $\eta_t$ is a scalar scale factor
(the determinant / metric coefficient), not a Jacobian matrix. Internal-usage flag only.

---

## Addendum — outside my three parts, but mechanically verified and severe

I was not asked to audit the numbers, but these fell out of reading the file and they are the
kind of error the brief was written to catch. **Every figure below is quoted from the .tex
itself; I resolved none of them against a checkpoint or log**, and doing so is NEEDS-SOURCE-CHECK
against `piwm/checkpoints/` and `figures/`.

**A-1 — The headline number in the abstract, introduction and conclusion contradicts Table 1.**
- Abstract (line 131), Introduction (line 271) and Conclusion (line 2091):
  **0.267 m** for ours against **0.383 m** for the strongest baseline.
- Table 1 (line 1687) and the Results text (line 1647):
  **0.463 ± 0.046 m** for ours against **0.703 ± 0.192 m** (Vid2Param).

Neither pair of numbers appears in the other location. Figure 5's caption (line 1875) repeats
the 0.383 m figure. Contribution 3 (line 316) and the Discussion (line 1983) repeat 0.267 m.

**A-2 — Table 2 contradicts the two paragraphs that describe it, including in sign.**
The prose at lines 1730–1736 states: map lookup **0.246 m**, ground-truth profile **0.263 m**,
encoder **0.267 m** — so the map is *best*, and "the entire cost of replacing a map lookup by
a camera is about 2 cm". Table 2 (lines 1797–1805) gives: known track map **0.505**,
ground-truth profile **0.466**, CNN-from-scratch **0.463** — so the map is *worst*, and the
camera row *beats* both privileged rows. The Discussion (line 1983) repeats the prose pair
(0.267 → 0.246). The later paragraph at lines 1767–1774 argues from the *table's* ordering
("Reading a fixed preview beats consulting a perfect map"), directly contradicting the earlier
paragraph on the same page.

**A-3 — $\mathrm{RMSE}_\kappa$ values contradict Table 2.**
Prose and Discussion quote **0.1193** / **0.119** m⁻¹ for the shipped encoder (lines 1732,
1751, 1998) and **0.190** m⁻¹ for VQ--Transformer (line 1750). Table 2 gives **0.154** for
CNN-from-scratch and **0.219** for VQ+Transformer. Line 1756 says curvature "varies by a
factor of two across the **four** camera-only rows (0.119–0.237)"; Table 2 has **eight**
camera rows spanning **0.122–0.898**. The caption's own "factor of $6.6$" matches the table
(0.898/0.136) while the body text's "factor of two" does not.

**A-4 — §4.1 quotes a third set of numbers.** Line 936 claims the VQ configuration is worse
"by \SI{0.071}{m^{-1}} of curvature error and \SI{8}{mm} of 100-step position error". Table 2
gives 0.219 − 0.154 = 0.065 m⁻¹ and 0.463 − 0.463 = 0.000 m. Line 1758's "8 mm" and "59%
worse" likewise do not follow from the table (0.219/0.154 = 42%).

**A-5 — validation-set size is stated two ways.** "$N = 201$ held-out validation windows"
(lines 1421, 1483, 1636, 1919) vs "**2483** evaluation windows of all five folds"
(lines 1653, 1685). Both are used as *the* evaluation set in adjacent sentences; Figure 6's
caption says "standard errors over the $201$ validation windows" while Table 1 is
five-fold. It is likely 201 = one fold and 2483 = all five, but the text never says so and
uses 201 to describe five-fold results.

These five are consistent with the manuscript having been partially updated from a
single-split evaluation to a five-fold one, with the abstract, introduction, conclusion,
§4.1 and two results paragraphs left on the old numbers. Per `CLAUDE.md`
("论文里的每个数字都要能追溯到 checkpoint 或日志"), every figure above needs to be re-derived
from the current checkpoints before either set is trusted.

---

## Summary

| Check | Result |
|---|---|
| Unresolved `\cite` keys | **0** |
| Duplicate bib keys | **0** |
| Duplicate works under different keys | **1** (`brockman2016openai` / `1606.01540`) |
| Missing mandatory fields | **0** (6 BibTeX `empty pages` / volume+number nags) |
| Malformed DOIs (verified in `.bbl`) | **2** (`BRUNTON2016710`, `ZHONG2023115664`) |
| arXiv entries showing a published venue in-entry | **0** — 18 arXiv-only entries flagged NEEDS-SOURCE-CHECK |
| Claims about cited works | **26 NEEDS-SOURCE-CHECK**, 0 verified by me |
| Citation that appears misplaced from its own bib title | **1** (`hussein2017imitation`) |
| Terminology inconsistencies | **8** (T-1…T-8) |
| Symbol problems | **9** (T-9…T-17) |
| US/UK prose mixing | **none** |
| Numeric contradictions (outside brief) | **5** (A-1…A-5) |

Nothing in this report was verified against any source outside this repository, and no
external lookup of any kind was performed.
