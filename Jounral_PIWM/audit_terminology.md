# Terminology Alignment Audit — `elsarticle-template-num.tex`

Date: 2026-08-14. Read-only audit. Sources actually read this session:
- **CONF** = conference PDF (`scratchpad/conf.pdf`, "Physically Interpretable World Models via Weakly Supervised Representation Learning", Mao/Umasudhan/Ruchkin, arXiv:2412.12870v6), all 15 pages.
- **DD** = Deep Dynamics (arXiv HTML 2312.04374, RA-L 2024) — full text fetched.
- **LEE** = terrain-aware kinodynamic (ar5iv 2305.00676, RA-L 2023) — full text fetched.
- **CADY** = Causal Structure Distributions (arXiv HTML 2508.06742v1, RA-L 2025) — full text fetched.
- `TERMINOLOGY.md` §3 (cross-checked; consistent with what I fetched).

Verdicts: **KEEP** / **RENAME-TO-X** / **NO-PRECEDENT-FOUND** (no precedent in any read source; standard-English cases still get KEEP with a note).

---

## A. Evaluation wording

| Manuscript string (line) | CONF says | Exemplars say | Verdict |
|---|---|---|---|
| "five-fold cross-validation" (L1641, L273, L1678, L1788, L2108) | "under **5-fold cross-validation**" (p.7, §4.1) | DD: none (single point estimates). LEE: none. CADY: none — uses "runs"/"trials" | **KEEP** (conference precedent; spelling out "five" is house-style, fine for Elsevier) |
| "mean $\pm$ **half-range** over the five fold-level means" (L1645); "mean $\pm$ half-range over five-fold cross-validation" (L1678, L1789, L1851) | CONF Table 1 reports "0.12 ± 0.04" etc. **without ever defining ±**; figure bands undefined | CADY (verbatim): "Statistics (**mean ± std**) computed over five runs"; "**Mean ± std. dev. over 10 trials**". LEE: "the **mean and standard deviation**". DD: no variance at all. **"half-range" appears in none of the four sources** | **RENAME-TO** "mean $\pm$ standard deviation over the five folds" (CADY/LEE authority). If the authors deliberately want the min–max statistic (defensible at n=5), keep the number but say it in standard English **once**: "where $\pm$ denotes half the range (min–max) across the five fold means" — and then "half-range" may be reused. Bare undefined "half-range" is NO-PRECEDENT-FOUND |
| "fold-to-fold variation" (L133) | not in CONF | not in exemplars | **KEEP** — plain-English description, not a coined term; but if half-range→std, consider "fold-to-fold standard deviation" where a number follows |
| "paired within folds" (L938, L1646, L1762) | not in CONF | not in exemplars | **KEEP** — "paired" is standard statistics vocabulary (paired comparison/paired test); usage at L1646 explains itself ("every model rolls out from the same initial states") |
| "fold-mean curves" (L1657) | not in CONF | not in exemplars | **KEEP** — transparent compound, standard English |
| "held-out episodes" (L1685, L1668), "held-out validation windows" | not in CONF (CONF: "held-out validation split shared across all models", §4.1 — **precedent for "held-out"**) | DD avoids "held-out" (uses $\mathcal{D}_{test}$); CADY/LEE: no | **KEEP** (conference itself uses "held-out") |
| "symmetric selection is not unbiased selection" (L2054) | not in CONF (CONF used early stopping on validation loss — the manuscript says so at L2056) | not in exemplars | **KEEP** — not a coined technical term; it back-references "We apply that rule symmetrically" one sentence earlier, so it is self-defining. Do not promote it to a heading or unexplained noun elsewhere |
| "crossover" (L1648, L1656, L1718, L1720, L1722) | not in CONF | not in DD/LEE/CADY (verified: CADY "does not contain wording describing one method overtaking another as horizon grows") | **KEEP** — standard English for curves intersecting; no exemplar models this rhetoric at all (TERMINOLOGY.md §3.5 reached the same conclusion), so there is nothing to borrow. It is used consistently and defined by the surrounding numbers |
| "the ordering does not carry over" (L1795), "the ordering inverts and does not invert back" (L1656) | not in CONF | not in exemplars | **KEEP** — descriptive English, no coinage |
| "permanently ahead" (L1657, L132) | not in CONF | not in exemplars | **KEEP**, minor style note: "permanently" is a strong universal claim from a 100-step window; "ahead for the remainder of the horizon" is the more defensible phrasing if a reviewer pushes. Not a terminology violation |

## B. Model/mechanism wording

| Term | CONF (exact form) | Exemplars | Verdict |
|---|---|---|---|
| "weak supervision" / "weakly supervised" | **YES** — in the CONF title ("Weakly Supervised Representation Learning"), abstract ("weak distribution-based supervision"), contribution 2 ("**distribution-based weak supervision**") | none use it | **KEEP** (conference authority; journal's "distribution-based weak supervision" at L173 matches contribution 2 verbatim) |
| "proxy labels" | **YES** — CONF §3.3: "the empirical mean of the **proxy labels** $\Xi_{t+H}$"; also "state **proxy samples**, $\Xi = \{\xi^{(l)}\}_{l=1}^L$" (§3.1) and "the **proxy supervision set** $\Xi$" (§4.1) | none | **KEEP** — all three conference variants are legitimate; journal may use any of them |
| "privileged" (33×) | **NOT in CONF** (checked all 15 pages) | not in DD/LEE/CADY | **KEEP** — established field term (Vapnik's *learning using privileged information*; teacher–student). The journal correctly anchors it: L493 "privileged-information and teacher--student learning", L1019 "teacher--student / learning-from-privileged-information". Authority is the cited literature, not the conference |
| "oracle" (L1480 etc.) | NOT in CONF | not in exemplars | **KEEP** — common robotics/RL usage for a ground-truth-access baseline row; L1480 defines it on first use as "a privileged **oracle** variant"; TERMINOLOGY.md §4.4 already endorses the privileged=property / oracle=row-name split. NO-PRECEDENT-FOUND in the four read sources, but standard, defined at first use |
| "map-free" (L1685, L1700, L1862) | NOT in CONF | not in exemplars | **KEEP with note** — NO-PRECEDENT-FOUND in the read sources this session; "map-free" is however a recognized autonomous-driving adjective (map-free relocalization/navigation) and the table caption defines it operationally ("recovers position without consulting the track map at all"). Keep the inline definition; do not use it undefined in the abstract |
| "advected" (L251, L310, L849, L869, L1136, L1186, L1289, L2104) | NOT in CONF (the Frenet mechanism is new to the journal) | not in exemplars | **KEEP** — standard physics English used metaphorically and explicitly glossed at first substantive use (L1289: "the road context is \emph{advected} past it, i.e.\ the profile is **re-indexed** by..."). Keep the gloss |
| "re-indexed" (L1186, L1225, L1293, L1770, L1987, L2104) | NOT in CONF | not in exemplars | **KEEP** — plain English, always paired with the advection gloss |
| "preview" (L663, L690, L1043–1097, L1229–1232, ...) | NOT in CONF | not in exemplars (LEE contains no "preview"/"curvature" at all) | **KEEP** — "preview" is long-established in vehicle dynamics/control ("preview control", curvature preview in MPC); manuscript quantifies it (\SI{4.5}{m}) at first use |
| "curvature profile" (7×) | NOT in CONF | not in exemplars | **KEEP** — standard road-geometry English; TERMINOLOGY.md §4.3 already fixed this as the canonical form over "curvature sequence" |

## C. Symbols — conference vs journal

Conference notation (read from CONF §2–§3 + Appendix):

| CONF symbol | CONF meaning (exact wording) |
|---|---|
| $x$ | "the true physical state" (not directly observable) |
| $y$, $Y$ | observation / observation set |
| $z$ | intermediate/standard latent, $z_t = \mathcal{E}_v(y_t)$ ("Continuous Latent") |
| $z^*$ | "interpretable latent state" |
| $z_p^*$ | "the **physically interpretable** part" (subscript p = physical) |
| $z_v^*$ | "captures the remaining **visual** information necessary for reconstruction" (subscript v = **VISUAL**) |
| $\mathbf{e}_k = [\mathbf{e}_k^p, \mathbf{e}_k^v]$ | codebook split: physical part / visual part |
| $\Xi = \{\xi^{(l)}\}_{l=1}^L$ | proxy supervision set ($L=50$); $\hat\mu_\xi$ its empirical mean |
| $\phi_\theta$, $\phi(z_t^*, z_{t+1}^*, a_{t+1}; \theta)$ | "partially known dynamics" — takes **two consecutive full interpretable states** (second-order window) + action |
| $\mathcal{E}/\mathcal{D}$; $\mathcal{E}_v/\mathcal{D}_v$; $\mathcal{E}_p/\mathcal{D}_p$ | encoder/decoder; **subscript v = vision**, subscript p = physical |
| $\delta$, $\mathcal{X}_i$ | supervision noise width; valid range of state dimension $i$ |
| $\mathcal{L}_{rec}, \mathcal{L}_{interp}, \mathcal{L}_{dyn}, \mathcal{L}_{VQ}, \mathcal{L}_{reg}$; $\beta$ | losses; commitment weight |
| $h$, $g$, $\theta$ | controller, observation function, learnable physical parameters |

Journal keeps $x, y, z, z^*, \Xi, \xi^{(l)}, \hat\mu_\Xi, \mathcal{E}_v/\mathcal{D}_v, \mathcal{E}_p, \mathcal{L}_{interp}, \phi_\theta, h$ — consistent — and **adds** $\zv \equiv z^{v}$ (vehicle-relative state), $\zr \equiv z^{r}$ (road-context state), $\Xi^v/\Xi^r$ (split proxy sets), $f_r(\zr_t, \zv_t, \hat\zv_{t+1}; \theta_r)$ (road transition, genuinely new — no conference counterpart, fine), $\lambda_v/\lambda_r$, $\varepsilon^d_{\theta_r}/\varepsilon^\psi_{\theta_r}$, $e_{\mathrm{CTE}}, e_\psi, \kappa$.

### Collisions found (all verified against both documents)

1. **v = visual vs v = vehicle — inside the journal itself.** L886 recaps the conference intrinsic architecture verbatim: "$\zstar = [z_p^*, z_v^*]$, where ... $z_v^*$ [is the visual part]" — while L67/L240 define $z^{v}$ = **vehicle**-relative state, used ~40×. A reader meets $z_v^*$ (visual) and $z^v$ (vehicle) pages apart, distinguished only by sub- vs superscript. **RENAME (journal side, recap only):** at L886/L894 rename the visual remainder to something unambiguous, e.g. $z^{*}_{\mathrm{app}}$ or $z^{*}_{o}$ ("appearance/observation remainder"), with a footnote "denoted $z_v^*$ in the conference version" — 2 call sites vs ~40 for $\zv$, so change the recap, not $\zv$.
2. **$\mathcal{E}^{v}$ (superscript, L764: "$\zv_t = \mathcal{E}^{v}(y_{1:t})$ approximates the vehicle-relative state") vs $\mathcal{E}_v$ (subscript, L814/L906: vision encoder).** Same letter, same font, opposite meanings, both live in the journal. Conference authority: subscript v on encoders = vision. **RENAME:** the factorized problem-statement encoders at L761–765 to $\mathcal{E}^{\mathrm{veh}}/\mathcal{E}^{\mathrm{road}}$, or fold them into $\mathcal{E}_p$ (which per L920 already "maps ... to the factorized state $[\zv,\zr]$", making $\mathcal{E}^v/\mathcal{E}^r$ redundant).
3. **$\phi_\theta$ signature drift (soft collision).** CONF: $\phi(z_t^*, z_{t+1}^*, a_{t+1}; \theta)$ — full state, two-step window. Journal L1126: $\hat\zv_{t+1} = \phi_\theta(\zv_t, a_t)$ — vehicle component only, single step. Same symbol, different domain and arity. Not wrong (the journal deliberately splits the dynamics), but say so once where $\phi_\theta$ is introduced: "the vehicle transition $\phi_\theta$, the restriction of the conference's full-state dynamics to $\zv$" — or rename to $\phi^v_\theta$ to mirror $f_r$.
4. **$\Xi^v$ superscript-v = vehicle** (L761–763, L962) inherits collision 1: conference $\mathbf{e}_k^v$ superscript-v = visual codebook part. Resolved automatically if 1–2 are fixed and the recap footnote is added; otherwise a reader of both papers will mis-expand $\Xi^v$ as "visual supervision".
5. **$\lambda_v$ (L962) = weight on the vehicle interpretability loss** — same subscript hazard; harmless if 1–2 are fixed since it only appears next to $\Xi^v$; consider $\lambda_{\mathrm{veh}}/\lambda_{\mathrm{road}}$ for free.
6. No collision on: $\Xi$ (journal splits it but keeps the conference meaning), $z$ (intermediate visual latent — same in both), $\mathcal{E}_p$, $\theta$, $h$, $\delta$, $\mathcal{L}_{interp}$ (journal Eq. at L952 matches CONF Eq. 4 exactly).

---

## Summary of recommended actions

1. **RENAME** "half-range" → "standard deviation over the five folds" (CADY: "mean ± std computed over five runs"; LEE: "mean and standard deviation"; DD reports no variance at all — nobody in the field writes "half-range"). Alternative: keep the min–max statistic but define it in standard English once ("$\pm$ denotes half the min–max range across the five fold means") before reusing the word. ~5 sites: L1645, L1678, L1789, L1851, plus L132–133 prose.
2. **RENAME** journal-recap $z_v^*$ (visual) → $z^*_{\mathrm{app}}$ or similar + footnote to conference notation (L886, L894), and $\mathcal{E}^v/\mathcal{E}^r$ (L764–765) → $\mathcal{E}^{\mathrm{veh}}/\mathcal{E}^{\mathrm{road}}$ or fold into $\mathcal{E}_p$. These two fix collisions 1, 2, 4, 5.
3. Optionally annotate or rename $\phi_\theta$ → $\phi^v_\theta$ (collision 3) so the vehicle transition visibly parallels $f_r$.
4. Everything in list B is either conference-attested (weak supervision, proxy labels) or standard field/plain English with an inline definition (privileged, oracle, map-free, advected, re-indexed, preview, curvature profile): **KEEP**.
5. Crossover-regime wording (crossover / does-not-carry-over / permanently ahead): **KEEP**; no exemplar offers alternative phrasing (verified — none of DD/LEE/CADY describes any metric crossover). Only style note: consider "ahead for the remainder of the horizon" over "permanently ahead".
