# Fix log — `elsarticle-template-num.tex`

Applied by the FIXER pass against `review_correctness.md` and `review_prose.md`.
Line numbers below are **pre-edit** (file was 2145 lines; it is now 2130).
Backup of the pre-edit file: scratchpad `elsarticle-backup.tex`.

Build after edits: `pdflatex → bibtex → pdflatex ×2`, **0 errors, 0 undefined
references, 0 undefined citations, 0 LaTeX warnings, 44 pages** (7 over/underfull
hboxes, the same 7 as before the edits).

---

## APPLIED — 31 edits

### 1. Cross-references (1)

| Line | Change |
|---|---|
| 1449 | `\subsection{Baselines}` → `\subsection{Baselines}\label{sec:baselines}` |
| 1664 | Table 1 caption: `Section~\ref{sec:setup}` → `Section~\ref{sec:baselines}` (the DVBF/GOKU-net dynamics-only reduction is stated in §6.3 Baselines, not §6.2 Setup) |

`sec:setup` remains defined (L1389) and still referenced (L579), so no dangling label.

### 2. US/UK spelling unification (17)

All conversions are orthographic only; no wording, number, symbol, or claim changed.

| Line | Change |
|---|---|
| 872 | `labelled` → `labeled` (Fig. 2 caption) |
| 1054 | TikZ node text `{lane centre}` → `{lane center}` |
| 1098 | `lane centre` → `lane center` (Fig. 3 caption) |
| 1163 | `centre of mass` → `center of mass` |
| 1199 | `lane centre` → `lane center` |
| 1226 | `travelled` → `traveled` |
| 1286 | TikZ node text `travelled arc length` → `traveled arc length` |
| 1425 | `centreline` → `centerline` |
| 1428 | `metres` → `meters` |
| 1440 | `favours` → `favors` |
| 1484 | `metres` → `meters` |
| 1500 | `centimetres` → `centimeters` |
| 1654 | `metres` → `meters` |
| 1659 | `metres` → `meters` (Table 1 caption) |
| 1770 | `metres` → `meters` (Table 2 caption) |
| 1816 | `metres` → `meters` (Table 3 caption) |
| 2013 | `centimetres` → `centimeters` |

**Note for the author:** four of these (1654, 1659, 1770, 1816) fall inside
Section 7.3 / results-table captions, which the brief otherwise fenced off. They
are spelling-only and cannot conflict semantically with the pending number
rewrite, but they are listed separately here so they can be reverted trivially if
7.3 is replaced wholesale.

### 3. LaTeX hygiene (2)

| Line | Change |
|---|---|
| 227 | Deleted the stray lone `\` sitting between two paragraphs |
| 318 | `\SI{100}{}-step` → `\num{100}-step` (empty siunitx unit argument; `\num` is provided by the already-loaded `siunitx`, and the rendered digits are unchanged) |

### 4. Prose — empty intensifiers, template phrases, restatement (11)

| Line | Change |
|---|---|
| 194 | `diverge substantially even under the same control inputs` → `diverge even under the same control inputs` |
| 228 | `are precisely what distinguishes` → `are what distinguishes` |
| 280 | `substantially extends the conference version` → `extends the conference version` |
| 323–326 | Deleted the meta-paragraph announcing the paper's "Challenge and motivation" habit; `\noindent` (L322) moved to sit before `The remainder of this article is organized as follows.` |
| 969–971 | `reflects its central role: it is the term that forces the encoder…` → `is what forces the encoder…` |
| 1022–1024 | Deleted the colon clause after `setup~\cite{chen2020learning}` (restates L1014–1021 verbatim in substance) |
| 1364–1365 | `establish continuity with the conference version and a clean baseline` → `establish a clean baseline` (continuity is already asserted at L1357) |
| 1938–1943 | Deleted the Discussion sentences duplicating the "What the crossover means" paragraph at L1701–1716 |
| 1960 | `It is worth noting what the baseline ordering itself says:` → `The baseline ordering itself points the same way:` |
| 1991–1992 | Deleted `The model must therefore learn to read road context from onboard images, guided during training by the privileged signal.` (restates L1986–1987 and §5.2) |

---

## SKIPPED — with reasons

### Excluded by the brief (not evaluated)

- **L749 `\in \mathbb{R}^6`** and the surrounding factorized-state dimension.
- **L752–755** Markov-property claim.
- **L1130 / L872** eq. (11) coupling contradiction and the Fig. 2 caption amendment.
- **L1163–1164 / 1174–1175 / 1220–1221** three-way "what is learned" inconsistency.
  (Only the `centre of mass` → `center of mass` spelling on L1163 was touched.)
- **L1496–1498** "all rollouts initialized from ground truth" vs Table 2.
- **All numbers, table values, results, Section 7.3 prose, results-table captions**
  — except the four spelling-only changes flagged above.
- **L1713–1716 / L2005** the "well inside the preview / cannot accumulate" rewrite:
  a claim about experimental outcomes, and the proposed replacement measurably
  weakens it. Author's call.
- **L1728 / L1972** 0.1193 → 0.119 precision unification: a number.
- **L1751–1752** `\SIrange{0.267}{0.290}{m}`: inside Section 7.3 and rewrites result
  numbers into a different macro. Left for the numbers pass.
- **L1444–1447** "identical conditions": a claim about the experimental protocol.
- **L1652–1655, 1678** "not quoting an error value": Section 7.3 result prose.
- **L1431–1433** conference "strongest configuration" caveat: a claim about the
  prior paper's findings.
- **L258, 291–293, 1413–1415** "validate in two environments" vs §7.2 withdrawal:
  a claim about what the article demonstrates.
- **L1504–1507** Metrics-vs-§7.1 promise: same.
- **L176–177** "a simulated DonkeyCar": a claim about the conference platform.
- **L1152** body-frame vs inertial-frame: the review itself routes this to the
  author (QUESTIONS §1) pending the bicycle-model reference.
- **L412–415, L1468–1471, L2112–2113** need external sources.

### Reviewer proposals declined on judgement

- **`review_correctness.md` §3 terminology rows other than spelling** — `front-facing`
  vs `forward-facing` (L264/267/1386), baseline-set naming (L274 and friends),
  `context-window length` naming (L997/1321/1435), `\emph` vs `\textbf` for
  environment names (L121/287 vs L260…), "the vehicle state" at L1198–1199,
  "privileged upper bounds" at L1774. These are terminology harmonization, not
  US/UK spelling, and several (`\textbf`→`\emph`, the L1198–1199 rewording) touch
  what the sentence asserts or how results are framed. Outside the authorized scope.
- **Symbol-collision renames** ($x_t$, $k$, $z_v^*$, `\mathcal{E}_v`; correctness rows
  at L196/610, L997, L890–892). Each is a global rename across dozens of sites with
  real breakage risk and a genuine notational decision behind it. Author's call.
- **L566–568 `Property~1` / `Property~2` → `\ref`** (correctness §5). Nothing is
  currently wrong; the numbers render correctly today. Adding labels to a plain
  `enumerate` is a robustness improvement, not a fix to malformed markup, and it is
  outside the three hygiene items in my brief. Left alone deliberately.
- **`review_prose.md` L452** `Compared with these approaches, PIWM recovers` →
  `PIWM instead recovers`. Categorized by the reviewer as "mechanical parallelism"
  (not intensifier / template / restatement) and marked optional-lowest-priority.
  It is a stylistic rewrite, not padding removal. Skipped.
- **`review_prose.md` L743–744** delete `Together, $\zr_t$ encodes the information
  needed to determine how the vehicle will interact with the road geometry over the
  coming steps.` This is a **sufficiency claim about $\zr_t$**, sitting immediately
  above the explicitly-fenced-off L749 dimension and L752–755 Markov paragraph. The
  reviewer's justification is that the claim is made "properly" at L752–755 — but
  L752–755 is exactly the text the author has been asked to reconcile, so deleting
  the L743–744 restatement now could silently remove the surviving statement of that
  claim. Skipped as an authorial judgement.
- **L34 `greyness`, L1039/1050/1549/1558 comment-only UK spellings** — inside `%%`
  comments, never typeset.
- **TikZ style identifiers `centre/.style` and `\draw[centre]`** (L652, 662, 672,
  1035, 1053, 1246, 1273) — internal style names, functionally identifiers like
  `\label` keys, not typeset text. Not renamed.
- **`stocco2020misbehaviour` (L386)** — a BibTeX cite key containing `behaviour`.
  Deliberately **not** touched; this is precisely the trap the brief warned about.
  The surrounding prose already reads `unsafe behavior`, correctly US.
- **L2107 `\section*{Acknowledgements}`** — Elsevier house-style heading supplied by
  the template, not authorial prose. Left as-is.
