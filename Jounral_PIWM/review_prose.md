# Prose review: machine-generated register in `elsarticle-template-num.tex`

Read-only review of `elsarticle-template-num.tex` (2146 lines) for prose that reads as
LLM-drafted rather than researcher-written. 12 findings.

**Scope note.** The brief excluded "Section 5.3 or the results tables". That reference is
ambiguous under the document's actual numbering, so I excluded **both** readings and
proposed nothing in either: Section 5.3 *Predicting the Road Environment* (lines
1111--1300) and Section 7.3 *Real DonkeyCar Results* (lines 1627--1888), plus every table
and every table caption. Nothing below touches a number, symbol, equation, citation,
`\ref`, `\label`, or table content.

**Reading the table.** The `.tex` source is hard-wrapped; quoted text is shown joined with
spaces, so a literal search must allow for the line breaks. Line numbers are the first
line of the quoted span.

| Line | Category | Exact current text | Exact proposed replacement | Why it is padding |
|---|---|---|---|---|
| 194 | Empty intensifier | `diverge substantially even under the same control inputs` | `diverge even under the same control inputs` | the point of the sentence is that they diverge at all; the magnitude is asserted nowhere and shown nowhere |
| 228 | Empty intensifier | `These two facets, encoding and predicting the external environment, are precisely what distinguishes` | `These two facets, encoding and predicting the external environment, are what distinguishes` | "precisely" adds emphasis, not exactness — nothing here is being pinned down more narrowly |
| 280 | Empty intensifier | `This journal article substantially extends the conference version~\cite{piwm_conf}` | `This journal article extends the conference version~\cite{piwm_conf}` | "along three axes, which constitute its three contributions" in the next clause already sizes the extension |
| 323 | Restatement (meta) | `Throughout, we precede the technical development of each extension with the specific challenge and motivation that makes it necessary, in keeping with the two facets of partial observability identified above.` | delete lines 323--326; move the `\noindent` on line 322 to sit immediately before `The remainder of this article is organized as follows.` on line 327 | describes the paper's own paragraph habit immediately before the real roadmap paragraph; the "Challenge and motivation" headings do this visibly without being announced |
| 452 | Mechanical parallelism | `Compared with these approaches, PIWM recovers physical state under weak` | `PIWM instead recovers physical state under weak` | three consecutive subsections (390, 425, 452) close with "Compared with th{is,ese}…, PIWM"; the practice is declared on line 354, so one variation is enough — optional, lowest priority in this table |
| 743 | Restatement | `Together, $\zr_t$ encodes the information needed to determine how the vehicle will interact with the road geometry over the coming steps.` | delete lines 743--744 | closes a three-sentence paragraph by restating it, and the sufficiency claim is made properly (and for the right object, the pair) at lines 752--755 |
| 969 | Restatement | `reflects its central role: it is the term that forces` | `is what forces` | "reflects its central role" is a vague placeholder that the clause after the colon then says concretely; the only rewrite (not deletion) in this table |
| 1022 | Restatement | `setup~\cite{chen2020learning}: the privileged road context teaches the encoder what to extract, and at deployment the encoder reconstructs it from images.` | `setup~\cite{chen2020learning}.` | the two preceding sentences (1014--1021) already state that training supplies privileged road context and that the deployed encoder must recover it from images |
| 1364 | Restatement | `These results establish continuity with the conference version and a clean baseline for physically interpretable prediction under simple dynamics.` | `These results establish a clean baseline for physically interpretable prediction under simple dynamics.` | third statement of the same fact — line 1357 says Tier 1 establishes continuity, lines 1361--1363 say the benchmarks are retained from the conference paper |
| 1938 | Restatement (cross-section) | `Before it, the task is smooth extrapolation and a flexible latent model does it better than we do. After it, the question is whether the model knows the geometry it has entered, and this is where a free-latent road representation fails: it must be propagated, its propagation error compounds, and the vehicle branch that consumes it inherits that error.` | delete lines 1938--1943 | reproduces "What the crossover means" (1701--1716) almost clause for clause; the surviving sentences (1935--1937, 1944--1949) carry the discussion's own contribution. Cut here rather than in Results, since Results is being revised separately |
| 1960 | Template phrase | `It is worth noting what the baseline ordering itself says: the single baseline` | `The baseline ordering itself points the same way: the single baseline` | "It is worth noting" is the only instance of the construction in the article and the sentence loses nothing without it |
| 1991 | Restatement | `The model must therefore learn to read road context from onboard images, guided during training by the privileged signal.` | delete lines 1991--1992 | restates line 1986--1987 four lines above it, and restates the encoder paragraph at 1011--1024 |

## What reads fine

- **Abstract, Section 2 (Related Work), Section 3 (Preliminaries), Section 4 (Partial
  Observability in Driving), Section 6.3 (Baselines), Section 6.4 (Metrics), and the
  Limitations paragraph** read as written by a researcher. Related Work in particular is
  dense with specific concessions ("their numbers should be read as characterizing a
  *dynamics-only* version of each method") that no template produces.
- The whole article contains **zero** instances of stacked "Moreover / Furthermore /
  Additionally", and none of "delve", "leverage", "paves the way", "sheds light",
  "plays a crucial role", "the utilization of", or "we propose a novel". One "It is worth
  noting" (row 11 above). That is unusually clean.
- Lines 927--948 ("Which encoder for driving") and 1449--1479 (Baselines) are the strongest
  prose in the paper — they concede things a generated draft would not.

## Deliberately not flagged

- Every hedge carrying technical content: "on the physical platform", "which if anything
  favours them", "we note that", "should be read as", "we have not characterized that
  regime", "approximately", "up to", "may limit accuracy". Several of these look like
  wordiness and are load-bearing.
- Line 1522 `substantially outperforming data-driven and physics-aware baselines`
  (Section 7.1). Borderline, and excluded on purpose: no numbers accompany it here, so
  "substantially" is the only magnitude claim in the sentence and deleting it would change
  what is asserted about the carried-over conference results.
- The conclusion's three consecutive `We introduced …` sentences (2052--2061). Mechanical,
  but it mirrors the numbered contributions and is normal for the genre.

## One non-prose observation (outside my remit, not an edit request)

Line 227 is a lone `\` on its own line between two paragraphs. It is not prose, so it is
not in the table above, but it looks like an editing artifact rather than intentional
spacing — worth a glance from whoever owns the LaTeX.
