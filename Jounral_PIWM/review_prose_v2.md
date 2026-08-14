# Prose review v2 — freshly rewritten results narrative only

Scope: abstract close (129–134), intro ¶ (258–276), 5.1 encoder ¶ (923–945),
architecture-and-training ¶ (1428–1450), long-horizon results (1630–1783),
delta-noise (1819–1844), Limitations (2037–2074), conclusion close (2089–2120).
Constraints honored: no numbers, symbols, citations, refs, labels, or table
content touched; no hedge removed; deletion preferred over rewriting.

## Findings

| line | exact current text | exact proposed replacement | reason |
|---|---|---|---|
| 269–271 | `This setup tests whether the factorized representation learned under privileged supervision transfers to the more challenging unaided real-world deployment setting.` | `This setup tests whether the factorized representation learned under privileged supervision transfers to the unaided real-world deployment setting.` | "more challenging" is an empty intensifier; "unaided real-world deployment" already carries the difficulty |
| 927 | `We do not, and the reason is worth stating rather than glossing: that comparison` | `We do not: that comparison` | self-referential meta-commentary ("worth stating rather than glossing") announces the reason instead of just giving it, which follows immediately anyway |
| 1635–1637 | `This is the setting the article is about: the camera sees a short stretch of road, the vehicle state alone is non-Markov, and nothing about the track is available to the deployed model beyond what it can read from its own images.` | *(delete sentence)* | restates the immediately preceding sentence ("supervised only during training and must be inferred from the front camera at test time") plus the non-Markov point already made in Sections 1 and 4, wrapped in a showman frame |
| 1656 | `Beyond the crossover the ordering inverts and does not invert back: on the fold-mean curves` | `Beyond the crossover the ordering inverts: on the fold-mean curves` | "does not invert back" duplicates "permanently ahead" later in the same sentence |
| 1744 | `Perceiving the road from the camera costs nothing measurable.` | *(delete sentence)* | restates the 3 mm paired difference quoted one sentence earlier as an absolute flourish — and 0.003 m was in fact measured, so the punchline overstates the number it summarizes |
| 1751–1752 | `The table also settles the encoder-architecture question, in the least flattering way possible: it does not matter.` | `The table also settles the encoder-architecture question: it does not matter.` | "in the least flattering way possible" is showmanship (flattering to whom is never resolved) and doubles the "flattered" already used at line 1450 |
| 1835–1836 | `It would be convenient to read this as evidence that the particular biased uniform noise of the conference formulation benefits a structured model. It does not.` | *(delete both sentences)* | dramatic-negation template ("It would be convenient... It does not."), the third punchy "It does not" construction within ~90 lines (cf. 1752, 1768); the Gaussian control and the sentence "The finding is the \emph{asymmetry}..." already make the point without the strawman setup |
| 2091–2092 | `The central observation is that driving is partially observable: the vehicle's physical state alone is non-Markov` | `The central observation is that the vehicle's physical state alone is non-Markov` | "driving is partially observable" repeats the previous sentence's "partially observable driving" verbatim; the non-Markov clause *is* the observation |

## Pattern noted, no single fix proposed

The results-and-conclusion stretch contains four self-framing declarations:
"This is the setting the article is about" (1635), "the crossover between them
is the central result" (1648), "the clearest evidence we have for the article's
central claim" (1767), "The central observation" (2091). Rows above remove two;
the two survivors (1648, 1767) each sit atop the evidence they describe and can
stand, but a fifth should not be added.

## Passages that read fine

- **Abstract final sentences (129–134):** every clause carries a distinct claim
  (per-fold win, spread, interpretability); the em-dash close is earned.
- **Architecture-and-training ¶ (1428–1450):** dense, factual, no filler; the
  "which if anything favors them" hedge is load-bearing and correctly kept.
- **"What the crossover means" ¶ (1718–1734):** the free-latent vs. re-read
  contrast is argument, not ornament.
- **"we return to why below" (1746–1747):** earns its place — the map-worse-
  than-camera result is genuinely counterintuitive and the pointer prevents the
  reader from stalling on it; the answering paragraph (1774–1781) delivers.
- **Kappa re-indexing ¶ (1774–1783):** reads as a researcher reasoning, not a
  template; "The corollary is practical" is followed by an actual corollary.
- **Delta-noise body apart from the row above (1829–1844):** "which is what
  fitting the label noise almost immediately looks like" is voice, but it
  carries a real diagnostic and should stay.
- **Limitations (2037–2074):** the First/.../Finally enumeration is mechanical
  by design and appropriate here; items five and seven are unusually candid and
  should not be softened.
- **Conclusion's "We introduced... We introduced... And we introduced" (2096,
  2098, 2102):** deliberate anaphora mirroring the three numbered
  contributions; keep.
