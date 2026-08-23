# JASMP submission — dataset / benchmark / re-evaluation paper

Branch: `dataset-study`. Target venue: **EURASIP Journal on Audio, Speech, and
Music Processing** (SpringerOpen, Springer Nature). APC covered by Manipal
University Jaipur.

Files added by this branch:

| File | What it is |
|---|---|
| `main_jasmp.tex` | The manuscript (Springer Nature `sn-jnl` class) |
| `refs_jasmp.bib` | Its bibliography — separate from the ICASSP paper's inline one |
| `leakage_ablation.py` | Optional experiment that turns the protocol audit into a measurement |
| `README_JASMP.md` | This file |

Nothing existing was modified. `main.tex`, `main_icassp.tex`, `results.tex`,
`report_tables.tex` and every result CSV are untouched.

---

## 1. Non-collision with the ICASSP paper

This is the constraint the paper was designed around, so it is worth being able
to answer it in one table if an editor asks.

| | **This paper (JASMP)** | **ICASSP paper (`main_icassp.tex`)** |
|---|---|---|
| Type | Benchmark + diagnosis + negative results | Method + mechanistic analysis |
| Corpora | ASVspoof2021-**DF**, LibriSpeech+TTS set, MLAAD v5 | ASVspoof2019-**LA**, In-the-Wild, Arabic ArAD, MLAAD |
| Detectors | MFCC + {GNB, LR, RF, kNN}; AttentiveSpecCNN (106,149 params) | Fine-tuned XLS-R (SSL) + 4 public third-party checkpoints |
| Central claim | The 99.93% figure is a protocol artefact; here is a leakage-free benchmark, an explainable detector, and a measured generalisation gap | Ranking transfers cross-corpus even when the threshold does not; TTA exploits that |
| Metric focus | EER + balanced acc + per-class F1 + AUC, single-seed | EER across 10 seeds, Wilcoxon + Holm |
| Cross-corpus role | The **problem** being measured | The **thing being fixed** |

No table, figure, equation, or number is shared. The overlap in *topic* is
handled explicitly in `main_jasmp.tex` §"Relation to companion work", which
cites the ICASSP manuscript as a companion study by the same authors and states
the disjointness. Do that rather than hiding it — SpringerOpen asks about
related submissions, and a declared companion is normal; an undeclared one is a
problem.

---

## 2. Building it

### Compile status

Springer does not ship `sn-jnl.cls` to CTAN, so no local TeX bundle has it and
the real class cannot be compiled here. What *was* compiled, with `tectonic`, is
the full manuscript body under a throwaway `article`-class shim that emulates
the sn-jnl interface (`\fnm`, `\sur`, `\affil`, `\abstract`, `\keywords`,
`\bmhead`, `\backmatter`) at Springer's single-column text measure (375 pt).

Result: **builds clean, 21 pages, BibTeX resolves, zero undefined references or
citations.** Every table, the algorithm float, the TikZ pipeline figure, and all
three PNG figures typeset correctly. Remaining warnings are five overfull hboxes
of 0.2–8.3 pt, which are ordinary tight lines and measure-dependent — they will
break differently under the real class.

Fixes the compile check forced, all now in `main_jasmp.tex`:

- the in-domain results table was **64 pt overfull** → `\small`, tighter
  `\tabcolsep`, `GNB+NMF transfer` → `GNB+NMF`, and the audited headline row
  marked with `$\dagger$` and glossed in the caption
- the repo URL was an unbreakable 52-character `\texttt` token → `\url` with
  `\usepackage[hyphens]{url}`
- the TikZ pipeline figure was wider than the measure → wrapped in
  `\resizebox{\textwidth}{!}{...}`, so it fits whatever the real measure is
- `\botrule` and table `\footnotetext` (sn-jnl-only) → booktabs `\bottomrule`
  plus caption notes, so the file is class-portable
- `\hyphenation` hints for `AttentiveSpecCNN` / `LibriSpeech`, which were
  forcing 14 pt overfulls
- Fig. 3's caption claimed "160 ms resolution" for a curve that `explain.py`
  interpolates onto the 401-frame input axis; corrected to say so

Reproduce the check:

```bash
tectonic -k --keep-logs shimtest.tex
```

(the shim generator is in this session's scratchpad; `main_jasmp_structurecheck.pdf`
at the repo root is its output — **shim layout, not Springer layout**, so read it
for content and structure only, never for how the submission will look)

### Real build

Build on Overleaf the way you build the other two.

1. Overleaf → **New Project → Templates** → search **"Springer Nature LaTeX
   Template"** → open as template. (This gives you `sn-jnl.cls` and the
   `sn-*.bst` files, which are what the manuscript needs and are not in this
   repo.)
2. Delete the template's `main.tex` (or rename it) and upload `main_jasmp.tex`.
3. Upload `refs_jasmp.bib`.
4. Upload the three figures from the repo root:
   `fig_method_comparison.png`, `fig_explain.png`, `fig_cross_dataset.png`.
   (The `\safefig` macro means it still compiles without them — you get a
   labelled placeholder box instead of a crash.)
5. Set the main document to `main_jasmp.tex`, compiler **pdfLaTeX**.

The document class line is:

```latex
\documentclass[sn-mathphys-num,pdflatex]{sn-jnl}
```

`sn-mathphys-num` gives numbered references, which is the right style for a
signal-processing journal. If the editor asks for Vancouver instead, swap the
option to `sn-vancouver`; nothing else changes.

---

## 3. Two decisions you should check before submitting

**Corresponding author is Dr. Sumit Srivastava** (the `\author*` marker). I set
it that way deliberately: institutional APC coverage normally requires the
corresponding author to hold the affiliation that is paying. If MUJ's
arrangement doesn't work that way, move the `*` to the first author — it's a
one-character change.

**First-author affiliation is IUST**, matching the version in
`Deepfakeaudio-DatasetStudy.pdf`. Note this differs from `main_icassp.tex`,
which lists Manipal University Jaipur for the same author. Pick one and make
both papers agree, or list both affiliations. The email
`roham_izadidoost@comp.iust.ac.ir` was reconstructed from the PDF (the
underscore is lost in text extraction) — verify it.

---

## 4. What's in the paper, and where each number comes from

Everything is traceable to a committed artefact — no number was invented.

| Claim in the paper | Source |
|---|---|
| Corpus counts (16,977 / 441,894 etc.), merged 19,251 / 595,513 | `build_manifest.py` output, recorded in `PROJECT_LOG.md` §2 and `report_tables.tex` |
| 614,764 total clips, 96.9% fake, ASVspoof = 74.2% of fakes | Arithmetic on the above |
| All 9 in-domain rows (EER / bal-acc / F1 / AUC) | `baseline_results.csv` + `PROJECT_LOG.md` §3 |
| AttentiveSpecCNN 4.81% EER / 93.84 / 93.49 / 0.99 | `PROJECT_LOG.md` §3, `report_tables.tex` |
| 106,149 parameters | Computed layer-by-layer from `model.py` |
| Feature map 128×5×25, attention resolution 160 ms | Derived from `model.py` + the front-end settings |
| Leave-one-source-out: 44.93/0.56, 57.50/0.42, 48.4% recall | `cross_dataset_results.csv` |
| The GNB→NMF degeneracy argument | Structural; `paper_baseline.py`'s own docstring records the same finding |
| The silent-decoder / "silence ⇒ real" shortcut | `deepfake_dataset.load_audio` docstring, `README.md` §Data-hygiene, `PROJECT_LOG.md` §3 |

### What the paper deliberately does *not* claim

The audit argues mechanisms **M1** (pre-split SMOTE) and **M2** (random k-fold
over a speaker-recurrent corpus) structurally, and says so — it does not put a
number on how much each one inflates a score, because that was never measured.
The Limitations section states this outright.

The decoder-failure section says "roughly half of the ASVspoof audio" and gives
qualitative evidence, because that is what was recorded. It does **not** report
a pre-fix EER, because none exists.

---

## 5. The one experiment worth running before submission

`leakage_ablation.py` closes the gap above. It re-runs the *same* classical
pipeline on the *same* features under four progressively leakier protocols and
reports both accuracy and EER for each:

```
P0  speaker-disjoint holdout          honest — matches the paper's main table
P1  random row holdout                M2 only
P2  random 5-fold                     M2, exactly as the audited paper runs it
P3  pre-split SMOTE + random 5-fold   M1 + M2, on a deliberately 4:1 pool
```

It is **CPU-only** — no GPU, no torch training, features cached after the first
pass — and it reuses `paper_baseline.py`'s feature extractor and classifiers so
the arms are genuinely comparable.

```bash
source env/bin/activate && python leakage_ablation.py --seeds 0 1 2 3 4
```

Prerequisite: `manifest_balanced.csv` must exist (it's gitignored, so regenerate
with `build_manifest.py` → `build_balanced_subset.py` if this is a fresh
checkout).

Output: `leakage_ablation_results.csv`, plus a printed "best accuracy per
protocol" line — which is the number a paper following each protocol *would have
reported*.

If you run it, send me the CSV and I'll write it up as a proper subsection with
its own table. It converts the audit from "we argue these mechanisms inflate
scores" to "we measured how much each one inflates them on our own data," which
is a materially stronger paper and the first thing a reviewer will want.

A second, much cheaper measurement worth having: the decoder-failure counts. A
few lines over `manifest.csv` — try `soundfile.read` on each file, tally
failures by `(dataset_source, label)` — would let §"Data hygiene" report an
exact figure instead of "roughly half."

---

## 6. Suggested pre-submission pass

- [ ] Compile on Overleaf, check it lands in a sensible page range (SpringerOpen
      has no hard limit, but ~14–18 typeset pages is the norm for this journal)
- [ ] Confirm affiliations and the corresponding-author choice (§3 above)
- [ ] Verify the GitHub repo URL in "Availability of data and materials" is
      public, and that the branch with this work is merged or referenced
- [ ] Decide whether to cite the ICASSP companion as arXiv (needs a preprint
      posted) or leave it as "manuscript under review" — currently the latter
- [ ] Run `leakage_ablation.py` and add §5's table (recommended)
