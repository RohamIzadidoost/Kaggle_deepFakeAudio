# JASMP submission — dataset / benchmark / re-evaluation paper

Branch: `dataset-study`. Target venue: **EURASIP Journal on Audio, Speech, and
Music Processing** (SpringerOpen, Springer Nature). APC covered by Manipal
University Jaipur.

Files added by this branch:

| File | What it is |
|---|---|
| `main_jasmp.tex` | The manuscript (Springer Nature `sn-jnl` class) |
| `refs_jasmp.bib` | Its bibliography — separate from the ICASSP paper's inline one |
| `leakage_ablation.py` | The protocol-inflation ablation — run, results in §5 |
| `make_leakage_table.py` | Generates `tab_leakage.tex` from the ablation CSV |
| `leakage_ablation_results.csv` | Raw per-seed/fold/classifier ablation output |
| `build_jasmp/sn-jnl.cls` | Structure-check compile shim (§2) |
| `README_JASMP.md` | This file |

`main.tex`, `main_icassp.tex`, `results.tex`, `report_tables.tex`, and every
result CSV belonging to the ICASSP paper are untouched. One shared file *was*
touched: `deepfake_dataset.py`, to fix a dead decoder fallback the ablation run
caught (§5) — that fix lands as its own commit, separate from the paper
changes, since every other pipeline using that module benefits from it too.

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
the real class cannot be compiled here. What *is* compiled, with `pdflatex` +
`bibtex`, is the full manuscript body under a throwaway `article`-class shim
(`build_jasmp/sn-jnl.cls`, checked into the repo) that emulates the sn-jnl
interface (`\fnm`, `\sur`, `\affil`, `\abstract`, `\keywords`, `\bmhead`,
`\backmatter`, starred `\author*`) at Springer's single-column text measure
(375 pt).

Result: **builds clean, 23 pages, BibTeX resolves, zero undefined references or
citations.** Every table (including the generated leakage-ablation table,
§5), the algorithm float, the TikZ pipeline figure, and all three PNG figures
typeset correctly. Remaining warnings are six overfull hboxes of 0.4–8.3 pt,
ordinary tight lines and measure-dependent — they will break differently under
the real class.

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
BIBINPUTS=.. bash -c 'cd build_jasmp && bibtex main_jasmp'
TEXINPUTS="./build_jasmp:" pdflatex -interaction=nonstopmode -output-directory=build_jasmp main_jasmp.tex
TEXINPUTS="./build_jasmp:" pdflatex -interaction=nonstopmode -output-directory=build_jasmp main_jasmp.tex
```

`build_jasmp/main_jasmp.pdf` is the output — **shim layout, not Springer
layout**, so read it for content and structure only, never for how the
submission will look. `build_jasmp/` is gitignored except for `sn-jnl.cls`
itself, which is checked in so the check is reproducible without regenerating
the shim from scratch.

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
| The silent-decoder / "silence ⇒ real" shortcut, and the exact per-source/per-label failure counts | `deepfake_dataset.load_audio` docstring, `README.md` §Data-hygiene, `PROJECT_LOG.md` §3; exact counts in `decoder_audit.csv` |
| Protocol-inflation table (P0–P3, 5-seed mean±s.d.) | `leakage_ablation_results.csv`, via `make_leakage_table.py` → `tab_leakage.tex` |

### What the paper deliberately does *not* claim

The audit argues mechanism **M1** (pre-split SMOTE) structurally, and its
marginal contribution on top of M2 is **not** cleanly isolated by the ablation
in §5 — P2 and P3 differ in pool composition as well as resampling, so the
paper reports the confound rather than a clean number for M1 alone. M2's
contribution *is* now measured (§5), which is new since the previous version
of this README.

The decoder-failure section previously said "roughly half of the ASVspoof
audio" from qualitative evidence; it now reports the exact cross-tabulation
(§5 table, and Table "Clips libsndfile fails to decode" in the paper). It still
does **not** report a pre-fix EER, because none exists — the fix was applied
before any detector was trained on this benchmark.

---

## 5. The leakage ablation — run, and one real bug it caught

`leakage_ablation.py` ran on the machine that holds the corpora, 5 seeds
(0–4), all four protocols. Results: `leakage_ablation_results.csv` (raw,
per-seed-per-fold-per-classifier) and `tab_leakage.tex` (the generated table,
via `make_leakage_table.py` — regenerate rather than hand-edit if the CSV
changes). Written up in §"Measuring the inflation mechanisms" in
`main_jasmp.tex`, and reflected in the abstract and Limitations.

Headline: at a matched train fraction, letting speakers recur across the split
(P0→P1) moves accuracy 84.21%→94.64% and EER 14.06%→3.71% (5-seed
mean±s.d.), with nothing else changed — the paper's strongest evidence for M2,
because it's a controlled measurement rather than a structural argument. It
also changes which model wins (RF honest → KNC leaky) and makes the honest
protocol look *noisier* across seeds than the leaky ones, which is itself
worth knowing. M1's marginal contribution on top of M2 (P2→P3) came out
**not** monotonic — P3 was slightly below P2 on both metrics — because P2 and
P3 don't share a pool (P3 replaces 75% of its real class with SMOTE
interpolants). The paper reports this plainly rather than smoothing it into a
clean story; see the "M1 on top of M2 does not stack cleanly here" paragraph
for the reasoning and what a clean isolation would need.

**Before trusting any of it, the run caught a real bug**, which is exactly why
this was worth running rather than just writing up as planned. `leakage_ablation.py`'s
preflight checked `import librosa` and passed — but an empty leftover
`librosa/` directory in `env/`'s `site-packages` satisfies that import as a
Python 3.14 namespace package without providing `librosa.load`. The decoder
fallback in `deepfake_dataset.py` therefore raised `AttributeError` on every
file `soundfile` couldn't decode, which a bare `except Exception:` swallowed,
silently zero-filling **31% of the benchmark** (43.9% of real clips, 18.8% of
fake — the exact "silence ⇒ real" shortcut §"Data hygiene" warns about). Fixed
in `deepfake_dataset.py` by shelling out to FFmpeg directly instead of via
librosa (kept out of `requirements.txt` deliberately — it drags in numba
against the pinned numpy 1.26.4); preflight now checks for the `ffmpeg` binary
instead of the hollow import. Verified 0/38,502 zero-feature clips post-fix.
That fix is a **separate commit** from the paper changes, since it touches
`deepfake_dataset.py`, which every other pipeline (`train.py` included) also
uses.

The exact pre-fix decoder failure count, cross-tabulated by source and label,
is now in the paper too (Table "Clips libsndfile fails to decode", §"Data
hygiene"), replacing the old "roughly half" estimate:

| Source | real failed/total | fake failed/total |
|---|---|---|
| ASVspoof 2021 DF | 8,456/16,977 (49.8%) | 3,613/8,539 (42.3%) |
| LibriSpeech+TTS | 0/2,274 | 0/2,173 |
| MLAAD v5 | — | 0/8,539 |
| **All** | **8,456/19,251 (43.9%)** | **3,613/19,251 (18.8%)** |

To reproduce or extend:

```bash
python leakage_ablation.py --seeds 0 1 2 3 4    # ~75 min, cache warm after seed 0
python make_leakage_table.py                     # regenerate tab_leakage.tex from the CSV
```

A genuinely open extension, not done here: isolating M1's marginal
contribution cleanly needs a fifth arm that holds the imbalanced-pool
subsampling fixed and varies only whether SMOTE runs before or after the
split (P2 and P3 currently differ in both the resampling *and* the pool).

---

## 6. Suggested pre-submission pass

- [ ] Compile on Overleaf, check it lands in a sensible page range (SpringerOpen
      has no hard limit, but ~14–18 typeset pages is the norm for this journal)
- [ ] Confirm affiliations and the corresponding-author choice (§3 above)
- [ ] Verify the GitHub repo URL in "Availability of data and materials" is
      public, and that the branch with this work is merged or referenced
- [ ] Decide whether to cite the ICASSP companion as arXiv (needs a preprint
      posted) or leave it as "manuscript under review" — currently the latter
- [x] Run `leakage_ablation.py` and add §5's table — done; see §5 above for
      the headline numbers and the decoder bug the run caught
