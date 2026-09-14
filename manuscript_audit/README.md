# ICASSP manuscript correction and score audit

The revised paper is `../main_icassp.tex`, compiled as `../main_icassp.pdf` and
copied to `../izadidoost.pdf`. The previous PDFs and source are in `original/`.
No new model training, target-label adaptation, or neural-network inference was
performed for this revision. All original score arrays and historical result
CSVs were retained unchanged.

## Reproduce

From the repository root, use a Python environment containing NumPy, pandas,
SciPy, and scikit-learn. The exact audit versions are in `provenance.json`.
This revision was verified with an isolated environment at
`/tmp/icassp-audit-env`, without changing the project's pinned training environment.

```bash
python -m unittest test_metrics
python audit_icassp.py
pdflatex -interaction=nonstopmode -halt-on-error main_icassp.tex
pdflatex -interaction=nonstopmode -halt-on-error main_icassp.tex
```

The audit needs the existing `.npy` files in `scores_protocol_a_public/` and
`scores_protocol_a_public_itw/`, their historical result CSVs, the cached source
confusion matrices, the official DF metadata, and the original local audio file
names. It does not decode audio or import PyTorch. It reconstructs the sorted
manifests and the original seeded sampling, then verifies legacy EER, AUC and
accuracy against historical rows before reporting new metrics. There are 89
audited score arrays and 157 full-pool/disjoint metric rows. Score, metadata,
manifest, script and auxiliary input hashes are recorded in `provenance.json`.

`spconf.sty` was downloaded without modification from the official ICASSP 2027
paper kit: https://cmsworkshops.com/ICASSP2027/papers/PaperFormat/spconf.sty.

## Corrections supported by the audit

| Issue | Verified correction |
|---|---|
| DF denominator | 611,829 covers **all phases**; eval alone has 533,928 trials. The available 400,435 eval clips are 74.9979% of eval. |
| Available labels | 389,299 spoof and 11,136 bona fide, or 97.2190% spoof. |
| Selection representativeness | Class coverage is similar (75.00% spoof, 74.89% bona fide); this does not prove an unbiased full-set EER. The revised paper makes no full-set/SOTA comparison. |
| Tent saturation | 98.0723% are at least `1-1e-6`; 93.0216% are **exactly** 1.0. The previous wording confused these. Its AUC 0.9765 is reproduced. |
| ETA saturation | 97.7080% are near one; 96.6459% are exactly one. AUC 0.9288 is reproduced. |
| EER computation | Replaced the nearest-ROC-point approximation with the interpolated FPR/FNR crossing. Exact ties move together. |
| DF-2 EER changes | Source 4.5101 → 4.5131; BBSE budget 4.0202 → 4.0600; Tent-style 4.0615 → 4.2771; ETA 7.4440 → 12.3110. All values are percentages. |
| Fixed-confidence AUC | Saved scores give 0.9918, correcting the manuscript's 0.9924. |
| Threshold deficit | Replaced `(100-EER)-accuracy` with exact best-threshold accuracy minus actual accuracy. The maximization uses labels and is explicitly an oracle diagnostic. |
| Threshold-only controls | Added median and BBSE-quantile thresholds selected from frozen-source adaptation scores, evaluated on the full and disjoint pools. No weights are updated. |
| Purity theorem | Applies to **realized** bucket size; perfect-purity feasibility is not a guarantee of useful or safe adaptation. |
| Quantile implementation | Inclusive comparisons can select more clips than nominal `q`; fake assignments overwrite overlaps. Existing runs were not converted into exact-count selection after the fact. |
| Loss equation | CE averages over selected examples per mini-batch; consistency averages over all examples and both class probabilities. The revised equation matches that implementation. |
| Baseline fidelity | Entropy is Tent-style under restricted parameters; ETA removes redundancy and Fisher terms; SAR uses a scaled reset; the probability-space IM-PL baseline is SHOT-inspired, not original feature-space SHOT. Its head is frozen (16,384 parameters, versus 16,706 for the others). |
| Repeated runs | Three checkpoints × three adaptation runs are described as sensitivity checks; overlapping data and reused checkpoints are not treated as nine independent datasets. P-values were removed. |
| Prior estimation | BBSE can work with uncalibrated predictors under label shift and an invertible confusion matrix. Unrestricted conditional shift invalidates that identification assumption; it does not imply that every output-based estimator is impossible. |
| Oracle prior | Improves DF-42 on In-the-Wild but is not universally best (WF is a counterexample). |
| Minority-class tradeoff | DF-2 BBSE accuracy is 98.95% versus source 98.87%, but balanced accuracy falls from 88.28% to 82.34%. Both are reported. |
| Disjoint DF-240 | BBSE EER 3.8330% versus source 3.8298%: no gain. The revised paper does not claim universal source improvement. |
| Small-q safety | At 97% spoof on resampled ITW, q=.02 gives 98.59% accuracy versus source 98.94%. It avoids the large collapse, not all harm. |
| Prevalence intervention | Thinning real clips changes pool size; all resampled pools are under 20,000 clips and evaluated transductively. |
| Related work | Added Label Shift Adapter, RLSbench, and the DeepFense framework paper. Removed claims that the field never studies imbalance. |

The earlier review's fifth-page concern needs qualification: the general author
guidelines say references only, but the detailed 2027 paper kit explicitly
permits funding acknowledgements and ethical-compliance statements there as
well. The revised PDF places all discussion and assistance/ethics text within
four pages, followed by a references-only fifth page, satisfying either wording.
Sources:
https://2027.ieeeicassp.org/author-guidelines/ and
https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php.

## Output files

- `scores.csv`: recomputed EER, legacy EER, AUC, accuracy, balanced accuracy,
  FPR/FNR, exact oracle accuracy/gap, and score-resolution statistics.
- `threshold_controls.csv`: frozen-source controls; thresholds are chosen only
  from adaptation scores and cached source confusion matrices.
- `initial_tails.csv`: realized initial selection sizes, label-based purity and
  counting bounds in a **cached-score replay**. These are not recovered original
  adaptation-time masks. Small numerical differences between scoring passes can
  change quantile boundaries. BBSE values here use the cached matrix; the main
  adaptation results retain their original trained weights and scores.
- `coverage.csv`: official and available metadata counts by phase and class.
- `table_*.tex`: tables generated from the recomputed results.
- `provenance.json`: input hashes, manifest hashes, counts and library versions.

## What remains scientifically unresolved

The revised manuscript is a narrower, auditable study, not a declaration that
the method is submission-ready or that every reviewer concern has been solved.
The following require new experiments or additional data:

1. Full DF eval coverage and independent architectures/corpora.
2. Properly tuned and faithful published baselines, including methods designed
   for joint label/conditional shift. Renaming existing variants does not replace
   those comparisons.
3. Fixed-size, repeated prevalence interventions to separate prevalence from
   sample-count effects; disjoint evaluation at each prevalence.
4. Exact-count tail selection or logged realized masks at every adaptation
   epoch, with all affected adaptation experiments rerun.
5. Held-out source calibration/confusion estimation and stronger uncertainty
   analysis accounting for checkpoint, corpus and recording dependence.
6. Reanalysis of the older 369-cell breadth grid with a defensible threshold
   diagnostic and corrected EER. Aggregate historical CSVs alone cannot recover
   exact ROC crossings or best-threshold accuracy. This grid, the proposed
   automatic guards, and the calibration-error correlation were omitted from
   the revised short paper; their original artifacts remain available.

Changing `metrics.compute_eer` affects **future** evaluations throughout the
project. Historical CSVs, the long paper, old plots and old reports still use
the previous estimator and must not be silently mixed with this audit. Future
`protocol_a_public.report` rows use explicit `eer_operating_*` names for the old
signed diagnostic and `oracle_accuracy`/`threshold_gap` for the correct one.
The new scripts and outputs are local; they have not been pushed or submitted.
