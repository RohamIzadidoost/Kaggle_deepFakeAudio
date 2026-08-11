# ICASSP 2027 Submission Plan

**Target:** ICASSP 2027, Toronto (May 16–21, 2027)
**Full paper deadline:** **September 16, 2026** (UTC-12)
**Today:** August 5, 2026 → **42 days**
**Internal freeze target:** **September 5, 2026** (4 weeks work + 11 days buffer)
**Manuscript:** `main_icassp.tex` (currently 4 pp. compiled, IEEEtran `conference`)
**Long version:** `main.tex` (11 pp., journal-length — not the submission artifact)

Review model is **single-anonymous** (per ICASSP 2027 editorial policies): reviewers
see author names. No anonymization work needed, and an arXiv preprint cannot break
blinding. Dual-submission rule applies to peer-reviewed venues only.

---

## 1. What a reviewer will actually reject on

A skeptical-reviewer pass over `main.tex` + `PROJECT_LOG.md` landed at **weak accept**.
The blocking objections, ranked:

| # | Objection | Severity | Fix cost |
|---|---|---|---|
| R1 | No paired significance test anywhere; "sign consistency" substituted for statistics | **High** | Zero GPU |
| R2 | Abstract leads with broad "improves cross-corpus" framing; real claim is narrower (works when source AUC ≳ 0.85) | **High** | Zero GPU |
| R3 | DANN/ASDG at 3 seeds vs. 5 for the main method — unjustified asymmetry in the head-to-head | Medium | ~5 GPU-h |
| R4 | No external SOTA anchor — is source-only 4.69% EER on ASVspoof2019 even competitive? | Medium | Zero GPU |
| R5 | Seed count (5, 4 on Arabic) too low to rule out a collapsed 6th seed for *our* method — the paper's own Tent result proves this | Medium | ~24 GPU-h |
| R6 | Citations unverified against original sources (flagged in `PROJECT_LOG.md` §10) | **Blocker** | Zero GPU |
| R7 | Table 4 text calls DANN "close" at 6.61 vs 3.36 — nearly 2× | Low | Zero GPU |
| R8 | Repetitive hedging prose ("we report this honestly" ×5+) | Low | Zero GPU |

**Key observation: five of eight fixes need no GPU at all.** The compute-bound items
(R3, R5) upgrade the paper from "weak accept" to "accept"; the zero-cost items are
what stop it sliding to "borderline". Do the free ones first and unconditionally.

---

## 2. The statistics problem — read this before planning seeds

Wilcoxon signed-rank at **n = 5 paired seeds cannot produce p < 0.05.** With all five
differences the same sign, the minimum attainable two-sided p is `2/2^5 = 0.0625`.
This is a hard combinatorial floor, not a power issue. Consequences:

- **Per-target significance claims need n ≥ 6 seeds** (`2/2^6 = 0.031`). Budget **8–10**
  so one collapsed seed does not destroy the claim.
- **Pooled-across-target claims work today.** Pairing (source, ours) over all 19
  (target, seed) runs already in `results_ext.csv` gives n = 19 → Wilcoxon can reach
  p < 0.001. **This is free and should be in the paper regardless of whether the seed
  expansion runs.**
- **Bootstrap CIs on EER are a separate, complementary test** — resample the eval clips
  (n = 5,580 on ASVspoof2019) to get a CI on a single run's EER. Answers "is this EER
  gap meaningful given the eval set size", which the seed test does not.

Do all three. They are cheap and they are exactly what R1 asks for.

---

## 3. Work packages

### P0 — Blockers (must ship, zero GPU) — Week 1

| ID | Task | Output | Est. |
|---|---|---|---|
| P0.1 | **Citation verification.** Every `\bibitem` in `main_icassp.tex` checked against the real paper: authors, title, venue, year, page range. `PROJECT_LOG.md` §10 flags DANN, AdaBN, Ben-David, In-the-Wild as never verified. `crossdomain` has no author list at all — fix or drop it. | Corrected `thebibliography` | 3 h |
| P0.2 | **Pooled Wilcoxon + bootstrap CIs.** New script `stats_tests.py` reading `results_ext.csv`: (a) Wilcoxon signed-rank source vs. ours over all 19 (target, seed) pairs; (b) same for ours vs. tent, ours vs. st_only, ours vs. DANN; (c) 10k-resample bootstrap 95% CI on per-target mean EER. | `stats_results.csv` + one sentence + a table footnote | 4 h |
| P0.3 | **Reframe abstract + conclusion.** Replace "improves EER on three of four targets" as the lead with the conditional claim: adaptation gain is predicted by source AUC (Fig. 4, r = +0.57); it delivers where ranking already transfers and is neutral where it does not. State the precondition as a *contribution*, not a caveat. | Rewritten abstract, conclusion | 3 h |
| P0.4 | **Fix Table 4 wording.** "close" → the actual multiple. Audit every comparative adjective in the results section against its number. | Diff on `main_icassp.tex` | 1 h |
| P0.5 | ~~Verify page limit.~~ **Done 2026-08-11**: confirmed on `2027.ieeeicassp.org/publishing-and-paper-presentation-options/` — "up to four pages for technical content including figures and references, and one optional fifth page containing only references." 4+1 holds. Current PDF is 4 pp. *including* refs, so there is ~1 page of headroom for P2.1/P2.4. | Confirmed budget | 0.5 h |

### P1 — Compute (upgrades the verdict) — Weeks 2–3

| ID | Task | GPU cost | Est. wall |
|---|---|---|---|
| P1.1 | **Seeds 5→10 on the main grid.** Re-run the extended pipeline for seeds 5–9, all 4 targets, all methods (source train + ours + tent + st_only + bn_only + oracle + rawnet2lite). Measured: the 5-seed run took **23 h wall** on one H200 NVL MIG 1g.35gb slice. | ~24 GPU-h | 1 day |
| P1.2 | **DANN + ASDG to 5 seeds.** DANN measured at 30 min/target-seed, ASDG at 7 min. 4 targets × 2 extra seeds each. | ~5 GPU-h | 4 h |
| P1.3 | **Re-run P0.2 statistics on 10 seeds.** Per-target Wilcoxon now attainable at p < 0.05. Update every table's ± and every seed-count mention in the text. | 0 | 3 h |
| P1.4 | **Arabic 5th seed.** Currently 4 (a cloud allocation ended). Fold into P1.1 — removes an awkward asymmetry a reviewer will notice. | included | — |

**Total P1 compute ≈ 29 GPU-hours ≈ 1.5 days on a single GPU.** This is one booking, not
a campaign.

### P2 — Strengthening (do if P0/P1 land early) — Week 4

| ID | Task | Est. |
|---|---|---|
| P2.1 | **External SOTA anchor.** One sentence + 2–3 rows citing published ASVspoof2019-LA and In-the-Wild EERs (Tak et al. wav2vec2+aug, AASIST, XLS-R+SLS are already cited). Establishes the source model is competitive in absolute terms, so the gains are meaningful and not lifting a weak baseline. | 3 h |
| P2.2 | **Prose de-hedging pass.** Collapse the repeated honesty meta-commentary to one occurrence in Limitations. Reads authentic once, defensive five times. Frees space too. | 2 h |
| P2.3 | **Scope the title/claim explicitly.** Consider making the ranking-calibration precondition part of the title rather than an internal finding. | 1 h |
| P2.4 | **Reposition the ASVspoof2021-DF failure.** Currently last in the limitations list; it is the target closest to a standard deployed protocol. Move up and frame as a scope boundary tied to class imbalance, not an afterthought. | 2 h |

### P3 — Deliberately NOT doing

- Per-target hyperparameter tuning of `q`, `λ`. `PROJECT_LOG.md` §10 says explicitly: do
  not re-litigate the LibriSpeech-TTS null by re-tuning λ. Reviewers reward the honest
  null; tuning it away invites a "why is this not overfitting to the target" question
  you cannot answer without target labels.
- Rescuing the ASVspoof2021-DF single-source regime. Already attempted (prevalence
  estimator), already failed, already documented. Out of scope for 4 pages.
- New targets or a 5th corpus. Data acquisition + source retraining does not fit 6 weeks
  alongside everything above.
- Re-running the hyperparameter sweep (`sec:sweep`) on 4 targets. Reported honestly as a
  2-target seed-0 artifact; leave it.

---

## 4. Schedule

| Week | Dates | Focus | Exit criterion |
|---|---|---|---|
| **1** | Aug 5–11 | P0 in full. Book GPU for week 2. | Citations verified; pooled Wilcoxon in the manuscript; abstract reframed; page budget confirmed |
| **2** | Aug 12–18 | P1.1 + P1.2 launch and complete. Monitor via `tail -f run_log_ext.txt`. | `results_ext.csv` has 10 seeds × 4 targets; DANN/ASDG at 5 |
| **3** | Aug 19–25 | P1.3 — regenerate every table, figure, and ± from the 10-seed data. Re-run `fig_auc_gain.png` (n grows 19 → 39 points). | All numbers in `main_icassp.tex` trace to the 10-seed CSVs |
| **4** | Aug 26–Sep 1 | P2. Full-paper editing pass. Fit to page limit. | Compiles at exactly the page limit, no overfull boxes |
| **5** | Sep 2–5 | Co-author review (Dr. Srivastava, Shweta Sharma). Final proofread. **Freeze Sep 5.** | Sign-off from both co-authors |
| **buffer** | Sep 6–16 | Submission mechanics, IEEE PDF eXpress validation, contingency | Submitted ≥ 3 days early |

**Do not** plan to use the buffer. IEEE PDF eXpress rejections for font embedding are
routine and cost a day.

---

## 5. Resources needed

### Compute
| Item | Spec | Note |
|---|---|---|
| GPU | 1× ≥24 GB (H200 MIG 1g.35gb slice is proven sufficient) | measured ~4 GB actual for the method |
| GPU-hours | **~29** | P1.1 24h + P1.2 5h |
| CPU | 16 cores | audio decode + ffmpeg is the bottleneck, not the GPU |
| RAM | 64 GB | 32 GB minimum |
| Storage | 150 GB | datasets ~65 GB + env ~20 GB + working ~10 GB + headroom |
| Booking | One contiguous ~36 h window | the 23 h run should not be interrupted |

### Data (must be present or re-downloadable on the instance)
`asvspoof2019_LA`, `in_the_wild`, `dataset_2`, `mlaad` (38 languages), `arabic_arad`.
The pipeline logs `already present:` when cached — verify before burning GPU time on
downloads.

### Checkpoints
`ckpt_ext/` (large, not committed). Seeds 0–4 source models exist and are reused.
**Seeds 5–9 need fresh source training** (~7 min/model × 4 targets × 5 seeds ≈ 2.5 h,
already inside the 24 h estimate). Confirm `ckpt_ext/` is on the instance or the run
re-trains seeds 0–4 unnecessarily.

### People
- **Roham Izadidoost** — all P0/P1/P2 execution
- **Dr. Sumit Srivastava, Shweta Sharma** — week 5 review; give them the draft by **Sep 1**, not Sep 4
- Optional: one external reader outside the project for the abstract-clarity check

### Accounts / access
- Cloud GPU allocation (the Arabic 4-seed gap came from an allocation ending — confirm the
  new booking covers the full window)
- ICASSP 2027 submission portal account (create in week 4, not week 5)
- IEEE PDF eXpress access for the final PDF check

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| A new seed collapses for *our* method | Medium | High — kills the core stability claim | Report it. Tent's single-seed collapse is already reported and *strengthened* the paper. Hiding it is worse than having it. |
| 10-seed run changes a headline number | Medium | Medium | Every table regenerates from CSVs; budget week 3 fully for this. Do not hand-copy numbers. |
| GPU booking slips | Medium | High | P0 is GPU-free and worth a full week. If compute never lands, the paper still submits with pooled-Wilcoxon statistics on 5 seeds — weaker but complete. |
| Page limit is 4 total, not 4+1 | Low | High | P0.5 resolves this in week 1. If 4 total, P2.1 (SOTA table) becomes one inline sentence and P2.2 becomes mandatory. |
| Citation check finds a fabricated or wrong reference | Medium | **Critical** | This is exactly why P0.1 is week 1 and not week 4. `PROJECT_LOG.md` §10 explicitly flags these as unverified. |
| Co-authors unavailable in early September | Medium | Medium | Send the draft Sep 1; ask for availability confirmation in week 1. |

---

## 7. Submission checklist

- [ ] Page limit confirmed and met, references within their allowance
- [ ] IEEEtran `conference` class, no `\pagestyle` hacks, no overfull hboxes
- [ ] Every `\bibitem` verified against the original source (P0.1)
- [ ] Every number in every table traceable to a committed CSV
- [ ] Seed counts stated consistently in abstract, setup, and every caption
- [ ] Significance test reported with the test named and n given
- [ ] Figures legible at print size and in greyscale
- [ ] Author list, affiliations, emails correct (single-anonymous — names *stay in*)
- [ ] Originality/no-simultaneous-submission declaration true
- [ ] IEEE PDF eXpress validation passed
- [ ] arXiv posting decision made (permitted; no blinding conflict)
- [ ] `PROJECT_LOG.md` updated with the 10-seed results and final framing

---

## 8. If you only do three things

1. **P0.1 — verify the citations.** A fabricated reference is the one failure mode that
   ends the paper regardless of merit.
2. **P0.2 — pooled Wilcoxon over the 19 existing runs.** Zero GPU, kills the single most
   damaging reviewer objection, available today.
3. **P0.3 — reframe the abstract around the precondition.** The narrower claim is the one
   the evidence actually supports, and it is a *more* interesting claim than the broad one.
