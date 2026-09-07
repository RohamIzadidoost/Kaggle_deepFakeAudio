"""Job queue for the autonomous run, ordered by scientific value.

Read by orchestrator.py. Ordering rule: if the GPU is lost at any point, the
work already done should be the work most worth having. So the first stage is
the one that decides whether the paper's main table changes, not the one that
adds breadth.

Checkpoint inventory after this run's additions (9 checkpoints, 5 families):
  XLS-R+AASIST   ssl_aasist_wavefake            WaveFake/LJSpeech
  XLS-R+AASIST   deepfense_w2v2_aasist_s{2,42,240}  ASVspoof2019 (siblings)
  w2v2+seqcls    hf_xlsr_gustking               undocumented (partial)
  w2v2+seqcls    hf_xlsr_stafford               commercial TTS vendors
  w2v2-base      hf_w2v2_mothecreator           undocumented
  w2v2-base      hf_w2v2_bisher                 undocumented
  AST            hf_ast_asv19                   ASVspoof2019   <- different input modality

Corpora (7): asvspoof2019, dataset2, in_the_wild, arabic, asvspoof2021df,
asvspoof2021la, asvspoof2021pa(replay -- different task, extreme-OOD arm only).
"""

PY = "python"
OUT = "--out results_public_ckpt_multi.csv --log run_log_multi2.txt"

ALL_CKPTS = ["ssl_aasist_wavefake", "deepfense_w2v2_aasist_s42",
             "deepfense_w2v2_aasist_s2", "deepfense_w2v2_aasist_s240",
             "hf_xlsr_gustking", "hf_xlsr_stafford",
             "hf_w2v2_mothecreator", "hf_w2v2_bisher", "hf_ast_asv19"]

# Corpora each checkpoint may be scored on. asvspoof2019 is excluded wherever the
# checkpoint was (or may have been) fitted on it -- public_ckpt_tta.py enforces
# this independently, this list just avoids queuing jobs that would be refused.
TRAINED_ON_ASV19 = {"deepfense_w2v2_aasist_s42", "deepfense_w2v2_aasist_s2",
                    "deepfense_w2v2_aasist_s240", "hf_xlsr_gustking",
                    "hf_xlsr_stafford", "hf_w2v2_bisher", "hf_ast_asv19"}
CORPORA = ["arabic", "dataset2", "in_the_wild", "asvspoof2021df", "asvspoof2021la"]


def targets_for(ck):
    t = list(CORPORA)
    if ck not in TRAINED_ON_ASV19:
        t.insert(0, "asvspoof2019")
    return t


JOBS = []


def add(name, cmd, est_min, tag):
    JOBS.append(dict(name=name, cmd=cmd, est_min=est_min, tag=tag))


# ---------------------------------------------------------------- stage B
# Port gates for the five new checkpoints: polarity on a corpus where each
# actually ranks. Cheap, and nothing downstream is trustworthy without them.
for ck in ["hf_xlsr_gustking", "hf_xlsr_stafford", "hf_w2v2_mothecreator",
           "hf_w2v2_bisher", "hf_ast_asv19"]:
    add(f"B-signcheck-{ck}",
        f"{PY} public_ckpt_tta.py --mode signcheck --ckpt {ck} "
        f"--manifest manifest_tgt_in_the_wild_seed0.csv --target in_the_wild "
        f"--limit 400 --batch 8 --log run_log_hfport.txt",
        4, "stageB")

# ---------------------------------------------------------------- stage A
# Our OWN model's E curve. This is the only experiment that can change the
# paper's main table, so it runs first. E=4 is already recorded in
# results_ext.csv; this adds 8/16/32/64 to map the shape. If the curve
# plateaus, "spend a fixed step budget" is a label-free recommendation; if it
# peaks and falls, locating the peak would need target labels and the whole
# finding stays an observation. Seed 0 first, for the shape at 1/3 the cost.
for seeds, est in (("0", 430), ("1,2", 860)):
    add(f"A-our-esweep-seed{seeds.replace(',','_')}",
        f'CACHE_ON_CPU=1 ADAPTIVE_SMOKE=0 OUR_SEEDS={seeds} OUR_E_SWEEP="8,16,32,64" '
        f'{PY} adaptive_pipeline.py >> our_esweep.out 2>&1',
        est, "stageA")

# ---------------------------------------------------------------- stage C
# Breadth: every checkpoint x every corpus at the published E=4, seed 0. This is
# what turns the calibration-deficit result from 2 model families into 5, and
# from 5 corpora into 7. Seed 0 only -- breadth first, depth in stage D.
for ck in ALL_CKPTS:
    for tgt in targets_for(ck):
        add(f"C-{ck}-{tgt}-s0",
            f"{PY} public_ckpt_tta.py --mode ours --ckpt {ck} "
            f"--manifest manifest_tgt_{tgt}_seed0.csv --target {tgt} --seed 0 "
            f"--batch 16 {OUT}",
            12, "stageC")

# ---------------------------------------------------------------- stage D
# Depth: seeds 1-9, so per-cell Wilcoxon is possible at all. n=10 paired seeds is
# the minimum that can reach p<0.05 (2/2^10 = 0.002); at n=5 the combinatorial
# floor is 0.0625 and NO result can be significant however large the effect.
#
# Restricted to a CORE subset rather than all ~50 cells: ten seeds everywhere is
# ~92 GPU-hours by itself, more than the whole budget. The split is deliberate --
# pooled claims (the calibration-deficit result) are strengthened by *breadth*,
# which stage C buys at seed 0, whereas only per-cell significance needs depth.
# The core spans one representative per model family x three corpora chosen to
# cover the source-AUC range (in_the_wild high, arabic middle, dataset2 the null).
CORE_CKPTS = ["ssl_aasist_wavefake",        # XLS-R+AASIST, WaveFake-trained
              "deepfense_w2v2_aasist_s42",  # XLS-R+AASIST, ASV19-trained
              "hf_xlsr_stafford",           # w2v2+seqcls, commercial-TTS-trained
              "hf_w2v2_mothecreator",       # w2v2-base+seqcls
              "hf_ast_asv19"]               # AST -- different input modality
CORE_CORPORA = ["in_the_wild", "arabic", "dataset2"]

for ck in CORE_CKPTS:
    for tgt in CORE_CORPORA:
        for s in range(1, 10):
            add(f"D-{ck}-{tgt}-s{s}",
                f"{PY} public_ckpt_tta.py --mode ours --ckpt {ck} "
                f"--manifest manifest_tgt_{tgt}_seed{s}.csv --target {tgt} --seed {s} "
                f"--batch 16 {OUT}",
                12, "stageD")

# ---------------------------------------------------------------- stage E
# Baselines on the new checkpoints: Tent (the load-bearing contrast) and
# st_only (the component decomposition), seed 0. Only on the new families --
# the original four already have these from the overnight run.
for ck in ["hf_xlsr_gustking", "hf_xlsr_stafford", "hf_w2v2_mothecreator",
           "hf_w2v2_bisher", "hf_ast_asv19"]:
    for tgt in targets_for(ck):
        for mode in ("tent", "st_only"):
            add(f"E-{mode}-{ck}-{tgt}-s0",
                f"{PY} public_ckpt_tta.py --mode {mode} --ckpt {ck} "
                f"--manifest manifest_tgt_{tgt}_seed0.csv --target {tgt} --seed 0 "
                f"--batch 16 {OUT}",
                12, "stageE")

# ---------------------------------------------------------------- stage F
# The replay arm. Different task (a real voice replayed, not synthesised), so
# every model should be near chance. Kept separate from every pooled statistic;
# its job is to answer whether adaptation is inert or harmful when there is no
# ranking to exploit.
for ck in ALL_CKPTS:
    add(f"F-replay-{ck}-s0",
        f"{PY} public_ckpt_tta.py --mode ours --ckpt {ck} "
        f"--manifest manifest_tgt_asvspoof2021pa_seed0.csv --target asvspoof2021pa "
        f"--seed 0 --batch 16 {OUT}",
        12, "stageF")

# ---------------------------------------------------------------- stage G
# Does the E-curve/ranking interaction found on OUR model reproduce on models we
# did not train? Stage A showed the shape is target-dependent: more epochs helped
# on three targets (arabic +1.85 -> +6.43 from E=4 to E=64) and monotonically
# HURT on dataset2, the one target with the weakest source ranking. If that is a
# real mechanism -- more epochs amplify whatever the pseudo-labels say, so good
# ranking compounds and bad ranking compounds too -- it must show up here as
# well, and the sign of the E slope should track source AUC.
#
# Cells chosen to span measured source AUC from 0.585 to 0.986, using stage C's
# actual numbers rather than guesses. Prediction to be falsified: the two lowest
# cells degrade with E, the two highest improve or plateau.
G_CELLS = [
    ("deepfense_w2v2_aasist_s42", "dataset2"),     # src AUC 0.585 -- expect harm
    ("hf_w2v2_bisher",            "dataset2"),     # 0.623, and a +13.83 outlier at E=4
    ("ssl_aasist_wavefake",       "dataset2"),     # 0.734
    ("deepfense_w2v2_aasist_s42", "in_the_wild"),  # 0.931
    ("ssl_aasist_wavefake",       "in_the_wild"),  # 0.986 -- expect plateau (no headroom)
]
for ck, tgt in G_CELLS:
    for E in (16, 32):
        add(f"G-E{E}-{ck}-{tgt}-s0",
            f"{PY} public_ckpt_tta.py --mode ours --ckpt {ck} "
            f"--manifest manifest_tgt_{tgt}_seed0.csv --target {tgt} --seed 0 "
            f"--batch 16 --tta_epochs {E} {OUT}",
            48 if E == 16 else 96, "stageG")

# ---------------------------------------------------------------- stage G2
# The 10-seed tests found FOUR significant degradations at E=4, the largest being
# hf_ast_asv19 on in_the_wild (-3.79 EER, p=0.002 Holm-adjusted) -- a cell where
# the threshold is repaired (32/32 on deficit) while ranking gets worse. If the
# "more epochs amplify whatever the pseudo-labels say" mechanism is right, raising
# E here must make the degradation LARGER, not smaller. That is a falsifiable
# prediction and the cheapest way to test the mechanism rather than just describe it.
for ck, tgt in [("hf_ast_asv19", "in_the_wild"), ("hf_ast_asv19", "dataset2")]:
    for E in (16, 32):
        add(f"G2-E{E}-{ck}-{tgt}-s0",
            f"{PY} public_ckpt_tta.py --mode ours --ckpt {ck} "
            f"--manifest manifest_tgt_{tgt}_seed0.csv --target {tgt} --seed 0 "
            f"--batch 16 --tta_epochs {E} {OUT}",
            40 if E == 16 else 80, "stageG2")

# ================= 2026-08-31 evening: our own model at ten seeds =============
# The main method has been stuck at n=3 locally because only ckpt_ext seeds 0-2
# survived the cloud. Source training turns out to cost 3.64 GiB peak and ~6 min
# per model -- XLSRDetector freezes the encoder and trains only the top-4
# transformer layers (50.6M of 315.7M params), so it was never the 300M
# fine-tune that was assumed not to fit. Gated on train_source_local.py
# --fidelity reproducing the cloud checkpoints first.

# --- stage H: train the missing source checkpoints, one job per seed ---------
for _s in range(3, 10):
    add(f"H-train-source-seed{_s}",
        f"CACHE_ON_CPU=1 ADAPTIVE_SMOKE=0 {PY} train_source_local.py "
        f"--seeds {_s} >> train_source.out 2>&1",
        26, "stageH")

# --- stage I: the PUBLISHED config at the new seeds --------------------------
# OUR_E_SWEEP="4" gives exactly {source, ours_E4}: the published E with no
# adaptive arms. This is what extends the paper's own main table from the three
# seeds we can currently verify locally to ten, on checkpoints we hold.
for _s in range(3, 10):
    add(f"I-ours-E4-seed{_s}",
        f'CACHE_ON_CPU=1 ADAPTIVE_SMOKE=0 OUR_SEEDS={_s} OUR_E_SWEEP="4" '
        f'{PY} adaptive_pipeline.py >> our_e4_10seed.out 2>&1',
        18, "stageI")

# --- stage J: E=32 at ten seeds, on the two decisive targets -----------------
# A full E sweep at ten seeds is 66 GPU-hours; E=32 alone on every target is
# 17.7. Restricted to arabic (largest E gain: +1.85 at E=4 -> +6.43 at E=64) and
# dataset2 (the one target that degrades monotonically with E). Those two carry
# the positive and the negative case, so a per-target Wilcoxon at n=10 on both is
# what turns the E finding from an observation into a result.
for _s in range(3, 10):
    add(f"J-E32-seed{_s}",
        f'CACHE_ON_CPU=1 ADAPTIVE_SMOKE=0 OUR_SEEDS={_s} OUR_TARGETS=arabic,dataset2 '
        f'OUR_E_SWEEP="32" {PY} adaptive_pipeline.py >> our_e32_10seed.out 2>&1',
        80, "stageJ")

# ================= 2026-09-01: closing the fairness hole, and more corpora =====

# --- stage K: baselines at MATCHED budget ------------------------------------
# We now claim E=32 beats the published E=4 for our method. That hands it 8x the
# gradient updates, so the obvious reviewer question is whether the baselines
# improve just as much on the same budget. Until they are run at matched E the
# comparison is confounded, which makes this a correctness fix rather than an
# extra result -- hence it runs first.
#
# Restricted to arabic and dataset2, the two targets the E claim is made on.
for _s in range(0, 10):
    add(f"K-baselines-E4E32-seed{_s}",
        f'CACHE_ON_CPU=1 ADAPTIVE_SMOKE=0 OUR_SEEDS={_s} OUR_TARGETS=arabic,dataset2 '
        f'OUR_BASELINE_E="4,32" {PY} adaptive_pipeline.py >> baselines_matched.out 2>&1',
        145, "stageK")

# --- stage M: finish the E curve at ten seeds on the other two targets -------
for _s in range(0, 10):
    add(f"M-E32-rest-seed{_s}",
        f'CACHE_ON_CPU=1 ADAPTIVE_SMOKE=0 OUR_SEEDS={_s} '
        f'OUR_TARGETS=asvspoof2019,in_the_wild OUR_E_SWEEP="32" '
        f'{PY} adaptive_pipeline.py >> our_e32_rest.out 2>&1',
        72, "stageM")

# --- stage L: three more targets for our own model, ten seeds ----------------
# Addresses the paper's stated scale limitation directly ("Seeds do not buy
# corpora"). Source training is now ~6.6 min, so one shared source model per seed
# plus TTA on three targets costs ~23 min/seed. See run_new_targets.sh for the
# protocol and the condition-shift caveat that must accompany these numbers.
for _s in range(0, 10):
    add(f"L-new-targets-seed{_s}",
        f"SEED={_s} ./run_new_targets.sh >> new_targets.out 2>&1",
        23, "stageL")

# --- stage N: is there a LABEL-FREE proxy for the calibration deficit? -------
# The deficit predicts the adaptation gain (r=+0.56, rho=+0.69 over 70 cells)
# where source AUC does not (r=-0.12). But it is computed from EER and accuracy,
# so it needs labels and can only explain results after the fact. If a statistic
# of the score distribution alone tracks it, the paper gets an actionable rule --
# measure this, then decide whether to adapt -- instead of a post-hoc account.
# Scoring only; no training, no adaptation.
add("N-calibration-proxy",
    "CACHE_ON_CPU=1 ADAPTIVE_SMOKE=0 INCLUDE_MLAAD=1 NEW_TARGETS=1 "
    "python probe_calibration_proxy.py >> calib_proxy.out 2>&1",
    45, "stageN")

# ================= 2026-09-02 overnight: mechanism, not more data ============

# --- stage P: the skew experiment -------------------------------------------
# Phase 1 found that a one-line label-free rule -- threshold at the median score
# -- recovers 94% of the full method's accuracy gain (4.6 of 4.9 pts; TTA wins by
# only 0.32, p=0.014, against an oracle ceiling 0.99 above TTA). The natural
# question is WHY they nearly coincide.
#
# Hypothesis: both assume a balanced pool. The median predicts exactly 50%
# positive; q=0.3 takes the top and bottom 30% as pseudo-fake/real regardless of
# the true prior. If that is the shared mechanism, skewing the pool should break
# BOTH together -- and it would explain the Protocol A collapse (26.33 -> 42.46
# EER on a 97%-spoof pool) that the paper reports without accounting for.
#
# Two skews x four targets x five seeds. Rows are tagged setting="skew0.9" etc so
# they can never be pooled with the balanced results.
for _sk in ("0.9", "0.7"):
    for _s in range(0, 5):
        add(f"P-skew{_sk}-seed{_s}",
            f'CACHE_ON_CPU=1 ADAPTIVE_SMOKE=0 INCLUDE_MLAAD=1 TARGET_SKEW={_sk} '
            f'OUR_SEEDS={_s} OUR_E_SWEEP="4" {PY} adaptive_pipeline.py '
            f'>> skew_experiment.out 2>&1',
            25, "stageP")

# --- stage Q: two genuinely independent corpora, our model, ten seeds --------
# The finding that most needs this: "extending the target set from four to seven
# erases the pooled EER benefit" currently rests on three ASVspoof2021 tracks,
# all relatives of a corpus that is itself in the source pool. WaveFake (vocoder
# artefacts over LJSpeech) and commercial TTS (ElevenLabs/Polly/Kokoro/Hume/
# Speechify, 2024) share no lineage with ASVspoof at all.
for _s in range(0, 10):
    add(f"Q-newcorpora-seed{_s}",
        f'CACHE_ON_CPU=1 ADAPTIVE_SMOKE=0 INCLUDE_MLAAD=1 NEW_TARGETS=1 '
        f'OUR_SEEDS={_s} OUR_TARGETS=hf_wavefake,hf_commercialtts OUR_E_SWEEP="4" '
        f'{PY} adaptive_pipeline.py >> new_corpora.out 2>&1',
        14, "stageQ")

# --- stage R: the same two corpora, third-party checkpoints, three seeds -----
# Extends the 196-cell calibration result onto corpora none of these models has
# seen. The two in-domain pairs are refused by public_ckpt_tta.py itself
# (ssl_aasist_wavefake on hf_wavefake, hf_xlsr_stafford on hf_commercialtts).
_R_CKPTS = [c for c in ALL_CKPTS]
for _ck in _R_CKPTS:
    for _tg in ("hf_wavefake", "hf_commercialtts"):
        if (_ck == "ssl_aasist_wavefake" and _tg == "hf_wavefake") or \
           (_ck == "hf_xlsr_stafford" and _tg == "hf_commercialtts"):
            continue
        for _s in range(0, 3):
            add(f"R-{_ck}-{_tg}-s{_s}",
                f"{PY} public_ckpt_tta.py --mode ours --ckpt {_ck} "
                f"--manifest manifest_tgt_{_tg}_seed{_s}.csv --target {_tg} --seed {_s} "
                f"--batch 16 {OUT}",
                8, "stageR")

# ================= 2026-09-07: autonomous pass to strengthen the paper ========
# The queue was fully drained (384/384). These stages exist because three claims
# the manuscript now leads with are thinner than the calibration-deficit result
# they sit beside:
#   S  -- "the method is a label-free threshold rule" is shown on our own model
#         only (threshold_control.csv, 70 cells). Put it to the third-party grid.
#   T  -- "E=32 reverses the AST degradations" rests on ONE seed (stage G2).
#   R2 -- the ASVspoof-independent third-party arm is 3 seeds; its deficit
#         closure is the one place in the study that misses significance.

# --- stage S: median/otsu threshold control on the third-party grid ----------
# One job; scores each (checkpoint, corpus) cell's seed-0 pool once, no
# adaptation. Writes threshold_control_multi.csv incrementally, so a kill leaves
# usable rows. ~76 cells, most 2-13 min; the AST cells dominate. Est 3.5 h.
add("S-threshold-control-multi",
    f"{PY} threshold_control_multi.py >> threshold_control_multi.out 2>&1",
    210, "stageS")

# --- stage T: E=32 depth on the two AST degradation cells --------------------
# hf_ast_asv19 on in_the_wild (-3.79 EER at E=4, the study's largest significant
# degradation) and dataset2 (-2.14). Stage G2 showed both flip strongly positive
# at E=32 -- but at seed 0 only. Seeds 1-4 give n=5 (with G2's seed 0) for a
# Wilcoxon of E=32-gain against source, and the E=4 rows at seeds 0-9 already
# exist from stage D for the paired contrast. E=32 on AST is ~65-90 min/cell.
for _ck, _tg in [("hf_ast_asv19", "in_the_wild"), ("hf_ast_asv19", "dataset2")]:
    for _s in range(1, 6):
        add(f"T-E32-{_ck}-{_tg}-s{_s}",
            f"{PY} public_ckpt_tta.py --mode ours --ckpt {_ck} "
            f"--manifest manifest_tgt_{_tg}_seed{_s}.csv --target {_tg} --seed {_s} "
            f"--batch 16 --tta_epochs 32 {OUT}",
            90, "stageT")

# --- stage R2: the ASVspoof-independent third-party arm, seeds 3-9 -----------
# Takes R from 3 seeds to 10. The R arm's deficit closure (6.18 -> 3.49) is the
# only pooled deficit result in the study that is not significant (p=0.24,
# n=48); ten seeds is the minimum at which its per-corpus Wilcoxon can reach
# p<0.05. Same job shape as stage R, which ran clean at seeds 0-2.
for _ck in ALL_CKPTS:
    for _tg in ("hf_wavefake", "hf_commercialtts"):
        if (_ck == "ssl_aasist_wavefake" and _tg == "hf_wavefake") or \
           (_ck == "hf_xlsr_stafford" and _tg == "hf_commercialtts"):
            continue
        for _s in range(3, 10):
            add(f"R-{_ck}-{_tg}-s{_s}",
                f"{PY} public_ckpt_tta.py --mode ours --ckpt {_ck} "
                f"--manifest manifest_tgt_{_tg}_seed{_s}.csv --target {_tg} --seed {_s} "
                f"--batch 16 {OUT}",
                8, "stageR2")
