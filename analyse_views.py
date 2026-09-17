"""Read the pilot score matrix and answer the three questions it was built for.

  1. View quality: does each view preserve spoofing evidence on its own?
  2. The frozen multi-view teacher: does averaging over views improve ranking
     over the single standard view? This is the baseline any adaptation must
     beat, not the single-view source.
  3. Pair weighting: does cross-view agreement on a pair carry information
     about whether that pair is ordered correctly?

Question 3 is the one the proposed loss depends on. Labels enter only here, as
diagnosis; nothing in the method would be allowed to use them.

    python analyse_views.py pilot_views_itw_ssl_aasist_wavefake.npz
"""
import sys

import numpy as np
from metrics import compute_eer
from sklearn.metrics import roc_auc_score

path = sys.argv[1] if len(sys.argv) > 1 else "pilot_views_itw_ssl_aasist_wavefake.npz"
z = np.load(path, allow_pickle=True)
s, y = z["scores"], z["label"]
n, K = s.shape
print(f"{path}: {n} clips, {K} views, spoof {100*y.mean():.2f}%\n")


def rank_metrics(v):
    eer, _ = compute_eer(y, v)
    return 100 * eer, roc_auc_score(y, v)


print("1. VIEW QUALITY  (view 0 is the standard leading crop)")
for v in range(K):
    e, a = rank_metrics(s[:, v])
    print(f"   view {v}: EER {e:6.3f}%  AUC {a:.4f}"
          f"{'   <- standard' if v == 0 else ''}")

print("\n2. FROZEN MULTI-VIEW TEACHER")
e0, a0 = rank_metrics(s[:, 0])
em, am = rank_metrics(s.mean(1))
er, ar = rank_metrics(np.argsort(np.argsort(s, 0), 0).mean(1))   # rank-average
print(f"   single standard view : EER {e0:6.3f}%  AUC {a0:.4f}")
print(f"   mean over {K} views     : EER {em:6.3f}%  AUC {am:.4f}"
      f"   ({100*(em-e0)/e0:+.1f}% relative EER)")
print(f"   rank-mean over {K} views: EER {er:6.3f}%  AUC {ar:.4f}"
      f"   ({100*(er-e0)/e0:+.1f}% relative EER)")

print("\n3. DOES CROSS-VIEW AGREEMENT IDENTIFY CORRECT ORDERINGS?")
rng = np.random.RandomState(0)
# sample discordant-label pairs: the only pairs whose correct order is defined
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
m = 400_000
i = pos[rng.randint(0, len(pos), m)]
j = neg[rng.randint(0, len(neg), m)]
pref = (s[i] > s[j])                     # (m,K) per-view preference "i is spoofier"
agree = pref.mean(1)                     # fraction of views preferring i
correct = pref.mean(1) > .5              # majority-vote ordering, i truly spoof
stability = np.abs(agree - .5) * 2       # 0 = views split, 1 = unanimous
print(f"   {m:,} spoof/bona-fide pairs sampled")
print(f"   majority-vote ordering accuracy overall: {100*correct.mean():.2f}%")
print("   by cross-view stability:")
edges = [0, .25, .5, .75, .999, 1.001]
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (stability >= lo) & (stability < hi)
    if sel.sum() < 100:
        continue
    print(f"     stability [{lo:.2f},{hi:.2f}): {sel.sum():7,} pairs "
          f"({100*sel.mean():5.1f}%)  ordering accuracy {100*correct[sel].mean():6.2f}%")
una = stability > .999
print(f"   unanimous pairs: {100*una.mean():.1f}% of all, accuracy {100*correct[una].mean():.2f}%")
print(f"   split pairs    : {100*(~una).mean():.1f}% of all, accuracy {100*correct[~una].mean():.2f}%")
single = s[i, 0] > s[j, 0]
print(f"\n   for reference, single-view ordering accuracy: {100*single.mean():.2f}%")
