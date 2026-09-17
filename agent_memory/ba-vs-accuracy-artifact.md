---
name: ba-vs-accuracy-artifact
description: The dramatic accuracy collapse in the ICASSP paper is largely a fixed-threshold artifact; BA tells a different story
metadata:
  type: project
---

On ASVspoof2021-DF (97.22% spoof, so the always-spoof baseline scores 97.22%),
symmetric q=0.3 adaptation looks catastrophic in accuracy and mild in every
threshold-free metric:

| checkpoint | acc source -> adapted | BA source -> adapted |
|---|---|---|
| DF-2  | 98.87 -> 62.96 | 88.28 -> 80.89 |
| DF-42 | 98.78 -> 61.80 | 78.35 -> **80.35** |
| DF-240| 99.11 -> 65.24 | 87.01 -> 82.12 |

On DF-42 the "collapse" is a balanced-accuracy **improvement**. BBSE's accuracy
"restoration" also lowers BA on 2 of 3 checkpoints (88.28->82.34, 87.01->77.66).

**Why:** a reviewer computing BA finds this immediately, so leading with the
accuracy number is a liability, not a strength. It is also what motivated
[[operating-point-law]].

**How to apply:** never quote an accuracy on a skewed pool without the trivial
baseline next to it.
