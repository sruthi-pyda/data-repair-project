# Cross-domain detection: the rule detector on nine datasets

Measured with `src/detection_eval` (PR-AUC primary; lift = PR-AUC /
prevalence, so **1.0x is chance**). Whole-dataset scoring, shipped detector
configuration. `soccer` and `tax` omitted here for runtime and are run
separately.

| dataset | prevalence | PR-AUC | **lift** | ROC-AUC | best-F1 (oracle) |
|---|---:|---:|---:|---:|---:|
| smartfactory | 0.0158 | 1.0000 | **63.3x** | 1.0000 | 1.0000 |
| hospital | 0.0255 | 0.9136 | **35.9x** | 0.9975 | 0.9443 |
| movies | 0.0558 | 0.6301 | **11.3x** | 0.9383 | 0.7988 |
| breast_cancer | 0.0752 | 0.6497 | **8.6x** | 0.8106 | 0.7664 |
| beers | 0.1266 | 0.7267 | **5.7x** | 0.9546 | 0.8623 |
| flights | 0.2958 | 0.9551 | **3.2x** | 0.9681 | 0.9670 |
| rayyan | 0.0862 | 0.1815 | **2.1x** | 0.6869 | 0.3738 |
| restaurants | 0.0012 | 0.0018 | **1.5x** | 0.5854 | 0.0052 |
| adult | 0.0025 | 0.0025 | **1.0x** | 0.4957 | 0.0050 |

## What this establishes

**The method generalises to six of nine datasets and fails on three.** That is
a far more informative result than the single-dataset evidence it replaces,
and it is the first time the approach has been tested outside the original
three.

**Where it works** (lift 3.2x to 63.3x): hospital, movies, breast_cancer,
beers, flights, smartfactory. These carry *syntactic* corruption — missing
values, format violations, out-of-range values — which rule checks are built
to see.

**Where it fails**: adult (1.0x = chance), restaurants (1.5x), rayyan (2.1x).

## The pattern is corruption type, not dataset size or prevalence

Prevalence alone does not predict failure. `hospital` at 2.55 % prevalence
achieves **35.9x** lift, while `restaurants` at 0.12 % achieves 1.5x and
`adult` at 0.25 % achieves chance. Two low-prevalence datasets behave
completely differently from a third.

What separates them is whether the corruption is *visible without context*.
Adult's corruption swaps a valid value for another valid value; no syntactic
rule can see it. That is the same mechanism documented in
`docs/research_program.md` §8, now reproduced on two further datasets.

## Honest reading

- **`rayyan` is a new failure we had not seen** (0.1815, 2.1x) and it is not
  an extreme-imbalance case — prevalence is 8.62 %, comparable to
  breast_cancer's 7.52 % where the detector reaches 8.6x. Something other
  than imbalance is wrong there and it has not yet been diagnosed.
- **`flights` has the highest PR-AUC (0.9551) but the lowest lift (3.2x)**
  because its prevalence is 29.58 %. A high PR-AUC at high prevalence is
  much less impressive than the raw number suggests — which is exactly why
  lift is reported beside it.
- best-F1 is an **oracle** threshold throughout, not an achievable operating
  point.
- `soccer` and `tax` are not in this table; omitting them is a runtime
  decision, not a result.
