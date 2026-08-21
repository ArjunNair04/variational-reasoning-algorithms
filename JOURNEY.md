# Notes from the ablations

This note summarises the ablations that were most useful while developing the methods.

## Finite support and responsibility weights

VIN and VOUT used the same joint latent posterior on persistent support. VIN refreshed the posterior during local optimisation; VOUT held it fixed for the round. Their similar results suggested that refresh timing was unlikely to be the main issue at these settings. Both lost stability under four repeated updates in these runs.

POLD replaced the posterior with uniform weights on fresh current-policy traces. Six of its 27 factorial cells passed the development gates, compared with 21 for PIS. These results suggested that current-policy sampling was not sufficient on its own, and that answer evidence in the responsibility weights was useful.

Strict-correct curation improved four-update uniform AUC by 12.82 points and trailed ordinary PIS by 0.82 points. In these runs, better support helped, although hard filtering gave no clear improvement over the ordinary importance-weighted update.

## Proposal conditioning

AC-PIS used an answer-derived proposal with proposal-density correction. In our runs, final extracted accuracy was 7.50 points lower than PIS, while the mean effective-support fraction fell from 0.608 to 0.387. The answer-conditioned proposal produced more concentrated finite-sample weights in this comparison.

In an earlier prompt study, literal answer-first conditioning performed worse. The derive-first alternative reduced the gap, but its three-seed difference was too uncertain to interpret, so we record it as a prompt observation.

## Signed and selective credit

Centred trace credit tested whether GRPO-like relative credit would help latent learning. AUC was 1.23 points lower than ordinary answer weighting, final strict accuracy was 6.00%, and format failure reached 93.33%. One possible explanation is that relative coefficients penalised traces that were weak only within an already poor sample.

The null-state update tested whether reducing total update mass under weak answer evidence might help. The three-seed result looked encouraging, while the seven-seed AUC gain over forced answer weighting was 1.63 points with an interval from -1.19 to 4.45. We therefore regard the result as uncertain.

## Supervised and self-training baselines

The final seven-seed comparison gave the following mean final extracted Acc@1 values:

| Method | Acc@1 |
|---|---:|
| STaR | 80.36% |
| RFT | 79.00% |
| ReST-EM | 78.11% |
| Frozen base | 76.18% |
| Gold-CoT SFT | 75.36% |

For this model and validation split, online generation and filtering were more competitive than gold-rationale supervision.

## Current development direction

Q5-MORE generates 32 proposals and compresses them to the same 16-trace support. In its three-seed screen, final extracted accuracy was 1.58 points higher than Q5, although the paired interval crossed zero. We therefore keep it as a preliminary development setting.

## Evaluation context

The table and comparisons use a fixed 400-question validation partition, with training seed as the comparison unit.
