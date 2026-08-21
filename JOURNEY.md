# What the ablations showed

This is not a catalogue of every experiment. It records the small algorithmic changes that materially changed our understanding of the method.

## Finite support and responsibility weights

VIN and VOUT used the same joint latent posterior on persistent support. VIN refreshed the posterior during local optimisation; VOUT held it fixed for the round. Their similar results showed that refresh timing was not the main problem. Both became unreliable with four repeated updates on the same support.

POLD replaced the posterior with uniform weights on fresh current-policy traces. Only 6 of its 27 factorial cells passed the development gates, compared with 21 for PIS. This was the clean evidence that current-policy sampling alone was not enough: the answer evidence in the responsibility weights mattered.

Strict-correct curation later improved four-update uniform AUC by 12.82 points, but still trailed ordinary PIS by 0.82 points and failed its coverage requirement. Better support helped, but hard filtering was not a replacement for the ordinary importance-weighted update.

## Proposal conditioning

AC-PIS used an answer-derived proposal and corrected it by its proposal density, but performed worse than native PIS. Final extracted accuracy fell by 7.50 points and the mean effective-support fraction fell from 0.608 to 0.387. The correction was valid; the answer-conditioned proposal made the finite-sample weights more concentrated.

An earlier prompt study found literal answer-first conditioning harmful. Its derive-first alternative was less damaging, but the three-seed improvement was unresolved. It therefore remains a prompt observation rather than another exported algorithm.

## Signed and selective credit

Centred trace credit tested whether GRPO-like negative relative credit would repair latent learning. It did not. AUC fell by 1.23 points against ordinary answer weighting, final strict accuracy was 6.00%, and format failure reached 93.33%. Negative rationale coefficients could punish traces that were only relatively weak inside a poor sampled set.

The null-state update tested a gentler hypothesis: reduce total update mass when all sampled traces have weak answer evidence. Its three-seed pilot was promising, but the seven-seed gain over forced answer weighting was 1.63 AUC points with an interval from -1.19 to 4.45. It remains an elegant diagnostic, not a selected method.

## Supervised and self-training baselines

The final seven-seed comparison gave the following mean final extracted Acc@1 values:

| Method | Acc@1 |
|---|---:|
| STaR | 80.36% |
| RFT | 79.00% |
| ReST-EM | 78.11% |
| Frozen base | 76.18% |
| Gold-CoT SFT | 75.36% |

These methods are included because they are simple and directly relevant to the reasoning-training question. Gold rationale supervision was not automatically best in this setting; online generation and filtering were more competitive. This does not establish a universal ordering across models or datasets.

## Current development direction

Q5-MORE generates 32 proposals and compresses them to the same 16-trace support. In its three-seed screen it improved final extracted accuracy by 1.58 points over Q5, but the paired interval crossed zero. It is kept as a development setting awaiting fresh confirmation.

## Deliberate exclusions

The compact repository does not include segment-flow credit, exact-signed credit, verifier combinations, Bayesian fusion, two-witness credit, ESS projections, replay/allocation systems, or mixed-proposal programmes. They were either unfinished, mechanically redundant, or combined enough moving parts that they would obscure rather than clarify the main research path.

All figures above use the fixed training-derived validation partition. Training seed is the inferential unit, and these are not official GSM8K-test estimates.
