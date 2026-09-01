# Selected-method posterity replays

Two same-seed replays preserve the selected Qwen3/GSM8K methods and the
proposal-sampling temperature experiment under current code. They are
reproducibility studies, not new hyperparameter searches.

## Common-protocol method replay

Run `68078ecc` contains thirteen methods over the seven final-confirmation seeds
`1201, 1213, 1217, 1223, 1229, 1231, 1237`, for 91 tasks:

1. frozen base;
2. Q5 with a regular question-only proposal prompt;
3. Q5 with the answer-derived proposal prompt and moving answer reader;
4. Q5 with the answer-derived proposal prompt and frozen answer reader;
5. Q5 with the answer-derived proposal prompt and a 0.50 minimum ESS floor;
6. Q5 with a rationale-only adaptive KL target of 0.03;
7. Q5 with answer-derived proposals sampled at temperature 1.2;
8. PIS with a regular question-only proposal;
9. PIS with the answer-derived proposal and its exact answer-conditioned
   importance correction;
10. ReST-EM;
11. faithful STaR;
12. TRICE-CV; and
13. RLOO.

Each trained method uses its previously frozen selected setting. No method is
factorially retuned. The shared contract retains Qwen3-1.7B-Base, three shots
from the five-example prompt bank, 128 optimisation questions, 32 rounds,
rank-16 attention-plus-MLP LoRA, the fixed 400-question train-derived
validation partition and strict terminal-answer evaluation.

The three Q5 stability cells are isolated one-factor changes from canonical
moving-reader Q5. The ESS arm uses the historical 0.50 floor, implemented by
the minimum one-sided responsibility-temperature increase needed to meet that
floor. The KL arm adaptively targets an anchor-gradient norm equal to 0.03 of
the method-gradient norm and masks the answer marker, answer and EOS. The
temperature arm changes only proposal sampling from 1.0 to 1.2.

The Q5 prompt pair changes only the proposal prompt and the validation profile
required to permit that prompt. The PIS pair repeats the two estimators already
run: regular Prior-IS for question-only proposals and exact
answer-conditioned importance sampling for answer-derived proposals. It is
therefore an estimator-level replication, not a claim that only the prompt
string differs.

Final Acc@1 is reported first, strict final accuracy second and normalized
trajectory AUC third. Paired seed is the independent replicate. Final round 32
is fixed before outcomes, and no method is selected from this replay.

## PIS proposal-temperature replay

Run `f3950b2e` exactly repeats the original two-cell temperature experiment on
seeds `1481, 1483, 1487, 1489, 1499, 1511, 1523`. The control draws eight
question-only rationales at temperature 1.0. The treatment draws four at
temperature 1.0 and four at temperature 1.2, then applies the exact 50:50
mixture-density correction used in the original run. All other scientific
settings are byte-for-byte inherited from the source YAML except the new run
identity and output path.

This replay repeats a previously negative coordinate for posterity. Its result
does not authorise outcome-based temperature retuning.

## Release gate

Both payloads must be submitted under user hold and remain held until the
active Herring development payload and validator finish cleanly. Release also
requires enough Beaker home-quota and file-count headroom for every expected
artifact plus reserve. A missing Herring marker, repeated quota failure,
scientific mismatch or ambiguous scheduler state blocks release.

The official GSM8K test split is prohibited in generation, training,
validation and analysis.
