# Qwen3 JEPO comparator

## Question

Does the multi-sample JEPO objective improve Qwen3-1.7B GSM8K reasoning under
the same 2,048-generation budget and paired validation protocol used for Q5
and PIS?

## Frozen design

- One new method cell only: `JEPO-MS4-LR1e-5`.
- Seven paired seeds: `1201, 1213, 1217, 1223, 1229, 1231, 1237`.
- Four current-policy, question-conditioned samples for each of 16 questions
  per round; 32 rounds and 2,048 training generations per seed.
- Qwen3-1.7B Base with rank-16 LoRA on attention and MLP projections.
- Strict `#### answer` plus natural tokenizer EOS training target.
- Paper-derived JEPO coefficients: supervised `0.01`, format penalty `10`,
  clipped normalized advantage `1`, and frozen-policy KL `0.001`.
- Invalid generations contribute zero to the evidence terms without changing
  the fixed generated-batch denominator.
- Fixed 400-question validation partition derived from GSM8K training data.
  The official GSM8K test split is not loaded or scored.

Historical same-seed controls are reused from run `68078ecc`; they are not
resubmitted. This isolates the new JEPO implementation and avoids spending GPU
time reproducing already validated cells.

## Interpretation boundary

This is a common-protocol JEPO comparator, not a reproduction of the JEPO
paper's full training setup. It retains the paper's multi-sample estimator and
core coefficients but uses the project's smaller Qwen model, LoRA adaptation,
learning rate, dataset and generation budget.

## Validation and reporting

The payload is held until generated-YAML equality, all seven dry-run
coordinates, Python compilation, shell syntax and frozen-analysis validation
pass. The dependent validator requires all receipts, receipt hashes, trained
adapters, terminal log markers, 32 diagnostics, seven checkpoints and explicit
official-test non-access.

Results will be reported as Final Acc@1 first, strict final accuracy second and
normalized trajectory AUC third. Comparisons with the same-seed frozen base
use paired training seeds. Mechanism reporting includes valid-format coverage,
trace-advantage clipping, answer-weight ESS, sampled policy KL, generated and
backward tokens, optimizer steps and accelerator-hours.
