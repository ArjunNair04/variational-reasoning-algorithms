# Experiment index

Each source archive contains the local Python dependency closure of an original
execution revision. Files are byte-identical to that revision; `source_manifest.json`
records individual and archive SHA-256 values. One archive is shared by studies
executed from the same revision. Configuration YAMLs are also unchanged.

The short NumPy package explains the principal operations. The archives execute
them in the actual model-training loop, including historical behavior that the
final method replaced. Original identifiers such as Q5, PIS, VIN, VOUT, B and U
remain in source/configuration records. In the thesis, Q5 is pseudo-posterior
delta, PIS is prior importance sampling, B denotes the question batch in the
common-grid cell identifiers, and U denotes repeated fitting steps J.

| Study key | Thesis comparison | Execution revision | Configuration |
|---|---|---|---|
| `initial_objectives` | 4.2: labelled and answer-only objective terms; historical trainer | `44eed6e7a00f` | [ac_alg1_objective_terms](configs/experiments_qwen3_17b_ac_alg1_objective_terms.yaml) |
| `initial_latent_terms` | 4.2: historical latent-only objective terms | `9ca8dfa34e26` | [ac_alg1_latent_only](configs/experiments_qwen3_17b_ac_alg1_latent_only.yaml) |
| `early_replication` | 4.2: five-seed short-horizon objective replication | `e44dff166979` | [ac_alg1_early_horizon_replication](configs/experiments_qwen3_17b_ac_alg1_early_horizon_replication.yaml) |
| `prompt_base` | 4.3: frozen prompt calibration | `ef3d8aac09d7` | [prompt_shot_base](configs/experiments_qwen3_17b_prompt_shot_base.yaml) |
| `prompt_training` | 4.3: trained three-/four-shot comparison | `ef3d8aac09d7` | [prompt_shot_training](configs/experiments_qwen3_17b_prompt_shot_training.yaml) |
| `eos` | 4.3: paired complete-answer target repair | `ba06c02dd429` | [barber_eos_repair_retry](configs/experiments_qwen3_17b_barber_eos_repair_retry.yaml) |
| `prompt_information` | 4.4: historical proposal and importance correction | `46fade649f9b` | [prompt_information_paths](configs/experiments_qwen3_17b_prompt_information_paths.yaml) |
| `joint_objective` | 4.3–4.4: joint/answer/trace fitting and proposals | `da711297c198` | [l2r_persistent_importance_bridge](configs/experiments_qwen3_17b_l2r_persistent_importance_bridge.yaml) |
| `prior_delta_retry` | Historical supplementary prior-proposal retry (outside current thesis) | `0cd1347f91e2` | [q5_question_lr_screen](configs/experiments_qwen3_17b_q5_question_lr_screen.yaml) |
| `common_grid` | 4.5: common grid of support, batch and repeated fitting | `4d9f528a59ab` | [l2r_common_protocol_factorial](configs/experiments_qwen3_17b_l2r_common_protocol_factorial.yaml) |
| `reader_ess` | 4.5: seven-seed frozen answer scorer / ESS floor | `f79b0f1d3cf5` | [l2r_reader_ess_closure](configs/experiments_qwen3_17b_l2r_reader_ess_closure.yaml) |
| `delta_confirmation` | 4.6: revisited-question delta confirmation | `5988113f682a` | [q5_reader_confirmation](configs/experiments_qwen3_17b_q5_reader_confirmation.yaml) |
| `importance_confirmation` | 4.6: first importance-sampling confirmation | `5988113f682a` | [pis_confirmation](configs/experiments_qwen3_17b_pis_confirmation.yaml) |
| `selected_methods` | 4.6: seven-seed selected-method comparison | `48bda822f94d` | [final_method_confirmation](configs/experiments_qwen3_17b_final_method_confirmation.yaml) |
| `pseudo_importance` | 4.4: corrected pseudo-posterior importance proposal | `38d134b091f3` | [answer_conditioned_pis_addon](configs/experiments_qwen3_17b_answer_conditioned_pis_addon.yaml) |
| `equal_counts` | 4.6: GRPO/RLOO at the first importance schedule | `056b86affe6b` | [gsm8k_matched_comparators](configs/experiments_qwen3_17b_gsm8k_matched_comparators.yaml) |
| `proposal_depth` | 4.7: three-seed proposal-depth screen | `5c38141222a8` | [q5_support_reallocation_retry](configs/experiments_qwen3_17b_q5_support_reallocation_retry.yaml) |
| `transfer` | 4.7: frozen SVAMP and MATH-500 evaluation | `9aa67a557f97` | Receipt-bound adapter manifest; see transfer below |

## Start with selected methods

Use `selected_methods` for the seven-seed comparison. To inspect the ordinary
delta configuration, use cell `Q5-LR1e-5-U1-K16`; for importance sampling use
`PIS-S8-B8-U4`. The YAML lists all other cell IDs. `--seed-index 0` chooses
seed 1201, and indices 1–6 choose 1213, 1217, 1223, 1229, 1231, 1237.

The launcher does not retune any field. It writes a temporary YAML with the
requested existing cell and a new results directory, then invokes the archived
driver. `replay_provenance.json` binds this runtime configuration to the original
source and config hashes. The driver receives the original execution revision
through its existing `EXPECTED_COMMIT` fallback, where supported; no enclosing
unrelated Git checkout is used as the implementation identity.

The ordinary selected delta configuration generates 16 candidates, retains up
to 16 distinct prefixes, fits four questions per round and takes one step. The
earlier `delta_confirmation` uses revisited questions, a different fitting
schedule and attention-only LoRA. `importance_confirmation` and the later
selected importance cell use the same S8/M8/J4 configuration on different seeds.
The exact batch-allocation fields in YAML are authoritative; the original
driver distinguishes response-batch sizes from question counts.

## Historical versions and partial continuations

`initial_objectives` and `initial_latent_terms` retain the older multi-term
objective, filtering and response treatment. The later `eos`, `joint_objective`
and proposal studies should not be silently replayed through those versions.
`prompt_information` is the historical refreshed-numerator importance check,
not the final fixed-weight importance method. Some original historical raw
question records are unavailable in the released evidence; their reported
aggregate values are identified as such in `results/README.md`.

The common grid has two recorded execution segments: tasks 1–236 used
`4d9f528a59ab65813d17df6c79c3d219b3f9cfad`; tasks 237–489 used
`4d0c8a5006013cd4f663749ebd3d371e6343f019`. Both used the same YAML.
`--continuation` selects the second **source version only**. It does not
select historical tasks 237–489, resume an interrupted result tree, or combine
old receipts with new outputs. A local replay is a new execution of whichever
configured cells/seeds the reader chooses, with that source version recorded.

## Data and response contracts

All GSM8K training studies use questions from the pinned official training split.
The 400-question development-validation reserve is separate from optimization
and demonstration pools. No official GSM8K test data are included or accessed
by preparation, dry-run, numeric verification or the public launcher.

The final known-answer training suffix includes EOS once. The evaluation
terminal-format checker is a text-format test: one terminal marker and the
correct answer, with no substantive trailing text. Natural EOS and token-limit
flags are separate. Extracted-answer accuracy uses the historical permissive
parser. The exact parser is available directly in the NumPy package; MATH-500
uses its own archived mathematical-answer parser. Evaluation prompts remain
question-only even when delta training uses the pseudo-posterior proposal.

## Transfer needs the trained adapters

`transfer_plan.json` exposes all 56 method/dataset/seed assignments and dataset
pins without the original checkpoint paths.

Prepare `transfer` to inspect `evaluate_qwen3_final_transfer.py` and
`prepare_qwen3_final_generality.py`. Transfer uses the already selected
round 32 adapters; it does not train on SVAMP or MATH-500. The original 56-task
manifest bound the original adapter directories and completion receipts, which
are not distributed here. After reproducing selected methods, use the original
preparation script with the new analysis decision, artifact root and output
manifest (`--help` lists its arguments), then invoke the evaluator with that
manifest, its SHA-256 and a task index. Do not substitute adapter paths while
keeping a receipt hash from a different execution.

Dataset revisions, row counts and content hashes are embedded in the original
preparation/evaluation modules. The public numeric summaries retain all four
methods, both datasets and all seven seeds. They allow metric/count checking
without redistributing 42,000 model responses or model weights.

## Analyses and numeric replay

Preparation also expands the analysis snapshot under `analysis_tools/`. These
are archived analysis recipes at revision 3472a14, with their own source hashes;
they are later than execution and never replace the training modules.
`studies.json` lists the analysis entry points. Their full audit paths require
the original expected receipt, configuration and raw-artifact set; they are not
advertised as accepting arbitrary newly trained results without a new analysis
binding. No scheduler scripts or private result trees are included.

For the released compact evidence, use `python experiments/results.py --verify`.
It checks all exported bytes, recomputes ESS/entropy from saved weights, checks
fixed-weight objective differences and their unobserved fourth-step scope, and
reconstructs transfer metrics from category counts. `--plot DIR` redraws selected
method, sample-budget and fixed-objective plots. It uses published aggregates
and seed summaries, not newly selected outcomes or new bootstrap intervals.

The project compute ledger is in `resources/compute/`; its 970 completed timing
records sum to 808.589722 H100 GPU-hours. This is a lower bound on the wider
project, including studies outside the 18 thesis source groups. It is not the
cost of a single algorithm comparison.
