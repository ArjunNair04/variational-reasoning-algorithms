# Variational reasoning algorithms

This repository accompanies the thesis experiments on learning mathematical
reasoning from questions and final answers. It contains two ways to read the
work: short NumPy implementations of the update rules, and the original code
and configurations that executed the experiments.

The delta method starts from Professor David Barber's *Learning to Reason*
formulation. Our experiments implement that finite-buffer update for an
autoregressive language model, examine its proposal and fitting choices, and
compare it with importance sampling and established training methods.

## Start with the update

```bash
git clone https://github.com/ArjunNair04/variational-reasoning-algorithms.git
cd variational-reasoning-algorithms
python -m pip install -e '.[test]'
python -m pytest
```

```python
from variational_reasoning import delta_weights, prior_importance_weights

# Two candidate traces for one question. Values are sequence log probabilities.
trace_logp = [-2.0, -1.0]
answer_logp = [-1.0, -3.0]  # complete known-answer suffix, including EOS
question_ids = [0, 0]

w_delta = delta_weights(trace_logp, answer_logp, question_ids)
w_importance = prior_importance_weights(answer_logp, question_ids)
```

Delta normalises joint probabilities over distinct retained traces. Prior
importance sampling uses fresh sampled occurrences and normalises their answer
likelihoods. [ALGORITHMS.md](ALGORITHMS.md) explains the distinction, the
complete-response objective and the comparison methods.

| Readable module | Contents |
|---|---|
| `em.py` | delta, importance, equal weights, usable-question objective, FIFO |
| `diagnostics.py` | entropy, effective sample size, temperature floor |
| `answer_events.py`, `answer_targets.py` | original answer parser and EOS targets |
| `policy_gradient.py`, `trice.py` | GRPO, RLOO and TRICE update kernels |
| `self_training.py` | RFT, ReST-EM and STaR selection rules |

These functions accept precomputed scores. The experiment snapshots below
contain tokenisation, model loading, LoRA, sampling, scoring and optimisation.
Historical `q5_weights` and `pis_weights` names remain available for existing
users; the documentation uses delta and importance sampling.

## Inspect or replay an experiment

List the studies, then unpack a chosen implementation without loading a model
or dataset. Preparation requires only Python's standard library.

```bash
python experiments/reproduce.py
python experiments/reproduce.py selected_methods \
  --workdir /tmp/reasoning-selected --prepare
```

The original files are now under `/tmp/reasoning-selected/lm_study/`. Begin with
`ac_alg1.py` for delta/importance, `run_yaml.py` for configuration expansion, and
`run_sweep_lm.py` for the experiment loop. The other named trainers are separate
modules. Archives keep multiple historical versions compact; no training source
is fetched from another repository.

For GPU execution, create a separate Python 3.11 environment on Linux with
CUDA 12.4 and install the archived dependency specification:

```bash
python -m pip install -r experiments/requirements.txt
python experiments/reproduce.py selected_methods \
  --workdir /tmp/reasoning-selected --cell Q5-LR1e-5-U1-K16 --seed-index 0
```

The last command uses the original driver's **dry-run** mode. Add `--execute`
to train the chosen cell. Seed index zero means the first seed in the frozen
configuration, not seed zero. Source files stay unchanged; a temporary YAML
changes output placement and, when requested, selects one existing cell.

[EXPERIMENTS.md](experiments/EXPERIMENTS.md) maps each chapter comparison to its
configuration, source revision and analysis. It also explains the two delta
schedules, historical changes, and the trained-adapter prerequisite for transfer.
GPU training is not part of the tests.

## Read the reported evidence without training

```bash
python experiments/results.py --verify
python -m pip install -e '.[plots]'
python experiments/results.py --plot /tmp/reasoning-figures
```

The checked numeric exports retain the published means, seed-level summaries,
contrasts and weight distributions. The plots are regenerated from those
exports; raw model responses and checkpoint weights are not included.
[Verification](experiments/VERIFICATION.md) distinguishes formula checks,
source/configuration checks and dry runs from a fresh GPU reproduction.
