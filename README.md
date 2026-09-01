# Variational reasoning algorithms

This package collects the update rules used in our reasoning experiments in a small NumPy implementation. The functions accept precomputed log probabilities, rewards, and sampled traces, so they can be used with different model-training stacks.

The two main finite-support methods are:

- **Q5**: answer-derived proposals, persistent finite support, and a joint latent posterior.
- **PIS**: fresh question-only proposals and answer-likelihood importance weights.

It also includes **TRICE**, **GRPO**, and **RLOO**, together with a few short ablations and self-training rules that help explain how the methods developed.

## Layout

```text
src/variational_reasoning/
  em.py               Q5, PIS, uniform credit, and finite support
  policy_gradient.py  GRPO and RLOO update kernels
  trice.py            persistent-chain transition and control variate
  self_training.py    RFT/ReST/STaR selection rules
  settings.py         settings used in the reported experiments
tests/
  test_algorithms.py  direct numerical checks
lm_study/              frozen Qwen3/GSM8K training and SGE harness
analysis/              preregistered result validators and analyzers
ALGORITHMS.md          derivations and method differences
JOURNEY.md             notes from the main ablations
```

## Install and test

```bash
python -m pip install -e '.[test]'
python -m pytest
```

## Example

```python
import numpy as np
from variational_reasoning import pis_weights

answer_logp = np.array([-4.0, -1.0, -2.0, -3.0])
question_id = np.array([0, 0, 1, 1])
weights = pis_weights(answer_logp, question_id)
```

Responsibilities are normalised separately for each question. The NumPy helper below evaluates the corresponding detached-weight objective:

```python
from variational_reasoning import weighted_joint_loss

loss = weighted_joint_loss(
    trace_logp=[-2.0, -1.4, -2.8, -2.1],
    answer_logp=answer_logp,
    weights=weights,
    question_ids=question_id,
)
```

The remaining modules follow the same pattern. A differentiable PyTorch or JAX training loop can apply the returned weights or advantages to its own model log probabilities.

## Experiment settings

| Method | Support / group | Questions per round | Updates | Learning rate |
|---|---:|---:|---:|---:|
| Q5 | 16 | 4 | 1 | `1e-5` |
| PIS | 8 | 8 | 4 | `1e-5` |
| TRICE | 1 chain | 64 | 1 | `1e-4` |
| GRPO | 16 | 4 | 4 | `1e-5` |
| RLOO | 16 | 8 | 4 | `2e-5` |

The settings used in the comparisons, including the KL and clipping values, are collected in `settings.py`. Q5-MORE records the preliminary proposal-depth variation explored after the main comparison.

## Reproducibility replay

The repository includes two frozen Qwen3-1.7B/GSM8K replay studies:

- `68078ecc`: 13 selected methods or controls across seven paired seeds, 91 tasks.
- `f3950b2e`: two PIS proposal-temperature conditions across seven paired seeds, 14 tasks.

The execution layer is preserved from the original implementation revision. It
uses the compact package as an independent numerical oracle during deployment;
training arithmetic is not rewritten by the port. See
`docs/POSTERITY_REPLAY.md` for the protocol, checks and AMN commands.
