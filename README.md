# Variational reasoning algorithms

This repository contains small reference implementations of the main algorithms used in the thesis experiments. The code is meant to make the update rules easy to inspect. It does not include the Hugging Face training stack, cluster scripts, experiment registry, or result archive.

The thesis-native methods are:

- **Q5**: answer-derived proposals, a persistent finite support, and a joint latent posterior.
- **PIS**: fresh question-only proposals and answer-likelihood importance weights.

The repository also includes the three main comparison methods used in the final study: **TRICE**, **GRPO**, and **RLOO**.

## Layout

```text
src/variational_reasoning/
  em.py               Q5, PIS, uniform credit, and finite support
  policy_gradient.py  GRPO and RLOO update kernels
  trice.py            persistent-chain transition and control variate
  settings.py         settings used in the selected experiments
tests/
  test_algorithms.py  direct numerical checks
ALGORITHMS.md          derivations and method differences
PROVENANCE.md          correspondence with the thesis repository
```

## Install and test

```bash
python -m pip install -e '.[test]'
python -m pytest
```

## Small example

```python
import numpy as np
from variational_reasoning import pis_weights

answer_logp = np.array([-4.0, -1.0, -2.0, -3.0])
question_id = np.array([0, 0, 1, 1])
weights = pis_weights(answer_logp, question_id)
```

Responsibilities are normalised separately for each question. They are detached before the model update in the full trainer.

## Selected settings

| Method | Support / group | Questions per round | Updates | Learning rate |
|---|---:|---:|---:|---:|
| Q5 | 16 | 4 | 1 | `1e-5` |
| PIS | 8 | 8 | 4 | `1e-5` |
| TRICE | 1 chain | 64 | 1 | `1e-4` |
| GRPO | 16 | 4 | 4 | `1e-5` |
| RLOO | 16 | 8 | 4 | `2e-5` |

The precise settings, including KL and clipping values, are in `settings.py`. Q5-MORE is included there as a development setting, not as a confirmed replacement for Q5.

The repository is private and currently carries no redistribution licence. The larger research repository remains the authoritative source for training orchestration and experimental records.
