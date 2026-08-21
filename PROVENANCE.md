# Provenance

This is a small semantic extraction from the thesis research repository, not a history mirror.

The selected seven-seed comparison ran from source commit:

```text
48bda822f94d486d15ac03c450449e0a66c6b998
```

Its frozen configuration SHA-256 was `196558334be56f1031a80d6c9abad05c063064b71b0a9a6b5be878de5263e5b3`.

The extraction was also checked against thesis branch commit:

```text
2c338924271d553bc7e8befe30e9d4f5c0bff1c9
```

| Reference code | Thesis source |
|---|---|
| `em.py`: Q5 and PIS responsibilities | `lm_study/ac_alg1.py` |
| `em.py`: strict answer evidence convention | `lm_study/answer_events.py`, `lm_study/answer_targets.py` |
| `em.py`: persistent support | `lm_study/ac_alg1.py` |
| `em.py`: centred trace credit | `lm_study/ac_alg1.py` |
| `em.py`: null-state posterior | `lm_study/ac_alg1_null_latent.py` |
| `em.py`: answer-conditioned correction | `lm_study/ac_alg1.py` |
| `policy_gradient.py`: GRPO | `lm_study/grpo.py` |
| `policy_gradient.py`: RLOO | `lm_study/rloo.py` |
| `trice.py` | `lm_study/ac_alg1_trice.py` |
| `self_training.py` | `lm_study/self_training.py` |
| `settings.py` | `lm_study/experiments_qwen3_17b_final_method_confirmation.yaml` and the registered Q5 support study |

The numerical tests cover the identities that matter for the update rules: per-question normalisation, PIS prior cancellation, Q5 joint weights, FIFO uniqueness, GRPO dead groups and clipping, RLOO leave-one-out credit, and TRICE transition/control-variate coefficients.

The full research repository remains authoritative for prompt construction, token masks, model loading, LoRA training, dataset splits, evaluation, and experiment provenance. No datasets, model weights, result artifacts, credentials, or cluster scripts are copied here.
