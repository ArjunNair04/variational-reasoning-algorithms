"""Settings used in the experiments summarised here."""

SELECTED_SETTINGS = {
    "Q5": {
        "proposal": "answer-derived",
        "support": "persistent token-unique FIFO",
        "proposals_per_question": 16,
        "support_size": 16,
        "questions_per_round": 4,
        "updates_per_round": 1,
        "rounds": 32,
        "learning_rate": 1e-5,
        "answer_event": "strict terminal marker",
        "answer_target_termination": "EOS",
    },
    "PIS": {
        "proposal": "question-only current policy",
        "support": "fresh multiset",
        "support_size": 8,
        "questions_per_round": 8,
        "updates_per_round": 4,
        "rounds": 32,
        "learning_rate": 1e-5,
        "answer_event": "strict terminal marker",
        "answer_target_termination": "EOS",
    },
    "TRICE": {
        "support": "one persistent chain per question",
        "estimator": "control variate",
        "questions_per_round": 64,
        "updates_per_round": 1,
        "rounds": 32,
        "learning_rate": 1e-4,
        "initializer_prompt": "answer-derived",
        "proposal_prompt": "question-only",
        "reward_requires_eos": True,
    },
    "GRPO": {
        "group_size": 16,
        "questions_per_round": 4,
        "updates_per_round": 4,
        "rounds": 32,
        "learning_rate": 1e-5,
        "clip": 0.2,
        "kl_coef": 0.02,
        "optimizer_step_scope": "batch",
        "reward_requires_eos": True,
    },
    "RLOO": {
        "group_size": 16,
        "questions_per_round": 8,
        "updates_per_round": 4,
        "rounds": 32,
        "learning_rate": 2e-5,
        "kl_coef": 0.03,
        "reward_requires_eos": True,
    },
}


DEVELOPMENT_SETTINGS = {
    "Q5-MORE": {
        **SELECTED_SETTINGS["Q5"],
        "proposals_per_question": 32,
        "status": "preliminary development setting",
    }
}


DIAGNOSTIC_SETTINGS = {
    "AC-PIS": {
        **SELECTED_SETTINGS["PIS"],
        "proposal": "answer-derived",
        "responsibility": "trace + answer - proposal log density",
        "status": "lower than PIS in the study",
    },
    "CENTERED-TRACE": {
        **SELECTED_SETTINGS["PIS"],
        "rationale_credit": "posterior minus uniform",
        "answer_credit": "posterior",
        "status": "lower than PIS in the study",
    },
    "NULL-LATENT": {
        **SELECTED_SETTINGS["PIS"],
        "null_prior": 0.5,
        "null_log_evidence": -2.1191687253117015,
        "status": "uncertain in the seven-seed comparison",
    },
}


SELF_TRAINING_SETTINGS = {
    "Gold-CoT-SFT": {
        "learning_rate": 3e-6,
        "epochs": 2,
        "questions_per_round": 8,
        "answer_target_termination": "EOS",
    },
    "RFT": {
        "learning_rate": 1e-5,
        "epochs": 2,
        "iterations": 1,
        "group_size": 16,
        "questions_per_round": 4,
        "accepted_per_question": 10,
        "reward_requires_eos": True,
    },
    "ReST-EM": {
        "learning_rate": 1e-5,
        "epochs": 1,
        "iterations": 4,
        "group_size": 4,
        "questions_per_round": 16,
        "accepted_per_question": 10,
        "reset_before_improve": True,
        "reward_requires_eos": True,
    },
    "STaR": {
        "learning_rate": 3e-6,
        "epochs": 2,
        "iterations": 4,
        "group_size": 1,
        "questions_per_round": 16,
        "accepted_per_question": 1,
        "reset_before_improve": True,
        "reward_requires_eos": True,
    },
}


EM_FAMILY_PRESETS = {
    "VIN": {
        "proposal": "question-only",
        "support": "persistent token-unique",
        "responsibility": "joint",
        "refresh": "each update",
        "support_size": 8,
        "questions_per_round": 4,
        "updates_per_round": 1,
    },
    "VOUT": {
        "proposal": "question-only",
        "support": "persistent token-unique",
        "responsibility": "joint",
        "refresh": "once per round",
        "support_size": 8,
        "questions_per_round": 8,
        "updates_per_round": 2,
    },
    "POLD": {
        "proposal": "question-only",
        "support": "fresh multiset",
        "responsibility": "uniform",
        "refresh": "once per round",
        "support_size": 8,
        "questions_per_round": 4,
        "updates_per_round": 1,
    },
}
