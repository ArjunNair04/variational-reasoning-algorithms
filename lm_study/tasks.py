"""Programmatic-reward tasks for the LM study, mirroring the tabular difficulty ladder.

Each task = (prompt, reward_fn over completion strings, target). Rewards are cheap to compute but
ACCOUNTED as a costly oracle: the study's cost axis is total completions scored. The ladder maps
the tabular boundary knob onto language:

  dense       keyword-fraction  ~ matching / low-threshold  (smooth gradient, EM predicted to win)
  structured  sortedness        ~ bigram                    (position-invariant pattern)
  sparse      all-or-nothing    ~ threshold 8-of-10         (EM predicted to LOSE without offpol)

Registered predictions (from the tabular map, notebook §5–§6b): see README.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from answer_events import parse_gsm8k_answer_event
from math_answer_events import math_answers_equivalent, parse_math_answer_event


class Task:
    max_new = 40                                          # default completion length (multi-prompt gen)
    floor = 0.0                                           # reward floor (tasks returning 0.05+correct set 0.05);
                                                          # trainers subtract it to report accuracy, never hardcode
    def __init__(self, name, prompt, reward_fn, best=1.0, target_frac=0.9):
        self.name, self.prompt, self._fn = name, prompt, reward_fn
        self.best = best
        self.target = target_frac * best

    def reward(self, completions, pids=None):            # list[str] -> np.ndarray
        # pids (per-completion prompt index) is accepted for answer-aware multi-prompt tasks
        # (e.g. GSM8K) and ignored by single-reward tasks. Multi-prompt runners pass it.
        return np.array([self._fn(c) for c in completions], dtype=np.float64)


_WORD = re.compile(r"[a-z']+")


def _words(text):
    return _WORD.findall(text.lower())


# --------------------------------------------------------------------------- #
#  dense: fraction of k required keywords present (smooth, separable-ish)
# --------------------------------------------------------------------------- #
KEYWORDS = ["sun", "water", "sand", "waves", "warm"]


def keyword_reward(completion):
    ws = set(_words(completion))
    return sum(k in ws for k in KEYWORDS) / len(KEYWORDS)


def dense_task():
    # continuation-style prompt: Pythia-160m is a BASE model, instruction prompts get ~0 signal
    # (measured: 0.0125 mean vs 0.055 for this prompt, 30/128 completions with >= 1 keyword)
    return Task("dense:keywords5",
                "It was a beautiful day at the beach. The",
                keyword_reward)


# --------------------------------------------------------------------------- #
#  structured: alphabetical sortedness of the word sequence (position-invariant
#  pairwise pattern -- the LM analogue of the bigram task)
# --------------------------------------------------------------------------- #
def sorted_reward(completion):
    ws = _words(completion)
    if len(ws) < 4:                                       # too short to demonstrate structure
        return 0.0
    pairs = list(zip(ws[:-1], ws[1:]))
    return sum(a <= b for a, b in pairs) / len(pairs)


def structured_task():
    return Task("structured:sorted",
                "Words sorted alphabetically: act, bed, cat, dog,",
                sorted_reward)


# --------------------------------------------------------------------------- #
#  sparse: ALL keywords present AND length constraint -- all-or-nothing bonus
#  (the LM analogue of threshold 8-of-10; floor keeps reward > 0 like tabular)
# --------------------------------------------------------------------------- #
def sparse_reward(completion):
    ws = set(_words(completion))
    n = len(_words(completion))
    hit = all(k in ws for k in KEYWORDS) and 8 <= n <= 25
    return 0.05 + float(hit)


def sparse_task():
    return Task("sparse:all5",
                "It was a beautiful day at the beach. The",
                sparse_reward, best=1.05)


def sparse3_reward(completion):
    ws = set(_words(completion))
    return 0.05 + float(sum(k in ws for k in KEYWORDS) >= 3)


def sparse3_task():
    """Recalibrated sparse task (MATHS.md M5). The original all-5+length bonus has measured base
    rate 0/1024 -- conjunction-grade, undiscoverable by anything at this budget (confirmed: every
    method flat at the floor for 2,560 completions). >=3-of-5, no length: base rate 7/1024 = 0.7%,
    threshold-grade (the tabular 8-of-10 analogue, which had 4e-4)."""
    return Task("sparse3:3of5", "It was a beautiful day at the beach. The",
                sparse3_reward, best=1.05)


_SCENES = ["It was a beautiful day at the beach. The",
           "We spent the whole morning by the sea, and the",
           "The afternoon at the shore felt endless. The",
           "Summer had finally arrived at the coast, and the",
           "They walked down to the bay at noon, where the",
           "The little seaside town woke slowly, and the",
           "Out on the boardwalk that day, the",
           "By the time we reached the ocean, the"]
_LINKS = ["", " Around us, the", " Soon enough, the", " Before long, the"]


class MultiPromptTask(Task):
    """M14: one reward, many prompts (the realistic fine-tuning shape)."""
    def __init__(self, name, prompts, reward_fn, best=1.0, target_frac=0.9):
        super().__init__(name, prompts[0], reward_fn, best, target_frac)
        self.prompts = prompts


def dense_mp_task():
    prompts = [s + l for s in _SCENES for l in _LINKS][:32]
    return MultiPromptTask("dense_mp:keywords5x32", prompts, keyword_reward)


# --------------------------------------------------------------------------- #
#  reasoning: a single (question, answer) pair with a hidden chain-of-thought and
#  a VERIFIER reward -- the LM instance of BarberDocs/LearningToReason.pdf (eq 1-10,
#  which is derived for one (q,a) pair). The model emits reasoning then states a number;
#  reward = 1 if the correct total appears among the stated numbers (deterministic
#  verifier, eq 11), floor keeps it > 0. This is the Tier-1 lift of the tabular
#  reasoning toy: compare the eq-8 weight (Barber) vs EM-L1 vs GRPO on real CoT tokens.
# --------------------------------------------------------------------------- #
def reasoning_task(a=2, b=3, c=4):
    ans = a + b + c                                       # multi-step: a scratchpad helps
    def reward(completion):
        nums = [int(x) for x in re.findall(r"-?\d+", completion)]
        return 0.05 + float(ans in nums)
    return Task(f"reason:sum:{a}+{b}+{c}",
                f"Question: what is {a} + {b} + {c}? Let's work it out step by step.\n",
                reward, best=1.05)


def reason_easy_task(a=5, b=3):
    """LEARNABLE reasoning variant: single-digit sum (answer <= 9, one token) with a few-shot prompt
    that PRIMES the base model to emit a number after '=' -> nonzero base rate, so EM/GRPO have
    contrast to climb (the 3-number `reason` task is undiscoverable for a 160M base model: it sits at
    the floor, cf. the tabular flat-reward finding). The first integer of the completion is the
    model's answer; verifier reward = 1 if it equals a+b. Use with the warm-started EM variants
    (`EM-l1-warm` / `EM-barber-warm`) for an even stronger initial signal."""
    ans = a + b                                           # keep a + b <= 9
    shots = "3 + 4 = 7\n1 + 5 = 6\n2 + 2 = 4\n"           # correct demos of OTHER sums (prime format)
    def reward(completion):
        nums = re.findall(r"-?\d+", completion)
        return 0.05 + float(bool(nums) and int(nums[0]) == ans)
    return Task(f"reason_easy:{a}+{b}", f"{shots}{a} + {b} =", reward, best=1.05)


def _first_int_task(name, shots, query, ans):
    def reward(completion):
        nums = re.findall(r"-?\d+", completion)
        return 0.05 + float(bool(nums) and int(nums[0]) == ans)
    return Task(name, f"{shots}{query}", reward, best=1.05)


def reason_hard_task(a=2, b=4, c=1):
    """Sparser reasoning: THREE single-digit terms (more to get right -> lower base rate than the
    2-term reason_easy). Tests the boundary's other side -- where reward is sparse, the doc's eq-8
    policy factor (bootstrap) should matter more relative to EM-L1."""
    return _first_int_task(f"reason_hard:{a}+{b}+{c}", "1 + 2 + 3 = 6\n2 + 2 + 1 = 5\n3 + 1 + 2 = 6\n",
                           f"{a} + {b} + {c} =", a + b + c)


def reason_2d_task(a=21, b=14):
    """Sparser still: two-digit addition (the answer is multi-token and arithmetic is harder for a
    160M base model -> low base rate, the regime where bootstrap / warm-start should help most)."""
    return _first_int_task(f"reason_2d:{a}+{b}", "11 + 22 = 33\n31 + 14 = 45\n12 + 25 = 37\n",
                           f"{a} + {b} =", a + b)


# --------------------------------------------------------------------------- #
#  GSM8K: real grade-school math word problems (the doc's suggested test case).
#  Multi-prompt + answer-aware: each prompt is a few-shot-CoT'd question, reward = 1 if the
#  completion's final number matches that question's gold_answer (verifier, doc eq 11).
# --------------------------------------------------------------------------- #
def _gsm8k_gold(answer_field):
    m = re.search(r"####\s*(-?[\d,]+)", answer_field)
    return int(m.group(1).replace(",", "")) if m else None


def _final_int(text, answer_event_mode="legacy"):
    """Return the parsed GSM8K answer under the declared event contract."""

    return parse_gsm8k_answer_event(
        text,
        mode=answer_event_mode,
    ).answer


_GSM8K_VALIDATION_SEED = 20260716
_GSM8K_VALIDATION_SIZE = 400
GSM8K_DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def _gsm8k_train_validation_pools(n_items):
    """Return fixed, disjoint indices into GSM8K's official training split."""
    if n_items <= _GSM8K_VALIDATION_SIZE:
        raise ValueError(
            f"GSM8K training split has {n_items} rows; need more than "
            f"{_GSM8K_VALIDATION_SIZE} to reserve validation data"
        )
    order = np.random.default_rng(_GSM8K_VALIDATION_SEED).permutation(n_items)
    validation = order[:_GSM8K_VALIDATION_SIZE]
    train = order[_GSM8K_VALIDATION_SIZE:]
    return train, validation


class GSM8KTask(MultiPromptTask):
    """GSM8K word problems with a shared few-shot CoT preamble. ``prompts`` carry the questions and
    ``self.gold_answer`` the per-prompt answers; ``reward`` looks each completion's answer up by its prompt
    id. Use with the multi-prompt runners (run_sweep_lm.py / the trainers in methods.py)."""
    floor = 0.05                                          # reward = floor + correct (see Task.floor)
    def __init__(self, n_prompts=64, n_shots=4, split="train", seed=0,
                 train_partition="all", shot_bank_size=None,
                 answer_event_mode="legacy"):
        from datasets import load_dataset
        ds = load_dataset(
            "openai/gsm8k",
            "main",
            split=split,
            revision=GSM8K_DATASET_REVISION,
        )
        rng = np.random.default_rng(seed)
        if train_partition == "all":
            pool = np.arange(len(ds), dtype=int)
        elif train_partition == "train":
            if split != "train":
                raise ValueError("train_partition='train' requires the GSM8K training split")
            pool, _validation = _gsm8k_train_validation_pools(len(ds))
        else:
            raise ValueError(f"unknown GSM8K train partition {train_partition!r}")
        if shot_bank_size is None:
            question_offset = n_shots
        else:
            shot_bank_size = int(shot_bank_size)
            if shot_bank_size < n_shots:
                raise ValueError(
                    "shot_bank_size must be at least n_shots, got "
                    f"{shot_bank_size} < {n_shots}"
                )
            question_offset = shot_bank_size
        if question_offset + n_prompts > len(pool):
            raise ValueError(
                f"requested a {question_offset}-row shot bank/offset and "
                f"{n_prompts} prompts from only {len(pool)} rows"
            )
        idx = rng.permutation(pool)
        shot_qi = idx[:n_shots]
        shots = "".join(f"Question: {ds[int(i)]['question']}\nAnswer: {ds[int(i)]['answer']}\n\n"
                        for i in shot_qi)
        qi = idx[question_offset:question_offset + n_prompts]
        self.questions = [ds[int(i)]["question"] for i in qi]
        prompts = [shots + f"Question: {question}\nAnswer:" for question in self.questions]
        self.gold_answer = [_gsm8k_gold(ds[int(i)]["answer"]) for i in qi]   # parsed final numeric answer
        self.gold_solution = [ds[int(i)]["answer"] for i in qi]              # full official solution
        self.shot_qi = [int(i) for i in shot_qi]          # few-shot provenance / validation-leakage audit
        self.shot_bank_size = question_offset
        self.train_qi = [int(i) for i in qi]             # exact train-split indices (provenance: contamination answer)
        self.train_partition = train_partition
        self.answer_event_mode = str(answer_event_mode)
        super().__init__(f"gsm8k:{n_prompts}q:{n_shots}shot", prompts, reward_fn=None, best=1.0)
        self.max_new = 256                               # CoT needs room to reach a final answer (eq-11)

    def reward(self, completions, pids=None):
        if pids is None:                                 # single-prompt fallback: use prompt 0's gold_answer
            pids = [0] * len(completions)
        out = np.empty(len(completions))
        for i, c in enumerate(completions):
            pred = _final_int(c, self.answer_event_mode)
            out[i] = 0.05 + float(pred is not None and pred == self.gold_answer[int(pids[i])])
        return out


def gsm8k_task():
    return GSM8KTask()


MATH_CALIBRATION_CONFIG = Path(__file__).with_name(
    "qwen3_8b_math_instruction_qualification.yaml"
)
MATH_CHAT_CALIBRATION_CONFIG = Path(__file__).with_name(
    "qwen3_8b_math_chat_calibration.yaml"
)


@lru_cache(maxsize=1)
def _hendrycks_math_partition():
    """Load the pinned train-only MATH partition shared with 8B calibration."""

    from evaluate_qwen3_8b_math_base_calibration import (
        load_math_train_records,
        load_protocol,
        partition_instruction_successor_records,
        partition_successor_records,
    )
    from math_prompting import MATH_INSTRUCTION_PROMPT_VERSION

    protocol, _digest = load_protocol(MATH_CALIBRATION_CONFIG)
    records, sources = load_math_train_records(protocol)
    partition = protocol["partition"]
    if protocol["prompt"]["version"] == MATH_INSTRUCTION_PROMPT_VERSION:
        (
            validation,
            demonstrations,
            prior_qualification,
            qualification,
            optimization,
        ) = partition_instruction_successor_records(
            records,
            seed=int(partition["seed"]),
            validation_size=int(partition["validation_size"]),
            shots=int(partition["shots"]),
            prior_qualification_size=int(partition["prior_qualification_size"]),
            qualification_size=int(partition["qualification_size"]),
        )
    else:
        validation, demonstrations, qualification, optimization = (
            partition_successor_records(
                records,
                seed=int(partition["seed"]),
                validation_size=int(partition["validation_size"]),
                shots=int(partition["shots"]),
                qualification_size=int(partition["qualification_size"]),
            )
        )
        prior_qualification = []
    if (
        len(optimization)
        + len(validation)
        + len(demonstrations)
        + len(prior_qualification)
        + len(qualification)
        != len(records)
    ):
        raise AssertionError("MATH train partition is not exhaustive")
    if {row["dataset_id"] for row in optimization} & {
        row["dataset_id"] for row in prior_qualification + qualification
    }:
        raise AssertionError("a MATH qualification row entered optimization")
    return protocol, optimization, validation, demonstrations, sources


def _math_preamble(demonstrations, n_shots):
    from math_prompting import build_math_preamble

    return build_math_preamble(demonstrations[:int(n_shots)])


class HendrycksMathTask(MultiPromptTask):
    """Pinned train-only Hendrycks MATH task with symbolic answer equivalence."""

    floor = 0.05

    def __init__(
        self,
        n_prompts=128,
        n_shots=4,
        seed=0,
        train_partition="train",
        shot_bank_size=None,
        answer_event_mode="strict_terminal_marker",
    ):
        if train_partition != "train":
            raise ValueError("Hendrycks MATH training requires train_partition='train'")
        protocol, optimization, _validation, demonstrations, sources = (
            _hendrycks_math_partition()
        )
        if shot_bank_size not in (None, 0, len(demonstrations)):
            raise ValueError(
                "Hendrycks MATH uses the fixed calibration demonstration bank"
            )
        if not 0 <= int(n_shots) <= len(demonstrations):
            raise ValueError("MATH shots exceed the fixed demonstration bank")
        if not 1 <= int(n_prompts) <= len(optimization):
            raise ValueError("invalid MATH optimization question count")
        order = np.random.default_rng(int(seed)).permutation(len(optimization))
        selected_indices = [int(index) for index in order[:int(n_prompts)]]
        selected = [optimization[index] for index in selected_indices]
        from math_prompting import MATH_PROMPT_VERSION, build_math_prompts

        prompts = build_math_prompts(
            selected,
            demonstrations[:int(n_shots)],
            version=str(
                protocol.get("prompt", {}).get("version", MATH_PROMPT_VERSION)
            ),
        )
        self.questions = [row["problem"] for row in selected]
        self.gold_answer = [row["gold"] for row in selected]
        self.gold_solution = [row["canonical_solution"] for row in selected]
        self.train_qi = [
            int(next_index)
            for next_index in selected_indices
        ]
        self.train_dataset_ids = [row["dataset_id"] for row in selected]
        self.shot_qi = [row["dataset_id"] for row in demonstrations[:int(n_shots)]]
        self.shot_bank_size = len(demonstrations)
        self.train_partition = "train"
        self.answer_event_mode = str(answer_event_mode)
        self.dataset_sources = sources
        self.dataset_revision = protocol["dataset"]["revision"]
        super().__init__(
            f"hendrycks_math:{n_prompts}q:{n_shots}shot",
            prompts,
            reward_fn=None,
            best=1.05,
        )
        self.max_new = int(protocol["generation"]["max_new_tokens"])

    @staticmethod
    def parse_answer_event(text, *, mode="legacy"):
        return parse_math_answer_event(text, mode=mode)

    @staticmethod
    def answers_equivalent(left, right):
        return math_answers_equivalent(left, right)

    def reward(self, completions, pids=None):
        if pids is None:
            pids = [0] * len(completions)
        out = np.empty(len(completions), dtype=np.float64)
        for index, (completion, pid) in enumerate(zip(completions, pids)):
            event = self.parse_answer_event(
                completion,
                mode=self.answer_event_mode,
            )
            correct = event.answer is not None and self.answers_equivalent(
                event.answer,
                self.gold_answer[int(pid)],
            )
            out[index] = self.floor + float(correct)
        return out


def hendrycks_math_task():
    return HendrycksMathTask()


@lru_cache(maxsize=1)
def _hendrycks_math_chat_partition():
    """Load the additive post-trained chat partition beginning at position 532."""

    from evaluate_qwen3_8b_math_base_calibration import (
        load_math_train_records,
        load_protocol,
        partition_instruction_successor_records,
    )

    protocol, _digest = load_protocol(MATH_CHAT_CALIBRATION_CONFIG)
    records, sources = load_math_train_records(protocol)
    partition = protocol["partition"]
    validation, demonstrations, prior, qualification, optimization = (
        partition_instruction_successor_records(
            records,
            seed=int(partition["seed"]),
            validation_size=int(partition["validation_size"]),
            shots=int(partition["shots"]),
            prior_qualification_size=int(partition["prior_qualification_size"]),
            qualification_size=int(partition["qualification_size"]),
        )
    )
    if [len(part) for part in (validation, demonstrations, prior, qualification, optimization)] != [
        400,
        4,
        96,
        32,
        6968,
    ]:
        raise AssertionError("Qwen3 chat MATH partition sizes changed")
    reserved_ids = {
        row["dataset_id"]
        for row in validation + demonstrations + prior + qualification
    }
    if reserved_ids & {row["dataset_id"] for row in optimization}:
        raise AssertionError("a reserved MATH row entered chat optimization")
    if partition["optimization_positions"] != [532, 7500]:
        raise AssertionError("Qwen3 chat optimization boundary changed")
    return protocol, optimization, validation, demonstrations, sources


class HendrycksMathChatTask(HendrycksMathTask):
    """Pinned post-trained Qwen3 chat task; rendered only after tokenizer binding."""

    def __init__(
        self,
        n_prompts=128,
        n_shots=4,
        seed=0,
        train_partition="train",
        shot_bank_size=None,
        answer_event_mode="strict_terminal_marker",
    ):
        if train_partition != "train":
            raise ValueError("MATH chat training requires train_partition='train'")
        protocol, optimization, _validation, demonstrations, sources = (
            _hendrycks_math_chat_partition()
        )
        if int(n_shots) != 4 or shot_bank_size not in (None, 0, 4):
            raise ValueError("MATH chat uses the exact four-turn demonstration bank")
        if not 1 <= int(n_prompts) <= len(optimization):
            raise ValueError("invalid MATH chat optimization question count")
        order = np.random.default_rng(int(seed)).permutation(len(optimization))
        selected_indices = [int(index) for index in order[:int(n_prompts)]]
        selected = [optimization[index] for index in selected_indices]

        from math_prompting import build_math_chat_messages

        version = str(protocol["prompt"]["version"])
        self._message_sets = build_math_chat_messages(
            selected,
            demonstrations,
            version=version,
        )
        self._prompt_version = version
        self._tokenizer = None
        self.rendered_chat_prompts = True
        self.questions = [row["problem"] for row in selected]
        self.gold_answer = [row["gold"] for row in selected]
        self.gold_solution = [row["canonical_solution"] for row in selected]
        self.train_qi = selected_indices
        self.train_dataset_ids = [row["dataset_id"] for row in selected]
        self.shot_qi = [row["dataset_id"] for row in demonstrations]
        self.shot_bank_size = 4
        self.train_partition = "train"
        self.answer_event_mode = str(answer_event_mode)
        self.dataset_sources = sources
        self.dataset_revision = protocol["dataset"]["revision"]
        self.protocol = protocol
        self.max_new = int(protocol["generation"]["max_new_tokens"])
        # Message objects are safe placeholders until the exact tokenizer is
        # loaded.  Every training entry point binds them before generation.
        MultiPromptTask.__init__(
            self,
            f"hendrycks_math_chat:{n_prompts}q:4shot",
            list(self._message_sets),
            reward_fn=None,
            best=1.05,
        )

    @staticmethod
    def parse_answer_event(text, *, mode="legacy"):
        return parse_math_answer_event(
            text,
            mode=mode,
            disallowed_exact_answers=("answer",),
        )

    def bind_runtime(self, model, tokenizer) -> dict:
        from math_prompting import (
            bind_math_chat_generation_runtime,
            render_math_chat_prompts,
        )

        contract = bind_math_chat_generation_runtime(
            model,
            tokenizer,
            version=self._prompt_version,
            reset_audit=True,
        )
        self.prompts = render_math_chat_prompts(
            tokenizer,
            self._message_sets,
            version=self._prompt_version,
        )
        rendered_ids = tokenizer(
            self.prompts,
            add_special_tokens=False,
        ).input_ids
        template_ids = [
            tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for messages in self._message_sets
        ]
        rendered_ids = [
            [int(token_id) for token_id in row]
            for row in rendered_ids
        ]
        template_ids = [
            [int(token_id) for token_id in row]
            for row in template_ids
        ]
        if rendered_ids != template_ids:
            raise ValueError(
                "rendered Qwen3 chat prompts do not match template token IDs"
            )
        max_prompt_tokens = max(len(row) for row in rendered_ids)
        if max_prompt_tokens + self.max_new > int(
            model.config.max_position_embeddings
        ):
            raise ValueError("Qwen3 chat prompt plus fixed cap exceeds context")
        prompt_ids_sha256 = hashlib.sha256(
            json.dumps(
                rendered_ids,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        contract.update(
            prompt_tokenization="rendered_add_special_tokens_false",
            prompt_count=len(rendered_ids),
            max_prompt_tokens=max_prompt_tokens,
            prompt_utf8_sha256=hashlib.sha256(
                json.dumps(
                    self.prompts,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            prompt_token_ids_sha256=prompt_ids_sha256,
            fixed_max_new_tokens=self.max_new,
            proposal_prompt_token_parity_checks=0,
        )
        self.prompt = self.prompts[0]
        self._tokenizer = tokenizer
        self.runtime_contract = contract
        return contract

    def build_proposal_prompt(self, question_id: int, mode: str) -> str:
        """Render answer-guided support proposals as a valid final user turn."""

        if self._tokenizer is None:
            raise RuntimeError("MATH chat task must bind its tokenizer before prompting")
        question_id = int(question_id)
        if mode == "question":
            return self.prompts[question_id]
        if mode not in {"answer_derive", "answer_derive_first"}:
            raise ValueError(f"unsupported MATH chat proposal mode: {mode}")
        target = str(self.gold_answer[question_id])
        messages = [dict(message) for message in self._message_sets[question_id]]
        if mode == "answer_derive":
            guidance = (
                f"For support construction, the correct final answer is {target}. "
                "Derive a complete step-by-step solution that leads to it and "
                f"end with #### {target}."
            )
        else:
            guidance = (
                f"For verification only, the correct final answer is {target}. "
                "Solve forward from the stated quantities without citing that "
                f"hint until the final line, then end with #### {target}."
            )
        messages[-1]["content"] = f"{guidance}\n\n{messages[-1]['content']}"
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rendered_ids = self._tokenizer(
            rendered,
            add_special_tokens=False,
        ).input_ids
        template_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if [int(value) for value in rendered_ids] != [
            int(value) for value in template_ids
        ]:
            raise ValueError("Qwen3 proposal prompt token parity changed")
        self.runtime_contract["proposal_prompt_token_parity_checks"] += 1
        return rendered


def hendrycks_math_chat_task():
    return HendrycksMathChatTask()


# --------------------------------------------------------------------------- #
#  DEPRECATED -- IMDB controlled sentiment. The study now focuses on tasks that REQUIRE
#  reasoning (a latent chain-of-thought before a verifiable answer), which is where AC-EM's
#  answer-conditioning applies; sentiment has no latent reasoning to optimise. Kept runnable
#  for reproducing old PO-vs-RL results only. Reasoning-dataset candidates to add instead:
#    MATH (Hendrycks, \boxed{} answers -- the harder drop-in for GSM8K), SVAMP / ASDiv / MAWPS
#    (cheap math-word-problem variety), GSM-Symbolic / GSM-Plus (robustness perturbations),
#    StrategyQA (multi-hop yes/no), CommonsenseQA / ARC-Challenge (multiple-choice reasoning).
#  All share GSM8K's shape: short verifiable gold answer + benefit from CoT -> AC-EM teacher-forces
#  the gold answer, reward = exact-match verifier. (See run_sweep_lm BENCHMARKS for the eval side.)
# --------------------------------------------------------------------------- #
class IMDBSentimentTask(MultiPromptTask):
    """DEPRECATED (non-reasoning). See the note above for reasoning-dataset replacements."""
    def __init__(self, n_prompts=32, prefix_words=8, seed=0, split="train"):
        from datasets import load_dataset
        ds = load_dataset("imdb", split=split)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(ds))[:n_prompts]
        prompts = [" ".join(ds[int(i)]["text"].split()[:prefix_words]) for i in idx]
        super().__init__(f"imdb_sentiment:{n_prompts}p", prompts, reward_fn=None, best=1.0)

    def reward(self, completions, pids=None):
        from benchmark import _sentiment_scores                # lazy: avoids loading clf at import
        return _sentiment_scores([c if c.strip() else " " for c in completions])  # P(positive) in (0,1)


def imdb_task():
    return IMDBSentimentTask()


TASKS = {"dense": dense_task, "structured": structured_task, "sparse": sparse_task,
         "sparse3": sparse3_task, "dense_mp": dense_mp_task, "reason": reasoning_task,
         "reason_easy": reason_easy_task, "reason_hard": reason_hard_task,
         "reason_2d": reason_2d_task, "gsm8k": gsm8k_task,
         "hendrycks_math": hendrycks_math_task,
         "hendrycks_math_chat": hendrycks_math_chat_task, "imdb": imdb_task}
