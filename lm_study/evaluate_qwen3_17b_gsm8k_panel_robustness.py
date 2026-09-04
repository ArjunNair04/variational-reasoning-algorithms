#!/usr/bin/env python3
"""Evaluate retained final-confirmation adapters on three fresh GSM8K panels."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from answer_events import parse_gsm8k_answer_event
from benchmark import _generate
from generate_qwen3_17b_gsm8k_panel_robustness import validate_payload
from methods_lm import DEV, cuda_dtype
from tasks import _gsm8k_gold


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _write_json_gz(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(json.dumps(payload, sort_keys=True).encode("utf-8"))
    temporary.replace(path)


def _adapter_path(root: Path, fragment: str, seed: int) -> Path:
    matches = sorted(root.glob(f"adapter_gsm8k*{fragment}_seed{seed}*"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one adapter for fragment={fragment!r}, seed={seed}; "
            f"found {len(matches)}"
        )
    path = matches[0]
    if not (path / "adapter_config.json").is_file():
        raise RuntimeError(f"adapter configuration missing: {path}")
    if not any(
        (path / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError(f"adapter weights missing: {path}")
    return path


def _load_model(config: dict, adapter: Path | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_spec = config["model"]
    kwargs = {"revision": model_spec["revision"]}
    tok = AutoTokenizer.from_pretrained(model_spec["hf_id"], **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_spec["hf_id"],
        torch_dtype=cuda_dtype(),
        **kwargs,
    )
    if adapter is None:
        model = base
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    return model.to(DEV), tok


def _prompts(dataset, shot_indices: list[int], panel_indices: list[int]):
    preamble = "".join(
        f"Question: {dataset[index]['question']}\nAnswer: {dataset[index]['answer']}\n\n"
        for index in shot_indices
    )
    return [
        preamble + f"Question: {dataset[index]['question']}\nAnswer:"
        for index in panel_indices
    ]


def _records(dataset, indices, completions, metadata):
    rows = []
    for index, completion, generation in zip(indices, completions, metadata):
        gold = _gsm8k_gold(dataset[index]["answer"])
        legacy = parse_gsm8k_answer_event(completion, mode="legacy")
        strict = parse_gsm8k_answer_event(completion, mode="strict_terminal_marker")
        rows.append(
            {
                "dataset_train_index": int(index),
                "question": dataset[index]["question"],
                "gold": int(gold),
                "completion": completion,
                "legacy_pred": legacy.answer,
                "legacy_correct": bool(legacy.answer == gold),
                "strict_pred": strict.answer,
                "strict_correct": bool(strict.answer == gold),
                "strict_correct_and_eos": bool(
                    strict.answer == gold and generation["generated_eos"]
                ),
                "answer_marker_count": int(strict.marker_count),
                "answer_marker_terminal": bool(strict.terminal_marker),
                "generated_eos": bool(generation["generated_eos"]),
                "generated_tokens_until_eos": int(
                    generation["generated_tokens_until_eos"]
                ),
                "hit_max_new_tokens": bool(generation["hit_max_new_tokens"]),
                "official_test_accessed": False,
                "dataset_splits_loaded": ["train"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    validate_payload(config)
    methods = config["methods"]
    seeds = [int(value) for value in config["design"]["seeds"]]
    task_count = len(methods) * len(seeds)
    if not 1 <= args.task_id <= task_count:
        raise SystemExit(f"task ID must be in [1,{task_count}]")
    zero = args.task_id - 1
    method_spec = methods[zero // len(seeds)]
    seed = seeds[zero % len(seeds)]
    method = method_spec["method"]

    if DEV != "cuda":
        raise RuntimeError("panel evaluation requires a CUDA GPU")
    from datasets import load_dataset

    dataset = load_dataset(
        config["dataset"]["id"],
        config["dataset"]["configuration"],
        split="train",
        revision=config["dataset"]["revision"],
    )
    if len(dataset) != config["dataset"]["row_count"]:
        raise RuntimeError("pinned GSM8K training row count changed")
    adapter_root = Path(config["design"]["source_adapter_root"]).expanduser()
    fragment = method_spec["adapter_name_fragment"]
    adapter = None if fragment is None else _adapter_path(adapter_root, fragment, seed)
    adapter_digest = None if adapter is None else _tree_sha256(adapter)
    model, tok = _load_model(config, adapter)
    output = Path(config["design"]["output_root"]).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / f"receipt__{method}__seed{seed}.json"
    if receipt_path.exists():
        raise RuntimeError(f"receipt already exists: {receipt_path}")

    started = time.time()
    files = []
    summaries = []
    shot_indices = [
        int(value)
        for value in config["source_contract"]["demonstration_indices_by_seed"][seed]
    ]
    for panel in config["source_contract"]["panels"]:
        panel_id = int(panel["panel_id"])
        indices = [int(value) for value in panel["dataset_train_indices"]]
        prompts = _prompts(dataset, shot_indices, indices)
        completions, metadata = _generate(
            model,
            tok,
            prompts,
            max_new=int(config["design"]["max_new_tokens"]),
            batch=int(config["design"]["batch_size"]),
            return_metadata=True,
        )
        rows = _records(dataset, indices, completions, metadata)
        path = output / f"panel__{method}__seed{seed}__p{panel_id}.json.gz"
        _write_json_gz(path, rows)
        files.append({"path": path.name, "sha256": _sha256(path), "records": len(rows)})
        summaries.append(
            {
                "panel_id": panel_id,
                "final_acc1": float(np.mean([row["legacy_correct"] for row in rows])),
                "strict_final_acc1": float(
                    np.mean([row["strict_correct"] for row in rows])
                ),
                "strict_final_acc1_and_eos": float(
                    np.mean([row["strict_correct_and_eos"] for row in rows])
                ),
                "natural_eos_rate": float(
                    np.mean([row["generated_eos"] for row in rows])
                ),
                "mean_generated_tokens": float(
                    np.mean([row["generated_tokens_until_eos"] for row in rows])
                ),
            }
        )
    receipt = {
        "status": "complete",
        "run_id": config["run_id"],
        "source_run_id": config["source_run_id"],
        "task_id": args.task_id,
        "method": method,
        "seed": seed,
        "training_performed": False,
        "official_test_used": False,
        "dataset_splits_loaded": ["train"],
        "execution_commit": os.environ.get("EXPECTED_COMMIT"),
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "adapter_path": None if adapter is None else str(adapter),
        "adapter_sha256": adapter_digest,
        "panels": summaries,
        "artifacts": files,
        "elapsed_seconds": time.time() - started,
        "gpu": torch.cuda.get_device_name(0),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
