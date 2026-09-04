#!/usr/bin/env python3
"""Validate and analyse the evaluation-only GSM8K panel-robustness study."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _bootstrap_difference(records, method, reference, seeds, panels, reps=10000):
    rng = np.random.default_rng(20260904)
    values = np.empty(reps, dtype=float)
    for rep in range(reps):
        seed_draw = rng.choice(seeds, size=len(seeds), replace=True)
        panel_draw = rng.choice(panels, size=len(panels), replace=True)
        terms = []
        for seed in seed_draw:
            for panel in panel_draw:
                candidate = records[(method, int(seed), int(panel))]
                base = records[(reference, int(seed), int(panel))]
                indices = rng.integers(0, len(candidate), size=len(candidate))
                terms.append(
                    float(
                        np.mean(
                            candidate[indices].astype(float)
                            - base[indices].astype(float)
                        )
                    )
                )
        values[rep] = float(np.mean(terms))
    return values


def load_and_validate(config_path: Path, result_root: Path):
    from generate_qwen3_17b_gsm8k_panel_robustness import validate_payload

    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    validate_payload(config)
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    methods = [row["method"] for row in config["methods"]]
    seeds = [int(value) for value in config["design"]["seeds"]]
    panel_ids = [int(row["panel_id"]) for row in config["source_contract"]["panels"]]
    expected_indices = {
        int(row["panel_id"]): [int(value) for value in row["dataset_train_indices"]]
        for row in config["source_contract"]["panels"]
    }
    records = {}
    receipts = []
    for method in methods:
        for seed in seeds:
            receipt_path = result_root / f"receipt__{method}__seed{seed}.json"
            if not receipt_path.is_file():
                raise FileNotFoundError(receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            required = {
                "status": "complete",
                "run_id": config["run_id"],
                "source_run_id": config["source_run_id"],
                "method": method,
                "seed": seed,
                "training_performed": False,
                "official_test_used": False,
                "configuration_sha256": config_sha,
            }
            for key, value in required.items():
                if receipt.get(key) != value:
                    raise ValueError(f"{receipt_path}: {key} mismatch")
            if receipt.get("dataset_splits_loaded") != ["train"]:
                raise ValueError(f"{receipt_path}: unexpected dataset access")
            artifacts = {row["path"]: row for row in receipt.get("artifacts", [])}
            if len(artifacts) != len(panel_ids):
                raise ValueError(f"{receipt_path}: incomplete artifact manifest")
            for panel_id in panel_ids:
                name = f"panel__{method}__seed{seed}__p{panel_id}.json.gz"
                path = result_root / name
                manifest = artifacts.get(name)
                if (
                    manifest is None
                    or not path.is_file()
                    or _sha256(path) != manifest["sha256"]
                ):
                    raise ValueError(f"{path}: missing or hash mismatch")
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    rows = json.load(handle)
                if len(rows) != 400:
                    raise ValueError(f"{path}: expected 400 records")
                observed = [int(row["dataset_train_index"]) for row in rows]
                if observed != expected_indices[panel_id]:
                    raise ValueError(f"{path}: panel membership/order mismatch")
                if any(
                    row.get("official_test_accessed") is not False
                    or row.get("dataset_splits_loaded") != ["train"]
                    for row in rows
                ):
                    raise ValueError(f"{path}: official-test prohibition violated")
                records[(method, seed, panel_id)] = np.asarray(
                    [bool(row["legacy_correct"]) for row in rows], dtype=bool
                )
                records[(method + "::strict", seed, panel_id)] = np.asarray(
                    [bool(row["strict_correct"]) for row in rows], dtype=bool
                )
            receipts.append(receipt)
    return config, methods, seeds, panel_ids, records, receipts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-design-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from generate_qwen3_17b_gsm8k_panel_robustness import validate_payload

    validate_payload(config)
    if args.validate_design_only:
        print(json.dumps({"run_id": config["run_id"], "tasks": 42, "panels": 3}))
        return 0
    result_root = (
        args.result_root or Path(config["design"]["output_root"])
    ).expanduser()
    output_dir = (args.output_dir or result_root / "analysis").expanduser()
    config, methods, seeds, panels, records, receipts = load_and_validate(
        args.config, result_root
    )
    reference = config["reporting"]["reference"]
    summary = []
    for method in methods:
        extracted = [
            records[(method, seed, panel)] for seed in seeds for panel in panels
        ]
        strict = [
            records[(method + "::strict", seed, panel)]
            for seed in seeds
            for panel in panels
        ]
        row = {
            "method": method,
            "final_acc1": float(np.mean(np.concatenate(extracted))),
            "strict_final_acc1": float(np.mean(np.concatenate(strict))),
            "paired_blocks": len(seeds) * len(panels),
            "evaluation_questions_per_block": 400,
        }
        if method != reference:
            draws = _bootstrap_difference(records, method, reference, seeds, panels)
            row["delta_final_acc1_vs_base"] = float(draws.mean())
            row["delta_final_acc1_vs_base_95"] = _interval(draws)
        summary.append(row)
    summary.sort(key=lambda row: row["final_acc1"], reverse=True)

    rank_rows = []
    wins = {method: 0.0 for method in methods}
    for seed in seeds:
        for panel in panels:
            scores = {
                method: float(records[(method, seed, panel)].mean())
                for method in methods
            }
            best = max(scores.values())
            winners = [method for method, score in scores.items() if score == best]
            for method in winners:
                wins[method] += 1.0 / len(winners)
            rank_rows.append(
                {"seed": seed, "panel": panel, "scores": scores, "winners": winners}
            )
    for row in summary:
        row["first_place_share"] = wins[row["method"]] / len(rank_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": config["run_id"],
                "official_test_used": False,
                "training_performed": False,
                "methods": summary,
                "paired_panel_rankings": rank_rows,
                "receipt_count": len(receipts),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "method_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# GSM8K fresh-panel ranking robustness",
        "",
        "Final Acc@1 is reported first. Strict final accuracy is second. This is final-adapter evaluation only, so trajectory AUC is unavailable.",
        "",
        "| Method | Final Acc@1 | Strict final | First-place share |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {100 * row['final_acc1']:.2f}% | "
            f"{100 * row['strict_final_acc1']:.2f}% | {100 * row['first_place_share']:.1f}% |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": config["run_id"], "methods": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
