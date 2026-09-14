"""Text-only boundary audit of saved Q5 evaluation completions; no inference."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def inspect_completion(text):
    rationale = text.split("####", 1)[0].strip()
    lines = [line.strip() for line in rationale.splitlines() if line.strip()]
    first = lines[0] if lines else ""
    return {
        "nonempty_lines": len(lines),
        "first_line_has_equation_sign": "=" in first,
        "first_line_has_numeric_operation": bool(re.search(r"\d\s*[+*/-]\s*\d", first)),
        "first_line_word_fraction": len(first.split())/len(rationale.split()) if rationale else None,
        "first_line": first,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, sources = [], []
    for seed in (1201, 1213, 1217):
        paths = list(args.artifact_dir.glob(f"dump_*68078ecc*Q5-AD-M-LR1e-5-U1-K16_seed{seed}__*.json"))
        if len(paths) != 1:
            raise ValueError(f"expected one historical moving-Q5 dump for seed {seed}")
        path = paths[0]
        payload = json.loads(path.read_text())
        if (payload["run_id"] != "68078ecc" or payload["seed"] != seed
                or payload.get("eval_official_test_accessed") is not False
                or payload.get("eval_dataset_splits_loaded") != ["train"]):
            raise ValueError("dump provenance or train-only boundary changed")
        samples = payload["samples"]
        if len(samples) != 100 or len({s["idx"] for s in samples}) != 100:
            raise ValueError("expected 100 distinct saved completions, not all 400 evaluations")
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for sample in samples:
            records.append(dict(seed=seed, question_id=sample["idx"], **inspect_completion(sample["completion"])))
    summary = {
        "scope": "300 saved final evaluation completions from three historical moving-Q5 seeds; not training buffers or the full 400-question evaluation pool per seed",
        "method": "Text-only syntactic audit; an equation sign is not a semantic understanding label. No tokenization, scoring or generation.",
        "records": len(records),
        "single_line_rationales": sum(r["nonempty_lines"] == 1 for r in records),
        "first_line_equation_sign": sum(r["first_line_has_equation_sign"] for r in records),
        "first_line_numeric_operation": sum(r["first_line_has_numeric_operation"] for r in records),
        "sources": sources, "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"sources", "samples"}}, indent=2))


if __name__ == "__main__":
    main()
