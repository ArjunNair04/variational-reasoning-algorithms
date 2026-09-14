"""Revalidate scoring evidence before the approved marker-fixed training screen."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile

from analyze_q5_segment_scoring import analyze
from analyze_qwen3_q5_prior_segments import validate_design
from audit_q5_prior_segments_pretraining import sha256
from generate_qwen3_17b_q5_prior_segments import SCORING_EVIDENCE, RUN_ID

EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "docs/experiments/qwen3_q5_prior_segments"


def require_training_gate(config, source, results, logs, evidence_dir=EVIDENCE_DIR):
    payload, _ = validate_design(config)
    if payload["diagnostic"]["pretraining_evidence"] != SCORING_EVIDENCE:
        raise ValueError("training evidence or approved marker convention changed")
    manifest = evidence_dir / "scoring_manifest.json"
    summary = evidence_dir / "scoring_summary_2026-09-14.json"
    for path, key in ((manifest, "manifest_sha256"), (summary, "summary_sha256")):
        if sha256(path) != SCORING_EVIDENCE[key]:
            raise ValueError("frozen scoring evidence checksum mismatch")
    expected = json.loads(summary.read_text())
    # Recheck original receipts, source files, token tapes and full logs. Never
    # edit the old partial gate or treat its boolean as training authorization.
    with tempfile.TemporaryDirectory(prefix="q5-segment-gate-") as directory:
        output = Path(directory) / "validation"
        with contextlib.redirect_stdout(io.StringIO()):
            analyze(source, results, manifest, SCORING_EVIDENCE["execution_commit"],
                    SCORING_EVIDENCE["job"], logs, output)
        observed = json.loads((output / "summary.json").read_text())
    for key in ("status", "audit_id", "supports", "traces", "execution_commit",
                "job", "manifest_sha256", "receipts"):
        if observed[key] != expected[key]:
            raise ValueError(f"revalidated scoring identity differs: {key}")
    return dict(status="ok", training_run_id=RUN_ID,
                segment_scope=SCORING_EVIDENCE["selected_scope"],
                scoring_job=SCORING_EVIDENCE["job"],
                supports=observed["supports"], traces=observed["traces"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-training-gate", type=Path, required=True, metavar="CONFIG")
    for key in ("source", "scoring-results", "scoring-logs"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(require_training_gate(args.check_training_gate, args.source,
                                          args.scoring_results, args.scoring_logs), sort_keys=True))


if __name__ == "__main__":
    main()
