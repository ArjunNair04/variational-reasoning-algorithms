#!/usr/bin/env python3
"""Verify the portable completed-run H100 compute ledger without model evaluation."""
from pathlib import Path
import csv, hashlib, json, math, re

root = Path(__file__).resolve().parent
ledger = root / 'h100_compute_ledger.csv'
summary = json.loads((root / 'h100_compute_audit.json').read_text())
rows = list(csv.DictReader(ledger.open(newline='')))
assert hashlib.sha256(ledger.read_bytes()).hexdigest() == summary['csv_sha256']
assert len(rows) == summary['record_count'] == 970
for field in ('scientific_tag', 'cell_fingerprint', 'source_path', 'source_sha256'):
    assert len({row[field] for row in rows}) == len(rows), field
for row in rows:
    seconds, hours = float(row['wall_seconds']), float(row['h100_hours'])
    assert math.isfinite(seconds) and seconds >= 0
    assert math.isfinite(hours) and hours >= 0
    assert int(row['accelerator_count']) == 1
    assert row['accelerator_name'] == 'NVIDIA H100 80GB HBM3'
    assert math.isclose(hours, seconds / 3600, abs_tol=1e-12)
    for field in ('source_sha256', 'completion_sha256', 'cell_fingerprint'):
        assert re.fullmatch('[0-9a-f]{64}', row[field]), field
    for field in ('source_path', 'completion_path'):
        parts = Path(row[field]).parts
        assert parts[0] in ('research_repository', 'saved_archive')
        assert not Path(row[field]).is_absolute() and '..' not in parts
hours = math.fsum(float(row['h100_hours']) for row in rows)
assert math.isclose(hours, summary['h100_gpu_hours'], abs_tol=1e-10)
assert math.isclose(hours, 808.5897222222222, abs_tol=1e-10)
assert math.isclose(math.fsum(float(r['wall_seconds']) for r in rows), summary['wall_seconds'], abs_tol=1e-8)
for study in summary['study_totals']:
    study_rows = [row for row in rows if row['run_id'] == study['run_id']]
    assert len(study_rows) == study['records']
    assert math.isclose(math.fsum(float(r['h100_hours']) for r in study_rows), study['h100_gpu_hours'], abs_tol=1e-10)
print(f'PASS: {len(rows)} distinct completed-run records, {hours:.9f} H100 GPU-hours')
