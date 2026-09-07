#!/usr/bin/env python3
"""Inspect or run one original thesis study. Preparation uses only the standard library."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack(archive, destination, manifest):
    """Verify all members before extracting ordinary files into a fresh directory."""
    if sha256(archive) != manifest['sha256']:
        raise ValueError(f'Archive hash mismatch: {archive.name}')
    destination = Path(destination)
    with tarfile.open(archive, 'r:gz') as handle:
        members = handle.getmembers()
        if set(m.name for m in members) != set(manifest['files']):
            raise ValueError('Archive member list differs from the source manifest')
        contents = {}
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Unsafe archive member: {member.name}')
            data = handle.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != manifest['files'][member.name]['sha256']:
                raise ValueError(f'Source hash mismatch: {member.name}')
            target = destination / path
            if target.is_symlink() or any(p.is_symlink() for p in target.parents):
                raise ValueError(f'Symbolic link in destination: {target}')
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Refusing to overwrite changed source: {target}')
            contents[target] = data
        for target, data in contents.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def prepare(study, workdir, continuation=False):
    manifest = json.loads((HERE / 'source_manifest.json').read_text())
    selected = study['continuation'] if continuation else study
    archive = HERE / selected['source_bundle']
    unpack(archive, workdir, manifest['bundles'][archive.name])
    # Analysers are separate: their later source never replaces execution modules.
    name = 'analysis-3472a14.tar.gz'
    unpack(HERE / 'frozen' / name, workdir / 'analysis_tools', manifest['bundles'][name])
    for config in study['configurations']:
        path = HERE / config
        if sha256(path) != manifest['configurations'][config]['sha256']:
            raise ValueError(f'Configuration hash mismatch: {config}')
        target = workdir / 'configs' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return workdir


def runtime_config(source, output, cell_id=None):
    import yaml
    config = yaml.safe_load(source.read_text())
    defaults = config['defaults']
    if defaults.get('task') != 'gsm8k' or defaults.get('eval_partition') != 'validation':
        raise ValueError('Only the frozen GSM8K training/validation studies are launchable here')
    if defaults.get('train_partition', 'train') != 'train':
        raise ValueError('Unexpected optimization partition')
    defaults['out'] = str(output.resolve())
    if cell_id:
        selected = {}
        for method, value in config['algos'].items():
            variants = value if isinstance(value, list) else [value]
            keep = [v for v in variants if v.get('cell_id', method) == cell_id]
            if keep:
                selected[method] = keep if isinstance(value, list) else keep[0]
        if sum(len(v) if isinstance(v, list) else 1 for v in selected.values()) != 1:
            raise ValueError(f'Expected one configured cell named {cell_id!r}')
        config['algos'] = selected
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', nargs='?', default='list')
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--prepare', action='store_true', help='unpack verified original source; no runtime imports')
    parser.add_argument('--execute', action='store_true', help='run the original GPU experiment; default is dry-run')
    parser.add_argument('--cell', help='exact cell_id in the frozen YAML; omission selects all configured cells')
    parser.add_argument('--seed-index', type=int, default=0, help='index into the configured seed list, not a new seed')
    parser.add_argument('--config-index', type=int, default=0)
    parser.add_argument('--continuation', action='store_true', help='use the separately recorded common-grid continuation source')
    args = parser.parse_args()
    studies = json.loads((HERE / 'studies.json').read_text())
    if args.study == 'list':
        for name, study in studies.items():
            print(f'{name:25} {study["thesis"]}')
        return
    if args.study not in studies:
        parser.error('unknown study; run without arguments for the study list')
    if args.workdir is None:
        parser.error('--workdir is required (source and results stay outside the package)')
    if args.prepare and args.execute:
        parser.error('--prepare and --execute are mutually exclusive')
    study = studies[args.study]
    if args.continuation and 'continuation' not in study:
        parser.error('this study has no second execution segment')
    root = prepare(study, args.workdir.resolve(), args.continuation)
    print(f'Original execution source: {root / "lm_study"}', flush=True)
    if args.prepare:
        return
    if study['kind'] == 'evaluation':
        parser.error('transfer needs receipt-bound trained adapters; use --prepare and follow EXPERIMENTS.md')
    if not 0 <= args.config_index < len(study['configurations']):
        parser.error('config-index outside available configurations')
    import yaml
    original = root / 'configs' / Path(study['configurations'][args.config_index]).name
    config = runtime_config(original, root / 'results', args.cell)
    defaults = config['defaults']
    seeds = defaults.get('seed_values', list(range(defaults.get('seeds', 1))))
    if not 0 <= args.seed_index < len(seeds):
        parser.error('seed-index outside the original seed list')
    runtime = root / 'runtime.yaml'
    runtime.write_text(yaml.safe_dump(config, sort_keys=False))
    revision = study["continuation"]["execution_revision"] if args.continuation else study["execution_revision"]
    provenance = {"study": args.study, "execution_revision": revision,
                  "original_config_sha256": sha256(original), "runtime_config_sha256": sha256(runtime),
                  "runtime_changes": ["output directory", "existing cell selection" if args.cell else "no cell filtering"],
                  "seed_index": args.seed_index, "configured_seed": seeds[args.seed_index],
                  "mode": "execute" if args.execute else "dry_run"}
    (root / "replay_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    command = [sys.executable, str(root / 'lm_study' / 'run_yaml.py'), str(runtime), '--seed', str(args.seed_index)]
    if not args.execute:
        command.append('--dry-run')
    print(f'Seed {seeds[args.seed_index]}; {"EXECUTE" if args.execute else "dry-run (no model or data loaded)"}', flush=True)
    if sys.version_info < (3, 11):
        parser.error("the original trainer requires Python 3.11 or newer; preparation works on older Python")
    env = {**os.environ, "EXPECTED_COMMIT": revision, "GIT_CEILING_DIRECTORIES": str(root)}
    subprocess.run(command, cwd=root / 'lm_study', env=env, check=True)


if __name__ == '__main__':
    main()
