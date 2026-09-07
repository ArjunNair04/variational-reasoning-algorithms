import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'experiments'


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_archived_source_and_config_hashes_and_syntax():
    manifest = json.loads((EXPERIMENTS / 'source_manifest.json').read_text())
    for name, record in manifest['bundles'].items():
        path = EXPERIMENTS / 'frozen' / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        with tarfile.open(path) as archive:
            assert set(archive.getnames()) == set(record['files'])
            for member in archive:
                assert member.isfile()
                content = archive.extractfile(member).read()
                meta = record['files'][member.name]
                assert hashlib.sha256(content).hexdigest() == meta['sha256'] == meta['source_sha256']
                assert meta['changes'] == []
                if member.name.endswith('.py'):
                    ast.parse(content)
                assert all(token not in content for token in [b'/Users/', b'/home/'])
    for name, record in manifest['configurations'].items():
        assert hashlib.sha256((EXPERIMENTS / name).read_bytes()).hexdigest() == record['sha256'] == record['original_sha256']


def test_runtime_selection_changes_only_location_and_existing_cell(tmp_path):
    yaml = pytest.importorskip('yaml')
    launcher = load('reproduce')
    source = EXPERIMENTS / 'configs/experiments_qwen3_17b_final_method_confirmation.yaml'
    original = yaml.safe_load(source.read_text())
    runtime = launcher.runtime_config(source, tmp_path / 'results', 'Q5-LR1e-5-U1-K16')
    assert len(runtime['algos']['AC-ALG1']) == 1
    cell = runtime['algos']['AC-ALG1'][0]
    assert cell in original['algos']['AC-ALG1']
    assert cell['proposal_prompt'] == 'answer_derive'
    assert cell['buffer_semantics'] == 'unique_set'
    assert cell['responsibility_refresh'] == 'inner_step'
    assert cell['answer_target_termination'] == 'eos'
    runtime['algos'] = original['algos'];runtime['defaults']['out'] = original['defaults']['out']
    assert runtime == original
    importance = launcher.runtime_config(source, tmp_path / 'results', 'PIS-S8-B8-U4')['algos']['AC-ALG1'][0]
    assert importance['variational_estimator'] == 'prior_importance'
    assert importance['buffer_lifecycle'] == 'fresh_round'
    assert importance['responsibility_refresh'] == 'outer_round'
    with pytest.raises(ValueError, match='Expected one'):
        launcher.runtime_config(source, tmp_path, 'invented-cell')


def test_prepare_refuses_to_overwrite_edited_source(tmp_path):
    launcher = load('reproduce')
    study = json.loads((EXPERIMENTS / 'studies.json').read_text())['selected_methods']
    launcher.prepare(study, tmp_path)
    edited = tmp_path / 'lm_study/ac_alg1.py';edited.write_text('# local edit\n')
    with pytest.raises(ValueError, match='overwrite changed source'):
        launcher.prepare(study, tmp_path)


def test_frozen_numeric_reconstruction():
    result = load('results').verify()
    assert result['transfer_responses_accounted_for'] == 42000
    assert result['fixed_Q_transitions'] == 288
