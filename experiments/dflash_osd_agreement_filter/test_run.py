"""Qualification must gate a completely fresh campaign; metadata never launches."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from . import run
from .test_experiment import Backend, references, stream
from .experiment import EXPECTED_STREAMS


def test_clean_qualification_then_full_bounded_campaign(tmp_path):
    streams = [stream(name) for name in EXPECTED_STREAMS]
    refs = {s['stream_id']: references(s) for s in streams}
    created = []
    def factory(config):
        value = Backend(config['agreement_arm'])
        created.append((dict(config), value))
        return value
    result = run.run_qualified_campaign(streams, refs, factory, tmp_path, config={}, fixtures={'status': 'passed'})
    assert result['status'] == 'complete'
    assert len(created) == 28 and len({id(x[1]) for x in created}) == 28
    assert all(x[0]['agreement_qualification'] for x in created[:4])
    assert all(not x[0]['agreement_qualification'] for x in created[4:])
    assert len({x[0]['output_root'] for x in created}) == 28
    assert all(x[1].training_state == 2 for x in created)
    assert len(list((tmp_path / 'campaign').glob('*/RESULT.json'))) == 6


def test_failed_qualification_does_not_start_campaign(tmp_path):
    source = stream(); created = []
    def factory(config):
        created.append(config)
        return Backend(config['agreement_arm'], qualifier=False)
    result = run.run_qualified_campaign([source], {source['stream_id']: references(source)}, factory,
        tmp_path, config={}, fixtures={'status': 'passed'})
    assert result['status'] == 'qualification_failed' and result['campaign'] is None
    assert not (tmp_path / 'campaign').exists()
    assert [x['agreement_arm'] for x in created] == ['H', 'L', 'G']


def test_failed_fixture_cannot_construct_backend(tmp_path):
    with pytest.raises(ValueError):
        run.run_qualified_campaign([], {}, lambda config: pytest.fail('unexpected worker'), tmp_path,
                                   config={}, fixtures={'status': 'failed'})


def test_qualification_only_success_never_constructs_campaign_workers(tmp_path):
    streams = [stream(name) for name in EXPECTED_STREAMS]
    refs = {s['stream_id']: references(s) for s in streams}
    created = []
    def factory(config):
        assert config['agreement_qualification'] is True
        backend = Backend(config['agreement_arm'])
        created.append((dict(config), backend))
        return backend
    result = run.run_qualified_campaign(streams, refs, factory, tmp_path, config={},
                                       fixtures={'status': 'passed'}, qualification_only=True)
    assert result['status'] == 'qualification_complete'
    assert result['qualification']['status'] == 'complete'
    assert result['campaign'] is None and result['campaign_started'] is False
    assert result['execution_scope'] == {'mode': 'qualification_only', 'campaign_execution_permitted': False}
    assert 'paused pending review' in result['stop_reason']
    assert [config['agreement_arm'] for config, _ in created] == ['H', 'L', 'G', 'N']
    assert all(backend.training_state == 2 for _, backend in created)
    assert len(list((tmp_path / 'qualification').glob('*/RESULT.json'))) == 1
    assert not (tmp_path / 'campaign').exists()
    assert 'actual_execution_wall_s' in result
    assert 'actual_campaign_wall_s_including_qualification' not in result
    assert json.loads((tmp_path / 'EXECUTION.json').read_text()) == result
    assert json.loads((tmp_path / 'EXECUTION_SCOPE.json').read_text()) == result['execution_scope']


def test_qualification_only_failure_preserves_failure_and_stops(tmp_path):
    source = stream()
    created = []
    def factory(config):
        created.append(config)
        return Backend(config['agreement_arm'], qualifier=False)
    result = run.run_qualified_campaign([source], {source['stream_id']: references(source)}, factory,
                                       tmp_path, config={}, fixtures={'status': 'passed'},
                                       qualification_only=True)
    assert result['status'] == 'qualification_failed'
    assert result['campaign'] is None and result['campaign_started'] is False
    assert 'failed' in result['stop_reason'] and 'paused' in result['stop_reason']
    assert [config['agreement_arm'] for config in created] == ['H', 'L', 'G']
    assert all(config['agreement_qualification'] for config in created)
    assert not (tmp_path / 'campaign').exists()
    record = json.loads(next((tmp_path / 'qualification').glob('*/RESULT.json')).read_text())
    assert record['status'] == 'failed' and record['failed_arm'] == 'G'
    assert json.loads((tmp_path / 'EXECUTION.json').read_text())['status'] == 'qualification_failed'


@pytest.mark.parametrize('action', ['prepare-references', 'plan', 'qualify-text'])
def test_qualification_only_flag_rejects_metadata_actions_before_loading(monkeypatch, tmp_path, action):
    monkeypatch.setattr(run, 'load_streams', lambda _: pytest.fail('metadata flag must reject before loading'))
    with pytest.raises(ValueError, match='applies only to the run action'):
        run.main([action, '--dataset', str(tmp_path / 'dataset'), '--out', str(tmp_path / 'output'),
                  '--qualification-only'])
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('status,expected_exit', [('qualification_complete', 0), ('qualification_failed', 1)])
def test_cli_forwards_explicit_qualification_only_scope(monkeypatch, tmp_path, status, expected_exit):
    streams = [stream(name) for name in EXPECTED_STREAMS]
    refs = {s['stream_id']: references(s) for s in streams}
    source_pins = {'fixture.py': 'fixture-digest'}
    reference_hash = 'a' * 64
    config_path = tmp_path / 'runtime.json'
    config_path.write_text(json.dumps({'output_cap': 1024, 'block_size': 10, 'remaining_diagnostic_cases': 0}))
    monkeypatch.setattr(run, 'load_streams', lambda _: streams)
    monkeypatch.setattr(run, 'source_inventory', lambda: source_pins)
    monkeypatch.setattr(run, 'load_references', lambda *_: refs)
    monkeypatch.setattr(run, 'file_binding', lambda _: {'sha256': reference_hash})
    monkeypatch.setattr(run, 'load_pinned_tokenizer', lambda _: object())
    monkeypatch.setattr(run, 'qualify_text', lambda *_: {'status': 'passed'})
    factory_marker = object()
    monkeypatch.setitem(sys.modules, run.__package__ + '.native',
                        SimpleNamespace(NativeBackend=factory_marker))
    monkeypatch.setitem(sys.modules, run.__package__ + '.qualification',
                        SimpleNamespace(run_deterministic_checks=lambda: {'status': 'passed'}))
    calls = []
    def execute(selected_streams, selected_refs, factory, out, **kwargs):
        calls.append(kwargs)
        assert selected_streams is streams and selected_refs is refs and factory is factory_marker
        assert kwargs['qualification_only'] is True
        return {'status': status, 'campaign': None, 'campaign_started': False}
    monkeypatch.setattr(run, 'run_qualified_campaign', execute)
    code = run.main(['run', '--dataset', str(tmp_path / 'dataset'), '--out', str(tmp_path / 'output'),
                     '--references', str(tmp_path / 'references.json'), '--tokenizer', str(tmp_path / 'tokenizer'),
                     '--runtime-config', str(config_path), '--allow-gpu-execution', '--qualification-only',
                     '--reviewed-code-sha256', run.digest(source_pins),
                     '--reviewed-reference-sha256', reference_hash])
    assert code == expected_exit and len(calls) == 1
