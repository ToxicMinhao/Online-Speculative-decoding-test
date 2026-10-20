"""No model or GPU startup: worker dispatch, source/arm isolation and admission."""
from pathlib import Path
import pytest
from . import native


def test_worker_targets_new_entry_and_preserves_runtime(monkeypatch):
    monkeypatch.setattr(native.per_turn, '_original_worker_command', lambda cfg: ['python', str(native.base.HERE / 'native.py')])
    assert native.worker_command({}) == ['python', str(native.HERE / 'native.py')]


def test_source_arm_qualification_checks(monkeypatch):
    monkeypatch.setattr(native.per_turn, 'check_sources', lambda cfg: None)
    cfg = dict(agreement_source_sha256=native.source_pins(), agreement_arm='G', agreement_qualification=True)
    native.check_sources(cfg)
    for key, bad in [('agreement_arm', 'F'), ('agreement_qualification', 1), ('agreement_source_sha256', {})]:
        with pytest.raises(ValueError):
            native.check_sources({**cfg, key: bad})


def test_no_unplanned_diagnostics():
    with pytest.raises(RuntimeError):
        native.NativeBackend({}).diagnostic_event()


def test_native_arm_and_diagnostic_budget_checked_before_start():
    for cfg in [dict(agreement_arm='F', agreement_qualification=False),
                dict(agreement_arm='G', agreement_qualification=False, remaining_diagnostic_cases=1)]:
        with pytest.raises(ValueError):
            native.NativeBackend(cfg).start_episode('wildchat_00')


def test_retirement_requires_valid_scores_and_durable_owned_result(tmp_path, monkeypatch):
    backend = native.NativeBackend(dict(agreement_arm='G'))
    backend.out = tmp_path
    calls = []
    monkeypatch.setattr(backend, 'retire', lambda *args, **kwargs: calls.append((args, kwargs)) or {'status': 'passed'})
    value = dict(status='complete', arm='G', cleanup={'status': 'passed'},
                 canonical=[{'turn': n} for n in (1, 2, 3)],
                 updates=[{'event': n, 'mask_and_schedule_check': True} for n in (1, 2)],
                 scores=[{'same_generation_valid': True} for _ in range(2)])
    invalid = {**value, 'scores': [{'same_generation_valid': False}, {'same_generation_valid': True}]}
    with pytest.raises(ValueError):
        backend.finalize_retention(result=invalid)
    assert not calls and not (tmp_path / 'ARM_RESULT.json').exists()
    result = backend.finalize_retention(result=value)
    assert result['status'] == 'passed' and len(calls) == 1
    assert Path(result['completion_evidence']['path']).read_text()


def test_deterministic_fixtures():
    from .qualification import run_deterministic_checks
    assert run_deterministic_checks()['status'] == 'passed'


def test_training_service_qualifies_once_and_keeps_receipt_for_both_events(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    class Trainer:
        def qualify_disabled_filter(self, responses):
            calls.append(responses)
            return {'status': 'passed', 'temporary_optimizer_steps': 4, 'persistent_optimizer_steps_after': 0}
    service = object.__new__(native.TrainerService)
    service.config = {'agreement_qualification': True, 'agreement_arm': 'G'}
    service.trainer = Trainer(); service.out = tmp_path; service.qualification_receipt = None
    service._check_responses = lambda responses: None
    def parent_train(owner, body):
        return {'event': body['event'], 'accounting': {'batches': [
            {'presentations': [{'gradient_sum_qualification': {'status': 'passed'}}]} for _ in range(2)]}}
    monkeypatch.setattr(native.per_turn.TrainerService, 'train', parent_train)
    first = service.train({'event': 1, 'responses': [{'record_id': 'first'}]})
    second = service.train({'event': 2, 'responses': [{'record_id': 'second'}]})
    assert len(calls) == 1
    assert first['qualification_checks']['status'] == second['qualification_checks']['status'] == 'passed'
    assert 'disabled_filter_qualification' in first and 'disabled_filter_qualification' not in second
    assert (tmp_path / 'DISABLED_FILTER_QUALIFICATION.json').exists()
