"""No model or GPU startup: worker dispatch, source/arm isolation and admission."""
from pathlib import Path
import pytest
from . import native


def _direct_full_presentation(*, sum_status='passed'):
    from .gradient_diagnostics import SUM_ERROR
    return dict(
        ordinary_proposal_gradient='direct full-loss native autograd gradient',
        gradient_backward_calls=3, numerical_protocol=native.training.GROUP_PROTOCOL['schema'],
        sum_tolerance_is_admission_gate=False,
        gradient_sum_diagnostic=dict(
            schema='agreement_gradient_sum_diagnostic_v1', status=sum_status,
            nonfinite_input_coordinates=0,
            nonfinite_counts=dict.fromkeys(
                ('a', 'b', 'sum', 'reference', 'error', 'tolerance', 'error_over_tolerance'), 0),
            first_failure={'error': SUM_ERROR} if sum_status == 'failed' else None))


def _training_result(*, sum_status='passed'):
    return {'accounting': {'batches': [
        {'presentations': [_direct_full_presentation(sum_status=sum_status)]}
        for _ in range(2)]}}


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
        return {'event': body['event'], **_training_result(sum_status='failed')}
    monkeypatch.setattr(native.per_turn.TrainerService, 'train', parent_train)
    first = service.train({'event': 1, 'responses': [{'record_id': 'first'}]})
    second = service.train({'event': 2, 'responses': [{'record_id': 'second'}]})
    assert len(calls) == 1
    assert first['qualification_checks']['status'] == second['qualification_checks']['status'] == 'passed'
    assert 'disabled_filter_qualification' in first and 'disabled_filter_qualification' not in second
    assert (tmp_path / 'DISABLED_FILTER_QUALIFICATION.json').exists()
    assert first['qualification_checks']['gradient_sum_diagnostics'][0]['status'] == 'failed'
    assert first['qualification_checks']['sum_tolerance_is_admission_gate'] is False


@pytest.mark.parametrize('arm', ('H', 'L', 'G', 'N'))
def test_effective_runtime_recipe_overrides_stale_inherited_counts_and_lr(arm):
    from types import SimpleNamespace
    inherited = dict(recipe={'responses_per_step': 2, 'optimizer_steps_per_event': 8},
                     args={'learning_rate': 3e-4, 'attention_backend': 'flex_attention'},
                     teacher_context_limit=32768)
    lr = 1e-4 if arm == 'L' else 3e-4
    cfg = dict(agreement_arm=arm, agreement_qualification=True, agreement_source_sha256={'test': 'source'})
    receipt = native.effective_training_receipt(
        inherited, SimpleNamespace(learning_rate=lr), cfg, {'effective_donated_buffer': False})
    assert receipt['recipe'] == {**native.training.RECIPE, 'learning_rate': lr}
    assert receipt['recipe']['responses_per_step'] == 1
    assert receipt['recipe']['optimizer_steps_per_event'] == 2
    assert receipt['recipe']['passes'] == 2
    assert receipt['args']['learning_rate'] == lr
    assert receipt['args']['attention_backend'] == 'flex_attention'
    assert receipt['teacher_context_limit'] == 32768
    assert receipt['agreement_numerics']['version'] == 'agreement_direct_full_loss_gradient_v2'
    assert receipt['gradient_backward_calls_per_optimizer_step'] == (3 if arm in ('G', 'N') else 1)
    assert receipt['gradient_backward_calls_per_event'] == (6 if arm in ('G', 'N') else 2)
    assert inherited['recipe']['responses_per_step'] == 2
    assert inherited['args']['learning_rate'] == 3e-4


def test_finite_failed_sum_is_diagnostic_and_remains_failed():
    result = native.direct_full_qualification_checks(_training_result(sum_status='failed'))
    assert result['status'] == 'passed'
    assert result['sum_tolerance_is_admission_gate'] is False
    assert result['gradient_backward_calls_per_optimizer_step'] == 3
    assert result['gradient_backward_calls_per_event'] == 6
    assert [c['status'] for c in result['gradient_sum_diagnostics']] == ['failed', 'failed']


@pytest.mark.parametrize('field,value', [
    ('ordinary_proposal_gradient', None),
    ('ordinary_proposal_gradient', 'sum of A and B'),
    ('gradient_backward_calls', 2),
    ('numerical_protocol', 'chronological_response_groups_v1'),
    ('sum_tolerance_is_admission_gate', True),
    ('gradient_sum_diagnostic', None),
])
def test_direct_full_qualification_rejects_missing_or_wrong_construction(field, value):
    result = _training_result()
    result['accounting']['batches'][0]['presentations'][0][field] = value
    with pytest.raises(ValueError):
        native.direct_full_qualification_checks(result)


@pytest.mark.parametrize('invalid', ['nonfinite_input', 'nonfinite_sum', 'missing_counts', 'other_failure'])
def test_direct_full_qualification_still_rejects_nonfinite_or_malformed_diagnostics(invalid):
    result = _training_result(sum_status='failed')
    diagnostic = result['accounting']['batches'][0]['presentations'][0]['gradient_sum_diagnostic']
    if invalid == 'nonfinite_input':
        diagnostic['nonfinite_input_coordinates'] = 1
    elif invalid == 'nonfinite_sum':
        diagnostic['nonfinite_counts']['sum'] = 1
    elif invalid == 'missing_counts':
        diagnostic.pop('nonfinite_counts')
    else:
        diagnostic['first_failure']['error'] = 'Unrelated correctness failure'
    with pytest.raises(ValueError):
        native.direct_full_qualification_checks(result)


def test_failed_disabled_filter_receipt_is_saved_before_original_error(tmp_path, monkeypatch):
    import json
    class Trainer:
        last_qualification_receipt = None
        def qualify_disabled_filter(self, responses):
            self.last_qualification_receipt = dict(
                status='failed', error='fixed displacement tolerance failed',
                temporary_optimizer_steps=4, persistent_optimizer_steps_after=0,
                restored_state_identity_sha256='restored', restoration_exact=True)
            raise ValueError('original qualification rejection')
    service = object.__new__(native.TrainerService)
    service.config = {'agreement_qualification': True, 'agreement_arm': 'G'}
    service.trainer = Trainer(); service.out = tmp_path; service.qualification_receipt = None
    service._check_responses = lambda responses: None
    def forbidden_parent_train(*args):
        raise AssertionError('A rejected qualification must not proceed to training')
    monkeypatch.setattr(native.per_turn.TrainerService, 'train', forbidden_parent_train)
    with pytest.raises(ValueError, match='original qualification rejection'):
        service.train({'event': 1, 'responses': [{'record_id': 'first'}]})
    receipt = json.loads((tmp_path / 'DISABLED_FILTER_QUALIFICATION.json').read_text())
    assert receipt['status'] == 'failed' and receipt['restoration_exact'] is True
    assert receipt['temporary_optimizer_steps'] == 4
    assert receipt['observed_wall_s'] >= 0
    assert not (tmp_path / 'UPDATE_1.json').exists()


def test_failed_disabled_filter_return_does_not_become_admission_success(tmp_path, monkeypatch):
    class Trainer:
        def qualify_disabled_filter(self, responses):
            return {'status': 'failed', 'temporary_optimizer_steps': 4}
    service = object.__new__(native.TrainerService)
    service.config = {'agreement_qualification': True, 'agreement_arm': 'G'}
    service.trainer = Trainer(); service.out = tmp_path; service.qualification_receipt = None
    service._check_responses = lambda responses: None
    def forbidden_parent_train(*args):
        raise AssertionError('A rejected qualification must not proceed to training')
    monkeypatch.setattr(native.per_turn.TrainerService, 'train', forbidden_parent_train)
    with pytest.raises(ValueError, match='Disabled-filter equivalence failed'):
        service.train({'event': 1, 'responses': [{'record_id': 'first'}]})
    assert (tmp_path / 'DISABLED_FILTER_QUALIFICATION.json').exists()


@pytest.mark.parametrize('arm', ('H', 'L', 'G', 'N'))
def test_compiler_setting_is_pinned_before_inherited_model_setup(tmp_path, monkeypatch, arm):
    import json
    import torch._functorch.config as config
    monkeypatch.setattr(native, 'check_sources', lambda cfg: None)
    def stop_before_models(owner, cfg):
        assert config.donated_buffer is False
        receipt = json.loads((tmp_path / 'COMPILER_RUNTIME.json').read_text())
        assert receipt['effective_donated_buffer'] is False
        assert receipt['setting'] == 'torch._functorch.config.donated_buffer'
        raise RuntimeError('test stopped before model construction')
    monkeypatch.setattr(native.followup.TrainerService, '__init__', stop_before_models)
    with config.patch(donated_buffer=True), pytest.raises(RuntimeError, match='test stopped before model construction'):
        native.TrainerService({'worker_out': str(tmp_path), 'agreement_arm': arm})
