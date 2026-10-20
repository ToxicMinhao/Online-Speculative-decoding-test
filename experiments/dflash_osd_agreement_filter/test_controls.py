"""Numerical control validity is separate from native measurement validity."""
import copy
import json
import math

import pytest
import torch

from .controls import norm_match, n_update_control, n_checkpoint_control, N_NORM_POLICY
from .test_core import setup, gradients, core
from .test_experiment import Backend, references, stream
from .experiment import run_stream
from .report import summarize, comparison
from .run import run_qualified_campaign


def set_norm(update, *, index=0, actual=.5, intended=.5):
    stats = update['accounting']['batches'][index]['agreement']
    stats.update(applied_step_norm=actual, hypothetical_filtered_step_norm=intended,
                 uniform_control=norm_match(actual, intended))


def full_result():
    source = stream()
    result = run_stream(source, references(source), lambda arm, _: Backend(arm))
    assert result['status'] == 'complete'
    for row in result['scores']:
        row['verification_calls'] = {'F': 8, 'H': 6, 'L': 5, 'N': 4, 'G': 3}[row['arm']]
    return result


def test_review_bf16_counterexample_rejects_norm_control_without_changing_update():
    params, opt = setup({'x': [.05, .05]}, dtype=torch.bfloat16)
    a, b = gradients(params, {'x': [1., 1.]}), gradients(params, {'x': [1., -2.]})
    full = {'x': torch.tensor([2., -1.], dtype=torch.bfloat16)}
    stats = core.apply_adamw_step(params, opt, a, b, full_loss_gradient=full, mode='N')
    assert stats['uniform_scale_alpha'] == pytest.approx(1 / math.sqrt(2))
    assert stats['hypothetical_filtered_step_norm'] == pytest.approx(.000244140625)
    assert stats['applied_step_norm'] == pytest.approx(stats['proposed_step_norm'])
    check = stats['uniform_control']
    assert check['status'] == 'failed'
    assert check['relative_error'] == pytest.approx(math.sqrt(2) - 1)
    assert stats['optimizer_steps_after'] == 1  # Evidence retained; no hidden alternative update.


@pytest.mark.parametrize('actual,intended,passed', [
    (0., 0., True), (1e-30, 0., False), (0., 1e-30, False),
    (100.5, 100., True), (101., 100., True), (101.0001, 100., False),
    (99., 100., True), (98.9999, 100., False),
    (-1., 1., False), (1., -1., False), (float('nan'), 1., False),
    (float('inf'), 1., False), (1., None, False), (True, 1., False)])
def test_fixed_norm_boundary_and_zero_rule(actual, intended, passed):
    check = norm_match(actual, intended)
    assert (check['status'] == 'passed') is passed
    assert check['policy']['relative_error_max'] == .01
    json.dumps(check, allow_nan=False)


def test_bad_ancestral_event_blocks_later_attribution_even_if_latest_event_passes():
    result = full_result()
    assert all(x['interpretation'] == 'useful_filter_signal' for x in summarize([result])['interpretations'])
    event = result['arms']['N']['updates'][0]
    set_norm(event, actual=.7)
    report = summarize([result])
    assert all(x['interpretation'] == 'selective_filter_attribution_unavailable_N_norm_control'
               and not x['selective_filter_attribution_eligible'] for x in report['interpretations'])
    assert n_checkpoint_control(result, 2)['status'] == n_checkpoint_control(result, 3)['status'] == 'failed'
    # Native measurements remain valid descriptive evidence; no gain is concealed.
    row, _ = comparison(result, 2, 'G', 'N')
    assert row['valid'] and row['verification_calls_saved'] == 1
    assert not row['selective_filter_attribution_eligible']
    pooled = next(x for x in report['pooled'] if x['source'] == 'all' and x['turn'] == 'all'
                  and x['candidate'] == 'G' and x['control'] == 'N')
    assert pooled['valid_requests'] == 2 and pooled['norm_control_qualified_requests'] == 0
    assert not pooled['selective_filter_attribution_eligible']


def test_bad_later_event_does_not_invalidate_earlier_checkpoint():
    result = full_result()
    set_norm(result['arms']['N']['updates'][1], actual=.4)
    assert n_checkpoint_control(result, 2)['status'] == 'passed'
    assert n_checkpoint_control(result, 3)['status'] == 'failed'


@pytest.mark.parametrize('mutation', ['missing_update', 'duplicate_update', 'missing_step', 'duplicate_step', 'stale_receipt'])
def test_incomplete_or_forged_control_evidence_fails_closed(mutation):
    result = full_result()
    updates = result['arms']['N']['updates']
    if mutation == 'missing_update':
        del updates[0]
    elif mutation == 'duplicate_update':
        updates.append(copy.deepcopy(updates[0]))
    elif mutation == 'missing_step':
        updates[0]['accounting']['batches'].pop()
    elif mutation == 'duplicate_step':
        batches = updates[0]['accounting']['batches']
        batches[1] = copy.deepcopy(batches[0])
    else:
        stats = updates[0]['accounting']['batches'][0]['agreement']
        stats['applied_step_norm'] = .8  # Leave the old 'passed' receipt untouched.
    assert n_checkpoint_control(result, 3)['status'] == 'failed'
    assert all(x['interpretation'] != 'useful_filter_signal' for x in summarize([result])['interpretations'])


class InvalidN(Backend):
    def train_event(self, event, responses):
        update = super().train_event(event, responses)
        set_norm(update, actual=.7)
        return update


@pytest.mark.parametrize('qualification', [False, True])
def test_n_failure_preserves_update_and_stops_before_next_request_or_retirement(qualification):
    source, created = stream(), {}
    def factory(arm, name):
        created[arm] = InvalidN(arm) if arm == 'N' else Backend(arm)
        return created[arm]
    result = run_stream(source, references(source), factory, qualification=qualification)
    assert result['status'] == 'failed' and result['failed_arm'] == 'N'
    assert len(result['arms']['N']['canonical']) == 1
    assert len(result['arms']['N']['updates']) == 1
    assert result['arms']['N']['updates'][0]['uniform_control']['status'] == 'failed'
    assert not created['N'].finalized
    assert result['implementation_checks']['status'] == 'failed'
    assert len(result['scores']) == 8  # Other arms' valid measurements are preserved.


def test_invalid_n_qualification_never_launches_main_campaign(tmp_path):
    source = stream()
    def factory(config):
        return InvalidN('N') if config['agreement_arm'] == 'N' else Backend(config['agreement_arm'])
    result = run_qualified_campaign([source], {source['stream_id']: references(source)}, factory,
        tmp_path, config={}, fixtures={'status': 'passed'})
    assert result['status'] == 'qualification_failed' and result['campaign'] is None
    assert not (tmp_path / 'campaign').exists()


@pytest.mark.parametrize('changed', ['score_identity', 'predecessor_identity'])
def test_norm_evidence_must_belong_to_actual_evaluated_checkpoint(changed):
    result = full_result()
    if changed == 'score_identity':
        score = next(row for row in result['scores'] if row['arm'] == 'N' and row['turn'] == 3)
        score['checkpoint_identity_sha256'] = 'unrelated-checkpoint'
    else:
        result['arms']['N']['updates'][1]['accounting']['capture_checkpoint_sha256'] = 'unrelated-predecessor'
    assert n_checkpoint_control(result, 3)['status'] == 'failed'
    assert n_checkpoint_control(result, 2)['status'] == 'passed'
