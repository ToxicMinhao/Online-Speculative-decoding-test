import copy
import math

import pytest

from .experiment import (CHECKPOINTS, checkpoint_order, digest, measurement_validation,
                         run_campaign, run_group, validate_measurement, validate_training)


def group(index=0):
    domain = ('gsm8k', 'codealpaca', 'sharegpt')[index // 2]
    return dict(group_id=f'{domain}_{index % 2}', domain=domain,
                **{role: [dict(request_id=f'r{index}_{role}_{i}', text=f'Question {index}/{role}/{i}',
                               input_ids=[10, 100 + index * 8 + i + (4 if role == 'evaluation' else 0)])
                          for i in range(4)] for role in ('feedback', 'evaluation')})


def measurement(ids, checkpoint, calls=2, latency=1.):
    output = list(range(1000, 1010)) + [128009]
    rounds, offset = [], 1
    for i in range(calls):
        committed = (10 - (offset - 1) + calls - i - 1) // (calls - i)
        rounds.append(dict(round_index=i, output_offset=offset, accepted_prefix_length=committed - 1,
            accepted_draft_tokens=committed - 1, committed_tokens=committed,
            committed_token_ids=output[offset:offset + committed], proposed_tokens=9,
            first_rejection_offset=committed - 1, stop_clipped=False))
        offset += committed
    return dict(status='passed', input_ids=list(ids), input_ids_sha256=digest(ids), output_ids=output,
                output_tokens=len(output), cap_hit=False, finish_reason={'type': 'stop'}, request_s=latency,
                capture=False, feedback=None, capture_costs=None, verification_calls=calls,
                accepted_draft_tokens=10 - calls, verification_committed_tokens=10, initial_output_tokens=1,
                rounds=rounds, checkpoint_identity_sha256={'F': 'a', 'S2': 'b', 'S8': 'c'}[checkpoint] * 64,
                committed_progress_per_call=10 / calls, emitted_tokens_per_verification_call=11 / calls,
                activation_s=0., native_activation_s=0., cache_maintenance_s=.01)


class Backend:
    def __init__(self, source, *, harmful=False, quiescent=True, mismatch=False, malformed=False):
        self.source = source
        self.harmful, self.quiescent, self.mismatch, self.malformed = harmful, quiescent, mismatch, malformed
        self.log = []; self.stopped = False
        self.checkpoints = {c: {'checkpoint_identity_sha256': letter * 64} for c, letter in zip(CHECKPOINTS, 'abc')}

    def start_group(self, source):
        self.log.append(('start', source['group_id']))
        return dict(status='passed', checkpoints={'F': self.checkpoints['F']}, fresh_optimizer=True, released_reset=True, seed=17)

    def serve(self, ids, checkpoint_id, role, request_id, capture=False):
        assert not capture
        if role == 'evaluation':
            assert self.stopped
        self.log.append(('serve', role, request_id, checkpoint_id))
        calls = {'F': 2, 'S2': 3, 'S8': 4}[checkpoint_id] if self.harmful else 2
        row = measurement(ids, checkpoint_id, calls, {'F': 1., 'S2': 1.1, 'S8': 1.2}[checkpoint_id])
        if self.mismatch and role == 'evaluation' and checkpoint_id == 'S2':
            row['output_ids'][1] += 1
            row['rounds'][0]['committed_token_ids'][0] += 1
        if self.malformed and role == 'feedback':
            row['request_s'] = float('nan')
        return row

    def prepare_support(self, canonical):
        self.log.append(('prepare',))
        return [dict(record_id=row['request_id'], input_ids=row['input_ids'], output_ids=row['output_ids']) for row in canonical]

    def train_pool(self, responses):
        self.log.append(('train',))
        batches = []
        for index in range(8):
            presentations = []
            for row in responses[2 * (index % 2):2 * (index % 2) + 2]:
                start, end = len(row['input_ids']), len(row['input_ids']) + len(row['output_ids'])
                positions = list(range(start + 1, end))
                presentations.append(dict(record_id=row['record_id'], response_start=start, response_end_exclusive=end,
                    loss_mask_sha256=digest([0] * start + [1] * (end - start) + [0] * 9),
                    unique_supervised_position_ids=positions, unique_supervised_positions=len(positions),
                    eligible_positions=len(positions), zero_supervision=False,
                    anchor_positions=list(range(start, end - 1)), normalization_denominator=10.))
            batches.append(dict(optimizer_steps_after=index + 1, pass_index=index // 2, presentations=presentations))
        losses = [dict(checkpoint_id=c, loss=1. - .05 * index, fixed_anchor_identity_sha256='d' * 64,
                       training_rng_preserved=True) for index, c in enumerate(CHECKPOINTS)]
        return dict(status='passed', checkpoints=self.checkpoints, source_losses=losses,
                    accounting=dict(optimizer_steps=8, response_presentations=16,
                        recipe=dict(passes=4, responses_per_step=2, learning_rate=3e-4, loss='forward_KL'), batches=batches),
                    costs=dict(teacher_preparation_s=2., training_s=5., export_s=1., optimizer_state_save_s=.5,
                               checkpoint_transfer_s=0., peak_allocated_gpu_bytes=1234), trainer_startup_s=.2)

    def stop_trainer(self):
        self.log.append(('stop',)); self.stopped = True
        return self.assert_training_quiescent()

    def assert_training_quiescent(self):
        return dict(status='passed', training_quiescent=self.stopped and self.quiescent,
                    trainer_processes=[] if self.quiescent else [91])

    def publish(self, checkpoint_id):
        assert self.stopped
        self.log.append(('publish', checkpoint_id))
        return dict(status='passed', activation_s=.3, native_activation_s=.1)

    def warmup(self, checkpoint_id):
        assert self.stopped
        self.log.append(('warmup', checkpoint_id))
        return dict(status='passed', request_s=.1, cache_maintenance_s=.01)

    def finish_group(self):
        self.log.append(('finish',)); return {'status': 'passed'}


def result(index=0, **kwargs):
    source = group(index)
    return run_group(source, Backend(source, **kwargs), group_index=index)


def test_complete_group_separates_training_and_rotates_after_hard_stop():
    source = group(); backend = Backend(source)
    value = run_group(source, backend)
    assert value['status'] == 'complete' and value['implementation_checks']['status'] == 'passed'
    assert len(value['feedback']) == 4 and len(value['scores']) == 12 and len(value['warmups']) == 3
    assert all(event[1] == 'feedback' for event in backend.log[1:5])
    stop = backend.log.index(('stop',))
    assert all(index > stop for index, event in enumerate(backend.log) if event[0] == 'warmup' or
               (event[0] == 'serve' and event[1] == 'evaluation'))
    request2 = [r for r in value['scores'] if r['request_id'] == source['evaluation'][1]['request_id']]
    assert [r['checkpoint_id'] for r in request2] == ['S2', 'S8', 'F']
    assert all(r['same_generation_valid'] for r in request2)


def test_first_group_harmful_results_continue_all_six_without_retuning():
    groups = [group(i) for i in range(6)]
    value = run_campaign(groups, lambda g: Backend(g, harmful=True))
    assert value['status'] == 'complete' and len(value['groups']) == 6
    assert sum(len(g['feedback']) + len(g['scores']) for g in value['groups']) == 96
    assert sum(g['training']['accounting']['optimizer_steps'] for g in value['groups']) == 48
    assert sum(g['training']['accounting']['response_presentations'] for g in value['groups']) == 96
    assert value['first_group_qualification']['status'] == 'passed'


def test_late_frozen_reference_mismatch_is_preserved_and_stops_group():
    source = group(1)
    records = []
    value = run_group(source, Backend(source, mismatch=True), group_index=1,
                      emit=lambda kind, row: records.append((kind, copy.deepcopy(row))))
    assert value['status'] == 'failed' and len(value['scores']) == 3
    assert [r['checkpoint_id'] for r in value['scores']] == ['S2', 'S8', 'F']
    bad = next(r for r in value['scores'] if r['checkpoint_id'] == 'S2')
    assert bad['measurement_validation']['status'] == 'passed' and not bad['same_generation_valid']
    assert any(kind == 'raw_measurement' and row.get('checkpoint_id') == 'S2' for kind, row in records)


def test_concurrent_training_blocks_all_warmups_and_evaluation():
    source = group(); backend = Backend(source, quiescent=False)
    value = run_group(source, backend)
    assert value['status'] == 'failed' and not value['scores'] and not value['warmups']
    assert backend.log[-1] == ('finish',)


def test_nonfinite_feedback_receipt_is_preserved_before_abort():
    source = group(); events = []
    value = run_group(source, Backend(source, malformed=True), emit=lambda kind, row: events.append((kind, row)))
    assert value['status'] == 'failed' and len(value['feedback']) == 1 and value['training'] is None
    receipt = value['feedback'][0]
    assert receipt['request_s']['nonfinite_float'] == 'nan'
    assert receipt['measurement_validation']['status'] == 'failed'
    assert events[0][0] == 'raw_measurement'


@pytest.mark.parametrize('mutate', [
    lambda r: r.update(accepted_draft_tokens=99),
    lambda r: r['rounds'][1].update(output_offset=99),
    lambda r: r.update(committed_progress_per_call=5.5),
    lambda r: r.update(capture=True),
    lambda r: r.update(request_s=float('inf')),
    lambda r: r['rounds'][0].update(first_rejection_offset=8),
])
def test_measurement_rejects_invalid_native_accounting(mutate):
    row = measurement([10, 11], 'F'); mutate(row)
    assert measurement_validation(row)['status'] == 'failed'


def test_one_token_eos_without_verification_is_valid():
    row = measurement([10, 11], 'F')
    row.update(output_ids=[128009], output_tokens=1, verification_calls=0, accepted_draft_tokens=0,
               verification_committed_tokens=0, initial_output_tokens=1, rounds=[],
               committed_progress_per_call=None, emitted_tokens_per_verification_call=None)
    assert validate_measurement(row)


def test_training_cannot_use_eval_or_duplicate_first_pass():
    source = group(); backend = Backend(source)
    support = [dict(record_id=r['request_id'], input_ids=r['input_ids'], output_ids=[1, 2, 128009]) for r in source['feedback']]
    training = backend.train_pool(support)
    assert validate_training(training, support)
    training['accounting']['batches'][2]['presentations'][0]['record_id'] = source['evaluation'][0]['request_id']
    with pytest.raises(ValueError, match='order or pool'):
        validate_training(training, support)


def test_first_group_correctness_failure_prevents_other_groups():
    calls = []
    def factory(source):
        calls.append(source['group_id']); return Backend(source, mismatch=True)
    campaign = run_campaign([group(i) for i in range(6)], factory)
    assert len(calls) == 1 and campaign['status'] == 'stopped_after_correctness_failure'
    assert len(campaign['unexecuted_groups']) == 5


def test_qualified_zero_exposure_retains_anchors_and_scheduled_steps():
    source = group(); backend = Backend(source)
    support = [dict(record_id=r['request_id'], input_ids=r['input_ids'], output_ids=[1, 128009])
               for r in source['feedback']]
    training = backend.train_pool(support)
    presentation = training['accounting']['batches'][0]['presentations'][0]
    anchor = presentation['response_end_exclusive'] - 1
    presentation.update(unique_supervised_position_ids=[], unique_supervised_positions=0,
                        eligible_positions=0, zero_supervision=True,
                        anchor_positions=[anchor], normalization_denominator=1e-6)
    assert validate_training(training, support)
    assert presentation['anchor_positions'] == [anchor]
    assert training['accounting']['optimizer_steps'] == 8
    assert training['accounting']['response_presentations'] == 16
    del presentation['zero_supervision']
    with pytest.raises(ValueError, match='zero-supervision'):
        validate_training(training, support)
    presentation.update(zero_supervision=True, eligible_positions=1)
    with pytest.raises(ValueError, match='zero-supervision'):
        validate_training(training, support)


def test_order_rotates_request_and_group():
    assert checkpoint_order(0, 0) == ['F', 'S2', 'S8']
    assert checkpoint_order(0, 1) == ['S2', 'S8', 'F']
    assert checkpoint_order(1, 0) == ['S2', 'S8', 'F']
