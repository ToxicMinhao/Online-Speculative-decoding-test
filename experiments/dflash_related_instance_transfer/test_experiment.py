"""CPU-only sealed synthetic groups; no retained real prompts or native runtime."""
import copy
import math

import pytest

from . import evidence as ev, experiment as ex
from .protocol import SERVING_CONTRACT
from .test_evidence import make_bundle, make_construction, make_renderer


def native_row(child, checkpoint, label, *, calls=4, output=None):
    output = list(output if output is not None else list(range(200, 224)) + [128009])
    n = len(output)
    if n == 1:
        calls = 0
    assert calls == 0 if n == 1 else 0 < calls <= n - 1 and (n - 1 + calls - 1) // calls <= 10
    offset, rounds = 1, []
    for i in range(calls):
        committed = (n - 1) // calls + (i < (n - 1) % calls)
        rounds.append(dict(round_index=i, output_offset=offset, accepted_prefix_length=committed - 1,
            accepted_draft_tokens=committed - 1, proposed_tokens=9, committed_tokens=committed,
            committed_token_ids=output[offset:offset + committed], first_rejection_offset=committed - 1,
            stop_clipped=False))
        offset += committed
    row = {key: child.get(key) for key in ex.IDENTITY}
    row.update(status='passed', checkpoint_id=label, checkpoint_identity_sha256=checkpoint,
        input_ids=copy.deepcopy(child['input_ids']), input_ids_sha256=ex.digest(child['input_ids']),
        output_ids=output, output_tokens=n, finish_reason={'type': 'stop', 'matched': 128009},
        cap_hit=n == 1024, eos_hit=output[-1] == 128009, request_s=.25,
        capture=False, feedback_capture_enabled=False, feedback=None, capture_costs=None,
        serving_contract=copy.deepcopy(SERVING_CONTRACT), verification_calls=calls,
        accepted_draft_tokens=sum(r['accepted_draft_tokens'] for r in rounds),
        verification_committed_tokens=n - 1, initial_output_tokens=1, rounds=rounds,
        committed_progress_per_call=(n - 1) / calls if calls else None,
        emitted_tokens_per_verification_call=n / calls if calls else None,
        cache_maintenance_s=.01)
    return row


def groups_fixture():
    bundle = make_bundle()
    for group in bundle['groups']:
        bindings = {k: group['checkpoints'][k]['checkpoint_identity_sha256'] for k in ex.ARMS}
        group['reference_pairs'] = {'source': [], 'heldout': []}
        for i, original in enumerate(group['feedback']):
            child = {**original, 'parent_request_id': None}
            output = original['output_ids'][:-1] + [128009]
            f = native_row(child, bindings['F'], 'F', calls=max(0, len(output)-1), output=output)
            s = native_row(child, bindings['S8'], 'S8', calls=max(0, len(output)-1), output=output)
            group['feedback'][i] = f
            group['reference_pairs']['source'].append(dict(request_id=f['request_id'], F={'native': f}, S8={'native': s}))
        for original in group['evaluation_requests']:
            child = dict(**original, group_id=group['group_id'], domain=group['domain'],
                         role='evaluation', parent_request_id=None)
            f = native_row(child, bindings['F'], 'F', calls=4)
            s = native_row(child, bindings['S8'], 'S8', calls=5)
            group['scores'].extend([f, s])
            group['reference_pairs']['heldout'].append(dict(request_id=f['request_id'], F={'native': f}, S8={'native': s}))
    bundle['parent_selection'] = ev.select_parents(bundle['groups'], bundle['manifest'])
    construction = make_construction(bundle)
    sealed = ev.seal_children(bundle, construction, render=make_renderer(bundle, construction))
    return ev.bind_children(bundle, sealed)


class FakeBackend:
    def __init__(self, *, fail_at=None, corrupt=None, cleanup=None, outcome='worse', returned_partial=False):
        self.fail_at, self.corrupt, self.cleanup = fail_at, corrupt, cleanup
        self.outcome, self.returned_partial = outcome, returned_partial
        self.calls, self.closed = [], False

    def start_group(self, group):
        self.group = copy.deepcopy(group)
        return {'status': 'passed', 'runtime_initialization_s': .1, 'explicit_generation_warmups': 0}

    def serve_request(self, child, label):
        self.calls.append((label, child['request_id']))
        if len(self.calls) == self.fail_at:
            error = RuntimeError('worker transport or normalization failed')
            if self.returned_partial:
                error.native_partial = dict(generation_returned=True, raw_receipt={'output_ids': [1, 2, 128009]}, stage='normalization')
            raise error
        calls = 4 if label == 'F' else {'worse': 5, 'better': 3, 'tied': 4}[self.outcome]
        row = native_row(child, self.group['checkpoints'][label]['checkpoint_identity_sha256'], label, calls=calls)
        if self.corrupt and label == 'S8':
            if self.corrupt == 'output':
                row['output_ids'][-2] += 1
                row['rounds'][-1]['committed_token_ids'][-2] += 1
            if self.corrupt == 'termination': row['finish_reason'] = {'type': 'length'}
            if self.corrupt == 'nonfinite': row['request_s'] = math.nan
            if self.corrupt == 'checkpoint': row['checkpoint_identity_sha256'] = 'x' * 64
            if self.corrupt == 'role': row['role'] = 'feedback'
            if self.corrupt == 'parent': row['parent_request_id'] = 'foreign'
            if self.corrupt == 'capture': row['feedback_capture_enabled'] = True
            if self.corrupt == 'runtime': row['serving_contract']['effective_prompt_limit'] = 31744
            if self.corrupt == 'counter': row['accepted_draft_tokens'] += 1
        return row

    def finish_group(self):
        self.closed = True
        row = dict(status='passed', checkpoint_weights_unchanged=True, training_quiescent=True,
                   optimizer_steps=0, response_kl_evaluations=0, new_checkpoints=0)
        if self.cleanup == 'status': row['status'] = 'failed'
        if self.cleanup == 'weights': row['checkpoint_weights_unchanged'] = False
        if self.cleanup == 'trainer': row['training_quiescent'] = False
        if self.cleanup == 'nonfinite': row['shutdown_s'] = math.inf
        if self.cleanup == 'raise':
            error = RuntimeError('cleanup failed')
            error.native_finish = {**row, 'status': 'failed'}
            raise error
        return row


def complete_result(outcome='worse'):
    return ex.run_campaign(groups_fixture(), lambda group: FakeBackend(outcome=outcome))


@pytest.mark.parametrize('outcome', ['worse', 'tied', 'better'])
def test_full_scope_order_and_first_gate_inside_budget(outcome):
    groups = groups_fixture(); backends = []
    def factory(group):
        b = FakeBackend(outcome=outcome); backends.append(b); return b
    result = ex.run_campaign(groups, factory)
    assert result['status'] == 'complete'
    assert result['actual_native_generations'] == result['returned_valid_native_generations'] == 24
    assert result['unknown_native_work_receipts'] == result['unattempted_native_generations'] == 0
    assert result['optimizer_steps'] == result['actual_response_kl_evaluations'] == result['new_checkpoints'] == result['retries'] == 0
    assert result['first_group_qualification']['native_generations'] == 4
    assert all(b.closed for b in backends)
    assert backends[0].calls == [(arm, c['request_id']) for arm in ('F', 'S8') for c in groups[0]['children']]
    assert result['groups'][0]['pairs'][0]['F']['native']['output_ids'] != groups[0]['feedback'][1]['output_ids']


@pytest.mark.parametrize('failure_call', [1, 2, 3, 4])
@pytest.mark.parametrize('returned_partial', [False, True])
def test_partial_work_accounting_each_failure_slot(failure_call, returned_partial):
    backend = FakeBackend(fail_at=failure_call, returned_partial=returned_partial)
    result = ex.run_campaign(groups_fixture(), lambda group: backend)
    assert result['status'] == 'failed' and backend.closed
    assert result['attempted_native_generations'] == failure_call
    assert result['actual_native_generations'] == failure_call - 1 + int(returned_partial)
    assert result['unknown_native_work_receipts'] == int(not returned_partial)
    assert result['returned_invalid_native_generations'] == int(returned_partial)
    assert result['unattempted_native_generations'] == 24 - failure_call
    assert len(result['unexecuted_groups']) == 5 and 'first_group_qualification' not in result


@pytest.mark.parametrize('corrupt', ['output', 'termination', 'nonfinite', 'checkpoint', 'role', 'parent', 'capture', 'runtime', 'counter'])
def test_invalid_return_kept_before_validation_and_stops(corrupt):
    snapshots, events = [], []
    backend = FakeBackend(corrupt=corrupt)
    result = ex.run_campaign(groups_fixture(), lambda group: backend,
        snapshot=lambda r: snapshots.append(r), emit=lambda k, r: events.append((k, r)))
    assert result['status'] == 'failed' and result['attempted_native_generations'] == 3
    assert result['actual_native_generations'] == 3 and result['returned_invalid_native_generations'] == 1
    assert result['unknown_native_work_receipts'] == 0 and result['unattempted_native_generations'] == 21
    assert any(s['groups'] and len(s['groups'][0]['native_raw']) == 3 and
               s['groups'][0]['native_attempts'][-1]['status'] == 'returned_unvalidated' for s in snapshots)
    assert len([e for e in events if e[0] == 'native_raw']) == 3
    if corrupt == 'nonfinite':
        assert result['groups'][0]['native_raw'][-1]['request_s']['nonfinite_float'] == 'nan'


@pytest.mark.parametrize('cleanup', ['status', 'weights', 'trainer', 'nonfinite', 'raise'])
def test_cleanup_and_state_failure_stop_before_first_gate(cleanup):
    result = ex.run_campaign(groups_fixture(), lambda group: FakeBackend(cleanup=cleanup))
    assert result['status'] == 'failed' and result['actual_native_generations'] == 4
    assert 'first_group_qualification' not in result and len(result['unexecuted_groups']) == 5
    assert all(p['group_execution_valid'] is False for p in result['groups'][0]['pairs'])


def test_factory_failure_has_zero_attempts_and_all_children_preserved():
    def fail(group): raise RuntimeError('constructor failed')
    result = ex.run_campaign(groups_fixture(), fail)
    assert result['status'] == 'failed' and result['unattempted_native_generations'] == 24
    assert result['unknown_native_work_receipts'] == 0
    assert sum(len(g['children']) for g in result['selected_groups']) == 12


def test_repetition_checks_parent_and_other_three_without_removal():
    result = complete_result()
    group = result['groups'][0]; pair = group['pairs'][0]
    own = next(r for r in group['feedback_answers'] if r['request_id'] == pair['parent_request_id'])
    other = next(r for r in group['feedback_answers'] if r['request_id'] != pair['parent_request_id'])
    first = ex.repeat_flags(pair['child'], {'output_ids': own['output_ids']}, group['feedback_answers'])
    second = ex.repeat_flags(pair['child'], {'output_ids': other['output_ids']}, group['feedback_answers'])
    assert first['matches_parent_answer'] and first['matches_any_feedback_answer'] and not first['non_exact_repeat']
    assert not second['matches_parent_answer'] and second['matches_any_feedback_answer'] and not second['non_exact_repeat']
    assert pair['repetition']['non_exact_repeat'] and len(result['selected_groups']) == 6


def test_historical_context_is_retained_and_separate_without_reruns():
    result = complete_result()
    context = [r for g in result['groups'] for r in g['historical_context']]
    assert len(context) == 36 and all(r['execution'] == 'not_rerun' for r in context)
    assert sum(r['population'] == 'parent_source' for r in context) == 12
    assert sum(r['population'] == 'ordinary_heldout' for r in context) == 24
    assert result['attempted_native_generations'] == 24


def test_group_order_is_fixed_before_any_backend_starts():
    groups = groups_fixture(); groups.reverse()
    with pytest.raises(ValueError, match='manifest group order'):
        ex.run_campaign(groups, lambda group: pytest.fail('Invalid schedule reached backend'))


@pytest.mark.parametrize('stage', ['startup', 'cleanup'])
@pytest.mark.parametrize('key', ['optimizer_steps', 'response_kl_evaluations', 'checkpoint_exports', 'warmup_generations', 'optimizer_created'])
def test_receipts_cannot_hide_forbidden_work(stage, key):
    class ForbiddenWork(FakeBackend):
        def start_group(self, group):
            row = super().start_group(group)
            if stage == 'startup': row[key] = True if key == 'optimizer_created' else 1
            return row
        def finish_group(self):
            row = super().finish_group()
            if stage == 'cleanup': row[key] = True if key == 'optimizer_created' else 1
            return row
    result = ex.run_campaign(groups_fixture(), lambda group: ForbiddenWork())
    assert result['status'] == 'failed' and 'first_group_qualification' not in result
    assert result['actual_native_generations'] == (0 if stage == 'startup' else 4)


def test_backend_cannot_mutate_the_controllers_sealed_child_objects():
    groups = groups_fixture(); original = copy.deepcopy(groups)
    class MutatingInput(FakeBackend):
        def start_group(self, group):
            answer = super().start_group(group)
            group['children'][0]['text'] = 'Backend mutation'
            return answer
    result = ex.run_campaign(groups, lambda group: MutatingInput())
    assert result['status'] == 'complete' and groups == original
