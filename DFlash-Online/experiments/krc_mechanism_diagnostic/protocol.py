"""Fixed, output-independent role/anchor bindings and bounded work contracts."""
from copy import deepcopy
import re

from experiments.krc_continuation.protocol import build_continuation_schedule as event2_schedule
from experiments.krc_events_3_4.protocol import build_continuation_schedule as late_schedule
from experiments.ok_continuation.resume import validate_event_receipt

WINDOWS = (1, 16, 32, 48, 64, 65, 80, 96, 112, 128)
CANDIDATES = ('R_FULL', 'R_HALF', 'D_FULL', 'K_ONLY')
BUDGET = dict(ordinary_Adam_steps=256, ordinary_presentations=512,
    baseline_response_forwards=1536, diagnostic_response_forwards=600,
    core_response_forwards=2136, maximum_response_backwards=1064,
    sampled_windows=20, candidate_slots=80, serving_generations=0,
    feedback_generations=0, startup_maximum_forwards=4,
    startup_maximum_backwards=4, startup_Adam_steps=0)


def reference_schedule(ordinary, event_number, *, evaluation_ids=()):
    if event_number == 2:
        return event2_schedule(ordinary, evaluation_ids=evaluation_ids)
    if event_number == 4:
        return late_schedule(ordinary, 4, evaluation_ids=evaluation_ids)
    raise ValueError('Only independent original events 2 and 4 are authorized')


def build_window_plan(ordinary, event_number, *, evaluation_ids=()):
    """Seal each role's own same-pass anchors, including end-of-pass wraps."""
    schedule = reference_schedule(ordinary, event_number, evaluation_ids=evaluation_ids)
    windows = []
    for step in WINDOWS:
        original = schedule['windows'][step - 1]
        pass_index, pair_index = divmod(step - 1, 64)
        diagnostic_index = (pair_index + 32) % 64
        offset = pass_index * 128 + diagnostic_index * 2
        diagnostic = [{key: deepcopy(row[key]) for key in ('record_id', 'pass', 'sampled_anchors')}
                      for row in ordinary['records'][offset:offset + 2]]
        roles = dict(A=deepcopy(original['learning']), R=deepcopy(original['reference']), D=diagnostic)
        ids = [binding['record_id'] for role in roles.values() for binding in role]
        if len(ids) != 6 or len(set(ids)) != 6 or set(ids) & set(evaluation_ids):
            raise ValueError('A/R/D must be mutually disjoint and exclude all evaluation IDs')
        windows.append(dict(event_number=event_number, event_id=f'event_{event_number:03d}',
            step=step, event_local_step=step, pass_index=pass_index, batch_index=pair_index,
            reference_batch_index=(pair_index + 1) % 64, diagnostic_batch_index=diagnostic_index,
            global_step_before=(event_number - 1) * 128 + step - 1,
            global_step_after_native_Adam=(event_number - 1) * 128 + step,
            candidate_order=list(CANDIDATES), **roles))
    return windows


def validate_window_plan(windows, ordinary, event_number, *, evaluation_ids=()):
    if windows != build_window_plan(ordinary, event_number, evaluation_ids=evaluation_ids):
        raise ValueError('Window plan differs from the fixed same-pass A/R/D bindings')
    return dict(status='passed', event_number=event_number, windows=list(WINDOWS),
        candidate_slots=40, evaluation_ids_excluded=True,
        D_scope='buffered training data; not unseen or independent; fitted data for D_FULL')


def validate_plan(plan):
    if plan.get('budget') != BUDGET:
        raise ValueError('The authorized mechanism work budget differs')
    episodes = plan.get('episodes', [])
    if [row.get('event_number') for row in episodes] != [2, 4]:
        raise ValueError('Require exactly independent original event-2 and event-4 episodes')
    for row in episodes:
        number = row['event_number']
        if (row.get('start_step') != (number - 1) * 128 or row.get('end_step') != number * 128
                or row.get('parent_event') != number - 1
                or not all(row.get(key) for key in ('parent_state', 'parent_checkpoint',
                    'parent_receipt', 'endpoint_state', 'endpoint_checkpoint', 'endpoint_receipt',
                    'ordinary_receipt', 'reference_schedule', 'window_plan', 'pool'))):
            raise ValueError('Incomplete independent parent/endpoint/pool binding')
    source = plan.get('execution_source', {})
    if (not re.fullmatch('[0-9a-f]{40}', source.get('execution_git_commit', ''))
            or source.get('dirty_files') != []):
        raise ValueError('Require a committed exact execution revision and clean published source')
    if plan.get('no_automatic_retry') is not True or plan.get('no_new_feedback') is not True:
        raise ValueError('No retry or new feedback is authorized')
    return dict(status='passed', independent_episodes=[2, 4], budget=deepcopy(BUDGET))


def validate_original_report(report, event_number):
    """The original KRC recorder remains isolated from all diagnostic work."""
    from experiments.krc_events_3_4.protocol import check_correction_report
    return check_correction_report(report, event_number)


def validate_diagnostic_report(report):
    """Use explicit counters; no inference of forwards from result row counts."""
    mechanism = report['mechanism']
    work = mechanism['work']
    expected = dict(pre_capture_response_forwards=40, candidate_response_forwards=240,
                    d_gradient_response_forwards=20)
    if (work.get('pre_capture_response_forwards') != 40
            or any(not 0 <= work.get(key, -1) <= value for key, value in expected.items())):
        raise ValueError('Diagnostic response-forward budget exceeded or pre-captures are incomplete')
    if not 0 <= work.get('d_gradient_backwards', -1) <= 20:
        raise ValueError('Diagnostic D backward budget exceeded')
    rows = mechanism['rows']
    if [row['step'] for row in rows] != list(WINDOWS):
        raise ValueError('Retain all 10 fixed sampled windows, including no-op windows')
    if any(row.get('candidate_order') != list(CANDIDATES) for row in rows):
        raise ValueError('Every sampled window must attempt all four fixed candidate slots')
    invalid = []
    for row in rows:
        if set(row.get('candidates', {})) != set(CANDIDATES):
            raise ValueError('Every sampled window must retain all four candidate slots')
        for candidate, entry in row['candidates'].items():
            if entry.get('status') not in ('valid', 'invalid'):
                raise ValueError('Candidate slot lacks a validity decision')
            if entry['status'] == 'invalid':
                if (candidate == 'K_ONLY' or not entry.get('error')
                        or not entry.get('restoration_after_failure')
                        or not all(entry['restoration_after_failure'].values())):
                    raise ValueError('An invalid candidate requires an explicit failure and exact restoration')
                invalid.append(dict(step=row['step'], candidate=candidate, error=entry['error']))
    if not invalid and any(work.get(key) != value for key, value in expected.items()):
        raise ValueError('All-valid diagnostic requires exactly 300 additional response forwards per episode')
    if any(row.get('restoration_passed') is not True or row.get('k_only_witness_passed') is not True
           for row in rows):
        raise ValueError('Every sampled window requires exact continuation and K_ONLY witnesses')
    return dict(status='passed', windows=10, candidate_slots=40,
        additional_response_forwards=sum(work[key] for key in expected),
        additional_response_backwards=work['d_gradient_backwards'], invalid_candidates=invalid,
        unattempted_scoring_reason='invalid temporary candidate; no replacement or retry' if invalid else None)
