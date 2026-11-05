"""Event-local schedules with unchanged global KRC counters and sampling."""
from copy import deepcopy

from experiments.ok_continuation.resume import validate_event_receipt
from online_feedback.reference_schedule import build_reference_schedule


def build_continuation_schedule(ordinary, event_number, *, evaluation_ids=()):
    if event_number not in (3, 4):
        raise ValueError('Only events 3 and 4 belong to this continuation')
    validate_event_receipt(ordinary, event_number=event_number, objective='osd')
    local = deepcopy(ordinary)
    start = (event_number - 1) * 128
    local.update(event_id='event_001', start_step=0, end_step=128)
    for row in local['records']:
        row['optimizer_step_after'] -= start
    schedule = build_reference_schedule(local, evaluation_ids=evaluation_ids)
    schedule['continuation'] = dict(event_id=ordinary['event_id'], start_global_step=start,
        end_global_step=start + 128, local_windows=128,
        global_counter_rule=f'global_step = {start} + event_local_step')
    return schedule


def validate_schedule(schedule, ordinary, event_number, *, evaluation_ids=()):
    if schedule != build_continuation_schedule(ordinary, event_number, evaluation_ids=evaluation_ids):
        raise ValueError('Sealed reference schedule differs from the retained event bindings')
    return dict(passed=True, event_number=event_number, windows=128,
        global_steps=[(event_number - 1) * 128 + 1, event_number * 128],
        ordinary_presentations=256, reference_presentations=256, within_pass_wraps=[64, 128])


def validate_retained(krc2, events):
    validate_event_receipt(krc2, event_number=2, objective='osd_first_error')
    if [number for number, _, _ in events] != [3, 4]:
        raise ValueError('Require exactly ordered events 3 and 4')
    for number, ordinary, k in events:
        validate_event_receipt(ordinary, event_number=number, objective='osd')
        validate_event_receipt(k, event_number=number, objective='osd_first_error',
                               reference_schedule=ordinary)
    return dict(passed=True, start_global_step=256, end_global_step=512,
                retained_feedback=256, ordinary_presentations=512, new_Adam_steps=256)


def check_correction_report(report, event_number):
    start = (event_number - 1) * 128
    if report['start_step'] != start or report['native_global_step'] != start + 128:
        raise ValueError('Correction report has the wrong global event boundary')
    first = report['first_window']
    if (first['status'] != 'passed' or first['extra_optimizer_steps'] != 0
            or first['extra_forwards'] != 0):
        raise ValueError('The actual first window must qualify without duplicate work')
    counts = dict(K_backward_attempted=256, K_backward_completed=256, K_backward_failed=0,
        Adam_attempted=128, Adam_completed=128, Adam_failed=0,
        presentations_completed=256, windows_completed=128)
    if report['status'] != 'completed' or any(report['counts'][k] != v for k, v in counts.items()):
        raise ValueError('Require 128 Adam steps and 256 ordinary presentations per event')
    forwards = dict(original_response_forwards=256, pre_reference_response_forwards=256,
        post_k_reference_response_forwards=256, post_c_reference_response_forwards=0,
        post_k_learning_response_forwards=0, post_c_learning_response_forwards=0)
    work = report['work']
    if any(work.get(k, 0) != v for k, v in forwards.items()):
        raise ValueError('Require 768 core forwards and zero optional probes per event')
    if (work.get('violation_backward_completed', 0) > 256
            or work.get('direct_corrections', 0) > 128 or work.get('violation_backward_failed', 0)):
        raise ValueError('Correction/backward budget or failure count differs')
    if [row['step'] for row in report['steps']] != list(range(1, 129)) or report['probes']:
        raise ValueError('Require local windows 1..128 and no optional probes')
    return dict(passed=True, event_number=event_number, start_global_step=start,
        end_global_step=start + 128, core_response_forwards=768, ordinary_backwards=256,
        reference_backwards=work.get('violation_backward_completed', 0))
