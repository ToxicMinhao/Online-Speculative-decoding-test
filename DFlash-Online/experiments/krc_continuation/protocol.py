"""KRC event-2 bindings: event-local windows and unchanged global Adam counters."""
from copy import deepcopy

from experiments.ok_continuation.resume import validate_event_receipt
from online_feedback.reference_schedule import build_reference_schedule


def validate_retained(krc1, ordinary2, k2):
    validate_event_receipt(krc1, event_number=1, objective='osd_first_error')
    validate_event_receipt(ordinary2, event_number=2, objective='osd')
    validate_event_receipt(k2, event_number=2, objective='osd_first_error',
                           reference_schedule=ordinary2)
    if not krc1.get('checkpoint_path') or not k2.get('checkpoint_path'):
        raise ValueError('Require retained corrected KRC1 and K2 exports')
    return dict(passed=True, start_global_step=128, end_global_step=256,
                retained_feedback=128, ordinary_presentations=256, schedule_matched=True)


def event_local_view(ordinary2):
    """Translate only counters for the existing builder; retain original receipt."""
    validate_event_receipt(ordinary2, event_number=2, objective='osd')
    local = deepcopy(ordinary2)
    local.update(event_id='event_001', start_step=0, end_step=128)
    for row in local['records']:
        row['optimizer_step_after'] -= 128
    return local


def build_continuation_schedule(ordinary2, *, evaluation_ids=()):
    schedule = build_reference_schedule(event_local_view(ordinary2), evaluation_ids=evaluation_ids)
    schedule['continuation'] = dict(event_id='event_002', start_global_step=128,
                                    end_global_step=256, local_windows=128,
                                    global_counter_rule='global_step = 128 + event_local_step')
    return schedule


def validate_continuation_schedule(schedule, ordinary2, *, evaluation_ids=()):
    if schedule != build_continuation_schedule(ordinary2, evaluation_ids=evaluation_ids):
        raise ValueError('Sealed event-2 reference schedule or anchor bindings differ')
    return dict(passed=True, local_windows=128, global_steps=[129,256],
                ordinary_presentations=256, pre_reference_presentations=256,
                post_reference_presentations=256, within_pass_wraps=[64,128])


def check_event(receipt, ordinary2, trainer_status):
    return validate_event_receipt(receipt, event_number=2, objective='osd_first_error',
                                   reference_schedule=ordinary2, trainer_status=trainer_status)


def check_correction_report(report):
    if report['start_step'] != 128 or report['native_global_step'] != 256:
        raise ValueError('Correction report must retain global steps 128 to 256')
    first = report['first_window']
    if (first['status'] != 'passed' or first['extra_optimizer_steps'] != 0
            or first['extra_forwards'] != 0):
        raise ValueError('First actual window must pass without duplicate qualification work')
    expected_counts = dict(K_backward_attempted=256, K_backward_completed=256, K_backward_failed=0,
                           Adam_attempted=128, Adam_completed=128, Adam_failed=0,
                           presentations_completed=256, windows_completed=128)
    if report['status'] != 'completed' or any(report['counts'][key] != value
                                              for key,value in expected_counts.items()):
        raise ValueError('Require exactly 128 new Adam steps and 256 ordinary K backwards')
    expected = dict(original_response_forwards=256, pre_reference_response_forwards=256,
                    post_k_reference_response_forwards=256, post_c_reference_response_forwards=0,
                    post_k_learning_response_forwards=0, post_c_learning_response_forwards=0)
    if any(report['work'].get(key,0) != value for key,value in expected.items()):
        raise ValueError('Require exactly 768 core forwards and zero diagnostic forwards')
    work=report['work']
    if (work.get('violation_backward_completed',0)>256 or work.get('direct_corrections',0)>128
            or work.get('violation_backward_failed',0)):
        raise ValueError('Correction/backward budget or failure count differs')
    if [row['step'] for row in report['steps']] != list(range(1,129)) or report['probes']:
        raise ValueError('Require local windows 1..128 and no optional probes')
    return dict(passed=True, local_windows=128, start_global_step=128,
                end_global_step=256, core_response_forwards=768,
                ordinary_backwards=256, reference_backwards=work.get('violation_backward_completed',0))
