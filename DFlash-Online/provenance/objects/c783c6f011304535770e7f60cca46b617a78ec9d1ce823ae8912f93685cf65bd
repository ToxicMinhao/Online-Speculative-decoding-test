"""Sealed ordinary-K schedule and zero-reference work for correction removal."""
from experiments.ok_continuation.resume import validate_event_receipt


def validate_retained(krc1, ordinary2, k2, krc2):
    validate_event_receipt(krc1, event_number=1, objective='osd_first_error')
    validate_event_receipt(ordinary2, event_number=2, objective='osd')
    for receipt in (k2, krc2):
        validate_event_receipt(receipt, event_number=2, objective='osd_first_error',
                               reference_schedule=ordinary2)
    if any(not receipt.get('checkpoint_path') for receipt in (krc1, k2, krc2)):
        raise ValueError('Require retained KRC1, K2 and KRC2 checkpoint identities')
    return dict(passed=True, start_global_step=128, end_global_step=256,
                retained_feedback=128, ordinary_presentations=256, schedule_matched=True)


def validate_schedule(schedule, ordinary2, *, evaluation_ids=()):
    """Seal the actual ordinary receipt; no reference pairs or forced RNG draws."""
    validate_event_receipt(ordinary2, event_number=2, objective='osd')
    if schedule != ordinary2:
        raise ValueError('Sealed event-2 presentation/anchor schedule differs')
    ids = {row['record_id'] for row in schedule['features']}
    if ids.intersection(evaluation_ids):
        raise ValueError('Evaluation request entered the training pool')
    return dict(passed=True, local_windows=128, global_steps=[129, 256],
                ordinary_presentations=256, anchor_draws=256,
                reference_operations=0, forced_anchors=False, reseeded=False)


def check_event(receipt, ordinary2, trainer_status):
    return validate_event_receipt(receipt, event_number=2, objective='osd_first_error',
                                   reference_schedule=ordinary2, trainer_status=trainer_status)


def check_off_report(report):
    """Require the ordinary event budget and absence of every reference path."""
    if report['status'] != 'completed':
        raise ValueError('OFF continuation is incomplete')
    if report['first_A']['status'] != 'passed':
        raise ValueError('First actual A update did not reproduce the retained ordinary candidate')
    expected = dict(response_forwards=256, K_backwards=256, Adam_steps=128,
                    presentations=256, reference_forwards=0, reference_backwards=0,
                    protected_prefix_captures=0, direct_corrections=0, probe_forwards=0)
    if any(report['work'].get(key) != value for key, value in expected.items()):
        raise ValueError('OFF work differs from 256 ordinary forwards/backwards and zero reference work')
    if (report['start_global_step'] != 128 or report['end_global_step'] != 256
            or report['extra_qualification_optimizer_steps'] != 0):
        raise ValueError('OFF global counters or qualification budget differs')
    return dict(passed=True, local_windows=128, start_global_step=128,
                end_global_step=256, core_response_forwards=256, core_backwards=256,
                reference_operations=0, direct_corrections=0)
