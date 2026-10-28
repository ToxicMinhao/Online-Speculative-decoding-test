"""The KR training budget and its original next-pair anchor bindings."""
from online_feedback.reference_schedule import build_reference_schedule
from online_feedback.reference_replay_event import expand_reference_schedule

DISTINCT_RESPONSES = 128
OUTER_WINDOWS = 128
PRESENTATIONS = 512
OPTIMIZER_STEPS = 256


def build_replay_schedule(ordinary, *, evaluation_ids=()):
    reference = build_reference_schedule(ordinary, evaluation_ids=evaluation_ids)
    return expand_reference_schedule(reference)


def validate_replay_schedule(schedule, ordinary, *, evaluation_ids=()):
    if schedule != build_replay_schedule(ordinary, evaluation_ids=evaluation_ids):
        raise ValueError('KR schedule differs from the original within-pass A/R binding')
    return dict(passed=True, distinct_responses=128, outer_windows=128,
                optimizer_steps=256, response_presentations=512,
                same_pass_anchors=True, cyclic_reference=True)


def check_training(receipt, schedule):
    expected = dict(status='completed', training_variant='KR', objective='osd_first_error',
                    passes=2, response_presentations=PRESENTATIONS,
                    optimizer_steps=OPTIMIZER_STEPS, start_step=0, end_step=OPTIMIZER_STEPS)
    if any(receipt[key] != value for key, value in expected.items()):
        raise ValueError('KR must complete exactly 512 presentations and 256 ordinary K steps')
    if receipt['features'] != schedule['features'] or len(receipt['records']) != PRESENTATIONS:
        raise ValueError('KR feedback pool or presentation accounting differs')
    for update in schedule['updates']:
        for offset, binding in enumerate(update['records']):
            row = receipt['records'][2*(update['optimizer_step']-1)+offset]
            for key in ('record_id', 'pass', 'sampled_anchors'):
                if row[key] != binding[key]:
                    raise ValueError('KR recorded binding differs: ' + key)
            if (row['role'] != update['role'] or row['outer_window'] != update['outer_window']
                    or row['optimizer_stepped'] != (offset == 1)
                    or row['optimizer_step_after'] != update['optimizer_step'] - int(offset == 0)):
                raise ValueError('KR update boundary or role differs')
    work = receipt['reference_replay']['counts']
    expected_work = dict(response_forwards_attempted=512, response_forwards_completed=512,
                         backward_attempted=512, backward_completed=512,
                         Adam_attempted=256, Adam_completed=256,
                         presentations_completed=512, updates_completed=256,
                         outer_windows_completed=128, shuffle_rng_draws=2, anchor_rng_draws=512)
    for role in ('A', 'R'):
        expected_work.update({role+'_presentations_completed': 256,
                              role+'_response_forwards_completed': 256,
                              role+'_backward_completed': 256, role+'_Adam_completed': 128})
    if any(work.get(key, 0) != value for key, value in expected_work.items()):
        raise ValueError('KR native forward/backward/optimizer work does not reconcile')
    if any(value for key, value in work.items() if key.endswith('_failed')):
        raise ValueError('KR contains failed native work')
    return dict(passed=True, **expected_work)
