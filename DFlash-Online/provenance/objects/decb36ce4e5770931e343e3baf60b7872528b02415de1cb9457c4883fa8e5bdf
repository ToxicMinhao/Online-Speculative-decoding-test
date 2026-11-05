"""Reconcile the bounded KR event from compact, retrieved evidence only.

No checkpoints are loaded and no models execute. Additive exposure on an
optimizer-boundary row already includes both responses; it is counted once.
"""
from collections import Counter, defaultdict
import argparse
import csv
import json
import math
from pathlib import Path

ADDITIVE = (
    'available_position_occurrences', 'selected_position_occurrences',
    'available_position_weight', 'selected_position_weight',
    'supervised_position_exposure', 'main_loss_numerator', 'loss_denominator',
    'no_error_blocks', 'empty_blocks',
    *(f'first_error_position_{position}' for position in range(1, 16)),
)
EXPECTED_WORK = dict(
    response_forwards_attempted=512, response_forwards_completed=512,
    backward_attempted=512, backward_completed=512,
    Adam_attempted=256, Adam_completed=256,
    presentations_completed=512, updates_completed=256,
    outer_windows_completed=128, anchor_rng_draws=512, shuffle_rng_draws=2,
)


def read(path):
    return json.loads(Path(path).read_text())


def anchor_count(bindings):
    return sum(sum(bool(value) for value in row)
               for draw in bindings for row in draw['keep_mask'])


def reconcile(receipt, costs, schedule, gates):
    """Return independently computed tables and an explicit issue list."""
    issues = []
    checks = []

    def check(name, actual, expected):
        passed = actual == expected
        checks.append(dict(name=name, passed=passed, actual=actual, expected=expected))
        if not passed:
            issues.append(name)

    rows = receipt['records']
    replay = receipt['reference_replay']
    work = replay['counts']
    check('completed_KR_event', [receipt.get('status'), receipt.get('training_variant'), receipt.get('objective')],
          ['completed', 'KR', 'osd_first_error'])
    check('receipt_budget', [receipt['response_presentations'], receipt['optimizer_steps'],
                            receipt['start_step'], receipt['end_step']], [512, 256, 0, 256])
    check('presentation_rows', len(rows), 512)
    check('sealed_updates', len(schedule['updates']), 256)
    check('distinct_feedback', len({row['record_id'] for row in receipt['features']}), 128)
    for name, expected in EXPECTED_WORK.items():
        check('work_' + name, work.get(name, 0), expected)
    check('failed_native_work', {key: value for key, value in work.items() if key.endswith('_failed') and value}, {})
    check('forbidden_correction_work', [replay['correction_steps'], replay['protected_capture'], replay['reference_model']],
          [0, False, False])
    check('replay_status', replay['status'], 'completed')

    bindings = [(update, index, binding) for update in schedule['updates']
                for index, binding in enumerate(update['records'])]
    check('schedule_presentations', len(bindings), 512)
    presentations = []
    response_exposure = defaultdict(Counter)
    role_totals = {role: Counter() for role in ('A', 'R')}
    for ordinal, row in enumerate(rows):
        role = row['role']
        if ordinal < len(bindings):
            update, offset, binding = bindings[ordinal]
            actual = [row['record_id'], row['pass'], row['sampled_anchors'], role,
                      row['outer_window'], row['scheduled_optimizer_step'],
                      row['optimizer_stepped'], row['optimizer_step_after']]
            expected = [binding['record_id'], binding['pass'], binding['sampled_anchors'], update['role'],
                        update['outer_window'], update['optimizer_step'], bool(offset),
                        update['optimizer_step'] - int(offset == 0)]
            check(f'binding_{ordinal:03d}', actual == expected, True)
        anchors = anchor_count(row['sampled_anchors'])
        metrics = row['metrics']
        result = {key: row[key] for key in ('record_id', 'pass', 'role', 'outer_window',
                                          'scheduled_optimizer_step', 'optimizer_step_after',
                                          'optimizer_stepped', 'learning_rate', 'metrics_scope', 'training_s')}
        result.update(presentation_ordinal=ordinal, sampled_anchors=anchors,
                      feature_load_s=row.get('feature_load_s'), reported_loss=metrics['loss'])
        is_boundary = row['optimizer_stepped']
        expected_scope = 'accumulation_window' if is_boundary else 'presentation'
        check(f'metric_scope_{ordinal:03d}', row['metrics_scope'], expected_scope)
        response_exposure[row['record_id']][role + '_presentations'] += 1
        response_exposure[row['record_id']][role + '_sampled_anchors'] += anchors
        role_totals[role]['presentations'] += 1
        role_totals[role]['sampled_anchors'] += anchors
        role_totals[role]['training_s'] += row['training_s']
        role_totals[role]['feature_load_s'] += row.get('feature_load_s', 0)
        previous = rows[ordinal - 1]['metrics'] if is_boundary and ordinal else {}
        for name in ADDITIVE:
            value = metrics[name]
            own_value = value - previous.get(name, 0) if is_boundary else value
            result[name + '_reported'] = value
            result[name + '_presentation'] = own_value
            response_exposure[row['record_id']][role + '_' + name] += own_value
            if is_boundary:
                role_totals[role][name] += value
        if is_boundary:
            role_totals[role]['optimizer_steps'] += 1
            check(f'constant_learning_rate_{ordinal:03d}', row['learning_rate'], 3e-4)
        presentations.append(result)

    response_rows = []
    for record_id, values in sorted(response_exposure.items()):
        check('two_A_two_R_' + record_id, [values['A_presentations'], values['R_presentations']], [2, 2])
        check('same_retained_anchor_exposure_' + record_id, values['A_sampled_anchors'], values['R_sampled_anchors'])
        response_rows.append(dict(record_id=record_id, **values))
    anchors_total = sum(value['sampled_anchors'] for value in role_totals.values())
    check('sampled_anchor_occurrences', anchors_total, 130192)
    check('role_presentations', [role_totals[role]['presentations'] for role in ('A', 'R')], [256, 256])
    check('role_optimizer_steps', [role_totals[role]['optimizer_steps'] for role in ('A', 'R')], [128, 128])

    updates = replay['updates']
    check('update_receipts', len(updates), 256)
    check('continuous_native_Adam_counters', [row['native_Adam_step'] for row in updates], list(range(1, 257)))
    check('update_step_identity', [row['optimizer_step'] for row in updates], list(range(1, 257)))
    check('alternating_update_roles', [row['role'] for row in updates], ['A', 'R'] * 128)
    for name in ADDITIVE:
        actual = sum(value[name] for value in role_totals.values())
        saved = receipt['selection_totals'][name]
        # Identical ordering is used within each role, then combined; FP totals
        # can differ in their last bit from the event's original summation.
        passed = math.isclose(actual, saved, rel_tol=1e-12, abs_tol=1e-8)
        check('selection_total_' + name, passed, True)

    for role in ('A', 'R'):
        selected = [row for row in updates if row['role'] == role]
        role_totals[role]['update_wall_s'] = sum(row['update_wall_s'] for row in selected)
        role_totals[role]['journal_s_within_updates'] = sum(row['journal_s'] for row in selected)
        for name in ('training_s', 'feature_load_s', 'update_wall_s'):
            check(role + '_cost_' + name,
                  math.isclose(role_totals[role][name], replay['costs'][role + '_' + name], rel_tol=1e-12, abs_tol=1e-8), True)

    startup = costs['startup']
    check('startup_work', [startup['response_evaluations'], startup['backwards'], startup['optimizer_steps']], [2, 2, 0])
    check('no_additional_qualification_updates',
          [costs['new_qualification_optimizer_steps'], costs['new_qualification_presentations']], [0, 0])
    check('startup_is_disposable_restored', [startup['status'], startup['separate_disposable_runtime'],
                                           startup['exact_initial_state_restored']], ['passed', True, True])
    check('first_A_record_gate', gates['first_A']['exact_equal'], True)
    check('first_AR_gate', gates['first_AR']['status'], 'passed')
    check('first_AR_first_A_gate', gates['first_AR']['first_A_record_gate_passed'], True)
    check('first_AR_no_added_work', [gates['first_AR']['additional_forwards'], gates['first_AR']['additional_optimizer_steps']], [0, 0])
    check('first_AR_native_steps', gates['first_AR']['native_Adam_steps'], [1, 2])
    check('fresh_initialization', gates['initialization'], dict(exact_pinned_initial_state=True,
          full_rng_restored_after_construction=True, fresh_optimizer=True, inherited_qualification_learning=False))
    check('trainer_closed_before_serving', [gates['closed']['closed'], gates['closed']['failed'],
                                          gates['closed']['accumulation_remainder'],
                                          gates['closed']['global_step'], gates['closed']['presentations']],
          [True, False, 0, 256, 512])
    summary = dict(
        status='passed' if not issues else 'failed', issues=issues, checks=len(checks),
        arm='KR', objective='ordinary current-first-error KL', distinct_feedback=128,
        outer_windows=work.get('outer_windows_completed'),
        presentations=len(rows), optimizer_steps=work.get('Adam_completed'),
        sampled_anchor_occurrences=anchors_total,
        roles={role: dict(values) for role, values in role_totals.items()},
        main_native_work=work,
        startup=dict(response_forwards=2, backwards=2, optimizer_steps=0,
                     recorded=costs['startup'], work_counts=costs.get('startup_work_counts', {})),
        first_A_gate_scope='Exact retained first-two-presentation ordinary K records plus fresh initialization; historical first-step post-update tensor equivalence is not claimed.',
        rng_policy=replay['rng_policy'], pass_shuffle_draws=replay['pass_draws'],
        continuity_evidence='The live trainer checks one native optimizer identity and counters after every update, and every saved Adam counter at completion. This analysis reconciles those compact receipts; it does not reload model-sized restart tensors.',
        cost_scopes=dict(
            response_training='Existing core train_step, backward/clip/Adam and metric materialization; excludes feature-file loading.',
            role_update_wall='Includes feature loading, ordinary training, sampling, metrics, row journal and bookkeeping; excludes the first-A gate. Nested in event wall.',
            journal='Already nested in role-update wall; do not add it again.',
            event='Complete train_replay_event, including feature validation, updates, checks, checkpoint persistence/export and progress writes.',
            objective='Teacher-head/selection timers are nested in training, not additional costs.',
            historical='Earlier KRC training costs are historical; this run is not compute matched and has no fresh KRC training control.',
        ),
        costs=dict(event_wall_s=costs['event_wall_s'], event=costs['event'],
                   replay=replay['costs'], objective=costs['objective_work'],
                   loading_and_initialization_s=costs['loading_and_initialization_s'],
                   startup=startup, peak_allocated_bytes=costs['peak_allocated_bytes'],
                   peak_reserved_bytes=costs['peak_reserved_bytes']),
    )
    tables = dict(PRESENTATIONS=presentations, UPDATES=updates,
                  ROLE_EXPOSURE_COSTS=[dict(role=role, **values) for role, values in role_totals.items()],
                  RESPONSE_EXPOSURE=response_rows)
    return summary, checks, tables


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, *, raw=None, schedule_path=None):
    root = Path(root)
    campaign = Path(raw) if raw else root / 'raw/campaign_001'
    event = campaign / 'KR/events/event_001'
    schedule_path = Path(schedule_path) if schedule_path else root / 'raw/sealed/REPLAY_SCHEDULE.json'
    gates = dict(first_A=read(campaign / 'FIRST_A_REPRODUCTION.json'),
                 first_AR=read(event / 'FIRST_AR_WINDOW.json'),
                 initialization=read(campaign / 'INITIALIZATION.json'),
                 closed=read(campaign / 'TRAINER_CLOSED.json'))
    summary, checks, tables = reconcile(read(event / 'RESULT.json'), read(campaign / 'COSTS.json'),
                                        read(schedule_path), gates)
    destination = root / 'analysis'
    destination.mkdir(exist_ok=True)
    (destination / 'TRAINING_SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (destination / 'TRAINING_CHECKS.json').write_text(json.dumps(checks, indent=2, allow_nan=False) + '\n')
    for name, rows in tables.items():
        write_csv(destination / (name + '.csv'), rows)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).parent)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--schedule', type=Path)
    options = parser.parse_args()
    result = analyze(options.root, raw=options.raw, schedule_path=options.schedule)
    print(json.dumps({key: result[key] for key in ('status', 'issues', 'presentations', 'optimizer_steps',
                                                'sampled_anchor_occurrences')}, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
