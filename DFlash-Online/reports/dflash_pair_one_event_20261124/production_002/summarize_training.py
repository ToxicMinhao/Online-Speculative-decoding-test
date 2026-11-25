#!/usr/bin/env python3
"""Summarize completed local PAIR records into public aggregates, with no model imports.

Read original response observations once; never pool accumulation-window metrics
with their component responses. Inputs remain private and are never modified.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ARMS = ('BASE', 'PAIR')
COMMIT = '416cb7931028621c09b8fa983e53fc8a7dcebaab'
WORK = dict(
    feedback_generation=128,
    teacher_preparation=128,
    response_forward=256,
    response_backward=256,
    adam=128,
)
PHYSICAL = (
    'draft_forwards',
    'layer_forwards',
    'objective_chunks',
    'vocabulary_head_calls',
    'response_forwards',
)
PAIR_FIELDS = (
    'repair_rows',
    'active_rows',
    'mu_sum',
    'complement_mass_sum',
    'conditional_kl_sum',
    'teacher_y_preferred_rows',
    'teacher_tie_rows',
    'teacher_r_preferred_rows',
    'log_mu_roundoff_clamps',
)
FIT_FIELDS = (
    'boundaries_repaired',
    'useful_repairs',
    'damaged_prefix_positions',
    'damaged_prefix_blocks',
    'full_valid_block_prefix_length',
    'net_prefix_length_change',
    'old_rival_addressed_but_still_wrong',
    'positive_original_margin_but_still_wrong',
    'addressed_still_wrong_with_rival_change',
    'rival_changed',
    'original_margin_ties',
    'current_margin_ties',
    'improved_blocks',
    'regressed_blocks',
    'tied_blocks',
    'M_original_change_mean',
    'M_current_change_mean',
)
FIT_COMMON = (
    'block_occurrences',
    'response_count',
    'origin_repair_occurrences',
    'original_prefix_positions',
    'useful_repair_PAIR_only',
    'useful_repair_BASE_only',
    'useful_repair_both',
    'useful_repair_neither',
    'PAIR_minus_BASE_prefix_length',
    'PAIR_improved_blocks',
    'PAIR_regressed_blocks',
    'PAIR_tied_blocks',
    'PAIR_minus_BASE_M_original_mean',
    'PAIR_minus_BASE_M_current_mean',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(
        type(value) in (int, float) and math.isfinite(value),
        'Nonfinite/non-numeric observation',
    )
    return value


def close(left, right):
    # Saved FP32 numerator additions/divisions can differ from Python FP64 by
    # rounding. This is an audit comparison, never a change to the objective.
    return math.isclose(left, right, rel_tol=2e-6, abs_tol=2e-6)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def distribution(values):
    values = sorted(finite(value) for value in values)
    require(bool(values), 'Cannot summarize an empty distribution')

    def quantile(p):
        point = (len(values) - 1) * p
        low = int(point)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (point - low)

    return dict(
        count=len(values),
        mean=math.fsum(values) / len(values),
        min=values[0],
        median=quantile(0.5),
        p95=quantile(0.95),
        max=values[-1],
    )


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.fixed_file_identities = {}

    def read(self, relative, *, lines=False, public_identity=True):
        payload = (self.root / relative).read_bytes()
        if public_identity:
            self.fixed_file_identities[str(relative)] = dict(
                bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
            )
        if lines:
            return [json.loads(line) for line in payload.splitlines() if line]
        return json.loads(payload)


def loss_summary(rows, arm):
    """Equal-response means are the training reduction; pooled ratios are separate."""
    output = {}
    for key in ('forward_kl', 'repair_pair_auxiliary', 'objective_loss'):
        enabled = key != 'repair_pair_auxiliary' or arm == 'PAIR'
        if enabled:
            pairs = [row['ratio_metrics'][key] for row in rows]
        else:
            require(
                all(key not in row['ratio_metrics'] for row in rows),
                'BASE unexpectedly contains an auxiliary metric',
            )
            pairs = [[0.0, row['ratio_metrics']['forward_kl'][1]] for row in rows]
        for numerator, denominator in pairs:
            finite(numerator)
            require(finite(denominator) > 0, 'Nonpositive response denominator')
        numerator = math.fsum(pair[0] for pair in pairs)
        denominator = math.fsum(pair[1] for pair in pairs)
        output[key] = dict(
            enabled=enabled,
            responses=len(rows),
            response_mean=math.fsum(a / b for a, b in pairs) / len(rows),
            numerator_sum=numerator,
            denominator_sum=denominator,
            denominator_weighted_mean=numerator / denominator,
        )
    output['recorded_total_loss'] = distribution([row['loss'] for row in rows])
    output['auxiliary_to_forward_kl_response_mean_ratio'] = ratio(
        output['repair_pair_auxiliary']['response_mean'], output['forward_kl']['response_mean']
    )
    return output


def activity_summary(rows, arm):
    sums = [row['sum_metrics'] for row in rows]
    repair = sum(
        row['kplus_priority1_error_count'] + row['kplus_priority2_error_count'] for row in sums
    )
    result = dict(
        repair_occurrences=repair,
        original_forward_observations=len(rows),
        selected_position_occurrences=sum(row['selected_position_occurrences'] for row in sums),
        available_position_occurrences=sum(
            row['available_position_occurrences'] for row in sums
        ),
        no_error_blocks=sum(row['no_error_blocks'] for row in sums),
        empty_blocks=sum(row['empty_blocks'] for row in sums),
    )
    result['first_error_position_counts'] = {
        str(offset): sum(row['first_error_position_' + str(offset)] for row in sums)
        for offset in range(1, 16)
    }
    require(
        close(sum(result['first_error_position_counts'].values()), repair),
        'Repair position counts do not reconcile',
    )
    if arm == 'PAIR':
        values = {
            name: math.fsum(row['repair_pair_' + name] for row in sums) for name in PAIR_FIELDS
        }
        require(
            values['repair_rows'] == repair, 'PAIR repair support differs from native roles'
        )
        require(
            0 <= values['active_rows'] <= repair, 'Invalid positive-auxiliary activity count'
        )
        require(
            sum(
                values[name]
                for name in (
                    'teacher_y_preferred_rows',
                    'teacher_tie_rows',
                    'teacher_r_preferred_rows',
                )
            )
            == repair,
            'Teacher preference counts do not partition repair rows',
        )
        result['auxiliary'] = dict(
            enabled=True,
            **values,
            active_fraction=ratio(values['active_rows'], repair),
            pair_mass_mean=ratio(values['mu_sum'], repair),
            complement_mass_mean=ratio(values['complement_mass_sum'], repair),
            conditional_kl_mean=ratio(values['conditional_kl_sum'], repair)
        )
    else:
        result['auxiliary'] = dict(
            enabled=False,
            active_rows=0,
            active_fraction=0.0,
            pair_mass_mean=None,
            complement_mass_mean=None,
            conditional_kl_mean=None,
        )
    return result


def validate_observations(event, responses, windows, arm):
    require(
        event['status'] == 'completed'
        and event['event_id'] == 'event_001'
        and event['start_step'] == 0
        and event['end_step'] == 128
        and event['response_presentations'] == 256
        and event['optimizer_steps'] == 128
        and event['passes'] == 2
        and event['reduction'] == 'response_mean',
        'Incomplete or foreign event',
    )
    require(
        len(responses) == len(event['records']) == 256 and len(windows) == 128,
        'Response/window counts differ from the fixed event',
    )
    for index, (observation, record) in enumerate(zip(responses, event['records'])):
        require(
            observation == record['response_observation'], 'Original response copies differ'
        )
        require(
            observation['scope'] == 'one_original_response_forward'
            and observation['presentation_ordinal'] == index
            and observation['pass_index'] == record['pass'] == index // 128
            and observation['record_id'] == record['record_id'],
            'Response binding differs',
        )
        for value in observation['sum_metrics'].values():
            finite(value)
        metrics = observation['ratio_metrics']
        ordinary, denominator = metrics['forward_kl']
        total, total_denominator = metrics['objective_loss']
        auxiliary = metrics['repair_pair_auxiliary'][0] if arm == 'PAIR' else 0.0
        require(
            denominator == total_denominator
            and (arm != 'PAIR' or denominator == metrics['repair_pair_auxiliary'][1]),
            'FKL/auxiliary/total response denominators differ',
        )
        require(
            close(total, ordinary + auxiliary)
            and close(observation['loss'], total / denominator),
            'Recorded FKL plus auxiliary does not reconcile with total loss',
        )
    ids = [row['record_id'] for row in responses]
    require(
        len(set(ids)) == 128
        and Counter(ids[:128]) == Counter(ids[128:])
        and all(count == 2 for count in Counter(ids).values()),
        'Own responses were not used once per pass',
    )
    for index, window in enumerate(windows):
        require(
            window['window'] == window['global_step'] == index + 1
            and window['record_ids'] == ids[2 * index : 2 * index + 2]
            and window['learning_rate'] == 3e-4,
            'Native optimizer window differs',
        )
        require(
            type(window['clipped']) is bool
            and window['clipped'] == (finite(window['gradient_norm']) > 1.0),
            'Clipping flag disagrees with preclip gradient norm',
        )
        require(
            close(
                window['mean_response_loss'],
                sum(row['loss'] for row in responses[2 * index : 2 * index + 2]) / 2,
            ),
            'Window loss differs from its two original responses',
        )


def gradient_summary(windows):
    clipped = sum(row['clipped'] for row in windows)
    return dict(
        preclip_gradient_norm=distribution([row['gradient_norm'] for row in windows]),
        clip_threshold=1.0,
        clipped_windows=clipped,
        windows=len(windows),
        clipped_fraction=clipped / len(windows),
    )


def physical_summary(receipt):
    initialization = receipt['initialization']
    require(
        all(row['status'] == 'completed' for row in initialization),
        'Incomplete native graph initialization',
    )
    return dict(
        graph_capture_operations=len(initialization),
        graph_forward_calls=sum(row['model_forward_calls'] for row in initialization),
        graph_forward_completions=sum(
            row['model_forward_completions'] for row in initialization
        ),
        graph_forward_failures=sum(row['model_forward_failures'] for row in initialization),
        graph_capture_wall_s=sum(row['wall_s'] for row in initialization),
        peak_gpu_allocated_bytes=receipt.get('peak_gpu_allocated_bytes'),
        peak_gpu_reserved_bytes=receipt.get('peak_gpu_reserved_bytes'),
        extra_model_calls=receipt['extra_model_calls'],
    )


def arm_summary(inputs, arm):
    base = Path('arms') / arm
    status = inputs.read(base / 'STATUS.json')
    require(
        status['status'] == status['phase'] == 'completed'
        and status['execution_commit'] == COMMIT
        and not status['cleanup_errors'],
        'Arm did not complete cleanly',
    )
    work = inputs.read(base / 'WORK.json')
    require(
        not work['pending'] and work['attempted'] == work['completed'] == WORK,
        'Arm work ledger is incomplete',
    )
    event_root = base / 'trainer/events/event_001'
    event = inputs.read(event_root / 'RESULT.json')
    responses = inputs.read(event_root / 'RESPONSE_OBSERVATIONS.jsonl', lines=True)
    windows = inputs.read(event_root / 'WINDOW_OBSERVATIONS.jsonl', lines=True)
    validate_observations(event, responses, windows, arm)
    require(event['costs'] == status['training'], 'Arm and event cost copies differ')
    require(
        close(event['costs']['training_s'], sum(row['training_s'] for row in event['records'])),
        'Training timer total does not reconcile',
    )
    startup = inputs.read(base / 'startup/RESULT.json')
    restoration = inputs.read(base / 'startup/EXACT_RESTORATION.json')
    require(
        startup['status'] == 'completed'
        and startup['optimizer_steps'] == 0
        and restoration['exact_equal'] is True,
        'Startup was not exactly restored',
    )
    initial = inputs.read(base / 'trainer/INITIAL_CHECK.json')
    require(
        initial['status'] == 'passed'
        and initial['global_step'] == initial['presentations'] == 0
        and initial['adam_state_entries'] == 0
        and initial['export_exact'] is True,
        'Arm did not start from exact released weights and empty Adam',
    )
    if arm == 'PAIR':
        require(
            initial['paired_initialization']['exact_weights_empty_Adam_RNG'] is True,
            'BASE/PAIR origins differ',
        )
    loop = base / 'event_001/loop'
    require((inputs.root / loop).is_dir(), 'Expected retained local_single feedback records')
    collection_paths = sorted((inputs.root / loop / 'serving').glob('*.costs.json'))
    require(len(collection_paths) == 128, 'Missing own-feedback cost receipts')
    collection = [
        inputs.read(path.relative_to(inputs.root), public_identity=False)
        for path in collection_paths
    ]
    update = inputs.read(loop / 'events/event_001.costs.json')
    loop_event = inputs.read(loop / 'events/event_001.json')
    require(len(loop_event['preparation']) == 128, 'Missing native preparation receipts')
    preparation = loop_event['preparation']
    require(
        all(
            row['status'] == 'prepared'
            and not row['label_mismatches']
            and row['terminal_feature']['status'] == 'passed'
            and row['terminal_feature']['final_feature_logits_used_for_labels'] is False
            for row in preparation
        ),
        'Native teacher preparation did not pass',
    )
    require(
        close(
            status['teacher_preparation_s'],
            sum(row['teacher_preparation_s'] for row in preparation),
        )
        and close(
            status['feature_persistence_s'],
            sum(row['feature_persistence_s'] for row in preparation),
        ),
        'Preparation costs do not reconcile',
    )
    physical = inputs.read(base / 'PHYSICAL_WORK.json')
    costs = dict(
        arm_wall_s=status['wall_s'],
        trainer_loading_s=status['trainer_loading_s'],
        serving_loading_s=status['backend_loading_s'],
        collection={
            key: math.fsum(finite(row[key]) for row in collection)
            for key in (
                'serve_call_s',
                'buffer_persistence_s',
                'generation_validation_and_recording_s',
                'native_request_latency_s',
            )
        },
        update_call_s=update['update_call_s'],
        teacher_preparation_s=status['teacher_preparation_s'],
        feature_persistence_s=status['feature_persistence_s'],
        training=event['costs'],
        objective_nested_timers={
            key: finite(value)
            for key, value in event['objective_work'].items()
            if key.endswith('_s')
        },
        startup_wall_s=startup['wall_s'],
        boundary_validation_s=status['boundary'].get('wall_s'),
        post_event_activation_s=loop_event['activation']['activation_s'],
        publication_check_activation_s=status['publication']['activation_s'],
    )
    return dict(
        work=work['completed'],
        unique_own_responses=128,
        own_feedback_output_tokens=sum(row['response_length'] for row in preparation),
        own_feedback_output_length=distribution(
            [row['response_length'] for row in preparation]
        ),
        native_terminal_feature_work={
            key: sum(row['terminal_feature'][key] for row in preparation)
            for key in ('target_forward_count', 'forward_input_rows', 'retained_feature_rows')
        },
        overall=dict(
            losses=loss_summary(responses, arm),
            activity=activity_summary(responses, arm),
            gradients=gradient_summary(windows),
        ),
        by_pass={
            str(index): dict(
                losses=loss_summary(responses[index * 128 : (index + 1) * 128], arm),
                activity=activity_summary(responses[index * 128 : (index + 1) * 128], arm),
                gradients=gradient_summary(windows[index * 64 : (index + 1) * 64]),
            )
            for index in (0, 1)
        },
        costs=costs,
        recorded_training_physical_work={
            key: event['objective_work']['work'].get(key, 0) for key in PHYSICAL
        },
        startup=dict(
            work=startup['work'],
            physical_counts={
                key: startup['physical_work']['counts'].get(key, 0) for key in PHYSICAL
            },
        ),
        native_serving_worker=physical_summary(physical),
        training_allocator_peak_bytes=None,
    )


def fit_summary(inputs):
    comparison = inputs.read('analysis/WITNESS_COMPARISON.json')['summary']
    result = {key: comparison[key] for key in FIT_COMMON}
    result.update({arm: {key: comparison[arm][key] for key in FIT_FIELDS} for arm in ARMS})
    result['readouts'] = {}
    for name, step in (('origin', 0), ('BASE1', 128), ('PAIR1', 128)):
        readout = inputs.read('witness/' + name + '.json')
        require(
            readout['status'] == 'passed'
            and readout['global_step'] == step
            and readout['teacher_forwards']
            == readout['backward_calls']
            == readout['optimizer_steps']
            == 0,
            'Witness readout is incomplete',
        )
        margins = [
            row
            for response in readout['rows']
            for row in response['margin_observations']
            if row is not None
        ]
        require(
            len(margins) == result['origin_repair_occurrences'], 'Frozen margin support changed'
        )
        result['readouts'][name] = dict(
            wall_s=readout['wall_s'],
            draft_forwards=readout['diagnostic_draft_forwards'],
            head_calls=readout['diagnostic_head_calls'],
            repair_occurrences=len(margins),
            M_original_mean=ratio(
                math.fsum(row['M_original'] for row in margins), len(margins)
            ),
            M_current_mean=ratio(math.fsum(row['M_current'] for row in margins), len(margins)),
        )
    result['scope'] = (
        'Small fixed BASE-training witness; overlapping block occurrences are not independent examples or held-out transfer evidence.'
    )
    return result


def summarize(root):
    inputs = Inputs(root)
    launch = inputs.read('LAUNCH.json')
    require(
        launch['status'] == 'completed'
        and launch['execution_commit'] == COMMIT
        and launch['qualification_executed'] is False
        and not launch['pending']
        and not launch['unattempted']
        and not launch['clock']['active'],
        'Production run is incomplete',
    )
    arms = {arm: arm_summary(inputs, arm) for arm in ARMS}
    reuse = inputs.read('sealed/QUALIFICATION_REUSE.json')
    require(
        reuse['status'] == 'reused_passed' and reuse['optimizer_steps_executed'] == 0,
        'Historical qualification was not reused',
    )
    historical = reuse['historical_cost']
    current = launch['clock']
    fit = fit_summary(inputs)
    evaluation = physical_summary(inputs.read('evaluation_physical_work.json'))
    output = dict(
        schema='pair_one_event_public_training_summary_v1',
        status='completed',
        execution_commit=COMMIT,
        arms=arms,
        common_origin_fit=fit,
        candidate_to_baseline_cost_ratios={
            key: ratio(arms['PAIR']['costs'][key], arms['BASE']['costs'][key])
            for key in (
                'arm_wall_s',
                'trainer_loading_s',
                'serving_loading_s',
                'teacher_preparation_s',
                'feature_persistence_s',
                'update_call_s',
            )
        },
        training_timer_PAIR_to_BASE_ratio=ratio(
            arms['PAIR']['costs']['training']['training_s'],
            arms['BASE']['costs']['training']['training_s'],
        ),
        clocks=dict(
            new_production={key: current[key] for key in ('elapsed_s', 'occupied_s')},
            historical_qualification={
                key: historical[key] for key in ('elapsed_s', 'occupied_s')
            },
            combined_charged_elapsed_s=current['elapsed_s'] + historical['elapsed_s'],
            combined_occupied_card_s=current['occupied_s'] + historical['occupied_s'],
        ),
        evaluation_native_worker=evaluation,
        definitions={
            'loss_reduction': 'Arithmetic mean of saved per-response numerator/denominator values; pooled ratios are labeled separately and are not the equal-response training reduction.',
            'activity': 'active_rows counts positive computed auxiliary extra>0, not nonzero gradients; pair mass and conditional KL means use all current repair occurrences.',
            'gradient': 'Native preclip norm; clipping is norm>1. A larger norm does not establish a better Adam update.',
            'timers': 'Nested measured scopes are reported separately, never added. Arm/update timers include startup and witness work; training_s excludes those separate passes.',
            'runtime_comparison': 'Each arm uses its own responses. Observed runtime differences do not isolate auxiliary-kernel overhead from trajectory differences.',
            'memory': 'Native worker peaks cover serving graph/collection/preparation lifetime on A. No training allocator peak was recorded by these sources; sampled device usage is not substituted.',
            'historical_cost': 'PR58 qualification is reported separately and added only for cumulative cost accounting; it is not new production work or evidence of training-inclusive speedup.',
            'fit_validation': 'Witness scalars are extracted from saved records; separate audit_results.py independently checks the matched summary. No logits or model forwards are recomputed.',
        },
        model_work_executed=0,
        fixed_input_file_identities=inputs.fixed_file_identities,
    )
    # No raw request IDs, token IDs, anchor arrays or private paths enter output.
    json.dumps(output, allow_nan=False)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(
        json.dumps(
            dict(status=result['status'], output=str(args.output), model_work_executed=0)
        )
    )
