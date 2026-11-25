#!/usr/bin/env python3
"""Independent CPU audit of the fixed one-event BASE/PAIR retained results.

No experiment, Torch or model imports. All input reads are local. The audit
never launches, resumes, repairs or overwrites an experiment. Raw tokens and
matched block identities appear only in PRIVATE_RECOMPUTATION.json.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

ARMS = ('BASE', 'PAIR')
TIMES = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COUNTS = (
    'output_tokens',
    'verification_calls',
    'accepted_draft_tokens',
    'native_accepted_draft_tokens',
    'committed_progress',
)
COMMIT = '416cb7931028621c09b8fa983e53fc8a7dcebaab'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def compare(actual, expected, location='root'):
    """Compare complete structures, permitting only floating arithmetic roundoff."""
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and set(actual) == set(expected),
            'Different object keys at ' + location,
        )
        for key in expected:
            compare(actual[key], expected[key], location + '.' + key)
    elif isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected),
            'Different list length at ' + location,
        )
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, location + '[' + str(index) + ']')
    elif isinstance(expected, float):
        require(
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            'Different finite value at ' + location,
        )
    else:
        require(actual == expected, 'Different value at ' + location)


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.identities = {}

    def read(self, relative, lines=False):
        path = self.root / relative
        payload = path.read_bytes()
        self.identities[str(relative)] = dict(
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
        return (
            [json.loads(row) for row in payload.splitlines() if row]
            if lines
            else json.loads(payload)
        )


def audit_raw(record, row, request, canonical, checkpoint):
    """Rebuild measured counters from saved native blocks and exact output tokens."""
    for name in ('prompt_ids', 'response_ids', 'finish_reason', 'stop_token_ids', 'max_new_tokens'):
        compare(record[name], canonical[name], 'raw canonical ' + name)
    compare(record['prompt_ids'], request['prompt_ids'], 'raw prompt')
    require(
        record['checkpoint_id'] == checkpoint
        and record['purpose'] == 'evaluation'
        and record['session_id'] == 'opencode_final58'
        and record['request_index'] == request['request_index'],
        'Raw execution identity changed',
    )
    label = f"a{row['activation']:03d}_{row['phase']}_{row['request_ordinal']:04d}_{row['method']}"
    require(record['record_id'] == label, 'Raw slot label changed')
    output = record['response_ids']
    require(1 <= len(output) <= 2048 and record['max_new_tokens'] == 2048, 'Output cap changed')
    stops = record['stop_token_ids']
    require(not any(token in stops for token in output[:-1]), 'Interior EOS in raw output')
    require(
        (record['finish_reason'] == 'eos' and output[-1] in stops)
        or (
            record['finish_reason'] == 'length' and len(output) == 2048 and output[-1] not in stops
        ),
        'Raw termination changed',
    )
    metadata = record['metadata']
    trace = metadata['trace']
    capture = metadata['capture']
    require(
        metadata['feedback_capture_in_timed_request'] is False
        and metadata['block_size'] == 16
        and metadata['context_limit'] == 4096
        and metadata['runtime'] == 'pinned_SGLang_FA3'
        and metadata['cache_policy'] == 'idle_full_flush_before_every_request_and_publication'
        and capture['collect_features'] is False
        and capture['feature_bytes'] == 0
        and capture['native_capture_validated'] is True,
        'Raw serving/capture contract differs',
    )
    compare(capture['prompt_ids'], record['prompt_ids'], 'capture prompt')
    compare(capture['response_ids'], output, 'capture output')
    compare(capture['blocks'], trace, 'capture blocks')
    committed = []
    for index, block in enumerate(trace):
        draft, teacher = block['draft_tokens'], block['target_argmax']
        width, progress = block['verify_size'], block['committed_progress']
        require(
            type(width) is int
            and width == len(draft) + 1 == len(teacher)
            and 1 <= width <= 16
            and type(progress) is int
            and 1 <= progress <= width,
            'Invalid native block geometry',
        )
        accepted = next((i for i, token in enumerate(draft) if token != teacher[i]), len(draft))
        expected_tokens = (draft[:accepted] + [teacher[accepted]])[:progress]
        require(
            progress <= accepted + 1
            and block['committed_tokens'] == expected_tokens
            and block['committed_tokens'] == teacher[:progress],
            'Native posterior decision differs',
        )
        require(
            block['output_start'] == len(committed)
            and block['anchor_token'] == output[len(committed)],
            'Native trace output offset differs',
        )
        start = len(record['prompt_ids']) + block['output_start']
        require(
            block['positions'] == list(range(start, start + width)), 'Native trace positions differ'
        )
        require(
            block['accepted_draft_tokens'] == min(accepted, progress)
            and block['native_accepted_draft_tokens'] == progress - 1
            and block['raw_accepted_draft_tokens_before_stop'] == accepted
            and block['stop_clipped'] == (progress < accepted + 1),
            'Native acceptance counters differ',
        )
        committed.extend(block['committed_tokens'])
    require(committed == output[1:], 'Trace does not reconstruct output after the prefill token')
    rebuilt = dict(
        output_tokens=len(output),
        verification_calls=len(trace),
        accepted_draft_tokens=sum(block['accepted_draft_tokens'] for block in trace),
        native_accepted_draft_tokens=sum(block['native_accepted_draft_tokens'] for block in trace),
        committed_progress=len(committed),
    )
    for key, value in rebuilt.items():
        require(row[key] == value, 'Raw-to-measurement count differs: ' + key)
        if key not in ('output_tokens', 'native_accepted_draft_tokens'):
            require(metadata[key] == value, 'Raw metadata count differs: ' + key)
    for key in TIMES:
        value = capture[key] if key == 'cache_maintenance_s' else metadata[key]
        require(
            math.isfinite(value) and value >= 0 and (key == 'cache_maintenance_s' or value > 0),
            'Invalid raw timing',
        )
        require(row[key] == value, 'Raw-to-measurement timing differs')
    require(
        metadata['native_generate_s'] <= metadata['request_latency_s'], 'Timing scopes inverted'
    )
    return len(trace)


def serving(rows):
    """Request means, ratio of sums and paired whole-request percentile bootstrap."""
    grouped = defaultdict(list)
    for row in rows:
        require(row['valid_measurement'] is True and not row['errors'], 'Invalid measurement')
        if row['phase'] == 'scored':
            grouped[row['request_id'], row['method']].append(row)
    ids = sorted({key[0] for key in grouped})
    require(len(ids) == 128 and len(grouped) == 256, 'Not the fixed 128-request two-arm panel')
    paired = []
    values = {arm: {} for arm in ARMS}
    for request in ids:
        item = {'request_id': request}
        require(
            len({row['source_id'] for arm in ARMS for row in grouped[request, arm]}) == 1,
            'Paired source identity differs',
        )
        for arm in ARMS:
            repeated = sorted(grouped[request, arm], key=lambda row: row['repeat'])
            require(
                [row['repeat'] for row in repeated] == [0, 1, 2],
                'Timing repeats missing or duplicated',
            )
            first = repeated[0]
            for key in (
                *COUNTS,
                'finish_reason',
                'checkpoint_id',
                'physical_gpu_uuid',
                'source_id',
            ):
                require(
                    all(row[key] == first[key] for row in repeated),
                    'Repeat identity/count disagreement',
                )
            item[arm] = {key: first[key] for key in (*COUNTS, 'finish_reason')}
            item[arm]['repeat_times'] = {key: [row[key] for row in repeated] for key in TIMES}
            item[arm].update({key: sum(item[arm]['repeat_times'][key]) / 3 for key in TIMES})
        require(
            item['BASE']['output_tokens'] == item['PAIR']['output_tokens']
            and item['BASE']['finish_reason'] == item['PAIR']['finish_reason'],
            'Paired termination differs',
        )
        paired.append(item)
    totals = {}
    for arm in ARMS:
        for key in (*TIMES, *COUNTS):
            values[arm][key] = np.array([item[arm][key] for item in paired], dtype=np.float64)
        sums = {key: float(vector.sum()) for key, vector in values[arm].items()}
        calls = sums['verification_calls']
        for name, metric in [
            ('native_accepted_per_call', 'native_accepted_draft_tokens'),
            ('stop_clipped_accepted_per_call', 'accepted_draft_tokens'),
            ('committed_per_call', 'committed_progress'),
        ]:
            sums[name] = sums[metric] / calls if calls else None
        sums['tokens_per_s'] = sums['output_tokens'] / sums['request_latency_s']
        sums['repeat_totals'] = {
            key: [sum(item[arm]['repeat_times'][key][r] for item in paired) for r in range(3)]
            for key in TIMES
        }
        sums['termination_counts'] = dict(Counter(item[arm]['finish_reason'] for item in paired))
        sums['termination_counts'] = {
            kind: sums['termination_counts'].get(kind, 0) for kind in ('eos', 'length')
        }
        lengths = values[arm]['output_tokens']
        sums['output_length'] = dict(
            min=int(min(lengths)),
            max=int(max(lengths)),
            mean=float(np.mean(lengths)),
            median=float(np.median(lengths)),
        )
        totals[arm] = sums
    draws = np.random.Generator(np.random.PCG64(20261124)).integers(0, 128, size=(10000, 128))
    contrast = dict(candidate='PAIR', baseline='BASE')
    metrics = [(key + '_reduction_pct', key) for key in TIMES]
    metrics.append(('verification_call_reduction_pct', 'verification_calls'))
    for name, key in metrics:
        left, right = values['PAIR'][key], values['BASE'][key]
        if right.sum() == 0 or np.any(right[draws].sum(axis=1) == 0):
            contrast[name] = dict(value=None, ci95=None, status='undefined')
            continue
        point = 100 * (1 - left.sum() / right.sum())
        sampled = 100 * (1 - left[draws].sum(axis=1) / right[draws].sum(axis=1))
        contrast[name] = dict(
            value=float(point),
            ci95=np.quantile(sampled, [0.025, 0.975], method='linear').tolist(),
            status=(
                'prespecified_primary' if key == 'request_latency_s' else 'descriptive_unadjusted'
            ),
        )
    delta = values['BASE']['verification_calls'] - values['PAIR']['verification_calls']
    contrast.update(
        wins=int(sum(delta > 0)),
        regressions=int(sum(delta < 0)),
        ties=int(sum(delta == 0)),
        gross_calls_saved=int(sum(delta[delta > 0])),
        gross_calls_added=int(-sum(delta[delta < 0])),
        net_calls_saved=int(sum(delta)),
    )
    return dict(
        requests=128,
        paired=paired,
        totals=totals,
        contrasts={'PAIR_vs_BASE': contrast},
        bootstrap=dict(
            draws=10000,
            generator='PCG64',
            seed=20261124,
            input_order=ids,
            input_order_digest=digest(ids),
            draw_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
            percentiles='linear 2.5/97.5; paired whole requests; average timing repeats within each request',
            counts='one validated repeat per request; timing repeats are not independent training runs',
        ),
    )


def block_state(row, index):
    block = row['blocks']
    predictions, labels, valid = [block[key][0][index] for key in ('proposals', 'labels', 'valid')]
    require(
        len(predictions) == len(labels) == len(valid) == 16 and valid[0] is False,
        'Witness block geometry/known seed differs',
    )
    require(
        not any(valid[j] and not valid[j - 1] for j in range(2, 16)),
        'Witness validity reopened after gap',
    )
    prefix = [False] * 16
    error = [False] * 16
    matching = True
    first_error = None
    for position in range(1, 16):
        matching = matching and valid[position] and predictions[position] == labels[position]
        prefix[position] = matching
        if first_error is None and valid[position] and predictions[position] != labels[position]:
            first_error = position
            error[position] = True
    require(
        block['prefix'][0][index] == prefix
        and block['first_error'][0][index] == error
        and block['prefix_length'][0][index] == sum(prefix),
        'Witness derived masks/count differ',
    )
    return predictions, labels, valid, prefix, first_error


def margin_check(observation, frozen, predictions, labels, valid):
    if frozen is None:
        require(observation is None, 'Invented margin on no-error origin block')
        return
    for key in ('position', 'canonical_token_id', 'original_rival_token_id'):
        require(observation[key] == frozen[key], 'Frozen repair identity changed')
    position, target, rival = [
        observation[key] for key in ('position', 'canonical_token_id', 'original_rival_token_id')
    ]
    wrong = observation['current_wrong_token_id']
    original, current = observation['M_original'], observation['M_current']
    require(
        valid[position] and labels[position] == target and target != rival and target != wrong,
        'Invalid witness repair binding',
    )
    require(
        math.isfinite(original) and math.isfinite(current) and current <= original,
        'Nonfinite or impossible strongest-wrong margin',
    )
    if wrong == rival:
        require(current == original, 'Same wrong rival has inconsistent margin')
    correct = predictions[position] == target
    require(
        observation['native_predicted_token_id'] == predictions[position]
        and observation['correct'] == correct,
        'Witness native correctness differs',
    )
    require(
        correct == (current > 0 or (current == 0 and target < wrong)),
        'Full-vocabulary margin/tie contradicts native correctness',
    )
    if not correct:
        require(
            predictions[position] == wrong,
            'Native wrong prediction differs from strongest wrong ID',
        )
    expected = dict(
        original_margin_positive=original > 0,
        original_margin_tie=original == 0,
        current_margin_tie=current == 0,
        original_rival_addressed=original > 0 or (original == 0 and target < rival),
        rival_changed=wrong != rival,
    )
    for key, value in expected.items():
        require(observation[key] == value, 'Witness margin predicate differs: ' + key)


def witnesses(origin, base, pair):
    require(
        origin['arm'] == base['arm'] == 'BASE' and pair['arm'] == 'PAIR',
        'Witness arm binding differs',
    )
    require(
        len(origin['rows']) == len(base['rows']) == len(pair['rows']) == 8,
        'Witness is not first eight responses',
    )
    for item, step in ((origin, 0), (base, 128), (pair, 128)):
        require(
            item['status'] == 'passed'
            and item['global_step'] == step
            and item['margin_schema'] == 'repair_pair_origin_margin_v1',
            'Incomplete witness readout',
        )
        require(item['witness_identity'] == origin['witness_identity'], 'Witness identity changed')
        require(
            item['teacher_forwards'] == item['backward_calls'] == item['optimizer_steps'] == 0,
            'Witness performed disallowed work',
        )
        require(item['rng_preserved'] is True, 'Witness RNG restoration failed')
        require(
            item['diagnostic_draft_forwards_attempted'] == item['diagnostic_draft_forwards'] <= 8,
            'Witness diagnostic attempts do not reconcile',
        )
        require(
            item['diagnostic_head_calls_attempted'] == item['diagnostic_head_calls'] <= 8,
            'Witness head attempts do not reconcile',
        )
        if step:
            require(
                item['origin_readout_digest'] == digest(origin)
                and item['presentations'] == 256
                and item['completed_events'] == 1,
                'Endpoint does not bind the completed event and origin',
            )
    table = []
    for response_index, first in enumerate(origin['rows']):
        require(len(first['anchors']) <= 8, 'More than eight fixed anchors')
        for endpoint in (base, pair):
            current = endpoint['rows'][response_index]
            require(
                current['record_id'] == first['record_id']
                and current['anchors'] == first['anchors'],
                'Witness response/anchor changed',
            )
        if first['blocks'] is None:
            require(
                not first['anchors']
                and not first['margin_observations']
                and all(
                    endpoint['rows'][response_index]['blocks'] is None
                    and not endpoint['rows'][response_index]['margin_observations']
                    for endpoint in (base, pair)
                ),
                'Empty witness support changed',
            )
            continue
        for index, anchor in enumerate(first['anchors']):
            op, labels, valid, prefix, error = block_state(first, index)
            frozen = first['margin_observations'][index]
            require((error is None) == (frozen is None), 'Origin error/margin support differs')
            if frozen is not None:
                require(
                    frozen['position'] == error and frozen['original_rival_token_id'] == op[error],
                    'Origin rival is not the native first error',
                )
            margin_check(frozen, frozen, op, labels, valid)
            row = dict(
                witness_identity=origin['witness_identity'],
                record_id=first['record_id'],
                response_index=response_index,
                block_index=index,
                anchor=anchor,
                origin_prefix_length=sum(prefix),
                origin_margin=frozen,
            )
            for arm, endpoint in (('BASE', base), ('PAIR', pair)):
                current = endpoint['rows'][response_index]
                predictions, y, support, new_prefix, _ = block_state(current, index)
                require(labels == y and valid == support, 'Witness labels or validity changed')
                margin = current['margin_observations'][index]
                margin_check(margin, frozen, predictions, labels, valid)
                damage = sum(
                    keep and prediction != label
                    for keep, prediction, label in zip(prefix, predictions, labels)
                )
                repaired = margin is not None and margin['correct']
                row[arm] = dict(
                    margin=margin,
                    boundary_repaired=repaired,
                    useful_repair=repaired and damage == 0,
                    damaged_prefix_positions=damage,
                    damaged_prefix_block=damage > 0,
                    prefix_length=sum(new_prefix),
                    prefix_length_change=sum(new_prefix) - sum(prefix),
                )
                if frozen is not None:
                    for key in ('M_original', 'M_current'):
                        row[arm][key + '_change'] = margin[key] - frozen[key]
            row['PAIR_minus_BASE_prefix_length'] = (
                row['PAIR']['prefix_length'] - row['BASE']['prefix_length']
            )
            if frozen is not None:
                for key in ('M_original', 'M_current'):
                    row['PAIR_minus_BASE_' + key] = (
                        row['PAIR']['margin'][key] - row['BASE']['margin'][key]
                    )
            table.append(row)
    repairs = [row for row in table if row['origin_margin'] is not None]
    summary = dict(
        block_occurrences=len(table),
        response_count=8,
        origin_repair_occurrences=len(repairs),
        original_prefix_positions=sum(row['origin_prefix_length'] for row in table),
        PAIR_minus_BASE_prefix_length=sum(row['PAIR_minus_BASE_prefix_length'] for row in table),
    )
    for label, predicate in [
        (
            'PAIR_only',
            lambda row: row['PAIR']['useful_repair'] and not row['BASE']['useful_repair'],
        ),
        (
            'BASE_only',
            lambda row: row['BASE']['useful_repair'] and not row['PAIR']['useful_repair'],
        ),
        ('both', lambda row: row['BASE']['useful_repair'] and row['PAIR']['useful_repair']),
        (
            'neither',
            lambda row: not row['BASE']['useful_repair'] and not row['PAIR']['useful_repair'],
        ),
    ]:
        summary['useful_repair_' + label] = sum(predicate(row) for row in repairs)
    for label, sign in [('improved', 1), ('regressed', -1), ('tied', 0)]:
        summary['PAIR_' + label + '_blocks'] = sum(
            (
                (row['PAIR_minus_BASE_prefix_length'] > 0)
                - (row['PAIR_minus_BASE_prefix_length'] < 0)
            )
            == sign
            for row in table
        )
    for arm in ARMS:
        local = [row[arm] for row in table]
        margins = [row[arm]['margin'] for row in repairs]
        result = {
            'boundaries_repaired': sum(row['boundary_repaired'] for row in local),
            'useful_repairs': sum(row['useful_repair'] for row in local),
            'damaged_prefix_positions': sum(row['damaged_prefix_positions'] for row in local),
            'damaged_prefix_blocks': sum(row['damaged_prefix_block'] for row in local),
            'full_valid_block_prefix_length': sum(row['prefix_length'] for row in local),
            'net_prefix_length_change': sum(row['prefix_length_change'] for row in local),
            'old_rival_addressed_but_still_wrong': sum(
                row['original_rival_addressed'] and not row['correct'] for row in margins
            ),
            'positive_original_margin_but_still_wrong': sum(
                row['M_original'] > 0 and not row['correct'] for row in margins
            ),
            'addressed_still_wrong_with_rival_change': sum(
                row['original_rival_addressed'] and not row['correct'] and row['rival_changed']
                for row in margins
            ),
            'rival_changed': sum(row['rival_changed'] for row in margins),
            'original_margin_ties': sum(row['M_original'] == 0 for row in margins),
            'current_margin_ties': sum(row['M_current'] == 0 for row in margins),
        }
        for label, sign in [('improved', 1), ('regressed', -1), ('tied', 0)]:
            result[label + '_blocks'] = sum(
                ((row['prefix_length_change'] > 0) - (row['prefix_length_change'] < 0)) == sign
                for row in local
            )
        for key in ('M_original', 'M_current'):
            changes = [row[arm]['margin'][key] - row['origin_margin'][key] for row in repairs]
            result[key + '_change_mean'] = sum(changes) / len(changes) if changes else None
        summary[arm] = result
    for key in ('M_original', 'M_current'):
        changes = [row['PAIR']['margin'][key] - row['BASE']['margin'][key] for row in repairs]
        summary['PAIR_minus_BASE_' + key + '_mean'] = (
            sum(changes) / len(changes) if changes else None
        )
    return summary, table


def audit(root, destination):
    reader = Inputs(root)
    destination = Path(destination).resolve()
    require(
        destination != reader.root and not destination.is_relative_to(reader.root),
        'Write the audit outside the immutable execution tree',
    )
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(
        status='running',
        extra_model_work=0,
        execution_commit=COMMIT,
        raw_generations_checked=0,
        native_blocks_checked=0,
    )
    try:
        plan = reader.read('sealed/PLAN.json')
        require(
            plan['binding']
            == digest({key: value for key, value in plan.items() if key != 'binding'}),
            'Plan digest differs',
        )
        require(
            plan['execution_commit'] == COMMIT
            and plan['execution_mode'] == 'one_event_production'
            and plan['production_admitted'] is True
            and plan['maximum_events'] == 1,
            'Wrong or unadmitted experiment',
        )
        reuse = reader.read('sealed/QUALIFICATION_REUSE.json')
        require(
            reuse['status'] == 'reused_passed'
            and reuse['optimizer_steps_executed'] == 0
            and reuse['qualified_execution_commit'] == '79b8c076610e9b4d310169b3891cb76ac6d0f9bb',
            'Qualification was not explicitly reused',
        )
        launch = reader.read('LAUNCH.json')
        require(
            launch['status'] == 'completed'
            and launch['qualification_executed'] is False
            and launch['qualification_reused'] is True
            and not launch['pending']
            and not launch['unattempted'],
            'Launch incomplete or repeated qualification',
        )
        require(
            [(row['phase'], row['arm']) for row in launch['phases']]
            == [('train', 'BASE'), ('train', 'PAIR'), ('evaluate', None)],
            'Phase order differs',
        )
        expected_arm = dict(
            feedback_generation=128,
            teacher_preparation=128,
            response_forward=256,
            response_backward=256,
            adam=128,
        )
        arm_statuses = {}
        for arm in ARMS:
            status = reader.read('arms/' + arm + '/STATUS.json')
            arm_statuses[arm] = status
            initial = reader.read('arms/' + arm + '/trainer/INITIAL_CHECK.json')
            require(
                initial['status'] == 'passed'
                and initial['global_step'] == 0
                and initial['presentations'] == initial['adam_state_entries'] == 0
                and initial['actual_state_unchanged'] is True
                and initial['export_exact'] is True,
                'Fresh released initialization failed',
            )
            if arm == 'PAIR':
                require(
                    initial['paired_initialization']['status'] == 'passed'
                    and initial['paired_initialization']['exact_weights_empty_Adam_RNG'] is True,
                    'BASE/PAIR origins differ',
                )
            restored = reader.read('arms/' + arm + '/startup/EXACT_RESTORATION.json')
            require(restored['exact_equal'] is True, 'Native startup did not restore exact state')
            require(
                status['status'] == status['phase'] == 'completed'
                and status['training_card'] == plan['devices']['B']
                and status['serving_card'] == plan['devices']['A'],
                'Arm incomplete or misallocated',
            )
            require(
                status['startup_work'] == dict(response_forward=2, response_backward=2, adam=0),
                'Qualified startup work changed',
            )
            ledger = reader.read('arms/' + arm + '/WORK.json')
            require(
                ledger['attempted'] == ledger['completed'] == expected_arm
                and not ledger['pending'],
                'Arm physical work has failed/unknown/excess operations',
            )
        evaluation = reader.read('evaluation/STATUS.json')
        require(
            evaluation['status'] == 'completed'
            and all(
                evaluation[key] == 792
                for key in ('planned', 'attempted', 'returned', 'completed_valid')
            )
            and not any(
                evaluation[key] for key in ('invalid', 'pending', 'unattempted', 'unknown')
            ),
            'Evaluation has incomplete/invalid/unknown work',
        )
        ledger = reader.read('evaluation_work.json')
        expected_evaluation = dict(scored_generation=768, warmup_generation=24)
        require(
            ledger['attempted'] == ledger['completed'] == expected_evaluation
            and not ledger['pending'],
            'Evaluation ledger differs from 768 scored plus 24 warmups',
        )
        inputs = reader.read('sealed/INPUTS.json')
        requests, warmups = inputs['evaluation_requests'], inputs['warmup_requests']
        require(len(requests) == 128 and len(warmups) == 2, 'Panel sizes differ')
        protocol = reader.read('evaluation/PROTOCOL.json')
        require(
            protocol['gpu'] == plan['devices']['A'] and protocol['feedback_capture'] is False,
            'Scoring device or capture changed',
        )
        expected_schedule, expected_slots = [], []
        for repeat in range(3):
            for group in range(2):
                order = ARMS if (repeat + group) % 2 == 0 else ARMS[::-1]
                panel = requests[group * 64 : (group + 1) * 64]
                for position, arm in enumerate(order):
                    activation = len(expected_schedule)
                    expected_schedule.append(
                        dict(
                            activation=activation,
                            repeat=repeat,
                            group=group,
                            position=position,
                            method=arm,
                            warmups=2,
                            request_ids=[row['request_id'] for row in panel],
                        )
                    )
                    for phase, batch in (('warmup', warmups), ('scored', panel)):
                        for request in batch:
                            expected_slots.append(
                                (
                                    dict(
                                        activation=activation,
                                        repeat=repeat,
                                        group=group,
                                        method=arm,
                                        phase=phase,
                                        request_ordinal=request['request_index'],
                                    ),
                                    request,
                                )
                            )
        compare(protocol['schedule'], expected_schedule, 'evaluation schedule')
        rows = reader.read('evaluation/MEASUREMENTS.jsonl', lines=True)
        require(len(rows) == len(expected_slots) == 792, 'Raw measurement count differs')
        canonicals = reader.read('sealed/CANONICALS.json')
        warmup_canonicals = reader.read('sealed/WARMUP_CANONICALS.json')
        for row, (slot, request) in zip(rows, expected_slots):
            for key, value in slot.items():
                require(row[key] == value, 'Measurement schedule order differs')
            require(
                row['request_id'] == request['request_id']
                and row['source_id'] == request['source_id']
                and row['physical_gpu_uuid'] == plan['devices']['A'],
                'Measurement input/device identity differs',
            )
            require(
                row['valid_measurement'] is True and not row['errors'],
                'Invalid recorded measurement',
            )
            canonical = (warmup_canonicals if slot['phase'] == 'warmup' else canonicals)[
                request['request_id']
            ]
            if set(canonical) == {'AR', 'F'}:
                canonical = canonical['F']
            require(
                canonical['binding_digest']
                == digest(
                    {key: value for key, value in canonical.items() if key != 'binding_digest'}
                )
                and canonical['verified'] is True
                and canonical['reference_kind'] == 'frozen_DFlash',
                'Frozen canonical is not authenticated',
            )
            raw_relative = Path(row['raw_path']).relative_to(Path(plan['root']))
            require(raw_relative.parent == Path('evaluation/raw'), 'Raw path escaped evaluation')
            record = reader.read(raw_relative)
            report['native_blocks_checked'] += audit_raw(
                record, row, request, canonical, protocol['checkpoints'][slot['method']]
            )
            report['raw_generations_checked'] += 1
        repeat_counts = defaultdict(list)
        for row in rows:
            repeat_counts[row['phase'], row['method'], row['request_id']].append(row)
        for repeated in repeat_counts.values():
            first = tuple(repeated[0][key] for key in (*COUNTS, 'finish_reason'))
            require(
                all(
                    tuple(row[key] for key in (*COUNTS, 'finish_reason')) == first
                    for row in repeated
                ),
                'Warmup or scored repeat-count disagreement',
            )
        recomputed = serving(rows)
        compare(reader.read('analysis/SERVING.json'), recomputed, 'serving analysis')
        origin = reader.read('witness/origin.json')
        base = reader.read('witness/BASE1.json')
        pair = reader.read('witness/PAIR1.json')
        seal = arm_statuses['BASE']['witness_seal']
        require(
            seal['optimizer_steps'] == seal['diagnostic_forwards'] == 0
            and seal['rng_preserved'] is True,
            'Witness was not frozen before updates',
        )
        features = reader.read('arms/BASE/FEATURE_IDENTITIES.json')
        require(
            len(features) == 128 and len(seal['rows']) == 8,
            'Own features or witness seal incomplete',
        )
        for index, frozen in enumerate(seal['rows']):
            require(
                frozen['ordinal'] == index
                and frozen['record_id'] == inputs['train_requests'][index]['request_id']
                and frozen['record_id'] == origin['rows'][index]['record_id']
                and frozen['anchors'] == origin['rows'][index]['anchors']
                and frozen['feature_identity'] == features[index]['identity'],
                'Witness is not the first eight ordered own BASE responses',
            )
        witness_summary, per_block = witnesses(origin, base, pair)
        recorded = reader.read('analysis/WITNESS_COMPARISON.json')
        compare(recorded['summary'], witness_summary, 'witness summary')
        compare(recorded['per_block'], per_block, 'witness per-block table')
        clock = launch['clock']
        require(
            clock['elapsed_s'] < 10800 and clock['occupied_s'] < 18000,
            'New clock exceeds authorized limit',
        )
        public = dict(
            status='passed',
            execution_commit=COMMIT,
            requests=128,
            raw_generations_checked=792,
            scored_generations=768,
            warmup_generations=24,
            production_adam_steps=256,
            startup_adam_steps=0,
            qualification_reexecuted=False,
            serving_totals=recomputed['totals'],
            contrasts=recomputed['contrasts'],
            bootstrap={
                key: value for key, value in recomputed['bootstrap'].items() if key != 'input_order'
            },
            witness_summary=witness_summary,
            new_cost=dict(elapsed_s=clock['elapsed_s'], occupied_s=clock['occupied_s']),
            historical_qualification_cost={
                key: reuse['historical_cost'][key] for key in ('elapsed_s', 'occupied_s')
            },
            combined_historical_plus_new_cost={
                key: clock[key] + reuse['historical_cost'][key]
                for key in ('elapsed_s', 'occupied_s')
            },
            witness_scope='Overlapping blocks from eight training responses; no independent training replication',
            margin_limit='Saved values and tie/correctness logic checked; unretained logits were not regenerated',
            resource_limit='Separate cross-host storage/cleanup audit is required; this script does not inspect remote free space',
            extra_model_work=0,
        )
        private = dict(
            serving=recomputed, witness_summary=witness_summary, witness_per_block=per_block
        )
        write(destination / 'PRIVATE_RECOMPUTATION.json', private)
        write(destination / 'PUBLIC_SUMMARY.json', public)
        report.update(
            status='passed',
            witness_block_occurrences=len(per_block),
            serving_summary_matches=True,
            witness_summary_and_blocks_match=True,
            bootstrap_draw_sha256=recomputed['bootstrap']['draw_sha256'],
        )
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['input_identities'] = reader.identities
        report['auditor_identity'] = dict(
            bytes=Path(__file__).stat().st_size,
            sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        write(destination / 'AUDIT_PRIVATE.json', report)
    return report


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run-root', required=True, help='Local retained execution_002 tree, including raw JSON'
    )
    parser.add_argument(
        '--output-dir', required=True, help='New private audit directory outside the execution tree'
    )
    args = parser.parse_args()
    result = audit(args.run_root, args.output_dir)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ('status', 'raw_generations_checked', 'native_blocks_checked')
            }
        )
    )
