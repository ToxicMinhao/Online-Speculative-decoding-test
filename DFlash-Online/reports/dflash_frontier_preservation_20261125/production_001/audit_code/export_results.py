#!/usr/bin/env python3
"""Offline scalar export and independent audit of the fixed PR60 production.

No experiment/model imports, networking, GPU operations or artifact mutation.
The serving audit reuses the independent PR59 auditor arithmetic with explicit
CONTROL/FRONTIER_P labels and the new prospective bootstrap seed20261125.
Private identifiers/tokens are read only to verify retained native records.
Public exports contain explicit ordinal/scalar allowlists; detailed audit paths
and input fingerprints remain under PRIVATE_AUDIT.json.
"""

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

ARMS = ('CONTROL', 'FRONTIER_P')
TIMES = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COUNTS = (
    'output_tokens',
    'verification_calls',
    'accepted_draft_tokens',
    'native_accepted_draft_tokens',
    'committed_progress',
)
COMMIT = '225802657d630c0985abd68d6e7453f3e47c5b74'
PARENT_ROOT = '/data/sources/minhao/runs/dflash_pair_one_event_20261124/execution_002'
# Authenticated in the retained PR59 RETRIEVAL_MANIFEST.json, not new model work.
PARENT_FEATURE_IDENTITY = dict(
    bytes=43268,
    sha256='f88d8f1a4e26e3a369ee8fabf7bd2d262722978c4e94fb2aa8ff5aaf32fa5500',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
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
    def __init__(self, root, dependencies=None):
        self.root = Path(root).resolve()
        self.dependencies = (
            Path(dependencies).resolve()
            if dependencies is not None
            else self.root.parent / 'dependencies'
        )
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

    def read_dependency(self, name, expected_identity):
        require(Path(name).name == name, 'Dependency must be one fixed basename')
        payload = (self.dependencies / name).read_bytes()
        actual = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        self.identities['dependencies/' + name] = actual
        require(
            actual == expected_identity, 'Retained dependency identity changed: ' + name
        )
        return json.loads(payload)


def sealed_read(reader, plan, name):
    relative = 'sealed/' + name
    value = reader.read(relative)
    key = plan['root'] + '/' + relative
    require(
        reader.identities[relative] == plan['sealed_files'][key],
        'Sealed input bytes changed: ' + name,
    )
    return value


def validate_serving_bindings(reader, plan, inputs, references, protocol, checkpoints):
    """Bind compact evaluation evidence to the authorized inputs and endpoints."""
    compare(protocol['checkpoints'], checkpoints, 'served endpoint exports')
    require(protocol['execution_commit'] == COMMIT, 'Serving execution commit differs')
    selection = sealed_read(reader, plan, 'INPUT_SELECTION.json')
    old_inputs = reader.read_dependency(
        'PARENT_INPUTS.json', plan['sealed_files'][PARENT_ROOT + '/sealed/INPUTS.json']
    )
    for group, old_key, ids_key, reference_name in (
        ('panel', 'evaluation_requests', 'evaluation_request_ids', 'CANONICALS.json'),
        ('warmup', 'warmup_requests', 'warmup_request_ids', 'WARMUP_CANONICALS.json'),
    ):
        requests = inputs[group]
        compare(requests, old_inputs[old_key], 'authorized original ' + group)
        compare(
            [row['request_id'] for row in requests],
            selection[ids_key],
            'sealed ordered ' + group,
        )
        source = sealed_read(reader, plan, reference_name)
        normalized = {
            key: value['F'] if set(value) == {'AR', 'F'} else value
            for key, value in source.items()
        }
        compare(references[group], normalized, 'sealed frozen-F ' + group)


def audit_raw(record, row, request, canonical, checkpoint):
    """Rebuild measured counters from saved native blocks and exact output tokens."""
    for name in (
        'prompt_ids',
        'response_ids',
        'finish_reason',
        'stop_token_ids',
        'max_new_tokens',
    ):
        compare(record[name], canonical[name], 'raw canonical ' + name)
    compare(record['prompt_ids'], request['prompt_ids'], 'raw prompt')
    require(
        row['finish_reason'] == record['finish_reason'],
        'Raw-to-measurement termination differs',
    )
    require(row['checkpoint_id'] == checkpoint, 'Measurement endpoint export differs')
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
    require(
        1 <= len(output) <= 2048 and record['max_new_tokens'] == 2048,
        'Output cap changed',
    )
    stops = record['stop_token_ids']
    require(
        not any(token in stops for token in output[:-1]), 'Interior EOS in raw output'
    )
    require(
        (record['finish_reason'] == 'eos' and output[-1] in stops)
        or (
            record['finish_reason'] == 'length'
            and len(output) == 2048
            and output[-1] not in stops
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
        and metadata['cache_policy']
        == 'idle_full_flush_before_every_request_and_publication'
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
        accepted = next(
            (i for i, token in enumerate(draft) if token != teacher[i]), len(draft)
        )
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
            block['positions'] == list(range(start, start + width)),
            'Native trace positions differ',
        )
        require(
            block['accepted_draft_tokens'] == min(accepted, progress)
            and block['native_accepted_draft_tokens'] == progress - 1
            and block['raw_accepted_draft_tokens_before_stop'] == accepted
            and block['stop_clipped'] == (progress < accepted + 1),
            'Native acceptance counters differ',
        )
        require(
            type(block['stop_clipped']) is bool
            and type(block['finished']) is bool
            and block['finished'] == (index == len(trace) - 1)
            and (not block['stop_clipped'] or block['finished']),
            'Native clipping/terminal block differs',
        )
        committed.extend(block['committed_tokens'])
    require(
        committed == output[1:],
        'Trace does not reconstruct output after the prefill token',
    )
    rebuilt = dict(
        output_tokens=len(output),
        verification_calls=len(trace),
        accepted_draft_tokens=sum(block['accepted_draft_tokens'] for block in trace),
        native_accepted_draft_tokens=sum(
            block['native_accepted_draft_tokens'] for block in trace
        ),
        committed_progress=len(committed),
    )
    for key, value in rebuilt.items():
        require(
            type(row[key]) is int and row[key] == value,
            'Raw-to-measurement count differs: ' + key,
        )
        if key not in ('output_tokens', 'native_accepted_draft_tokens'):
            require(metadata[key] == value, 'Raw metadata count differs: ' + key)
    for key in TIMES:
        value = capture[key] if key == 'cache_maintenance_s' else metadata[key]
        require(
            type(value) in (int, float)
            and math.isfinite(value)
            and value >= 0
            and (key == 'cache_maintenance_s' or value > 0),
            'Invalid raw timing',
        )
        require(row[key] == value, 'Raw-to-measurement timing differs')
    require(
        metadata['native_generate_s'] <= metadata['request_latency_s'],
        'Timing scopes inverted',
    )
    return len(trace)


def serving(rows):
    """Request means, ratio of sums and paired whole-request percentile bootstrap."""
    grouped = defaultdict(list)
    for row in rows:
        require(
            row['valid_measurement'] is True and not row['errors'],
            'Invalid measurement',
        )
        if row['phase'] == 'scored':
            grouped[row['request_id'], row['method']].append(row)
    ids = sorted({key[0] for key in grouped})
    require(
        len(ids) == 128 and len(grouped) == 256,
        'Not the fixed 128-request two-arm panel',
    )
    paired = []
    values = {arm: {} for arm in ARMS}
    for request in ids:
        item = {'request_id': request}
        require(
            len({row['source_id'] for arm in ARMS for row in grouped[request, arm]})
            == 1,
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
            item[arm]['repeat_times'] = {
                key: [row[key] for row in repeated] for key in TIMES
            }
            item[arm].update(
                {key: sum(item[arm]['repeat_times'][key]) / 3 for key in TIMES}
            )
        require(
            item['CONTROL']['output_tokens'] == item['FRONTIER_P']['output_tokens']
            and item['CONTROL']['finish_reason'] == item['FRONTIER_P']['finish_reason'],
            'Paired termination differs',
        )
        paired.append(item)
    totals = {}
    for arm in ARMS:
        for key in (*TIMES, *COUNTS):
            values[arm][key] = np.array(
                [item[arm][key] for item in paired], dtype=np.float64
            )
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
            key: [
                sum(item[arm]['repeat_times'][key][r] for item in paired)
                for r in range(3)
            ]
            for key in TIMES
        }
        sums['termination_counts'] = dict(
            Counter(item[arm]['finish_reason'] for item in paired)
        )
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
    draws = np.random.Generator(np.random.PCG64(20261125)).integers(
        0, 128, size=(10000, 128)
    )
    contrast = dict(candidate='FRONTIER_P', baseline='CONTROL')
    metrics = [(key + '_reduction_pct', key) for key in TIMES]
    metrics.append(('verification_call_reduction_pct', 'verification_calls'))
    for name, key in metrics:
        left, right = values['FRONTIER_P'][key], values['CONTROL'][key]
        if right.sum() == 0 or np.any(right[draws].sum(axis=1) == 0):
            contrast[name] = dict(value=None, ci95=None, status='undefined')
            continue
        point = 100 * (1 - left.sum() / right.sum())
        sampled = 100 * (1 - left[draws].sum(axis=1) / right[draws].sum(axis=1))
        contrast[name] = dict(
            value=float(point),
            ci95=np.quantile(sampled, [0.025, 0.975], method='linear').tolist(),
            status=(
                'prespecified_primary'
                if key == 'request_latency_s'
                else 'descriptive_unadjusted'
            ),
        )
    delta = (
        values['CONTROL']['verification_calls']
        - values['FRONTIER_P']['verification_calls']
    )
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
        contrasts={'FRONTIER_P_vs_CONTROL': contrast},
        bootstrap=dict(
            draws=10000,
            generator='PCG64',
            seed=20261125,
            input_order=ids,
            input_order_digest=digest(ids),
            draw_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
            percentiles='linear 2.5/97.5; paired whole requests; average timing repeats within each request',
            counts='one validated repeat per request; timing repeats are not independent training runs',
        ),
    )


def read_optional(reader, relative, *, lines=False):
    return (
        reader.read(relative, lines=lines)
        if (reader.root / relative).is_file()
        else None
    )


def safe_scalar(value):
    if value is None or isinstance(value, bool):
        return value
    if type(value) in (int, float) and math.isfinite(value):
        return value
    return None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_repeats(rows, path):
    fields = [
        'phase',
        'warmup',
        'request_ordinal',
        'warmup_ordinal',
        'arm',
        'repeat',
        'activation',
        'group',
        'valid',
        'error_count',
        *COUNTS,
        'finish_reason',
        *TIMES,
    ]
    output = []
    for row in rows:
        phase = (
            row.get('phase') if row.get('phase') in ('warmup', 'scored') else 'unknown'
        )
        ordinal = safe_scalar(row.get('request_ordinal'))
        errors = row.get('errors')
        current = dict(
            phase=phase,
            warmup=phase == 'warmup',
            request_ordinal=ordinal if phase == 'scored' else None,
            warmup_ordinal=ordinal if phase == 'warmup' else None,
            arm=row.get('method') if row.get('method') in ARMS else 'unknown',
            valid=row.get('valid_measurement') is True
            and isinstance(errors, list)
            and not errors,
            error_count=len(errors) if isinstance(errors, list) else None,
            finish_reason=(
                row.get('finish_reason')
                if row.get('finish_reason') in ('eos', 'length')
                else 'unknown'
            ),
        )
        current.update(
            {
                key: safe_scalar(row.get(key))
                for key in ('repeat', 'activation', 'group', *COUNTS, *TIMES)
            }
        )
        output.append(current)
    write_csv(path, fields, output)


def export_requests(summary, requests, path):
    fields = ['request_ordinal']
    for arm in ARMS:
        fields.extend(arm + '_' + key for key in (*COUNTS, 'finish_reason', *TIMES))
    fields += ['request_latency_saved_s', 'verification_calls_saved']
    records = []
    if summary is not None:
        ordinal = {row['request_id']: row['request_index'] for row in requests}
        for item in summary['paired']:
            row = dict(request_ordinal=ordinal[item['request_id']])
            for arm in ARMS:
                row.update(
                    {
                        arm + '_' + key: item[arm][key]
                        for key in (*COUNTS, 'finish_reason', *TIMES)
                    }
                )
            row['request_latency_saved_s'] = (
                item['CONTROL']['request_latency_s']
                - item['FRONTIER_P']['request_latency_s']
            )
            row['verification_calls_saved'] = (
                item['CONTROL']['verification_calls']
                - item['FRONTIER_P']['verification_calls']
            )
            records.append(row)
        records.sort(key=lambda row: row['request_ordinal'])
    write_csv(path, fields, records)


def audit_serving(reader, report):
    plan = reader.read('sealed/PLAN.json')
    require(
        plan['execution_commit'] == COMMIT
        and plan['bootstrap_seed'] == 20261125
        and plan['arms'] == list(ARMS),
        'Execution source/labels/bootstrap contract changed',
    )
    require(
        plan['binding'] == digest({k: v for k, v in plan.items() if k != 'binding'}),
        'Plan digest changed',
    )
    checkpoints = {}
    for arm in ARMS:
        endpoint = reader.read('arms/' + arm + '/STATUS.json')
        require(
            endpoint['status'] == 'completed'
            and endpoint['global_step'] == 256
            and endpoint['presentations'] == 512
            and endpoint['completed_event'] == 2,
            'Serving requires both complete new endpoints',
        )
        require(
            endpoint['boundary']['status'] == 'passed'
            and endpoint['boundary']['export_exact'] is True
            and endpoint['boundary']['live']['exact_resume'] is True,
            'Endpoint state/export validation failed',
        )
        checkpoints[arm] = endpoint['checkpoint']
        work = reader.read('arms/' + arm + '/WORK.json')
        expected = dict(response_forward=256, response_backward=256, adam=128)
        require(
            work['attempted'] == work['completed'] == expected and not work['pending'],
            'Committed training work is incomplete or unknown',
        )
    shared = reader.read('shared/WORK.json')
    expected_shared = dict(feedback_generation=128, teacher_preparation=128)
    require(
        shared['attempted'] == shared['completed'] == expected_shared
        and not shared['pending'],
        'Shared feedback work is incomplete or unknown',
    )
    status = reader.read('evaluation/STATUS.json')
    require(
        status['status'] == 'completed'
        and all(
            status[k] == 792
            for k in ('planned', 'attempted', 'returned', 'completed_valid')
        )
        and not any(
            status[k] for k in ('invalid', 'pending', 'unattempted', 'unknown')
        ),
        'Evaluation is incomplete, invalid or unknown',
    )
    ledger = reader.read('evaluation_work.json')
    expected = dict(scored_generation=768, warmup_generation=24)
    require(
        ledger['attempted'] == ledger['completed'] == expected
        and not ledger['pending'],
        'Evaluation work does not reconcile',
    )
    inputs = reader.read('evaluation/INPUTS.json')
    requests, warmups = inputs['panel'], inputs['warmup']
    require(len(requests) == 128 and len(warmups) == 2, 'Evaluation panel size changed')
    require(
        [r['request_index'] for r in requests] == list(range(128)),
        'Evaluation request ordinals changed',
    )
    protocol = reader.read('evaluation/PROTOCOL.json')
    require(
        protocol['gpu'] == plan['devices']['A']
        and protocol['feedback_capture'] is False,
        'Scored physical card/capture differs',
    )
    schedule, slots = [], []
    for repeat in range(3):
        for group in range(2):
            order = ARMS if (repeat + group) % 2 == 0 else ARMS[::-1]
            batch = requests[group * 64 : (group + 1) * 64]
            for position, arm in enumerate(order):
                activation = len(schedule)
                schedule.append(
                    dict(
                        activation=activation,
                        repeat=repeat,
                        group=group,
                        position=position,
                        method=arm,
                        warmups=2,
                        request_ids=[row['request_id'] for row in batch],
                    )
                )
                for phase, selected in (('warmup', warmups), ('scored', batch)):
                    for request in selected:
                        slots.append(
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
    compare(protocol['schedule'], schedule, 'fixed schedule')
    rows = reader.read('evaluation/MEASUREMENTS.jsonl', lines=True)
    require(len(rows) == len(slots) == 792, 'Evaluation row count changed')
    references = reader.read('evaluation/REFERENCES.json')
    validate_serving_bindings(reader, plan, inputs, references, protocol, checkpoints)
    for row, (slot, request) in zip(rows, slots):
        for key, value in slot.items():
            require(row[key] == value, 'Measurement schedule order changed')
        require(
            row['request_id'] == request['request_id']
            and row['source_id'] == request['source_id']
            and row['physical_gpu_uuid'] == plan['devices']['A'],
            'Measurement identity/device differs',
        )
        reference = references['warmup' if slot['phase'] == 'warmup' else 'panel'][
            request['request_id']
        ]
        require(
            reference['reference_kind'] == 'frozen_DFlash'
            and reference['verified'] is True
            and reference['provenance'] == plan['canonical_provenance']
            and reference['binding_digest']
            == digest({k: v for k, v in reference.items() if k != 'binding_digest'}),
            'Frozen-F canonical binding differs',
        )
        label = f"a{slot['activation']:03d}_{slot['phase']}_{slot['request_ordinal']:04d}_{slot['method']}"
        raw = reader.read('evaluation/raw/' + label + '.json')
        report['native_blocks_checked'] += audit_raw(
            raw, row, request, reference, protocol['checkpoints'][slot['method']]
        )
        report['raw_generations_checked'] += 1
    computed = serving(rows)
    saved = reader.read('analysis/SERVING.json')
    compare(saved, computed, 'independent serving reduction')
    return computed, requests


METRICS = (
    'raw_repair',
    'useful_repair',
    'damaged_prefix_positions',
    'damaged_prefix_blocks',
    'surviving_followers',
    'reachable_followers',
    'matching_prefix_length',
    'fixed_K_shortfall',
    'assessment',
)
LOWER_BETTER = {
    'damaged_prefix_positions',
    'damaged_prefix_blocks',
    'fixed_K_shortfall',
    'assessment',
}


def roles_from_readout(block):
    current, proposals = block['current'], block['proposals']
    labels, valid = current['labels'], current['valid']
    require(
        len(labels) == len(valid) == len(proposals) == 16 and valid[0] is False,
        'Witness block geometry changed',
    )
    require(
        not any(valid[j] and not valid[j - 1] for j in range(2, 16)),
        'Witness validity reopened after gap',
    )
    m = sum(valid)
    errors = [j for j in range(1, 16) if valid[j] and proposals[j] != labels[j]]
    e = errors[0] if errors else -1
    length = e - 1 if errors else m
    endpoint = errors[1] - 1 if len(errors) > 1 else m
    horizon = endpoint - length if errors else 0
    expected = dict(
        labels=labels,
        valid=valid,
        prefix=[bool(valid[j] and j <= length) for j in range(16)],
        repair=[bool(valid[j] and j == e) for j in range(16)],
        L=length,
        e=e,
        H=horizon,
        K=endpoint,
        m=m,
    )
    compare(current, expected, 'witness current roles')
    return expected


def witness_metrics(block, parent):
    current = roles_from_readout(block)
    origin, q = block['origin'], block['proposals']
    compare(
        parent['origin'],
        roles_from_readout(parent),
        'parent origin roles from native predictions',
    )
    compare(origin, parent['origin'], 'frozen witness roles')
    require(
        current['labels'] == origin['labels'] and current['valid'] == origin['valid'],
        'Witness labels/support changed',
    )
    require(
        len(block['positions']) == origin['K'], 'Witness assessment support changed'
    )
    for j, point in enumerate(block['positions'], 1):
        old = parent['positions'][j - 1]
        require(
            point['position'] == j
            and point['canonical'] == origin['labels'][j]
            and point['prediction'] == q[j]
            and point['origin_rival'] == old['rival'],
            'Witness token/rival binding changed',
        )
        y, wrong, old_wrong = point['canonical'], point['rival'], point['origin_rival']
        margin, original_margin = point['margin'], point['origin_rival_margin']
        require(
            math.isfinite(margin)
            and math.isfinite(original_margin)
            and y not in (wrong, old_wrong),
            'Invalid witness margin identity',
        )
        require(
            margin <= original_margin
            and point['correct'] == (q[j] == y)
            and point['correct'] == (margin > 0 or (margin == 0 and y < wrong)),
            'Witness native tie/correctness differs',
        )
        require(
            point['margin_tie'] == (margin == 0)
            and point['rival_changed'] == (wrong != old_wrong)
            and point['origin_rival_addressed']
            == (original_margin > 0 or (original_margin == 0 and y < old_wrong)),
            'Witness tie/old-rival predicate differs',
        )
    damage = sum(
        flag and q[j] != origin['labels'][j] for j, flag in enumerate(origin['prefix'])
    )
    repaired = origin['e'] >= 1 and q[origin['e']] == origin['labels'][origin['e']]
    followers = (
        list(range(origin['e'] + 1, origin['K'] + 1)) if origin['e'] >= 1 else []
    )
    expected = dict(
        raw_repair=int(repaired),
        useful_repair=int(repaired and not damage),
        damaged_prefix_positions=damage,
        damaged_prefix_blocks=int(damage > 0),
        surviving_followers=sum(q[j] == origin['labels'][j] for j in followers),
        reachable_followers=sum(current['L'] >= j for j in followers),
        matching_prefix_length=current['L'],
        fixed_K_shortfall=max(0, origin['K'] - current['L']),
    )
    for key, value in expected.items():
        require(
            block['metrics'][key] == value, 'Witness derived metric differs: ' + key
        )
    margins = np.asarray([p['margin'] for p in block['positions']], dtype=np.float32)
    weighted = (
        np.arange(origin['K'], 0, -1, dtype=np.float32)
        * np.logaddexp(np.float32(0), -margins)
        / np.float32(math.log(2))
    )
    require(
        math.isclose(
            block['metrics']['assessment'],
            float(weighted.sum(dtype=np.float32)),
            rel_tol=5e-6,
            abs_tol=5e-5,
        ),
        'Retained-margin FP32 assessment does not reconcile',
    )


def paired_witness_comparison(first, second, label):
    result = {}
    for metric in METRICS:
        changes = [
            b['metrics'][metric] - a['metrics'][metric] for a, b in zip(first, second)
        ]
        sign = -1 if metric in LOWER_BETTER else 1
        result[metric] = dict(
            comparison=label,
            first_sum=sum(b['metrics'][metric] for b in first),
            second_sum=sum(b['metrics'][metric] for b in second),
            delta=sum(changes),
            gains=sum(sign * delta > 0 for delta in changes),
            losses=sum(sign * delta < 0 for delta in changes),
            ties=sum(delta == 0 for delta in changes),
            lower_is_better=sign < 0,
        )
    return result


def audit_witness_bindings(reader, receipts):
    """Authenticate selection bytes and their original ordered feature lists."""
    plan = reader.read('sealed/PLAN.json')
    selection = sealed_read(reader, plan, 'INPUT_SELECTION.json')
    shared = reader.read('shared/FEATURE_IDENTITIES.json')
    require(len(shared) == 128, 'Shared feature list is incomplete')
    for ordinal, (item, selected) in enumerate(zip(shared, selection['training'])):
        require(
            item['local_ordinal'] == selected['local_ordinal'] == ordinal
            and item['source_index'] == selected['source_index'] == ordinal + 128
            and item['source_request_id'] == selected['request_id'],
            'Shared feature source selection changed',
        )
    require(len(selection['training']) == 128, 'Sealed shared selection size differs')
    parent_features = reader.read_dependency(
        'PARENT_FEATURE_IDENTITIES.json', PARENT_FEATURE_IDENTITY
    )
    require(len(parent_features) == 128, 'Parent feature list is incomplete')
    binding = reader.read('witness/BINDINGS.json')
    schema = 'frontier_fixed_training_witness_v1'
    require(
        binding['schema'] == schema
        and binding['status'] == 'passed'
        and binding['origin_counters'] == [128, 256, 1]
        and binding['live_rng_before'] == binding['live_rng_after']
        and binding['diagnostic_forwards'] == binding['optimizer_steps'] == 0
        and binding['maximum_anchors_per_response'] == 8
        and binding['diagnostic_only'] is True
        and set(binding['sets']) == {'new_buffer', 'parent_training'},
        'Witness freeze contract changed',
    )
    identifiers, paths = set(), set()
    for name, features in (
        ('new_buffer', shared),
        ('parent_training', parent_features),
    ):
        items = binding['sets'][name]
        require(len(items) == 8, 'Witness freeze lost first-eight coverage')
        for ordinal, (item, feature) in enumerate(zip(items, features[:8])):
            require(
                item['ordinal'] == ordinal
                and item['feature_path'] == feature['path']
                and item['feature_identity'] == feature['identity']
                and len(item['anchors']) <= 8
                and item['anchors'] == sorted(set(item['anchors']))
                and all(type(a) is int and a >= 0 for a in item['anchors']),
                'Witness feature order, identity or anchors changed',
            )
            if name == 'new_buffer':
                require(
                    item['record_id'] == feature['record_id'],
                    'Shared witness record changed',
                )
            require(
                item['record_id'] not in identifiers
                and item['feature_path'] not in paths,
                'Witness response sets overlap',
            )
            identifiers.add(item['record_id'])
            paths.add(item['feature_path'])
    for endpoint, receipt in receipts.items():
        require(
            receipt['schema'] == schema
            and receipt['diagnostic_only'] is True
            and receipt['binding_identity']
            == reader.identities['witness/BINDINGS.json']
            and set(receipt['sets']) == set(binding['sets']),
            'Witness readout is not bound to the actual freeze file',
        )
        for name, items in binding['sets'].items():
            rows = receipt['sets'][name]
            require(
                len(rows) == len(items), 'Witness readout count differs from binding'
            )
            for row, item in zip(rows, items):
                require(
                    row['record_id'] == item['record_id']
                    and row['anchors'] == item['anchors'],
                    'Witness readout response or anchors differ from binding',
                )
        if endpoint == 'parent':
            require(
                receipt['origin_digest'] is None,
                'Parent cannot use a substituted origin',
            )
    return binding


def audit_witnesses(reader):
    endpoints = ('parent', *ARMS)
    receipts = {
        endpoint: reader.read('witness/' + endpoint + '.json') for endpoint in endpoints
    }
    audit_witness_bindings(reader, receipts)
    parent = receipts['parent']
    for endpoint, receipt in receipts.items():
        require(
            receipt['status'] == 'passed'
            and receipt['endpoint'] == endpoint
            and receipt['binding_identity'] == parent['binding_identity']
            and receipt['counters']
            == ([128, 256, 1] if endpoint == 'parent' else [256, 512, 2])
            and receipt['rng_preserved'] is True
            and receipt['parameters_and_counters_unchanged'] is True,
            'Witness state/binding or completion differs',
        )
        require(
            receipt['teacher_forwards']
            == receipt['backwards']
            == receipt['optimizer_steps']
            == 0,
            'Witness performed forbidden training/teacher work',
        )
        require(
            receipt['draft_forwards_attempted']
            == receipt['draft_forwards_completed']
            <= 16
            and receipt['head_calls_attempted']
            == receipt['head_calls_completed']
            <= 16,
            'Witness physical work differs',
        )
        if endpoint != 'parent':
            require(
                receipt['origin_digest'] == digest(parent),
                'Witness origin digest differs',
            )
    saved = reader.read('analysis/WITNESSES.json')
    require(
        set(saved) == {'schema', 'diagnostic_only', 'units', 'definitions', 'sets'},
        'Unexpected witness public-summary fields',
    )
    require(
        saved['diagnostic_only'] is True
        and set(saved['sets']) == {'new_buffer', 'parent_training'},
        'Witness summary scope differs',
    )
    for name in ('new_buffer', 'parent_training'):
        flat = {}
        for endpoint, receipt in receipts.items():
            rows = receipt['sets'][name]
            require(len(rows) == 8, 'Witness lost an ordered response')
            for row, old in zip(rows, parent['sets'][name]):
                require(
                    row['record_id'] == old['record_id']
                    and row['anchors'] == old['anchors']
                    and len(row['blocks'])
                    == len(old['blocks'])
                    == len(row['anchors'])
                    <= 8,
                    'Matched witness anchors changed',
                )
                for block, origin in zip(row['blocks'], old['blocks']):
                    witness_metrics(block, origin)
                nonempty = sum(b['origin']['m'] > 0 for b in row['blocks'])
                expected = (
                    sum(b['metrics']['assessment'] for b in row['blocks']) / nonempty
                    if nonempty
                    else 0.0
                )
                require(
                    row['nonempty_blocks'] == nonempty,
                    'Witness response support differs',
                )
                compare(
                    row['response_assessment'], expected, 'witness response assessment'
                )
            flat[endpoint] = [block for row in rows for block in row['blocks']]
        target = saved['sets'][name]
        original = flat['parent']
        counts = dict(
            responses=8,
            block_occurrences=len(original),
            nonempty_blocks=sum(b['origin']['m'] > 0 for b in original),
            original_errors=sum(b['origin']['e'] >= 1 for b in original),
            original_prefix_positions=sum(b['origin']['L'] for b in original),
            usable_followers=sum(max(0, b['origin']['H'] - 1) for b in original),
        )
        require(
            set(target)
            == set(counts)
            | {
                'parent_to_CONTROL',
                'parent_to_FRONTIER_P',
                'CONTROL_to_FRONTIER_P',
                'endpoints',
            },
            'Unexpected witness set-summary fields',
        )
        for key, value in counts.items():
            compare(target[key], value, 'witness support counts')
        for first, second, key, label in (
            ('parent', 'CONTROL', 'parent_to_CONTROL', 'CONTROL minus parent'),
            ('parent', 'FRONTIER_P', 'parent_to_FRONTIER_P', 'FRONTIER_P minus parent'),
            (
                'CONTROL',
                'FRONTIER_P',
                'CONTROL_to_FRONTIER_P',
                'FRONTIER_P minus CONTROL',
            ),
        ):
            compare(
                target[key],
                paired_witness_comparison(flat[first], flat[second], label),
                'witness aggregate comparison',
            )
        for endpoint, blocks in flat.items():
            points = [point for block in blocks for point in block['positions']]
            expected = dict(
                response_mean_assessment=sum(
                    row['response_assessment']
                    for row in receipts[endpoint]['sets'][name]
                )
                / 8,
                assessed_positions=len(points),
                margin_mean=(
                    sum(p['margin'] for p in points) / len(points) if points else None
                ),
                origin_rival_margin_mean=(
                    sum(p['origin_rival_margin'] for p in points) / len(points)
                    if points
                    else None
                ),
                margin_ties=sum(p['margin_tie'] for p in points),
                wrong_rival_ties=sum(p['wrong_rival_tie'] for p in points),
                rival_changes=sum(p['rival_changed'] for p in points),
                old_rival_addressed_but_wrong=sum(
                    p['origin_rival_addressed'] and not p['correct'] for p in points
                ),
            )
            compare(target['endpoints'][endpoint], expected, 'witness endpoint summary')
    return dict(
        status='audited',
        **saved,
        audit_limit='Full vocabulary logits and feature tensors are not re-projected; strongest-rival identity and wrong-rival ties rely on retained native readouts.',
    )


def counts_receipt(value):
    if value is None:
        return dict(
            status='missing', attempted=None, completed=None, pending_operations=None
        )
    schema_ok = (
        isinstance(value, dict)
        and isinstance(value.get('attempted'), dict)
        and isinstance(value.get('completed'), dict)
        and isinstance(value.get('pending'), (dict, list))
        and all(
            type(v) is int and v >= 0
            for key in ('attempted', 'completed')
            for v in value[key].values()
        )
    )
    if not schema_ok:
        return dict(
            status='malformed_or_unknown',
            attempted=None,
            completed=None,
            pending_operations=None,
        )
    unknown = value['pending'] or value['attempted'] != value['completed']
    return dict(
        status='pending_or_unknown' if unknown else 'recorded',
        attempted=value.get('attempted'),
        completed=value.get('completed'),
        pending_operations=len(value.get('pending', {})),
    )


def validate_cost_receipts(plan, launch, clock, windows):
    """Reconcile completed physical phases/windows without imputing missing work."""
    checks = {}
    expected_phases = ['collect', 'train_CONTROL', 'train_FRONTIER_P', 'evaluate']
    finite_nonnegative = (
        lambda value: type(value) in (int, float)
        and math.isfinite(value)
        and value >= 0
    )
    clock = clock if isinstance(clock, dict) else {}
    finished = clock.get('finished', [])
    checks['final_phase_order'] = [
        item.get('phase') for item in finished
    ] == expected_phases and all(item.get('status') == 'completed' for item in finished)
    checks['no_unresolved_phase'] = clock.get('active', 'missing') is None
    checks['phase_times_finite'] = all(
        finite_nonnegative(item.get('elapsed_s')) for item in finished
    )
    completed = clock.get('completed_card_s')
    checks['phase_sum'] = (
        finite_nonnegative(completed)
        and checks['phase_times_finite']
        and math.isclose(
            sum(item['elapsed_s'] for item in finished),
            completed,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    )
    final = launch.get('clock') or {}
    production, cumulative = final.get('production_occupied_card_s'), final.get(
        'cumulative_occupied_card_s'
    )
    prior = plan.get('prior_occupied_card_s')
    checks['final_clock_totals'] = (
        all(finite_nonnegative(v) for v in (production, cumulative, prior, completed))
        and final.get('prior_occupied_card_s') == prior
        and math.isclose(production, completed, rel_tol=1e-12, abs_tol=1e-9)
        and math.isclose(cumulative, prior + completed, rel_tol=1e-12, abs_tol=1e-9)
        and final.get('active', 'missing') is None
    )
    elapsed = final.get('new_phase_elapsed_s')
    checks['four_hour_limit'] = finite_nonnegative(elapsed) and elapsed <= 14400
    checks['eight_card_hour_limit'] = (
        finite_nonnegative(cumulative) and cumulative <= 28800
    )
    checks['launch_completed'] = launch.get('status') == 'completed'
    checks['fixed_window_ids'] = [row.get('window') for row in windows] == list(
        range(1, 129)
    )
    checks['window_steps_and_work'] = (
        len(windows) == 128
        and all(
            row.get('status') == 'completed'
            and row.get('outcome_window') == row.get('window')
            and row.get('native_adam_steps') == 1
            and row.get('work', {}).get('status') == 'recorded'
            and row.get('native', {}).get('start_step') == 127 + row['window']
            and row.get('native', {}).get('end_step') == 128 + row['window']
            and row.get('native', {}).get('presentations') == 2
            and row.get('native', {}).get('native_adam_steps') == 1
            for row in windows
            if type(row.get('window')) is int
        )
        and all(type(row.get('window')) is int for row in windows)
    )
    checks['window_times_finite_ordered'] = all(
        finite_nonnegative(row.get('score_s'))
        and finite_nonnegative(row.get('total_window_s'))
        and row['score_s'] <= row['total_window_s']
        for row in windows
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        failures=[name for name, passed in checks.items() if not passed],
    )


def flat_numeric(value):
    return {
        key: number
        for key, number in (value or {}).items()
        if type(number) in (int, float, bool) and math.isfinite(number)
    }


def physical_receipt(value):
    if value is None:
        return dict(status='missing')
    # Counter names are fixed by the native worker, never request identifiers.
    result = flat_numeric(value)
    for key in (
        'counts',
        'attempted',
        'attempts',
        'completed',
        'unknown',
        'limits',
        'physical',
    ):
        if isinstance(value.get(key), dict):
            result[key] = flat_numeric(value[key])
    if isinstance(value.get('initialization'), list):
        result['initialization'] = [
            dict(flat_numeric(row), status=row.get('status', 'unknown'))
            for row in value['initialization']
        ]
    result['status'] = value.get('status', 'recorded')
    return result


def cost_summary(reader):
    plan = read_optional(reader, 'sealed/PLAN.json') or {}
    launch = read_optional(reader, 'LAUNCH.json') or {}
    clock = read_optional(reader, 'OCCUPIED_CARDS.json')
    result = dict(
        execution_commit=COMMIT,
        status=launch.get('status', 'missing'),
        scope='Measured scopes overlap; do not sum nested times. Prior card cost includes all earlier PR60 qualifications.',
        prior_occupied_card_s=plan.get('prior_occupied_card_s'),
        final_clock=launch.get('clock'),
        shared_work=counts_receipt(read_optional(reader, 'shared/WORK.json')),
        evaluation_work=counts_receipt(read_optional(reader, 'evaluation_work.json')),
        arms={},
        allocation_windows=[],
        phase_times=[],
        storage={},
        physical_work=dict(
            shared=physical_receipt(read_optional(reader, 'shared/PHYSICAL_WORK.json')),
            evaluation=physical_receipt(
                read_optional(reader, 'evaluation_physical_work.json')
            ),
        ),
        witness_work={},
    )
    if clock is not None:
        result['production_completed_card_s'] = clock['completed_card_s']
        result['occupied_phase_unresolved'] = clock['active'] is not None
        result['phase_times'] = [
            {key: row[key] for key in ('phase', 'elapsed_s', 'status')}
            for row in clock['finished']
        ]
    for stage in ('COLLECTION_ADMISSION', 'CONTINUATION_ADMISSION', 'STORAGE_STAGE'):
        value = read_optional(reader, stage + '.json')
        if value is not None:
            result['storage'][stage] = {
                key: scalar
                for key, scalar in value.items()
                if key != 'stage' and safe_scalar(scalar) is not None
            }
    for arm in ARMS:
        base = 'arms/' + arm + '/'
        status = read_optional(reader, base + 'STATUS.json')
        value = dict(
            status='missing' if status is None else status['status'],
            work=counts_receipt(read_optional(reader, base + 'WORK.json')),
        )
        if status is not None:
            value.update(
                {
                    key: status.get(key)
                    for key in ('global_step', 'presentations', 'completed_event')
                }
            )
            value['training_costs'] = {
                key: number
                for key, number in status.get('training', {}).get('costs', {}).items()
                if type(number) in (int, float) and math.isfinite(number)
            }
            value['startup_wall_s'] = status.get('startup', {}).get('wall_s')
            startup = status.get('startup', {})
            value['startup_work'] = dict(
                status=startup.get('status', 'missing'),
                optimizer_steps=startup.get('optimizer_steps'),
                work=physical_receipt(startup.get('work')),
                physical_work=physical_receipt(startup.get('physical_work')),
            )
            value['recovery_size'] = {
                key: number
                for key, number in status.get('recovery_size', {}).items()
                if type(number) in (int, float, bool)
            }
        result['arms'][arm] = value
    for endpoint in ('parent', *ARMS):
        receipt = read_optional(reader, 'witness/' + endpoint + '.json')
        result['witness_work'][endpoint] = (
            dict(status='missing')
            if receipt is None
            else {
                key: receipt.get(key)
                for key in (
                    'status',
                    'wall_s',
                    'draft_forwards_attempted',
                    'draft_forwards_completed',
                    'head_calls_attempted',
                    'head_calls_completed',
                    'teacher_forwards',
                    'backwards',
                    'optimizer_steps',
                )
            }
        )
    allocation_root = reader.root / 'arms/FRONTIER_P/trainer/events/event_002'
    for directory in sorted(allocation_root.glob('allocation_*')):
        relative = directory.relative_to(reader.root)
        work = read_optional(reader, relative / 'WORK.json')
        outcome = read_optional(reader, relative / 'RESULT.json')
        row = dict(
            window=int(directory.name.split('_')[-1]),
            status='completed' if outcome is not None else 'incomplete',
            work=counts_receipt(work),
            native_adam_steps=None if work is None else work['native_adam_steps'],
        )
        if outcome is not None:
            row.update({key: outcome[key] for key in ('score_s', 'total_window_s')})
            row['outcome_window'] = outcome['window']
            row['native'] = {
                key: outcome['native'][key]
                for key in (
                    'start_step',
                    'end_step',
                    'presentations',
                    'native_adam_steps',
                )
            }
            row['preview'] = {
                key: value
                for key, value in outcome['preview'].items()
                if safe_scalar(value) is not None
            }
            row['response_scores'] = (
                None
                if outcome.get('response_scores') is None
                else [
                    {
                        key: safe_scalar(item.get(key))
                        for key in (
                            'minimum',
                            'maximum',
                            'mass_moved_from_beta',
                            'mass_moved_from_old_C',
                        )
                    }
                    for item in outcome['response_scores']
                ]
            )
        result['allocation_windows'].append(row)
    complete = [
        row for row in result['allocation_windows'] if row['status'] == 'completed'
    ]
    result['allocation_observed'] = dict(
        windows=len(result['allocation_windows']),
        completed=len(complete),
        score_s_sum=sum(row['score_s'] for row in complete),
        total_window_s_sum=sum(row['total_window_s'] for row in complete),
        unfinished_windows=len(result['allocation_windows']) - len(complete),
        completeness_is_not_imputed=len(complete) == 128
        and len(result['allocation_windows']) == 128,
    )
    for kind in ('attempted', 'completed'):
        totals = Counter()
        for row in result['allocation_windows']:
            totals.update(row['work'].get(kind) or {})
        result['allocation_observed'][kind + '_operation_counts'] = dict(totals)
    result['selected_update_displacement_norms'] = dict(
        status='unavailable',
        windows=[1, 64, 128],
        reason='Production receipts did not retain actual/shadow parameter-displacement norms; the disposable integration diagnostic is a different scope.',
        additional_model_work=0,
    )
    result['reconciliation'] = validate_cost_receipts(
        plan, launch, clock, result['allocation_windows']
    )
    startup_expected = dict(
        response_forwards=2, response_backwards=2, optimizer_steps=0
    )
    startup_zero = {key: 0 for key in startup_expected}
    result['paid_scope_checks'] = {}
    for arm in ARMS:
        startup = result['arms'][arm].get('startup_work') or {}
        work = startup.get('work') or {}
        result['paid_scope_checks']['startup_' + arm] = (
            startup.get('status') == 'completed'
            and startup.get('optimizer_steps') == 0
            and work.get('attempts') == work.get('completed') == startup_expected
            and work.get('unknown') == startup_zero
            and isinstance((startup.get('physical_work') or {}).get('counts'), dict)
        )
    for endpoint, work in result['witness_work'].items():
        result['paid_scope_checks']['witness_' + endpoint] = (
            work.get('status') == 'passed'
            and work.get('teacher_forwards')
            == work.get('backwards')
            == work.get('optimizer_steps')
            == 0
            and type(work.get('draft_forwards_attempted')) is int
            and type(work.get('head_calls_attempted')) is int
            and work['draft_forwards_attempted']
            == work.get('draft_forwards_completed')
            <= 16
            and work['head_calls_attempted'] == work.get('head_calls_completed') <= 16
        )
    for phase, work in result['physical_work'].items():
        initialization = work.get('initialization')
        result['paid_scope_checks']['serving_initialization_' + phase] = (
            work.get('status') == 'recorded'
            and isinstance(initialization, list)
            and bool(initialization)
            and all(
                row.get('status') == 'completed'
                and row.get('model_forward_calls')
                == row.get('model_forward_completions')
                == 3
                and row.get('model_forward_failures') == 0
                for row in initialization
            )
        )
    result['complete_work_records'] = (
        result['reconciliation']['passed']
        and all(result['paid_scope_checks'].values())
        and result['status'] == 'completed'
        and clock is not None
        and clock['active'] is None
        and result['allocation_observed']['completeness_is_not_imputed']
        and all(
            row['native_adam_steps'] == 1 and row['work']['status'] == 'recorded'
            for row in complete
        )
        and all(
            result['arms'][arm]['status'] == 'completed'
            and result['arms'][arm]['work']['status'] == 'recorded'
            for arm in ARMS
        )
        and result['shared_work']['status']
        == result['evaluation_work']['status']
        == 'recorded'
    )
    return result


def export_allocation_windows(costs, path):
    fields = [
        'window',
        'status',
        'score_s',
        'total_window_s',
        'native_adam_steps',
        'pending_operations',
        'window_assessment',
        'gradient_norm',
        'clip',
        'clipping_active',
    ]
    rows = []
    for value in costs.get('allocation_windows', []):
        row = {
            key: value.get(key)
            for key in (
                'window',
                'status',
                'score_s',
                'total_window_s',
                'native_adam_steps',
            )
        }
        row['pending_operations'] = value['work']['pending_operations']
        row.update({key: value.get('preview', {}).get(key) for key in fields[6:]})
        rows.append(row)
    write_csv(path, fields, rows)


def export(root, destination, dependencies=None):
    reader = Inputs(root, dependencies)
    destination = Path(destination).resolve()
    require(
        destination != reader.root and not destination.is_relative_to(reader.root),
        'Exports must be outside the immutable raw tree',
    )
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(
        status='running',
        execution_commit=COMMIT,
        extra_model_work=0,
        raw_generations_checked=0,
        native_blocks_checked=0,
        sections={},
    )
    rows = []
    measurements = reader.root / 'evaluation/MEASUREMENTS.jsonl'
    if measurements.is_file():
        payload = measurements.read_bytes()
        reader.identities['evaluation/MEASUREMENTS.jsonl'] = dict(
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
        for ordinal, line in enumerate(payload.splitlines()):
            if not line:
                continue
            try:
                value = json.loads(line)
                require(isinstance(value, dict), 'Measurement row is not an object')
                rows.append(value)
            except Exception as error:
                report['measurement_decode_error'] = dict(
                    line=ordinal + 1, error=repr(error)
                )
                break
    export_repeats(rows, destination / 'ALL_REPEATS.csv')
    summary, requests = None, []
    try:
        summary, requests = audit_serving(reader, report)
        report['sections']['serving'] = dict(status='passed')
        public = {key: summary[key] for key in ('requests', 'totals', 'contrasts')}
        public.update(
            status='audited_complete',
            execution_commit=COMMIT,
            bootstrap={
                key: value
                for key, value in summary['bootstrap'].items()
                if key != 'input_order'
            },
        )
    except Exception as error:
        report['sections']['serving'] = dict(
            status='incomplete_or_invalid',
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        public = dict(
            status='incomplete_or_invalid',
            execution_commit=COMMIT,
            observed_repeat_rows=len(rows),
            gain_estimates_withheld=True,
        )
    write_json(destination / 'SERVING_SUMMARY.json', public)
    export_requests(summary, requests, destination / 'PER_REQUEST.csv')
    try:
        witness = audit_witnesses(reader)
        report['sections']['witnesses'] = dict(status='passed')
    except Exception as error:
        witness = dict(status='incomplete_or_invalid', diagnostic_only=True)
        report['sections']['witnesses'] = dict(
            status='incomplete_or_invalid',
            error=repr(error),
            traceback=traceback.format_exc(),
        )
    write_json(destination / 'WITNESS_SUMMARY.json', witness)
    try:
        costs = cost_summary(reader)
        report['sections']['costs'] = dict(
            status=(
                'recorded'
                if costs['complete_work_records']
                else 'incomplete_or_unknown'
            )
        )
    except Exception as error:
        costs = dict(status='incomplete_or_invalid', execution_commit=COMMIT)
        report['sections']['costs'] = dict(
            status='incomplete_or_invalid',
            error=repr(error),
            traceback=traceback.format_exc(),
        )
    write_json(destination / 'COSTS_SUMMARY.json', costs)
    export_allocation_windows(costs, destination / 'ALLOCATION_WINDOWS.csv')
    report['status'] = (
        'passed'
        if all(
            row['status'] in ('passed', 'recorded')
            for row in report['sections'].values()
        )
        else 'incomplete_or_invalid'
    )
    report['input_identities'] = reader.identities
    write_json(destination / 'PRIVATE_AUDIT.json', report)
    (destination / 'TABLE_DEFINITIONS.md').write_text(
        'Execution: `'
        + COMMIT
        + '`. These files contain scalar metrics and local ordinals only.\n\n'
        '`ALL_REPEATS.csv` retains every observed scored/warmup row, including invalid rows; incomplete rows have blank missing values. Warmups are excluded from gain estimates. '
        '`PER_REQUEST.csv` contains one row per original panel ordinal only after all 792 generations and both endpoints validate; time columns average the three repeats, while deterministic counts use one checked repeat.\n\n'
        '`accepted_draft_tokens` is the stop-clipped accepted-proposal count. `native_accepted_draft_tokens` is committed progress minus verification calls. '
        '`committed_progress` excludes the initial prefill token and equals output tokens minus one.\n\n'
        'Primary gain is 100×(1−sum(FRONTIER_P request means)/sum(CONTROL request means)). The 95% interval uses10,000 paired whole-request PCG64 resamples, seed20261125, linear2.5/97.5percentiles. '
        'This reused development panel conditions on two fixed endpoints; it is not training-seed replication.\n\n'
        'Witnesses are overlapping training-block diagnostics. Full vocabulary logits are not retained, so strongest-wrong identities and ties cannot be independently re-projected. '
        'Cost scopes overlap and must not be added. Missing or pending work stays unknown. Private audit input fingerprints and failure detail remain in `PRIVATE_AUDIT.json`; do not publish that file.\n'
    )
    return dict(
        status=report['status'],
        serving=report['sections']['serving']['status'],
        witnesses=report['sections']['witnesses']['status'],
        observed_repeat_rows=len(rows),
        extra_model_work=0,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--dependencies',
        help='Retained compact parent metadata; default ROOT/../dependencies',
    )
    args = parser.parse_args()
    result = export(args.root, args.output, args.dependencies)
    print(json.dumps(result))
    sys.exit(0 if result['status'] == 'passed' else 2)
