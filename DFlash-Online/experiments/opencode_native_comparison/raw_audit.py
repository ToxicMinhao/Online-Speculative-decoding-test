"""Independent CPU reduction of sealed CODE evidence, without model execution.

The auditor does not call the execution validator or statistical reducer. Raw
tokens remain private; public results contain identities, counts and timings.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np

from .contract import ARMS, METHODS, PRODUCTION_WORK, QUALIFICATION_CEILINGS, digest, qualification_root

TIMES = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
COUNTS = ('output_tokens', 'verification_calls', 'accepted_draft_tokens',
          'native_accepted_draft_tokens', 'committed_progress')
CONTRASTS = (('C_CODE58', 'AREPO_CODE58'), ('C_CODE58', 'O_CODE58'),
             ('AREPO_CODE58', 'O_CODE58'), *((m, 'F') for m in ARMS),
             *((m, 'AR') for m in METHODS if m != 'AR'))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def identity(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(piece)
    return dict(bytes=Path(path).stat().st_size, sha256=checksum.hexdigest())


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def planned_schedule(requests):
    """Rebuild the prespecified order without importing the execution schedule."""
    require(len(requests) == 1024, 'The final panel must contain 1024 requests')
    orders = []
    for offset in range(5):
        forward = list(METHODS[offset:] + METHODS[:offset])
        orders.extend((forward, forward[::-1]))
    result = []
    for repeat in range(3):
        for group in range(8):
            ids = [r['request_id'] for r in requests[128 * group:128 * (group + 1)]]
            for position, method in enumerate(orders[(8 * repeat + group) % 10]):
                result.append(dict(activation=len(result), repeat=repeat, group=group,
                    position=position, method=method, warmups=2, request_ids=ids))
    return result


def planned_slots(requests, warmups):
    by_id = {r['request_id']: r for r in requests}
    slots = []
    for activation in planned_schedule(requests):
        for phase, rows in (('warmup', warmups),
                            ('scored', [by_id[k] for k in activation['request_ids']])):
            for request in rows:
                slots.append(dict(activation=activation['activation'],
                    repeat=activation['repeat'], group=activation['group'],
                    method=activation['method'], phase=phase,
                    request_ordinal=request['request_index'], request_id=request['request_id']))
    return slots


def label(slot):
    return f"a{slot['activation']:03d}_{slot['phase']}_{slot['request_ordinal']:04d}_{slot['method']}"


def check_canonical(reference, request, provenance, *, method='AR'):
    kind = 'target_only_AR' if method == 'AR' else 'frozen_DFlash'
    require(reference['reference_kind'] == kind and reference['verified'] is True,
            'Canonical is not a verified '+method+' reference')
    require(reference['provenance'] == provenance, 'Canonical producer identity changed')
    require(reference['binding_digest'] == digest({k:v for k,v in reference.items() if k != 'binding_digest'}),
            'Canonical content binding changed')
    for key in ('request_id', 'prompt_ids', 'max_new_tokens', 'stop_token_ids'):
        require(reference[key] == request[key], 'Canonical input mismatch: ' + key)


def cross_path_comparison(ar, frozen):
    """Independent cross-path observation; never substitute it for canonical checks."""
    for key in ('prompt_ids', 'max_new_tokens', 'stop_token_ids'):
        require(ar[key] == frozen[key], 'Cross-path input mismatch: '+key)
    left, right = ar['response_ids'], frozen['response_ids']
    first = None
    for index in range(max(len(left), len(right))):
        if index >= len(left) or index >= len(right) or left[index] != right[index]:
            first = index
            break
    tokens_equal, termination_equal = left == right, ar['finish_reason'] == frozen['finish_reason']
    return dict(token_ids_equal=tokens_equal, termination_equal=termination_equal,
        output_equal=tokens_equal and termination_equal, first_difference=first,
        record_output_tokens=len(left), reference_output_tokens=len(right),
        record_finish_reason=ar['finish_reason'], reference_finish_reason=frozen['finish_reason'])


def check_bundle(bundle, request, provenance):
    require(set(bundle) == {'AR', 'F'}, 'Missing separate AR/F canonical references')
    for method in ('AR', 'F'):
        check_canonical(bundle[method], request, provenance, method=method)
    return cross_path_comparison(bundle['AR'], bundle['F'])


def token_sequence(value):
    return isinstance(value, list) and all(type(token) is int and token >= 0 for token in value)


def raw_metrics(record, request, canonical, *, method, checkpoint, slot=None):
    """Recompute a native record's numbers, including the full commit chain."""
    meta, response = record['metadata'], record['response_ids']
    require(token_sequence(response) and 0 < len(response) <= request['max_new_tokens'],
            'Invalid response token sequence')
    require(token_sequence(request['prompt_ids']) and len(request['prompt_ids'])>0
            and len(request['prompt_ids'])+request['max_new_tokens']<=4096, 'Invalid logical request context')
    for key in ('prompt_ids', 'stop_token_ids', 'max_new_tokens'):
        require(record[key] == request[key], 'Raw input binding differs: ' + key)
    if canonical is not None:
        for key in ('prompt_ids', 'response_ids', 'finish_reason', 'stop_token_ids', 'max_new_tokens'):
            require(record[key] == canonical[key], 'Raw canonical mismatch: ' + key)
    require(record['checkpoint_id'] == checkpoint, 'Wrong serving checkpoint')
    stops = request['stop_token_ids']
    require(not any(token in stops for token in response[:-1]), 'Tokens appear after an EOS')
    finish = 'eos' if response[-1] in stops else 'length'
    require(finish == record['finish_reason'] and (finish == 'eos' or len(response) == request['max_new_tokens']),
            'Termination is inconsistent with tokens/cap')
    if slot is not None:
        require(record['record_id'] == label(slot) and record['session_id'] == 'opencode_final58'
                and record['request_index'] == request['request_index']
                and record['purpose'] == 'evaluation', 'Raw execution slot identity differs')
        require(meta['feedback_capture_in_timed_request'] is False, 'Timed feature capture was enabled')
    require(meta['runtime'] == 'pinned_SGLang_FA3' and meta['context_limit'] == 4096
            and meta['cache_policy'] == 'idle_full_flush_before_every_request_and_publication',
            'Serving runtime/context/cache policy changed')
    result = dict(output_tokens=len(response), finish_reason=finish,
                  request_latency_s=meta['request_latency_s'], native_generate_s=meta['native_generate_s'])
    if method == 'AR':
        require(meta['speculation_disabled'] is True and meta['physical_context_length'] == 4112,
                'AR must use the genuine target-only engine')
        native = meta['native_meta_info']
        require(native['prompt_tokens'] == len(request['prompt_ids'])
                and native['completion_tokens'] == len(response) and native['cached_tokens'] == 0,
                'AR native token accounting/cache evidence differs')
        termination = native['finish_reason']
        if termination['type'] == 'length':
            require(len(response) == request['max_new_tokens'] and termination['length'] == request['max_new_tokens'],
                    'AR native cap differs')
        else:
            match = termination.get('matched')
            require(termination['type'] == 'stop' and finish == 'eos'
                    and (response[-1] == match or isinstance(match, list) and response[-1] in match),
                    'AR native stop differs')
        require(all(meta.get(key) is None for key in COUNTS if key != 'output_tokens'),
                'AR speculative counts must be N/A')
        result.update({key:None for key in COUNTS if key != 'output_tokens'})
        result['cache_maintenance_s'] = meta['cache_maintenance_s']
        return checked_times(result)
    require(method in METHODS and meta['block_size'] == 16, 'Unknown native method/block size')
    capture, trace = meta['capture'], meta['trace']
    require(capture['prompt_ids'] == request['prompt_ids'] and capture['response_ids'] == response
            and capture['blocks'] == trace, 'Native trace and capture disagree')
    if slot is not None:
        require(capture['collect_features'] is False and capture['feature_bytes'] == 0
                and capture['native_capture_validated'] is True, 'Timed capture policy differs')
    # output_start identifies the already-known seed. Each committed suffix
    # starts one position later; the first prefill token never counts as progress.
    cursor, clipped, accepted = 0, 0, 0
    for block in trace:
        proposals, posterior = block['draft_tokens'], block['target_argmax']
        width, progress = block['verify_size'], block['committed_progress']
        require(token_sequence(proposals) and token_sequence(posterior)
                and type(width) is int and 1 <= width <= 16
                and width == len(proposals) + 1 == len(posterior), 'Invalid verifier block geometry')
        first = next((i for i, token in enumerate(proposals) if token != posterior[i]), width - 1)
        require(type(progress) is int and 1 <= progress <= first + 1, 'Invalid committed progress')
        committed = (proposals[:first] + [posterior[first]])[:progress]
        require(block['output_start'] == cursor and block['anchor_token'] == response[cursor]
                and block['committed_tokens'] == committed
                and committed == posterior[:progress] == response[cursor+1:cursor+1+progress],
                'Native block does not reproduce the canonical commit chain')
        require(block['positions'] == list(range(len(request['prompt_ids']) + cursor,
                                                 len(request['prompt_ids']) + cursor + width)),
                'Native absolute verification positions differ')
        require(block['accepted_draft_tokens'] == min(first, progress)
                and block['native_accepted_draft_tokens'] == progress - 1
                and block['raw_accepted_draft_tokens_before_stop'] == first
                and block['stop_clipped'] is (progress < first + 1), 'Native acceptance fields disagree')
        clipped += min(first, progress)
        accepted += progress - 1
        cursor += progress
    require(cursor == len(response)-1, 'Native trace ends before or after the response')
    result.update(verification_calls=len(trace), accepted_draft_tokens=clipped,
                  native_accepted_draft_tokens=accepted, committed_progress=cursor,
                  cache_maintenance_s=capture['cache_maintenance_s'])
    for key in ('verification_calls', 'accepted_draft_tokens', 'committed_progress'):
        require(meta[key] == result[key], 'Raw aggregate differs from blocks: ' + key)
    return checked_times(result)


def checked_times(result):
    for key in TIMES:
        value = result[key]
        require(type(value) in (int, float) and math.isfinite(value)
                and value >= 0 and (key == 'cache_maintenance_s' or value > 0), 'Invalid time: ' + key)
    require(result['native_generate_s'] <= result['request_latency_s'], 'Engine time exceeds complete request time')
    return result


def reduce_endpoint(rows, *, expected_requests=1024):
    """Independent request grouping and paired bootstrap; never calls analysis.py."""
    selected = [row for row in rows if row['phase'] == 'scored']
    require(all(row['valid_measurement'] is True and row['errors'] == [] for row in rows),
            'Invalid evidence cannot supply endpoint effects')
    grouped = defaultdict(dict)
    for row in selected:
        key, repeat = (row['request_id'], row['method']), row['repeat']
        require(row['method'] in METHODS and repeat in (0, 1, 2) and repeat not in grouped[key],
                'Duplicate, unknown arm or unknown repeat')
        grouped[key][repeat] = row
    ids = sorted({key[0] for key in grouped})
    require(len(ids) == expected_requests and len(grouped) == len(ids) * 5, 'Incomplete five-arm panel')
    require(len({r['physical_gpu_uuid'] for r in rows}) == 1, 'Timing spans different physical GPUs')
    for method in METHODS:
        require(len({r['checkpoint_id'] for r in selected if r['method'] == method}) == 1,
                'Endpoint changed during evaluation')
    pairs = []
    for request_id in ids:
        comparison = grouped[request_id, 'AR'][0]['ar_spec_comparison']
        require(all(row['ar_spec_comparison'] == comparison for method in METHODS
                    for row in grouped[request_id, method].values()), 'Cross-path observation changed')
        item = dict(request_id=request_id, ar_spec_comparison=comparison)
        sources = set()
        for method in METHODS:
            repeats = grouped[request_id, method]
            require(set(repeats) == {0, 1, 2}, 'Incomplete three-repeat request')
            first = repeats[0]
            for field in (*COUNTS, 'finish_reason', 'source_id', 'checkpoint_id'):
                require(all(repeats[r][field] == first[field] for r in (1, 2)), 'Repeat mismatch: ' + field)
            sources.add(first['source_id'])
            item[method] = {key:first[key] for key in (*COUNTS, 'finish_reason')}
            item[method]['repeat_times'] = {key:[checked_times(repeats[r])[key] for r in range(3)] for key in TIMES}
            item[method].update({key:sum(item[method]['repeat_times'][key]) / 3 for key in TIMES})
            if method == 'AR':
                require(all(first[key] is None for key in COUNTS if key != 'output_tokens'), 'AR counts are not N/A')
            else:
                require(all(type(first[key]) is int and first[key] >= 0 for key in COUNTS), 'Invalid count')
                require(first['committed_progress'] == first['output_tokens'] - 1
                        and first['native_accepted_draft_tokens'] == first['committed_progress'] - first['verification_calls']
                        and first['accepted_draft_tokens'] <= first['committed_progress'], 'Count identity differs')
        require(len(sources) == 1 and len({(item[m]['output_tokens'], item[m]['finish_reason'])
                    for m in METHODS if m != 'AR'}) == 1, 'Paired source or speculative output differs')
        require(comparison['record_output_tokens'] == item['AR']['output_tokens']
                and comparison['reference_output_tokens'] == item['F']['output_tokens']
                and comparison['record_finish_reason'] == item['AR']['finish_reason']
                and comparison['reference_finish_reason'] == item['F']['finish_reason']
                and comparison['termination_equal'] == (item['AR']['finish_reason'] == item['F']['finish_reason'])
                and comparison['output_equal'] == (comparison['token_ids_equal'] and comparison['termination_equal']),
                'Cross-path observation contradicts measured endpoints')
        pairs.append(item)
    draws = np.random.Generator(np.random.PCG64(20261120)).integers(0, len(ids), size=(10000, len(ids)))
    identical = np.array([row['ar_spec_comparison']['output_equal'] for row in pairs], dtype=bool)
    matched_ids = [row['request_id'] for row in pairs if row['ar_spec_comparison']['output_equal']]
    matched_draws = (np.random.Generator(np.random.PCG64(20261120)).integers(
        0,len(matched_ids),size=(10000,len(matched_ids))) if matched_ids else None)
    arrays = {m:{key:np.array([r[m][key] for r in pairs], dtype=np.float64)
                 for key in (*TIMES, *COUNTS) if not (m == 'AR' and key in COUNTS and key != 'output_tokens')}
              for m in METHODS}
    totals = {}
    for method, values in arrays.items():
        total = {key:float(values[key].sum()) for key in values}
        if method == 'AR':
            total.update({key:None for key in COUNTS if key != 'output_tokens'})
        for name, key in (('native_accepted_per_call', 'native_accepted_draft_tokens'),
                          ('stop_clipped_accepted_per_call', 'accepted_draft_tokens'),
                          ('committed_per_call', 'committed_progress')):
            total[name] = total[key] / total['verification_calls'] if total['verification_calls'] else None
        total['tokens_per_s'] = total['output_tokens'] / total['request_latency_s']
        total['repeat_totals'] = {key:[sum(r[method]['repeat_times'][key][i] for r in pairs) for i in range(3)] for key in TIMES}
        total['termination_counts'] = dict(Counter(r[method]['finish_reason'] for r in pairs))
        for kind in ('eos', 'length'):
            total['termination_counts'].setdefault(kind, 0)
        lengths = values['output_tokens']
        total['output_length'] = dict(min=int(lengths.min()), max=int(lengths.max()),
                                     mean=float(lengths.mean()), median=float(np.median(lengths)))
        totals[method] = total
    contrasts = {}
    for candidate, baseline in CONTRASTS:
        comparison = dict(candidate=candidate, baseline=baseline)
        for key in TIMES:
            comparison[key + '_reduction_pct'] = bootstrap_ratio(arrays[candidate][key], arrays[baseline][key], draws)
        if baseline == 'AR':
            a, b = arrays[candidate]['request_latency_s'], arrays[baseline]['request_latency_s']
            comparison['all_request_time_ratio'] = bootstrap_ratio(a, b, draws, speedup=True)
            comparison.update(identical_output_requests=len(matched_ids),
                different_output_requests=len(ids)-len(matched_ids),
                absolute_speedup_scope='AR/F identical-token-and-termination subset')
            comparison['absolute_speedup'] = (bootstrap_ratio(a[identical],b[identical],matched_draws,speedup=True)
                if matched_ids else dict(value=None,ci95=None,status='undefined'))
            if not identical.all():
                for metric in (*[key+'_reduction_pct' for key in TIMES], 'all_request_time_ratio'):
                    if comparison[metric]['status'] != 'undefined':
                        comparison[metric]['status'] = 'descriptive_different_outputs_no_identical_output_gain_claim'
            comparison['verification_call_reduction_pct'] = None
        else:
            a, b = arrays[candidate]['verification_calls'], arrays[baseline]['verification_calls']
            comparison['verification_call_reduction_pct'] = bootstrap_ratio(a, b, draws)
            differences = b - a
            comparison.update(wins=int(np.count_nonzero(differences > 0)), regressions=int(np.count_nonzero(differences < 0)),
                ties=int(np.count_nonzero(differences == 0)), gross_calls_saved=int(differences[differences > 0].sum()),
                gross_calls_added=int(-differences[differences < 0].sum()), net_calls_saved=int(differences.sum()))
        contrasts[candidate + '_vs_' + baseline] = comparison
    contrasts['C_CODE58_vs_AREPO_CODE58']['request_latency_s_reduction_pct']['status'] = 'prespecified_primary'
    output_agreement = dict(requests=len(ids), identical_outputs=int(identical.sum()),
        differing_outputs=int((~identical).sum()),
        differing_tokens=sum(not row['ar_spec_comparison']['token_ids_equal'] for row in pairs),
        differing_termination=sum(not row['ar_spec_comparison']['termination_equal'] for row in pairs),
        differing_lengths=sum(row['AR']['output_tokens']!=row['F']['output_tokens'] for row in pairs))
    return dict(requests=len(ids), totals=totals, contrasts=contrasts, paired=pairs, ar_spec_output_agreement=output_agreement,
        bootstrap=dict(draws=10000, generator='PCG64', seed=20261120, input_order=ids,
            input_order_digest=digest(ids), draw_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),
            percentiles='linear 2.5/97.5; shared whole-request draws across full-panel contrasts',
            identical_output_subset=dict(input_order=matched_ids,seed=20261120,
                draw_sha256=None if matched_draws is None else hashlib.sha256(matched_draws.astype('<i8').tobytes()).hexdigest())))


def bootstrap_ratio(a, b, draws, *, speedup=False):
    with np.errstate(divide='ignore', invalid='ignore'):
        point = b.sum() / a.sum() if speedup else 100 * (1 - a.sum() / b.sum())
        distribution = b[draws].sum(1) / a[draws].sum(1) if speedup else 100 * (1 - a[draws].sum(1) / b[draws].sum(1))
    if not math.isfinite(point) or not np.isfinite(distribution).all():
        return dict(value=None, ci95=None, status='undefined')
    return dict(value=float(point), ci95=np.quantile(distribution, [.025, .975], method='linear').tolist(),
                status='descriptive_unadjusted')


def compare_reductions(actual, expected, path='root'):
    """Check two implementations without imposing byte equality on FP64 sums."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Reducer keys differ: ' + path)
        for key in expected:
            compare_reductions(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'Reducer list differs: ' + path)
        for i, value in enumerate(expected):
            compare_reductions(actual[i], value, path + f'[{i}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
                'Independent numeric reduction differs: ' + path)
    else:
        require(actual == expected, 'Independent reduction differs: ' + path)


def work_accounting(state, *, plan=None):
    expected = {**{'production.'+k:v for k,v in PRODUCTION_WORK.items()},
                **{'qualification.'+k:v for k,v in QUALIFICATION_CEILINGS.items()}}
    if plan is not None and 'policy_resume' in plan:
        from .policy_resume import effective_work_limits
        expected = effective_work_limits(plan['root'])
    require(state['limits'] == expected, 'Work ceiling changed')
    pending = Counter()
    for operation in state['pending'].values():
        pending.update(operation['counts'])
    result = []
    for kind, limit in expected.items():
        attempted, completed = state['attempted'][kind], state['completed'][kind]
        require(type(attempted) is int and type(completed) is int
                and 0 <= completed <= attempted <= limit and attempted - completed == pending[kind],
                'Work accounting fails: ' + kind)
        result.append(dict(kind=kind, limit=limit, attempted=attempted, completed=completed,
            unknown=pending[kind], unattempted=(limit-attempted if kind.startswith('production.') else None),
            unused_allowance=(limit-attempted if kind.startswith('qualification.') else None)))
    return result


def stream_archive(archive, consume, *, zstd='zstd'):
    """Verify all exact members once, passing one member at a time to a reducer."""
    index = read(str(archive) + '.index.json')
    require(index['binding_digest'] == digest({k:v for k,v in index.items() if k != 'binding_digest'}),
            'Compact index binding differs')
    require(identity(archive) == dict(bytes=index['archive_bytes'], sha256=index['archive_sha256']),
            'Compact archive bytes differ')
    seen = set()
    process = subprocess.Popen([zstd, '-q', '-d', '-c', str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|', ignore_zeros=True) as stream:
            for member in stream:
                require(member.isfile() and member.name in index['members'] and member.name not in seen,
                        'Unknown, duplicate or nonregular archive member')
                expected = index['members'][member.name]
                payload = stream.extractfile(member).read()
                require(len(payload) == expected['bytes'] and hashlib.sha256(payload).hexdigest() == expected['sha256'],
                        'Compact member bytes differ: ' + member.name)
                consume(member.name, payload)
                seen.add(member.name)
        process.stdout.close()
        error = process.stderr.read().decode('utf-8', 'replace')
        require(process.wait() == 0, 'Archive decompression failed: ' + error)
    finally:
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    require(seen == set(index['members']), 'Missing compact members')
    return dict(arm=index['arm'], event=index['event'], members=len(seen),
                bytes=index['archive_bytes'], sha256=index['archive_sha256'])


def audit_evaluation(plan):
    root = Path(plan['root']) / 'evaluation/final'
    if not (root / 'STATUS.json').exists():
        return dict(status='not_started', rows=[], errors=[], planned=15600, unattempted=15600)
    status, protocol, inputs, references = (read(root/name) for name in
        ('STATUS.json', 'PROTOCOL.json', 'INPUTS.json', 'REFERENCES.json'))
    manifest = read(plan['manifest'])
    require(inputs == dict(panel=manifest['evaluation_requests'], warmup=manifest['warmup_requests']),
            'Final inputs differ from sealed manifest')
    require(protocol['execution_commit'] == plan['execution_commit'] and protocol['gpu'] == plan['devices']['A'],
            'Execution commit or physical GPU differs')
    require(protocol['schedule'] == planned_schedule(inputs['panel']), 'Final schedule changed')
    require(set(protocol['checkpoints']) == set(METHODS), 'Missing/extra endpoint')
    slots = planned_slots(inputs['panel'], inputs['warmup'])
    request_map = {r['request_id']:r for r in inputs['panel'] + inputs['warmup']}
    for role in ('panel', 'warmup'):
        require(set(references[role]) == {r['request_id'] for r in inputs[role]}, 'Incomplete AR/F canonical coverage')
        for request in inputs[role]:
            check_bundle(references[role][request['request_id']], request, plan['canonical_provenance'])
    rows = json_lines(root/'MEASUREMENTS.jsonl') if (root/'MEASUREMENTS.jsonl').exists() else []
    require(len(rows) <= len(slots), 'More executions than planned')
    errors, valid = [], []
    repeat_values = {}
    for index, row in enumerate(rows):
        slot = slots[index]
        try:
            require(all(row[key] == value for key,value in slot.items()), 'Execution order/source slot differs')
            request, method = request_map[slot['request_id']], slot['method']
            require(row['source_id'] == request['source_id'] and row['physical_gpu_uuid'] == plan['devices']['A'],
                    'Source identity or physical GPU differs')
            path = root/'raw'/(label(slot)+'.json')
            require(Path(row['raw_path']).resolve() == path.resolve(), 'Raw path does not match its slot')
            raw = read(path)
            bundle = references['panel' if slot['phase']=='scored' else 'warmup'][request['request_id']]
            metrics = raw_metrics(raw, request, bundle['AR' if method=='AR' else 'F'],
                                  method=method, checkpoint=protocol['checkpoints'][method], slot=slot)
            metrics['ar_spec_comparison'] = cross_path_comparison(bundle['AR'], bundle['F'])
            for key,value in metrics.items():
                require(row[key] == value, 'Saved measurement differs from raw evidence: '+key)
            require(row['checkpoint_id'] == protocol['checkpoints'][method], 'Measurement endpoint differs')
            require(row['prompt_tokens']==len(request['prompt_ids'])
                    and row['cap_hit'] is (metrics['finish_reason']=='length'), 'Prompt length/cap flag differs')
            require(row['valid_measurement'] is True and row['errors'] == [], 'Execution rejected this measurement')
            key = slot['phase'], method, request['request_id']
            counts = tuple(metrics[k] for k in (*COUNTS, 'finish_reason'))
            require(key not in repeat_values or repeat_values[key] == counts, 'Native repeat counts changed')
            repeat_values[key] = counts
            valid.append(row)
        except (KeyError, ValueError, TypeError, IndexError, OSError, json.JSONDecodeError) as error:
            errors.append(dict(index=index, slot=slot, error=str(error)))
    expected_files = {label(slot)+'.json' for slot in slots[:status['attempted']]}
    extras = {p.name for p in (root/'raw').glob('*.json')} - expected_files
    require(not extras, 'Unexpected raw executions outside the attempted schedule')
    require(status['planned'] == 15600 and status['returned'] == len(rows)
            and status['completed_valid'] + status['invalid'] == len(rows)
            and status['attempted'] == len(rows) + status['unknown']
            and status['unattempted'] == 15600-status['attempted'], 'Evaluation ledger does not reconcile')
    complete = status['status'] == 'completed' and len(rows) == 15600 and not errors and status['invalid'] == 0
    result = dict(status='passed' if complete else ('failed' if errors else 'incomplete'),
                  execution_status=status, rows=rows, errors=errors,
                  independently_validated=len(valid), planned=15600, unattempted=status['unattempted'])
    if complete:
        activations=json_lines(root/'ACTIVATIONS.jsonl')
        require(len(activations)==120, 'Incomplete activation ledger')
        for expected,actual in zip(planned_schedule(inputs['panel']),activations):
            require(all(actual[key]==value for key,value in expected.items()), 'Activation order differs')
        boundaries=json_lines(root/'BOUNDARIES.jsonl')
        require([row['activation'] for row in boundaries]==[*range(120),'finished'], 'Missing timing quiet boundary')
        require(all(row['both_cards_checked'] is True and row['physical_gpu_uuid']==plan['devices']['A']
                    and row['timing']['status']=='passed' and not row['timing']['transfers']
                    and not row['timing']['training'] and not row['timing']['foreign_gpu_processes']
                    for row in boundaries), 'Timing interference evidence differs')
        result['independent_summary'] = reduce_endpoint(valid)
    return result


def audit_precheck(plan):
    root=Path(plan['root'])
    destination=root/'precheck/AR_F_PRECHECK.json'
    if not destination.exists():
        return dict(status='not_completed',validated_generations=0)
    require(read(destination)['status']=='passed', 'The native AR/F precheck did not pass')
    manifest=read(plan['manifest'])
    canonical=read(root/'precheck/CANONICALS.json')
    require(set(canonical)=={r['request_id'] for r in manifest['evaluation_requests']}, 'Precheck panel coverage differs')
    count=0
    observations=[]
    for request in manifest['evaluation_requests']:
        card='A' if request['request_index']%2==0 else 'B'
        bundle=canonical[request['request_id']]
        observations.append(dict(request_id=request['request_id'],
            **check_bundle(bundle,request,plan['canonical_provenance'])))
        for method in ('AR','F'):
            name=f"{card}_{method}_precheck_{request['request_index']}"
            record=read(root/'precheck'/card/'records'/(name+'.json'))
            require(record['record_id']==name and record['session_id']==plan['session']
                    and record['request_index']==request['request_index'] and record['purpose']=='evaluation',
                    'Precheck execution identity differs')
            raw_metrics(record,request,bundle[method],method=method,
                        checkpoint=plan['target'] if method=='AR' else plan['draft'])
            require(record['metadata']['feedback_capture_in_timed_request'] is False, 'Precheck captured training feedback')
            count+=1
            require(bundle[method]['source_record_id']==name
                    and bundle[method]['source_checkpoint']==(plan['target'] if method=='AR' else plan['draft']),
                    'Canonical is not bound to its genuine producer')
    return dict(status='passed',validated_generations=count,canonical_requests=1024,
                separately_validated_AR_and_frozen_DFlash=True,
                identical_outputs=sum(row['output_equal'] for row in observations),
                differing_outputs=sum(not row['output_equal'] for row in observations),
                output_comparisons=observations)


def scalar_costs(value, prefix=''):
    """Expose measured fields separately; nested timers are never added together."""
    rows = []
    if isinstance(value, dict):
        for key,item in value.items():
            path = prefix + '.' + key if prefix else key
            if type(item) in (int, float) and (key.endswith('_s') or 'bytes' in key):
                require(math.isfinite(item) and item >= 0, 'Nonfinite/negative measured cost: '+path)
                rows.append(dict(metric=path, value=item))
            elif isinstance(item, dict):
                rows.extend(scalar_costs(item, path))
    return rows


def audit_arm_work(path):
    """Check chronology directly from before/after operation receipts."""
    started, completed = Counter(), Counter()
    pending = set()
    known = {'online_generations','correctness_generations','terminal_preparation_forwards',
             'response_forwards','response_backwards','optimizer_steps'}
    for row in json_lines(path):
        kind, ordinal, phase = row['kind'], row['ordinal'], row['phase']
        require(kind in known and type(ordinal) is int and ordinal > 0, 'Unknown native work receipt')
        if phase == 'completed':
            require((kind,ordinal) in pending and ordinal==completed[kind]+1, 'Work completion has no ordered start')
            pending.remove((kind,ordinal))
            completed[kind] += 1
            continue
        require(phase == 'started' and ordinal == started[kind]+1, 'Repeated or unordered work start')
        steps = completed['optimizer_steps']
        if kind == 'online_generations':
            require(steps == ((ordinal-1)//128)*128, 'Feedback used the wrong pre-update endpoint')
        elif kind == 'terminal_preparation_forwards':
            event = (ordinal-1)//128+1
            require(completed['online_generations'] == event*128 and steps == (event-1)*128,
                    'Teacher work precedes the complete own-feedback window')
        elif kind in ('response_forwards','response_backwards'):
            require(completed['terminal_preparation_forwards'] == ((ordinal-1)//256+1)*128,
                    'Training precedes complete teacher preparation')
            if kind == 'response_forwards':
                require(steps == (ordinal-1)//2, 'Response/optimizer chronology changed')
            else:
                require(completed['response_forwards'] == ordinal, 'Backward lacks its response forward')
        elif kind == 'optimizer_steps':
            require(completed['response_forwards'] == completed['response_backwards'] == 2*ordinal,
                    'Adam did not consume exactly two response presentations')
        else:
            require(completed['online_generations'] == 128 and steps == (0 if ordinal==1 else 128),
                    'Inherited correctness guard moved')
        started[kind] += 1
        pending.add((kind,ordinal))
    return dict(attempts={k:started[k] for k in sorted(known)},
                completed={k:completed[k] for k in sorted(known)},
                unknown={k:started[k]-completed[k] for k in sorted(known)})


def audit_event_training(receipt, *, arm, event, requests, feedback):
    expected = dict(event_id=f'event_{event:03d}',method=arm,dataset='OpenCodeInstruct',
        passes=2,response_presentations=256,optimizer_steps=128,start_step=(event-1)*128,
        end_step=event*128,reduction='response_mean',status='completed')
    require(all(receipt.get(k)==v for k,v in expected.items()), 'Training event counters/identity changed')
    features = {row['record_id']:row for row in receipt['features']}
    require(len(receipt['features'])==128 and set(features)==set(feedback) and len(receipt['records'])==256,
            'Training did not use its 128 own responses twice')
    request_map = {r['request_id']:r for r in requests}
    for key,feature in features.items():
        require(feature['prompt_length']==len(request_map[key]['prompt_ids'])
                and feature['response_tokens']==len(feedback[key]['response_ids']),
                'Prepared feature lengths differ from own response')
    exposures=[]
    for pass_index in (0,1):
        records=receipt['records'][128*pass_index:128*(pass_index+1)]
        require(sorted(r['record_id'] for r in records)==sorted(feedback), 'Missing/duplicate per-pass response')
        totals=Counter()
        for index,row in enumerate(records, start=128*pass_index):
            stepped=index%2==1
            require(row['pass']==pass_index and row['optimizer_stepped'] is stepped
                    and row['optimizer_step_after']==(event-1)*128+(index+1)//2
                    and row['learning_rate']==(3e-4 if stepped else None), 'Native optimizer boundary changed')
            require(len(row['sampled_anchors'])==1, 'Presentation must have one retained anchor draw')
            draw=row['sampled_anchors'][0]
            require(len(draw['positions'])==len(draw['keep_mask'])==1
                    and 0<len(draw['positions'][0])<=512
                    and len(draw['positions'][0])==len(draw['keep_mask'][0]), 'Malformed retained anchor draw')
            require(all(type(i) is int and i>=0 for i in draw['positions'][0])
                    and all(type(i) is bool for i in draw['keep_mask'][0]), 'Invalid anchor coordinates/mask')
            observation=row['response_observation']
            require(observation['record_id']==row['record_id'] and observation['pass_index']==pass_index
                    and observation['sampled_anchors']==row['sampled_anchors'], 'Observation lost its own anchor binding')
            require(all(math.isfinite(float(value)) for value in row['metrics'].values()), 'Nonfinite ordinary metrics')
            require(math.isfinite(observation['loss']), 'Nonfinite loss observation')
            for metric,value in observation['sum_metrics'].items():
                require(math.isfinite(float(value)), 'Nonfinite exposure metric')
                totals['sum.'+metric]+=value
            for metric,pair in observation['ratio_metrics'].items():
                require(len(pair)==2 and all(math.isfinite(float(v)) for v in pair), 'Nonfinite ratio observation')
                totals['numerator.'+metric]+=pair[0]
                totals['denominator.'+metric]+=pair[1]
            totals['active_anchors']+=sum(draw['keep_mask'][0])
            totals['presentations']+=1
        exposures.extend(dict(arm=arm,event=event,pass_index=pass_index,metric=key,value=value)
                         for key,value in sorted(totals.items()))
    return exposures


def audit_training(plan):
    root, inputs = Path(plan['root']), read(plan['manifest'])
    results, curves, costs, archives, exposures = {}, [], [], [], []
    for arm in ARMS:
        base = root/'campaign_001/arms'/arm
        if not (base/'STATUS.json').exists():
            results[arm] = dict(status='not_started', events_completed=0)
            continue
        status = read(base/'STATUS.json')
        require(status['arm'] == arm and status['binding'] == plan['binding'], 'Training arm binding differs')
        events = status['events_completed']
        actual_work=audit_arm_work(base/'WORK.jsonl')
        require(actual_work==status['actual_work'], 'Arm STATUS differs from independently counted work')
        require(0 <= events <= 58 and [r['event'] for r in status['event_costs']] == list(range(1, events+1)),
                'Completed event cost receipts are missing or duplicated')
        for item in status['event_costs']:
            costs.extend(dict(arm=arm,event=item['event'],**row) for row in scalar_costs(item['costs']))
        costs.extend(dict(arm=arm,event=None,**row) for row in scalar_costs({k:v for k,v in status.items() if k.endswith('_s')}))
        for event in range(1, events+1):
            archive = root/'compact'/arm/f'event_{event:03d}.tar.zst'
            feedback, resources, training_receipts = {}, [], []
            def consume(name, payload):
                if name.endswith('/feedback_buffer/state.json'):
                    value = json.loads(payload)
                    for record in value['records']:
                        require(record['record_id'] not in feedback, 'Duplicate archived feedback identity')
                        feedback[record['record_id']] = record
                elif name.endswith('/RESOURCE_COSTS.json'):
                    resources.append(json.loads(payload))
                elif name.endswith(f'/trainer/events/event_{event:03d}/RESULT.json'):
                    training_receipts.append(json.loads(payload))
            archives.append(stream_archive(archive,consume,zstd=plan['zstd']))
            requests = inputs['train_requests'][(event-1)*128:event*128]
            require(set(feedback) == {r['request_id'] for r in requests}, 'Feedback pool differs from sealed event inputs')
            expected_checkpoint = plan['draft'] if event == 1 else str(Path(
                read(base/'boundaries'/f'event_{event-1:03d}.json')['weights']).parent)
            require(len(training_receipts)==1, 'Completed event lacks its unique training receipt')
            exposures.extend(audit_event_training(training_receipts[0],arm=arm,event=event,
                requests=requests,feedback=feedback))
            aggregates = Counter()
            for request in requests:
                record = feedback[request['request_id']]
                require(record['request_index'] == request['request_index']
                        and record['session_id'] == 'opencode_train_'+arm and record['purpose']=='serving',
                        'Feedback source/order identity differs')
                capture=record['metadata']['capture']
                require(capture['collect_features'] is True and capture['native_capture_validated'] is True
                        and capture['covered_token_ids']==request['prompt_ids']+record['response_ids'][:-1]
                        and capture['target_argmax_ids']==record['response_ids']
                        and capture['label_binding_scope']=='native_prefill_and_committed_verification',
                        'Feedback features lost native canonical conditioning')
                metrics = raw_metrics(record,request,None,method=arm,checkpoint=expected_checkpoint)
                for key in (*COUNTS,*TIMES):
                    aggregates[key] += metrics[key]
            curves.append(dict(arm=arm,event=event,requests=128,
                pre_update_checkpoint=expected_checkpoint,**dict(aggregates)))
            for resource in resources:
                costs.extend(dict(arm=arm,event=event,**row) for row in scalar_costs(resource,'resources'))
        results[arm] = dict(status=status['status'],events_completed=events,
            actual_work=status.get('actual_work'), final_checkpoint=status.get('final_checkpoint'))
        if status['status'] == 'completed':
            require(events == 58 and read(base/'FINAL_STATE_CHECK.json')['status'] == 'passed',
                    'Completed training lacks its full validated endpoint')
            expected=dict(online_generations=7424,correctness_generations=4,terminal_preparation_forwards=7424,
                          response_forwards=14848,response_backwards=14848,optimizer_steps=7424)
            require(actual_work['attempts']==actual_work['completed']==expected
                    and not any(actual_work['unknown'].values()), 'Completed arm has incomplete work')
    return dict(arms=results, event_curves=curves, costs=costs, archives=archives, exposure=exposures,
                timer_scope='Nested cost fields overlap; do not sum arbitrary fields into a total')


def overhead_accounting(root):
    """Retain actual engine and activation timers without treating them as serving."""
    root = Path(root)
    rows, physical = [], []
    # A continuation moves phase administration into attempts/. Qualification
    # engines may also remain outside evaluation/engines; retain both histories.
    paths = sorted(set(root.glob('**/LIFECYCLE.json')) | set(root.glob('**/CLOSE.json')))
    for path in paths:
        for item in scalar_costs(read(path)):
            relative = path.relative_to(root)
            rows.append(dict(source=str(relative),archived=relative.parts[0]=='attempts',**item))
    activations=Path(root)/'evaluation/final/ACTIVATIONS.jsonl'
    if activations.exists():
        for row in json_lines(activations):
            for item in scalar_costs(row):
                rows.append(dict(source=f"activation_{row['activation']:03d}:{row['method']}",archived=False,**item))
    for path in sorted(Path(root).glob('**/PHYSICAL_WORK.json')):
        physical.append(dict(source=str(path.relative_to(root)),receipt=read(path)))
    phases=[]
    for path in sorted(root.glob('**/phases/*/STATUS.json')):
        value=read(path)
        relative=path.relative_to(root)
        phases.append(dict(source=str(relative),archived=relative.parts[0]=='attempts',
            phase=path.parent.name,status=value['status'],wall_s=value.get('wall_s'),
            started_epoch=value['started_epoch'],host_memory=value.get('host_memory')))
    return dict(costs=rows,physical_work=physical,phases=phases,
        overlap='Activation and phase wall times include nested engine loading/teardown. '
            'Current and archived source receipts are separate views, not additive totals; copied receipts must not be charged again. '
            'The cumulative CLOCK.json alone supplies occupied-card time, including historical failed attempts.')


def authenticate_sources(plan):
    manifest=read(plan['source_manifest'])
    require(manifest['commit']==plan['execution_commit'] and manifest['dirty'] is False, 'Execution source is not pinned/clean')
    root=Path(plan['source_root'])
    for name,pin in manifest['files'].items():
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Source member escapes its root')
        path=root/relative
        if pin.get('kind')=='symlink':
            require(path.is_symlink() and str(path.readlink())==pin['target'] and path.exists()
                    and path.resolve().is_relative_to(root.resolve()), 'Source link changed')
        else:
            require(not path.is_symlink() and identity(path)==pin, 'Imported source changed: '+name)
    for entry in plan['authenticated_files']:
        require(identity(entry['path'])==entry['expected_identity'], 'Read-only dependency changed: '+entry['path'])
    return dict(status='passed',execution_commit=manifest['commit'],source_files=len(manifest['files']),
        source_manifest_identity=identity(plan['source_manifest']),readonly_files=len(plan['authenticated_files']))


def audit_run(plan, output):
    """Audit only after model workers and timed serving have stopped."""
    root, output = Path(plan['root']), Path(output)
    started=time.perf_counter()
    result = dict(status='incomplete', execution_commit=plan['execution_commit'],
        scope='Independent CPU raw/metric reduction; shared project authorship, not independent GPU replication',
        errors=[], additional_model_work=0)
    try:
        require(plan['binding'] == digest({k:v for k,v in plan.items() if k!='binding'}), 'Plan binding changed')
        for path, expected in plan['input_identities'].items():
            require(identity(path) == expected, 'Sealed input artifact changed: '+path)
        result['input_identities'] = plan['input_identities']
        if (root/'CLOCK.json').exists():
            clock = read(root/'CLOCK.json')
            require(not clock['active'], 'Do not audit/compress while a model worker is active')
            result['clock'] = clock
        if 'policy_resume' in plan:
            from .policy_resume import _receipt as authenticated_policy_receipt
            result['historical_startup_retry'] = authenticated_policy_receipt(root,plan).get('historical_startup_retry')
        if 'budget_resume' in plan:
            from .budget_resume import parent_plan, effective_limits
            qualified = parent_plan(plan)
            continuation = read(plan['budget_resume']['receipt'])
            result['budget_continuation'] = dict(
                status='authenticated', qualified_execution_commit=qualified['execution_commit'],
                production_execution_commit=plan['execution_commit'],
                previous_limits=qualified['limits'], limits=effective_limits(plan),
                original_first_model_load_epoch=continuation['original_clock']['first_model_load_epoch'],
                reused_qualification_generations=continuation['reused_qualification_generations'],
                reused_precheck_generations=continuation['reused_precheck_generations'],
                additional_qualification_work=continuation['additional_qualification_work'])
        result['source'] = authenticate_sources(plan)
        result['precheck'] = audit_precheck(plan)
        result['evaluation'] = audit_evaluation(plan)
        result['training'] = audit_training(plan)
        result['overhead'] = overhead_accounting(root)
        if (root/'WORK.json').exists():
            work = read(root/'WORK.json')
            result['work'] = work_accounting(work, plan=plan)
            production_complete = all(r['completed'] == r['limit'] for r in result['work'] if r['kind'].startswith('production.'))
            if result['evaluation']['status'] == 'passed' and all(r['status']=='completed' for r in result['training']['arms'].values()):
                from .policy_resume import assert_historical_work_only, expected_qualification_totals
                require(production_complete, 'Full endpoint has incomplete production work')
                assert_historical_work_only(plan,work)
                totals=expected_qualification_totals(plan,dict(generation=64,adam=17,
                    response_forward=40,response_backward=40,teacher_preparation=16,training_initialization=3))
                for phase,values in totals.items():
                    require(all(work[phase]['qualification.'+kind]==number for kind,number in values.items()),
                            'Qualification totals do not match the authorized completed/unknown accounting')
                require(result['precheck']['status']=='passed'
                        and read(qualification_root(plan)/'QUALIFICATION.json')['status']=='passed',
                        'Full endpoint lacks passed qualification/precheck')
                exports = read(root/'phases/final_A/EXPORTS_AFTER.json')
                before=read(root/'phases/final_A/EXPORTS_BEFORE.json')
                require(exports['unchanged'] is True and exports['identities']==before, 'Final endpoint weights changed')
                protocol=read(root/'evaluation/final/PROTOCOL.json')
                for method in METHODS[:-1]:
                    require(identity(Path(protocol['checkpoints'][method])/'model.safetensors')==before[method],
                            'Endpoint no longer matches its final timing identity')
                result['endpoint_identities']=before
                result['status'] = 'passed'
        if result['evaluation']['errors']:
            result['status'] = 'failed'
    except (KeyError, ValueError, TypeError, IndexError, OSError, json.JSONDecodeError, tarfile.TarError) as error:
        result['errors'].append(str(error))
        result['status'] = 'failed'
    result['audit_wall_s']=time.perf_counter()-started
    result['audit_observed_epoch']=time.time()
    write(output/'RAW_AUDIT.json', result)
    return result
