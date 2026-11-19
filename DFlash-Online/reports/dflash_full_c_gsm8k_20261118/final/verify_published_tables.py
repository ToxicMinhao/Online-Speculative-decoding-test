#!/usr/bin/env python3
"""Reproduce published scalar tables with Python and NumPy only.

Run from any directory:
    python verify_published_tables.py --output /tmp/C58-public-check.json

This is public scalar reproduction, no native-token or model replay. It cannot
verify hidden token arrays, checkpoint bytes, native traces, or training state.
No experiment modules, Torch, model weights, network calls or GPUs are used.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

METHODS = ('O58', 'KPLUS58', 'KPLUS_C58')
COUNTS = ('verification_calls', 'native_accepted_draft_tokens', 'accepted_draft_tokens',
          'committed_progress', 'output_tokens')
TIMES = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')
CONTRASTS = (('KPLUS_C58', 'KPLUS58'), ('KPLUS_C58', 'O58'), ('KPLUS58', 'O58'))
SEEDS = {'gsm8k':20261118, 'mathqa':20261119}
PANEL_SIZE, REPEATS, DRAWS = 1024, 3, 10000
SCALAR_TOLERANCE = 1e-12


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []
        self.max_numeric_error = 0.0

    def equal(self, actual, expected, label):
        self.count += 1
        if actual != expected:
            self.failures.append({'field':label, 'actual':actual, 'expected':expected})

    def number(self, actual, expected, label):
        self.count += 1
        if actual is None or expected is None:
            same = actual is expected
        else:
            error = abs(float(actual)-float(expected))
            self.max_numeric_error = max(self.max_numeric_error, error)
            same = math.isfinite(float(actual)) and math.isfinite(float(expected)) and math.isclose(
                float(actual), float(expected), rel_tol=SCALAR_TOLERANCE, abs_tol=SCALAR_TOLERANCE)
        if not same:
            self.failures.append({'field':label, 'actual':actual, 'expected':expected})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    content = path.read_bytes()
    return {'bytes':len(content), 'sha256':hashlib.sha256(content).hexdigest()}


def read_csv(path, *, repeat_table):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    integer_fields = ('panel_ordinal', *COUNTS, *(('repeat',) if repeat_table else ()))
    for row in rows:
        for name in integer_fields:
            row[name] = int(row[name])
        for name in TIMES:
            row[name] = float(row[name])
        if repeat_table:
            require(row['cap_hit'] in ('True', 'False'), 'Invalid cap flag')
            row['cap_hit'] = row['cap_hit'] == 'True'
    return rows


def reduce_repeats(rows, dataset, checks):
    require(len(rows) == PANEL_SIZE*len(METHODS)*REPEATS, 'Incomplete scored table: '+dataset)
    groups = defaultdict(list)
    identities = {}
    for row in rows:
        require(row['dataset'] == dataset and row['method'] in METHODS, 'Foreign dataset or arm')
        request = row['request_id']
        binding = (row['source_id'], row['panel_ordinal'])
        require(request not in identities or identities[request] == binding, 'Changed request identity')
        identities[request] = binding
        groups[request, row['method']].append(row)
    require(len(identities) == PANEL_SIZE, 'Missing or duplicated request identity')
    require(len({v[0] for v in identities.values()}) == PANEL_SIZE, 'Duplicated source identity')
    require(sorted(v[1] for v in identities.values()) == list(range(PANEL_SIZE)), 'Panel ordinal coverage differs')
    paired = []
    for request in sorted(identities):
        source, ordinal = identities[request]
        item = {'request_id':request, 'source_id':source, 'panel_ordinal':ordinal}
        for method in METHODS:
            repeats = sorted(groups[request, method], key=lambda row:row['repeat'])
            require([row['repeat'] for row in repeats] == list(range(REPEATS)), 'Missing or duplicate repeat')
            first = repeats[0]
            for row in repeats:
                for field in COUNTS:
                    require(row[field] >= 0, 'Negative count')
                    checks.equal(row[field], first[field], f'{dataset}.{request}.{method}.repeat.{field}')
                checks.equal(row['finish_reason'], first['finish_reason'], 'repeat termination')
                checks.equal(row['cap_hit'], first['cap_hit'], 'repeat cap flag')
                for field in TIMES:
                    require(math.isfinite(row[field]) and row[field] >= 0, 'Invalid scalar latency')
                    require(field == 'cache_maintenance_s' or row[field] > 0, 'Nonpositive request latency')
            calls, native, clipped, progress, output = (first[field] for field in COUNTS)
            checks.equal(native, progress-calls, 'native acceptance accounting')
            checks.equal(progress, output-1, 'output progress accounting')
            require(clipped <= progress, 'Clipped acceptance exceeds progress')
            require(calls != 0 or progress == 0, 'Zero-call nonempty progress')
            require(first['finish_reason'] in ('eos', 'length'), 'Invalid termination')
            checks.equal(first['cap_hit'], first['finish_reason'] == 'length', 'termination cap agreement')
            require(not first['cap_hit'] or output == 2048, 'Capped output length differs')
            item[method] = {field:first[field] for field in COUNTS}
            item[method].update({field:float(np.mean([row[field] for row in repeats])) for field in TIMES})
            item[method]['repeat_times'] = [row['request_latency_s'] for row in repeats]
            item[method]['finish_reason'] = first['finish_reason']
            item[method]['cap_hit'] = first['cap_hit']
        require(len({(item[m]['output_tokens'], item[m]['finish_reason']) for m in METHODS}) == 1,
                'Paired scalar length/termination differs')
        paired.append(item)
    return paired


def safe_ratio(a, b):
    return float(a/b) if b > 0 else None


def primary_metrics(paired, draws):
    def values(method, field):
        return np.asarray([row[method][field] for row in paired], dtype=np.float64)
    c = {field:values('KPLUS_C58', field) for field in COUNTS+TIMES}
    k = {field:values('KPLUS58', field) for field in COUNTS+TIMES}
    cv, kv = c['verification_calls'], k['verification_calls']
    ca, ka = c['native_accepted_draft_tokens'], k['native_accepted_draft_tokens']
    ct, kt = c['request_latency_s'], k['request_latency_s']
    c_accept = safe_ratio(ca.sum(), cv.sum())
    k_accept = safe_ratio(ka.sum(), kv.sum())
    with np.errstate(divide='ignore', invalid='ignore'):
        c_draw_accept = ca[draws].sum(axis=1)/cv[draws].sum(axis=1)
        k_draw_accept = ka[draws].sum(axis=1)/kv[draws].sum(axis=1)
        sampled = {
            'time_reduction':1-ct[draws].sum(axis=1)/kt[draws].sum(axis=1),
            'call_reduction':1-cv[draws].sum(axis=1)/kv[draws].sum(axis=1),
            'acceptance_improvement':c_draw_accept/k_draw_accept-1,
            'acceptance_difference':c_draw_accept-k_draw_accept,
        }
    points = {
        'time_reduction':float(1-ct.sum()/kt.sum()),
        'call_reduction':None if kv.sum() == 0 else float(1-cv.sum()/kv.sum()),
        'acceptance_improvement':None if c_accept is None or k_accept in (None, 0) else c_accept/k_accept-1,
        'acceptance_difference':None if c_accept is None or k_accept is None else c_accept-k_accept,
    }
    return {name:{'point':points[name],
            'ci95':np.quantile(values, [.025,.975], method='linear').tolist() if np.isfinite(values).all() else None,
            'undefined_draws':int((~np.isfinite(values)).sum())} for name,values in sampled.items()}


def verify_dataset(base, dataset, checks):
    folder = base/dataset
    names = ('ALL_REPEATS.csv', 'PER_REQUEST.csv', 'SUMMARY.json')
    inputs = {dataset+'/'+name:identity(folder/name) for name in names}
    rows = read_csv(folder/'ALL_REPEATS.csv', repeat_table=True)
    requests = read_csv(folder/'PER_REQUEST.csv', repeat_table=False)
    summary = json.loads((folder/'SUMMARY.json').read_text())
    checks.equal(summary['status'], 'completed', dataset+'.status')
    checks.equal(summary['request_count'], PANEL_SIZE, dataset+'.request_count')
    checks.equal(summary['count_trajectory_repeat'], 0, dataset+'.count_repeat')
    checks.equal(summary['primary_comparison'], 'KPLUS_C58_vs_KPLUS58', dataset+'.primary')
    paired = reduce_repeats(rows, dataset, checks)
    indexed = {(row['request_id'], row['method']):row for row in requests}
    require(len(indexed) == len(requests) == PANEL_SIZE*len(METHODS), 'Incomplete per-request table')
    for row in paired:
        for method in METHODS:
            saved = indexed[row['request_id'], method]
            for key in ('source_id', 'request_id', 'panel_ordinal'):
                checks.equal(row[key], saved[key], dataset+'.per_request.'+key)
            for field in COUNTS+TIMES:
                compare = checks.equal if field in COUNTS else checks.number
                compare(row[method][field], saved[field], dataset+'.per_request.'+field)
    totals = {}
    for method in METHODS:
        values = {field:sum(row[method][field] for row in paired) for field in COUNTS+TIMES}
        totals[method] = values
        for field,value in values.items():
            compare = checks.equal if field in COUNTS else checks.number
            compare(value, summary['method_totals'][method][field], dataset+'.total.'+method+'.'+field)
        for field, numerator, denominator in (
            ('native_accepted_tokens_per_call',values['native_accepted_draft_tokens'],values['verification_calls']),
            ('matched_accepted_tokens_per_call',values['accepted_draft_tokens'],values['verification_calls']),
            ('committed_progress_per_call',values['committed_progress'],values['verification_calls']),
            ('calls_per_1000_output_tokens',1000*values['verification_calls'],values['output_tokens'])):
            checks.number(safe_ratio(numerator,denominator),summary['method_totals'][method][field],dataset+'.ratio.'+field)
        for repeat in range(REPEATS):
            selected = [row for row in rows if row['method'] == method and row['repeat'] == repeat]
            for field in COUNTS+TIMES:
                compare = checks.equal if field in COUNTS else checks.number
                compare(sum(row[field] for row in selected), summary['repeat_totals'][str(repeat)][method][field],
                              dataset+'.repeat_total.'+method+'.'+str(repeat)+'.'+field)
    contrasts = {}
    for candidate, reference in CONTRASTS:
        name = candidate+'_vs_'+reference
        effect = summary['comparisons'][name]
        saved = [row[reference]['verification_calls']-row[candidate]['verification_calls'] for row in paired]
        accounting = {
            'requests_fewer_calls':sum(value > 0 for value in saved),
            'requests_more_calls':sum(value < 0 for value in saved),
            'request_call_ties':sum(value == 0 for value in saved),
            'gross_calls_saved':sum(max(value,0) for value in saved),
            'gross_calls_added':sum(max(-value,0) for value in saved),
            'net_calls_saved':sum(saved),
        }
        for field,value in accounting.items():
            checks.equal(value,effect[field],dataset+'.'+name+'.'+field)
        repeat_changes = [1-sum(row[candidate]['repeat_times'][i] for row in paired)
                            /sum(row[reference]['repeat_times'][i] for row in paired) for i in range(REPEATS)]
        for i,value in enumerate(repeat_changes):
            checks.number(value,effect['repeat_time_reductions'][i],dataset+'.'+name+'.repeat_time.'+str(i))
        contrasts[name] = dict(accounting, repeat_time_reductions=repeat_changes)
    source_order = [{'source_id':row['source_id'],'request_id':row['request_id']} for row in paired]
    order_hash = hashlib.sha256(json.dumps(source_order,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    bootstrap = summary['bootstrap']
    for field,value in {'seed':SEEDS[dataset],'draws':DRAWS,'generator':'PCG64','method':'linear',
                        'order':'lexicographic request_id','source_order_sha256':order_hash}.items():
        checks.equal(value,bootstrap[field],dataset+'.bootstrap.'+field)
    if 'source_order' in bootstrap:
        checks.equal(source_order,bootstrap['source_order'],dataset+'.bootstrap.source_order')
    draws = np.random.Generator(np.random.PCG64(SEEDS[dataset])).integers(0,PANEL_SIZE,size=(DRAWS,PANEL_SIZE))
    draw_hash = hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    checks.equal(draw_hash,bootstrap['draw_matrix_sha256'],dataset+'.bootstrap.draw_matrix_sha256')
    metrics = primary_metrics(paired,draws)
    reported = summary['comparisons']['KPLUS_C58_vs_KPLUS58']
    for name,value in metrics.items():
        checks.number(value['point'],reported[name]['point'],dataset+'.primary.'+name+'.point')
        checks.equal(value['undefined_draws'],reported[name]['undefined_draws'],dataset+'.primary.'+name+'.undefined')
        if value['ci95'] is None or reported[name]['ci95'] is None:
            checks.equal(value['ci95'],reported[name]['ci95'],dataset+'.primary.'+name+'.interval')
        else:
            for i in (0,1):
                checks.number(value['ci95'][i],reported[name]['ci95'][i],dataset+'.primary.'+name+'.ci95.'+str(i))
    return {'inputs':inputs,'scored_rows':len(rows),'requests':len(paired),'methods':totals,
            'contrasts':contrasts,'primary_metrics':metrics,'source_order_sha256':order_hash,
            'draw_matrix_sha256':draw_hash,'execution_commit':summary.get('execution_commit'),
            'bootstrap_seed':SEEDS[dataset],'bootstrap_draws':DRAWS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    output = args.output or args.root/'PUBLIC_TABLE_REPRODUCTION.json'
    require(not output.exists(), 'Choose a new output path; preserve the previous verification receipt')
    checks = Checks()
    result = {'status':'running','scope':'public scalar reproduction, no native-token or model replay',
        'dependencies':{'python':platform.python_version(),'numpy':np.__version__},
        'verifier':identity(Path(__file__)), 'scalar_comparison_tolerance':SCALAR_TOLERANCE,
        'integer_counts_and_bootstrap_identities_exact':True,
        'count_reduction':'one repeat-consistent trajectory per request',
        'time_reduction':'sum of per-request means over all three repeats',
        'bootstrap':'whole paired requests, lexicographic request_id, shared draws across four primary metrics',
        'datasets':{},'limitations':['Does not authenticate model, checkpoint, prompt or output-token bytes.',
            'Does not rerun or independently validate private native traces or training state.',
            'Only the primary C/K contrast receives bootstrap interval reproduction here; all three contrasts receive scalar call accounting.',
            'One fitted seed17 trajectory and reused panels; repeats are timing repeats, not independent training samples.']}
    try:
        for dataset in SEEDS:
            result['datasets'][dataset] = verify_dataset(args.root,dataset,checks)
        result['status'] = 'passed' if not checks.failures else 'failed'
    except Exception as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
    result.update(checks=checks.count, failures=checks.failures,
                  maximum_scalar_absolute_difference=checks.max_numeric_error)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'checks':checks.count,'output':str(output)}))
    return 0 if result['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
