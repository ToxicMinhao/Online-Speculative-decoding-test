"""Prespecified paired endpoint reduction; repeats are timings, not samples."""
from collections import defaultdict
import hashlib
import math

import numpy as np

from .contract import METHODS, EVALUATION_REQUESTS, digest
from .evaluation import COUNT_FIELDS

TIMES = ('request_latency_s','native_generate_s','cache_maintenance_s')
CONTRASTS = (('C_CODE58','AREPO_CODE58'),('C_CODE58','O_CODE58'),
    ('AREPO_CODE58','O_CODE58'),('O_CODE58','F'),('AREPO_CODE58','F'),
    ('C_CODE58','F'), *((method,'AR') for method in METHODS if method!='AR'))


def reduce_requests(rows, *, expected_requests=EVALUATION_REQUESTS):
    groups = defaultdict(list)
    for row in rows:
        if row['phase']=='warmup':
            continue
        if (row['phase']!='scored' or row['method'] not in METHODS
                or row['valid_measurement'] is not True or row['errors']):
            raise ValueError('Invalid scored evidence cannot enter gain calculations')
        groups[row['request_id'],row['method']].append(row)
    ids = sorted({key[0] for key in groups})
    if len(ids)!=expected_requests or len(groups)!=len(ids)*len(METHODS):
        raise ValueError('Incomplete five-arm request panel')
    for method in METHODS:
        endpoints={row['checkpoint_id'] for group in groups.values() for row in group if row['method']==method}
        if len(endpoints)!=1:
            raise ValueError('A method changed endpoint across requests')
    for request_id in ids:
        sources={row['source_id'] for method in METHODS for row in groups[request_id,method]}
        if len(sources)!=1:
            raise ValueError('Source identity differs across arms')
    result = []
    for request_id in ids:
        item = dict(request_id=request_id)
        for method in METHODS:
            records = sorted(groups[request_id,method], key=lambda row:row['repeat'])
            if [row['repeat'] for row in records]!=[0,1,2]:
                raise ValueError('Exactly three distinct timing repeats are required')
            first = records[0]
            for key in (*COUNT_FIELDS,'finish_reason','checkpoint_id','physical_gpu_uuid','source_id'):
                if any(row[key]!=first[key] for row in records):
                    raise ValueError('Repeat counts or identity changed: '+key)
            for key in TIMES:
                if any(type(row[key]) not in (int,float) or not math.isfinite(row[key])
                       or row[key]<0 or (key!='cache_maintenance_s' and row[key]==0) for row in records):
                    raise ValueError('Invalid latency: '+key)
            if method=='AR':
                if any(first[key] is not None for key in COUNT_FIELDS if key!='output_tokens'):
                    raise ValueError('AR speculative metrics must remain N/A')
            else:
                if any(type(first[key]) is not int or first[key]<0 for key in COUNT_FIELDS):
                    raise ValueError('Native counts must be nonnegative integers')
                calls, progress = first['verification_calls'],first['committed_progress']
                if (progress!=first['output_tokens']-1
                        or first['native_accepted_draft_tokens']!=progress-calls
                        or first['accepted_draft_tokens']>progress or (calls==0 and progress!=0)):
                    raise ValueError('Native count identities do not reconcile')
            item[method] = {key:first[key] for key in (*COUNT_FIELDS,'finish_reason')}
            item[method].update({key:float(np.mean([row[key] for row in records])) for key in TIMES})
            item[method]['repeat_times'] = {key:[row[key] for row in records] for key in TIMES}
        if len({(item[m]['output_tokens'],item[m]['finish_reason']) for m in METHODS})!=1:
            raise ValueError('Paired output length or termination differs')
        result.append(item)
    if len({row['physical_gpu_uuid'] for row in rows})!=1:
        raise ValueError('Final timing must use one fixed physical card')
    return result


def _interval(point, samples):
    if not np.isfinite(samples).all() or not math.isfinite(point):
        return dict(value=None,ci95=None,status='undefined')
    bounds = np.percentile(samples,[2.5,97.5],method='linear').tolist()
    return dict(value=float(point),ci95=bounds,status='descriptive_unadjusted')


def summarize(rows, *, expected_requests=EVALUATION_REQUESTS):
    paired = reduce_requests(rows, expected_requests=expected_requests)
    draws = np.random.Generator(np.random.PCG64(20261120)).integers(0,len(paired),size=(10000,len(paired)))
    values = {method:{key:np.asarray([row[method][key] for row in paired],dtype=np.float64)
                     for key in (*TIMES,*COUNT_FIELDS) if not(method=='AR' and key in COUNT_FIELDS and key!='output_tokens')}
              for method in METHODS}
    totals = {}
    for method, metrics in values.items():
        sums = {key:float(value.sum()) for key,value in metrics.items()}
        if method=='AR':
            sums.update({key:None for key in COUNT_FIELDS if key!='output_tokens'})
            sums.update(native_accepted_per_call=None,stop_clipped_accepted_per_call=None,committed_per_call=None)
        else:
            calls = sums['verification_calls']
            for name,key in (('native_accepted_per_call','native_accepted_draft_tokens'),
                             ('stop_clipped_accepted_per_call','accepted_draft_tokens'),
                             ('committed_per_call','committed_progress')):
                sums[name] = sums[key]/calls if calls else None
        sums['tokens_per_s'] = sums['output_tokens']/sums['request_latency_s']
        sums['repeat_totals'] = {key:[sum(row[method]['repeat_times'][key][r] for row in paired) for r in range(3)] for key in TIMES}
        sums['termination_counts'] = {kind:sum(row[method]['finish_reason']==kind for row in paired) for kind in ('eos','length')}
        sums['output_length'] = dict(min=int(metrics['output_tokens'].min()),max=int(metrics['output_tokens'].max()),
            mean=float(metrics['output_tokens'].mean()),median=float(np.median(metrics['output_tokens'])))
        totals[method] = sums
    contrasts = {}
    for candidate, baseline in CONTRASTS:
        current, reference = values[candidate], values[baseline]
        comparison = dict(candidate=candidate,baseline=baseline)
        for key in TIMES:
            a,b = current[key],reference[key]
            point = 100*(1-a.sum()/b.sum()) if b.sum() else float('nan')
            with np.errstate(divide='ignore',invalid='ignore'):
                samples = 100*(1-a[draws].sum(1)/b[draws].sum(1))
            comparison[key+'_reduction_pct'] = _interval(point,samples)
        if baseline=='AR':
            a,b = current['request_latency_s'],reference['request_latency_s']
            comparison['absolute_speedup'] = _interval(b.sum()/a.sum(),b[draws].sum(1)/a[draws].sum(1))
            comparison['verification_call_reduction_pct'] = None
        else:
            a,b = current['verification_calls'],reference['verification_calls']
            with np.errstate(divide='ignore',invalid='ignore'):
                point = 100*(1-a.sum()/b.sum()) if b.sum() else float('nan')
                samples = 100*(1-a[draws].sum(1)/b[draws].sum(1))
            comparison['verification_call_reduction_pct'] = _interval(point,samples)
            difference = b-a
            comparison.update(wins=int((difference>0).sum()),regressions=int((difference<0).sum()),
                ties=int((difference==0).sum()),gross_calls_saved=int(np.maximum(difference,0).sum()),
                gross_calls_added=int(np.maximum(-difference,0).sum()),net_calls_saved=int(difference.sum()))
        contrasts[candidate+'_vs_'+baseline] = comparison
    contrasts['C_CODE58_vs_AREPO_CODE58']['request_latency_s_reduction_pct']['status']='prespecified_primary'
    return dict(requests=len(paired),totals=totals,contrasts=contrasts,
        paired=paired,bootstrap=dict(draws=10000,generator='PCG64',seed=20261120,
            input_order=[row['request_id'] for row in paired],input_order_digest=digest([row['request_id'] for row in paired]),
            draw_sha256=hashlib.sha256(draws.astype('<i8',copy=False).tobytes()).hexdigest(),
            percentiles='linear 2.5/97.5; shared whole-request draws across every contrast'))
