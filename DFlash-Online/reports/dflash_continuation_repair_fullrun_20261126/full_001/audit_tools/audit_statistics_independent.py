"""Independent CPU arithmetic for the fixed BASE/TAIL/AREPO/F full-run outputs.

No experimental reducer, bootstrap helper or model library is imported.
Shared context and earlier implementation authorship remain disclosed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ARMS=('BASE','TAIL','AREPO','F')
PAIRS=(('TAIL','BASE'),('BASE','AREPO'),('TAIL','AREPO'),('BASE','F'),('TAIL','F'))
TIMES=('request_latency_s','native_generate_s','cache_maintenance_s')
COUNTS=('output_tokens','verification_calls','accepted_draft_tokens','native_accepted_draft_tokens','committed_progress')


def reduce_independently(rows,n=1024):
    data=defaultdict(list)
    for r in rows:
        if not r['valid_measurement'] or r['errors']:raise ValueError('Invalid execution')
        if r['phase']=='scored':data[r['request_id'],r['method']].append(r)
    ids=sorted({key for key,_ in data})
    if len(ids)!=n or len(data)!=n*4:raise ValueError('Incomplete fixed endpoint panel')
    matrices={arm:{k:[] for k in (*TIMES,*COUNTS)} for arm in ARMS}
    for key in ids:
        for arm in ARMS:
            r=sorted(data[key,arm],key=lambda x:x['repeat'])
            if [x['repeat'] for x in r]!=[0,1,2]:raise ValueError('Repeated or missing timing repeat')
            for k in (*COUNTS,'finish_reason','checkpoint_id'):
                if len({x[k] for x in r})!=1:raise ValueError('Counts or endpoint changed')
            for k in TIMES:matrices[arm][k].append(math.fsum(x[k] for x in r)/3)
            for k in COUNTS:matrices[arm][k].append(r[0][k])
        if len({data[key,arm][0]['output_tokens'] for arm in ARMS})!=1:raise ValueError('Paired output differs')
    arrays={arm:{k:np.asarray(v,dtype=np.float64) for k,v in values.items()} for arm,values in matrices.items()}
    rng=np.random.Generator(np.random.PCG64(20261126))
    digest=hashlib.sha256();samples={pair:{k:[] for k in (*TIMES,'verification_calls')} for pair in PAIRS}
    # Draw one whole-request vector at a time; identical RNG stream, separate reduction layout.
    for _ in range(10000):
        selected=rng.integers(0,n,size=n);digest.update(selected.astype('<i8').tobytes())
        for candidate,baseline in PAIRS:
            for k in (*TIMES,'verification_calls'):
                ratio=arrays[candidate][k][selected].sum()/arrays[baseline][k][selected].sum()
                samples[candidate,baseline][k].append(100*(1-ratio) if k=='verification_calls' else 100*(ratio-1))
    contrasts={}
    for candidate,baseline in PAIRS:
        row={}
        for k in (*TIMES,'verification_calls'):
            ratio=math.fsum(matrices[candidate][k])/math.fsum(matrices[baseline][k])
            row['verification_call_reduction_pct' if k=='verification_calls' else k+'_degradation_pct']=dict(
                value=100*(1-ratio) if k=='verification_calls' else 100*(ratio-1),
                ci95=np.quantile(samples[candidate,baseline][k],[.025,.975],method='linear').tolist())
        differences=[b-c for b,c in zip(matrices[baseline]['verification_calls'],matrices[candidate]['verification_calls'])]
        row.update(wins=sum(d>0 for d in differences),regressions=sum(d<0 for d in differences),ties=sum(d==0 for d in differences),
            gross_calls_saved=sum(max(d,0) for d in differences),gross_calls_added=sum(max(-d,0) for d in differences),net_calls_saved=sum(differences))
        contrasts[candidate+'_vs_'+baseline]=row
    primary=contrasts['TAIL_vs_BASE']['request_latency_s_degradation_pct']
    return dict(requests=n,draw_sha256=digest.hexdigest(),contrasts=contrasts,
        totals={arm:{k:math.fsum(v) for k,v in values.items()} for arm,values in matrices.items()},
        decision=dict(selected='TAIL' if primary['value']<=.1 else 'BASE',practical_rule_met=primary['value']<=.1,
            degradation_bound_supported=primary['ci95'][1]<=.1,speed_improvement_supported=primary['ci95'][1]<0))


def compare(expected,actual,path='root'):
    if isinstance(expected,dict):
        return sum(compare(v,actual[k],path+'/'+k) for k,v in expected.items())
    if isinstance(expected,list):
        if len(expected)!=len(actual):raise ValueError(path+' length')
        return sum(compare(x,y,path) for x,y in zip(expected,actual))
    if isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not math.isclose(expected,actual,rel_tol=2e-12,abs_tol=1e-10):raise ValueError(path+' scalar')
    elif expected!=actual:raise ValueError(path+' value')
    return 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[json.loads(l) for l in (args.raw_root/'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
    if len(rows)!=12384 or sum(r['phase']=='warmup' for r in rows)!=96:raise ValueError('Complete serving work required')
    report=reduce_independently(rows)
    final=json.loads((args.raw_root/'FINAL_ANALYSIS.json').read_text())
    expected={k:report[k] for k in ('requests','totals','contrasts','decision')}
    checks=compare(expected,final)
    if report['draw_sha256']!=final['bootstrap']['draw_sha256']:raise ValueError('Bootstrap draw identities differ')
    report.update(status='passed',checks=checks+1,scope='Independent arithmetic only; canonical/native validity and provenance require companion audit',
        independence='Standard-library/NumPy reducer without production analysis imports; shared context/model family and earlier implementation authorship disclosed',new_model_work=0)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
