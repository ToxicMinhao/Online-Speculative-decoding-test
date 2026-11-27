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


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def audit(plan,raw_root,external=None):
    """Bind arithmetic to the sealed inputs; leave raw-token validity separate."""
    root=Path(raw_root).resolve();original=Path(plan['root']);external=external or {}
    inputs={}
    def read(locator,pin=None):
        path=Path(locator)
        if str(path) in external:path=Path(external[str(path)])
        elif not path.is_absolute():path=root/path
        elif path.is_relative_to(original):path=root/path.relative_to(original)
        else:raise FileNotFoundError('Missing external evidence mapping: '+str(locator))
        data=path.read_bytes();identity=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        inputs[str(locator)]=identity
        if pin is not None and identity!=pin:raise ValueError('Input identity differs: '+str(locator))
        return data
    result=dict(status='running',plan_binding=plan['binding'],execution_commit=plan['execution_commit'],
        scope='Independent arithmetic only; canonical/native validity and live endpoint identities require companion audits',
        independence='Standard-library/NumPy reducer without production analysis imports; shared context and earlier serving implementation authorship disclosed',new_model_work=0)
    try:
        if plan['binding']!=digest({k:v for k,v in plan.items() if k!='binding'}):raise ValueError('Sealed plan binding differs')
        source=json.loads(read(plan['source_manifest'],plan['source_identity']))
        if source['commit']!=plan['execution_commit'] or source['clean'] is not True:raise ValueError('Source identity differs')
        sealed=json.loads(read(plan['evaluation_inputs'],plan['input_identities'][plan['evaluation_inputs']]))
        rows=[json.loads(line) for line in read('evaluation/MEASUREMENTS.jsonl').splitlines() if line.strip()]
        if len(rows)!=12384 or sum(row['phase']=='warmup' for row in rows)!=96:raise ValueError('Complete serving work required')
        if {row['request_id'] for row in rows if row['phase']=='scored'}!={row['request_id'] for row in sealed['evaluation_requests']}:
            raise ValueError('Scored request identities differ from sealed panel')
        if {row['request_id'] for row in rows if row['phase']=='warmup'}!={row['request_id'] for row in sealed['warmup_requests']}:
            raise ValueError('Warmup identities differ from sealed inputs')
        final=json.loads(read('FINAL_ANALYSIS.json'))
        if final['execution_commit']!=plan['execution_commit']:raise ValueError('Analysis execution source differs')
        reduced=reduce_independently(rows)
        expected={k:reduced[k] for k in ('requests','totals','contrasts','decision')}
        checks=compare(expected,final)
        if reduced['draw_sha256']!=final['bootstrap']['draw_sha256']:raise ValueError('Bootstrap draw identities differ')
        result.update(reduced,status='passed',checks=checks+1)
    except FileNotFoundError as error:result.update(status='incomplete',missing=str(error))
    except (KeyError,TypeError,ValueError,IndexError,OSError,ZeroDivisionError) as error:result.update(status='failed',error=repr(error))
    result['input_manifest_digest']=digest(inputs)
    return result,inputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--external-map',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    external=json.loads(args.external_map.read_text()) if args.external_map else None
    report,inputs=audit(plan,args.raw_root,external)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    with args.output.with_name(args.output.stem+'_PRIVATE_INPUTS.json').open('x') as stream:
        json.dump(inputs,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],checks=report.get('checks',0))))
    return 0 if report['status']=='passed' else 2

if __name__=='__main__':raise SystemExit(main())
