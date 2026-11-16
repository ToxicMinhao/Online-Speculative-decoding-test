#!/usr/bin/env python3
"""Independent CPU-only reconstruction of fixed PR49 event-seven collectors.

This reader does not import the main reducer, any experimental helper or Torch.
It authenticates small JSON evidence and never reads model/feature tensors.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np

ARMS=('N','E','S')
CONTRASTS=(('E','N'),('S','N'),('E','S'))
SOURCE_COMMIT='db6e1221d8d1b234d9d07e246025d1afd5dc30d5'
BASE='/data/sources/minhao/runs/dflash_persistent_prefix_withholding_20261115/'


def identity(payload):
    return dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())


class Reader:
    def __init__(self,delivery):
        self.root=Path(delivery);self.errors=[];self.checks=0;self.inputs={}
        self.manifest=json.loads((self.root/'DELIVERY_MANIFEST.json').read_text())
    def check(self,condition,label):
        self.checks+=1
        if not condition:self.errors.append(label)
        return bool(condition)
    def resolve(self,path):
        text=str(path)
        if text.startswith(BASE):return self.root.parent/text.removeprefix(BASE)
        return Path(path)
    def read(self,path,pin=None):
        path=self.resolve(path);payload=path.read_bytes();actual=identity(payload)
        self.inputs[str(path)]=actual
        if path.is_relative_to(self.root):
            relative=str(path.relative_to(self.root))
            if relative!='DELIVERY_MANIFEST.json':
                self.check(self.manifest.get(relative)==actual,'delivery bytes '+relative)
        if pin is not None:self.check(pin==actual,'explicit identity '+str(path))
        return json.loads(payload)


def valid_tokens(value):
    return isinstance(value,list) and bool(value) and all(type(x) is int and x>=0 for x in value)


def count_record(reader,record,request,arm,export):
    """Derive V/P/A/M from native blocks, including the final clipped block."""
    tag=arm+'/'+request['request_id'];before=len(reader.errors)
    p,y,m=record['prompt_ids'],record['response_ids'],record['metadata'];c=m['capture'];blocks=m['trace']
    reader.check(record['record_id']==request['request_id'] and record['request_index']==request['request_index'],tag+': request identity/order')
    reader.check(record['session_id']==f'persistent_{arm}_event8' and record['checkpoint_id']==export and record['purpose']=='serving',tag+': immutable collector identity')
    reader.check(valid_tokens(p) and p==request['prompt_ids'] and valid_tokens(y),tag+': token arrays')
    stops=request['stop_token_ids']
    reader.check(record['stop_token_ids']==stops==[151645,151643] and record['max_new_tokens']==request['max_new_tokens']==2048,tag+': stop/cap contract')
    reader.check(len(p)+2048<=4096 and 0<len(y)<=2048 and not any(t in stops for t in y[:-1]),tag+': context/termination geometry')
    finish='eos' if y and y[-1] in stops else 'length'
    reader.check(record['finish_reason']==finish and (finish=='eos' or len(y)==2048),tag+': producer termination')
    reader.check(m['block_size']==c['block_size']==16 and m['context_limit']==4096 and m['runtime']=='pinned_SGLang_FA3',tag+': pinned runtime')
    reader.check(m['feedback_capture_in_timed_request'] is True and c['collect_features'] is True and c['native_capture_validated'] is True,tag+': native feature capture')
    reader.check(c['status']=='passed' and c['label_binding_scope']=='native_prefill_and_committed_verification',tag+': captured target binding')
    reader.check(c['prompt_ids']==p and c['response_ids']==y and c['target_argmax_ids']==y and c['covered_token_ids']==(p+y)[:-1] and c['covered_rows']==len(p)+len(y)-1,tag+': exact target coverage')
    reader.check(c['blocks']==blocks and reader.read(m['native_trace_path'])==c,tag+': independent raw native JSON')
    offset=0;native=matched=0;hist=Counter();clipped=0
    for number,b in enumerate(blocks):
        name=tag+'/block'+str(number);draft,target=b['draft_tokens'],b['target_argmax'];w,progress=b['verify_size'],b['committed_progress']
        integer_fields=('verify_size','committed_progress','output_start','anchor_token','raw_accepted_draft_tokens_before_stop','accepted_draft_tokens','native_accepted_draft_tokens')
        integers=all(type(b[k]) is int for k in integer_fields)
        reader.check(integers,name+': integer fields')
        geometry=integers and 1<=w<=16 and len(draft)+1==len(target)==w and 1<=progress<=w
        reader.check(geometry and all(type(t) is int and t>=0 for t in draft+target),name+': width/token geometry')
        if not geometry:continue
        a=next((i for i in range(len(draft)) if draft[i]!=target[i]),len(draft))
        reader.check(progress<=a+1 and b['committed_tokens']==(draft[:a]+[target[a]])[:progress]==target[:progress],name+': native rejection/bonus')
        reader.check(offset<len(y) and b['output_start']==offset and b['anchor_token']==y[offset],name+': block seed')
        reader.check(b['committed_tokens']==y[offset+1:offset+progress+1] and b['positions']==list(range(len(p)+offset,len(p)+offset+w)),name+': absolute committed positions')
        reader.check(b['raw_accepted_draft_tokens_before_stop']==a and b['accepted_draft_tokens']==min(a,progress) and b['native_accepted_draft_tokens']==progress-1,name+': clipping counters')
        reader.check(type(b['stop_clipped']) is bool and b['stop_clipped']==(progress<a+1),name+': clipping flag')
        reader.check(type(b['finished']) is bool and b['finished']==(number==len(blocks)-1),name+': req.finished native convention')
        offset+=progress;native+=b['native_accepted_draft_tokens'];matched+=b['accepted_draft_tokens'];hist[str(progress-1)]+=1;clipped+=int(progress<a+1)
    V=len(blocks)
    reader.check(offset==len(y)-1 and native==offset-V,tag+': complete V/P/A identity')
    reader.check(all(type(m[k]) is int for k in ('verification_calls','accepted_draft_tokens','committed_progress')) and m['verification_calls']==V and m['accepted_draft_tokens']==matched and m['committed_progress']==offset,tag+': stored metadata totals')
    if 'native_accepted_draft_tokens' in m:reader.check(m['native_accepted_draft_tokens']==native,tag+': optional native total')
    reader.check(m['committed_progress_per_call']==(offset/V if V else None) and math.isclose(m['calls_per_1000_output_tokens'],1000*V/len(y),rel_tol=1e-12),tag+': stored derived metrics')
    return dict(source_id=request['source_id'],request_id=request['request_id'],request_index=request['request_index'],
        native_source_id=request.get('native_source_id'),arm=arm,V=V,P=offset,A=native,M=matched,
        output_tokens=len(y),finish_reason=record['finish_reason'],cap_hit=finish=='length',zero_call=V==0,
        terminal_clipped_blocks=clipped,native_histogram=dict(hist),valid=len(reader.errors)==before),y


def compute(rows):
    """Only counts enter these prespecified pooled/bootstrap comparisons."""
    keys=sorted({(r['source_id'],r['request_id']) for r in rows})
    paired={k:{r['arm']:r for r in rows if (r['source_id'],r['request_id'])==k} for k in keys}
    totals={}
    for arm in ARMS:
        values=[paired[k][arm] for k in keys]
        totals[arm]={field:sum(r[field] for r in values) for field in ('V','P','A','M','output_tokens')}
        totals[arm].update(requests=len(values),eos=sum(r['finish_reason']=='eos' for r in values),cap=sum(r['cap_hit'] for r in values),zero_call=sum(r['zero_call'] for r in values))
        v=totals[arm]['V'];totals[arm]['pooled_native_acceptance']=totals[arm]['A']/v if v else None
        totals[arm]['progress_per_call']=totals[arm]['P']/v if v else None
    draws=np.random.Generator(np.random.PCG64(1708)).integers(0,len(keys),size=(10000,len(keys)),dtype=np.int64)
    draw_sha=hashlib.sha256(draws.astype('<i8',copy=False).tobytes(order='C')).hexdigest()
    key_sha=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest()
    def interval(point,values):
        finite=np.isfinite(values);return dict(point=point,ci95=np.quantile(values,[.025,.975],method='linear').tolist() if finite.all() else None,undefined_draws=int((~finite).sum()))
    contrasts={}
    for candidate,reference in CONTRASTS:
        a={arm:np.array([paired[k][arm]['A'] for k in keys],dtype=np.float64) for arm in (candidate,reference)}
        v={arm:np.array([paired[k][arm]['V'] for k in keys],dtype=np.float64) for arm in (candidate,reference)}
        with np.errstate(divide='ignore',invalid='ignore'):
            av={arm:a[arm][draws].sum(1)/v[arm][draws].sum(1) for arm in (candidate,reference)}
            absolute=av[candidate]-av[reference];relative=100*(av[candidate]/av[reference]-1)
            call=100*(v[reference][draws].sum(1)-v[candidate][draws].sum(1))/v[reference][draws].sum(1)
        x,y=totals[candidate]['pooled_native_acceptance'],totals[reference]['pooled_native_acceptance']
        differences=[paired[k][reference]['V']-paired[k][candidate]['V'] for k in keys]
        base=totals[reference]['V']
        contrasts[candidate+'_vs_'+reference]=dict(
            absolute_acceptance=interval(x-y if x is not None and y is not None else None,absolute),
            relative_acceptance_pct=interval(100*(x/y-1) if x is not None and y else None,relative),
            call_reduction_pct=interval(100*sum(differences)/base if base else None,call),
            net_calls_saved=sum(differences),gross_saved=sum(max(d,0) for d in differences),gross_added=sum(max(-d,0) for d in differences),
            fewer=sum(d>0 for d in differences),more=sum(d<0 for d in differences),tied=sum(d==0 for d in differences))
    table=[]
    for key in keys:
        row=dict(source_id=key[0],request_id=key[1])
        for arm in ARMS:
            row.update({arm+'_'+field:paired[key][arm][field] for field in ('V','P','A','M','output_tokens','finish_reason','cap_hit','zero_call')})
        for a,b in CONTRASTS:row[a+'_vs_'+b+'_calls_saved']=paired[key][b]['V']-paired[key][a]['V']
        table.append(row)
    return dict(totals=totals,contrasts=contrasts,bootstrap=dict(draws=10000,seed=1708,generator='PCG64',numpy_version=np.__version__,
        percentile_method='linear',sorted_identity_order='source_id then request_id',ordered_source_list_sha256=key_sha,
        draw_matrix_sha256=draw_sha,draw_hash_encoding='C-order little-endian int64 bytes')),table


def run(delivery,output):
    started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    r=Reader(delivery);rows=[];canonical={};summary=None;paired=[];chronology={}
    try:
        plan=r.read(r.root/'sealed/PLAN.json');source=r.read(r.root/'SOURCE.json')
        r.check(source['commit']==plan['execution_commit']==SOURCE_COMMIT and not source['dirty'],'clean collection execution identity')
        requests=r.read(plan['event8_inputs'],pin=plan['input_identities'][plan['event8_inputs']])['requests']
        r.check(len(requests)==128 and len({x['source_id'] for x in requests})==128 and len({x['request_id'] for x in requests})==128 and len({tuple(x['prompt_ids']) for x in requests})==128,'128 distinct sealed inputs')
        for arm in ARMS:
            base=r.root/'tasks'/(arm+'_collect');status=r.read(base/'STATUS.json');result=r.read(base/'RESULT.json');features=r.read(base/'FEATURES.json')
            training=r.read(r.root/'tasks'/(arm+'8')/'STATUS.json')
            end=status['started_epoch']+status['wall_s'];chronology[arm]=dict(collection_completed_epoch=end,own_training_started_epoch=training['started_epoch'],own_order_valid=end<=training['started_epoch'])
            r.check(end<=training['started_epoch'],arm+': own collection precedes own event8 update')
            export=plan['native_n7_export'] if arm=='N' else r.read(r.root/'tasks'/(arm+'7')/'RESULT.json')['endpoint']['export']
            r.check(status['status']==result['status']==features['status']=='completed' and status['execution_commit']==SOURCE_COMMIT,arm+': completed immutable worker')
            r.check(result['own_checkpoint']==features['own_checkpoint']==export and result['generated_responses']==128 and result['teacher_preparations']==128 and len(features['feature_paths'])==128,arm+': complete own feedback identity')
            paths=sorted((base/'records').glob('*.json'));r.check({p.stem for p in paths}=={q['request_id'] for q in requests} and len(paths)==128,arm+': exact record population')
            for request in requests:
                try:
                    record=r.read(base/'records'/(request['request_id']+'.json'));row,tokens=count_record(r,record,request,arm,export)
                    key=request['request_id'];signature=(tokens,record['finish_reason'],row['cap_hit'])
                    if key in canonical:r.check(signature==canonical[key],arm+'/'+key+': exact cross-arm response/termination')
                    else:canonical[key]=signature
                    rows.append(row)
                except (OSError,KeyError,TypeError,ValueError,IndexError) as error:r.check(False,arm+'/'+request['request_id']+': missing or malformed '+repr(error))
        r.check(len(rows)==384,'all384 count rows')
        if not r.errors:summary,paired=compute(rows)
    except (OSError,KeyError,TypeError,ValueError,IndexError) as error:r.check(False,'prerequisite failure '+repr(error))
    for name,values in [('INDEPENDENT_ROWS.csv',rows),('INDEPENDENT_PAIRED.csv',paired)]:
        if values:
            fields=[k for k in values[0] if k!='native_histogram']
            with (output/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:row[k] for k in fields} for row in values)
    report=dict(status='passed' if not r.errors else 'incomplete',source_commit=SOURCE_COMMIT,checks=r.checks,error_count=len(r.errors),errors=r.errors,
        identified_count_rows=len(rows),complete_pairs=len(paired),summary=summary,chronology=chronology,inputs=r.inputs,
        wall_s=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        new_gpu_work=0,model_imports=False,root_reducer_read=False,cpu_threads=1,latency_analysis=False)
    (output/'INDEPENDENT_RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','checks','error_count','identified_count_rows','complete_pairs','wall_s','peak_rss_bytes')}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delivery',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    report=run(a.delivery,a.output);raise SystemExit(0 if report['status']=='passed' else 1)
