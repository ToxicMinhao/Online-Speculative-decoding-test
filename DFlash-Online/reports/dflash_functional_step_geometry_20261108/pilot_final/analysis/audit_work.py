"""Audit all declared diagnostic work from JSON receipts, without loading models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

WINDOWS=(1,32,65,128)
LIMITS={'adam':0,'ordinary_forward':0,'ordinary_backward':0,
        'additional_forward':40,'additional_backward':4,'serving_generation':0}


def ledger_counts(rows):
    counts=Counter();pending=None
    for row in rows:
        kind,status=row['kind'],row['status']
        if kind not in LIMITS or status not in ('attempted','completed','failed'):
            raise ValueError('Unknown diagnostic work category/status')
        if status=='attempted':
            if pending is not None or counts[kind,status]>=LIMITS[kind]:
                raise ValueError('Overlapping or over-budget diagnostic attempt')
            pending=kind
        elif pending!=kind:
            raise ValueError('Unmatched diagnostic terminal record')
        else:
            pending=None
        counts[kind,status]+=1
    return counts


def audit(root,output,datasets=('gsm8k','mathqa')):
    root=Path(root);output=Path(output);files={};records=[]
    def read(path,lines=False):
        data=(root/path).read_bytes()
        files[str(path)]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        return [json.loads(line) for line in data.splitlines()] if lines else json.loads(data)
    for dataset in datasets:
        for event in (5,6):
            folder=Path(f'campaign_001/{dataset}/event_{event:03d}/FSG/diagnostics')
            observers=read(folder/'RESULTS.json')
            if [r['window'] for r in observers]!=list(WINDOWS):
                raise ValueError('Diagnostic observer coverage differs')
            for window,observer in zip(WINDOWS,observers):
                base=folder/f'window_{window:03d}'
                receipt=read(base/'RESULT.json');counts=ledger_counts(read(base/'MODEL_WORK.jsonl',True))
                if receipt['status']!='completed' or not receipt['restoration_exact'] or receipt['extra_adam_steps']!=0:
                    raise ValueError('Diagnostic status/state failure')
                for kind in LIMITS:
                    if counts[kind,'failed'] or counts[kind,'attempted']!=counts[kind,'completed']:
                        raise ValueError('Failed or unknown diagnostic work')
                recorded={f'{k}_{s}':v for (k,s),v in counts.items()}
                if recorded!=receipt['model_work']:
                    raise ValueError('Durable ledger and model-work receipt disagree')
                for kind,prefix in [('additional_forward','additional_response_forwards'),('additional_backward','additional_backwards')]:
                    for status in ('attempted','completed','failed'):
                        if receipt['actual_work'][prefix+'_'+status]!=counts[kind,status]:
                            raise ValueError('Actual callback work differs from ledger')
                if receipt['response_forwards']!=counts['additional_forward','attempted'] or receipt['backwards']!=counts['additional_backward','attempted']:
                    raise ValueError('Reported response totals differ')
                records.append(dict(dataset=dataset,event=event,window=window,status='passed',
                    forwards=counts['additional_forward','completed'],backwards=counts['additional_backward','completed'],
                    adam=counts['adam','attempted'],unknown=0,failed=0,
                    observer_wall_s=observer['observer_wall_s'],
                    physical_head_calls=receipt['physical_head_calls'],
                    checkpoint_proxy_calls=receipt['actual_work'].get('checkpoint_proxy_calls'),
                    costs=receipt['costs'],actual_work=receipt['actual_work']))
    totals={dataset:{'windows':sum(r['dataset']==dataset for r in records),
        **{k:sum(r[k] for r in records if r['dataset']==dataset) for k in
           ('forwards','backwards','adam','unknown','failed','observer_wall_s','physical_head_calls','checkpoint_proxy_calls')}}
        for dataset in datasets}
    result=dict(status='passed',windows=len(records),records=records,totals=totals,
                bounds=dict(windows=8*len(datasets),forwards=320*len(datasets),backwards=32*len(datasets)),
                new_model_work=0,input_files=files)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=audit(a.evidence,a.output);print(json.dumps({'status':r['status'],'totals':r['totals']}))
