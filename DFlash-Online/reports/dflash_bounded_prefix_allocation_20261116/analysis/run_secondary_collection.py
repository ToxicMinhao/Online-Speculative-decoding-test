"""Authenticate and reduce already collected C7/N7/S7 trajectories; CPU only.

Does not read model tensors, access A800, generate responses, or rank collection
latency. A failed secondary provenance gate cannot change the primary decision.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def discover_source(checkout=None, *, start=None):
    """Find the same analysis checkout from a task or published report path."""
    if checkout is not None:
        bases=[Path(checkout).resolve()]
    else:
        here=Path(start).resolve() if start is not None else Path(__file__).resolve().parent
        bases=[here,*here.parents]
    for base in bases:
        for candidate in (base,base/'DFlash-Online',base/'implementation_checkout/DFlash-Online'):
            if (candidate/'experiments').is_dir() and (candidate/'online_feedback').is_dir():
                return candidate.resolve()
    raise FileNotFoundError('Cannot locate DFlash-Online analysis source; supply --checkout')


_bootstrap_parser=argparse.ArgumentParser(add_help=False)
_bootstrap_parser.add_argument('--checkout')
_bootstrap_args,_=_bootstrap_parser.parse_known_args() if __name__=='__main__' else (_bootstrap_parser.parse_args([]),[])
SRC=discover_source(_bootstrap_args.checkout)
sys.path.insert(0,str(SRC))
from experiments.bounded_prefix_allocation.collection_analysis import analyze
from experiments.pr49_collection_cpu import provenance as historical


def identity(path):
    path=Path(path)
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(bytes=path.stat().st_size,sha256=digest)


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


class Evidence:
    """Read the small JSON receipt chain; never interpret a tensor payload."""
    def __init__(self,root):
        self.root=Path(root).resolve();self.inputs=[];self.errors=[];self.checks=0
        self.manifest=json.loads((self.root/'RETRIEVAL_MANIFEST.json').read_text())
        self.inputs.append(dict(local_path=str(self.root/'RETRIEVAL_MANIFEST.json'),identity=identity(self.root/'RETRIEVAL_MANIFEST.json'),role='transport_manifest'))
        self.remote_root=None

    def require(self,condition,label):
        self.checks+=1
        if not condition:self.errors.append(label)

    def read(self,relative):
        relative=str(relative);path=self.root/relative
        self.require(path.resolve().is_relative_to(self.root),'inside_delivery:'+relative)
        self.require(path.suffix=='.json','compact_json_only:'+relative)
        actual=identity(path);self.require(actual==self.manifest.get(relative),'manifest:'+relative)
        self.inputs.append(dict(local_path=str(path),identity=actual,role='new_delivery',relative_path=relative))
        return json.loads(path.read_text())

    def remote(self,path):
        return self.read(Path(path).relative_to(self.remote_root))

    def prior(self,path,expected):
        path=Path(path);actual=identity(path)
        self.require(actual==expected,'retained_source:'+str(path))
        self.inputs.append(dict(local_path=str(path),identity=actual,role='rechecked_old_provenance'))
        return json.loads(path.read_text())


def candidate_provenance(evidence,old,checkout):
    """Bind the C7 export and collection to saved source/state receipts."""
    e=evidence;plan=e.read('sealed/PLAN.json');e.remote_root=Path(plan['root'])
    launch=e.read('LAUNCH.json')
    e.require(launch['status'] in ('completed','stopped'),'all_native_tasks_finished_before_analysis')
    source=e.remote(plan['source_manifest']);preflight=e.read('CPU_PREFLIGHT.json')
    e.require(source['commit']==plan['execution_commit'] and source['dirty'] is False,'published_clean_source')
    e.require(preflight['status']=='passed' and preflight['execution_commit']==plan['execution_commit'],'new_cpu_preflight')
    source_names=('experiments/bounded_prefix_allocation/task_runner.py',
        'experiments/bounded_prefix_allocation/training.py','experiments/persistent_prefix_withholding/worker.py',
        'online_feedback/sglang_worker.py','online_feedback/sglang_backend.py','online_feedback/native.py')
    for name in source_names:
        payload=subprocess.check_output(['git','show',plan['execution_commit']+':DFlash-Online/'+name],cwd=checkout)
        observed=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        e.require(observed==source['files'][name],'executed_git_source:'+name)
        e.inputs.append(dict(git_object=plan['execution_commit']+':DFlash-Online/'+name,identity=observed,role='published_source'))
    runtime=old['runtime']
    e.require(plan['target']==runtime['target'] and plan['runtime_contract']==runtime['runtime_contract'],'same_pinned_runtime_and_target')
    for key,expected in plan['runtime_contract']['native_runtime_sha256'].items():
        e.require(preflight['runtime']['installed_sources'][key]['sha256']==expected,'installed_runtime:'+key)
    for path,pin in runtime['inherited_tokenizer_identities'].items():
        e.require(plan['input_identities'][path]==pin==preflight['inputs'][path],'target_tokenizer:'+path)
    selected=e.remote(plan['event8_inputs']);requests=selected['requests']
    e.require(identity(e.root/Path(plan['event8_inputs']).relative_to(e.remote_root))==plan['input_identities'][plan['event8_inputs']],'sealed_event8_identity')
    e.require(len(requests)==128 and len({r['request_id'] for r in requests})==128,'fixed128_collection_requests')
    prior_map={(r['arm'],r['request_id']):r for r in old['request_mappings']}
    e.require({r['request_id'] for r in requests}=={rid for arm,rid in prior_map if arm=='N'},'same128_retained_population')
    for ordinal,request in enumerate(requests):
        oldrow=prior_map['N',request['request_id']]
        e.require(oldrow['ordinal']==ordinal and all(request[k]==oldrow[k] for k in ('request_id','source_id','request_index')),'same_ordered_source:'+request['request_id'])
    c7=e.read('tasks/C7/RESULT.json');boundary=e.read('tasks/C7/BOUNDARY.json');resume=e.read('tasks/C7/trainer/RESUME_CHECK.json')
    e.require(c7['status']=='completed' and c7['arm']=='C' and c7['event']==7 and c7['execution_commit']==plan['execution_commit'] and c7['binding']==plan['binding'],'C7_complete_bound_endpoint')
    endpoint=c7['endpoint'];expected_base=e.remote_root/'tasks/C7/trainer'
    e.require(endpoint['state']==str(expected_base/'training_state.pt') and endpoint['export']==str(expected_base/'events/event_007/export') and endpoint['global_step']==896 and endpoint['presentations']==1792,'C7_own_paths_counters')
    for field in ('weights_identity','state_identity'):
        pin=endpoint[field]
        e.require(type(pin['bytes']) is int and pin['bytes']>0 and len(pin['sha256'])==64 and all(c in '0123456789abcdef' for c in pin['sha256']),'C7_'+field)
    e.require(c7['boundary']==boundary and boundary['status']=='passed' and boundary['export_exact'] is True and boundary['global_step']==896 and boundary['live']['exact_resume'] is True and not boundary['live']['mismatches'],'C7_exact_state_export_boundary')
    e.require(resume['status']=='passed' and resume['exact_resume'] is True and not resume['mismatches'] and resume['global_step']==768 and resume['origin_state']==plan['origin_state'] and resume['origin_export']==plan['origin_export'],'C7_exact_K6_origin')
    result=e.read('tasks/C_collect/RESULT.json');features=e.read('tasks/C_collect/FEATURES.json')
    expected_features=[str(e.remote_root/'tasks/C_collect/loop/features'/f"{r['request_id']}.pt") for r in requests]
    e.require(result==features and result['status']=='completed' and result['arm']=='C' and result['binding']==plan['binding'] and result['execution_commit']==plan['execution_commit'],'C_collection_complete_binding')
    e.require(result['own_checkpoint']==endpoint['export'] and result['own_weights_identity']==endpoint['weights_identity'] and result['generated_responses']==128 and result['teacher_preparations']==128 and result['feature_paths']==expected_features,'C_collection_own_C7_weights_all128')
    engine=e.read('tasks/C_collect/native/engine_settings.json');ready=e.read('tasks/C_collect/native/ready.json')
    e.require(all(engine[k]==v for k,v in runtime['engine'].items()) and engine['speculative_draft_model_path']==endpoint['export'],'C_native_engine_configuration')
    e.require(ready['status']['status']=='passed' and ready['status']['publication'] is None,'C_fresh_native_worker')
    statuses={name:e.read('tasks/'+name+'/STATUS.json') for name in ('C7','C_collect','C8')}
    for name,row in statuses.items():
        e.require(row['status']=='completed' and row['execution_commit']==plan['execution_commit'] and row['binding']==plan['binding'],'task_completion:'+name)
    e.require(statuses['C7']['started_epoch']+statuses['C7']['wall_s']<=statuses['C_collect']['started_epoch'],'C7_before_collection')
    e.require(statuses['C_collect']['started_epoch']+statuses['C_collect']['wall_s']<=statuses['C8']['started_epoch'],'collection_before_own_C8')
    chronology=dict(C=dict(collection_start=statuses['C_collect']['started_epoch'],collection_finish=statuses['C_collect']['started_epoch']+statuses['C_collect']['wall_s'],own_event8_training_start=statuses['C8']['started_epoch'],own_collection_precedes_training=True),**{a:old['chronology'][a] for a in ('N','S')})
    checkpoints={a:old['checkpoints'][a] for a in ('N','S')};checkpoints['C']=endpoint
    return plan,requests,checkpoints,chronology,prior_map


def run(delivery,old_delivery,qualification_delivery,restart_delivery,output,checkout=SRC):
    began=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='INVALID_OR_INCOMPLETE_SECONDARY',latency_excluded=True,primary_endpoint_decision_affected=False,new_model_work=0)
    e=None
    try:
        # Re-execute the old receipt/source authentication, not the old metrics.
        old=historical.run(old_delivery,qualification_delivery,restart_delivery,output/'old_provenance_recheck')
        if old['status']!='passed':raise ValueError('Retained N/S provenance did not reauthenticate')
        e=Evidence(delivery)
        plan,requests,checkpoints,chronology,mapping=candidate_provenance(e,old,checkout)
        prior_inputs={row['local_path']:row['identity'] for row in old['consumed_inputs']}
        records={a:[] for a in ('N','C','S')};captures={a:{} for a in records}
        for arm in records:
            ids={p.stem for p in ((Path(old_delivery)/f'tasks/{arm}_collect/records') if arm!='C' else (Path(delivery)/'tasks/C_collect/records')).glob('*.json')}
            e.require(ids=={r['request_id'] for r in requests},'no_missing_or_extra_records:'+arm)
            for ordinal,request in enumerate(requests):
                rid=request['request_id']
                if arm=='C':
                    record=e.read(f'tasks/C_collect/records/{rid}.json')
                    trace_path=Path(record['metadata']['native_trace_path'])
                    e.require(trace_path==e.remote_root/f'tasks/C_collect/native/request_{ordinal+1:06d}.json','C_native_serial:'+rid)
                    captured=e.remote(trace_path)
                else:
                    row=mapping[arm,rid]
                    record=e.prior(row['record_path'],prior_inputs[row['record_path']])
                    captured=e.prior(row['native_path'],prior_inputs[row['native_path']])
                records[arm].append(record);captures[arm][rid]=captured
        proof=dict(status='passed' if not e.errors else 'incomplete',checkpoints=checkpoints,chronology=chronology,
            checks=e.checks,errors=e.errors,consumed_inputs=e.inputs,old_provenance_receipt=dict(path=str(output/'old_provenance_recheck/PROVENANCE.json'),identity=identity(output/'old_provenance_recheck/PROVENANCE.json')),
            source_commit=plan['execution_commit'],model_tensors_reread=False,checkpoint_authentication='Pinned executed source, exact native boundary/export checks, saved hash identities, own-checkpoint collector and runtime receipts; no tensor reconstruction.')
        write(output/'PROVENANCE.json',proof)
        result=analyze(requests,records,captures,checkpoints,proof)
        for key,name in (('per_request','PER_REQUEST.csv'),('paired_requests','PAIRED_REQUESTS.csv')):
            rows=result[key];fields=sorted({k for row in rows for k in row})
            with (output/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        write(output/'VALIDATION.json',result['validation'])
    except (OSError,ValueError,KeyError,TypeError,IndexError,subprocess.CalledProcessError) as error:
        result.update(error=repr(error))
        write(output/'PROVENANCE.json',dict(status='incomplete',errors=([repr(error)] if e is None else [*e.errors,repr(error)]),consumed_inputs=[] if e is None else e.inputs))
    result.update(analysis_wall_s=time.perf_counter()-began,remote_operations=0,tensors_read=0,threads=1)
    write(output/'SUMMARY.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery',required=True);p.add_argument('--old-delivery',required=True)
    p.add_argument('--qualification-delivery',required=True);p.add_argument('--restart-delivery',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkout',type=Path,default=SRC,help='DFlash-Online source tree used for analysis imports and pinned Git verification')
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise SystemExit('Set CUDA_VISIBLE_DEVICES empty for CPU-only analysis')
    result=run(**{k.replace('-','_'):v for k,v in vars(args).items()})
    print(json.dumps(dict(status=result['status'],wall_s=result['analysis_wall_s'])))
    raise SystemExit(result['status']!='completed')
