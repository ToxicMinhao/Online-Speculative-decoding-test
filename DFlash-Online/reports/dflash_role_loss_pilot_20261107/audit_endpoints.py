"""Independent read-only endpoint audit from raw native JSON; no production imports."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

METHODS=('KPLUS','MIX','ROLE')
REMOTE=Path('/data/sources/minhao/runs/dflash_role_loss_pilot_20261107/production_001')

class Audit:
    def __init__(self, root):
        self.root=root
        self.checks=0
        self.files={}
        self.rows=0
        self.blocks=0
        self.stages={}
        self.plan=self.read(root/'sealed/PLAN.json')
    def check(self, condition, context):
        self.checks+=1
        if not condition: raise AssertionError(context)
    def read(self, p, lines=False):
        p=Path(p)
        payload=p.read_bytes()
        self.files[str(p.relative_to(self.root))]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
        return [json.loads(line) for line in payload.splitlines() if line] if lines else json.loads(payload)
    def local(self,p):
        return self.root/Path(p).relative_to(REMOTE)
    def status(self,s,expected):
        for k,v in expected.items(): self.check(s.get(k)==v,'status '+k)
    def publication(self,s,checkpoint):
        self.check(s['errors']==[], 'publication errors')
        self.check(s['checkpoint_id']==checkpoint and s['checkpoint_structure_valid'] is True,'publication identity')
        n=s['native_status'];p=n['publication']
        self.check(n['status']=='passed' and n['poisoned'] is None and n['capture_pending'] is False,'worker healthy')
        self.check(n['cache_tokens']==p['cache_after_tokens']==0,'empty publication caches')
        self.check(n['attention_backend']=='fa3' and n['block_size']==16,'publication runtime')
        self.check(p['status']=='passed' and p['checkpoint_path']==checkpoint,'published path')
        for k in ('helper_matches_current_layers','helper_norm_matches_current_layers','fused_enabled'):
            self.check(p[k] is True,'publication '+k)
    def raw(self, raw, request, canonical, checkpoint, label, row):
        m=raw['metadata'];c=m['capture'];trace=m['trace'];ids=raw['response_ids'];prompt=raw['prompt_ids'];stops=raw['stop_token_ids']
        self.check(prompt==request['prompt_ids']==canonical['prompt_ids'],label+' exact prompt')
        self.check(ids==canonical['response_ids'] and raw['finish_reason']==canonical['finish_reason'],label+' exact output/termination')
        self.check(raw['checkpoint_id']==checkpoint,label+' checkpoint')
        self.check(raw['record_id']==label and raw['request_index']==request['panel_ordinal'] and raw['session_id']=='role_loss_pilot_evaluation',label+' request identity')
        self.check(raw['purpose']=='evaluation' and m['feedback_capture_in_timed_request'] is False,label+' capture disabled')
        self.check(raw['max_new_tokens']==request['max_new_tokens']==2048 and stops==request['stop_token_ids'],label+' cap/stops')
        self.check(m['block_size']==16 and m['context_limit']==4096 and m['runtime']=='pinned_SGLang_FA3',label+' runtime')
        self.check(m['cache_policy']=='idle_full_flush_before_every_request_and_publication',label+' cache policy')
        self.check(0<len(ids)<=2048 and len(prompt)+2048<=4096 and all(type(v) is int and v>=0 for v in prompt+ids),label+' token geometry')
        self.check(not any(v in stops for v in ids[:-1]),label+' no premature stop token')
        self.check((raw['finish_reason']=='eos' and ids[-1] in stops) or (raw['finish_reason']=='length' and len(ids)==2048 and ids[-1] not in stops),label+' termination boundary')
        self.check(c['prompt_ids']==prompt and c['response_ids']==ids and c['blocks']==trace,label+' capture binding')
        self.check(c['status']=='passed' and c['collect_features'] is False and c['feature_bytes']==0 and c['native_capture_validated'] is True,label+' native capture')
        sidecar=self.read(self.local(m['native_trace_path']))
        self.check(sidecar==c,label+' native sidecar')
        offset=0;accepted_total=0;native_total=0
        for bi,b in enumerate(trace):
            self.blocks+=1;w=b['verify_size'];d=b['draft_tokens'];p=b['target_argmax'];n=b['committed_progress']
            self.check(type(w) is int and 1<=w<=16 and len(d)==w-1 and len(p)==w,label+' block geometry')
            self.check(type(n) is int and 1<=n<=w,label+' progress bounds')
            self.check(all(type(v) is int and v>=0 for v in d+p),label+' native token types')
            first=next((j for j in range(w-1) if d[j]!=p[j]),w-1)
            proposed=d[:first]+[p[first]]
            self.check(n<=first+1 and b['committed_tokens']==proposed[:n]==p[:n],label+' native decision')
            self.check(b['output_start']==offset and b['anchor_token']==ids[offset],label+' contiguous starts/seeds')
            self.check(b['committed_tokens']==ids[offset+1:offset+n+1],label+' committed output')
            self.check(b['positions']==list(range(len(prompt)+offset,len(prompt)+offset+w)),label+' absolute positions')
            self.check(b['accepted_draft_tokens']==min(first,n) and b['native_accepted_draft_tokens']==n-1 and b['raw_accepted_draft_tokens_before_stop']==first,label+' acceptance counters')
            self.check(b['stop_clipped']==(n<first+1),label+' clipping flag')
            offset+=n;accepted_total+=min(first,n);native_total+=n-1
            self.check(b['finished']==(bi==len(trace)-1),label+' finished flag')
        self.check(offset==len(ids)-1 and m['committed_progress']==offset,label+' complete coverage')
        self.check(m['verification_calls']==len(trace) and m['accepted_draft_tokens']==accepted_total and native_total==offset-len(trace),label+' total counts')
        for name in ('request_latency_s','native_generate_s'):
            self.check(isinstance(m[name],(int,float)) and math.isfinite(m[name]) and m[name]>0,label+' time '+name)
        self.check(math.isfinite(c['cache_maintenance_s']) and c['cache_maintenance_s']>=0,label+' cache time')
        expected=dict(checkpoint_id=checkpoint,prompt_tokens=len(prompt),output_tokens=len(ids),cap_hit=raw['finish_reason']=='length',finish_reason=raw['finish_reason'],verification_calls=len(trace),accepted_draft_tokens=accepted_total,native_accepted_draft_tokens=native_total,committed_progress=offset,request_latency_s=m['request_latency_s'],native_generate_s=m['native_generate_s'],cache_maintenance_s=c['cache_maintenance_s'])
        for k,v in expected.items():self.check(row[k]==v,label+' saved scalar '+k)
        self.check(row['validation_status']=='valid' and row['valid_measurement'] is True and row['errors']==[] and row['structural_errors']==[],label+' valid flag')
        self.rows+=1
        return expected
    def stage(self,dataset,event):
        folder=self.root/'campaign_001/evaluation'/dataset/f'event_{event:03d}'
        protocol=self.read(folder/'PROTOCOL.json');reqs=self.read(folder/'REQUESTS.json');canon=self.read(folder/'CANONICAL.json');rows=self.read(folder/'MEASUREMENTS.jsonl',True);status=self.read(folder/'STATUS.json')
        repeats=1 if event==2 else 2;expected_count=256*3*repeats
        item=self.plan['datasets'][dataset]
        sealed_panel=self.read(self.local(item['panel_manifest']))
        sealed_requests=sealed_panel if isinstance(sealed_panel,list) else next(sealed_panel[k] for k in ('evaluation_requests','requests','panel_requests') if k in sealed_panel)
        self.check(reqs==sealed_requests,'exact presealed panel')
        self.check(canon==self.read(self.local(item['references'])),'exact presealed canonicals')
        self.check(protocol['schedule']==item['evaluation_schedules'][str(event)],'exact presealed schedule')
        self.check(len(reqs)==256 and len(canon)==256 and len({r['request_id'] for r in reqs})==256,'stage panel cardinality')
        self.check([r['panel_ordinal'] for r in reqs]==list(range(256)),'panel ordinals')
        checkpoints=protocol['checkpoints']
        self.check(set(checkpoints)==set(METHODS) and len(set(checkpoints.values()))==3,'three checkpoints')
        self.check(protocol['capture'] is False and protocol['extra_warmup_generations']==0 and protocol['count_trajectory_repeat']==0,'stage protocol')
        self.status(status,dict(status='completed',planned_generations=expected_count,attempted_generations=expected_count,returned_generations=expected_count,completed_valid_generations=expected_count,invalid_generations=0,unknown_generation_attempts=0,unattempted_generations=0,pending_canonical_generations=0,optimizer_steps=0,feedback_collections=0,teacher_preparations=0,extra_generations=0))
        orders=((0,1,2),(1,2,0),(2,0,1),(2,1,0),(1,0,2),(0,2,1));slots=[];batches=[]
        for repeat in range(repeats):
            for block in range(8):
                for pos,mi in enumerate(orders[(repeat*8+block)%6]):
                    batches.append([block,repeat,pos])
                    for ordinal in range(block*32,(block+1)*32):
                        slots.append(dict(seed=17,dataset=dataset,event=event,phase='final' if event==4 else 'intermediate',repeat=repeat,block_ordinal=block,panel_ordinal=ordinal,position=pos,request_id=reqs[ordinal]['request_id'],method=METHODS[mi]))
        self.check(protocol['schedule']==slots and len(rows)==len(slots)==expected_count,'sealed schedule')
        for method in METHODS:
            w=self.read(folder/f'WEIGHTS_{method}.json');n=w['native_reply']
            self.check(w['errors']==[] and n['status']=='passed' and n['all_equal'] is True,'loaded tensor equality')
            self.check(n['tensors']==n['source_tensor_count']==58 and n['native_tensor_count']==43,'weight tensor counts')
            self.check(n['checkpoint_path']==checkpoints[method],'weight checkpoint')
            for k in ('helper_matches_current_layers','helper_norm_matches_current_layers','fused_enabled'):self.check(n[k] is True,'weights '+k)
            boundary=self.read(self.root/'campaign_001/trajectories'/dataset/method/'boundaries'/f'event_{event:03d}.json')
            self.check(boundary['status']=='passed' and boundary['weights']==checkpoints[method]+'/model.safetensors','selected boundary')
        acts=self.read(folder/'ACTIVATIONS.jsonl',True);active=None;activation_slots=[]
        for s in slots:
            if s['method']!=active:activation_slots.append(s);active=s['method']
        self.check(len(acts)==len(activation_slots),'activation count')
        for a,s in zip(acts,activation_slots):
            self.check(all(a.get(k)==v for k,v in s.items()),'activation slot');self.publication(a,checkpoints[s['method']])
        boundary_checks=self.read(folder/'BOUNDARY_CHECKS.jsonl',True)
        self.check([b['batch'] for b in boundary_checks]==batches,'boundary schedule')
        for b in boundary_checks:
            q=b['check']['quiescence'];s=b['check']['storage']
            self.check(q['status']=='passed' and not any(q[k] for k in ('training','transfers','foreign_gpu_processes')),'timing quiescence')
            self.check(s['status']=='passed' and s['free_bytes']>=s['required_bytes'],'timing storage')
        grouped=defaultdict(list);caps=Counter()
        for row,slot in zip(rows,slots):
            self.check(all(row[k]==v for k,v in slot.items()),'measurement slot')
            label=f"{slot['panel_ordinal']:04d}_r{slot['repeat']}_{slot['method']}";request=reqs[slot['panel_ordinal']]
            raw=self.read(folder/(label+'.json'))
            self.check(self.local(row['record_path'])==folder/(label+'.json'),'saved record path')
            measured=self.raw(raw,request,canon[request['request_id']],checkpoints[slot['method']],label,row)
            validity=self.read(folder/(label+'_VALIDATION.json'))
            self.check(validity==dict(label=label,status='valid',errors=[]),'separate validity record')
            grouped[(slot['method'],slot['panel_ordinal'])].append(measured)
            caps[slot['method']]+=measured['cap_hit']
        count_fields=('verification_calls','accepted_draft_tokens','native_accepted_draft_tokens','committed_progress','output_tokens','finish_reason','checkpoint_id','cap_hit')
        for key,group in grouped.items():
            self.check(len(group)==repeats,'repeat cardinality')
            self.check(all(tuple(g[k] for k in count_fields)==tuple(group[0][k] for k in count_fields) for g in group),'repeat consistency')
        totals={m:{k:sum(grouped[(m,i)][0][k] for i in range(256)) for k in count_fields[:5]} for m in METHODS}
        self.stages[f'{dataset}/event_{event:03d}']=dict(executions=len(rows),requests=256,repeats=repeats,raw_count_totals=totals,cap_executions=dict(caps),native_publications=len(acts),timing_boundaries=len(boundary_checks))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    a=Audit(args.root.resolve());start=time.perf_counter();result=dict(status='running',execution_commit='cdb640543d8fe3a43c8e63af4ccaaa002e09a599',scope='Independent local raw-JSON assertions; no production validators imported; no model execution.')
    try:
        for d in ('gsm8k','mathqa'):
            for e in (2,4):a.stage(d,e)
        a.check(a.rows==4608,'full 4608 execution budget')
        result['status']='passed'
    except Exception as e:
        result.update(status='failed',error=repr(e));raise
    finally:
        result.update(assertions=a.checks,raw_executions=a.rows,native_verification_rounds=a.blocks,stages=a.stages,files_read=len(a.files),wall_s=time.perf_counter()-start,model_work=0)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'ENDPOINT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
        (args.output/'ENDPOINT_AUDIT_FILES.json').write_text(json.dumps(a.files,indent=2)+'\n')
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
