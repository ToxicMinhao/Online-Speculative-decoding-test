"""Read-only identity, inherited-correctness, and cost-reconciliation addendum."""
import hashlib,json,pathlib,collections,os,subprocess
R=pathlib.Path('/data/sources/minhao/runs/dflash_full_k_gsm8k_20261031/production_002')
checks=collections.Counter();errors=[]
def ck(ok,label,group):
    checks[group]+=1
    if not ok: errors.append(label)
def read(p): return json.loads(pathlib.Path(p).read_text())
seal=read(R/'sealed/SEAL.json'); plan=read(R/'sealed/PLAN.json')
for name,expected in seal['files'].items():
    p=pathlib.Path(name); b=p.read_bytes()
    ck({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}==expected,name,'source_config_hash')
manifest=read(plan['manifest']); reports=read(R/'final_report/COSTS.json'); output={'source_files':len(seal['files']),'arms':{}}
for arm in ('K','K_HALF'):
    base=R/'campaign_001/arms'/arm; event=base/'events/event_001'
    first=read(event/'loop/serving'/f"{manifest['train_requests'][0]['request_id']}.json")
    replay=read(event/'loop/evaluation/first_capture_disabled.json')
    cc=read(event/'loop/evaluation/first_capture_disabled.comparison.json')
    ck(cc['record']==replay and cc['same_input'] is True and cc['output_equal'] is True and cc['learning_unchanged'] is True,'capture control receipt','correctness')
    ck(replay['response_ids']==first['response_ids'] and replay['finish_reason']==first['finish_reason'] and replay['metadata']['verification_calls']==first['metadata']['verification_calls'],'capture-disabled exact tokens/counts','correctness')
    records={n:read(event/f'qualification_{n}.json') for n in ('updated','frozen','reloaded')}
    ck(len({tuple(v['response_ids']) for v in records.values()})==1 and len({v['finish_reason'] for v in records.values()})==1,'publication exact tokens/termination','correctness')
    ck(records['updated']['metadata']['verification_calls']==records['reloaded']['metadata']['verification_calls'],'publication exact reload calls','correctness')
    control=read(event/'INHERITED_CORRECTNESS.json')
    ck(control['generations']==4 and control['disposable_optimizer_steps']==0 and control['feedback_collected'] is False and control['performance_win_required'] is False and control['first_capture_disabled_in_loop'] is True,'exact inherited correctness budget','correctness')
    for rec in [replay,*records.values()]:
        ck(rec['metadata']['feedback_capture_in_timed_request'] is False and rec['metadata']['capture']['collect_features'] is False,'control no feedback capture','correctness')
        ck(pathlib.Path(rec['metadata']['native_trace_path']).is_file(),'control native evidence retained','correctness')
    status=read(base/'STATUS.json'); costs=reports['arms'][arm]
    raw={
        'loading_s':sum(s[k] for s in status['segments'] for k in ('backend_loading_s','trainer_loading_s')),
        'online_wall_s':sum(status['worker_segments_s']),
        'initial_validation_s':sum(s['initial_validation_s'] for s in status['segments'] if not s['resume']),
        'publication_validation_s':sum(e['costs']['publication_validation_s'] for e in status['event_costs']),
        'retention_s':sum(e['costs']['retention_s'] for e in status['event_costs']),
        'correctness_publication_replay_s':control['wall_s'],
        'correctness_capture_disabled_replay_s':read(event/'loop/evaluation/first_capture_disabled.costs.json')['evaluation_call_s'],
    }
    tails=[read(base/'tail'/f"{request['request_id']}.json") for request in manifest['train_requests'][7424:]]
    raw.update(tail_native_request_s=sum(r['metadata']['request_latency_s'] for r in tails),tail_native_generate_s=sum(r['metadata']['native_generate_s'] for r in tails),tail_cache_maintenance_s=sum(r['metadata']['capture']['cache_maintenance_s'] for r in tails))
    for key,value in raw.items(): ck(value==costs[key],'cost '+arm+'/'+key,'cost')
    output['arms'][arm]={'correctness_generations':4,'additional_reconstructed_costs_s':raw}
accounting=read(R/'final_report/ACCOUNTING.json')
final=next(x for x in accounting['evaluation']['stages'] if x['phase']=='final')
ck(final['planned_generations']==final['attempted_generations']==final['returned_generations']==final['completed_valid_generations']==12288,'final generation receipt','budget')
ck(7473*2+4*2+384*2+final['completed_valid_generations']==28010,'complete generation total','budget')
ck(accounting['training']['status']=='passed' and not accounting['training']['errors'],'author full training audit','receipt')
output.update(checks_by_category=dict(checks),checks_total=sum(checks.values()),errors=errors,status='passed' if not errors else 'findings',
    final_generations_from_evaluation_receipt=final['completed_valid_generations'],combined_complete_generations=28010,
    final_generation_scope='Final raw evaluation/bootstrap checked by separate evaluation reviewer; this addendum reconciles completed receipts.',
    final_storage_free_bytes=os.statvfs('/data/sources/minhao').f_bavail*os.statvfs('/data/sources/minhao').f_frsize,
    gpu_status=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
    gpu_compute_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))
print(json.dumps(output,indent=2))
