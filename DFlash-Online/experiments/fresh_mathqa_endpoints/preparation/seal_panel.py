"""Seal the approved input-only selection and reuse authenticated warmup witnesses.
All writes stay in panel_audit/sealed. No model work, network, or remote access.
"""
from collections import Counter
from itertools import permutations
from pathlib import Path
import hashlib, json, shutil, sys
from audit_pool import OUT, REV, M26, M53, M54, canon, h, identity, norm, tokens_h

DEST=OUT/'sealed'
SEED=20261119
SCOPE='FRESH_MATHQA_C58_AREPO58_O58_20261119'
METHODS=('O58','AREPO58','KPLUS_C58')


def read(p):return json.loads(Path(p).read_text())
def check(x,s):
    if not x:raise AssertionError(s)
def put(name,x):
    data=(json.dumps(x,indent=2,ensure_ascii=False)+'\n').encode()
    p=DEST/name
    if p.exists():check(p.read_bytes()==data,'Existing immutable seal differs: '+name)
    else:p.write_bytes(data)
def copy_exact(source,name):
    p=DEST/name
    if p.exists():check(p.read_bytes()==source.read_bytes(),'Existing exact copy differs: '+name)
    else:shutil.copyfile(source,p)

def main():
    audit=read(OUT/'POOL_AUDIT.json');eligible=read(OUT/'ELIGIBLE_INPUTS_PRIVATE.json')['requests']
    excluded=read(OUT/'EXCLUSIONS_PRIVATE.json')['rows']
    check(audit['status']=='passed_pool_availability','Pool gate failed')
    check(len(eligible)==1186 and len(excluded)==1799,'Audited counts differ')
    check(len(eligible)>=1024,'Insufficient input pool: stop')
    exclusion_ids={r['source_id'] for r in excluded}
    ranked=[]
    for item in eligible:
        row=dict(item)
        key={'seed':SEED,'dataset_revision':REV,'split':'test','source_id':row['source_id']}
        text=canon(key)
        row['selection_input']=key;row['selection_canonical_json']=text;row['selection_key']=h(text)
        ranked.append(row)
    ranked.sort(key=lambda r:(r['selection_key'],canon(r['source_identity'])))
    selected=[]
    for ordinal,r in enumerate(ranked[:1024]):
        r.update(dataset='mathqa',scope=SCOPE,role='scored',ordinal=ordinal,panel_ordinal=ordinal,original_index=r['source_row'],request_index=ordinal,request_id=f'mathqa-fresh-20261119-{ordinal:04d}',max_new_tokens=2048,stop_token_ids=[151645,151643],collect_feedback=False,block_ordinal=ordinal//64)
        check(r['source_id'] not in exclusion_ids,'Known request entered panel')
        selected.append(r)
    check(len({r['source_id'] for r in selected})==1024,'Duplicate selected source')
    for f in ('source_identity_sha256','normalized_prompt_sha256','prompt_ids_sha256'):
        check(len({r[f] for r in selected})==1024,'Duplicate selected identity '+f)
        check(not ({r[f] for r in selected} & {r[f] for r in excluded if f in r}),'Excluded hash intersects selection '+f)
    old=M54/'package_001/sealed';plan=read(old/'PLAN.json')
    source_warmups=old/'mathqa/WARMUPS.json';source_canonicals=old/'mathqa/WARMUP_CANONICALS.json'
    for p in (source_warmups,source_canonicals):
        remote=plan['root']+'/sealed/'+str(p.relative_to(old))
        check(identity(p)==plan['input_identities'][remote],'PR54 warmup input identity')
    warms=read(source_warmups)['requests'];warm_canon=read(source_canonicals)
    check(len(warms)==2 and [w['panel_ordinal'] for w in warms]==[0,1],'Two fixed ordered warmups')
    witnesses=[]
    for w in warms:
        check(w['source_id'] in exclusion_ids,'Warmup was not previously used')
        check(w['source_id'] not in {r['source_id'] for r in selected},'Warmup/scored source overlap')
        check(h(norm(w['rendered_prompt'])) not in {r['normalized_prompt_sha256'] for r in selected},'Warmup/scored prompt overlap')
        check(tokens_h(w['prompt_ids']) not in {r['prompt_ids_sha256'] for r in selected},'Warmup/scored token overlap')
        c=warm_canon[w['request_id']]
        check(c['request_id']==w['request_id'] and c['source_id']==w['source_id'] and c['prompt_ids']==w['prompt_ids'],'Warmup reference input binding')
        check(c['verified'] is True and c['reference_kind']=='native_greedy_target_verifier','Warmup qualified native producer')
        payload={k:v for k,v in c.items() if k!='binding_digest'}
        check(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()==c['binding_digest'],'Warmup canonical content digest')
        source_record=M53/'delivery/final_execution_001'/Path(c['source_record_path']).relative_to('/data/sources/minhao/runs/dflash_c_paired_confirmation_20261117/execution_001')
        check(identity(source_record)==c['source_record_identity'],'Warmup raw witness identity')
        raw=read(source_record)
        for field in ('prompt_ids','response_ids','finish_reason','stop_token_ids','max_new_tokens'):
            check(raw[field]==c[field],'Warmup raw canonical mismatch: '+field)
        check(raw['checkpoint_id']==c['reference_checkpoint'],'Warmup original producer checkpoint')
        check(c['stop_token_ids']==[151645,151643] and c['max_new_tokens']==2048,'Warmup stopping contract')
        response=c['response_ids']
        check(0<len(response)<=2048,'Warmup output cap')
        stops=[i for i,t in enumerate(response) if t in c['stop_token_ids']]
        check(c['finish_reason']=='eos' and stops==[len(response)-1],'Warmup terminal EOS witness')
        witnesses.append({'request_id':w['request_id'],'source_id':w['source_id'],'source_record':str(source_record),'source_record_identity':identity(source_record),'canonical_binding_digest':c['binding_digest'],'producer':c['reference_checkpoint'],'finish_reason':c['finish_reason'],'response_tokens':len(response),'response_ids_sha256':tokens_h(response),'reused_without_generation':True})
    # Literal schedule prescribed in handoff: aggregate balance across 48 groups.
    orders=tuple(sorted(permutations(METHODS)));schedule=[]
    for repeat in range(3):
        for group in range(16):
            order=orders[(16*repeat+group)%6]
            for position,method in enumerate(order):
                schedule.append({'activation':len(schedule),'repeat':repeat,'block_ordinal':group,'position':position,'method':method,'request_ordinals':list(range(group*64,(group+1)*64))})
    check(len(schedule)==144,'Activation total')
    check(Counter(tuple(schedule[3*i+j]['method'] for j in range(3)) for i in range(48))==Counter({x:8 for x in orders}),'Permutation balance')
    coverage={k:audit[k] for k in ('counts','source','tokenizer','normalization','source_identity','prompt_token_identity','prompt_template','role_inventory','historical_coverage','checks')}
    coverage.update(status='passed_pool_and_panel_seal',scope=SCOPE,scored=1024,warmup=2,unselected_eligible=162,selection_uses_outputs=False,missing_essential_manifests=[],freshness_scope=audit['scope'])
    order={'panel_order':[r['request_id'] for r in selected],'source_id_order':[r['source_id'] for r in selected],'lexicographic_request_id_order':sorted(r['request_id'] for r in selected),'bootstrap_seed':SEED,'bootstrap_order_sha256':h(canon(sorted(r['request_id'] for r in selected))),'digest_encoding':'SHA256 UTF-8 canonical JSON array, sorted object keys, compact separators; request IDs lexicographically sorted'}
    selection={'seed':SEED,'dataset_revision':REV,'split':'test','identity_keys':['seed','dataset_revision','split','source_id'],'canonical_json':'UTF-8 json.dumps(object, sort_keys=True, separators=(comma,colon), ensure_ascii=False); all rank fields ASCII','ranking':'ascending SHA256 of canonical JSON; ties ascending canonical JSON source identity','required':1024,'eligible_count':1186,'first_unselected_rank':1024,'no_answer_or_response_or_performance_inputs':True,'ordered_eligible_id_sha256':h(canon([r['source_id'] for r in ranked])),'ordered_selected_id_sha256':h(canon([r['source_id'] for r in selected])),'ranked_eligible':[{'source_identity':r['source_identity'],'selection_key':r['selection_key']} for r in ranked]}
    DEST.mkdir(exist_ok=True)
    put('PANEL.json',{'requests':selected,'scope':SCOPE,'source':audit['source'],'selection':{k:v for k,v in selection.items() if k!='ranked_eligible'}})
    copy_exact(source_warmups,'WARMUPS.json');copy_exact(source_canonicals,'WARMUP_CANONICALS.json');copy_exact(OUT/'EXCLUSIONS_PRIVATE.json','EXCLUSIONS.json')
    put('SOURCE_AND_COVERAGE.json',coverage);put('SELECTION.json',selection);put('ORDER.json',order)
    put('WARMUP_AUTHENTICATION.json',{'status':'passed','source_warmups':str(source_warmups),'source_warmups_identity':identity(source_warmups),'source_canonicals':str(source_canonicals),'source_canonicals_identity':identity(source_canonicals),'references':witnesses,'new_warmup_generations':0,'copied_byte_identically':True})
    put('SCHEDULE.json',{'activations':schedule,'methods':list(METHODS),'repeat_count':3,'scheduling_group_size':64,'draft_block_size':16,'permutation_occurrences':8,'per_request_position_balance_claim':False,'scored_generation':9216,'warmup_generation':288,'canonical_generation':1024,'qualification_generation_max':24,'total_generation_max':10552})
    files={p.name:identity(p) for p in sorted(DEST.iterdir()) if p.is_file() and p.name!='SEAL.json'}
    put('SEAL.json',{'schema':'fresh_mathqa_input_seal_v1','status':'sealed_before_new_model_work','scope':SCOPE,'files':files,'input_audit':{'path':str(OUT/'POOL_AUDIT.json'),'identity':identity(OUT/'POOL_AUDIT.json')},'scripts':{str(OUT/n):identity(OUT/n) for n in ('audit_pool.py','seal_panel.py')},'scored':1024,'warmup':2,'source_pool':2985,'excluded':1799,'eligible':1186,'unused_eligible_after_selection':162,'canonicals_pending':True,'fresh_canonical_work_performed':0,'new_model_work':0,'network_operations':0,'remote_operations':0,'deletions':0,'bootstrap_input_order_sha256':order['bootstrap_order_sha256'],'old_warmup_canonical_values_and_bytes_unchanged':True})
    print(json.dumps({'status':'sealed','panel':str(DEST/'PANEL.json'),'panel_identity':identity(DEST/'PANEL.json'),'seal_identity':identity(DEST/'SEAL.json'),'scored':1024,'warmups':2,'unused_eligible':162,'warmup_references_authenticated':2,'bootstrap_order_sha256':order['bootstrap_order_sha256']}))

if __name__=='__main__':main()
