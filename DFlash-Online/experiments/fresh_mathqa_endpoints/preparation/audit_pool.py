"""Read-only input inventory audit; writes only this task's panel_audit directory.

Uses authenticated historical inventory receipts, extends them with actual
PR51/52/53/54 role manifests, and rerenders the complete pinned MathQA test pool.
It never loads model weights, outputs, quality metrics, or remote resources.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib, json, os, re, time, unicodedata, zipfile

W = Path('/home/minhao2/EAGLE_SPEC')
R = W / 'research'
OUT = R / 'dflash_fresh_mathqa_c_arepo_osd_20261119/panel_audit'
REV = '4f958013c164a40745a5b771dfb005df308deb4a'
BLOB = '193fb150706468110459ddcdc52ad48e3f0b7491'
M26 = R / 'dflash_mathqa_transfer_20261104'
M51 = R / 'dflash_mathqa_fixed_lr_transfer_20261116'
M52 = R / 'dflash_unchanged_c_mathqa_transfer_20261117'
M53 = R / 'dflash_c_paired_confirmation_20261117'
M54 = R / 'dflash_full_c_gsm8k_20261118'
receipts = {}
checks = Counter()

def sha(b): return hashlib.sha256(b).hexdigest()
def h(s): return sha(s.encode('utf-8'))
def canon(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
def norm(s): return ' '.join(unicodedata.normalize('NFKC', s).split())
def tokens_h(x): return h(json.dumps(x, separators=(',', ':')))
def identity(p):
    b = Path(p).read_bytes()
    return {'bytes': len(b), 'sha256': sha(b)}
def require(ok, label):
    if not ok: raise AssertionError(label)
    checks[label] += 1

def read(p, expected=None):
    p=Path(p); pin=identity(p)
    if expected is not None:
        require(all(pin[k] == expected[k] for k in ('bytes','sha256')), 'authenticated_file_identity')
    receipts[str(p)] = pin
    return json.loads(p.read_text())

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')

def main():
    started=time.perf_counter()
    prior=read(M51/'data_readiness/READINESS.json')
    seal51=read(M51/'data_readiness/sealed/DATA_SEAL.json',prior['data_seal']['identity'])
    for name,pin in seal51['preparation_inputs'].items():
        # Scripts are provenance rather than a source of exclusion decisions.
        p=M51/'data_readiness'/name
        require(identity(p)==pin, 'prior_preparation_identity')
        receipts[str(p)]=pin
    early=read(M51/'data_readiness/READINESS_PRELIMINARY.json',prior['preliminary_audit']['identity'])
    inventory=read(M51/'data_readiness/INVENTORY_FILES.json',prior['consumed_inventory']['identity'])
    ancient=read(Path(prior['old_history']['path']),prior['old_history']['identity'])
    require(not early['errors'] and not early['unknown_ids'] and not early['malformed'],'historical_audit_no_declared_gaps')
    require(not ancient['unreadable_files'] and not ancient['unavailable_referenced_inventories'],'older_audit_no_declared_gaps')
    remote=read(M26/'retrieved_002/production_002/sealed/REMOTE_MATHQA_INVENTORY_002.json')
    require(remote['status']=='completed' and not remote['skipped'] and not remote['matches'],'historical_remote_inventory_no_reported_gap_or_match')
    reconstructed={f'mathqa:test:{i}' for i in ancient['overlapping_source_rows']}
    for row in inventory['files']:
        reconstructed.update(f'mathqa:test:{i}' for i in row['matched_test_rows'])
    original_exclusions={x['source_id'] for x in early['excluded']}
    require(reconstructed==original_exclusions,'inherited_exclusion_union_reproduced_from_inventory')
    require(len(original_exclusions)==1025,'inherited_1025_exclusions')

    archive=M26/'data_preparation/MathQA.zip'; raw=archive.read_bytes()
    require(identity(archive)==early['source']['archive'],'pinned_archive_sha256')
    require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==BLOB,'pinned_archive_git_blob')
    receipts[str(archive)]=identity(archive)
    with zipfile.ZipFile(archive) as z:
        sources={}
        for split,pin in early['source']['members'].items():
            raw_member=z.read(split+'.json'); require(len(raw_member)==pin['bytes'] and sha(raw_member)==pin['sha256'],'pinned_split_sha256')
            sources[split]=json.loads(raw_member)
            require(len(sources[split])==pin['rows'],'pinned_split_rows')
    require(len(sources['test'])==2985 and early['source']['revision']==REV,'pinned_test_pool')
    tokenizer_path=Path(prior['tokenizer_path'])
    for name,pin in prior['tokenizer'].items():
        require(identity(tokenizer_path/name)==pin,'pinned_tokenizer_file')
        receipts[str(tokenizer_path/name)]=pin
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    rendered=read(M51/'data_readiness/RENDERED_TEST_INPUTS_PRIVATE.json',seal51['preparation_inputs']['RENDERED_TEST_INPUTS_PRIVATE.json'])['requests']
    require(len(rendered)==2985,'rendered_catalog_size')
    candidates=[];lookups=defaultdict(set);source_content={}
    for split, rows in sources.items():
        for i,row in enumerate(rows):
            source_content[f'mathqa:{split}:{i}']=(norm(row['Problem']),norm(row['Problem']+'\n'+row['options']))
    fields=('normalized_question_sha256','normalized_content_sha256','rendered_prompt_sha256','normalized_prompt_sha256','prompt_ids_sha256')
    for i,row in enumerate(sources['test']):
        message=row['Problem'].strip()+'\n\nOptions: '+row['options'].strip()
        prompt=tokenizer.apply_chat_template([dict(role='user',content=message)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        fp=dict(normalized_question_sha256=h(norm(row['Problem'])),normalized_content_sha256=h(norm(row['Problem']+'\n'+row['options'])),rendered_prompt_sha256=h(prompt),normalized_prompt_sha256=h(norm(prompt)),prompt_ids_sha256=tokens_h(ids))
        old=rendered[i]
        require(old['source_id']==f'mathqa:test:{i}' and old['source_row']==i,'retained_source_row_order')
        require(old['user_message']==message and old['rendered_prompt']==prompt and old['prompt_ids']==ids,'all_2985_render_and_token_parity')
        require(all(old[k]==fp[k] for k in fields),'all_2985_fingerprint_parity')
        record=dict(dataset_revision=REV,split='test',stable_source_id=old['source_id'],source_id=old['source_id'],source_row=i,question=row['Problem'].strip(),options=row['options'],user_message=message,rendered_prompt=prompt,prompt_ids=ids,prompt_tokens=len(ids),**fp)
        record['source_identity']={'dataset_revision':REV,'split':'test','source_id':old['source_id']}
        record['source_identity_sha256']=h(canon(record['source_identity']))
        candidates.append(record)
        for text in (norm(row['Problem']),norm(row['Problem']+'\n'+row['options']),norm(message),norm(prompt)):
            lookups[('text',text)].add(i)
        for key in fields:lookups[('hash',record[key])].add(i)
        lookups[('token',tokens_h(ids))].add(i)
    # Current tree roots are reconciled to the previously audited roots and
    # explicit later manifests. A new unidentified research task fails closed.
    supplement=read(M52/'data_readiness/sealed/EXPOSURE_SUPPLEMENT.json')
    roots=set(early['discovery']['tasks'])|set(supplement['subsequent_roots'])|{M51.name,M52.name,M53.name,M54.name,'dflash_post_pr52_origin_analysis_20261117',OUT.parent.name}
    present=sorted(p.name for p in R.iterdir() if p.is_dir() and (m:=re.search(r'_(2026\d{4})',p.name)) and m.group(1)>='20261116')
    unknown_roots=sorted(set(present)-roots)
    require(not unknown_roots,'all_later_dated_task_roots_accounted_for')
    post=read(R/'dflash_post_pr52_origin_analysis_20261117/analysis/reduction_001/STATUS.json')
    require(post['new_model_work']==0,'post_origin_analysis_no_model_work')

    roles=[]
    for label,task in [('PR51',M51),('PR52',M52)]:
        delivery=task/'delivery/execution_001'; manifest=read(delivery/'RETRIEVAL_MANIFEST.json')
        work=read(delivery/'WORK.json',manifest['WORK.json'])
        require(not work['pending'] and work['attempted']==work['completed'],label+'_completed_work_ledger')
        for name in ('PANEL.json','WARMUPS.json','TRAINING.json','EVENT5_INPUTS.json','EVENT6_INPUTS.json'):
            path=delivery/'sealed'/name; roles.append((label+':'+name,path,read(path,manifest['sealed/'+name])))
        if label=='PR51':
            path=delivery/'sealed/ORIGIN_TRAINING.json'; roles.append((label+':ORIGIN_TRAINING.json',path,read(path,manifest['sealed/ORIGIN_TRAINING.json'])))
    sealed53=M53/'data_preparation/sealed_003/mathqa';seal53=read(sealed53/'DATA_SEAL.json')
    for name,pin in seal53['inputs'].items():require(identity(name)==pin,'PR53_inherited_input_identity')
    d53=M53/'delivery/final_execution_001'; mf53=read(d53/'COMPACT_FINAL_MANIFEST.json')
    pins53={x['path']:x for x in mf53['files']}
    work53=read(d53/'WORK.json',pins53['WORK.json'])
    require(not work53['pending'] and work53['attempted']==work53['completed'],'PR53_completed_work_ledger')
    require(work53['completed']['scored_generation']==3072 and work53['completed']['canonical_generation']==516,'PR53_actual_completed_panels')
    for name in ('PANEL.json','WARMUPS.json','TRAINING.json','EVENT5_INPUTS.json','EVENT6_INPUTS.json','ORIGIN_ANCESTRY.json'):
        path=d53/'sealed/mathqa'/name;value=read(path,pins53['sealed/mathqa/'+name])
        require(identity(path)==seal53['files'][name],'PR53_executed_input_equals_preseal')
        roles.append(('PR53:'+name,path,value))
    sealed54=M54/'package_001/sealed';plan54=read(sealed54/'PLAN.json')
    for name in ('mathqa/PANEL.json','mathqa/WARMUPS.json','ORIGINAL_MANIFEST.json'):
        path=sealed54/name;remote_path=plan54['root']+'/sealed/'+name
        roles.append(('PR54:'+name,path,read(path,plan54['input_identities'][remote_path])))
    # PR54 warmups are previously qualified requests, and its scored requests
    # are the original MathQA transfer panel. Check all three identity modes.
    # Input-only walker never recurses into output/answer/statistical fields.
    ignored={'answer','response','output','completion','reference_answer','generated_text','output_text','response_text','assistant','metrics','exclusions','rejections','candidate_pools','sources','source_catalog','history_files','rationale','correct','annotated_formula','linear_formula','category','eligible_order','unselected_eligible_ids'}
    text_fields={'text','prompt','content','value','user','user_text','question','problem','user_message','rendered_prompt','prompt_text','instruction','normalized_stem','options'}
    token_fields={'prompt_ids','input_ids','prompt_token_ids','rendered_prompt_token_ids'}
    token_hash_fields={'prompt_ids_sha256','prompt_token_ids_sha256','rendered_token_sha256'}
    id_fields={'source_id','dataset_id','request_id','id','source_key'}
    evidence=defaultdict(list)
    for sid in sorted(original_exclusions):evidence[int(sid.rsplit(':',1)[1])].append({'role':'inherited_project_history','path':str(M51/'data_readiness/READINESS_PRELIMINARY.json'),'match_modes':['authenticated_exclusion_inventory']})
    role_summary=[];known_mathqa_ids=set();unknown_ids=set()
    for label,path,value in roles:
        matches=defaultdict(set);identities=set();counts=Counter()
        def text_match(s):
            variants=[s,*re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',s,re.S)]
            for v in variants:
                for t in (norm(v),norm(re.split(r'\n\s*Options\s*:',v,maxsplit=1)[0])):
                    for i in lookups.get(('text',t),()):matches[i].add('normalized_input_text')
        def walk(v,field=''):
            field=field.lower()
            if field in ignored:return
            if isinstance(v,dict):
                if v.get('role',v.get('from')) in ('assistant','gpt','chatgpt','bing','bard'):return
                for k,x in v.items():walk(x,k)
            elif isinstance(v,list):
                if field in token_fields and v and all(type(x) is int for x in v):
                    counts['token_arrays']+=1
                    for i in lookups.get(('token',tokens_h(v)),()):matches[i].add('exact_prompt_token_identity')
                else:
                    for x in v:walk(x,field)
            elif isinstance(v,str):
                if field in text_fields:counts['input_text_fields']+=1;text_match(v)
                if field in fields or field in token_hash_fields:
                    for i in lookups.get(('token' if field in token_hash_fields else 'hash',v),()):matches[i].add('retained_input_hash')
                if field in id_fields:
                    sid=v.replace('mathqa_test:','mathqa:test:')
                    if re.fullmatch(r'mathqa:(train|dev|test|challenge_test):\d+',sid):
                        identities.add(sid);known_mathqa_ids.add(sid)
                        if sid not in source_content:unknown_ids.add(sid)
                        else:
                            if sid.startswith('mathqa:test:'):matches[int(sid.rsplit(':',1)[1])].add('revision_split_source_id')
                            for t in source_content[sid]:
                                for i in lookups.get(('text',t),()):matches[i].add('source_content_identity')
        walk(value)
        new=set(matches)-set(evidence)
        for i,modes in matches.items():evidence[i].append({'role':label,'path':str(path),'match_modes':sorted(modes)})
        role_summary.append({'role':label,'path':str(path),'identity':receipts[str(path)],'input_counts':dict(counts),'distinct_mathqa_source_ids':len(identities),'matching_test_rows':len(matches),'new_exclusions':len(new),'match_mode_counts':dict(Counter(m for ms in matches.values() for m in ms))})
    require(not unknown_ids,'no_unresolved_later_mathqa_source_ids')
    exclusions=[];eligible=[];seen={f:{} for f in fields};duplicate_counts=Counter();invalid_counts=Counter()
    for row in candidates:
        why=[];i=row['source_row']
        if i in evidence:why.append('known_project_input_use_or_reservation')
        if not row['question'] or not row['options'].strip() or not row['prompt_ids'] or row['prompt_tokens']+2048>4096:
            why.append('input_ineligible');invalid_counts['invalid_or_context']+=1
        for f in fields:
            if row[f] in seen[f]:why.append('duplicate_'+f);duplicate_counts[f]+=1
            seen[f].setdefault(row[f],row['source_id'])
        if why:exclusions.append({'source_identity':row['source_identity'],'source_id':row['source_id'],'source_row':i,'reasons':why,'evidence':evidence.get(i,[]),**{f:row[f] for f in fields}})
        else:eligible.append(row)
    counts=dict(source_rows=len(candidates),inherited_exclusions=len(original_exclusions),known_used_or_reserved_rows=len(evidence),total_excluded=len(exclusions),eligible_rows=len(eligible),required_scored=1024,excess_after_required=len(eligible)-1024,duplicates=dict(duplicate_counts),input_ineligible=dict(invalid_counts))
    result={'schema':'fresh_mathqa_pool_audit_v1','status':'passed_pool_availability' if len(eligible)>=1024 else 'blocked_insufficient_pool','mode':'CPU input-inventory validation','scope':'All enumerated retained project input histories, including training/probes/qualification/evaluation and reserved inputs; not unknown pretraining/private history or semantic near-duplicates.','source':early['source'],'tokenizer':{'path':str(tokenizer_path),'files':prior['tokenizer']},'normalization':'Unicode NFKC, then Python str.split whitespace collapse joined by one ASCII space, preserving case/punctuation and option order. Raw model prompt unchanged.','source_identity':'canonical JSON of dataset_revision, split, source_id; UTF-8, sorted keys, compact separators','prompt_token_identity':'SHA256 UTF-8 compact JSON integer list; no spaces','prompt_template':prior['model_prompt_rendering'] if 'model_prompt_rendering' in prior else seal51['model_prompt_rendering'],'counts':counts,'role_inventory':role_summary,'historical_coverage':{'pre_Nov4_distinct_input_carriers':ancient['distinct_scanned_files'],'pre_Nov4_duplicate_carriers':ancient['duplicate_files'],'remote_metadata_receipts':len(remote['checked']),'Nov4_Nov16_distinct_carriers':len(inventory['files']),'Nov4_Nov16_duplicate_carriers':len(inventory['duplicates']),'Nov4_Nov16_tasks':early['discovery']['tasks'],'current_later_roots':present,'unknown_later_roots':unknown_roots,'missing_essential_manifests':[],'historic_receipts_reauthenticated':True,'old_raw_files_not_wholly_rescanned':True},'checks':dict(checks),'consumed_files':receipts,'no_model_work':True,'no_remote_operations':True,'no_network':True,'no_deletion':True,'selection_performed':False,'elapsed_s':time.perf_counter()-started}
    write('EXCLUSIONS_PRIVATE.json',{'dataset_revision':REV,'split':'test','normalization':result['normalization'],'rows':exclusions})
    write('ELIGIBLE_INPUTS_PRIVATE.json',{'dataset_revision':REV,'split':'test','normalization':result['normalization'],'requests':eligible,'selection_performed':False})
    write('POOL_AUDIT.json',result)
    print(json.dumps({'status':result['status'],'counts':counts,'checks':sum(checks.values()),'elapsed_s':result['elapsed_s']}))

if __name__=='__main__':main()
