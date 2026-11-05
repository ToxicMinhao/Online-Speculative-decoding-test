"""Question-only sealing for fixed KW-v1 events9--12; no model executions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from experiments.ok_events_5_8.prepare import (
    REVISION, TOKENIZER_REVISION, SUFFIX, REMOTE_RUNS, require, read, identity,
    digest, normalized, save, save_bytes, select_panel, select_training,
)

SEED = 20261030
REMOTE_ROOT = REMOTE_RUNS / 'dflash_kw_v1_20261029'
ARMS = ('O', 'K', 'KW')
NEW_BUDGET = dict(feedback=512, evaluation=4032, generations=4544,
    presentations=3072, optimizer_steps=1536, optimizer_previews=1024,
    credit_opportunities=504, additional_controller_model_passes=0)
QUALIFICATION_BUDGET = dict(response_forwards=8, response_backwards=8,
    disposable_optimizer_applications=16, serving_generations=0, persistent_optimizer_updates=0)


def previous_source_cursor(audit, previous_requests):
    """Recover the consumed SOURCE boundary, including rejected candidates."""
    indices = audit['selected_dataset_indices']
    require(len(indices) == 512 and indices == sorted(set(indices))
            and indices == [row['dataset_index'] for row in previous_requests],
            'Previous request source identities disagree with selection audit')
    skipped = [row['source_row'] for row in audit['skipped_training_candidates']]
    require(not set(indices) & set(skipped), 'Previous accepted and rejected source rows overlap')
    prior_cursor = audit.get('source_cursor_before', audit['prior_training_count'])
    cursor = indices[-1] + 1
    require(sorted(indices + skipped) == list(range(prior_cursor, cursor)),
            'Previous selection does not account for every consumed source candidate')
    if 'source_cursor_after' in audit:
        require(audit['source_cursor_after'] == cursor, 'Recorded previous source cursor disagrees')
    return cursor


def qualification_rows(features):
    require(len(features) == 128 and len({r['record_id'] for r in features}) == 128,
            'Require all128 retained event8 training features')
    require(all(type(r['length']) is int and r['length'] > 0 for r in features),
            'Qualification selection requires stored positive lengths')
    return [dict(row) for row in sorted(features, key=lambda r: (-r['length'], r['record_id']))[:4]]


def retain_panels(prior_panels, canonical):
    require([p['name'] for p in prior_panels] == ['development', 'additional', 'fresh'],
            'Require exactly the PR18 panel inventory')
    panels = json.loads(json.dumps(prior_panels))
    fresh = panels[2]
    require(set(canonical) == {r['request_id'] for r in fresh['requests']},
            'O8 canonical inventory differs from retained fresh panel')
    for row in fresh['requests']:
        output = canonical[row['request_id']]
        require(output['source_method'] == 'O' and '/event_008/evaluation/fresh/' in output['source_record_path'],
                'Retained fresh canonical must be native O8')
        row.update(response_ids=output['response_ids'], finish_reason=output['finish_reason'],
                   canonical_source_method='O8')
    fresh.update(canonical_mode='retained', provenance='PR18 first native O8 output; exact tokens and termination retained')
    return panels


def prepare(research, output, *, remote_root=REMOTE_ROOT):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from .campaign import validate_plan
    from .evaluation_protocol import method_order
    from .storage import full_campaign_budget

    research, output, remote_root = Path(research).resolve(), Path(output).resolve(), Path(remote_root)
    require(remote_root.is_absolute(), 'Production root must be absolute')
    prepared, sealed = output/'data_preparation', output/'sealed'
    prepared.mkdir(parents=True, exist_ok=True); sealed.mkdir(parents=True, exist_ok=True)
    paths = set()
    def load(path):
        path = Path(path); paths.add(path); return read(path)

    previous = research/'dflash_ok_events_5_8_20261029'
    prior = load(previous/'raw/sealed/PLAN.json')
    audit = load(previous/'data_preparation/SELECTION_AUDIT.json')
    previous_requests_path = previous/'data_preparation/CONTINUATION_REQUESTS.json'
    previous_requests = load(previous_requests_path)
    require(identity(previous_requests_path)['sha256'] == audit['output_sha256']['CONTINUATION_REQUESTS.json'],
            'Previous selected training input seal changed')
    exclusions_path = previous/'data_preparation/EXCLUSIONS.json'
    exclusions = load(exclusions_path)
    require(identity(exclusions_path)['sha256'] == audit['output_sha256']['EXCLUSIONS.json'],
            'Previous exclusion seal changed')
    cursor = previous_source_cursor(audit, previous_requests)
    original_path = research/'dflash_online_osd_rerun_20261023/sealed/MANIFEST.json'
    original = load(original_path)
    require(identity(original_path)['sha256'] == audit['original_stream_sha256'], 'Original source stream seal changed')
    train = original['train_requests']
    require(len(train) == 7473 and all(r['dataset_index'] == i and r['request_id'] == f'train_{i:05d}'
        for i, r in enumerate(train)), 'Original chronological stream identities differ')
    used_indices = exclusions['previous_training_rows'] + exclusions['planned_training_rows']
    require(len(used_indices) == len(set(used_indices)) == 1024
            and exclusions['planned_training_rows'] == audit['selected_dataset_indices'], 'Prior accepted training inventory differs')
    questions = {}
    require(audit['source']['revision'] == REVISION and audit['tokenizer']['revision'] == TOKENIZER_REVISION,
            'Pinned dataset/tokenizer revisions differ')
    for split, source in audit['source']['files'].items():
        path = Path(source['path']); paths.add(path)
        require(identity(path)['sha256'] == source['sha256'], 'Question source changed: ' + split)
        questions[split] = pq.read_table(path, columns=['question']).column('question').to_pylist()
        require(len(questions[split]) == source['rows'], 'Question source size differs')
    # The prior inventory explicitly bound qualification/native probes to these
    # identities. Bind it again instead of assuming other early probes are absent.
    inventory_path = research/'dflash_krc_confirmation_20261028/inventory_audit/CURRENT_STUDY_INVENTORY.json'
    inventory = load(inventory_path)
    require(str(inventory_path) in audit['source_files']
            and identity(inventory_path) == audit['source_files'][str(inventory_path)], 'Current-study probe inventory changed')
    require(not inventory['missing_evidence'], 'Earlier qualification/probe inventory is incomplete')
    old_panel_path = research/'dflash_first_error_fresh_confirmation_20261027/inventory_audit/NEW_PANEL.json'
    old_panel = load(old_panel_path)
    require(identity(old_panel_path) == audit['source_files'][str(old_panel_path)], 'Seed20261027 confirmation panel changed')
    canonical_path = previous/'raw/campaign_001/event_008/evaluation/fresh/CANONICAL.json'
    canonical = load(canonical_path)
    for rid, reference in canonical.items():
        relative = Path(reference['source_record_path']).relative_to(REMOTE_RUNS/'dflash_ok_events_5_8_20261029')
        record = load(previous/'raw'/relative)
        require(record['response_ids'] == reference['response_ids']
                and record['finish_reason'] == reference['finish_reason']
                and record['checkpoint_id'].endswith('/O/events/event_008/export'),
                'Saved O8 canonical differs from its actual native record: '+rid)
    panels = retain_panels(prior['panels'], canonical)
    old_eval = [r for p in panels for r in p['requests']] + old_panel
    source_rows = dict(exclusions['earlier_request_to_actual_source_row'])
    source_rows.update({r['request_id']:r['original_index'] for r in panels[2]['requests']})
    require(len(old_eval) == len(source_rows) == len(set(source_rows.values())) == 256,
            'Require all four prior disjoint evaluation panels including20261027')
    earlier_test_ids = set(source_rows.values())
    excluded_questions = [questions['test'][i] for i in sorted(earlier_test_ids)]
    selected, skipped = select_training(questions['train'], previous_count=cursor,
        excluded_questions=excluded_questions, count=512)
    excluded_normalized = {normalized(q) for q in excluded_questions}
    excluded_normalized.update(normalized(questions['train'][i]) for i in used_indices + selected)
    final_ids, final_skips, eligible = select_panel(questions['test'], excluded_test_ids=earlier_test_ids,
        excluded_normalized=excluded_normalized, seed=SEED)

    pin = audit['tokenizer']; tokenizer_path = Path(pin['path'])
    for name, expected in pin['file_sha256'].items():
        path = tokenizer_path/name; paths.add(path)
        require(identity(path)['sha256'] == expected, 'Pinned tokenizer changed: ' + name)
    helper = Path(audit['preprocessing']['template_helper_path']); paths.add(helper)
    require(identity(helper)['sha256'] == audit['preprocessing']['template_helper_sha256'], 'Native template helper changed')
    syntax = ast.parse(helper.read_text())
    functions = ast.Module(body=[n for n in syntax.body if isinstance(n, ast.FunctionDef)
        and n.name in {'_reasoning_kwargs','apply_chat_template'}], type_ignores=[])
    namespace = {}; exec(compile(functions, str(helper), 'exec'), namespace)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    stops = load(tokenizer_path/'generation_config.json')['eos_token_id']
    require(stops == [151645,151643], 'Pinned stopping tokens differ')
    def render(split, index):
        text = questions[split][index] + (SUFFIX if split == 'test' else '')
        return tokenizer.encode(namespace['apply_chat_template'](tokenizer,
            [dict(role='user',content=text)], 'off'), add_special_tokens=False)
    for i in used_indices + selected:
        require(render('train',i) == train[i]['prompt_ids'], 'Training token reconstruction differs: ' + str(i))
    for row in old_eval:
        require(render('test',source_rows[row['request_id']]) == row['prompt_ids'], 'Retained evaluation tokens differ')
    # Exposure annotations are applied only after selection; never inspect outcomes.
    older_rows = load(research/'adaflash_reproduction_20260930/online_analysis_001/ONLINE_ROWS.json')['arms']['osd']
    older_done = load(research/'adaflash_reproduction_20260930/online_analysis_001/receipts/online_reference_001/osd/RESULT.json')
    require(older_done['status'] == 'passed' and older_done['generated_requests'] == len(older_rows) == 7473
            and older_done['checks']['all7473_chronological_responses'] is True
            and older_done['checks']['inference_completed'] is True, 'Older training exposure evidence incomplete')
    old_inventory = load(research/'dflash_first_error_fresh_confirmation_20261027/inventory_audit/INVENTORY_AUDIT.json')
    old_test_exposure = {}
    for source in old_inventory['primary_completed_execution_sets']:
        old_test_exposure.update({i:source['directory'] for i in source['source_rows']})
    for key in ('oct22_additional_completed_source_rows','async_final_additional_completed_source_rows'):
        for i in old_inventory[key]: old_test_exposure.setdefault(i,key)
    training, identities = [], []
    for ordinal,index in enumerate(selected):
        base, proof = train[index], older_rows[index]
        require(proof['dataset_index'] == index and proof['source_record_id'] == base['dataset_id']
                and proof['prompt_token_ids_sha256'] == digest(base['prompt_ids']) and proof['generation_end_utc'],
                'Older training exposure identity differs')
        require(base['max_new_tokens'] == 2048 and len(base['prompt_ids'])+2048 <= 2272,
                'Selected training input cannot fit unchanged; stop without substitution')
        number = 9 + ordinal//128
        row = dict(base, event_id=f'event_{number:03d}', original_event_id=base['event_id'],
            stop_token_ids=stops, source_id=f'gsm8k:main:train:{index}', source_split='train',
            source_revision=REVISION, continuation_ordinal=ordinal, window_ordinal=ordinal%128,
            older_project_execution_verified=True)
        training.append(row)
        identities.append(dict(request_id=row['request_id'],source_row=index,source_id=row['source_id'],
            normalized_question_sha256=hashlib.sha256(normalized(questions['train'][index]).encode()).hexdigest(),
            prompt_ids_sha256=digest(row['prompt_ids'])))
    final = []
    for ordinal,index in enumerate(final_ids):
        tokens = render('test',index)
        require(tokens and len(tokens)+2048+16 <= 4096, 'Final input overlength; no replacement')
        final.append(dict(request_id=f'kw_final_eval_{ordinal:05d}',panel_ordinal=ordinal,original_index=index,
            dataset_id=f'gsm8k:main:test:{index}',prompt_ids=tokens,stop_token_ids=stops,max_new_tokens=2048,
            source_revision=REVISION,source_split='test',
            question_sha256=hashlib.sha256(questions['test'][index].encode()).hexdigest(),
            normalized_question_sha256=hashlib.sha256(normalized(questions['test'][index]).encode()).hexdigest(),
            prompt_ids_sha256=digest(tokens),older_project_execution_verified=index in old_test_exposure,
            older_project_execution_evidence=old_test_exposure.get(index)))
    panels.append(dict(name='final',canonical_mode='fresh',requests=final,selection_seed=SEED,
        freshness_scope='Current study fresh; known unrelated older-project exposure disclosed after selection.'))
    train_prompts = {tuple(train[i]['prompt_ids']) for i in used_indices+selected}
    evaluation_prompts = {tuple(r['prompt_ids']) for r in old_eval+final}
    require(len(train_prompts)==1536 and len(evaluation_prompts)==320 and not train_prompts & evaluation_prompts,
            'Exact training/evaluation content duplicates or overlaps')
    windows = [dict(event_index=n,event_id=f'event_{n:03d}',request_count=128,
        requests=training[(n-9)*128:(n-8)*128]) for n in (9,10,11,12)]
    initial, remote_ids, initial_receipts = {}, {}, {}
    for arm in ('O','K'):
        result_path = previous/f'raw/campaign_001/{arm}/events/event_008/RESULT.json'
        result = load(result_path); initial_receipts[arm] = result
        require(result['status']=='completed' and result['start_step']==896 and result['end_step']==1024,
                'Original own event8 receipt differs')
        identities_path = previous/f'raw/campaign_001/resume_008_{arm}/STATE_IDENTITIES.json'
        bindings = load(identities_path)
        source = dict(state=result['full_state_path'],export=result['export_path'],
            receipt=str(REMOTE_RUNS/f'dflash_ok_events_5_8_20261029/campaign_001/{arm}/events/event_008/RESULT.json'),
            state_identity=bindings[result['full_state_path']])
        initial[arm]=source; remote_ids.update(bindings);remote_ids[source['receipt']]=identity(result_path)
    initial['KW'] = dict(initial['K'], migration='exact_complete_K8_to_KW_v1')
    require(initial_receipts['O']['features']==initial_receipts['K']['features'], 'O8/K8 immutable feature inventories differ')
    qualification = dict(selection_rule='descending stored length; record_id ascending ties',
        rows=qualification_rows(initial_receipts['K']['features']), budgets=QUALIFICATION_BUDGET,
        source_receipt=initial['K']['receipt'], source_state=initial['K']['state'],source_state_identity=initial['K']['state_identity'])
    for name in ('config.yaml','RUNTIME_PIN.json','MODEL_PIN.json','TRAINING_PIN.json'):
        source = previous/'raw/sealed'/name;paths.add(source);save_bytes(sealed/name,source.read_bytes())
    schedule = [dict(event=n,panel=p['name'],request_id=r['request_id'],ordinal=i,repeat=repeat,
        order=list(method_order(i,repeat))) for n in (10,12) for p in panels
        if n==12 or p['name']!='final' for i,r in enumerate(p['requests']) for repeat in range(3)]
    require(sum(len(r['order']) for r in schedule)==4032,'Evaluation schedule budget differs')
    plan = dict(config=str(remote_root/'sealed/config.yaml'),windows=windows,panels=panels,initial=initial,
        target=prior['target'],draft=prior['draft'],worker_python=prior['worker_python'],stop_step=1536,
        inherited_horizon=7424,new_budget=NEW_BUDGET,qualification_budget=QUALIFICATION_BUDGET,
        no_automatic_retry=True,serving=prior['serving'],evaluation_events=[10,12],
        shared_feedback_source='preceding O8,O9,O10,O11 checkpoint',generation_warmups=0,
        source_cursor_before=cursor,source_cursor_after=selected[-1]+1,
        state_retention=dict(required_events=[10,12],rolling_events=[9,11],preserve_original_events=[8],
            complete_own_states_and_exports=True,all_feedback_features=True),
        qualification=str(remote_root/'sealed/QUALIFICATION_INPUTS.json'),
        evaluation_schedule_sha256=digest(schedule),
        storage=dict(reserve_bytes=8*1024**3,mode='retain_local',offload_assumed=False))
    plan['storage']['campaign_capacity']=full_campaign_budget(plan)
    plan_check=validate_plan(plan)
    outputs={'CONTINUATION_REQUESTS.json':training,'WINDOWS.json':windows,'NEW_PANEL.json':final,
        'SELECTED_IDENTITIES.json':identities,'EVALUATION_SCHEDULE.json':schedule,
        'EXCLUSIONS.json':dict(previous_training_rows=used_indices,planned_training_rows=selected,
            earlier_test_rows=sorted(earlier_test_ids),normalized_questions=sorted(excluded_normalized),
            earlier_request_to_actual_source_row=source_rows,
            exact_previous_test_request_ids=[r['request_id'] for r in old_eval],
            qualification_and_probe_inventory=str(inventory_path))}
    for window in windows: outputs[f"WINDOW_{window['event_index']:03d}.json"]=window['requests']
    for name,value in outputs.items(): save(prepared/name,value)
    save(sealed/'PLAN.json',plan);save(sealed/'PLAN_CHECK.json',plan_check)
    save(sealed/'REMOTE_INPUT_IDENTITIES.json',remote_ids);save(sealed/'QUALIFICATION_INPUTS.json',qualification)
    paths.add(Path(__file__).resolve())
    result=dict(schema='kw_v1_preparation_v1',selection_complete=True,selected_count=512,
        selected_dataset_indices=selected,selected_request_ids=[r['request_id'] for r in training],
        source_cursor_before=cursor,source_cursor_after=selected[-1]+1,skipped_training_candidates=skipped,
        prior_training_count=1024,all_training_count=1536,prior_evaluation_count=256,final_seed=SEED,
        final_count=64,final_source_rows=final_ids,eligible_test_rows=eligible,excluded_test_candidates=final_skips,
        older_executed_training_count=512,older_executed_final_count=sum(r['older_project_execution_verified'] for r in final),
        global_freshness=False,answers_loaded=False,outcomes_used_for_selection=False,scientific_generations=0,
        model_forwards=0,exact_prompt_overlap=0,normalized_question_overlap=0,source=audit['source'],
        tokenizer=pin,preprocessing=audit['preprocessing'],source_files={str(p):identity(p) for p in sorted(paths)},
        output_sha256={name:identity(prepared/name)['sha256'] for name in outputs},
        original_stream_sha256=identity(original_path)['sha256'],remote_checks_pending=True)
    save(prepared/'SELECTION_AUDIT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research',type=Path,default=Path('/home/minhao2/EAGLE_SPEC/research'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--remote-root',type=Path,default=REMOTE_ROOT)
    args=parser.parse_args(); result=prepare(args.research,args.output,remote_root=args.remote_root)
    print(json.dumps({k:result[k] for k in ('selection_complete','selected_count','source_cursor_before','source_cursor_after','final_count','answers_loaded')}))
