"""Prepare the fixed O/K5--8 inputs locally, without answers or model calls.

Selection uses pinned question columns and retained identities only. These
artifacts precede the executor's source-commit seal and remote availability gates.
No remote access, generation, optimizer step, or outcome-based selection occurs.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import unicodedata

REVISION = '740312add88f781978c0658806c59bc2815b9866'
TOKENIZER_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
SEED = 20261029
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
REMOTE_RUNS = Path('/data/sources/minhao/runs')
REMOTE_ROOT = REMOTE_RUNS / 'dflash_ok_events_5_8_20261029'
NEW_BUDGET = dict(feedback=512, evaluation=1920, generations=2432,
                  presentations=2048, optimizer_steps=1024)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def save(path, value):
    save_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())


def save_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, 'Refusing to replace different prepared evidence: ' + str(path))
    else:
        path.write_bytes(payload)


def select_training(questions, *, previous_count, excluded_questions, count):
    """Continue chronological source rows; never shuffle the feedback stream."""
    require(0 <= previous_count < len(questions) and count > 0, 'Invalid training selection boundary')
    seen = {normalized(text) for text in questions[:previous_count]}
    excluded = {normalized(text) for text in excluded_questions}
    selected, skipped = [], []
    for index in range(previous_count, len(questions)):
        text = normalized(questions[index])
        reasons = []
        if text in excluded:
            reasons.append('earlier_evaluation_normalized_question')
        if text in seen:
            reasons.append('previous_or_selected_training_normalized_duplicate')
        seen.add(text)
        if reasons:
            skipped.append(dict(source_row=index, reasons=reasons))
        else:
            selected.append(index)
        if len(selected) == count:
            return selected, skipped
    raise ValueError('Insufficient next eligible training requests; no substitution or smaller event')


def select_panel(questions, *, excluded_test_ids, excluded_normalized, seed=SEED, count=64):
    """Preserve the additional-panel recipe, including dedup before shuffle."""
    eligible, skipped, seen = [], [], set()
    for index, question in enumerate(questions):
        text = normalized(question)
        reasons = []
        if index in excluded_test_ids:
            reasons.append('earlier_evaluation_source_id')
        if text in excluded_normalized:
            reasons.append('training_or_earlier_evaluation_normalized_question')
        if text in seen:
            reasons.append('duplicate_normalized_test_question')
        seen.add(text)
        if reasons:
            skipped.append(dict(source_row=index, reasons=reasons))
        else:
            eligible.append(index)
    require(len(eligible) >= count, 'Insufficient eligible test requests; do not shrink or replace the panel')
    shuffled = list(eligible)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:count], skipped, eligible


def retained_counts(rows, method, requests):
    result = {}
    for request in requests:
        found = [row for row in rows if row['method'] == method and row['request_id'] == request['request_id']]
        require(len(found) == 3 and {row['repeat'] for row in found} == {0, 1, 2}
                and all(row.get('valid_measurement') is True and not row.get('errors') for row in found),
                'Incomplete/invalid historical measurements: ' + method + '/' + request['request_id'])
        values = {row['verification_calls'] for row in found}
        require(len(values) == 1 and all(type(v) is int and v > 0 for v in values),
                'Historical repeats disagree on call counts')
        result[request['request_id']] = values.pop()
    return result


def bind_source_rows(requests, sources):
    """Historical development original_index is an ordinal, not a source row."""
    bindings = {row['request_id']: row for row in sources}
    require(len(bindings) == len(sources) == len(requests)
            and set(bindings) == {row['request_id'] for row in requests}, 'Source-ID inventory membership differs')
    result = {}
    for row in requests:
        source = bindings[row['request_id']]
        require(digest(row['prompt_ids']) == source['prompt_ids_sha256'], 'Source-ID inventory prompt binding differs')
        index = source['source_row']
        require(type(index) is int and index >= 0, 'Malformed original source row')
        result[row['request_id']] = index
    require(len(set(result.values())) == len(result), 'Earlier evaluation source rows overlap')
    return result


def build_panels(prior_panels, measurements, fresh_requests):
    require([row['name'] for row in prior_panels] == ['development', 'additional'],
            'Use the two latest retained PR16 panels')
    panels = []
    for original in prior_panels:
        name = original['name']
        requests = json.loads(json.dumps(original['requests']))
        counts = {arm: retained_counts(measurements[name], arm, requests) for arm in ('O4', 'K4')}
        expected = {'development': {'O4': 2994, 'K4': 2942}, 'additional': {'O4': 3130, 'K4': 3058}}
        require({arm: sum(values.values()) for arm, values in counts.items()} == expected[name],
                'Retained event4 panel totals differ')
        panels.append(dict(name=name, canonical_mode='retained', requests=requests,
            historical_counts=counts, provenance=original['provenance'],
            historical_counts_source=str(REMOTE_RUNS / 'dflash_krc_events_3_4_20261028' /
                                         'campaign_001/evaluation' / name / 'MEASUREMENTS.jsonl')))
    panels.append(dict(name='fresh', canonical_mode='fresh', requests=fresh_requests,
                       selection_seed=SEED, freshness_scope='Current study only; older exposure disclosed.'))
    return panels


def prepare(research, output, *, remote_root=REMOTE_ROOT):
    """Prepare local artifacts with explicit production paths, never touch A800."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from .campaign import validate_plan

    research, output, remote_root = Path(research).resolve(), Path(output).resolve(), Path(remote_root)
    require(remote_root.is_absolute(), 'Production root must be absolute')
    prepared, sealed = output / 'data_preparation', output / 'sealed'
    prepared.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)
    paths = set()

    def load(path):
        path = Path(path)
        paths.add(path)
        return read(path)

    continuation = research / 'dflash_ok_continuation_20261027'
    latest = research / 'dflash_krc_events_3_4_20261028'
    original_path = research / 'dflash_online_osd_rerun_20261023/sealed/MANIFEST.json'
    original = load(original_path)
    original_seal = load(original_path.with_name('SEAL.json'))
    require(identity(original_path)['sha256'] == original_seal['manifest_sha256'], 'Original stream seal changed')
    require(original['provenance']['dataset_revision'] == REVISION, 'Dataset revision changed')
    train = original['train_requests']
    require(len(train) == 7473 and all(row['dataset_index'] == i and row['request_index'] == i + 1
            and row['request_id'] == f'train_{i:05d}' for i, row in enumerate(train)),
            'Original chronological training identities differ')
    previous = load(continuation / 'data_preparation/CONTINUATION_REQUESTS.json')
    require([row['dataset_index'] for row in previous] == list(range(128, 512)), 'Previous feedback boundary differs')
    event_one = load(research / 'dflash_osd_one_event_20261025/sealed/MANIFEST.json')
    require(train[:128] == event_one['train_requests'], 'Event1 is not the original stream prefix')
    for row in previous:
        require(row['prompt_ids'] == train[row['dataset_index']]['prompt_ids'], 'Old continuation prompt changed')

    prior_plan = load(latest / 'raw/sealed/PLAN.json')
    old_panel_path = research / 'dflash_first_error_fresh_confirmation_20261027/inventory_audit/NEW_PANEL.json'
    old_panel = load(old_panel_path)
    pins = load(research / 'dflash_krc_confirmation_20261028/inventory_audit/SELECTION_AUDIT.json')
    current_inventory_path = research / 'dflash_krc_confirmation_20261028/inventory_audit/CURRENT_STUDY_INVENTORY.json'
    current_inventory = load(current_inventory_path)
    inventory_hashes = [value for name, value in pins['bindings'].items() if Path(name).name == current_inventory_path.name]
    require(inventory_hashes == [identity(current_inventory_path)['sha256']], 'Current source-ID inventory binding changed')
    old_panel_hashes = [value for name, value in current_inventory['evidence_sha256'].items()
                       if 'dflash_first_error_fresh_confirmation_20261027' in name and Path(name).name == 'NEW_PANEL.json']
    require(old_panel_hashes == [identity(old_panel_path)['sha256']], 'Earlier confirmation panel binding changed')
    additional_path = research / 'dflash_krc_confirmation_20261028/inventory_audit/NEW_PANEL.json'
    additional = load(additional_path)
    require(identity(additional_path)['sha256'] == pins['panel_sha256'], 'Retained additional panel changed')
    old_inventory = load(research / 'dflash_first_error_fresh_confirmation_20261027/inventory_audit/INVENTORY_AUDIT.json')
    require(pins['source']['revision'] == REVISION and pins['tokenizer']['revision'] == TOKENIZER_REVISION,
            'Pinned dataset/tokenizer revision differs')
    questions = {}
    for split, source in pins['source']['files'].items():
        path = Path(source['path'])
        require(identity(path)['sha256'] == source['sha256'], 'Pinned question source changed')
        questions[split] = pq.read_table(path, columns=['question']).column('question').to_pylist()
        require(len(questions[split]) == source['rows'], 'Pinned source row count differs')
        paths.add(path)
    require(set(questions) == {'train', 'test'}, 'Require the pinned train/test question columns')

    old_evaluation = [row for panel in prior_plan['panels'] for row in panel['requests']] + old_panel
    previous_rows = {row['request_id']: row['source_row'] for row in current_inventory['groups']['prior_confirmation']}
    require(previous_rows == {row['request_id']: row['original_index'] for row in old_panel},
            'Earlier confirmation source rows differ from the pinned inventory')
    sources = current_inventory['groups']['development'] + [dict(request_id=row['request_id'],
        source_row=previous_rows[row['request_id']], prompt_ids_sha256=digest(row['prompt_ids'])) for row in old_panel]
    sources += [dict(request_id=row['request_id'], source_row=row['original_index'],
                     prompt_ids_sha256=row['prompt_ids_sha256']) for row in additional]
    source_rows = bind_source_rows(old_evaluation, sources)
    prior_test_ids = set(source_rows.values())
    require(len(old_evaluation) == len(prior_test_ids) == 192, 'Require all three distinct earlier64 panels')
    excluded_questions = [questions['test'][index] for index in sorted(prior_test_ids)]
    selected, skipped = select_training(questions['train'], previous_count=512,
                                        excluded_questions=excluded_questions, count=512)
    excluded_normalized = {normalized(text) for text in excluded_questions}
    excluded_normalized.update(normalized(questions['train'][i]) for i in list(range(512)) + selected)
    fresh_ids, panel_skipped, eligible = select_panel(questions['test'],
        excluded_test_ids=prior_test_ids, excluded_normalized=excluded_normalized)

    tokenizer_pin = pins['tokenizer']
    tokenizer_path = Path(tokenizer_pin['path'])
    for name, expected in tokenizer_pin['file_sha256'].items():
        path = tokenizer_path / name
        require(identity(path)['sha256'] == expected, 'Pinned tokenizer file changed: ' + name)
        paths.add(path)
    helper = Path(pins['preprocessing']['template_helper_path'])
    require(identity(helper)['sha256'] == pins['preprocessing']['template_helper_sha256'], 'Author template helper changed')
    paths.add(helper)
    syntax = ast.parse(helper.read_text())
    helpers = ast.Module(body=[node for node in syntax.body if isinstance(node, ast.FunctionDef)
                               and node.name in {'_reasoning_kwargs', 'apply_chat_template'}], type_ignores=[])
    namespace = {}
    exec(compile(helpers, str(helper), 'exec'), namespace)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    stops = load(tokenizer_path / 'generation_config.json')['eos_token_id']
    require(stops == [151645, 151643], 'Qualified stop tokens differ')

    def render(split, index):
        text = questions[split][index] + (SUFFIX if split == 'test' else '')
        rendered = namespace['apply_chat_template'](tokenizer, [dict(role='user', content=text)], 'off')
        return tokenizer.encode(rendered, add_special_tokens=False)

    for i in list(range(512)) + selected:
        require(render('train', i) == train[i]['prompt_ids'], 'Original training token reconstruction differs: ' + str(i))
    for row in old_evaluation:
        require(render('test', source_rows[row['request_id']]) == row['prompt_ids'], 'Earlier panel token reconstruction differs')

    # Exposure is annotation only, after both selections are fixed.
    older_rows = load(research / 'adaflash_reproduction_20260930/online_analysis_001/ONLINE_ROWS.json')['arms']['osd']
    older_done = load(research / 'adaflash_reproduction_20260930/online_analysis_001/receipts/online_reference_001/osd/RESULT.json')
    require(older_done['status'] == 'passed' and older_done['generated_requests'] == 7473
            and older_done['checks']['all7473_chronological_responses'] is True
            and older_done['checks']['inference_completed'] is True and len(older_rows) == 7473,
            'Older complete training exposure cannot be established')
    old_test_exposure = {}
    for source in old_inventory['primary_completed_execution_sets']:
        old_test_exposure.update({i: source['directory'] for i in source['source_rows']})
    for key in ('oct22_additional_completed_source_rows', 'async_final_additional_completed_source_rows'):
        for i in old_inventory[key]:
            old_test_exposure.setdefault(i, key)

    training, identities = [], []
    for ordinal, index in enumerate(selected):
        base, proof = train[index], older_rows[index]
        require(proof['dataset_index'] == index and proof['source_record_id'] == base['dataset_id']
                and proof['prompt_token_ids_sha256'] == digest(base['prompt_ids']) and proof['generation_end_utc'],
                'Older training exposure identity differs')
        require(base['max_new_tokens'] == 2048 and len(base['prompt_ids']) + 2048 <= 2272,
                'Selected input cannot fit unchanged training context; stop without replacement')
        number = 5 + ordinal // 128
        row = dict(base, event_id=f'event_{number:03d}', original_event_id=base['event_id'],
            stop_token_ids=stops, source_id=f'gsm8k:main:train:{index}', source_split='train',
            source_revision=REVISION, continuation_ordinal=ordinal, window_ordinal=ordinal % 128,
            older_project_execution_verified=True)
        training.append(row)
        identities.append(dict(request_id=row['request_id'], source_id=row['source_id'], source_row=index,
            normalized_question_sha256=hashlib.sha256(normalized(questions['train'][index]).encode()).hexdigest(),
            prompt_ids_sha256=digest(row['prompt_ids']), older_execution_evidence_row=index))
    fresh = []
    for ordinal, index in enumerate(fresh_ids):
        tokens = render('test', index)
        require(tokens and len(tokens) + 2048 + 16 <= 4096, 'Fresh input fails unchanged context; no resampling')
        fresh.append(dict(request_id=f'ok58_fresh_eval_{ordinal:05d}', panel_ordinal=ordinal,
            original_index=index, dataset_id=f'gsm8k:main:test:{index}', prompt_ids=tokens,
            stop_token_ids=stops, max_new_tokens=2048, source_revision=REVISION, source_split='test',
            question_sha256=hashlib.sha256(questions['test'][index].encode()).hexdigest(),
            normalized_question_sha256=hashlib.sha256(normalized(questions['test'][index]).encode()).hexdigest(),
            prompt_ids_sha256=digest(tokens), older_project_execution_verified=index in old_test_exposure,
            older_project_execution_evidence=old_test_exposure.get(index)))
    train_prompts = {tuple(row['prompt_ids']) for row in train[:512] + training}
    eval_prompts = {tuple(row['prompt_ids']) for row in old_evaluation + fresh}
    require(len(train_prompts) == 1024 and len(eval_prompts) == 256 and not train_prompts & eval_prompts,
            'Exact training/evaluation prompts overlap or duplicate')
    windows = [dict(event_index=n, event_id=f'event_{n:03d}', request_count=128,
                    requests=training[(n - 5) * 128:(n - 4) * 128]) for n in (5, 6, 7, 8)]
    measurements = {}
    for name in ('development', 'additional'):
        path = latest / 'raw/campaign_001/evaluation' / name / 'MEASUREMENTS.jsonl'
        paths.add(path)
        measurements[name] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    panels = build_panels(prior_plan['panels'], measurements, fresh)

    state_audit = load(continuation / 'raw/FINAL_STATE_AUDIT.json')
    require(state_audit['status'] == 'passed' and not state_audit['errors']
            and not state_audit['final_O_K_rng_mismatches'], 'Retained own O/K state audit failed')
    initial, remote_identities = {}, {}
    for arm in ('O', 'K'):
        check = state_audit['arms'][arm]
        require(check['global_step'] == 512 and check['presentations'] == 1024
                and check['completed_event'] == 4 and check['export_exact'], 'Parent boundary differs')
        remote_receipt = REMOTE_RUNS / 'dflash_ok_continuation_20261027/campaign_001' / arm / 'events/event_004/RESULT.json'
        local_receipt = continuation / 'raw/campaign_001' / arm / 'events/event_004/RESULT.json'
        result = load(local_receipt)
        require(result['full_state_path'] == check['state_path'] and result['export_path'] == check['export_path']
                and result['status'] == 'completed' and result['end_step'] == 512, 'Own parent receipt differs')
        initial[arm] = dict(state=check['state_path'], export=check['export_path'], receipt=str(remote_receipt),
            state_identity=state_audit['files'][check['state_path']])
        for path in (check['state_path'], check['export_path'] + '/config.json', check['export_path'] + '/model.safetensors'):
            remote_identities[path] = state_audit['files'][path]
        remote_identities[str(remote_receipt)] = identity(local_receipt)
    old_plan = load(continuation / 'raw/sealed/PLAN.json')
    for name in ('config.yaml', 'RUNTIME_PIN.json'):
        source = continuation / 'raw/sealed' / name
        require(source.read_bytes() == (latest / 'raw/sealed' / name).read_bytes(), 'Inherited configuration/pin differs')
        paths.update((source, latest / 'raw/sealed' / name))
        save_bytes(sealed / name, source.read_bytes())
    for name in ('TRAINING_PIN.json', 'MODEL_PIN.json'):
        source = latest / 'raw/sealed' / name
        paths.add(source)
        save_bytes(sealed / name, source.read_bytes())
    plan = dict(config=str(remote_root / 'sealed/config.yaml'), windows=windows, panels=panels, initial=initial,
        target=old_plan['target'], draft=old_plan['draft'], worker_python=old_plan['worker_python'],
        stop_step=1024, inherited_horizon=7424, new_budget=NEW_BUDGET, no_automatic_retry=True,
        serving=old_plan['serving'], evaluation_events=[6, 8], shared_feedback_source='O pre-update checkpoint',
        additional_optimizer_qualification_steps=0, generation_warmups=0,
        state_retention=dict(required_events=[6, 8], complete_own_states_and_exports=True,
                            verified_offload_before_retirement=True),
        preparation_scope='Local question-only preparation; execution commit and remote availability sealed separately.')
    plan_check = validate_plan(plan)
    outputs = {'CONTINUATION_REQUESTS.json': training, 'WINDOWS.json': windows, 'NEW_PANEL.json': fresh,
        'SELECTED_IDENTITIES.json': identities,
        'EXCLUSIONS.json': dict(previous_training_rows=list(range(512)), planned_training_rows=selected,
            earlier_test_rows=sorted(prior_test_ids),
            normalized_questions=sorted(excluded_normalized), exact_previous_test_request_ids=[r['request_id'] for r in old_evaluation],
            earlier_request_to_actual_source_row=source_rows,
            development_original_index_scope='Historical manifest ordinal, preserved; actual GSM8K rows come from the bound inventory.')}
    for window in windows:
        outputs[f"WINDOW_{window['event_index']:03d}.json"] = window['requests']
    for name, value in outputs.items():
        save(prepared / name, value)
    save(sealed / 'PLAN.json', plan)
    save(sealed / 'PLAN_CHECK.json', plan_check)
    save(sealed / 'REMOTE_INPUT_IDENTITIES.json', remote_identities)
    paths.add(Path(__file__).resolve())
    audit = dict(schema='ok_events_5_8_preparation_v1', selection_complete=True,
        selected_count=512, selected_dataset_indices=selected, selected_request_ids=[r['request_id'] for r in training],
        skipped_training_candidates=skipped, fresh_seed=SEED, fresh_count=64, fresh_source_rows=fresh_ids,
        eligible_test_count=len(eligible), eligible_test_rows=eligible, excluded_test_candidates=panel_skipped,
        prior_training_count=512, all_training_count=1024, prior_evaluation_count=192,
        original_stream_rows=7473, original_stream_sha256=identity(original_path)['sha256'],
        selection_rule='Chronological next512 excluding earlier evaluation content and training duplicates; then ascending eligible test rows shuffled with isolated seed20261029, first64.',
        freshness_scope='Current O/K study only; older unrelated project exposure is permitted and annotated after selection.',
        global_freshness=False, older_executed_training_count=512,
        older_executed_fresh_count=sum(r['older_project_execution_verified'] for r in fresh),
        answers_loaded=False, outcomes_used_for_selection=False, scientific_generations=0, model_forwards=0,
        retained_training_prompts_reconstructed=1024, retained_evaluation_prompts_reconstructed=192,
        exact_prompt_overlap=0, normalized_question_overlap=0,
        source=pins['source'], tokenizer={**tokenizer_pin,
            'transformers_version': importlib.metadata.version('transformers'),
            'tokenizers_version': importlib.metadata.version('tokenizers')},
        preprocessing=pins['preprocessing'], training_preprocessing='Original unchanged question; no evaluation reasoning suffix.',
        generation_contract=dict(max_new_tokens=2048, stop_token_ids=stops, train_context=2272,
            serving_context=4096, block_size=16, concurrency=1, truncation=False, resampling_after_output=False),
        source_files={str(path): identity(path) for path in sorted(paths)},
        output_sha256={name: identity(prepared / name)['sha256'] for name in outputs},
        plan_sha256_before_execution_binding=identity(sealed / 'PLAN.json')['sha256'],
        remote_checks_pending=True, scientific_acceptance_settings_changed=False)
    save(prepared / 'SELECTION_AUDIT.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research', type=Path, default=Path('/home/minhao2/EAGLE_SPEC/research'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', type=Path, default=REMOTE_ROOT)
    args = parser.parse_args()
    result = prepare(args.research, args.output, remote_root=args.remote_root)
    print(json.dumps({key: result[key] for key in ('selection_complete', 'selected_count', 'fresh_count',
        'eligible_test_count', 'prior_evaluation_count', 'answers_loaded', 'model_forwards')}))
