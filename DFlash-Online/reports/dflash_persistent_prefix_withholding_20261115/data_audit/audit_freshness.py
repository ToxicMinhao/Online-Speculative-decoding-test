"""CPU-only proof of the fresh-panel upper bound; no selection or model work."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

import pyarrow.parquet as pq

ROOT = Path('/home/minhao2/EAGLE_SPEC/research')
OUT = Path(__file__).resolve().parent
REVISION = '740312add88f781978c0658806c59bc2815b9866'
evidence = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    evidence[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return json.loads(path.read_text())


def rows(path):
    path = Path(path)
    evidence[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def main():
    source = read(ROOT / 'projected_target_feasibility_20261022/data/approved/manifest.json')
    mapping, questions, source_details = {}, {}, {}
    for split in ('train', 'test'):
        meta = source['sources']['gsm8k_' + split]
        path = Path(meta['local_path'])
        assert meta['revision'] == REVISION and sha(path) == meta['sha256']
        evidence[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
        # Only the source question column enters the audit.
        question = [r['question'] for r in pq.read_table(path, columns=['question']).to_pylist()]
        lookup = {normalize(text): i for i, text in enumerate(question)}
        assert len(question) == len(lookup) == meta['rows']
        native_path = ROOT / f'dflash_paid_learning_20261004/data_audit_sources/gsm8k_{split}.jsonl'
        native = rows(native_path)
        aliases = {}
        for row in native:
            user = [m['content'] for m in row['conversations'] if m['role'] == 'user']
            assert len(user) == 1
            aliases[row['id']] = lookup[normalize(user[0])]
        assert len(aliases) == len(question) and set(aliases.values()) == set(range(len(question)))
        mapping[split], questions[split] = aliases, question
        source_details[split] = dict(rows=len(question), source=meta, opaque_mapping_bijective=True)

    completed = set()
    per_source = []
    first = ROOT / 'adaflash_reproduction_20260930/runs/serving_c1_reference_001/gpu0_ar'
    second = ROOT / 'dflash_objective_check_20261005/retrieved_001/run_001/evaluation/code_s0_buffered_gsm8k'
    original_serving_prompts = {}
    for directory, filename in [(first, 'REQUESTS.jsonl'), (second, 'EVALUATION.jsonl')]:
        manifest = read(directory / 'REQUEST_MANIFEST.json')
        result = read(directory / 'RESULT.json')
        actual_rows = rows(directory / filename)
        assert result['status'] == 'passed'
        if filename == 'REQUESTS.jsonl':
            assert result['benchmark_returncode'] == 0
            expected = {r['dataset_id'] for r in manifest['requests'] if r['kind'] == 'timed'}
            actual = [r['dataset_id'] for r in actual_rows if r['kind'] == 'timed']
            assert len(actual) == len(expected) == 1024
            original_serving_prompts = {r['dataset_id']: r['rendered_prompt_token_ids']
                for r in manifest['requests'] if r['kind'] == 'timed'}
        else:
            assert result['count'] == result['completed'] == 276
            assert sha(directory / filename) == result['output_sha256']
            expected, actual = set(manifest['ids']), [r['dataset_id'] for r in actual_rows]
            assert len(actual) == len(expected) == 276
        assert set(actual) == expected
        indices = {mapping['test'][x] for x in actual}
        per_source.append(dict(source=str(directory), completed_unique=len(indices),
            additional=len(indices - completed), source_rows=sorted(indices)))
        completed.update(indices)
    assert len(completed) == 1300

    oct22_rows = [r for g in source['groups'] for r in g['requests'] if r['source_key'] == 'gsm8k_test']
    added = set()
    for row in oct22_rows:
        path = ROOT / f"projected_target_feasibility_20261022/retrieved/final_recovery/campaign_002/gsm8k/F/records/{row['request_id']}.json"
        actual = read(path)
        assert actual['record_id'] == row['request_id'] and actual['prompt_ids'] == row['prompt_ids']
        assert actual['response_ids'] and actual['finish_reason']
        assert normalize(row['text']) == normalize(questions['test'][row['source_index']])
        added.add(row['source_index'])
    assert len(added) == 8
    per_source.append(dict(source='projected_target_feasibility_20261022/gsm8k/F',
        completed_unique=8, additional=len(added-completed), source_rows=sorted(added)))
    completed.update(added)

    async_root = ROOT / 'dflash_async_passes_20261006/recovery_001/retrieved_001/run_001/final_one_pass'
    result = read(async_root / 'RESULT.json')
    receipt = read(async_root / 'inference.completed.json')
    assert result['status'] == 'passed' and receipt['status'] == 'completed' and receipt['completed'] == 256
    expected = {r['source_record_id'] for r in rows(async_root / 'request_manifest.jsonl')}
    actual = [r['source_record_id'] for r in rows(async_root / 'responses.jsonl')]
    assert len(actual) == len(expected) == 256 and set(actual) == expected
    indices = {mapping['test'][x] for x in actual}
    per_source.append(dict(source=str(async_root), completed_unique=len(indices),
        additional=len(indices-completed), source_rows=sorted(indices)))
    completed.update(indices)
    assert len(completed) == 1311

    # Current inherited native campaign uses this exact revision and test split.
    full_path = ROOT / 'dflash_full_k_gsm8k_20261031/production_002/sealed/ORIGINAL_MANIFEST.json'
    native = read(full_path)
    assert native['provenance']['dataset_revision'] == REVISION
    assert len(native['eval_requests']) == 1024
    assert all(r['dataset_id'] in mapping['test'] for r in native['eval_requests'])
    assert all(r['prompt_ids'] == original_serving_prompts[r['dataset_id']] for r in native['eval_requests'])
    train = native['train_requests']
    assert len(train) == 7473 and all(r['dataset_id'] in mapping['train'] for r in train)
    through7, event8 = train[:896], train[896:1024]
    def keys(data):
        return ({r['dataset_id'] for r in data},
                {normalize(questions['train'][mapping['train'][r['dataset_id']]]) for r in data},
                {token_hash(r['prompt_ids']) for r in data})
    old_keys, new_keys = keys(through7), keys(event8)
    assert len(event8) == 128 and all(len(k) == 128 for k in new_keys)
    assert all(not (a & b) for a, b in zip(old_keys, new_keys))
    assert all(r['event_id'] == 'event_008' for r in event8)
    features_path = ROOT / 'dflash_lost_repair_20261114/delivery/retained/campaign_001/census/FEATURES.json'
    schedule_path = ROOT / 'dflash_lost_repair_20261114/delivery/retained/campaign_001/training/SCHEDULE.json'
    features, schedule = read(features_path), read(schedule_path)
    current_plan = read(ROOT / 'dflash_recent_repair_retention_event7_20261112/delivery/final_001/retained/TRAINING_PLAN.json')
    assert sha(features_path) == current_plan['input_identities'][current_plan['feature_manifest']]['sha256']
    assert sha(schedule_path) == current_plan['input_identities'][current_plan['schedule']]['sha256']
    event7_ids = {r['request_id'] for r in train[768:896]}
    assert {Path(p).parent.name for p in features['feature_paths']} == event7_ids
    assert Counter(r['record_id'] for r in schedule['records']) == Counter({k: 2 for k in event7_ids})

    previous = read(ROOT / 'dflash_first_error_fresh_confirmation_20261027/inventory_audit/INVENTORY_AUDIT.json')
    for path, identity in evidence.items():
        if path in previous['evidence_sha256']:
            assert identity['sha256'] == previous['evidence_sha256'][path]
    remaining = sorted(set(range(len(questions['test']))) - completed)
    result = dict(status='FAIL_FRESH_PANEL_INSUFFICIENT', requested=128,
        pinned_test_rows=1319, proven_completed_unique_test_requests=len(completed),
        maximum_possible_fresh_requests=len(remaining), minimum_shortfall=128-len(remaining),
        remaining_rows_not_certified_fresh=remaining, sources=source_details,
        completed_use_evidence=per_source,
        current_native_binding=dict(original_manifest=str(full_path), revision=REVISION,
            evaluation_split='test', matched_native_serving_prompt_sequences=1024,
            match_scope='Each retained current native eval prompt equals the earlier completed serving prompt token IDs exactly.'),
        event8=dict(input_range=[896,1023], count=128, ids_text_tokens_disjoint_from_first896=True,
            event7_feature_and_schedule_identity_passed=True,
            event7_current_input_request_ids_match_native_stream=True,
            scope='Input-only retained-manifest validation; live recovery ancestry and remote hashes remain separate gates.',
            older_project_exposure='These requests exist in the original full58-event stream; no globally-new training claim.'),
        interpretation='An upper bound of8, not a claim that these8 remain fresh. Further exclusions, aliases or context checks cannot raise this bound. No panel was selected.',
        scope_limits=['Bounded completed-use proof sufficient to fail128-request gate; not an exhaustive census of every later project use.',
            'No new tokenizer rendering or candidate tokenization is required to prove the cardinality shortfall;1024 inherited exact token identities were independently checked.',
            'No benchmark answer column used, no model, no GPU, no serving, no training, no network.'],
        files=evidence)
    (OUT / 'FRESHNESS_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT / 'EXCLUSION_LOWER_BOUND.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['source_id','source_row','normalized_question_sha256','completed_evidence_sources'])
        writer.writeheader()
        for idx in sorted(completed):
            writer.writerow(dict(source_id=f'gsm8k:main:test:{idx}',source_row=idx,
                normalized_question_sha256=hashlib.sha256(normalize(questions['test'][idx]).encode()).hexdigest(),
                completed_evidence_sources=';'.join(s['source'] for s in per_source if idx in s['source_rows'])))
    (OUT / 'EVENT8_INPUT_IDENTITIES.json').write_text(json.dumps([
        dict(request_id=r['request_id'],native_source_id=r['dataset_id'],
             source_id=f"gsm8k:main:train:{mapping['train'][r['dataset_id']]}",
             native_stream_index=r['dataset_index'],prompt_ids_sha256=token_hash(r['prompt_ids']),
             normalized_question_sha256=hashlib.sha256(normalize(questions['train'][mapping['train'][r['dataset_id']]]).encode()).hexdigest())
        for r in event8],indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','requested','pinned_test_rows','proven_completed_unique_test_requests','maximum_possible_fresh_requests','minimum_shortfall','remaining_rows_not_certified_fresh','event8']},indent=2))


if __name__ == '__main__':
    main()
