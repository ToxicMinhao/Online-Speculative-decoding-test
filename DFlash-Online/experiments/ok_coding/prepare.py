"""CPU-only source, history, prompt and schedule seal for the fixed coding study.

Only the parquet's instruction/input columns are read. Historical source
catalogs resolve identities; only retained allocated/selected inputs exclude
requests. No model weights or historical performance are used.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata


SOURCE_KEY = 'codealpaca_train'
SOURCE_REVISION = '3bd2de8099ee8a140653ad9b9df783554d5efabb'
SOURCE_SHA256 = '5a2fa0c510d42843292c583f357086abb59f95796e0f55727b97e449aa93bed9'
TOKENIZER_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
PRIOR_AUDIT_SHA256 = '3f8c96f9c0710d0bcf611fea7bc2fe02bdd0d5e6e2ea21c09a233c79f1474cb5'
SOURCE_ROWS = 20022
SOURCE_BYTES = 3339837
EVALUATION_RESERVOIR = 1024
PROMPT_CAP = 224
REQUESTS = {'training': 512, 'evaluation': 128}
SELECTION_SEEDS = {'training': 20261030, 'evaluation': 20261031}
DEFAULT_WORKSPACE = Path('/home/minhao2/EAGLE_SPEC')
OLD_DATA = Path('research/dflash_baseline_learning_curve_20261020/execution_001/data')
PRIOR_AUDIT = Path('research/projected_target_feasibility_20261022/data/approved/prior_exclusion_audit.json')
LOOKUP_CATALOG = Path('research/dflash_data_scaling_20260917/source_cache/codealpaca_prompt_only.jsonl')
HISTORY_MARKERS = ('manifest', 'panel', 'select', 'request', 'prompt', 'probe', 'reserv', 'window', 'continuation', 'plan')
NONALLOCATION_MARKERS = ('exclusion', 'source', 'audit', 'retrieval', 'retirement', 'archive', 'export')
IGNORED_HISTORY_FIELDS = {
    'output', 'answer', 'response', 'completion', 'reference_answer', 'generated_text',
    'output_text', 'response_text', 'assistant', 'metrics', 'results', 'exclusions',
    'rejections', 'candidate_pools', 'sources', 'source_catalog', 'history_files',
}
PROMPT_FIELDS = {'text', 'prompt', 'content', 'value', 'user', 'user_text', 'question', 'rendered_prompt', 'prompt_text'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def canonical_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def assemble(row):
    instruction, context = row.get('instruction'), row.get('input')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('instruction must be a nonempty string')
    if not isinstance(context, str):
        raise ValueError('input must be a string')
    return instruction + ('\n\n' + context if context else '')


def load_source(workspace):
    import pyarrow.parquet as pq

    old = workspace / OLD_DATA
    paths = load(old / 'source_paths.json')
    source = Path(paths[SOURCE_KEY])
    receipt = load(old / 'codealpaca_train_ACQUISITION.json')
    binding = load(old / 'prepared/source_bindings.json')[SOURCE_KEY]
    expected = {'bytes': SOURCE_BYTES, 'sha256': SOURCE_SHA256}
    actual = identity(source)
    for field, value in expected.items():
        if actual[field] != value or receipt[field] != value or binding[field] != value:
            raise ValueError('Retained CodeAlpaca source identity differs: ' + field)
    if (receipt['source_revision'] != SOURCE_REVISION or binding['revision'] != SOURCE_REVISION
            or binding['repository_path'] != 'default/train/0000.parquet'
            or binding['dataset'] != 'sahil2801/CodeAlpaca-20k'):
        raise ValueError('Source repository/revision binding differs')
    rows = pq.read_table(source, columns=['instruction', 'input']).to_pylist()
    if len(rows) != SOURCE_ROWS or binding['rows'] != SOURCE_ROWS:
        raise ValueError('Original row count differs')
    catalog_path = workspace / LOOKUP_CATALOG
    catalog = [json.loads(line) for line in catalog_path.read_text().splitlines()]
    if len(catalog) != len(rows):
        raise ValueError('Legacy source-ID lookup has a different row count')
    for index, (row, old_row) in enumerate(zip(rows, catalog)):
        if old_row['id'] != f'codealpaca:{index}' or old_row['prompt'] != assemble(row):
            raise ValueError('Legacy source-ID lookup order/text differs at ' + str(index))
    provenance = {
        **actual, 'dataset': binding['dataset'], 'revision': SOURCE_REVISION,
        'repository_path': binding['repository_path'], 'rows': len(rows),
        'columns_read': ['instruction', 'input'], 'source_answers_read_or_used': False,
        'partition_boundary': len(rows) - EVALUATION_RESERVOIR,
        'source_receipts': [identity(old / name) for name in
                            ('source_paths.json', 'codealpaca_train_ACQUISITION.json', 'prepared/source_bindings.json')],
        'lookup_only_catalog': identity(catalog_path),
        'legacy_catalog_exact_order_and_prompt_matches': len(rows),
        'partition_scope': 'Project raw-row partition before all filtering, not an official train/test split.',
    }
    return rows, provenance


def history_candidates(workspace):
    """Inspect selection carriers, not catalogs, raw outputs or performance."""
    ignored_directories = {'github_checkout', 'provenance', 'source_cache', 'source_artifacts',
                           'preparation_dependencies', 'dflash_ok_coding_20261030'}
    for path in sorted((workspace / 'research').rglob('*')):
        if path.suffix not in ('.json', '.jsonl') or ignored_directories.intersection(path.parts):
            continue
        name = path.name.lower()
        if any(marker in name for marker in HISTORY_MARKERS) and not any(
                marker in name for marker in NONALLOCATION_MARKERS):
            yield path


def history_matches(value, normalized_rows):
    found = defaultdict(set)

    def scan(item, field=''):
        if isinstance(item, dict):
            if item.get('role', item.get('from')) in ('assistant', 'gpt', 'chatgpt', 'bing', 'bard'):
                return
            if item.get('source_key') == SOURCE_KEY and type(item.get('source_index')) is int:
                found[item['source_index']].add('source_id')
            for name, content in item.items():
                if name not in IGNORED_HISTORY_FIELDS:
                    scan(content, name)
        elif isinstance(item, list):
            for content in item:
                if not isinstance(content, (int, float, bool)):
                    scan(content, field)
        elif isinstance(item, str):
            match = re.fullmatch(r'(?:codealpaca_train|codealpaca):(\d+)', item)
            if match:
                found[int(match[1])].add('source_id')
            if field in PROMPT_FIELDS:
                variants = [item, *re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', item, re.S)]
                for text in variants:
                    for index in normalized_rows.get(normalize(text), ()):
                        found[index].add('normalized_user_prompt')

    scan(value)
    return found


def inventory_history(workspace, source_rows):
    prior_path = workspace / PRIOR_AUDIT
    if digest(prior_path) != PRIOR_AUDIT_SHA256:
        raise ValueError('Retained prior-exclusion inventory changed')
    prior = load(prior_path)
    normalized_rows = defaultdict(set)
    for index, row in enumerate(source_rows):
        normalized_rows[normalize(assemble(row))].add(index)
    evidence = defaultdict(list)
    for row in prior['excluded_source_rows']:
        if row['source_key'] == SOURCE_KEY:
            evidence[row['source_index']].append({'path': row['first_evidence'], 'reason': 'retained_prior_allocation'})
    prior_count = len(evidence)
    # The existing conservative inventory includes a separately sealed 2,000-row
    # reserved candidate pool. Its 20,022-row source catalog is lookup-only.
    first_evidence = sorted({item['path'] for items in evidence.values() for item in items})
    prior_evidence = [identity(path) for path in first_evidence]
    receipts, seen = [], {}
    scanned_files = 0
    for path in history_candidates(workspace):
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        scanned_files += 1
        if sha in seen:
            continue
        seen[sha] = str(path)
        if path.suffix == '.jsonl':
            contents = [json.loads(line) for line in data.splitlines()]
        else:
            contents = json.loads(data)
        matched = history_matches(contents, normalized_rows)
        receipt = {'path': str(path), 'bytes': len(data), 'sha256': sha,
                   'matched_source_rows': sorted(matched)}
        receipts.append(receipt)
        for index, reasons in matched.items():
            if not 0 <= index < len(source_rows):
                raise ValueError('Historical source index outside the retained corpus')
            evidence[index].append({'path': str(path), 'sha256': sha, 'reason': '+'.join(sorted(reasons))})
        if len(receipts) % 5000 == 0:
            print(f'Inspected {len(receipts)} distinct history files; excluded rows {len(evidence)}', flush=True)
    return dict(evidence), {
        'retained_prior_audit': identity(prior_path), 'prior_excluded_source_rows': prior_count,
        'prior_first_evidence': prior_evidence, 'scanned_paths': scanned_files,
        'distinct_scanned_files': len(receipts), 'scan_receipts': receipts,
        'final_excluded_source_rows': len(evidence), 'additional_source_rows': len(evidence) - prior_count,
        'allocation_rule': 'Retained allocated/selected/reserved inputs, including unexecuted reservations. Source catalogs only resolve IDs.',
        'scan_rule': {'filename_markers': list(HISTORY_MARKERS), 'excluded_filename_markers': list(NONALLOCATION_MARKERS),
                      'ignored_fields': sorted(IGNORED_HISTORY_FIELDS)},
        'limits': 'Inventoried retained local project history only; not inaccessible history, pretraining or semantic-near-duplicate absence.',
    }


def load_renderer(tokenizer_path, author_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tree = ast.parse(author_path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('_reasoning_kwargs', 'apply_chat_template')]
    if len(functions) != 2:
        raise ValueError('Author template helper functions unavailable')
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(author_path), 'exec'), namespace)
    author_render = namespace['apply_chat_template']

    def render(text):
        messages = [{'role': 'user', 'content': text}]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False)
        ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
        original = author_render(tokenizer, messages, 'off')
        if list(ids) != tokenizer.encode(original, add_special_tokens=False):
            raise ValueError('Direct tokenizer output differs from author template helper')
        if not all(type(token) is int and token >= 0 for token in ids):
            raise ValueError('Malformed tokenizer IDs')
        return list(ids)

    stops = load(tokenizer_path / 'generation_config.json')['eos_token_id']
    if stops != [151645, 151643]:
        raise ValueError('Qualified stop-token identities differ')
    bindings = {'model': 'Qwen/Qwen3-8B', 'revision': TOKENIZER_REVISION,
                'path': str(tokenizer_path), 'stop_token_ids': stops,
                'template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                'files': [identity(tokenizer_path / name) for name in
                          ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'generation_config.json')],
                'author_helper_source': identity(author_path),
                'template': {'explicit_system_message': None, 'add_generation_prompt': True, 'enable_thinking': False},
                'normalization_for_matching_only': 'NFC then whitespace collapse; actual prompt preserved exactly.'}
    return render, stops, bindings


def prepare_population(source_rows, historical, render, *, evaluation_reservoir=1024):
    boundary = len(source_rows) - evaluation_reservoir
    if boundary <= 0:
        raise ValueError('Source too small for the fixed raw-row partition')
    text, keys, invalid = {}, {}, {}
    for index, row in enumerate(source_rows):
        try:
            text[index] = assemble(row)
            keys[index] = normalize(text[index])
        except ValueError as error:
            invalid[index] = str(error)
    reservoirs = {'training': range(boundary), 'evaluation': range(boundary, len(source_rows))}
    key_sets = {role: {keys[index] for index in indices if index in keys} for role, indices in reservoirs.items()}
    across = key_sets['training'] & key_sets['evaluation']
    historical_keys = {keys[index] for index in historical if index in keys}
    eligible, exclusions, counts = {}, [], {}
    for role, indices in reservoirs.items():
        first, selected, reasons = {}, [], Counter()
        for index in indices:
            cause, detail, ids = [], {}, None
            if index in invalid:
                cause.append('invalid_instruction_or_input')
                detail['detail'] = invalid[index]
            else:
                key = keys[index]
                if key in first:
                    cause.append('duplicate_within_reservoir')
                    detail['lowest_source_index'] = first[key]
                else:
                    first[key] = index
                if key in across:
                    cause.append('duplicate_across_reservoirs')
                if index in historical:
                    cause.append('historical_source_id')
                    detail['evidence'] = historical[index]
                elif key in historical_keys:
                    cause.append('historical_normalized_prompt')
                if not cause:
                    ids = render(text[index])
                    if not 1 <= len(ids) <= PROMPT_CAP:
                        cause.append('rendered_prompt_outside_1_224')
                        detail['prompt_tokens'] = len(ids)
            if cause:
                reasons.update(cause)
                exclusions.append({'source_index': index, 'source_id': f'{SOURCE_KEY}:{index}',
                                   'reservoir': role, 'reasons': cause, **detail})
            else:
                selected.append({'source_index': index, 'source_id': f'{SOURCE_KEY}:{index}',
                                 'text': text[index], 'prompt_ids': ids, 'prompt_tokens': len(ids),
                                 'normalized_prompt_sha256': hashlib.sha256(keys[index].encode()).hexdigest()})
        eligible[role] = selected
        counts[role] = {'raw_rows': len(indices), 'eligible_rows': len(selected),
                        'excluded_rows': len(indices) - len(selected), 'reasons': dict(reasons)}
    return eligible, exclusions, {'partition_boundary': boundary, 'counts': counts,
                                 'cross_reservoir_duplicate_keys': len(across),
                                 'deduplication_order': 'Lowest raw source index before history and token-cap filtering.'}


def select_requests(eligible, stops):
    selected, shortfalls = {}, []
    for role, required in REQUESTS.items():
        ordered = list(eligible[role])
        random.Random(SELECTION_SEEDS[role]).shuffle(ordered)
        if len(ordered) < required:
            shortfalls.append({'reservoir': role, 'required': required, 'available': len(ordered),
                               'shortfall': required - len(ordered)})
        result = []
        for ordinal, row in enumerate(ordered[:required]):
            request = {**row, 'request_id': f'coding_{role}_{ordinal:05d}', 'request_index': ordinal,
                       'original_index': row['source_index'], 'source_key': SOURCE_KEY,
                       'messages': [{'role': 'user', 'content': row['text']}],
                       'max_new_tokens': 2048, 'stop_token_ids': list(stops)}
            if role == 'evaluation':
                request['panel_ordinal'] = ordinal
            result.append(request)
        selected[role] = result
    return selected, shortfalls


def evaluation_schedule(requests):
    slots = []
    for seed, event in ((17, 2), (17, 4), (29, 2), (29, 4)):
        for ordinal, request in enumerate(requests):
            base = ('F', 'O', 'K') if ordinal % 2 == 0 else ('F', 'K', 'O')
            for repeat in range(3):
                shift = (ordinal + repeat) % 3
                for position, method in enumerate(base[shift:] + base[:shift]):
                    slots.append({'seed': seed, 'event': event, 'request_id': request['request_id'],
                                  'panel_ordinal': ordinal, 'repeat': repeat, 'method': method, 'position': position})
    return slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=Path('/home/minhao2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots') / TOKENIZER_REVISION)
    parser.add_argument('--author-helper', type=Path, default=Path('/home/minhao2/DFlash-Research/inference/dflash/benchmark.py'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing preparation; choose a new output directory')
    args.output.mkdir(parents=True)
    rows, source = load_source(args.workspace)
    save(args.output / 'SOURCE.json', source)
    history, history_audit = inventory_history(args.workspace, rows)
    save(args.output / 'HISTORY_INVENTORY.json', history_audit)
    render, stops, tokenizer = load_renderer(args.tokenizer, args.author_helper)
    eligible, exclusions, selection_audit = prepare_population(rows, history, render)
    selected, shortfalls = select_requests(eligible, stops)
    save(args.output / 'EXCLUSIONS.json', exclusions)
    save(args.output / 'TOKENIZER.json', tokenizer)
    selection_audit.update(selection_seeds=SELECTION_SEEDS, required=REQUESTS,
                           eligible_source_indices={role: [row['source_index'] for row in values]
                                                    for role, values in eligible.items()},
                           source_answers_used=False, historical_performance_used=False,
                           shortfalls=shortfalls, model_generations=0, model_forwards=0,
                           author_template_equivalence_checked=sum(len(values) for values in eligible.values()))
    save(args.output / 'SELECTION_AUDIT.json', selection_audit)
    print(json.dumps(selection_audit['counts']), flush=True)
    if shortfalls:
        save(args.output / 'BLOCKED.json', {'status': 'blocked_before_generation', 'shortfalls': shortfalls})
        raise SystemExit(2)
    save(args.output / 'TRAINING_REQUESTS.json', selected['training'])
    save(args.output / 'EVALUATION_REQUESTS.json', selected['evaluation'])
    windows = [{'event_index': event, 'event_id': f'event_{event:03d}',
                'requests': selected['training'][(event - 1) * 128:event * 128]} for event in range(1, 5)]
    save(args.output / 'WINDOWS.json', windows)
    schedule = evaluation_schedule(selected['evaluation'])
    save(args.output / 'EVALUATION_SCHEDULE.json', schedule)
    contract = {'status': 'sealed_before_generation', 'data_binding': canonical_digest(selected),
                'source_identity': source, 'training_seeds': [17, 29], 'selection_seeds': SELECTION_SEEDS,
                'training_requests': 512, 'evaluation_requests': 128, 'evaluation_slots': len(schedule),
                'evaluation_schedule_sha256': canonical_digest(schedule), 'prompt_cap': PROMPT_CAP,
                'output_cap': 2048, 'training_max_length': 2272, 'source_answers_used': False,
                'population': 'Short-prompt CodeAlpaca requests disjoint from inventoried retained allocations.',
                'selection_order': 'Source-order eligibility then isolated Python Random shuffle, first fixed count.',
                'normalization': 'NFC plus whitespace collapse for matching only; text/indentation unchanged.'}
    save(args.output / 'DATA_CONTRACT.json', contract)
    files = {path.name: identity(path) for path in sorted(args.output.iterdir()) if path.is_file()}
    save(args.output / 'SEAL.json', {'status': 'sealed_before_generation', 'files': files,
                                   'preparation_code': identity(Path(__file__)), 'data_binding': contract['data_binding'],
                                   'model_work': 0, 'source_answers_used': False})
    print('Preparation sealed: ' + contract['data_binding'], flush=True)


if __name__ == '__main__':
    main()
