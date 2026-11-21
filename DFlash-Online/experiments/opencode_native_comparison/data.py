"""Input-only OpenCode selection, with fixed reservoirs and auditable exclusions.

Source acquisition supplies only ``input`` plus immutable source coordinates.
No answer, generated length or model score is consulted here.
"""
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .contract import SELECTION_SEED, digest, validate_manifest

DATASET = 'nvidia/OpenCodeInstruct'
SOURCE_REF = 'refs/convert/parquet'
NORMALIZATION = 'Unicode NFKC, then whitespace collapse; case and punctuation preserved'
COUNTS = dict(train=7424, evaluation=1024, warmup=2, qualification=16)
MATCH_FIELDS = ('source_identity_sha256', 'normalized_raw_sha256',
                'rendered_prompt_sha256', 'prompt_ids_sha256')


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize(text):
    """Duplicate detection only: the tokenizer always receives original text."""
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def source_key(identity):
    return json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def reservoir_plan(splits):
    """Match the author's split branch, with the prospectively enlarged panel."""
    splits = list(splits)
    if not splits:
        raise ValueError('The pinned source has no split')
    if 'train' in splits and 'test' in splits:
        return dict(branch='official_train_test', ranges=[
            dict(pool='train', split='train', start=0, stop=30000),
            dict(pool='evaluation', split='test', start=0, stop=20000)])
    split = 'train' if 'train' in splits else 'test' if 'test' in splits else splits[0]
    return dict(branch='single_source_index_partition', ranges=[
        dict(pool='train', split=split, start=0, stop=30000),
        dict(pool='evaluation', split=split, start=30000, stop=50000)])


def prepare_input(row, tokenizer):
    """Render one source input exactly; return an exclusion rather than repair it."""
    identity = row['source_identity']
    base = dict(source_identity=identity, source_identity_sha256=digest(identity),
                pool=row['pool'])
    raw = row.get('input')
    if not isinstance(raw, str) or not raw.strip():
        return base | dict(exclusion='empty_or_nonstring_input')
    base.update(raw_input=raw, raw_input_sha256=text_hash(raw),
                normalized_raw_sha256=text_hash(normalize(raw)))
    messages = [dict(role='user', content=raw)]
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
    except (ValueError, TypeError, UnicodeError) as error:
        return base | dict(exclusion='invalid_tokenization', error=str(error))
    if (not ids or any(type(token) is not int or token < 0 for token in ids)):
        return base | dict(exclusion='invalid_tokenization')
    base.update(rendered_prompt=rendered, rendered_prompt_sha256=text_hash(rendered),
                prompt_ids=ids, prompt_ids_sha256=digest(ids), prompt_tokens=len(ids))
    if len(ids) > 2048:
        base['exclusion'] = 'prompt_exceeds_2048_tokens'
    return base


def history_fingerprints(value):
    """Extract input evidence, never recurse through assistant/answer payloads."""
    found = {key: set() for key in MATCH_FIELDS}
    text_fields = {'input', 'raw_input', 'user_message', 'user_text', 'question', 'text',
                   'instruction', 'prompt', 'prompt_text', 'rendered_prompt', 'content', 'value'}
    token_fields = {'prompt_ids', 'prompt_token_ids', 'input_ids', 'rendered_prompt_token_ids'}
    ignored = {'output', 'outputs', 'answer', 'answers', 'response', 'responses',
               'solution', 'solutions', 'rationale', 'completion', 'completions',
               'assistant', 'generated_text', 'output_text', 'response_text',
               'reference_answer', 'metrics', 'candidate_pools', 'source_catalog',
               'eligible_order', 'unselected_eligible_ids'}
    aliases = {'normalized_content_sha256': 'normalized_raw_sha256',
               'normalized_prompt_sha256': 'normalized_raw_sha256',
               'prompt_token_ids_sha256': 'prompt_ids_sha256',
               'rendered_token_sha256': 'prompt_ids_sha256'}

    def visit(item, field=''):
        field = field.lower()
        if field in ignored:
            return
        if isinstance(item, dict):
            if item.get('role', item.get('from')) in ('assistant', 'gpt', 'chatgpt', 'bard', 'bing'):
                return
            identity = item.get('source_identity')
            if isinstance(identity, dict) and identity.get('dataset') == DATASET:
                found['source_identity_sha256'].add(digest(identity))
            for key, child in item.items():
                visit(child, key)
        elif isinstance(item, list):
            if field in token_fields and item and all(type(token) is int for token in item):
                found['prompt_ids_sha256'].add(digest(item))
            else:
                for child in item:
                    visit(child, field)
        elif isinstance(item, str):
            if field in text_fields:
                found['normalized_raw_sha256'].add(text_hash(normalize(item)))
                found['rendered_prompt_sha256'].add(text_hash(item))
                # Old requests sometimes retain only the rendered chat prompt.
                for raw in re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', item, re.S):
                    found['normalized_raw_sha256'].add(text_hash(normalize(raw)))
            mapped = aliases.get(field, field)
            if mapped in found:
                found[mapped].add(item)

    visit(value)
    return found


def merge_history(inventories):
    merged = {key: set() for key in MATCH_FIELDS}
    for inventory in inventories:
        for key in MATCH_FIELDS:
            merged[key].update(inventory.get(key, ()))
    return merged


def inventory_history(paths):
    """Read an explicitly enumerated input-history inventory with file receipts.

The caller chooses history artifacts, rather than treating a downloaded source
catalog as evidence that every source row was used. Parse failures are retained
as coverage gaps and must be disclosed by the readiness report.
"""
    merged = {key: set() for key in MATCH_FIELDS}
    files, gaps = [], []
    content_seen = set()
    for path in sorted({Path(p).resolve() for p in paths}):
        try:
            with path.open('rb') as stream:
                sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            identity = dict(bytes=path.stat().st_size, sha256=sha)
            if sha in content_seen:
                files.append(dict(path=str(path), identity=identity, duplicate_content=True))
                continue
            content_seen.add(sha)
            found = {key: set() for key in MATCH_FIELDS}
            if path.suffix == '.jsonl':
                with path.open() as stream:
                    for line in stream:
                        if line.strip():
                            current = history_fingerprints(json.loads(line))
                            for key in MATCH_FIELDS:
                                found[key].update(current[key])
            else:
                found = history_fingerprints(json.loads(path.read_text()))
            for key in MATCH_FIELDS:
                merged[key].update(found[key])
            files.append(dict(path=str(path), identity=identity,
                              extracted={key:len(values) for key,values in found.items()}))
        except (OSError, ValueError, UnicodeError) as error:
            gaps.append(dict(path=str(path), error=str(error)))
    return dict(fingerprints={key:sorted(values) for key, values in merged.items()}, files=files,
                coverage=dict(scope='enumerated retained input histories and reservations',
                              normalization=NORMALIZATION, gaps=gaps,
                              excludes_unknown_history=True, semantic_near_duplicates_checked=False))


def acquire_inputs(loader_resolution, metadata, destination):
    """Read only required input-column row groups through HTTP range requests.

Immutable LFS content hashes are source-provided identities. We do not claim a
locally recomputed complete-shard hash after reading only one column.
"""
    import fsspec
    import pyarrow.parquet as parquet

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Never replace a retained source projection')
    revision = metadata['sha']
    data_files = loader_resolution['data_files']
    plan = reservoir_plan(data_files)
    shards = {row['rfilename']:dict(bytes=row['size'], sha256=row['lfs']['sha256'],
                git_pointer_blob=row['blobId'], hash_source='immutable HF LFS metadata')
              for row in metadata['siblings'] if row['rfilename'].endswith('.parquet')}
    counts, reads, row_count = {}, [], 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as output:
        for split in dict.fromkeys(r['split'] for r in plan['ranges']):
            stop = max(r['stop'] for r in plan['ranges'] if r['split']==split)
            offset = 0
            for original in data_files[split]:
                # datasets resolves hf://...@revision/path, not a moving URL.
                marker = '@'+revision+'/'
                if marker not in original:
                    raise ValueError('Loader shard path is not pinned to resolved revision')
                shard = original.split(marker,1)[1]
                if shard not in shards:
                    raise ValueError('Loader shard has no immutable LFS identity')
                url = f'https://hf-mirror.com/datasets/{DATASET}/resolve/{revision}/{shard}'
                with fsspec.open(url, mode='rb', block_size=65536, cache_type='none') as stream:
                    table = parquet.ParquetFile(stream)
                    shard_offset = 0
                    for group in range(table.metadata.num_row_groups):
                        size = table.metadata.row_group(group).num_rows
                        start = offset + shard_offset
                        if start >= stop:
                            break
                        values = table.read_row_group(group, columns=['input']).column('input').to_pylist()
                        take = min(size, stop-start)
                        reads.append(dict(split=split, shard=shard, row_group=group,
                            decoded_rows=size, inspected_rows=take, column='input'))
                        for index, raw in enumerate(values[:take]):
                            raw_index = start+index
                            pool = next(r['pool'] for r in plan['ranges']
                                        if r['split']==split and r['start']<=raw_index<r['stop'])
                            identity = dict(dataset=DATASET, revision=revision, split=split,
                                            shard=shard, raw_index=raw_index,
                                            shard_row_index=shard_offset+index)
                            output.write(json.dumps(dict(pool=pool, input=raw, source_identity=identity),
                                                    ensure_ascii=False, separators=(',',':'))+'\n')
                            row_count += 1
                        shard_offset += size
                    offset += table.metadata.num_rows
                if offset >= stop:
                    break
            counts[split] = offset
    with destination.open('rb') as stream:
        projection_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(schema='opencode_source_acquisition_v1', revision=revision,
        source_ref=SOURCE_REF, split_order=list(data_files), split_rows=counts,
        split_row_count_scope='metadata rows in visited shards; unvisited remainder not scanned',
        branch=plan['branch'], shards=shards, row_groups_read=reads, inspected_rows=row_count,
        projected_columns=['input'], reference_solutions_cached=False,
        complete_shard_bytes_rehashed=False,
        input_projection_identity=dict(bytes=destination.stat().st_size, sha256=projection_hash))


def length_summary(rows):
    values = sorted(row['prompt_tokens'] for row in rows if 'prompt_tokens' in row)
    if not values:
        return dict(count=0)
    return dict(count=len(values), minimum=values[0], maximum=values[-1],
                mean=sum(values)/len(values),
                quantiles={str(q): values[round(q*(len(values)-1))] for q in (0.25, 0.5, 0.75, 0.95, 0.99)})


def exclude_inputs(rows, history):
    """Exclude cross-pool duplicates from both sides before deterministic dedup."""
    history = {key: set(values) for key, values in history.items()}
    identities = [row['source_identity_sha256'] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('Source rows were read twice')
    by_pool = {pool: {key: set() for key in MATCH_FIELDS[1:]} for pool in ('train', 'evaluation')}
    for row in rows:
        for key, values in by_pool[row['pool']].items():
            if key in row:
                values.add(row[key])
    crossed = {key: by_pool['train'][key] & by_pool['evaluation'][key]
               for key in MATCH_FIELDS[1:]}
    seen = {pool: {key: set() for key in MATCH_FIELDS[1:]} for pool in by_pool}
    eligible, excluded = [], []
    for row in sorted(rows, key=lambda row: source_key(row['source_identity'])):
        reasons = [row['exclusion']] if 'exclusion' in row else []
        prior = [key for key in MATCH_FIELDS if row.get(key) in history.get(key, ())]
        cross = [key for key in crossed if row.get(key) in crossed[key]]
        duplicate = [key for key in seen[row['pool']] if row.get(key) in seen[row['pool']][key]]
        if prior:
            reasons.append('known_project_exposure_or_reservation')
        if cross:
            reasons.append('duplicate_across_source_pools')
        if duplicate:
            reasons.append('duplicate_within_source_pool')
        if reasons:
            excluded.append(dict(source_identity=row['source_identity'], pool=row['pool'],
                reasons=reasons, historical_match_fields=prior, cross_pool_match_fields=cross,
                within_pool_match_fields=duplicate, prompt_tokens=row.get('prompt_tokens')))
        else:
            eligible.append(row)
            for key, values in seen[row['pool']].items():
                if key in row:
                    values.add(row[key])
    return eligible, excluded


def rank(row, role):
    return (digest(dict(seed=SELECTION_SEED, role=role, source_identity=row['source_identity'])),
            source_key(row['source_identity']))


def select_inputs(eligible, counts=None):
    counts = COUNTS if counts is None else counts
    train = sorted((r for r in eligible if r['pool'] == 'train'), key=lambda r: rank(r, 'train'))
    evaluation = sorted((r for r in eligible if r['pool'] == 'evaluation'), key=lambda r: rank(r, 'evaluation'))
    required_train = counts['train'] + counts['warmup'] + counts['qualification']
    if len(train) < required_train or len(evaluation) < counts['evaluation']:
        raise ValueError(f'Bounded reservoir insufficient: train={len(train)}/{required_train}, evaluation={len(evaluation)}/{counts["evaluation"]}')
    result = dict(train=train[:counts['train']], evaluation=evaluation[:counts['evaluation']])
    remaining = train[counts['train']:]
    result['warmup'] = sorted(remaining, key=lambda r: rank(r, 'warmup'))[:counts['warmup']]
    used = {r['source_identity_sha256'] for group in result.values() for r in group}
    remaining = [r for r in remaining if r['source_identity_sha256'] not in used]
    by_length = sorted(remaining, key=lambda r: (r['prompt_tokens'], source_key(r['source_identity'])))
    shortest = by_length[:2]
    shortest_ids = {r['source_identity_sha256'] for r in shortest}
    longest = sorted((r for r in remaining if r['source_identity_sha256'] not in shortest_ids),
                     key=lambda r: (-r['prompt_tokens'], source_key(r['source_identity'])))[:2]
    extremes = shortest + longest
    if len({r['source_identity_sha256'] for r in extremes}) != 4:
        raise ValueError('Qualification length extremes are not four distinct inputs')
    used.update(r['source_identity_sha256'] for r in extremes)
    middle = sorted((r for r in remaining if r['source_identity_sha256'] not in used), key=lambda r: rank(r, 'qualification'))
    result['qualification'] = extremes + middle[:counts['qualification']-4]
    return result


def manifest_requests(selected):
    result = {}
    for role, rows in selected.items():
        requests = []
        for ordinal, row in enumerate(rows):
            entry = dict(row, request_id='opencode:'+row['source_identity_sha256'],
                source_id='opencode:'+row['source_identity_sha256'],
                original_index=row['source_identity']['raw_index'],
                request_index=ordinal, max_new_tokens=2048, stop_token_ids=[151645, 151643],
                selection_role=role, selection_rank=rank(row, role)[0])
            if role == 'evaluation':
                entry['panel_ordinal'] = ordinal
            if role == 'qualification':
                entry['qualification_selection'] = ('two_shortest' if ordinal < 2 else
                    'two_longest_remaining' if ordinal < 4 else 'hash_ranked')
            requests.append(entry)
        result[role+'_requests'] = requests
    validate_manifest(result)
    return result


def training_length_bounds(requests):
    """Input-only upper bounds: every output may consume its complete 2048 cap."""
    windows = []
    for start in range(0, len(requests), 128):
        rows = requests[start:start+128]
        prompt_sum = sum(len(r['prompt_ids']) for r in rows)
        windows.append(dict(event=1+start//128, requests=len(rows), prompt_tokens=prompt_sum,
                            maximum_prompt_tokens=max(len(r['prompt_ids']) for r in rows),
                            maximum_sequence_length=max(len(r['prompt_ids']) for r in rows)+2048,
                            maximum_sequence_tokens=prompt_sum+2048*len(rows)))
    return dict(windows=windows, maximum_window_sequence_tokens=max(w['maximum_sequence_tokens'] for w in windows),
                maximum_window_prompt_tokens=max(w['prompt_tokens'] for w in windows),
                maximum_sequence_length=max(w['maximum_sequence_length'] for w in windows))
