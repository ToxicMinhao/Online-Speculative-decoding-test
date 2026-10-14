#!/usr/bin/env python3
"""Pinned-source sampling and auditable quality-only selection (no model calls).

Candidate extraction needs pyarrow; final selection uses only the stdlib.
Source assistant messages are never included in emitted files.
"""
import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path
from build_streams import strict_object, validate_selection

SEED = 20261014
SOURCES = {
    'code': {'repo': 'allenai/tulu-3-sft-personas-code',
             'revision': '1412abe88dd2976af977260788e033013449f7b2', 'rows': 34999,
             'files': [('data/train-00000-of-00001.parquet', 38600399,
                        'e343c319aa0b4c577236d1e52433528ea7801dabd5696242ba3244242f76eb57')]},
    'math': {'repo': 'allenai/tulu-3-sft-personas-math',
             'revision': '46add032f5203fd96efbd54d074198981ed57d25', 'rows': 149960,
             'files': [('data/train-00000-of-00002.parquet', 177548235,
                        '4b12d90cfb2c370cfe26937c6b711d284f6f67737b27ed3873744f5ba31c5adc'),
                       ('data/train-00001-of-00002.parquet', 184073832,
                        '95c16943dec1faeabfde648cb36ddcd3ac83cd72532b40e987c642655dc5f697')]},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Refusing to replace different existing file: {path}')
    else:
        path.write_bytes(data)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def jsonl_bytes(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode()


def candidates(args):
    import pyarrow.parquet as pq
    if not 128 <= args.count_per_label <= min(s['rows'] for s in SOURCES.values()):
        raise ValueError('Candidate count must be between 128 and the smaller source size')
    cache = Path(args.cache_dir)
    all_rows, receipts = [], []
    for label, source in SOURCES.items():
        rows = []
        for name, size, expected in source['files']:
            path = cache / label / Path(name).name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{name}"
                partial = path.with_suffix('.partial')
                # No credential is needed or sent for these public datasets.
                with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as output:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                if partial.stat().st_size != size or file_digest(partial) != expected:
                    raise ValueError('Downloaded source identity mismatch: ' + name)
                partial.rename(path)
            if path.stat().st_size != size or file_digest(path) != expected:
                raise ValueError('Cached source identity mismatch: ' + name)
            receipts.append({'label': label, 'repo': source['repo'], 'revision': source['revision'],
                             'source_path': name, 'bytes': size, 'sha256': expected})
            # Read messages transiently only to verify the exact sole user message.
            table = pq.read_table(path, columns=['id', 'prompt', 'messages'])
            for row in table.to_pylist():
                users = [m.get('content') for m in row['messages'] if m.get('role') == 'user']
                rows.append({'id': row['id'], 'prompt': row['prompt'], 'user_messages': users,
                             'roles': [m.get('role') for m in row['messages']]})
            print(f'{label}: read {len(rows)} source rows', flush=True)
        if len(rows) != source['rows']:
            raise ValueError('Pinned source row count changed')
        order = list(range(len(rows)))
        random.Random(SEED).shuffle(order)
        for rank, index in enumerate(order[:args.count_per_label], 1):
            row = rows[index]
            valid = (isinstance(row['prompt'], str) and bool(row['prompt'].strip())
                     and row['user_messages'] == [row['prompt']] and row['roles'] == ['user', 'assistant']
                     and isinstance(row['id'], str) and bool(row['id']))
            all_rows.append({'question_id': f"{label}:{row['id']}", 'source_dataset': source['repo'],
                'source_revision': source['revision'], 'source_split': 'train', 'source_row_index': index,
                'source_id': row['id'], 'label': label, 'prompt': row['prompt'],
                'prompt_sha256': digest(row['prompt'].encode()) if isinstance(row['prompt'], str) else None,
                'sampling_rank': rank, 'seed': SEED, 'extraction_valid': valid})
    output = Path(args.output_dir)
    save(output/'candidates.jsonl', jsonl_bytes(all_rows))
    save(output/'source_manifest.json', json_bytes({'seed': SEED, 'sources': SOURCES, 'downloads': receipts,
        'sampling': 'Independent Random(20261014).shuffle of all row indices for each label, in pinned parquet-file order; inspect in increasing rank, retain first 128 eligible per label.',
        'count_per_label': args.count_per_label, 'assistant_answers_exported': False}))


def select(args):
    rows = [json.loads(line, object_pairs_hook=strict_object) for line in Path(args.candidates).read_text().split('\n') if line.strip()]
    candidate_fields = {'extraction_valid', 'label', 'prompt', 'prompt_sha256', 'question_id',
        'sampling_rank', 'seed', 'source_dataset', 'source_id', 'source_revision', 'source_row_index', 'source_split'}
    orders = {}
    for label, source in SOURCES.items():
        orders[label] = list(range(source['rows']))
        random.Random(SEED).shuffle(orders[label])
    for row in rows:
        if not isinstance(row, dict) or set(row) != candidate_fields or type(row['extraction_valid']) is not bool:
            raise ValueError('Unexpected candidate fields or invalid extraction flag')
        if not isinstance(row['prompt'], str) or not isinstance(row['source_id'], str):
            raise ValueError('Malformed candidate text or source identifier; re-extract valid source records')
        source = SOURCES.get(row['label'])
        if source is None or row['source_dataset'] != source['repo'] or row['source_revision'] != source['revision']:
            raise ValueError('Candidate does not match a pinned source')
        rank = row['sampling_rank']
        if type(rank) is not int or not 1 <= rank <= source['rows'] or row['source_row_index'] != orders[row['label']][rank-1]:
            raise ValueError('Candidate index does not match seeded full-source permutation')
        if row['seed'] != SEED or row['source_split'] != 'train' or row['prompt_sha256'] != digest(row['prompt'].encode()):
            raise ValueError('Candidate seed/split/text binding is invalid')
        if row['question_id'] != row['label'] + ':' + row['source_id']:
            raise ValueError('Candidate ID is inconsistent')
    decisions = []
    for path in args.reviews:
        decisions.extend(json.loads(Path(path).read_text(), object_pairs_hook=strict_object)['decisions'])
    review = {}
    for item in decisions:
        qid = item['question_id']
        if qid in review:
            raise ValueError('Duplicate quality decision: ' + qid)
        if item['decision'] not in ('accept', 'reject') or not item.get('reason'):
            raise ValueError('Incomplete quality decision: ' + qid)
        review[qid] = item
    candidate_ids = {r['question_id'] for r in rows}
    if not set(review) <= candidate_ids:
        raise ValueError('Quality decisions refer to unknown candidates')
    selected, audit, seen_ids, seen_text = [], [], set(), set()
    for label in SOURCES:
        accepted = 0
        ordered = sorted((r for r in rows if r['label'] == label), key=lambda r: r['sampling_rank'])
        if [r['sampling_rank'] for r in ordered] != list(range(1, len(ordered)+1)):
            raise ValueError('Candidate ranks are not contiguous')
        for row in ordered:
            if accepted == 128:
                break
            decision = review.get(row['question_id'])
            if decision is None:
                raise ValueError('Unreviewed candidate before selection boundary: ' + row['question_id'])
            if decision.get('prompt_sha256') != row['prompt_sha256']:
                raise ValueError('Review is not bound to candidate text')
            normalized = ' '.join(row['prompt'].split()).casefold()
            reason = decision['reason']
            take = decision['decision'] == 'accept' and row['extraction_valid']
            if row['question_id'] in seen_ids or normalized in seen_text:
                take, reason = False, 'duplicate ID or casefolded whitespace-normalized prompt'
            if not row['extraction_valid']:
                reason = 'source does not have one exact user prompt and one assistant response'
            audit.append({'question_id': row['question_id'], 'label': label, 'sampling_rank': row['sampling_rank'],
                          'prompt_sha256': row['prompt_sha256'], 'selected': take, 'reason': reason})
            if not take:
                continue
            item = {k: v for k, v in row.items() if k != 'extraction_valid'}
            item.update(replica=accepted//32+1, block=('C' if label == 'code' else 'M')+str((accepted%32)//8+1),
                        position_in_block=accepted%8+1)
            selected.append(item)
            seen_ids.add(row['question_id']); seen_text.add(normalized); accepted += 1
        if accepted != 128:
            raise ValueError(f'Insufficient reviewed eligible {label} candidates: {accepted}')
    out = Path(args.output_dir)
    validate_selection(selected)
    save(out/'selected_prompts.jsonl', jsonl_bytes(selected))
    save(out/'selection_audit.json', json_bytes({'seed': SEED, 'rules': 'English, self-contained, clear source-consistent label; no malformed, duplicate or missing-resource/history questions. No performance screening.',
       'candidate_file_sha256': file_digest(Path(args.candidates)),
       'review_file_sha256': [file_digest(Path(p)) for p in args.reviews], 'decisions_through_selection_boundary': audit}))
    print(json.dumps({'unique_questions': len(selected), 'inspected_through_boundary': len(audit)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cand = sub.add_parser('candidates')
    cand.add_argument('--cache-dir', required=True)
    cand.add_argument('--output-dir', required=True)
    cand.add_argument('--count-per-label', type=int, default=192)
    choice = sub.add_parser('select')
    choice.add_argument('--candidates', required=True)
    choice.add_argument('--reviews', required=True, nargs='+')
    choice.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    candidates(args) if args.command == 'candidates' else select(args)


if __name__ == '__main__':
    main()
