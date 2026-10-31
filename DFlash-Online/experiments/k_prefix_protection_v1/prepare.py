"""Seal the new KP evaluation panel without changing the original feedback pools."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import random

from experiments.ok_coding import prepare as original

SEEDS = (17, 29)
METHODS = ('F', 'K', 'K_HALF', 'KP')
PANEL_SIZE = 128
SELECTION_SEED = 20261101
UPSTREAM_BINDING = '6def4832d12698fc2733e91c427db77558f3fe50e7f51063e9c2a11f9bc961af'
COMPLETED_RELATIVE = Path('research/dflash_ok_coding_20261030/production_002')


def upstream_requests(directory, source_rows):
    """Bind the actual completed study's 512 training and 128 evaluation inputs."""
    contract = original.load(directory / 'DATA_CONTRACT.json')
    selected = {role: original.load(directory / (role.upper() + '_REQUESTS.json'))
                for role in ('training', 'evaluation')}
    if (contract['data_binding'] != UPSTREAM_BINDING
            or original.canonical_digest(selected) != UPSTREAM_BINDING):
        raise ValueError('Completed upstream request binding differs')
    for role, expected in (('training', 512), ('evaluation', 128)):
        if len(selected[role]) != expected:
            raise ValueError('Completed upstream request count differs: ' + role)
        for request in selected[role]:
            index = request['source_index']
            if (request['source_id'] != f'{original.SOURCE_KEY}:{index}'
                    or request['text'] != original.assemble(source_rows[index])):
                raise ValueError('Completed upstream source identity or text differs')
    return selected


def add_allocation_evidence(history, source_rows, paths):
    """Add explicit coding-study allocations, including normalized-text aliases."""
    normalized = defaultdict(set)
    for index, row in enumerate(source_rows):
        normalized[original.normalize(original.assemble(row))].add(index)
    result = {index: list(records) for index, records in history.items()}
    receipts = []
    seen = set()
    for path in paths:
        receipt = original.identity(path)
        if receipt['sha256'] in seen:
            continue
        seen.add(receipt['sha256'])
        contents = original.load(path) if path.suffix == '.json' else [
            json.loads(line) for line in path.read_text().splitlines()]
        matches = original.history_matches(contents, normalized)
        receipts.append({**receipt, 'matched_source_rows': sorted(matches)})
        for index, reasons in matches.items():
            if not 0 <= index < len(source_rows):
                raise ValueError('Allocation source index outside retained corpus')
            result.setdefault(index, []).append({**receipt, 'reason': '+'.join(sorted(reasons))})
    return result, receipts


def coding_allocation_paths(workspace):
    """The old inventory intentionally skipped its own study; include it explicitly."""
    coding_root = workspace / 'research/dflash_ok_coding_20261030'
    for path in sorted(coding_root.rglob('*')):
        if path.suffix not in ('.json', '.jsonl'):
            continue
        name = path.name.lower()
        if (any(marker in name for marker in original.HISTORY_MARKERS)
                and not any(marker in name for marker in original.NONALLOCATION_MARKERS)):
            yield path


def select_evaluation(eligible, stop_ids, *, count=PANEL_SIZE):
    """Select by prompt-only eligibility and an isolated, prespecified RNG."""
    ordered = sorted(eligible, key=lambda row: row['source_index'])
    if len(ordered) < count:
        raise ValueError(f'Insufficient eligible evaluation requests: {len(ordered)} < {count}')
    random.Random(SELECTION_SEED).shuffle(ordered)
    return [{**row, 'request_id': f'kp_evaluation_{ordinal:05d}',
             'request_index': ordinal, 'panel_ordinal': ordinal,
             'original_index': row['source_index'], 'source_key': original.SOURCE_KEY,
             'messages': [{'role': 'user', 'content': row['text']}],
             'max_new_tokens': 2048, 'stop_token_ids': list(stop_ids)}
            for ordinal, row in enumerate(ordered[:count])]


def method_order(ordinal, repeat):
    base = METHODS if (ordinal // 4) % 2 == 0 else ('F', 'KP', 'K_HALF', 'K')
    shift = (ordinal + repeat) % 4
    return base[shift:] + base[:shift]


def evaluation_schedule(requests, *, seed=None, event=4):
    """Balance all method positions and pairwise precedence within each eight requests."""
    if event != 4 or seed not in (None, *SEEDS):
        raise ValueError('Only seed 17/29 event 4 belongs to this experiment')
    slots = []
    for cell_seed in SEEDS if seed is None else (seed,):
        for ordinal, request in enumerate(requests):
            for repeat in range(3):
                order = method_order(ordinal, repeat)
                for position, method in enumerate(order):
                    slots.append({'seed': cell_seed, 'event': event,
                                  'request_id': request['request_id'], 'panel_ordinal': ordinal,
                                  'repeat': repeat, 'method': method, 'position': position})
    return slots


def schedule_balance(schedule):
    """Return exact position and precedence counts for execution-seal inspection."""
    positions = Counter((row['method'], row['position']) for row in schedule)
    orders = defaultdict(dict)
    for row in schedule:
        key = row['seed'], row['event'], row['request_id'], row['repeat']
        orders[key][row['method']] = row['position']
    precedence = {}
    for left, right in combinations(METHODS, 2):
        precedence[left + '_before_' + right] = sum(
            order[left] < order[right] for order in orders.values())
        precedence[right + '_before_' + left] = sum(
            order[right] < order[left] for order in orders.values())
    return {'generations': len(schedule), 'method_positions': {
        method: [positions[method, position] for position in range(4)] for method in METHODS},
        'pairwise_precedence': precedence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=original.DEFAULT_WORKSPACE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=Path(
        '/home/minhao2/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots') / original.TOKENIZER_REVISION)
    parser.add_argument('--author-helper', type=Path, default=Path(
        '/home/minhao2/DFlash-Research/inference/dflash/benchmark.py'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing preparation; choose a new output directory')
    rows, source = original.load_source(args.workspace)
    completed = args.workspace / COMPLETED_RELATIVE
    upstream = completed / 'data_preparation'
    old_requests = upstream_requests(upstream, rows)
    history, history_audit = original.inventory_history(args.workspace, rows)
    paths = [upstream / name for name in ('TRAINING_REQUESTS.json', 'EVALUATION_REQUESTS.json')]
    paths.extend(coding_allocation_paths(args.workspace))
    history, coding_receipts = add_allocation_evidence(history, rows, paths)
    history_audit.update(explicit_completed_coding_allocations=coding_receipts,
                         final_excluded_source_rows=len(history),
                         completed_study_training_requests=512, completed_study_evaluation_requests=128)
    render, stops, tokenizer = original.load_renderer(args.tokenizer, args.author_helper)
    eligible, exclusions, audit = original.prepare_population(rows, history, render)
    args.output.mkdir(parents=True)
    for name, value in (('SOURCE', source), ('HISTORY_INVENTORY', history_audit),
                        ('EXCLUSIONS', exclusions), ('TOKENIZER', tokenizer)):
        original.save(args.output / (name + '.json'), value)
    audit.update(selection_seed=SELECTION_SEED, required_evaluation_requests=PANEL_SIZE,
                 eligible_evaluation_source_indices=[row['source_index'] for row in eligible['evaluation']],
                 source_answers_used=False, historical_performance_used=False,
                 model_generations=0, model_forwards=0)
    original.save(args.output / 'SELECTION_AUDIT.json', audit)
    print(audit['counts'], flush=True)
    if len(eligible['evaluation']) < PANEL_SIZE:
        original.save(args.output / 'BLOCKED.json', {
            'reason': 'evaluation_selection_shortfall', 'required': PANEL_SIZE,
            'available': len(eligible['evaluation']), 'scientific_work': 0})
        raise SystemExit(2)
    requests = select_evaluation(eligible['evaluation'], stops)
    old_ids = {row['source_id'] for values in old_requests.values() for row in values}
    old_text = {original.normalize(row['text']) for values in old_requests.values() for row in values}
    if any(row['source_id'] in old_ids or original.normalize(row['text']) in old_text for row in requests):
        raise ValueError('New evaluation overlaps the completed coding allocations')
    schedule = evaluation_schedule(requests)
    original.save(args.output / 'EVALUATION_REQUESTS.json', requests)
    original.save(args.output / 'EVALUATION_SCHEDULE.json', schedule)
    balance = {str(seed): {
        'full_cell': schedule_balance([row for row in schedule if row['seed'] == seed]),
        'included_gate': schedule_balance([row for row in schedule if row['seed'] == seed and row['panel_ordinal'] < 8])}
        for seed in SEEDS}
    original.save(args.output / 'SCHEDULE_BALANCE.json', balance)
    contract = {
        'status': 'sealed_before_model_work', 'upstream_training_binding': UPSTREAM_BINDING,
        'upstream_completed_root': str(completed),
        'upstream_files': {name: original.identity(upstream / name) for name in
                           ('DATA_CONTRACT.json', 'TRAINING_REQUESTS.json', 'WINDOWS.json')},
        'evaluation_binding': original.canonical_digest(requests),
        'source_identity': source, 'training_seeds': list(SEEDS), 'selection_seed': SELECTION_SEED,
        'training_requests_reused': 512, 'new_feedback_generations': 0,
        'evaluation_requests': PANEL_SIZE, 'evaluation_slots': len(schedule),
        'evaluation_schedule_sha256': original.canonical_digest(schedule),
        'prompt_cap': 224, 'output_cap': 2048, 'training_max_length': 2272,
        'source_answers_used': False, 'methods': list(METHODS), 'evaluation_events': [4],
        'selection_order': 'Ascending eligible source index; isolated Random(20261101) shuffle once; first 128.',
        'history_scope': 'Retained project allocations plus explicit completed coding 512/128 and later recoverable allocations.',
        'normalization': 'NFC plus whitespace collapse for identity only; serving text preserved exactly.',
    }
    original.save(args.output / 'DATA_CONTRACT.json', contract)
    from .statistics import write_precision_note
    write_precision_note(completed / 'final_report/PAIRED_REQUESTS.csv', args.output)
    files = {path.name: original.identity(path) for path in sorted(args.output.iterdir()) if path.is_file()}
    original.save(args.output / 'SEAL.json', {
        'status': 'sealed_before_model_work', 'files': files,
        'preparation_code': original.identity(Path(__file__)),
        'inherited_preparation_code': original.identity(Path(original.__file__)),
        'statistics_code': original.identity(Path(__file__).with_name('statistics.py')),
        'upstream_training_binding': UPSTREAM_BINDING,
        'evaluation_binding': contract['evaluation_binding'], 'model_work': 0,
        'source_answers_used': False})
    print('New evaluation binding: ' + contract['evaluation_binding'], flush=True)


if __name__ == '__main__':
    main()
