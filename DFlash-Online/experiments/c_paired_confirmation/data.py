"""Input-only selection and fixed two-arm schedule for the paired RNG-29 run.

Normalization is solely an identity operation. Model prompts retain their
historical rendering. Eligibility/exposure authentication precedes this module.
"""
from copy import deepcopy
from experiments.mathqa_fixed_lr_transfer import data as original

normalize = original.normalize
digest = original.digest
token_digest = original.token_digest
fingerprints = original.fingerprints
history_matches = original.history_matches
FINGERPRINT_FIELDS = original.FINGERPRINT_FIELDS
METHODS = ('N', 'C')
PANEL_SIZE, WARMUP_SIZE, BLOCK_SIZE = 256, 2, 64
SELECTION_PREFIX = 'kplus_c_pair_confirm_r29_20261117_v1'
DATASETS = {
    'gsm8k': dict(source_prefix='gsm8k:main:test:', size=1319, first_ordinal=768,
                  revision='740312add88f781978c0658806c59bc2815b9866'),
    'mathqa': dict(source_prefix='mathqa:test:', size=2985, first_ordinal=512,
                   revision='4f958013c164a40745a5b771dfb005df308deb4a'),
}


def rank_key(dataset, source_id, normalized_content_sha256):
    if dataset not in DATASETS:
        raise ValueError('Unknown paired-confirmation dataset')
    return digest(f'{SELECTION_PREFIX}|{dataset}|{source_id}|{normalized_content_sha256}')


def rank_eligible(dataset, rows):
    return sorted(rows, key=lambda row: (rank_key(dataset, row['source_id'], row['normalized_content_sha256']), row['source_id']))


def seal_requests(dataset, eligible):
    ranked = rank_eligible(dataset, eligible)
    if len(ranked) < PANEL_SIZE + WARMUP_SIZE:
        raise ValueError('Fewer than 258 audited eligible requests; no replacement population')
    roles = {}
    for role, selected in (('scored', ranked[:256]), ('warmup', ranked[256:258])):
        rows = []
        for ordinal, original_row in enumerate(selected):
            row = deepcopy(original_row)
            row.update(role=role, panel_ordinal=ordinal, ordinal=ordinal, request_index=ordinal,
                block_ordinal=ordinal // 64 if role == 'scored' else None,
                request_id=f'{dataset}-c-pair29-{row["source_row"]:04d}', original_index=row['source_row'],
                max_new_tokens=2048, stop_token_ids=[151645, 151643], collect_feedback=False, dataset=dataset,
                selection_key=rank_key(dataset, row['source_id'], row['normalized_content_sha256']))
            rows.append(row)
        roles[role] = rows
    return roles


def validate_request(dataset, row, role, ordinal):
    spec = DATASETS[dataset]
    if (row['role'] != role or row['panel_ordinal'] != ordinal or row['ordinal'] != ordinal
            or row['request_index'] != ordinal or row['dataset'] != dataset
            or row['block_ordinal'] != (ordinal // 64 if role == 'scored' else None)
            or row['max_new_tokens'] != 2048 or row['stop_token_ids'] != [151645, 151643]
            or row['collect_feedback'] is not False):
        raise ValueError('Paired-confirmation input contract differs')
    if (type(row['source_row']) is not int or not 0 <= row['source_row'] < spec['size']
            or row['source_id'] != spec['source_prefix'] + str(row['source_row'])
            or row['original_index'] != row['source_row']
            or row['request_id'] != f'{dataset}-c-pair29-{row["source_row"]:04d}'):
        raise ValueError('Paired-confirmation source identity differs')
    ids = row['prompt_ids']
    if not ids or len(ids) > 2048 or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError('Malformed or over-context prompt IDs')
    expected = fingerprints(row['question'], row['options'], row['rendered_prompt'], ids)
    if any(row[k] != v for k, v in expected.items()):
        raise ValueError('Input fingerprint differs')
    if row['selection_key'] != rank_key(dataset, row['source_id'], row['normalized_content_sha256']):
        raise ValueError('Fixed input rank differs')


def validate_manifests(dataset, panel, warmups, adaptation):
    scored, warm = panel['requests'], warmups['requests']
    if len(scored) != PANEL_SIZE or len(warm) != WARMUP_SIZE:
        raise ValueError('Require both complete 256+2 inputs')
    for role, rows in (('scored', scored), ('warmup', warm)):
        for ordinal, row in enumerate(rows):
            validate_request(dataset, row, role, ordinal)
    selected = scored + warm
    for field in ('request_id', 'source_id', *FINGERPRINT_FIELDS):
        if len({row[field] for row in selected}) != 258:
            raise ValueError('Duplicate selected input: ' + field)
    if [row['source_id'] for row in selected] != [row['source_id'] for row in rank_eligible(dataset, selected)]:
        raise ValueError('Sealed rank order differs')
    train = adaptation['train_requests'];start = DATASETS[dataset]['first_ordinal']
    if (adaptation['start_ordinal'] != start or adaptation['stop_ordinal'] != start + 256
            or len(train) != 256 or [r['request_id'] for r in train] != [f'train_{i:05d}' for i in range(start, start + 256)]
            or len({r['source_id'] for r in train}) != 256):
        raise ValueError('Original paired continuation input stream differs')
    source_prefix = 'gsm8k:main:train:' if dataset == 'gsm8k' else 'mathqa:train:'
    for ordinal, row in enumerate(train, start):
        if (not row['source_id'].startswith(source_prefix) or row['request_index'] != ordinal + 1
                or row['max_new_tokens'] != 2048 or row['stop_token_ids'] != [151645, 151643]
                or row['collect_feedback'] is not True or len(row['prompt_ids']) + 2048 > 4096):
            raise ValueError('Original continuation input contract differs')
    for field in ('source_id', 'prompt_ids_sha256'):
        historical = {row[field] for row in train}
        if any(row[field] in historical for row in selected):
            raise ValueError('Held-out panel overlaps continuation inputs')
    return dict(status='passed', dataset=dataset, scored=256, warmup=2, adaptation=256,
                new_canonical_generations=258, no_new_model_work=True)


def evaluation_schedule(requests):
    if len(requests) != 256 or [r['panel_ordinal'] for r in requests] != list(range(256)):
        raise ValueError('Require the sealed 256-request panel')
    rows = []
    for repeat in range(3):
        for block in range(4):
            methods = ('N', 'C') if (repeat + block) % 2 == 0 else ('C', 'N')
            for position, method in enumerate(methods):
                rows.append(dict(activation=len(rows), repeat=repeat, block_ordinal=block, position=position,
                    method=method, request_ordinals=list(range(64 * block, 64 * (block + 1)))))
    return rows
