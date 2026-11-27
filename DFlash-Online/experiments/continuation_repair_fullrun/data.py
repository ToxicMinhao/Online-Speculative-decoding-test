"""Fresh input-only confirmation selection; the PR56 training stream is immutable."""
from collections import Counter
from pathlib import Path
import json

from experiments.ok_coding.storage import identity
from experiments.opencode_native_comparison.data import (
    DATASET, MATCH_FIELDS, exclude_inputs, history_fingerprints, merge_history,
    prepare_input, source_key,
)
from .contract import INPUT_IDENTITY, SOURCE_REVISION, SELECTION_SEED, digest

EVALUATION_ROLE = 'fullrun_confirm_evaluation_v1'
WITNESS_ROLE = 'fullrun_training_witness_v1'


def rank(row, role):
    source = row['source_identity']
    return digest(dict(seed=SELECTION_SEED, role=role, source_identity=source)), source_key(source)


def original_inputs(path):
    """Authenticate the original serialized manifest before reading any requests."""
    if identity(path) != INPUT_IDENTITY:
        raise ValueError('PR56 INPUTS identity differs from the handoff')
    return json.loads(Path(path).read_text())


def projected_rows(path, expected_identity):
    """Read the cached input column, without model outputs or reference answers."""
    if identity(path) != expected_identity:
        raise ValueError('Cached input-only projection changed')
    counts = Counter()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            source = row['source_identity']
            index = source['raw_index']
            if (set(row) != {'pool','input','source_identity'} or source['dataset'] != DATASET
                    or source['revision'] != SOURCE_REVISION or source['split'] != 'train'
                    or not 0 <= index < 50000 or row['pool'] != ('train' if index < 30000 else 'evaluation')):
                raise ValueError('Cached projection does not match the pinned input-only reservoirs')
            counts[index] += 1
            yield row
    if set(counts) != set(range(50000)) or any(n != 1 for n in counts.values()):
        raise ValueError('The original bounded source reservoirs are incomplete or duplicated')


def choose_evaluation(prepared_rows, history, original, *, count=1024):
    """Apply the original four-key exclusion rule, then the new sealed hash rank."""
    merged = merge_history([history, history_fingerprints(original)])
    eligible, excluded = exclude_inputs(prepared_rows, merged)
    candidates = sorted((row for row in eligible if row['pool'] == 'evaluation'),
                        key=lambda row: rank(row, EVALUATION_ROLE))
    if len(candidates) < count:
        raise ValueError(f'Fixed evaluation reservoir has {len(candidates)} eligible requests; requires {count}')
    selected = []
    for ordinal, row in enumerate(candidates[:count]):
        selected.append(dict(row, request_id='opencode:'+row['source_identity_sha256'],
            source_id='opencode:'+row['source_identity_sha256'],
            original_index=row['source_identity']['raw_index'], request_index=ordinal,
            panel_ordinal=ordinal, max_new_tokens=2048, stop_token_ids=[151645,151643],
            selection_role=EVALUATION_ROLE, selection_rank=rank(row, EVALUATION_ROLE)[0]))
    return dict(evaluation_requests=selected, warmup_requests=original['warmup_requests']), dict(
        selection_seed=SELECTION_SEED, selection_role=EVALUATION_ROLE,
        eligible_evaluation_requests=len(candidates), selected_evaluation_requests=count,
        excluded=excluded, exclusion_reasons=dict(Counter(reason for row in excluded for reason in row['reasons'])),
        fingerprint_counts={key:len(merged[key]) for key in MATCH_FIELDS},
        training_reselected=False, answers_used=False, generated_lengths_used=False,
        ranking='SHA256 canonical JSON {seed,role,source_identity}; canonical source identity tie-break')


def build_evaluation(projection, tokenizer, history, original):
    prepared = [prepare_input(row, tokenizer) for row in projection]
    return choose_evaluation(prepared, history, original)


def select_witness(original):
    """Choose requests now; BASE feedback will determine the independent anchors."""
    first_event = original['train_requests'][:128]
    selected = sorted(first_event, key=lambda row: rank(row, WITNESS_ROLE))[:8]
    return dict(selection_seed=SELECTION_SEED, selection_role=WITNESS_ROLE,
        origin='released_draft', conditioning='BASE_event1_shared_TRAIN',
        responses=[dict(request_id=row['request_id'], source_identity=row['source_identity'],
            training_ordinal=row['request_index'], selection_rank=rank(row,WITNESS_ROLE)[0]) for row in selected],
        anchor_seed=SELECTION_SEED, anchors_per_response=8, anchor_replacement=False,
        anchor_binding='seal independently after BASE event1 feedback, before its first optimizer update',
        endpoints=['released','BASE1','BASE16','BASE32','BASE58','TAIL1','TAIL16','TAIL32','TAIL58'],
        maximum_response_forwards=72)
