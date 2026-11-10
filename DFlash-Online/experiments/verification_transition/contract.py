"""Fixed scientific scope and outcome-independent request selection."""
from __future__ import annotations

import hashlib

DATASETS = ('gsm8k', 'mathqa')
CHECKPOINTS = ('KPLUS4', 'KPLUS6')
REQUESTS_PER_DATASET = 64
PANEL_SIZE = 256
BLOCK_SIZE = 16
CONTEXT_LIMIT = 4096
OUTPUT_CAP = 2048
BOOTSTRAP_SEED = 20261110
BOOTSTRAP_RESAMPLES = 10000
EXECUTION_SOURCE = '6440e8ff5af1a845f830d65c884f6711d7e4a687'
SOURCE_REPORT = 'f88a4f42dec8f9b01adc9b8125abd122e9428c43'
LIMITS = {
    'original_completion': 256,
    'crossed_suffix': 256,
    'neutral_suffix': 8,
    'draft_block': 32768,
}
GPU_SECONDS = 4 * 3600


def selection_key(dataset: str, request_id: str) -> tuple[str, str]:
    key = f'kplus_transition_replay_v1\n{dataset}\n{request_id}'
    return hashlib.sha256(key.encode('utf-8')).hexdigest(), request_id


def select_requests(dataset: str, panel: list[dict]) -> list[dict]:
    """Look only at recorded stable IDs; never inspect lengths or outcomes."""
    if dataset not in DATASETS or len(panel) != PANEL_SIZE:
        raise ValueError('Require each original 256-request development panel')
    ids = [row['request_id'] for row in panel]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != PANEL_SIZE:
        raise ValueError('Panel IDs must be nonempty and unique')
    ordered = sorted(ids, key=lambda value: selection_key(dataset, value))
    return [dict(dataset=dataset, selection_rank=rank, request_id=request_id,
                 selection_sha256=selection_key(dataset, request_id)[0])
            for rank, request_id in enumerate(ordered[:REQUESTS_PER_DATASET])]


def interleaved_requests(selected: dict[str, list[dict]]) -> list[dict]:
    """Finish each rank in GSM8K, MathQA order, including unfavorable cases."""
    if set(selected) != set(DATASETS):
        raise ValueError('Both fixed datasets are required')
    for dataset in DATASETS:
        rows = selected[dataset]
        if (len(rows) != REQUESTS_PER_DATASET
                or [r['selection_rank'] for r in rows] != list(range(REQUESTS_PER_DATASET))
                or any(r['dataset'] != dataset for r in rows)):
            raise ValueError('Incomplete or reordered selected panel')
    return [selected[d][i] for i in range(REQUESTS_PER_DATASET) for d in DATASETS]


def neutral_boundaries(rounds: list[dict]) -> list[int]:
    """One-based successors after round 1 and ceil(R/2), excluding terminal."""
    if not rounds:
        return []
    candidates = sorted({1, (len(rounds) + 1) // 2})
    return [index for index in candidates if not rounds[index - 1]['terminal']]


def scientific_config() -> dict:
    return dict(datasets=list(DATASETS), checkpoints=list(CHECKPOINTS),
                requests_per_dataset=REQUESTS_PER_DATASET, block_size=BLOCK_SIZE,
                draft_layers=5, precision='bfloat16', attention='fa3', greedy=True,
                concurrency=1, context_limit=CONTEXT_LIMIT, output_cap=OUTPUT_CAP,
                capture=False, optimizer_steps=0, backwards=0,
                limits=LIMITS.copy(), gpu_seconds=GPU_SECONDS,
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=BOOTSTRAP_RESAMPLES,
                source_execution_commit=EXECUTION_SOURCE, source_report_commit=SOURCE_REPORT)
