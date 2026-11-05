"""Admit collection, then the remaining writes for one fresh OSD event.

Native captures already occupy disk at the second gate. The pending trainer
state is renamed to latest, so this fresh event needs one state and one export.
Nothing is retired or offloaded by these checks.
"""
from pathlib import Path
import shutil

from online_feedback.storage import write_json

GIB = 2**30
FEATURE_ROW_BYTES = 4096 * (5 + 1) * 2
RESERVE_BYTES = 8 * GIB
WORK_BYTES = 2 * GIB
COMPACT_BYTES = 8 * GIB


def budget(remaining_tensor_bytes, free_bytes, *, compact_present=0):
    remaining_compact = max(0, COMPACT_BYTES - compact_present)
    tensors_with_margin = (remaining_tensor_bytes * 105 + 99) // 100
    required = tensors_with_margin + remaining_compact + WORK_BYTES + RESERVE_BYTES
    return {
        'remaining_tensor_bytes': remaining_tensor_bytes,
        'tensor_bytes_with_five_percent_margin': tensors_with_margin,
        'compact_allowance_bytes': COMPACT_BYTES,
        'compact_present_bytes': compact_present,
        'remaining_compact_allowance_bytes': remaining_compact,
        'work_allowance_bytes': WORK_BYTES,
        'reserve_bytes': RESERVE_BYTES,
        'required_additional_and_reserve_bytes': required,
        'free_bytes': free_bytes,
        'admitted': free_bytes >= required and compact_present <= COMPACT_BYTES,
    }


def _admit(root, phase, remaining_tensor_bytes, details):
    compact_present = sum(p.stat().st_size for p in root.rglob('*')
                          if p.is_file() and p.suffix not in {'.pt', '.safetensors'})
    receipt = {
        'phase': phase,
        **details,
        **budget(remaining_tensor_bytes, shutil.disk_usage(root).free,
                 compact_present=compact_present),
    }
    write_json(root / f'STORAGE_{phase.upper()}.json', receipt)
    if not receipt['admitted']:
        raise RuntimeError(f'{phase} peak storage would cross the declared reserve')
    return receipt


def admit_collection(root, draft, requests):
    """Only collection is admitted; a second gate precedes teacher preparation."""
    root = Path(root)
    rows = sum(len(r['prompt_ids']) + r['max_new_tokens'] - 1 for r in requests)
    return _admit(root, 'collection', rows * FEATURE_ROW_BYTES, {
        'planned_responses': len(requests),
        'capture_rows_at_full_output_cap': rows,
        'feature_row_bytes': FEATURE_ROW_BYTES,
        'training_is_not_admitted_by_this_receipt': True,
    })


def admit_update(root, draft, records):
    """Use actual completed lengths without changing training's 2272-token cap."""
    root = Path(root)
    lengths = [len(r['prompt_ids']) + len(r['response_ids']) for r in records]
    if len(records) != 128 or max(lengths) > 2272:
        raise ValueError('Single-event feedback count or prepared-length contract changed')
    weights = sum(p.stat().st_size for p in Path(draft).glob('*.safetensors'))
    if weights <= 0 or (root / 'trainer/training_state.pt').exists():
        raise ValueError('Require released weights and no previous trainer checkpoint')
    capture_paths = [Path(r['metadata']['capture']['feature_path']) for r in records]
    capture_bytes = sum(p.stat().st_size for p in capture_paths)
    prepared = sum(lengths) * (FEATURE_ROW_BYTES + 8 + 4)
    terminals = len(records) * FEATURE_ROW_BYTES
    # BF16 parameters, two BF16 Adam moments, and one inference export.
    state_and_export = 4 * weights
    return _admit(root, 'update', prepared + terminals + state_and_export, {
        'completed_responses': len(records),
        'actual_sequence_lengths': lengths,
        'native_capture_bytes_already_present': capture_bytes,
        'prepared_feature_bytes': prepared,
        'terminal_feature_bytes': terminals,
        'trainer_state_bound_bytes': 3 * weights,
        'export_bound_bytes': weights,
        'fresh_state_slots': 1,
        'export_slots': 1,
        'atomic_state_save': 'temporary -> pending -> latest by rename, no old state',
        'native_captures_are_retained_and_already_charged_to_free_space': True,
    })
