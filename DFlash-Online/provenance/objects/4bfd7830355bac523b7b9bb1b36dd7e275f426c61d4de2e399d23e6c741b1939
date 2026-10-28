"""Seal disjoint next-pair reference bindings from the original K/O trajectory.

This module does not sample anchors or inspect model outputs. A reference pair
always uses its own recorded anchors from the same pass, including cyclic wrap.
"""
from copy import deepcopy

SCHEMA = 'next_pair_reference_schedule_v1'
PASSES = 2
BATCHES_PER_PASS = 64
RESPONSES_PER_BATCH = 2


def _binding(record):
    return deepcopy({key: record[key] for key in ('record_id', 'pass', 'sampled_anchors')})


def build_reference_schedule(receipt, *, evaluation_ids=()):
    """Return the complete 128-window scientific schedule before generation."""
    records = receipt['records']
    features = receipt['features']
    if len(records) != 256 or len(features) != 128:
        raise ValueError('The reference schedule requires 128 responses and 256 presentations')
    feature_ids = {row['record_id'] for row in features}
    if len(feature_ids) != 128:
        raise ValueError('Feature identities must be unique')
    if feature_ids & set(evaluation_ids):
        raise ValueError('Evaluation requests cannot enter learning or reference exposure')
    windows = []
    for pass_index in range(PASSES):
        pass_records = records[128*pass_index:128*(pass_index+1)]
        if (any(row['pass'] != pass_index for row in pass_records)
                or {row['record_id'] for row in pass_records} != feature_ids):
            raise ValueError('Each pass must retain the original complete response pool exactly once')
        for offset, row in enumerate(pass_records):
            if (row['optimizer_stepped'] is not bool(offset % 2)
                    or row['optimizer_step_after'] != 64*pass_index+(offset+1)//2):
                raise ValueError('Original two-response optimizer boundaries changed')
            if not row['sampled_anchors']:
                raise ValueError('Every presentation requires its original sampled-anchor binding')
        pairs = [pass_records[2*b:2*b+2] for b in range(BATCHES_PER_PASS)]
        for batch_index, learning in enumerate(pairs):
            reference_index = (batch_index+1) % BATCHES_PER_PASS
            reference = pairs[reference_index]
            learning_ids = [row['record_id'] for row in learning]
            reference_ids = [row['record_id'] for row in reference]
            if set(learning_ids) & set(reference_ids):
                raise ValueError('Learning and reference pair identities must be disjoint')
            windows.append(dict(
                step=pass_index*BATCHES_PER_PASS+batch_index+1,
                pass_index=pass_index, batch_index=batch_index,
                reference_batch_index=reference_index,
                learning_record_ids=learning_ids, reference_record_ids=reference_ids,
                learning=[_binding(row) for row in learning],
                reference=[_binding(row) for row in reference]))
    return dict(schema=SCHEMA, passes=PASSES, batches_per_pass=BATCHES_PER_PASS,
                responses_per_batch=RESPONSES_PER_BATCH, features=deepcopy(features), windows=windows,
                exposure_scope='Reference responses are additional training exposure, not held-out evaluation.')


def validate_reference_schedule(schedule, receipt, *, evaluation_ids=()):
    """Rebuild and compare every identity, pass boundary and anchor binding."""
    expected = build_reference_schedule(receipt, evaluation_ids=evaluation_ids)
    if schedule != expected:
        raise ValueError('Sealed reference schedule differs from the original next-pair trajectory')
    return dict(passed=True, windows=128, learning_presentations=256,
                reference_presentations=256, disjoint_pairs=128,
                within_pass_wraps=[64, 128], original_per_pass_anchors=True,
                no_evaluation_overlap=True)
