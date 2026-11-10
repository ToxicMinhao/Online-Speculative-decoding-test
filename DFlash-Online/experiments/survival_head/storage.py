"""Own-run peak budget; no credit for future PR36 retirement or offloading."""
from experiments.functional_step_geometry.pilot_storage import check_current_capacity, retire_exact
from experiments.adaflash_c1.storage import RAW_BYTES_PER_GENERATION


def capacity_plan(weight_bytes, state_bytes, *, feature_bytes=None, max_feature_bytes=None):
    if any(type(value) is not int or value <= 0 for value in (weight_bytes, state_bytes)):
        raise ValueError('Use measured complete origin sizes')
    # Three final states/exports and one atomic overlap. Qualification instead
    # retains one baseline + one current endpoint, retired after exact checks.
    boundaries = 4 * (weight_bytes + state_bytes)
    measured = feature_bytes is not None or max_feature_bytes is not None
    if measured and (any(type(value) is not int or value <= 0 for value in
                         (feature_bytes, max_feature_bytes)) or max_feature_bytes > feature_bytes):
        raise ValueError('Require complete pool bytes and its largest original packet')
    serialization = 128 * 64 * 1024
    if measured:
        # Two qualification replays and one capture/terminal overlap remain
        # above the exact fixed pool, before the existing 5% tensor padding.
        features = feature_bytes + 3 * max_feature_bytes + serialization
    else:
        features = ((128 + 2 + 1) * 2272 + 1) * (4096 * 6 * 2 + 12)
    tensors = boundaries + features + 2**30  # compact head state/qualification
    compact = 774 * RAW_BYTES_PER_GENERATION + 2 * 2**30
    padded = (tensors * 105 + 99)//100
    return dict(policy='three_final_boundaries_one_atomic_overlap',
        weight_bytes=weight_bytes, state_bytes=state_bytes,
        tensor_peak_bytes=tensors, tensor_peak_with_margin_bytes=padded,
        shared_feature_peak_bytes=features, compact_bytes=compact,
        feature_budget_basis='fixed_original_pool_plus_three_largest_packets' if measured else 'maximum_length_pool',
        original_pool_feature_bytes=feature_bytes, largest_original_feature_bytes=max_feature_bytes,
        feature_overlap_copies=3 if measured else None,
        feature_serialization_margin_bytes=serialization if measured else 0,
        compiler_bytes=4*2**30, workspace_bytes=2*2**30, reserve_bytes=8*2**30,
        required_initial_free_bytes=padded+compact+14*2**30,
        no_future_deletion_credit=True, offload_assumed=False)


def require_capacity(plan, output):
    from online_feedback.storage import write_json
    receipt = check_current_capacity(plan['root'], plan['main_storage'])
    write_json(output, receipt)
    if receipt['status'] != 'passed':
        raise ValueError('Insufficient disk for the fixed remaining peak and reserve')
    return receipt
