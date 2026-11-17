"""Peak admission and authenticated rolling retirement for this new run only."""
from pathlib import Path

from experiments.adaflash_c1.storage import RAW_BYTES_PER_GENERATION
from experiments.functional_step_geometry.pilot_storage import (
    check_current_capacity, retire_previous_boundary, retire_exact,
)
from .contract import ARMS

GIB = 2**30


def capacity_plan(weight_bytes, state_bytes, *, retained_qualification_bytes=0,
                  retrieval_staging=False):
    if (any(type(v) is not int or v <= 0 for v in (weight_bytes, state_bytes))
            or type(retained_qualification_bytes) is not int or retained_qualification_bytes < 0
            or type(retrieval_staging) is not bool):
        raise ValueError('Use measured tensor sizes and explicit retrieval policy')
    boundaries = 9*(weight_bytes+state_bytes)
    features = (128*2272+2272+1)*(4096*6*2+12)
    tensors = boundaries+features+retained_qualification_bytes
    compact = (512+4096+8)*RAW_BYTES_PER_GENERATION+4*GIB
    staging = compact if retrieval_staging else 0
    padded = (tensors*105+99)//100
    return dict(policy='abt_validated_rolling_shared_pools_and_event5_boundaries',
        weight_bytes=weight_bytes, state_bytes=state_bytes,
        tensor_peak_bytes=tensors, tensor_peak_with_margin_bytes=padded,
        shared_feature_peak_bytes=features, retained_final_boundaries=8,
        atomic_boundary_overlap=1, retained_qualification_bytes=retained_qualification_bytes,
        compact_bytes=compact+staging, raw_compact_bytes=compact,
        retrieval_staging_bytes=staging, compiler_bytes=4*GIB, workspace_bytes=2*GIB,
        reserve_bytes=8*GIB, no_future_deletion_credit=True, offload_assumed=False,
        required_initial_free_bytes=padded+compact+staging+14*GIB,
        retention='Retain eight final complete states/exports and compact diagnostics; '
            'retire only own event5 boundaries after exact event6 successors; '
            'retire each shared pool after all four arms and fixed diagnostics validate.')


def retire_shared_pool(plan, dataset, event, pool, results):
    from .data import verify_shared_packets
    from .campaign import validate_update
    root = Path(plan['root'])
    verify_shared_packets(pool)
    if set(results) != set(ARMS):
        raise ValueError('All four completed arm receipts are required')
    for arm, result in results.items():
        validate_update(result, dataset=dataset, arm=arm, event=event,
            baseline=results['KPLUS'] if arm != 'KPLUS' else None)
        if (result['boundary']['status'] != 'passed' or not result['boundary']['export_exact']
                or [r['record_id'] for r in result['features']] != [r['record_id'] for r in pool['records']]
                or [str(Path(r['path']).resolve()) for r in result['features']]
                    != [r['feature']['path'] for r in pool['records']]):
            raise ValueError('Exact shared consumption or boundary evidence is incomplete')
    candidates = []
    for row in pool['records']:
        path = Path(row['feature']['path'])
        if path.is_relative_to(root/'campaign_001'):
            candidates.append((path, {k: row['feature'][k] for k in ('bytes', 'sha256')},
                               'Consumed by all four arms and fixed diagnostics'))
    output = root/'campaign_001'/dataset/f'event_{event:03d}'/'SHARED_RETIREMENT.json'
    return retire_exact(root, candidates, output,
        retained=dict(pool_manifest=str(output.parent/'shared/POOL.json'),
                      qualification_packets='unchanged outside campaign',
                      boundaries={arm:r['checkpoint_path'] for arm,r in results.items()}),
        lost='Prepared-feature replay is retired, not archived; canonical outputs, native traces, '
             'anchor/learning diagnostics and final recovery states remain.')
