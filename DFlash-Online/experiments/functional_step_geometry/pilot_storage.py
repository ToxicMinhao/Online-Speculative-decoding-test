"""Peak storage and narrowly gated rolling retirement for the declared pilot.

Original recovery material and qualification packets are outside retirement.
All sixteen pre/native diagnostic endpoint pairs and all six final boundaries
remain available. Features retire only after all arms and diagnostics finish.
"""
import json
from pathlib import Path
import shutil
import time

from experiments.adaflash_c1.storage import RAW_BYTES_PER_GENERATION
from experiments.ok_coding.storage import identity
from experiments.osd_persistent.retention import _inside
from online_feedback.storage import write_json
from .contract import ARMS, DATASETS

GIB = 2**30


def capacity_plan(weight_bytes, state_bytes):
    if any(type(value) is not int or value <= 0 for value in (weight_bytes,state_bytes)):
        raise ValueError('Use measured complete state and export sizes')
    # Six latest boundaries, plus one new atomic state/export overlap. A prior
    # event-five boundary is retired only after its own event-six is validated.
    boundaries = 7*(weight_bytes+state_bytes)
    # Every original diagnostic saves prestate/native active BF16 tensors.
    diagnostics = 32*weight_bytes
    # Only one shared pool is live. Validated incremental preparation retires
    # redundant raw captures; one capture/terminal row may overlap preparation.
    features = (128*2272+2272+1)*(4096*6*2+12)
    tensors = boundaries+diagnostics+features
    compact = (512+3072+512)*RAW_BYTES_PER_GENERATION+4*GIB
    padded = (tensors*105+99)//100
    return dict(policy='validated_rolling_shared_pools_and_event5_boundaries',
        weight_bytes=weight_bytes,state_bytes=state_bytes,
        tensor_peak_bytes=tensors,tensor_peak_with_margin_bytes=padded,
        diagnostic_endpoint_bytes=diagnostics,diagnostic_endpoint_pairs=16,
        shared_feature_peak_bytes=features,retained_final_boundaries=6,
        atomic_boundary_overlap=1,compact_bytes=compact,compiler_bytes=4*GIB,
        workspace_bytes=2*GIB,reserve_bytes=8*GIB,
        no_future_deletion_credit=True,offload_assumed=False,
        required_initial_free_bytes=padded+compact+14*GIB,
        retention='retain final six complete states/exports and all diagnostic endpoints; '
                  'retire each event5 boundary only after its exact own event6 successor; '
                  'retire shared features only after all three arms and diagnostics validate')


def check_current_capacity(root, plan):
    """Subtract only already-written allocated bytes, never a proposed deletion."""
    root = Path(root)
    seen = set()
    present = 0
    tensors = compact = 0
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Pilot storage must not follow symlinks')
        if path.is_file():
            stat = path.stat()
            key = stat.st_dev,stat.st_ino
            if key not in seen:
                allocated = stat.st_blocks*512
                present += allocated
                if path.suffix in ('.pt','.safetensors'):
                    tensors += allocated
                else:
                    compact += allocated
                seen.add(key)
    required = max(plan['reserve_bytes'], plan['required_initial_free_bytes']-present)
    free = shutil.disk_usage(root).free
    bounded = tensors <= plan['tensor_peak_with_margin_bytes'] and compact <= plan['compact_bytes']
    return dict(status='passed' if free>=required and bounded else 'failed',free_bytes=free,
        required_bytes=required,existing_allocated_bytes=present,
        existing_tensor_bytes=tensors,existing_compact_bytes=compact,within_sealed_byte_bounds=bounded,
        reserve_bytes=plan['reserve_bytes'],future_deletion_credit=0)


def retire_previous_boundary(root, previous, current, output):
    """Retire only this arm's event5 state/export after its decoded event6 check."""
    root = Path(root)
    if (previous['end_step'] != 640 or current['end_step'] != 768
            or previous['method'] != current['method'] or previous['dataset'] != current['dataset']
            or previous['status'] != 'completed' or current['status'] != 'completed'
            or current['boundary']['status'] != 'passed' or not current['boundary']['export_exact']):
        raise ValueError('Only a validated own event-six successor can retire event five')
    for key,path in (('state_identity',current['state_path']),
                     ('weights_identity',Path(current['checkpoint_path'])/'model.safetensors')):
        if identity(path) != current[key]:
            raise ValueError('Successor boundary differs before retirement')
    candidates = [(previous['state_path'],previous['state_identity'],'superseded event-five recovery'),
        (Path(previous['checkpoint_path'])/'model.safetensors',previous['weights_identity'],
         'superseded event-five serving export')]
    return retire_exact(root,candidates,output,retained=dict(state=current['state_path'],
        export=current['checkpoint_path']),lost='Exact event-five optimizer continuation/export replay is retired, not archived.')


def retire_exact(root, candidates, output, *, retained, lost):
    """Write the exact authenticated list before unlinking, preserving failures."""
    output = Path(output)
    if output.exists():
        raise ValueError('A prior retirement must not be retried silently')
    rows = []
    for path,expected,reason in candidates:
        path = _inside(path,root)
        if identity(path) != expected:
            raise ValueError('Retirement candidate changed: '+str(path))
        rows.append(dict(path=str(path),identity=expected,reason=reason))
    receipt = dict(status='started',planned=rows,removed=[],retained=retained,
        lost_capability=lost,before_free_bytes=shutil.disk_usage(root).free)
    write_json(output,receipt)
    started = time.perf_counter()
    try:
        for row in rows:
            Path(row['path']).unlink()
            receipt['removed'].append(row)
            write_json(output,receipt)
        receipt['status']='completed'
    except BaseException as error:
        receipt.update(status='failed',error=repr(error))
        raise
    finally:
        receipt.update(after_free_bytes=shutil.disk_usage(root).free,
            removed_bytes=sum(row['identity']['bytes'] for row in receipt['removed']),
            wall_s=time.perf_counter()-started)
        write_json(output,receipt)
    return receipt


def retire_shared_pool(plan, dataset, event, pool, results):
    """All arms must have consumed exactly the same immutable response packets."""
    from .data import verify_shared_packets
    from .pilot import validate_update
    root = Path(plan['root'])
    verify_shared_packets(pool)
    if set(results) != set(ARMS):
        raise ValueError('All three completed arm receipts are required')
    candidates = []
    for arm in ARMS:
        result = results[arm]
        validate_update(result,dataset=dataset,arm=arm,event=event,
                        baseline=results['KPLUS'] if arm!='KPLUS' else None)
        if (result['boundary']['status']!='passed' or not result['boundary']['export_exact']
                or [row['record_id'] for row in result['features']] != [row['record_id'] for row in pool['records']]
                or [str(Path(row['path']).resolve()) for row in result['features']]
                    != [row['feature']['path'] for row in pool['records']]):
            raise ValueError('Shared consumption or decoded boundary validation is incomplete')
    for row in pool['records']:
        feature = Path(row['feature']['path'])
        if feature.is_relative_to(root/'campaign_001'):
            candidates.append((feature,{key:row['feature'][key] for key in ('bytes','sha256')},
                               'shared prepared feature consumed by all three arms and diagnostics'))
    output=root/'campaign_001'/dataset/f'event_{event:03d}'/'SHARED_RETIREMENT.json'
    return retire_exact(root,candidates,output,
        retained=dict(original_qualification_packets='unchanged outside campaign',
                      pool_manifest=str(output.parent/'shared/POOL.json'),
                      training_receipts={arm:result['checkpoint_path'] for arm,result in results.items()}),
        lost='Prepared-feature replay for retired event packets is unavailable; canonical answers, token traces, '
             'anchor/learning diagnostics and exact diagnostic pre/native weights remain.')
