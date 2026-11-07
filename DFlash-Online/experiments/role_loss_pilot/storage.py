"""Conservative six-trajectory peak and evidence-gated own-event retirement."""
import json
from pathlib import Path
import shutil

from experiments.kplus_full_gsm8k import storage as native
from experiments.adaflash_c1.storage import RAW_BYTES_PER_GENERATION
from experiments.ok_coding.storage import identity
from experiments.osd_persistent.retention import _inside, _json
from online_feedback.storage import write_json
from .contract import ARMS, DATASETS, trajectory_root

GIB = 2**30
retire_redundant_features = native.retire_redundant_features


def capacity_plan(weight_bytes):
    if type(weight_bytes) is not int or weight_bytes <= 0:
        raise ValueError('Require measured positive draft weight bytes')
    # Five finished cells use 6w each (initial + recovery + two exports).
    # Active peak uses 10w: initial, old/new complete states, retained event2,
    # predecessor export and new export. No historical artifact is retired.
    feature_peak = (128*2272+2272+1)*(4096*6*2+12)
    diagnostic_features = 32*4096*(4096*6*2+12)
    phases = {}
    for name, equivalents, features, generations in (
        ('training',40,feature_peak+diagnostic_features,3072+8+4),
        ('evaluation',36,diagnostic_features,3072+1536+3072+512+4+8)):
        tensors=equivalents*weight_bytes+features
        padded=(tensors*105+99)//100
        compact=generations*RAW_BYTES_PER_GENERATION+4*GIB
        phases[name]=dict(state_weight_equivalents=equivalents,
            tensor_peak_bytes=tensors,tensor_peak_with_margin_bytes=padded,
            compact_bytes=compact,raw_budget_generations=generations,
            required_initial_free_bytes=padded+compact+14*GIB)
    return dict(weight_bytes=weight_bytes,phases=phases,
        feature_peak_bytes=feature_peak,diagnostic_features_bytes=diagnostic_features,
        compiler_bytes=4*GIB,workspace_bytes=2*GIB,reserve_bytes=8*GIB,
        workspace_scope='diagnostic single reconstruction overlap plus retrieval staging',
        required_initial_free_bytes=max(p['required_initial_free_bytes'] for p in phases.values()),
        raw_bytes_per_generation=RAW_BYTES_PER_GENERATION,raw_budget_generations=8204,
        lifecycle='six serial trajectories then both-checkpoint evaluation; all event2/4 exports and final recovery',
        offload_assumed=False,future_retirement_credit=False)


def boundary_files(root):
    paths=[]
    for dataset in DATASETS:
        for arm in ARMS:
            base=trajectory_root(root,dataset,arm)
            initial=base/'trainer/initial_state.pt'
            if initial.exists():
                check=_json(base/'trainer/INITIAL_CHECK.json',root)
                if check['status'] != 'passed' or not check['export_exact']:
                    raise ValueError('Cannot credit unqualified initial state')
                paths.append(initial)
            for marker in sorted((base/'boundaries').glob('event_[0-9][0-9][0-9].json')):
                row=_json(marker,root)
                if row['status']!='passed':
                    raise ValueError('Cannot credit unvalidated own boundary')
                paths.extend(Path(row[key]) for key in ('state','weights') if Path(row[key]).exists())
    for pool_file in (Path(root)/'campaign_001/diagnostics').glob('*/pool/POOL.json'):
        pool=_json(pool_file,root)
        if pool.get('status')!='completed':
            continue
        for feature in pool['features']:
            path=_inside(feature['path'],root)
            paths.append(path)
    return paths


def admit(root,plan,*,receipt_path=None):
    root=_inside(root,root,directory=True)
    if not (root/'COMPILER_BASELINE.json').exists() and any((root/'campaign_001/trajectories').glob('*/*/WORK.jsonl')):
        raise ValueError('Compiler baseline is missing after pilot work started')
    compiler=native.compiler_growth(root)
    present=native.allocated_bytes(_inside(p,root) for p in boundary_files(root))
    compact_paths=[root]
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError('Pilot storage must not follow symlinks')
        if p.is_dir() or p.suffix not in {'.pt','.safetensors'}:
            compact_paths.append(p)
    compact=native.allocated_bytes(compact_paths)
    phases={}
    for name,phase in plan['phases'].items():
        required=max(0,phase['tensor_peak_with_margin_bytes']-present)+max(0,phase['compact_bytes']-compact)
        required+=plan['compiler_bytes']+plan['workspace_bytes']+plan['reserve_bytes']
        phases[name]=required
    required=max(phases.values())
    free=shutil.disk_usage(root).free
    max_compact=max(p['compact_bytes'] for p in plan['phases'].values())
    passed=free>=required and compact<=max_compact and compiler['compiler_growth_bytes']<=plan['compiler_bytes']
    receipt=dict(status='passed' if passed else 'failed',free_bytes=free,
        required_additional_and_reserve_bytes=required,phase_requirements=phases,
        validated_existing_tensor_bytes=present,compact_bytes=compact,
        no_future_deletion_credit=True,plan=plan,**compiler)
    if receipt_path is not None:
        write_json(receipt_path,receipt)
    if not passed:
        raise ValueError('Six-trajectory peak storage admission failed')
    return receipt


check_growth=admit


def retire_event(*,root,event_root,records,receipt,previous_export,previous_boundary,current_boundary,
                 dataset,arm):
    """Retain selected exports; all obsolete optimizer states require a successor."""
    event_root=_inside(event_root,root,directory=True)
    training=receipt['training']; event_id=training['event_id']
    saved=_json(event_root/'loop/events'/f'{event_id}.json',root)
    buffer=_json(event_root/'loop/feedback_buffer/state.json',root)
    marker=_json(event_root/'PUBLICATION_CHECK.json',root)
    if (saved!=receipt or training['status']!='completed' or current_boundary['status']!='passed'
        or marker['status']!='passed' or marker['all_equal'] is not True
        or marker['checkpoint_path']!=training['checkpoint_path'] or marker['event_id']!=event_id
        or buffer['records']!=records or len(buffer['events'])!=1
        or buffer['events'][0]['status']!='committed' or buffer['events'][0]['receipt']!=receipt):
        raise ValueError('Retirement requires a complete validated published event')
    for key in ('state','weights'):
        if identity(current_boundary[key])!=current_boundary[key+'_identity']:
            raise ValueError('Current retained boundary identity changed')
    if [p['record_id'] for p in receipt['preparation']]!=[r['record_id'] for r in records]:
        raise ValueError('Prepared response order differs')
    candidates=[]
    for record,prepared in zip(records,receipt['preparation']):
        prior=_json(event_root/'preparation_retirement'/(record['record_id']+'.json'),root)
        if prior['status']!='completed' or prior['retained']['prepared_packet']!=prepared['feature_path']:
            raise ValueError('Incremental exact feature replacement has not passed')
        candidates.append((_inside(prepared['feature_path'],root),'prepared_feature'))
    retained=dict(current_state=current_boundary['state'],current_weights=current_boundary['weights'],
        initial_state=str(trajectory_root(root,dataset,arm)/'trainer/initial_state.pt'))
    if previous_export is not None:
        previous=_inside(previous_export,root,directory=True)
        if (previous_boundary is None or current_boundary['event']!=previous_boundary['event']+1
            or Path(previous_boundary['weights'])!=previous/'model.safetensors'
            or any(r['checkpoint_id']!=str(previous) for r in records)):
            raise ValueError('Predecessor is not the checkpoint that served this event')
        if identity(previous_boundary['state'])!=previous_boundary['state_identity']:
            raise ValueError('Obsolete own state differs from its validated boundary')
        candidates.append((Path(previous_boundary['state']),'previous_state'))
        if previous_boundary['event'] in (2,4):
            retained['selected_event_export']=previous_boundary['weights']
        else:
            if identity(previous_boundary['weights'])!=previous_boundary['weights_identity']:
                raise ValueError('Obsolete own export differs from its validated boundary')
            candidates.append((Path(previous_boundary['weights']),'previous_export'))
    evidence=list(event_root.rglob('*.json'))
    evidence.append(Path(training['checkpoint_path']).parent/'RESULT.json')
    result=native._retire(root,event_root/'RETIREMENT.json',candidates,evidence,
        reason='Exact successor state/export and publication passed; event2/4 exports remain selected.',
        retained=retained)
    return result
