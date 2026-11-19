"""Bound the new full-C run and retire only validated, redundant payloads.

The 20 GB compact allowance sizes future writes; it is not a deletion trigger
or separate cap. All actual run, source and compiler growth is charged against
100 GB, while the shared filesystem keeps 400 GB after projected peak writes.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

from experiments.kplus_full_gsm8k import storage as native
from experiments.ok_coding.storage import identity
from experiments.osd_persistent.retention import _inside
from online_feedback.storage import write_json

ALLOCATION_LIMIT = 100_000_000_000
SHARED_RESERVE = 400_000_000_000
COMPACT_ALLOWANCE = 20_000_000_000
ARM = 'KPLUS_C'


def capacity_plan(weight_bytes, *, complete_state_bytes=None, source_runtime_bytes=1_000_000_000,
                  external_roots=(), compiler_bytes=4*2**30, workspace_bytes=1_000_000_000):
    """Price all rolling/atomic overlap, without credit for future retirement.

    Complete-state metadata can grow with all 58 compact event histories. The
    1 GB state-metadata allowance is separate from the response/trace allowance.
    Qualification payloads are retired before production and fit below this
    production peak; its initial, active recovery and two exports are listed.
    """
    if weight_bytes <= 0:
        raise ValueError('A positive measured draft weight size is required')
    complete = max(3*weight_bytes, complete_state_bytes or 0) + 1_000_000_000
    initial = weight_bytes + 10_000_000
    rows = 128*2272
    features = (rows+2272+1)*(4096*6*2+12)
    rolling = initial + 3*complete + 2*weight_bytes
    qualification = initial + complete + 2*weight_bytes
    tensor_peak = max(rolling+features, qualification)
    padded = (tensor_peak*105+99)//100
    peak = padded+COMPACT_ALLOWANCE+source_runtime_bytes+compiler_bytes+workspace_bytes
    if peak > ALLOCATION_LIMIT:
        raise ValueError('Measured full-C peak exceeds the fixed 100 GB allocation limit')
    return dict(schema='full_c_storage_v1', allocation_limit_bytes=ALLOCATION_LIMIT,
        reserve_bytes=SHARED_RESERVE, weight_bytes=weight_bytes,
        complete_state_bound_bytes=complete, initial_state_bound_bytes=initial,
        complete_state_metadata_allowance_bytes=1_000_000_000,
        rolling_complete_copies=3, rolling_complete_roles=['latest','predecessor','atomic_temporary'],
        initial_retained_separately=True, publication_export_copies=2,
        feature_peak_bytes=features, rolling_tensor_bytes=rolling,
        qualification_tensor_bytes=qualification, tensor_peak_bytes=tensor_peak,
        tensor_peak_with_margin_bytes=padded, compact_bytes=COMPACT_ALLOWANCE,
        compact_is_subcap=False, source_runtime_bytes=source_runtime_bytes,
        external_roots=list(map(str,external_roots)), compiler_bytes=compiler_bytes,
        workspace_bytes=workspace_bytes, projected_peak_bytes=peak,
        required_initial_free_bytes=peak+SHARED_RESERVE,
        offload_assumed=False, lifecycle='validated rolling successor; one current feedback window')


def _tree(root):
    if not root.exists():
        return []
    return [root,*root.rglob('*')] if root.is_dir() else [root]


def allocation_snapshot(root, external_roots=(), *, source_policy=None):
    """Count physical bytes once per inode across every explicitly charged root."""
    root = _inside(root, root, directory=True)
    categories = {'tensor':[], 'compact':[], 'external':[]}
    for path in _tree(root):
        categories['tensor' if path.suffix in ('.pt','.safetensors') else 'compact'].append(path)
    for value in external_roots:
        external = Path(value).absolute()
        if external.resolve() != external or external == root or external.is_relative_to(root) or root.is_relative_to(external):
            raise ValueError('Charge external copies through separate real, non-overlapping roots')
        categories['external'].extend(_tree(external))
    source_links = None
    seen, counts = set(), {}
    for name, paths in categories.items():
        count = 0
        for path in paths:
            if path.is_symlink():
                if source_links is None:
                    from experiments.persistent_prefix_withholding.task_runner import _authenticated_source_links
                    source_links = {} if source_policy is None else _authenticated_source_links(source_policy)
                declared = source_links.get(path.absolute())
                if (declared is None or str(path.readlink()) != declared[1]
                        or not path.exists() or not path.resolve().is_relative_to(declared[0].resolve())):
                    raise ValueError('Only authenticated internal source symlinks may be charged: '+str(path))
            try:
                stat = path.lstat()
            except FileNotFoundError:  # Atomic metadata replacement or verified retirement.
                continue
            key=(stat.st_dev,stat.st_ino)
            if key not in seen:
                count += stat.st_blocks*512;seen.add(key)
        counts[name+'_bytes'] = count
    return dict(**counts, total_bytes=sum(counts.values()),
                accounting='st_blocks*512; unique device/inode including directory metadata')


def admit(root, plan, *, receipt_path=None, arm=ARM):
    if arm != ARM or (plan['allocation_limit_bytes'],plan['reserve_bytes'],plan['compact_is_subcap']) != (ALLOCATION_LIMIT,SHARED_RESERVE,False):
        raise ValueError('Full C storage policy or method changed')
    root = _inside(root,root,directory=True)
    source_keys = ('source_root','source_manifest','execution_commit')
    source_policy = {key:plan[key] for key in source_keys} if all(key in plan for key in source_keys) else None
    if source_policy is not None:
        source_policy['prior_source_manifests'] = plan.get('prior_source_manifests', [])
    current = allocation_snapshot(root,plan['external_roots'],source_policy=source_policy)
    compiler = native.compiler_growth(root)
    cache = compiler['compiler_growth_bytes']
    charged = current['total_bytes']+cache
    remaining = dict(tensors=max(0,plan['tensor_peak_with_margin_bytes']-current['tensor_bytes']),
        compact=max(0,plan['compact_bytes']-current['compact_bytes']),
        source_runtime=max(0,plan['source_runtime_bytes']-current['external_bytes']),
        compiler=max(0,plan['compiler_bytes']-cache), workspace=plan['workspace_bytes'])
    remaining_bytes = sum(remaining.values())
    peak = charged+remaining_bytes
    free = shutil.disk_usage(root).free
    passed = charged <= ALLOCATION_LIMIT and peak <= ALLOCATION_LIMIT and free >= SHARED_RESERVE+remaining_bytes
    result = dict(status='passed' if passed else 'failed', free_bytes=free,
        charged_allocation_bytes=charged, projected_peak_bytes=peak,
        remaining_projected_write_bytes=remaining_bytes,
        required_additional_and_reserve_bytes=remaining_bytes+SHARED_RESERVE,
        remaining_components=remaining, compact_allowance_excess_bytes=max(0,current['compact_bytes']-COMPACT_ALLOWANCE),
        compact_is_subcap=False, current=current, plan=plan, **compiler)
    if receipt_path is not None:
        write_json(receipt_path,result)
    if not passed:
        raise ValueError('Full C storage admission fails the 100 GB allocation or 400 GB shared reserve')
    return result


def check_growth(root, plan, *, arm=ARM):
    return admit(root,plan,arm=arm)


def checkpoint_fingerprints(path, *, expected_identity, export_path, receipt_path):
    """Identify every model/Adam/RNG tensor before retiring its recovery file."""
    import torch
    from experiments.ok_coding.training import _export_equal
    from experiments.persistent_prefix_withholding.qualification import tree_identity
    began=time.perf_counter();path=Path(path)
    if identity(path) != expected_identity:
        raise ValueError('Predecessor recovery changed before fingerprinting')
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    _export_equal(state,export_path)
    events=state['events']
    encoded=json.dumps(events,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    result=dict(status='passed',path=str(path),identity=expected_identity,
        byte_convention='tensor contiguous CPU bytes, little-endian native dtype; SHA256',
        tensors=tree_identity({'draft_state_dict':state['draft_state_dict'],'backend':state['backend'],'rng':state['rng']}),
        structure_keys=sorted(state),global_step=state['global_step'],presentations=state['presentations'],
        horizon=state['horizon'],accumulation_steps=state['accumulation_steps'],
        algorithm_contract=state['algorithm_contract'],config_text=state['config_text'],run_id=state['run_id'],
        event_history=dict(count=len(events),event_ids=[e['event_id'] for e in events],
            canonical_json_sha256=hashlib.sha256(encoded).hexdigest(),canonical_json_bytes=len(encoded)),
        export_path=str(export_path),export_identity=identity(Path(export_path)/'model.safetensors'),
        exact_export_tensors=True,wall_s=time.perf_counter()-began,
        lost_capability='Fingerprint is not an archive; direct resume from this predecessor is retired.')
    write_json(receipt_path,result)
    return result


def feature_fingerprints(path):
    """Preserve actual file identity and alignment/shape evidence for one packet."""
    import torch
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    tensors={name:dict(shape=list(value.shape),dtype=str(value.dtype),elements=value.numel())
             for name,value in payload.items() if torch.is_tensor(value)}
    return dict(path=str(path),identity=identity(path),tensors=tensors,
        record_id=payload.get('record_id'),prompt_length=payload.get('prompt_length'),
        response_tokens=payload.get('response_tokens'),label_binding=payload.get('label_binding'))


def retire_redundant_features(record, prepared_receipt, root, event_root):
    """Retain the original exact native/terminal-to-prepared equivalence check."""
    return native.retire_redundant_features(record,prepared_receipt,root,event_root)


def retire_event(root, event_root, records, receipt, previous_export,
                 previous_boundary, current_boundary, *, arm=ARM):
    """Fingerprint first, then invoke PR25's validated transactional retirement."""
    if arm != ARM:
        raise ValueError('This retirement adapter owns only the new KPLUS_C trajectory')
    root=_inside(root,root,directory=True);event_root=_inside(event_root,root,directory=True)
    if (event_root/'RETIREMENT.json').exists() or (event_root/'RETIREMENT.plan.json').exists():
        raise ValueError('Retirement already attempted; preserve and reconcile its receipt')
    if previous_export is not None:
        if previous_boundary is None:
            raise ValueError('Previous publication needs its authenticated complete boundary')
        previous_state = _inside(previous_boundary['state'],root)
        if previous_state == Path(current_boundary['state']):
            raise ValueError('The current complete recovery cannot be a retirement candidate')
        if identity(Path(previous_export)/'model.safetensors') != previous_boundary['weights_identity']:
            raise ValueError('Predecessor export changed before retirement')
        checkpoint_fingerprints(previous_state,
            expected_identity=previous_boundary['state_identity'],export_path=previous_export,
            receipt_path=event_root/'PREDECESSOR_TENSOR_FINGERPRINTS.json')
    packets=[]
    for item in receipt['preparation']:
        packets.append(feature_fingerprints(_inside(item['feature_path'],root)))
    write_json(event_root/'FEATURE_RETIREMENT_IDENTITIES.json',dict(status='passed',packets=packets,
        lost_capability='Prepared tensor packets are retired, not archived; compact response/anchor evidence remains.'))
    return native.retire_event(root,event_root,records,receipt,previous_export,
        previous_boundary,current_boundary,arm=arm)


def retire_qualification(plan, qualification):
    """Retire this run's disposable payloads only after both native gates pass.

    Durable qualification receipts, tensor fingerprints and CPU export checks
    remain. No production initial/recovery or historical path is eligible.
    """
    from experiments.ok_continuation.resume import differences
    root=_inside(plan['root'],plan['root'],directory=True)
    training=qualification['training'];serving=qualification['serving']
    detail=training['result']
    if (qualification.get('status')!='passed' or training.get('status')!='completed'
            or serving.get('status')!='completed' or detail.get('status')!='passed'
            or detail.get('cpu_reconstruction_verified') is not True
            or detail['cpu_export'].get('status')!='passed'
            or detail['cpu_export'].get('exact_weights_file_identity') is not True):
        raise ValueError('Disposable retirement requires passed training, CPU export and serving gates')
    base=_inside(root/'qualification/training_B',root,directory=True)
    initial=base/'trainer/initial_state.pt';active=base/'active_state.pt'
    export=base/'active_export';reconstructed=base/'cpu_reconstructed_export'
    expected_paths=[initial,active,export,reconstructed]
    if (detail['initialization']['state_path']!=str(initial) or detail['snapshot_path']!=str(active)
            or detail['export']!=str(export)
            or set(detail['disposable_payload_paths'])!=set(map(str,expected_paths))):
        raise ValueError('Disposable allowlist does not match the sealed qualification layout')
    destination=root/'QUALIFICATION_RETIREMENT.json'
    if destination.exists() or destination.with_suffix('.plan.json').exists():
        raise ValueError('Qualification retirement already attempted; reconcile the original receipt')
    before=checkpoint_fingerprints(initial,expected_identity=identity(initial),export_path=plan['draft'],
        receipt_path=base/'INITIAL_RETIREMENT_FINGERPRINTS.json')
    observed=dict(weights=before['tensors']['draft_state_dict'],optimizer=before['tensors']['backend'],
                  rng=before['tensors']['rng'])
    if differences(observed,detail['origin_before']):
        raise ValueError('Disposable initial state differs from the qualified released witness')
    endpoint=checkpoint_fingerprints(active,expected_identity=identity(active),export_path=export,
        receipt_path=base/'ACTIVE_RETIREMENT_FINGERPRINTS.json')
    expected=next(row['endpoint'] for row in detail['cases'] if row['case']=='active')
    observed=dict(weights=endpoint['tensors']['draft_state_dict'],optimizer=endpoint['tensors']['backend']['optimizer'],
        rng=endpoint['tensors']['rng'],global_step=endpoint['global_step'],presentations=endpoint['presentations'])
    if differences(observed,{key:expected[key] for key in observed}):
        raise ValueError('Disposable active recovery differs from the qualified endpoint')
    for folder in (export,reconstructed):
        if identity(folder/'model.safetensors')!=detail['cpu_export']['weights_files']['model.safetensors']:
            raise ValueError('Disposable serving export differs from the qualified CPU reconstruction')
    actual_files=[initial,active,*sorted(p for p in export.rglob('*') if p.is_file()),
                              *sorted(p for p in reconstructed.rglob('*') if p.is_file())]
    if set(map(str,actual_files))!=set(detail['disposable_payload_files']):
        raise ValueError('Disposable payload inventory changed after qualification')
    evidence=[root/'QUALIFICATION.json',base/'DETAILS.json',base/'CPU_EXPORT_QUALIFICATION.json',
        base/'INITIAL_RETIREMENT_FINGERPRINTS.json',base/'ACTIVE_RETIREMENT_FINGERPRINTS.json',
        root/'phases/qualification_training/STATUS.json',root/'phases/qualification_serving/STATUS.json']
    return native._retire(root,destination,[(p,'disposable_qualification_payload') for p in actual_files],evidence,
        reason='Passed native training/serving and exact CPU reconstruction; disposable model states never enter production.',
        retained={'qualification_records':str(base),'original_initial_witness':plan['initial_witness'],
                  'production_state':'not initialized or touched by this operation'})
