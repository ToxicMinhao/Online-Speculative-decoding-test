"""Keep two complete CODE boundaries; retire only this study's audited payloads."""
from pathlib import Path

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.kplus_full_gsm8k.storage import (
    _retire as native_retire, retire_redundant_features as native_retire_redundant_features,
)
from experiments.full_c_gsm8k.storage import checkpoint_fingerprints, feature_fingerprints
from .contract import ARMS, RESOURCE_LIMITS
from .runtime import read
from . import protected

BYTES_PER_FEATURE_ROW = 49164  # five BF16 target layers + final layer + IDs/mask


def _retire(root, receipt_path, candidates, compact_paths, **kwargs):
    """Supplement the inherited atomic plan with allocated-byte evidence."""
    candidates=list(candidates)
    protected.require_retirement_paths(root,[path for path,_ in candidates])
    allocation_path=Path(receipt_path).with_suffix('.allocated.plan.json')
    if allocation_path.exists():
        raise FileExistsError('A retirement allocation receipt already exists')
    entries=[]
    for path,kind in candidates:
        path=Path(path)
        stat=path.stat()
        entries.append(dict(path=str(path),kind=kind,allocated_bytes=stat.st_blocks*512,
                            links=stat.st_nlink,uid=stat.st_uid))
    write_json(allocation_path,dict(status='planned',entries=entries))
    try:
        result=native_retire(root,receipt_path,candidates,[*compact_paths,allocation_path],**kwargs)
    finally:
        protected.check(root,purpose='tensor_retirement:'+str(receipt_path))
    write_json(Path(receipt_path).with_suffix('.allocated.json'),dict(status='completed',entries=entries,
        retired_allocated_bytes=sum(row['allocated_bytes'] for row in entries),
        all_exclusive_links=all(row['links']==1 for row in entries),
        shared_free_change_bytes=result['free_after_bytes']-result['free_before_bytes'],
        free_space_difference_may_include_other_users=True))
    return result


def retire_redundant_features(record, prepared_receipt, root, event_root):
    paths=[record['metadata']['capture']['feature_path'],
           prepared_receipt['terminal_feature']['tensor_path']]
    protected.require_retirement_paths(root,paths)
    try:
        return native_retire_redundant_features(record,prepared_receipt,root,event_root)
    finally:
        protected.check(root,purpose='native_feature_retirement:'+record['record_id'])


def capacity_plan(requests, weight_bytes, complete_state_bytes, *,
                  compact_bytes, compiler_bytes, source_bytes, workspace_bytes,
                  state_metadata_growth_bytes, qualification_peak_bytes,
                  final_compact_bytes=None):
    """Use every sealed window's prompt sum and the unchanged 2048 output cap.

    The last arm overlaps two completed arms (two states/one export each),
    three active state copies (current, previous, atomic successor), two active
    exports, and one current feature window. Final qualification tensors must
    be retired after their evidence is authenticated, before production begins.
    """
    if len(requests)!=7424 or any(not 1<=len(r['prompt_ids'])<=2048 for r in requests):
        raise ValueError('Storage planning needs the actual sealed 7424 bounded inputs')
    windows = [sum(len(r['prompt_ids'])+2048 for r in requests[start:start+128])
               for start in range(0,7424,128)]
    replacement_rows = max(len(row['prompt_ids'])+2048 for row in requests)+1
    feature_peak = (max(windows)+replacement_rows)*BYTES_PER_FEATURE_ROW
    state = complete_state_bytes+state_metadata_growth_bytes
    retained_and_atomic = 7*state+4*weight_bytes
    tensors = max(retained_and_atomic+feature_peak, qualification_peak_bytes)
    tensor_margin = (tensors*105+99)//100
    final_compact_bytes = compact_bytes if final_compact_bytes is None else final_compact_bytes
    common = dict(compiler_bytes=compiler_bytes,source_bytes=source_bytes,
                  workspace_bytes=workspace_bytes)
    phases = dict(
        training=dict(tensor_peak_with_five_percent_margin_bytes=tensor_margin,
            compact_bytes=compact_bytes,complete_state_copies=7,export_copies=4,
            feature_peak_bytes=feature_peak,**common),
        evaluation=dict(tensor_peak_with_five_percent_margin_bytes=((6*state+3*weight_bytes)*105+99)//100,
            compact_bytes=final_compact_bytes,complete_state_copies=6,export_copies=3,
            feature_peak_bytes=0,**common))
    for phase in phases.values():
        phase['projected_peak_bytes'] = sum(phase[key] for key in (
            'tensor_peak_with_five_percent_margin_bytes','compact_bytes',
            'compiler_bytes','source_bytes','workspace_bytes'))
    peak = max(phase['projected_peak_bytes'] for phase in phases.values())
    result = dict(projected_peak_bytes=peak, bound_bytes=RESOURCE_LIMITS['new_allocated_bytes'],
        windows_full_cap_rows=windows,max_window_full_cap_rows=max(windows),
        feature_replacement_overlap_rows=replacement_rows,
        bytes_per_feature_row=BYTES_PER_FEATURE_ROW,feature_peak_bytes=feature_peak,
        complete_state_bound_bytes=state,complete_state_copies_at_peak=7,
        export_copies_at_peak=4,retained_and_atomic_bytes=retained_and_atomic,
        tensor_peak_with_five_percent_margin_bytes=tensor_margin,
        compact_bytes=compact_bytes,compiler_bytes=compiler_bytes,
        source_bytes=source_bytes,workspace_bytes=workspace_bytes,
        qualification_peak_bytes=qualification_peak_bytes,
        phases=phases,phase_change_requires='All three event58 arms complete; no remaining feedback/training work',
        initial_witness='compact per-arm exact released weight/empty Adam/RNG/config identities',
        final_retention='events57/58 complete states and event58 export for each arm',
        offload_credit=False,output_length_assumed=2048)
    if any(type(x) is not int or x<0 for x in (
            weight_bytes,complete_state_bytes,compact_bytes,compiler_bytes,source_bytes,
            workspace_bytes,state_metadata_growth_bytes,qualification_peak_bytes,final_compact_bytes)):
        raise ValueError('Storage components must be nonnegative measured/bounded integers')
    return {**result,'status':'admitted' if peak<=RESOURCE_LIMITS['new_allocated_bytes'] else 'blocked'}


def _authenticated_marker(marker, *, export=True):
    keys = ('state', 'weights') if export else ('state',)
    if marker['status'] != 'passed' or any(identity(marker[key]) != marker[key+'_identity'] for key in keys):
        raise ValueError('Retained CODE boundary identity changed')


def _export_retirement_evidence(base, marker):
    """Authenticate the exact tensor proof made before this export was retired."""
    number = marker['event']
    receipt_path = base/'boundaries'/f'event_{number:03d}_EXPORT_RETIREMENT.json'
    fingerprint_path = base/'boundaries'/f'event_{number:03d}_TENSOR_FINGERPRINTS.json'
    receipt, fingerprint = read(receipt_path), read(fingerprint_path)
    _authenticated_marker(marker, export=False)
    if (receipt['status'] != 'completed'
            or [row['path'] for row in receipt['removed']] != [marker['weights']]
            or Path(marker['weights']).exists()
            or receipt['compact_evidence'].get(str(fingerprint_path)) != identity(fingerprint_path)
            or fingerprint['status'] != 'passed' or fingerprint['identity'] != marker['state_identity']
            or fingerprint['path'] != marker['state']
            or fingerprint['export_identity'] != marker['weights_identity']
            or fingerprint['exact_export_tensors'] is not True):
        raise ValueError('Earlier export retirement lacks unchanged exact tensor evidence')
    return receipt_path, fingerprint_path


def retire_predecessor_export(root, arm, current):
    """Keep two full states but only the current serving export at a boundary."""
    root = Path(root).resolve()
    if arm not in ARMS or not 2 <= current['event'] <= 58:
        raise ValueError('Require a validated successor CODE boundary')
    base = root/'campaign_001/arms'/arm
    old = read(base/'boundaries'/f"event_{current['event']-1:03d}.json")
    for marker in (old, current):
        _authenticated_marker(marker)
    fingerprint_path = base/'boundaries'/f"event_{old['event']:03d}_TENSOR_FINGERPRINTS.json"
    if fingerprint_path.exists():
        raise FileExistsError('Predecessor retirement was already attempted; inspect its evidence')
    fingerprint = checkpoint_fingerprints(old['state'], expected_identity=old['state_identity'],
        export_path=Path(old['weights']).parent, receipt_path=fingerprint_path)
    fingerprint['lost_capability'] = ('Only the duplicate serving export is retired. The complete recovery '
        'remains; serving this predecessor requires rematerializing its authenticated weights.')
    write_json(fingerprint_path, fingerprint)
    result = _retire(root, base/'boundaries'/f"event_{old['event']:03d}_EXPORT_RETIREMENT.json",
        [(Path(old['weights']), 'redundant_predecessor_export')],
        [fingerprint_path, base/'boundaries'/f"event_{current['event']:03d}.json",
         base/'boundaries'/f"event_{old['event']:03d}.json"],
        reason='Exact complete predecessor recovery retains these weights; the successor export is published.',
        retained=dict(previous_state=old['state'], current_state=current['state'], current_export=current['weights']))
    _export_retirement_evidence(base, old)
    return result


def retire_window(root, arm, event_root, records, receipt, current):
    """Retain current and predecessor states; never reuse the old one-state policy."""
    root, event_root = Path(root).resolve(), Path(event_root).resolve()
    if arm not in ARMS or not event_root.is_relative_to(root/'campaign_001/arms'/arm):
        raise ValueError('Retirement must stay inside its new CODE arm')
    packets = [feature_fingerprints(Path(item['feature_path'])) for item in receipt['preparation']]
    write_json(event_root/'FEATURE_RETIREMENT_IDENTITIES.json',dict(packets=packets,
        lost_capability='Retired prepared tensors are not archived; exact feature replay is unavailable.'))
    training = receipt['training']
    event_id = training['event_id']
    saved = read(event_root/'loop/events'/(event_id+'.json'))
    buffer = read(event_root/'loop/feedback_buffer/state.json')
    published = read(event_root/'PUBLICATION_CHECK.json')
    if (saved!=receipt or training['status']!='completed' or current['status']!='passed'
            or published['status']!='passed' or published['all_equal'] is not True
            or published['event_id']!=event_id
            or published['checkpoint_path']!=training['checkpoint_path']
            or buffer['records']!=records or len(buffer['events'])!=1
            or buffer['events'][0]['status']!='committed' or buffer['events'][0]['receipt']!=receipt):
        raise ValueError('Feature retirement needs a committed and validated publication')
    for key in ('state','weights'):
        if identity(current[key])!=current[key+'_identity']:
            raise ValueError('Published recovery/export changed')
    if [row['record_id'] for row in receipt['preparation']]!=[row['record_id'] for row in records]:
        raise ValueError('Preparation/response order changed')
    candidates = []
    for record, prepared in zip(records,receipt['preparation']):
        earlier = read(event_root/'preparation_retirement'/(record['record_id']+'.json'))
        if earlier['status']!='completed' or earlier['retained']['prepared_packet']!=prepared['feature_path']:
            raise ValueError('Incremental native/prepared feature equivalence is missing')
        candidates.append((Path(prepared['feature_path']),'prepared_feature'))
    initial_witness = root/'campaign_001/arms'/arm/'trainer/INITIAL_CHECK.json'
    read(initial_witness)
    feature_receipt = _retire(root,event_root/'RETIREMENT.json',candidates,
        [*event_root.rglob('*.json'),Path(training['checkpoint_path']).parent/'RESULT.json',initial_witness],
        reason='Complete event published; exact feature fingerprints and compact evidence retained.',
        retained=dict(current_state=current['state'],current_weights=current['weights'],
                      initial_witness=str(initial_witness)))
    event = current['event']
    base = root/'campaign_001/arms'/arm
    spare_export = retire_predecessor_export(root, arm, current) if event>=2 else None
    if event<=2:
        return dict(features=feature_receipt, predecessor_export=spare_export, old_boundary=None)
    old = read(base/'boundaries'/f'event_{event-2:03d}.json')
    predecessor = read(base/'boundaries'/f'event_{event-1:03d}.json')
    _authenticated_marker(current)
    _export_retirement_evidence(base, predecessor)
    export_receipt, fingerprint = _export_retirement_evidence(base, old)
    retired = _retire(root,event_root/'OLDER_BOUNDARY_RETIREMENT.json',
        [(Path(old['state']),'older_state')],
        [fingerprint,export_receipt,base/'boundaries'/f'event_{event:03d}.json',
         base/'boundaries'/f'event_{event-1:03d}.json'],
        reason='Two newer complete CODE boundaries validated and published; outside rolling retention.',
        retained=dict(current=current,previous=predecessor))
    return dict(features=feature_receipt,predecessor_export=spare_export,old_boundary=retired)


def retire_penultimate_export(root, arm):
    """After arm58, keep full57/58 recovery but only the needed58 serving export."""
    root = Path(root).resolve()
    base = root/'campaign_001/arms'/arm
    old, final = [read(base/'boundaries'/f'event_{number:03d}.json') for number in (57,58)]
    if arm not in ARMS or final['status']!='passed':
        raise ValueError('Only a completed CODE arm can retire its spare57 export')
    _authenticated_marker(final)
    receipt, _ = _export_retirement_evidence(base, old)
    return read(receipt)  # Window58 already retired this duplicate, without repeating deletion.
