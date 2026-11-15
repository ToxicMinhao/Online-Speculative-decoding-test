"""Authenticate historical measurements independently of absent E tensors."""
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import time

from online_feedback.storage import write_json
from experiments.threatened_repair_replay.prepare import identity, read, authenticate_source
from experiments.native_prefix_loss.probe_client import score_observation
from .identity_gate import check_identity, norm_agrees

WINDOWS = tuple(range(8,129,8))


def historical_bindings(spec, output):
    """Bind full-precision norms and raw verdicts to the original source/plan."""
    started=time.perf_counter(); output=Path(output);output.mkdir(parents=True,exist_ok=True)
    check_identity(spec['historical_plan'],spec['historical_plan_identity'])
    old=read(spec['historical_plan'])
    check_identity(spec['historical_archive_manifest'],spec['historical_archive_manifest_identity'])
    archive=read(spec['historical_archive_manifest'])
    if old['execution_commit']!='2549f02d9b1edba8fb86f8e91a497e68c224d586':
        raise ValueError('Require original PR47 execution002 source')
    source=authenticate_source(old['source_manifest'],old['input_identities'][old['source_manifest']],old['execution_commit'])
    original_root=Path(old['root'])
    for path,expected in spec['historical_receipt_identities'].items():
        check_identity(path,expected)
    exports=read(original_root/'training/PROBE_EXPORTS.json')
    updates=read(original_root/'training/UPDATE_RECEIPTS.json')
    csv_rows=list(csv.DictReader(Path(spec['public_norms']).open()))
    norms={int(row['window']):row for row in csv_rows}
    export_map={(row['window'],row['state']):row for row in exports}
    update_map={(row['window'],row['state']):row for row in updates}
    if set(norms)!=set(WINDOWS) or len(export_map)!=80:
        raise ValueError('Original norm/export population changed')
    windows={};live=deepcopy(old['input_identities']); inventory=[]
    for window in WINDOWS:
        value={}
        for state in ('P','N','E'):
            row=deepcopy(export_map[(window,state)])
            path=Path(row['checkpoint'])/'model.safetensors'
            row['available']=path.is_file()
            if row['available']:
                check_identity(path,row['weights_identity']);live[str(path)]=row['weights_identity']
            inventory.append(dict(window=window,state=state,path=str(path),available=row['available'],identity=row['weights_identity']))
            value[state]=row
        for state in ('E','N'):
            raw=update_map[(window,state)]
            if raw['status']!='completed' or raw['window']!=window or raw['state']!=state or raw['start_step']!=767+window:
                raise ValueError('Historical update/window/prestate binding differs')
            norm=raw['displacement']['difference_norm']
            if float(norms[window][state])!=norm:
                raise ValueError('Published norm differs from full-precision original receipt')
            value[state+'_norm']=norm
        witness=original_root/f'training/witnesses/P_{window:03d}_fixed_boundary.pt'
        value['witness']=str(witness) if window in (8,128) else None
        if value['witness']:
            # CPU inventory authenticates the surviving witness file now; the
            # original PR47 witness receipt supplies its retained path/size.
            recorded=next(row for row in read(original_root/'training/WITNESSES.json') if row['window']==window)
            if str(witness)!=recorded['path'] or witness.stat().st_size!=recorded['bytes']:
                raise ValueError('Boundary witness path/size changed')
            live[str(witness)]=identity(witness)
        windows[str(window)]=value
    missing=Counter(row['state'] for row in inventory if not row['available'])
    if missing!=Counter(P=14,N=14,E=14):
        raise ValueError('Surviving/missing inventory differs; review rather than guessing replacements')
    keys=read(old['probe_keys']); key_map={(k['request_id'],k['canonical_start'],k['round_ordinal']):k for k in keys}
    if len(key_map)!=256 or Counter(k['request_id'] for k in keys)!=Counter({row['request_id']:4 for row in old['probe_requests']}):
        raise ValueError('Probe population changed')
    parents={rid:read(path) for rid,path in old['parent_records'].items()}
    outcomes=original_root/'probes/PROBE_OUTCOMES.jsonl'; rows=[]; raw_bindings=[];seen=set()
    for line in outcomes.read_text().splitlines():
        row=json.loads(line)
        if row['state'] not in ('P','N','E'):
            continue
        k=(row['request_id'],row['canonical_start'],row['round_ordinal'])
        key=(row['window'],row['state'],k)
        if key in seen or k not in key_map or row['window'] not in WINDOWS:
            raise ValueError('Historical outcome duplicated or outside sealed keys')
        seen.add(key)
        export=export_map[(row['window'],row['state'])]
        if row['checkpoint']!=export['checkpoint'] or not row['valid']:
            raise ValueError('Historical outcome checkpoint or validity differs')
        raw_path=original_root/'probes'/row['execution_id']
        retained_raw_identity=archive['probes/'+row['execution_id']]
        check_identity(raw_path,retained_raw_identity)
        raw=read(raw_path)
        scored=score_observation(parents[row['request_id']],key_map[k],raw,window=row['window'],state=row['state'],checkpoint=row['checkpoint'])
        for field in ('accepted_prefix','valid_proposals','first_error','matched_eos','censored_at_cap','valid'):
            if scored[field]!=row[field]:
                raise ValueError('Raw historical verdict does not reproduce retained score: '+str(key))
        raw_bindings.append(dict(key=list(key[:2])+list(k),path=str(raw_path),identity=identity(raw_path),
                                 checkpoint_identity=export['weights_identity']))
        rows.append(row)
    if len(rows)!=3*16*256:
        raise ValueError('Historical P/N/E outcome grid is incomplete')
    # Bind the supplied public table too; it is not used as a replacement for raw receipts.
    public=list(csv.DictReader(Path(spec['public_outcomes']).open()))
    public_map={(int(r['window']),r['state'],r['request_id'],int(r['canonical_start']),int(r['round_ordinal'])):r for r in public}
    for row in rows:
        pub=public_map[(row['window'],row['state'],row['request_id'],row['canonical_start'],row['round_ordinal'])]
        if int(pub['accepted_prefix'])!=row['accepted_prefix'] or pub['valid'].lower()!='true':
            raise ValueError('Published historical outcome does not reconcile')
    write_json(output/'HISTORICAL_RAW_BINDINGS.json',raw_bindings)
    write_json(output/'EXPORT_INVENTORY.json',inventory)
    receipt=dict(status='passed',execution_commit=old['execution_commit'],historical_execution_commit=old['execution_commit'],historical_source=source,
                 old_plan_identity=identity(spec['historical_plan']),outcomes_identity=identity(outcomes),
                 update_receipts_identity=identity(original_root/'training/UPDATE_RECEIPTS.json'),
                 export_receipts_identity=identity(original_root/'training/PROBE_EXPORTS.json'),
                 raw_bindings_identity=identity(output/'HISTORICAL_RAW_BINDINGS.json'),original_archive_manifest_identity=spec['historical_archive_manifest_identity'],windows=16,matched_rows=len(rows),
                 all_raw_verdicts_rescored=True,missing=dict(missing),full_E_tensor_validation='available only at8/128',
                 cpu_s=time.perf_counter()-started)
    write_json(output/'HISTORICAL_REUSE.json',receipt)
    return old,windows,live,receipt
