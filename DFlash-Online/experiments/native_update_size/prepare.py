"""Build the new live dependency set without rewriting historical seals."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import pwd
import subprocess
import time

from online_feedback.storage import write_json
from experiments.threatened_repair_replay.prepare import read, identity, authenticate_source
from .provenance import historical_bindings
from .identity_gate import gate0, check_identity
from .storage import initialize, check_write


def artifact_row(path, category, required_by, expected=None, release='retain through experiment/audit/next decision'):
    path=Path(path)
    row=dict(host='A800',path=str(path),realpath=str(path.resolve()),owner=None,device=None,inode=None,links=None,
             logical_bytes=0,allocated_bytes=0,identity=expected,classification=category,required_by=required_by,
             canonical_replacement=None,release_condition=release,status='present' if path.exists() else 'absent')
    if path.exists():
        stat=path.stat()
        try:
            owner=pwd.getpwuid(stat.st_uid).pw_name
        except KeyError:
            # Mounted files can belong to a UID absent from the container's
            # account database. Retain that identity without inventing a name.
            owner=f'uid:{stat.st_uid}'
        row.update(owner=owner,owner_uid=stat.st_uid,device=stat.st_dev,inode=stat.st_ino,
                                   links=stat.st_nlink,logical_bytes=stat.st_size,allocated_bytes=stat.st_blocks*512,
                                   symlink=path.is_symlink())
    else:
        row['classification']='ALREADY_ABSENT'
    return row


def build(spec):
    root=Path(spec['root']);root.mkdir(parents=True,exist_ok=True)
    new_roots=[str(root),spec['source_root']]+spec.get('prior_new_roots',[])
    skeleton=dict(root=str(root),allocation_baseline=str(root/'ALLOCATION_BASELINE.json'),
                  allocation_roots=new_roots+['/data/sources/minhao/.cache','/data/sources/minhao/tmp'])
    # The staging receipt accounts for source and plan bytes written before this
    # baseline. Remove these new-root inode credits so they still count as new.
    baseline=initialize(skeleton)
    from .storage import allocated_files
    for key in allocated_files(new_roots):
        baseline['files'].pop(key,None)
    write_json(skeleton['allocation_baseline'],baseline)
    check_write(skeleton,spec['peak_new_bytes'],'admission_from_pre_Gate0')
    source=authenticate_source(spec['source_manifest'],spec['source_manifest_identity'],spec['execution_commit'])
    old,windows,live,reuse=historical_bindings(spec,root/'preparation')
    for label,sha in [('origin_state','15ac28e1235851a15158e789af1480fd2d82535903b1dec5d992ec843c9ae3a1'),
                      ('historical_native_state','be5cccf3a7d4cee1ca65d0df390d5f4de3d198643a97fc5125afc68442e1fa45'),
                      ('schedule','6ac18ca2c16ded078f831ab90b72b58333403e5bece9553943de491e5193a1d0'),
                      ('probe_keys','ff20e335776f151d17eb47f4ed71c91ea00df6616fbbbfc08b245607ad6c1779')]:
        expected=live.get(old[label],identity(old[label]))
        if expected['sha256']!=sha:
            raise ValueError('Authoritative input pin differs: '+label)
        live[old[label]]=expected
    for path,expected in spec['historical_receipt_identities'].items():
        live[path]=expected
    plan=deepcopy(old)
    # Prior attempt allowances and old80-export requirements belong only to PR47.
    for key in ('prior_attempt','budget_accounting','storage','constants','immutable_files','previous_plan'):
        plan.pop(key,None)
    plan.update(skeleton, constants=dict(windows=list(range(8,129,8))), schema='native-update-size-v1',execution_commit=spec['execution_commit'],
                code_path=source['source_path'],source_manifest=spec['source_manifest'],source_verification=source,
                historical_plan=spec['historical_plan'],historical_execution_commit=old['execution_commit'],
                historical_windows=windows, live_input_identities=live,
                source_identity=spec['source_manifest_identity'],input_identities=live,
                historical_probe_outcomes=old['root']+'/probes/PROBE_OUTCOMES.jsonl',
                historical_probe_root=old['root']+'/probes',historical_probe_exports=old['root']+'/training/PROBE_EXPORTS.json',
                reuse_identity_receipt=str(root/'preparation/HISTORICAL_REUSE.json'),
                sentinel_exports=[windows['8'][state] for state in ('P','N','E')],
                control_seal=str(root/'training/CONTROL_SEAL.json'),
                cpu_gate0=str(root/'gate0/GATE0.json'),peak_new_bytes=spec['peak_new_bytes'],
                storage_ceiling_bytes=110_000_000_000,free_reserve_bytes=400_000_000_000,
                host_reserve_bytes=32*1024**3,training_host_required_bytes=80*1024**3,
                maximum_adam=294,maximum_response_forwards=260,maximum_response_backwards=260,
                maximum_main_blocks=8192,maximum_extra_blocks=32,card_seconds_limit=10800,
                remaining_probe_estimate_s=spec['remaining_probe_estimate_s'],
                validation_export_estimate_s=spec['validation_export_estimate_s'],
                remaining_overhead_estimate_s=spec['remaining_overhead_estimate_s'],
                prior_failed_work=[],full_E_geometry_available_windows=[8,128])
    # Existing source is immutable provenance; the new controller source has its
    # own manifest. We still validate the historical inputs, not stale outputs.
    artifacts=[artifact_row(path,'KEEP_INPUT' if ('features.pt' in path or path in
                [old['origin_state'],old['historical_native_state'],old['schedule'],old['origin_config']]
                or 'model' in path or 'tokenizer' in path) else 'KEEP_EVIDENCE',
                'native reconstruction / parent scoring / authenticated predecessor provenance',pin)
                for path,pin in live.items()]
    inventory=read(root/'preparation/EXPORT_INVENTORY.json')
    artifacts += [artifact_row(row['path'],'KEEP_INPUT' if row['available'] else 'ALREADY_ABSENT',
                               'original endpoint/boundary check; absent E tensor not a live input',row['identity']) for row in inventory]
    for row in read(root/'preparation/HISTORICAL_RAW_BINDINGS.json'):
        artifacts.append(artifact_row(row['path'],'KEEP_EVIDENCE','CPU verdict rescoring / historical reuse',row['identity']))
    keepers=read(spec['paper_keepers'])
    # The paper keeper draft is an explicit list, not authority to remove other files.
    entries=(keepers.get('selected_paper_assets',[]) + keepers.get('uncertain_closed_assets',[])) if isinstance(keepers,dict) else keepers
    for row in entries:
        if row.get('path'):
            path=Path(row['path'])
            if row.get('class')=='KEEP_PAPER' and path.is_file():
                expected=row.get('identity')
                if expected and expected.get('sha256'):check_identity(path,expected)
                row['verified_identity']=identity(path)
                row['status']='present; current content authenticated'
            elif not path.exists():row['status']='already absent; no substitution or deletion'
            artifacts.append(artifact_row(row['path'],row.get('class','KEEP_PAPER'),'selected paper/native dependency',row.get('verified_identity',row.get('identity'))))
    write_json(root/'PAPER_KEEPERS.json',keepers)
    for window in range(8,129,8):
        for state in ('S_low','S_high'):
            row=artifact_row(root/f'training/exports/window_{window:03d}_{state}/model.safetensors',
                             'TEMP_CONTROL','sealed scalar scoring and geometry',
                             release='all16windows, scoring, raw CPU audit and report verified; unique NEW files only')
            row.update(classification='TEMP_CONTROL',status='planned, not created')
            artifacts.append(row)
    row=artifact_row(root/'rolling_validation/model.safetensors','TEMP_VALIDATION','exact original-file comparison',
                     release='only after own SHA256/size pass and durable receipt; failed file becomes REVIEW')
    row.update(classification='TEMP_VALIDATION',status='planned, not created');artifacts.append(row)
    unique={row['path']:row for row in artifacts}
    write_json(root/'ARTIFACT_MANIFEST.json',list(unique.values()))
    with (root/'ARTIFACT_MANIFEST.csv').open('w',newline='') as stream:
        fields=sorted({key for row in unique.values() for key in row})
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(unique.values())
    (root/'sealed').mkdir(exist_ok=True)
    write_json(root/'sealed/PLAN.json',plan)
    from .resources import resource_snapshot
    resources=resource_snapshot(plan)
    write_json(root/'preparation/RESOURCE_PREFLIGHT.json',resources)
    if resources['status']!='passed':raise ValueError('Runtime/import/resource inventory did not qualify')
    gate=gate0(plan,root/'gate0')
    # No additional old asset was positively proven disposable in this audit.
    # Ample headroom is not a reason to invent a reclamation target.
    write_json(root/'RETIREMENT_PLAN.json',dict(status='no additional proven candidates',rows=[],
               historical_cleanup_counted_again=False,gate0_status=gate['status']))
    write_json(root/'RETIREMENT_LOG.json',dict(rows=[],new_files_deleted=0,new_bytes_freed=0))
    return plan


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();build(read(a.spec))
