"""Retire only explicitly recorded disposable qualification tensor payloads."""
from pathlib import Path
import os

from .contract import qualification_root
from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from .contract import ARMS
from .runtime import read
from .storage import _retire


def retire(plan):
    root=Path(plan['root']).resolve()
    complete=qualification_root(plan)/'QUALIFICATION.json'
    if read(complete)['status']!='passed':
        raise ValueError('Never retire an incomplete qualification')
    candidates={}
    evidence=[complete]
    def add(value,kind,*,expected=None):
        path=Path(value)
        if (not path.is_absolute() or path.resolve()!=path or not path.is_relative_to(qualification_root(plan))
                or path in candidates or (expected is not None and path!=expected)
                or path.stat().st_uid!=os.getuid()):
            raise ValueError('Disposable path is not its exclusive expected study payload: '+str(path))
        candidates[path]=kind
    for arm in ARMS:
        base=qualification_root(plan)/arm
        details=read(base/'DETAILS.json')
        if details['status']!='passed':
            raise ValueError('A disposable training gate is incomplete')
        evidence.extend([base/'DETAILS.json',base/'WORK.json'])
        expected=[base/'disposable_origin.pt',base/'disposable_active.pt']
        if list(map(Path,details['disposable_states']))!=expected:
            raise ValueError('Disposable qualification states differ from the prescribed pair')
        for value in expected:
            add(value,'disposable_training_state')
        if arm=='C_CODE58':
            add(Path(details['active_C_export'])/'model.safetensors','disposable_serving_export',
                expected=base/'active_export/model.safetensors')
        elif 'active_C_export' in details:
            raise ValueError('Only active C publishes a disposable serving export')
    synthetic=qualification_root(plan)/'synthetic4096/IDENTITIES.json'
    evidence.append(synthetic)
    pins=read(synthetic)['files']
    if set(pins)!={'synthetic4096_0.pt','synthetic4096_1.pt'}:
        raise ValueError('Synthetic resource fixture has unexpected members')
    for name,pin in pins.items():
        path=synthetic.parent/name
        add(path,'synthetic_resource_fixture')
        if identity(path)!=pin:
            raise ValueError('Synthetic resource fixture changed')
    feature_index=qualification_root(plan)/'A/FEATURES.json'
    evidence.append(feature_index)
    features=read(feature_index)
    if len(features)!=16:
        raise ValueError('Qualification feature retirement needs all16 reserved inputs')
    for value in features.values():
        record_path=Path(value['record_path'])
        slot=value['source_slot']
        if record_path!=feature_index.parent/'records'/(slot+'.json'):
            raise ValueError('Qualification record belongs to a different slot')
        record=read(record_path)
        receipt=value['receipt']
        if (receipt['status']!='prepared' or record['record_id']!=slot
                or receipt['record_id']!=slot):
            raise ValueError('Qualification feature/record identity changed')
        evidence.append(record_path)
        prepared=feature_index.parent/'features'/(slot+'.pt')
        add(receipt['feature_path'],'qualification_prepared_features',expected=prepared)
        add(receipt['terminal_feature']['tensor_path'],'qualification_terminal_features',
            expected=prepared.with_suffix('.terminal.pt'))
        add(record['metadata']['capture']['feature_path'],'qualification_native_features',
            expected=Path(record['metadata']['native_trace_path']).with_suffix('.pt'))
    # Complete content identities supplement the retained numerical checks.
    # These are deletion witnesses, not backups or a claim of replayability.
    identities={str(path):identity(path) for path in candidates}
    pins=qualification_root(plan)/'DISPOSABLE_IDENTITIES.json'
    write_json(pins,dict(files=identities,lost_capability='Exact disposable tensor replay; records and tests remain'))
    return _retire(root,qualification_root(plan)/'DISPOSABLE_RETIREMENT.json',list(candidates.items()),
        [*evidence,pins],reason='All native qualification checks and publication passed; discard temporary models.',
        retained=dict(qualification=str(complete),identities=str(pins),production_initialization='released fresh'))
