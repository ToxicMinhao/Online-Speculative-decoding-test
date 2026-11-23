"""Only the prescribed completed disposable payloads can be retired."""
import json

import pytest

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from . import qualification_cleanup as C
from .contract import ARMS
from .test_protected import seed_protection


def fixture(root):
    seed_protection(root)
    write_json(root/'qualification/QUALIFICATION.json',dict(status='passed'))
    tensors=[]
    def tensor(path):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(str(path.name).encode())
        tensors.append(path)
        return str(path)
    for arm in ARMS:
        base=root/'qualification'/arm
        details=dict(status='passed',disposable_states=[tensor(base/name)
            for name in ('disposable_origin.pt','disposable_active.pt')])
        if arm=='C_CODE58':
            tensor(base/'active_export/model.safetensors')
            details['active_C_export']=str(base/'active_export')
        write_json(base/'DETAILS.json',details)
        write_json(base/'WORK.json',dict(status='completed'))
    base=root/'qualification/synthetic4096'
    for i in range(2):tensor(base/f'synthetic4096_{i}.pt')
    write_json(base/'IDENTITIES.json',dict(files={path.name:identity(path) for path in base.glob('*.pt')}))
    features={}
    for i in range(16):
        slot=f'A_F_reserved_{i}'
        base=root/'qualification/A'
        prepared=base/'features'/(slot+'.pt')
        native=base/'A_F/native'/f'request_{i:06d}.pt'
        record_path=base/'records'/(slot+'.json')
        write_json(record_path,dict(record_id=slot,metadata=dict(native_trace_path=str(native.with_suffix('.json')),
            capture=dict(feature_path=tensor(native)))))
        features[str(i)]=dict(source_slot=slot,record_path=str(record_path),receipt=dict(
            status='prepared',record_id=slot,feature_path=tensor(prepared),
            terminal_feature=dict(tensor_path=tensor(prepared.with_suffix('.terminal.pt')))))
    write_json(root/'qualification/A/FEATURES.json',features)
    return tensors


def test_completed_qualification_retires_exact57_payloads_and_keeps_evidence(tmp_path):
    tensors=fixture(tmp_path)
    result=C.retire(dict(root=str(tmp_path)))
    assert result['status']=='completed' and len(result['removed'])==57
    assert all(not path.exists() for path in tensors)
    assert (tmp_path/'qualification/A/FEATURES.json').exists()
    assert (tmp_path/'qualification/DISPOSABLE_IDENTITIES.json').exists()


def test_changed_state_path_fails_before_any_unlink(tmp_path):
    tensors=fixture(tmp_path)
    path=tmp_path/'qualification/O_CODE58/DETAILS.json'
    details=json.loads(path.read_text())
    details['disposable_states'][0]=str(tmp_path/'qualification/other.pt')
    write_json(path,details)
    with pytest.raises(ValueError,match='prescribed pair'):
        C.retire(dict(root=str(tmp_path)))
    assert all(path.exists() for path in tensors)


def test_symlinked_payload_fails_before_hash_or_unlink(tmp_path):
    tensors=fixture(tmp_path)
    path=tensors[0]
    external=tmp_path/'outside.pt'
    external.write_bytes(b'preserve')
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(ValueError,match='exclusive expected'):
        C.retire(dict(root=str(tmp_path)))
    assert path.is_symlink() and external.read_bytes()==b'preserve'
    assert all(path.exists() for path in tensors)
