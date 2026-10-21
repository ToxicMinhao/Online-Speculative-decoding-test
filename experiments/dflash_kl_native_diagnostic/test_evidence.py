import copy
import json
from pathlib import Path

import pytest

from . import evidence as ev
from .test_experiment import native_row, loss_row


def make_baseline(tmp_path, monkeypatch):
    root = tmp_path / 'original'; root.mkdir()
    package = tmp_path / 'package'; package.mkdir()
    original = Path('/data/sources/minhao/EAGLE_SPEC/research/dflash_baseline_learning_curve_20261020/execution_001/attempt_001')
    groups, selections = [], []
    for i in range(6):
        gid, domain = f'g{i}', ('math','code','chat')[i//2]
        frozen_files = {'model.safetensors': {'bytes':1,'sha256':'f'*64}}
        adapted_files = {'model.safetensors': {'bytes':1,'sha256':str(i)*64}}
        checkpoints = {'F':dict(checkpoint_id='F',path='/models/released',files=frozen_files,checkpoint_identity_sha256=ev.digest(frozen_files)),
            'S8':dict(checkpoint_id='U2',path=str(original/'native'/gid/'checkpoints/U2'),files=adapted_files,checkpoint_identity_sha256=ev.digest(adapted_files))}
        g=dict(group_id=gid,domain=domain,status='complete',implementation_checks={'status':'passed'},
            feedback=[],scores=[],checkpoints=checkpoints,training={'source_losses':[]},evaluation_requests=[])
        selected=dict(group_id=gid,domain=domain,feedback=[],evaluation=[])
        for role in ('feedback','evaluation'):
            for j in range(4):
                rows=[]
                for label in (('F',) if role=='feedback' else ('F','S8')):
                    row=native_row(gid,role,j,label)
                    row['input_ids']=[10+100*i+10*(role=='evaluation')+j,11]
                    row['input_ids_sha256']=ev.digest(row['input_ids'])
                    row['checkpoint_identity_sha256']=checkpoints[label]['checkpoint_identity_sha256']
                    row['rounds'][0]['absolute_token_offset']=3
                    name=f'native/{gid}/worker/{role}_{j}_{label}.json'
                    trace=dict(input_token_ids=row['input_ids'],final_output_token_ids=row['output_ids'],max_new_tokens=1024,
                        reconciliation={'status':'passed'},events=[dict(kind='prefill',committed_token_ids=[20]),
                        dict(kind='verification',output_offset=1,committed_token_ids=[21,128009],
                            draft_proposed_token_ids=[21,22],rejection_boundary_proposal_index=1,
                            committed_provenance=['accepted_draft','correction'],stop_clipped=False)])
                    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(trace))
                    row['trajectory']={**ev.file_binding(p),'path':str(original/name)}
                    rows.append(row)
                g['feedback' if role=='feedback' else 'scores'].extend(rows)
                if role=='evaluation':g['scores'].append({**copy.deepcopy(rows[0]),'checkpoint_id':'S2'})
                selected[role].append(dict(request_id=rows[0]['request_id'],input_ids=rows[0]['input_ids']))
        g['evaluation_requests']=selected['evaluation']
        for label in ('F','S8'):
            losses=[loss_row({**row,'checkpoint_id':label,'checkpoint_identity_sha256':checkpoints[label]['checkpoint_identity_sha256']}) for row in g['feedback']]
            g['training']['source_losses'].append(dict(checkpoint_id=label,checkpoint_sha256=checkpoints[label]['checkpoint_identity_sha256'],
                training_rng_preserved=True,diagnostic_model_mode='eval_no_grad',response_losses=losses))
        groups.append(g);selections.append(selected)
    docs={'RESULT.json':dict(status='complete',unexecuted_groups=[],groups=groups),
        'SAMPLE_MANIFEST.json':{'groups':selections},'RUNTIME_CONFIG.json':{}}
    for name,value in docs.items():(root/name).write_text(json.dumps(value))
    commit='165e25b7a85cabfe16f7b9d642fb575b23d82ad6'
    source=dict(git={'commit':commit,'dirty':False},sample_manifest=ev.file_binding(root/'SAMPLE_MANIFEST.json'))
    (root/'SOURCE_MANIFEST.json').write_text(json.dumps(source))
    pins=dict(baseline_commit=commit,original_root=str(original),files={name:ev.file_binding(root/name) for name in (*docs,'SOURCE_MANIFEST.json')})
    (package/'baseline_bindings.json').write_text(json.dumps(pins))
    monkeypatch.setattr(ev,'HERE',package)
    return root,package


def repin(root, package, name, mutate):
    path=root/name;value=json.loads(path.read_text());mutate(value);path.write_text(json.dumps(value))
    pins=json.loads((package/'baseline_bindings.json').read_text());pins['files'][name]=ev.file_binding(path)
    (package/'baseline_bindings.json').write_text(json.dumps(pins))


def test_complete_compact_evidence_without_loading_tensors(tmp_path,monkeypatch):
    root,_=make_baseline(tmp_path,monkeypatch)
    bundle=ev.load_evidence(root)
    assert len(bundle['groups'])==6 and len(bundle['bindings']['trajectories'])==72
    assert bundle['bindings']['compact_only']


@pytest.mark.parametrize('filename',['RESULT.json','SOURCE_MANIFEST.json','SAMPLE_MANIFEST.json','RUNTIME_CONFIG.json'])
def test_hash_tamper_and_missing_are_not_approximated(tmp_path,monkeypatch,filename):
    root,_=make_baseline(tmp_path,monkeypatch)
    (root/filename).write_text('{"modified":true}')
    with pytest.raises(ValueError,match='Original evidence changed'):ev.load_evidence(root)
    (root/filename).unlink()
    with pytest.raises(FileNotFoundError):ev.load_evidence(root)


@pytest.mark.parametrize('change',['foreign_checkpoint','token','source_checkpoint','duplicate','order'])
def test_deep_binding_checks(tmp_path,monkeypatch,change):
    root,package=make_baseline(tmp_path,monkeypatch)
    def mutate(d):
        g=d['groups'][0]
        if change=='foreign_checkpoint':g['checkpoints']['S8']['path']=d['groups'][1]['checkpoints']['S8']['path']
        if change=='token':g['feedback'][0]['input_ids'][0]+=1
        if change=='source_checkpoint':g['training']['source_losses'][0]['checkpoint_sha256']='x'*64
        if change=='duplicate':g['feedback'][1]=copy.deepcopy(g['feedback'][0])
        if change=='order':d['groups'].reverse()
    repin(root,package,'RESULT.json',mutate)
    with pytest.raises(ValueError):ev.load_evidence(root)


def test_trace_tamper_does_not_use_summary_instead(tmp_path,monkeypatch):
    root,_=make_baseline(tmp_path,monkeypatch)
    path=next((root/'native').rglob('feedback*.json'));path.write_text('{}')
    with pytest.raises(ValueError,match='Original evidence changed'):ev.load_evidence(root)
