"""CPU-only controller adapters and strict checkpoint/source/storage guards."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from . import native as n
from .test_native_probe import anchor


def make_checkpoint(path,label,**extra):
    path.mkdir(parents=True)
    (path/'config.json').write_text('{}')
    (path/'model.safetensors').write_bytes(label.encode())
    files={p.name:dict(bytes=p.stat().st_size,sha256=n.sha(p)) for p in path.iterdir()}
    return dict(path=str(path),checkpoint_id=label,files=files,checkpoint_identity_sha256=n.digest(files),**extra)


def setup_backend(monkeypatch,tmp_path,*,empty=False):
    gid='gsm8k_00';old=tmp_path/'old';monkeypatch.setattr(n.kl,'ORIGINAL_NATIVE_ROOT',old)
    checkpoints=dict(F=make_checkpoint(tmp_path/'F','F'),
        S8=make_checkpoint(old/gid/'checkpoints/U2','U2'),
        L8=make_checkpoint(tmp_path/'l8'/gid/'checkpoints/L8','L8',group_id=gid,optimizer_steps=8,learning_rate=3e-5))
    targetpath=tmp_path/'target';targetpath.mkdir();(targetpath/'config.json').write_text('{}')
    target=dict(target_id='T0',model_path=str(targetpath),identity_sha256='c'*64,
        checkpoint_files={'config.json':dict(bytes=2,sha256=n.sha(targetpath/'config.json'))})
    models=dict(targets=[target],released_draft=dict(path=checkpoints['F']['path'],checkpoint_files=checkpoints['F']['files']),
        tokenizer_path=str(targetpath))
    registry=tmp_path/'models.json';n.write_json(registry,models)
    a=anchor();a['checkpoints']=checkpoints
    group=dict(group_id=gid,domain='gsm8k',checkpoints=checkpoints,anchors=[] if empty else [a],requests=[])
    cfg=dict(output_root=str(tmp_path/'out'),workspace_root=str(tmp_path),model_bindings=n.base.binding(registry),
        runtime_source_sha256={'runtime':'pinned'},matched_prefix_native_runtime=dict(target=target,released=checkpoints['F'],
            tokenizer_path=str(targetpath),effective_limits=dict(total_context_tokens=32768,chunked_prefill_tokens=8192)))
    monkeypatch.setattr(n,'check_sources',lambda _:None)
    monkeypatch.setattr(n.base,'check_environment',lambda _:None)
    monkeypatch.setattr(n.base,'check_dependencies',lambda _:dict(runtime_source_sha256={'runtime':'pinned'},
        qualified_model_bindings={'sha256':n.sha(registry)}))
    monkeypatch.setattr(n.shutil,'disk_usage',lambda _:SimpleNamespace(free=30*2**30))
    calls=[]
    class Worker:
        def __init__(self,config,kind,out):
            calls.append(('load',kind));self.out=Path(out);self.out.mkdir(parents=True)
            self.config=config;self.process=SimpleNamespace(pid=123)
        def call(self,action,value):
            calls.append((action,value));return dict(status='passed',native_draft_block_evaluations=1,
                native_head_calls=1,target_prefill_chunks=[2],costs=dict(activation_s=.1,cache_maintenance_s=.02,head_inspection_s=.03))
    monkeypatch.setattr(n.base,'_WorkerClient',Worker)
    monkeypatch.setattr(n.kl,'close_owned_worker',lambda worker:dict(status='passed'))
    return n.NativeBackend(cfg),group,calls


def test_import_does_not_import_torch_or_projection_packages():
    code=('import sys; from experiments.dflash_matched_prefix_diagnostic import native; '
          'assert "torch" not in sys.modules; '
          'assert not any("projections" in k or "correction_bank" in k for k in sys.modules); '
          'assert native.base.OUTPUT_CAP == 1024')
    result=subprocess.run([sys.executable,'-c',code],cwd=n.HERE.parents[1],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_backend_starts_only_serving_and_probes_each_declared_arm_once(monkeypatch,tmp_path):
    backend,group,calls=setup_backend(monkeypatch,tmp_path)
    old=n.base.worker_command
    receipt=backend.start(group)
    assert receipt['native_worker_started'] and calls==[('load','sd')] and n.base.worker_command is old
    a=group['anchors'][0]
    for arm in n.ARMS:assert backend.probe(a,arm)['status']=='passed'
    with pytest.raises(ValueError,match='repeated'):backend.probe(a,'F')
    assert backend.costs['attempted_probes']==3 and backend.costs['native_draft_block_evaluations']==3
    assert backend.costs['target_prefill_chunk_count']==3
    finish=backend.close()
    assert finish['costs']['head_inspection_s']==pytest.approx(.09) and backend.worker is None
    assert not hasattr(backend,'train_event') and not hasattr(backend,'serve')


def test_empty_group_never_loads_model_and_is_insufficient(monkeypatch,tmp_path):
    backend,group,calls=setup_backend(monkeypatch,tmp_path,empty=True)
    result=backend.start(group)
    assert not result['native_worker_started'] and result['qualification']=='insufficient_no_selected_locations'
    assert not calls
    with pytest.raises(ValueError,match='No active'):backend.probe(anchor(),'F')
    backend.close()


def test_backend_refuses_changed_anchor_without_work(monkeypatch,tmp_path):
    backend,group,calls=setup_backend(monkeypatch,tmp_path);backend.start(group)
    a=copy.deepcopy(group['anchors'][0]);a['seed_token_id']=999
    with pytest.raises(ValueError,match='sealed'):backend.probe(a,'F')
    assert backend.costs['attempted_probes']==0 and calls==[('load','sd')]


def test_backend_recovers_raw_zero_forward_failure_and_does_not_retry(monkeypatch,tmp_path):
    backend,group,_=setup_backend(monkeypatch,tmp_path);backend.start(group);a=group['anchors'][0]
    raw=dict(status='failed',native_draft_block_evaluations=0,native_head_calls=0,target_prefill_chunks=[2],
             reconstructed_target_seed_id=99,costs={})
    path=backend.out/'probe_worker/blocks';path.mkdir()
    n.write_json(path/(n.digest([a['anchor_id'],'F'])+'.json'),raw)
    def fail(*_):raise RuntimeError('HTTP500')
    backend.worker.call=fail
    with pytest.raises(RuntimeError) as error:backend.probe(a,'F')
    assert error.value.diagnostic_partial==raw and error.value.partial_binding['sha256']
    assert backend.costs['attempted_probes']==1 and backend.costs['native_draft_block_evaluations']==0
    with pytest.raises(ValueError,match='repeated'):backend.probe(a,'F')


@pytest.mark.parametrize('change',['group','lr','step','label','file'])
def test_exact_retained_l8_artifact_no_substitute(monkeypatch,tmp_path,change):
    backend,group,_=setup_backend(monkeypatch,tmp_path);cp=copy.deepcopy(group['checkpoints']['L8'])
    cfg=dict(episode_id=group['group_id'],matched_prefix_checkpoints=group['checkpoints'],released=group['checkpoints']['F'])
    assert n.admit_checkpoint(cfg,cp)=='L8'
    if change=='file':(Path(cp['path'])/'model.safetensors').write_bytes(b'changed')
    else:cp[{'group':'group_id','lr':'learning_rate','step':'optimizer_steps','label':'checkpoint_id'}[change]]={
        'group':'other','lr':3e-4,'step':2,'label':'U2'}[change]
    with pytest.raises(ValueError):n.admit_checkpoint(cfg,cp)


def test_original_runtime_binding_must_match_registry(monkeypatch,tmp_path):
    backend,group,calls=setup_backend(monkeypatch,tmp_path)
    backend.config['matched_prefix_native_runtime']['effective_limits']['total_context_tokens']=17408
    with pytest.raises(ValueError,match='PR11'):backend.start(group)
    assert not calls


def test_storage_includes_compact_all_groups_staging_and_reserve(monkeypatch,tmp_path):
    required=6*64*2**20+64*2**20+8*2**30
    monkeypatch.setattr(n.shutil,'disk_usage',lambda _:SimpleNamespace(free=required))
    row=n.storage_preflight(dict(output_root=str(tmp_path/'new')))
    assert row['required_bytes']==required and row['checkpoint_copies']==row['optimizer_states']==0
    monkeypatch.setattr(n.shutil,'disk_usage',lambda _:SimpleNamespace(free=required-1))
    with pytest.raises(ValueError,match='exceed free'):n.storage_preflight(dict(output_root=str(tmp_path)))
    with pytest.raises(ValueError,match='8 GiB'):n.storage_preflight(dict(output_root=str(tmp_path),reserve_bytes=1))


def test_source_pins_include_predecessor_provenance_and_qualified_runtime(monkeypatch):
    pins=n.source_pins()
    assert 'dflash_osd_attribution/native_probe.py' in pins
    assert 'dflash_lower_lr_control/native.py' in pins and 'dflash_two_update/native.py' in pins
    assert not any('correction_bank' in key or key.endswith('bound.py') for key in pins)
    called=[];monkeypatch.setattr(n.qualified,'check_sources',lambda cfg:called.append(cfg))
    n.check_sources({'matched_prefix_source_sha256':pins});assert len(called)==1
    with pytest.raises(ValueError,match='source changed'):n.check_sources({'matched_prefix_source_sha256':{}})


def test_worker_transport_points_only_to_readonly_probe():
    cfg=dict(workspace_root='/tmp/work',conda_bootstrap='/tmp/conda.sh',conda_prefix='/tmp/env',python_executable='/tmp/env/bin/python')
    assert n.worker_command(cfg)[-1]==str(n.HERE/'native_probe.py')
