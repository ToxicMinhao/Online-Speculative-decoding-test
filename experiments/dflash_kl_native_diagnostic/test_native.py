"""CPU wrapper checks, not native GPU qualification."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from . import native as n


def checkpoints(tmp_path):
    released = tmp_path / 'released'; released.mkdir()
    retained = tmp_path / 'original/g/checkpoints/U2'; retained.mkdir(parents=True)
    result = {}
    for label, path, physical in [('F',released,'F'), ('S8',retained,'U2')]:
        (path/'config.json').write_text('{}')
        (path/'model.safetensors').write_bytes(label.encode())
        files = {p.name:dict(bytes=p.stat().st_size,sha256=n.sha(p)) for p in path.iterdir()}
        result[label] = dict(path=str(path),files=files,checkpoint_id=physical,
                            checkpoint_identity_sha256=n.digest(files))
    cfg=dict(diagnostic_checkpoints=result,released=result['F'],episode_id='g')
    return cfg,result


def test_exact_bound_checkpoint_admission_and_group_ownership(tmp_path):
    cfg,cps=checkpoints(tmp_path)
    with patch.object(n,'ORIGINAL_NATIVE_ROOT',tmp_path/'original'):
        assert n.admit_checkpoint(cfg,cps['F'])=='F'
        assert n.admit_checkpoint(cfg,cps['S8'])=='S8'
        other=copy.deepcopy(cps['S8']);other['checkpoint_id']='U1'
        with pytest.raises(ValueError,match='exact bound'):n.admit_checkpoint(cfg,other)
        cfg['episode_id']='other'
        with pytest.raises(ValueError,match='different original group'):n.admit_checkpoint(cfg,cps['S8'])


def test_mutated_checkpoint_file_is_rejected(tmp_path):
    cfg,cps=checkpoints(tmp_path)
    (Path(cps['S8']['path'])/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):n.admit_checkpoint(cfg,cps['S8'])


def test_retained_publication_keeps_native_cache_invalidation(tmp_path):
    cfg,cps=checkpoints(tmp_path)
    calls=[]
    receipt=dict(status='passed',success=True,helper_matches_current_layers=True,
        helper_norm_matches_current_layers=True,cache_after={'tree_total_size':0})
    bridge=SimpleNamespace(_two_update_admitted=set(),publish=lambda body:(calls.append(body) or receipt),
        state=lambda:{'draft_model_path':cps['S8']['path']})
    service=n.ServingService.__new__(n.ServingService)
    service.config,service.out,service.bridge=cfg,tmp_path,bridge
    with patch.object(n,'ORIGINAL_NATIVE_ROOT',tmp_path/'original'):
        result=service.publish(cps['S8'])
    assert calls==[{'model_path':cps['S8']['path'],'reset':True}]
    assert result['logical_checkpoint_id']=='S8' and result['activation_s']>=0
    assert service.active==cps['S8']
    receipt['cache_after']['tree_total_size']=1
    with patch.object(n,'ORIGINAL_NATIVE_ROOT',tmp_path/'original'):
        with pytest.raises(ValueError,match='cache invalidation'):service.publish(cps['S8'])


def trace_fixture(tmp_path):
    ids=[99,98];output=[10,11,12,13]
    doc=dict(input_token_ids=ids,final_output_token_ids=output,reconciliation={'status':'passed'},events=[
        dict(kind='target_prefill',committed_token_ids=[10]),
        dict(kind='verification',output_offset=1,draft_proposed_token_ids=[11,12,50],
             rejection_boundary_proposal_index=2,committed_token_ids=[11,12,13],
             committed_provenance=['accepted_draft','accepted_draft','target_correction'],stop_clipped=False)])
    path=tmp_path/'trace.json';n.write_json(path,doc)
    canonical=dict(request_id='r',input_ids=ids,output_ids=output,checkpoint_id='F',role='feedback')
    measured=dict(status='passed',output_ids=output,verification_calls=1,accepted_draft_tokens=2,
        verification_committed_tokens=3,trajectory=n.base.binding(path),finish_reason={'type':'stop','matched':128009})
    return canonical,measured


def test_source_generation_uses_exact_input_no_answer_forcing_no_capture(tmp_path):
    row,result=trace_fixture(tmp_path)
    backend=n.NativeBackend({});backend.episode_id='g';backend.group={'feedback':[row]}
    calls=[]
    def serve(_self,ids,label,capture,**kwargs):
        calls.append((ids,label,capture,kwargs));return dict(result)
    with patch.object(n.base.NativeBackend,'serve',serve):
        actual=backend.serve_source(copy.deepcopy(row))
        with pytest.raises(ValueError,match='One natural'):backend.serve_source(row)
    assert calls==[([99,98],'U2',False,{'turn_index':1})]
    assert actual['checkpoint_id']=='S8' and actual['capture'] is False
    assert actual['finish_reason']==result['finish_reason']
    assert actual['accepted_draft_tokens']==2 and actual['committed_progress_per_call']==3
    assert actual['initial_output_tokens']==1 and actual['rounds'][0]['first_rejection_offset']==2


def test_source_record_modification_and_after_loss_phase_are_forbidden(tmp_path):
    row,_=trace_fixture(tmp_path)
    backend=n.NativeBackend({});backend.episode_id='g';backend.group={'feedback':[row]}
    changed={**row,'output_ids':[15]}
    with pytest.raises(ValueError,match='differs from retained'):backend.serve_source(changed)
    backend.loss_phase=True
    with pytest.raises(ValueError,match='precede'):backend.serve_source(row)


def test_serving_service_forbids_capture_and_fresh_f_generations():
    service=n.ServingService.__new__(n.ServingService)
    with pytest.raises(ValueError,match='capture-disabled'):service.generate({'capture':True,'checkpoint_id':'U2'})
    with pytest.raises(ValueError,match='capture-disabled'):service.generate({'capture':False,'checkpoint_id':'F'})
    with patch.object(n.base._ServingService,'generate',return_value={'feedback':None,'capture_costs':None}) as generate:
        assert service.generate({'capture':False,'checkpoint_id':'U2'})['feedback'] is None
        assert generate.call_count==1


def test_loss_phase_closes_serving_and_only_sends_evaluation(tmp_path):
    order=[]
    class Worker:
        def __init__(self,config,kind,out):
            order.append(('start',kind,Path(out).name));self.startup_s=.1
        def call(self,action,body):
            order.append(('rpc',action,body['checkpoint_id']))
            return dict(status='passed',checkpoint_id=body['checkpoint_id'],response_losses=[])
    records=[dict(request_id=str(i),group_id='g',role='evaluation',checkpoint_id='F',input_ids=[i+1],output_ids=[2]) for i in range(4)]
    backend=n.NativeBackend({});backend.episode_id='g';backend.group={'scores':copy.deepcopy(records)};backend.out=tmp_path
    backend._close_serving=lambda:order.append(('stop_serving',))
    with patch.object(n.base,'_WorkerClient',Worker):
        assert backend.evaluate_heldout(records,'F')['checkpoint_id']=='F'
        assert backend.evaluate_heldout(records,'S8')['checkpoint_id']=='S8'
        with pytest.raises(ValueError,match='once'):backend.evaluate_heldout(records,'F')
    assert order==[('stop_serving',),('start','train','read_only_loss_worker'),('rpc','evaluate','F'),
                   ('stop_serving',),('rpc','evaluate','S8')]
    assert backend.loss_phase and backend.trainer is None
    with pytest.raises(ValueError,match='forbids training'):backend.train_event(1,records)


def test_heldout_role_or_tokens_cannot_be_changed(tmp_path):
    records=[dict(request_id=str(i),group_id='g',role='evaluation',checkpoint_id='F',input_ids=[i+1],output_ids=[2]) for i in range(4)]
    backend=n.NativeBackend({});backend.episode_id='g';backend.group={'scores':copy.deepcopy(records)};backend.out=tmp_path
    changed=copy.deepcopy(records);changed[0]['role']='feedback'
    with pytest.raises(ValueError,match='identity or evaluation role'):backend.evaluate_heldout(changed,'F')
    changed=copy.deepcopy(records);changed[0]['output_ids']=[3]
    with pytest.raises(ValueError,match='identity or evaluation role'):backend.evaluate_heldout(changed,'S8')


def test_compact_storage_does_not_budget_model_or_optimizer_exports(tmp_path):
    cfg=dict(output_root=str(tmp_path/'new'),reserve_bytes=8*2**30)
    with patch.object(n.shutil,'disk_usage',return_value=SimpleNamespace(free=16*2**30)):
        receipt=n.storage_preflight(cfg)
    assert receipt['required_bytes']==(14*2**30+256*2**20)
    assert receipt['checkpoint_copies']==receipt['optimizer_states']==0
    with patch.object(n.shutil,'disk_usage',return_value=SimpleNamespace(free=14*2**30)):
        with pytest.raises(ValueError,match='exceed free'):n.storage_preflight(cfg)
    cfg['reserve_bytes']=2**30
    with pytest.raises(ValueError,match='8 GiB'):n.storage_preflight(cfg)


def test_native_worker_entrypoint_is_diagnostic_only():
    cfg=dict(workspace_root='/tmp/root',conda_bootstrap='/tmp/conda.sh',conda_prefix='/tmp/env',python_executable='/tmp/env/bin/python')
    assert n.worker_command(cfg)[-1]==str(n.HERE/'native.py')
    from experiments.dflash_two_update import native as historical
    assert historical.OUTPUT_CAP==256 and n.base.OUTPUT_CAP==1024


def test_readonly_service_has_no_training_or_export_interface():
    assert not issubclass(n.ReadOnlyLossService,n.base._TrainerService)
    assert not hasattr(n.ReadOnlyLossService,'train')
    assert not hasattr(n.ReadOnlyLossService,'_export')
    assert not hasattr(n.ReadOnlyLossService,'train_event')
    source=(n.HERE/'native.py').read_text()
    assert '.Adam' not in source and 'PersistentAuthorOSDTrainer(' not in source
    assert 'load_checkpoint_state(' not in source


def test_original_controller_config_gets_only_verified_derived_source_pins():
    cfg=dict(diagnostic_source_sha256=n.source_pins(),learning_curve_source_sha256=n.curve.source_pins())
    with patch.object(n.curve,'check_sources') as original_check:
        n.check_sources(cfg)
    assert cfg['followup_source_sha256']==n.followup.followup_sources()
    assert cfg['author_sources_sha256']==n.sha(n.followup.HERE/'author_sources.json')
    assert original_check.call_count==1
    cfg['followup_source_sha256']={}
    with pytest.raises(ValueError,match='derived source pin'):n.check_sources(cfg)


def test_native_source_row_passes_real_pair_validation(tmp_path):
    from .test_experiment import native_row,loss_row
    from .experiment import validate_pair
    f=native_row()
    trace=dict(input_token_ids=f['input_ids'],final_output_token_ids=f['output_ids'],
        reconciliation={'status':'passed'},events=[dict(kind='target_prefill',committed_token_ids=[20]),
        dict(kind='verification',output_offset=1,draft_proposed_token_ids=[21,30],
            rejection_boundary_proposal_index=1,committed_token_ids=[21,128009],
            committed_provenance=['accepted_draft','target_correction'],stop_clipped=False)])
    path=tmp_path/'trace.json';n.write_json(path,trace)
    generated={**f,'checkpoint_identity_sha256':'8'*64,'trajectory':n.base.binding(path)}
    backend=n.NativeBackend({});backend.episode_id='g0';backend.group={'feedback':[f],'domain':'test'}
    with patch.object(n.base.NativeBackend,'serve',return_value=generated):
        s=backend.serve_source(f)
    assert s['eos_hit'] is True and s['domain']=='test' and s['role']==f['role']
    pair=dict(population='source',request_id=f['request_id'],checkpoint_bindings={'F':'f'*64,'S8':'8'*64},
        F=dict(native=f,kl=loss_row(f)),S8=dict(native=s,kl=loss_row(s)))
    assert validate_pair(pair)=={'status':'passed','errors':[]}


def test_partial_loss_failure_is_durable_without_turning_into_pass(tmp_path):
    import torch
    service=n.ReadOnlyLossService.__new__(n.ReadOnlyLossService)
    service.config={'diagnostic_checkpoints':{'F':{'checkpoint_identity_sha256':'f'*64}}}
    service.out=tmp_path;service.completed=set();service.draft=torch.nn.Linear(1,1)
    service.components=SimpleNamespace(embed_tokens=torch.nn.Linear(1,1),lm_head=torch.nn.Linear(1,1))
    service._load_checkpoint=lambda label: .1
    service.teacher_features=lambda ids:None
    failure=ValueError('nonfinite loss')
    failure.diagnostic_partial=dict(status='failed',response_losses=[dict(request_id='completed',loss=.2)],
        error_loss=float('nan'),optimizer_steps=0)
    with patch('experiments.dflash_kl_native_diagnostic.loss.evaluate_responses',side_effect=failure), \
         patch.object(torch.cuda,'reset_peak_memory_stats'), \
         patch.object(torch.cuda,'max_memory_allocated',return_value=100), \
         patch.object(torch.cuda,'max_memory_reserved',return_value=200):
        with pytest.raises(ValueError,match='nonfinite loss'):
            service.evaluate({'checkpoint_id':'F','records':[]})
    saved=json.loads((tmp_path/'HELDOUT_F_PARTIAL.json').read_text())
    assert saved['status']=='failed' and saved['response_losses'][0]['request_id']=='completed'
    assert saved['error_loss']['nonfinite_float']=='nan' and saved['optimizer_steps']==0
    assert not (tmp_path/'HELDOUT_F.json').exists()


def test_owned_worker_waits_for_scheduler_child_exit():
    clock=[0.];slept=[];closed=[]
    worker=SimpleNamespace(process=SimpleNamespace(pid=123,poll=lambda:0),close=lambda:closed.append(123))
    def sleep(seconds):
        slept.append(seconds);clock[0]+=seconds
    with patch.object(n.curve,'process_group_alive',side_effect=[True,True,False]) as alive, \
         patch.object(n.time,'monotonic',side_effect=lambda:clock[0]), \
         patch.object(n.time,'perf_counter',side_effect=lambda:clock[0]), \
         patch.object(n.time,'sleep',side_effect=sleep):
        receipt=n.close_owned_worker(worker,timeout_s=.3,poll_s=.1)
    assert closed==[123] and slept==[.1,.1]
    assert [c.args for c in alive.call_args_list]==[(123,),(123,),(123,)]
    assert receipt['shutdown_s']==pytest.approx(.2) and receipt['terminated_process_group']==123


def test_owned_worker_surviving_child_fails_at_declared_bound():
    clock=[0.];closed=[]
    worker=SimpleNamespace(process=SimpleNamespace(pid=123,poll=lambda:0),close=lambda:closed.append(123))
    with patch.object(n.curve,'process_group_alive',return_value=True) as alive, \
         patch.object(n.time,'monotonic',side_effect=lambda:clock[0]), \
         patch.object(n.time,'perf_counter',side_effect=lambda:clock[0]), \
         patch.object(n.time,'sleep',side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)):
        with pytest.raises(ValueError,match='survived bounded shutdown'):
            n.close_owned_worker(worker,timeout_s=.25,poll_s=.1)
    assert clock[0]==pytest.approx(.25) and closed==[123]
    assert all(c.args==(123,) for c in alive.call_args_list)
