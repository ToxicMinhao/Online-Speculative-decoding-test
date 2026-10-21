"""CPU wrapper checks. Actual native model execution requires first-group qualification."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from . import native as n


def checkpoint(tmp_path):
    out=tmp_path/'g';path=out/'checkpoints/L8';path.mkdir(parents=True)
    (path/'config.json').write_text('{}');(path/'model.safetensors').write_bytes(b'L8')
    files={p.name:dict(bytes=p.stat().st_size,sha256=n.sha(p)) for p in path.iterdir()}
    cp=dict(path=str(path),files=files,checkpoint_id='L8',checkpoint_identity_sha256=n.digest(files),
        group_id='g',optimizer_steps=8,learning_rate=3e-5)
    cfg=dict(episode_out=str(out),episode_id='g')
    return cfg,cp


def test_l8_is_registered_under_own_label_group_path_and_learning_rate(tmp_path):
    cfg,cp=checkpoint(tmp_path)
    assert n.admit_checkpoint(cfg,cp)=='L8'
    for field,value in [('checkpoint_id','U2'),('group_id','another'),('learning_rate',3e-4),('optimizer_steps',2)]:
        with pytest.raises(ValueError):n.admit_checkpoint(cfg,{**cp,field:value})
    with pytest.raises(ValueError):n.admit_checkpoint({**cfg,'episode_out':str(tmp_path/'other')},cp)
    (Path(cp['path'])/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):n.admit_checkpoint(cfg,cp)


def test_publication_inherits_native_reset_helper_and_cache_checks(tmp_path):
    cfg,cp=checkpoint(tmp_path);calls=[]
    receipt=dict(status='passed',success=True,helper_matches_current_layers=True,
        helper_norm_matches_current_layers=True,cache_after={'tree_total_size':0})
    bridge=SimpleNamespace(_two_update_admitted=set(),publish=lambda body:(calls.append(body) or receipt),
        state=lambda:{'draft_model_path':cp['path']})
    service=n.ServingService.__new__(n.ServingService)
    service.config,service.out,service.bridge=cfg,tmp_path,bridge
    assert service.publish(cp)['checkpoint_id']=='L8'
    assert calls==[{'model_path':cp['path'],'reset':True}] and service.active==cp
    receipt['cache_after']['tree_total_size']=2
    with pytest.raises(ValueError,match='cache invalidation'):service.publish(cp)


def test_activation_timer_covers_validation_and_publication_once(tmp_path):
    cfg,cp=checkpoint(tmp_path);clock=[0.]
    backend=n.NativeBackend(cfg);backend.checkpoints={'L8':cp};backend.active_checkpoint='F'
    backend._ensure_worker=lambda _:0
    def call(action,body):
        assert action=='publish' and body==cp
        clock[0]+=1.25 # includes validation, RPC and publication.
        return dict(status='passed',activation_s=.5,checkpoint_id='L8')
    backend.worker=SimpleNamespace(call=call)
    with patch.object(n.base.time,'perf_counter',side_effect=lambda:clock[0]):
        result=backend.publish('L8');again=backend.publish('L8')
    assert result['activation_s']==1.25 and result['native_activation_s']==.5
    assert backend.totals['activation_s']==1.25 and again['activation_s']==0


def test_storage_includes_all_final_weights_overlap_evidence_reserve(tmp_path):
    bound=n.campaign_storage_bound();required=bound['planned_write_bytes']+8*2**30
    cfg={'output_root':str(tmp_path/'new')}
    with patch.object(n.shutil,'disk_usage',return_value=SimpleNamespace(free=required)):
        receipt=n.storage_preflight(cfg)
    assert receipt['model_tensor_bytes']==1048626432*2
    assert receipt['final_checkpoints']==6 and receipt['optimizer_state_exports']==receipt['intermediate_checkpoints']==0
    assert required==7*(1048626432*2+64*2**20)+6*256*2**20+8*2**30
    with patch.object(n.shutil,'disk_usage',return_value=SimpleNamespace(free=required-1)):
        with pytest.raises(ValueError,match='exceed free'):n.storage_preflight(cfg)
    with pytest.raises(ValueError,match='8 GiB'):n.storage_preflight({**cfg,'reserve_bytes':1})


def test_original_source_guards_remain_and_verified_derived_pin_is_supplied():
    cfg=dict(lower_lr_source_sha256=n.source_pins(),learning_curve_source_sha256=n.curve.source_pins())
    with patch.object(n.curve,'check_sources') as checked:n.check_sources(cfg)
    assert cfg['diagnostic_source_sha256']==n.kl.source_pins() and checked.call_count==1
    cfg['diagnostic_source_sha256']={}
    with pytest.raises(ValueError,match='PR10'):n.check_sources(cfg)


def canonical_fixture(tmp_path):
    ids=[99,98];output=[10,11,128009]
    trace=dict(input_token_ids=ids,final_output_token_ids=output,reconciliation={'status':'passed'},events=[
        dict(kind='target_prefill',committed_token_ids=[10]),
        dict(kind='verification',output_offset=1,draft_proposed_token_ids=[11,50],rejection_boundary_proposal_index=1,
             committed_token_ids=[11,128009],committed_provenance=['accepted_draft','target_correction'],stop_clipped=False)])
    path=tmp_path/'trace.json';n.write_json(path,trace)
    canonical=dict(request_id='r',input_ids=ids,output_ids=output,checkpoint_id='F',role='feedback')
    measured=dict(status='passed',output_ids=output,verification_calls=1,accepted_draft_tokens=1,
        verification_committed_tokens=2,trajectory=n.base.binding(path),finish_reason={'type':'stop','matched':128009})
    return canonical,measured


def test_source_and_heldout_are_natural_l8_no_capture_no_teacher_forcing(tmp_path):
    canonical,measured=canonical_fixture(tmp_path)
    heldout={**canonical,'request_id':'heldout','role':'evaluation'}
    backend=n.NativeBackend({});backend.episode_id='g';backend.training_closed=True
    backend.group=dict(domain='test',reference_pairs={'source':[{'F':{'native':canonical}}],
        'heldout':[{'F':{'native':heldout}}]})
    calls=[]
    def serve(_self,ids,label,capture,**kwargs):
        calls.append((ids,label,capture,kwargs));return dict(measured)
    with patch.object(n.base.NativeBackend,'serve',serve):
        result=backend.serve_request(canonical,'source')
        result2=backend.serve_request(heldout,'heldout')
        with pytest.raises(ValueError,match='no repetitions'):backend.serve_request(canonical,'source')
    assert calls==[([99,98],'L8',False,{'turn_index':1}),([99,98],'L8',False,{'turn_index':5})]
    assert result['checkpoint_id']=='L8' and result['eos_hit'] is True and result['cap_hit'] is False
    assert result['initial_output_tokens']==1 and result['committed_progress_per_call']==2
    assert result2['role']=='evaluation' and result2['feedback_capture_enabled'] is False
    assert result['rounds'][0]['first_rejection_offset']==1


def test_native_generation_is_blocked_until_training_group_is_gone(tmp_path):
    row,_=canonical_fixture(tmp_path);backend=n.NativeBackend({})
    with pytest.raises(ValueError,match='Stop training'):backend.serve_request(row,'source')
    backend.training_closed=True;backend.retired_trainer_pid=123
    with patch.object(n.curve,'process_group_alive',return_value=True):
        with pytest.raises(ValueError,match='child still alive'):backend.serve_request(row,'source')


def test_changed_saved_request_is_rejected(tmp_path):
    row,_=canonical_fixture(tmp_path);backend=n.NativeBackend({});backend.training_closed=True
    backend.group={'reference_pairs':{'source':[{'F':{'native':row}}]}}
    with pytest.raises(ValueError,match='differs from bound'):backend.serve_request({**row,'input_ids':[1]},'source')


def test_serving_guard_allows_only_real_l8_capture_disabled():
    service=n.ServingService.__new__(n.ServingService)
    with pytest.raises(ValueError,match='natural L8'):service.generate({'capture':True,'checkpoint_id':'L8'})
    with pytest.raises(ValueError,match='natural L8'):service.generate({'capture':False,'checkpoint_id':'U2'})
    with patch.object(n.ServingService,'native_generate',return_value={'feedback':None,'capture_costs':None}) as generate:
        assert service.generate({'capture':False,'checkpoint_id':'L8'})['feedback'] is None
        assert generate.call_count==1


def test_trace_registry_extension_does_not_alias_or_modify_historical_method():
    old=n.base._ServingService.generate
    import inspect
    assert "{'F': 0, 'U1': 1, 'U2': 2}" in inspect.getsource(old)
    fn=n._l8_generate_method()
    assert any(isinstance(v,tuple) and v==('F','L8') for v in fn.__code__.co_consts)
    assert fn is not old
    cfg=dict(workspace_root='/tmp/root',conda_bootstrap='/tmp/conda.sh',conda_prefix='/tmp/env',python_executable='/tmp/env/bin/python')
    assert n.worker_command(cfg)[-1]==str(n.HERE/'native.py')


def test_trainer_stops_and_all_children_are_verified_before_publication(tmp_path):
    backend=n.NativeBackend({});backend.training_started=True;backend.out=tmp_path;backend.checkpoints={'L8':{}}
    worker=SimpleNamespace(process=SimpleNamespace(pid=123));backend.trainer=worker
    with patch.object(n,'close_owned_worker',return_value=dict(status='passed',shutdown_s=.4)) as close, \
         patch.object(n.curve,'process_group_alive',return_value=False):
        result=backend.stop_training()
    assert close.call_args.args==(worker,) and backend.trainer is None and backend.training_closed
    assert result['training_quiescent'] and backend.totals['shutdown_s']==.4
    assert (tmp_path/'TRAINING_QUIESCENT.json').is_file()


def test_failed_training_retains_partial_without_registering_export(tmp_path):
    directory=tmp_path/'training_worker';directory.mkdir()
    n.write_json(directory/'PARTIAL.json',dict(status='failed',accounting={'optimizer_steps_total':2},losses={'source':[],'heldout':[]}))
    class Worker:
        startup_s=.1
        def __init__(self,*args):pass
        def call(self,*args):raise RuntimeError('worker failed')
    backend=n.NativeBackend({});backend.episode_id='g';backend.out=tmp_path;backend.checkpoints={'F':{}}
    with patch.object(n.base,'_WorkerClient',Worker):
        with pytest.raises(RuntimeError,match='worker failed') as error:backend.train_group()
    assert error.value.diagnostic_partial['accounting']['optimizer_steps_total']==2
    assert error.value.partial_binding['sha256']==n.sha(directory/'PARTIAL.json')
    assert 'L8' not in backend.checkpoints


def test_legacy_training_entrypoint_is_explicitly_unavailable():
    with pytest.raises(ValueError,match='explicit L8'):n.NativeBackend({}).train_event(1,[])
    assert not issubclass(n.TrainerService,n.followup.TrainerService)
    assert not hasattr(n.TrainerService,'_export') and not hasattr(n.TrainerService,'train')
    text=(n.HERE/'native.py').read_text()
    assert 'save_checkpoint_state(' not in text and 'load_checkpoint_state(' not in text
    assert 'source_diagnostic_steps =' not in text


def service_fixture(tmp_path):
    import torch
    records=[]
    for i in range(4):
        row=dict(record_id=f's{i}',episode_id='g',turn=i+1,input_ids=[10+i],output_ids=[20,21])
        path=tmp_path/f'support{i}.json';n.write_json(path,row)
        records.append({**row,'metadata_binding':n.base.binding(path)})
    refs={'source':[{'F':{'kl':{'id':f's{i}'}}} for i in range(4)],
          'heldout':[{'F':{'kl':{'id':f'h{i}'},'native':dict(request_id=f'h{i}',role='evaluation')}} for i in range(4)]}
    cfg={'episode_id':'g','original_group':{'support':records,'reference_pairs':refs}}
    service=n.TrainerService.__new__(n.TrainerService)
    service.config,service.out,service.completed=cfg,tmp_path,False
    service.draft=torch.nn.Linear(1,1)
    service.components=SimpleNamespace(embed_tokens=torch.nn.Linear(1,1),lm_head=torch.nn.Linear(1,1))
    service.teacher_features=lambda ids:None;service.check_frozen=lambda:None
    return service,records


def test_one_train_then_real_l8_export_then_both_readonly_losses(tmp_path):
    import torch
    service,records=service_fixture(tmp_path);order=[]
    cp={'checkpoint_id':'L8','checkpoint_identity_sha256':'a'*64,'export_s':.4}
    def train(rows):
        assert rows==records and all('role' not in r for r in rows)
        order.append('train8')
        return dict(optimizer_steps_total=8,response_presentations_total=16,costs={'training_s':1.})
    service.trainer=SimpleNamespace(train_to_step8=train,bind_l8_checkpoint=lambda value:order.append(('bind',value)),
        _check_owner=lambda:None)
    service._export_l8=lambda:(order.append('export_L8') or cp)
    def evaluate(draft,embed,head,teacher,rows,refs,**kwargs):
        population=kwargs['population'];order.append(('loss',population))
        assert kwargs['checkpoint_id']=='L8'
        if population=='source':
            assert all(r['role']=='feedback' and r['group_id']=='g' and r['request_id']==r['record_id'] for r in rows)
        else:assert all(r['role']=='evaluation' for r in rows)
        return dict(status='passed',response_losses=[dict(phase_seconds={'teacher_preparation_s':.01}) for _ in rows],
            diagnostic_wall_s=.2,optimizer_steps=0)
    with patch('experiments.dflash_lower_lr_control.loss.evaluate_responses',side_effect=evaluate), \
         patch.object(torch.cuda,'reset_peak_memory_stats'), \
         patch.object(torch.cuda,'max_memory_allocated',return_value=100), \
         patch.object(torch.cuda,'max_memory_reserved',return_value=200):
        result=service.train_group({})
        with pytest.raises(ValueError,match='One bounded'):service.train_group({})
    assert order==['train8','export_L8',('bind','a'*64),('loss','source'),('loss','heldout')]
    assert result['checkpoint']==cp and len(result['losses']['heldout'])==len(result['losses']['source'])==4
    assert result['costs']['source_loss_diagnostic_s']==.2 and result['costs']['optimizer_state_save_s']==0
    assert result['costs']['source_diagnostic_teacher_preparation_s']==.04


def test_partial_optimizer_progress_survives_native_service_failure(tmp_path):
    import torch
    service,_=service_fixture(tmp_path)
    failure=ValueError('sampling changed')
    failure.training_partial=dict(optimizer_steps=2,response_presentations=4,completed_batches=[{},{}])
    service.trainer=SimpleNamespace(train_to_step8=lambda rows:(_ for _ in ()).throw(failure))
    with patch.object(torch.cuda,'reset_peak_memory_stats'), \
         patch.object(torch.cuda,'max_memory_allocated',return_value=100), \
         patch.object(torch.cuda,'max_memory_reserved',return_value=200):
        with pytest.raises(ValueError,match='sampling changed'):service.train_group({})
    saved=json.loads((tmp_path/'PARTIAL.json').read_text())
    assert saved['accounting']['optimizer_steps_total']==2 and saved['accounting']['response_presentations_total']==4
    assert saved['checkpoint'] is None and saved['status']=='failed'


def test_partial_readonly_loss_does_not_discard_completed_rows(tmp_path):
    import torch
    service,_=service_fixture(tmp_path)
    service.trainer=SimpleNamespace(train_to_step8=lambda rows:dict(optimizer_steps_total=8),
        bind_l8_checkpoint=lambda _:None,_check_owner=lambda:None)
    service._export_l8=lambda:dict(checkpoint_id='L8',checkpoint_identity_sha256='a'*64)
    failure=ValueError('loss nonfinite')
    failure.diagnostic_partial=dict(response_losses=[dict(loss=.1)],bad_value=float('nan'))
    with patch('experiments.dflash_lower_lr_control.loss.evaluate_responses',side_effect=failure), \
         patch.object(torch.cuda,'reset_peak_memory_stats'), \
         patch.object(torch.cuda,'max_memory_allocated',return_value=100), \
         patch.object(torch.cuda,'max_memory_reserved',return_value=200):
        with pytest.raises(ValueError,match='loss nonfinite'):service.train_group({})
    saved=json.loads((tmp_path/'PARTIAL.json').read_text())
    assert saved['losses']['source']==[{'loss':.1}] and saved['losses']['heldout']==[]
    assert saved['failure_detail']['bad_value']['nonfinite_float']=='nan' and saved['checkpoint']['checkpoint_id']=='L8'


def test_before_training_metadata_failure_has_durable_zero_work(tmp_path):
    import torch
    service,records=service_fixture(tmp_path)
    Path(records[0]['metadata_binding']['path']).write_text('{"tampered":true}')
    called=[]
    service.trainer=SimpleNamespace(optimizer_steps=0,response_presentations=0,
        train_to_step8=lambda rows:called.append('training'))
    with patch.object(torch.cuda,'max_memory_allocated',return_value=100), \
         patch.object(torch.cuda,'max_memory_reserved',return_value=200):
        with pytest.raises(ValueError):service.train_group({})
    assert called==[]
    saved=json.loads((tmp_path/'PARTIAL.json').read_text())
    assert saved['accounting']['optimizer_steps_total']==saved['accounting']['response_presentations_total']==0
    assert saved['accounting']['status']=='failed' and saved['checkpoint'] is None
    assert (tmp_path/'TRAINING_BEGIN.json').is_file()


def test_failed_training_attempts_are_preserved_separately_from_applied_steps(tmp_path):
    import torch
    service,_=service_fixture(tmp_path)
    failure=ValueError('exposure mismatch')
    failure.training_partial=dict(optimizer_steps=2,response_presentations=4,completed_batches=[{},{}],
        attempted_response_calls=6,completed_backward_presentations=6,uncommitted_backward_presentations=2)
    service.trainer=SimpleNamespace(optimizer_steps=2,response_presentations=4,
        train_to_step8=lambda rows:(_ for _ in ()).throw(failure))
    with patch.object(torch.cuda,'reset_peak_memory_stats'), \
         patch.object(torch.cuda,'max_memory_allocated',return_value=100), \
         patch.object(torch.cuda,'max_memory_reserved',return_value=200):
        with pytest.raises(ValueError,match='exposure mismatch'):service.train_group({})
    saved=json.loads((tmp_path/'PARTIAL.json').read_text())
    assert saved['accounting']['optimizer_steps_total']==2 and saved['accounting']['response_presentations_total']==4
    assert saved['accounting']['attempted_response_calls']==6
    assert saved['accounting']['completed_backward_presentations']==6
    assert saved['accounting']['uncommitted_backward_presentations']==2
