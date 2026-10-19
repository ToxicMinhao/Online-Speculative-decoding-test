"""Controller, accounting and owned-retention checks without model execution."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from . import native
from experiments.dflash_correction_bank.artifacts import atomic_json, binding, save_snapshot


def fake_backend(tmp_path):
    backend=native.NativeBackend({})
    backend.out=tmp_path;backend.qualification=False;backend.projections={}
    backend.cost_profile=None
    backend._ensure_worker=lambda kind:0.
    backend.publish=lambda label:dict(activation_s=7.,native_activation_s=2.)
    return backend


def test_bank_request_timer_excludes_activation_and_cache_but_repeats_read(tmp_path):
    backend=fake_backend(tmp_path);calls=[]
    def call(action,body):
        calls.append((action,copy.deepcopy(body)))
        if action=='bank_activate':return {'loading_s':1.}
        if action=='flush':return {'seconds':3.}
        return dict(input_ids=[1,2],input_ids_sha256=native.digest([1,2]),output_ids=[3],read={})
    backend.worker=SimpleNamespace(call=call)
    state=dict(arm='BANK',turn=1,state_sha256='a'*64)
    with patch.object(native.time,'perf_counter',side_effect=[10.,12.,20.,25.]):
        result=backend.serve([1,2],'BANK',state,turn_index=1,repeat=0)
    assert result['request_s']==5. and result['activation_s']==9.
    assert result['native_activation_s']==2. and result['cache_maintenance_s']==3.
    assert result['deployment_activation_s']==0.
    assert [c[0] for c in calls]==['bank_activate','flush','generate']
    assert calls[-1][1]['capture'] is False and calls[-1][1]['canonical'] is False
    with patch.object(native.time,'perf_counter',side_effect=[30.,32.,40.,45.]):
        canonical=backend.serve([1,2],'BANK',state,turn_index=1,canonical=True)
    assert canonical['deployment_activation_s']==2. and canonical['benchmark_method_restore_s']==7.


def test_profile_waits_for_nonempty_native_blocks_instead_of_crashing_short_first_answer(tmp_path):
    backend=fake_backend(tmp_path);backend.qualification=True;calls=[]
    def call(action,body):
        calls.append(action)
        if action=='bank_activate':return {}
        if action=='flush':return {'seconds':0.}
        assert action=='generate'
        return dict(input_ids=[1],input_ids_sha256=native.digest([1]),output_ids=[128009],
                    correction_bank_cycle_observations=[])
    backend.worker=SimpleNamespace(call=call)
    result=backend.serve([1],'BANK',dict(arm='BANK',turn=1,state_sha256='a'*64),turn_index=1,canonical=True)
    assert result['qualification_cost_profile_status']=='deferred_no_native_draft_blocks'
    assert 'bank_profile' not in calls and backend.cost_profile is None


@pytest.mark.parametrize('arm,canonical,expected_capture', [('BANK',True,True),('BANK',False,False),('OSD',True,False),('F',False,False)])
def test_service_head_capture_is_distinct_from_inherited_feedback_capture(tmp_path,arm,canonical,expected_capture):
    service=native.ServingService.__new__(native.ServingService)
    service.out=tmp_path;service.serial=0;service.logical_checkpoint='D1' if arm=='OSD' else 'F'
    read={'forecast_saved_before_response':True} if arm=='BANK' else None
    trace=atomic_json(tmp_path/'trace.json',dict(correction_bank_read=read,bank_head_capture=None,
                                               correction_bank_cycle_observations=[]))
    body=dict(arm=arm,state={'arm':arm,'state_sha256':'a'*64},input_ids=[1,2],
        projections={},checkpoint_id=service.logical_checkpoint,canonical=canonical,capture=False,turn=2)
    with patch.object(native.per_turn.ServingService,'generate',return_value={'trajectory':trace}) as parent:
        result=service.generate(body)
    declaration=json.loads((tmp_path/'BANK_PENDING.json').read_text())
    assert declaration['capture'] is expected_capture
    assert parent.call_args.args[0]['capture'] is False
    assert result['read']==read


def test_native_profile_rpc_is_separate_from_generation(tmp_path):
    service=native.ServingService.__new__(native.ServingService);service.out=tmp_path
    def rpc(name):
        assert name=='correction_bank_profile'
        (tmp_path/'BANK_PROFILE.json').write_text('{"paths": []}')
    service.engine=SimpleNamespace(collective_rpc=rpc)
    assert service.bank_profile({})=={'paths':[]}


def test_worker_entry_and_sources_are_v02_without_mutating_historical_command():
    config=dict(workspace_root='/workspace',conda_bootstrap='/conda/bootstrap',conda_prefix='/env',python_executable='/env/bin/python')
    assert native.worker_command(config)[-1]==str(native.HERE/'native.py')
    assert native._original_worker_command(config)[-1]==str(native.base.HERE/'native.py')
    assert native.base.worker_command is native.worker_command
    assert native.base.OUTPUT_CAP==1024 and native.base.BLOCK_SIZE==10
    assert native.base.PREFILL_CHUNK_SIZE==8192
    assert {'native_hooks.py','core.py','state.py','feedback.py'} <= set(native.source_pins())


def test_storage_budget_includes_all_compact_states_one_rolling_osd_and_staging():
    M=2*2**30;budget=native.campaign_storage_bound(M)
    assert budget['active_episode_osd_peak_bytes']==11*M+512*2**20
    assert budget['compact_artifacts_bytes']==6*512*2**20+2*2**30
    assert budget['staging_bytes']>=2*2**30
    assert budget['planned_writes_bytes']==budget['active_episode_osd_peak_bytes']+budget['compact_artifacts_bytes']+budget['staging_bytes']
    assert budget['required_free_bytes']==budget['planned_writes_bytes']+8*2**30
    assert budget['assumptions']['retire_completed_osd_after_validated_result']
    with pytest.raises(ValueError,match='envelopes'):
        native.campaign_storage_bound(M,main_bank_bytes=1)


def capture_case(tmp_path):
    backend=fake_backend(tmp_path)
    folder=tmp_path/'worker_001_sd'/'bank_forecasts';folder.mkdir(parents=True)
    npz=folder/'000001.heads.npz';np.savez(npz,frozen_h=np.ones((1,9,4),np.float32))
    ids,outputs=[1,2],[3,4]
    cap=atomic_json(folder/'000001.heads.json',dict(input_ids_sha256=native.digest(ids),
        output_ids_sha256=native.digest(outputs),tensors=binding(npz)))
    memory={'h':np.ones((1,4),np.float32),'mu':np.ones((1,4),np.float32),'omega':np.ones(1),
            'token_ids':[4],'output_positions':[1]}
    feedback=save_snapshot(dict(response=dict(input_ids=ids,output_ids=outputs),memory=[memory],metadata={},
        reservation={},rng_before=np.arange(3),rng_after=np.arange(3),canonical_checks=[]),tmp_path/'feedback.json')
    solver=save_snapshot(dict(trials=[]),tmp_path/'solver.json')
    state=save_snapshot(dict(memory=[memory]),tmp_path/'bank.json')
    trajectory=atomic_json(tmp_path/'trajectory.json',dict(input_token_ids=ids,final_output_token_ids=outputs))
    item=dict(canonical=True,arm='BANK',turn=1,input_ids=ids,output_ids=outputs,bank_head_capture=cap,trajectory=trajectory)
    update=dict(status='passed',event=1,feedback=feedback,solver_diagnostics=solver,bank_snapshot=state)
    return backend,item,update,npz


def test_transient_capture_retirement_preserves_compact_evidence_and_metadata(tmp_path):
    backend,item,update,npz=capture_case(tmp_path)
    result=backend.retire_native_capture(item,update)
    assert result['status']=='retired' and not npz.exists()
    assert Path(item['bank_head_capture']['path']).exists()
    assert all(Path(update[key]['path']).exists() for key in ('feedback','solver_diagnostics','bank_snapshot'))
    assert 'not archived' in result['lost_capability']
    assert Path(result['receipt']['path']).exists()
    assert (tmp_path/'bank_capture_retirement'/'turn_01_PLAN.json').exists()


def test_qualification_original_capture_is_retained(tmp_path):
    backend,item,update,npz=capture_case(tmp_path);backend.qualification=True
    assert backend.retire_native_capture(item,update)['status']=='retained'
    assert npz.exists()


def test_corrupt_retained_array_blocks_capture_retirement(tmp_path):
    backend,item,update,npz=capture_case(tmp_path)
    feedback=json.loads(Path(update['feedback']['path']).read_text())
    array=Path(feedback['value']['memory'][0]['h']['__array__']['path'])
    array.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Artifact content changed'):
        backend.retire_native_capture(item,update)
    assert npz.exists()


def test_foreign_capture_cannot_be_retired_even_with_matching_feedback(tmp_path):
    backend,item,update,npz=capture_case(tmp_path)
    foreign=tmp_path/'foreign.npz';foreign.write_bytes(npz.read_bytes())
    doc=json.loads(Path(item['bank_head_capture']['path']).read_text());doc['tensors']=binding(foreign)
    item['bank_head_capture']=atomic_json(tmp_path/'foreign.json',doc)
    with pytest.raises(ValueError,match='outside owned'):
        backend.retire_native_capture(item,update)
    assert npz.exists() and foreign.exists()


def test_unchanged_osd_event_trains_only_own_response_once_via_existing_two_pass_trainer(tmp_path):
    backend=fake_backend(tmp_path);backend.osd_after_turn=0
    response=dict(canonical=True,arm='OSD',state_sha256='a'*64,turn=1)
    backend.prepare_support=Mock(return_value={'own_feedback':True,'collection_s':.2})
    backend.train_event=Mock(return_value=dict(costs={},accounting={'optimizer_steps_this_event':2}))
    backend.retire=Mock(return_value={'status':'passed'})
    result=backend.learn('OSD',response,{'state_sha256':'a'*64},turn_index=1)
    backend.train_event.assert_called_once_with(1,[{'own_feedback':True,'collection_s':.2}])
    assert result['accounting']['optimizer_steps_this_event']==2
    assert backend.osd_after_turn==1
    with pytest.raises(ValueError,match='Wrong OSD update'):
        backend.learn('OSD',response,{'state_sha256':'a'*64},turn_index=1,final=True)


def test_bank_learning_excludes_separate_diagnostics_but_counts_retirement_once(tmp_path):
    backend=fake_backend(tmp_path);backend.bank_after_turn=0;backend.bank_updates=[];backend.qualification_evidence=[]
    backend.target={'identity_sha256':'t'*64};backend.retire_native_capture=Mock(return_value={'status':'retired'})
    update=dict(status='passed',event=1,bank_snapshot={'path':'new'},
                costs=dict(qualification_only_s=.2,source_fit_diagnostic_only_s=.3))
    backend.trainer=SimpleNamespace(call=lambda action,body:copy.deepcopy(update))
    item=dict(canonical=True,arm='BANK',state_sha256='a'*64,input_ids=[1],output_ids=[2],read={})
    with patch.object(native.time,'perf_counter',side_effect=itertools.count(1.)),patch.object(native.base,'checked_binding'):
        result=backend.learn('BANK',item,{'state_sha256':'a'*64,'snapshot':{}},turn_index=1)
    assert result['feedback_rpc_wall_s']==1.
    assert result['nondeployment_diagnostic_s']==.5
    assert result['costs']['native_capture_retirement_s']==1.
    assert result['recurring_adaptation_s']==1.5


def test_admission_failure_closes_safely_without_model_or_prior_output_mutation():
    from .experiment import run_stream
    from .test_experiment import stream
    backend=native.NativeBackend({})
    with patch.object(native,'admit_campaign_storage',side_effect=ValueError('budget rejected')), \
         patch.object(native.base,'_WorkerClient') as worker:
        result=run_stream(stream(),backend,qualification=True)
    assert result['error']=='ValueError: budget rejected'
    assert result['cleanup']['status']=='passed' and not result['cleanup']['episode_initialized']
    worker.assert_not_called()
