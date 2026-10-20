"""Synthetic native-wrapper tests. These do not qualify the external GPU runtime."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from . import native as n


def test_physical_step_labels_use_existing_publication_path():
    backend=n.NativeBackend({})
    with patch.object(n.base.NativeBackend,'publish',side_effect=lambda label:{'checkpoint_id':label}):
        assert [backend.publish(x)['native_checkpoint_id'] for x in ('F','S2','S8')] == ['F','U1','U2']
        with pytest.raises(ValueError): backend.publish('S4')


def test_evaluation_cannot_overlap_live_training():
    backend=n.NativeBackend({})
    backend.training_closed=False
    backend.retired_trainer_pid=None
    with pytest.raises(ValueError,match='must be stopped'):
        backend.serve([1],'F','evaluation','test')
    backend.training_closed=True
    backend.trainer=object()
    with pytest.raises(ValueError,match='must be stopped'): backend.assert_training_quiescent()
    backend.trainer=None
    backend.retired_trainer_pid=123
    with patch.object(n,'process_group_alive',return_value=True):
        with pytest.raises(ValueError,match='survives'): backend.assert_training_quiescent()
    with patch.object(n,'process_group_alive',return_value=False):
        assert backend.assert_training_quiescent()['training_quiescent']


def test_stop_trainer_reaps_before_marking_quiescence(tmp_path):
    backend=n.NativeBackend({})
    backend.out=tmp_path
    backend.training_started=True
    backend.training_closed=False
    process=SimpleNamespace(pid=123,poll=lambda:0)
    marks=[]
    backend.trainer=SimpleNamespace(process=process,close=lambda:marks.append('closed'))
    with patch.object(n,'process_group_alive',return_value=False):
        receipt=backend.stop_trainer()
    assert marks==['closed'] and receipt['training_quiescent']
    assert backend.trainer is None
    assert json.loads((tmp_path/'TRAINING_QUIESCENT.json').read_text())['terminated_process_group']==123


def fixture():
    ids,output=[99,98],[10,11,12,13]
    events=[dict(kind='target_prefill',committed_token_ids=[10]),dict(kind='verification',output_offset=1,
        draft_proposed_token_ids=[11,12,50],rejection_boundary_proposal_index=2,
        committed_token_ids=[11,12,13],committed_provenance=['accepted_draft','accepted_draft','target_correction'],
        stop_clipped=False)]
    document=dict(input_token_ids=ids,final_output_token_ids=output,events=events,reconciliation={'status':'passed'})
    result=dict(output_ids=output,verification_calls=1,accepted_draft_tokens=2,verification_committed_tokens=3)
    return ids,document,result


def test_round_counts_distinguish_accepted_from_bonus_and_initial():
    ids,document,result=fixture()
    rounds,initial=n.normalize_rounds(document,result,ids)
    assert initial==1
    assert rounds[0]['accepted_draft_tokens']==2
    assert rounds[0]['committed_tokens']==3
    assert rounds[0]['output_offset']==1 and rounds[0]['absolute_token_offset']==3
    assert rounds[0]['first_rejection_offset']==2


@pytest.mark.parametrize('damage',['offset','accepted','sequence','calls'])
def test_corrupt_native_trace_fails(damage):
    ids,doc,result=fixture()
    if damage=='offset':doc['events'][1]['output_offset']=2
    if damage=='accepted':result['accepted_draft_tokens']=3
    if damage=='sequence':doc['events'][1]['committed_token_ids'][-1]=20
    if damage=='calls':result['verification_calls']=2
    with pytest.raises(ValueError):n.normalize_rounds(doc,result,ids)


def test_single_token_eos_without_verification_is_valid():
    result=dict(output_ids=[128009],verification_calls=0,accepted_draft_tokens=0,verification_committed_tokens=0)
    doc=dict(input_token_ids=[42],final_output_token_ids=[128009],reconciliation={'status':'passed'},
             events=[dict(kind='target_prefill',committed_token_ids=[128009])])
    assert n.normalize_rounds(doc,result,[42])==([],1)


def test_storage_budgets_every_group_without_retirement():
    size=2**30
    assert n.group_storage_bound(size)>=7*size
    assert n.campaign_storage_bound(size)==6*n.group_storage_bound(size)
    with pytest.raises(ValueError):n.campaign_storage_bound(size,5)


def test_import_keeps_historical_native_cap_unchanged():
    from experiments.dflash_two_update import native as old
    assert old.OUTPUT_CAP==256 and n.base.OUTPUT_CAP==1024


def test_worker_uses_reviewed_entrypoint():
    config=dict(workspace_root='/tmp/root',conda_bootstrap='/tmp/conda.sh',conda_prefix='/tmp/env',python_executable='/tmp/env/bin/python')
    assert n.worker_command(config)[-1]==str(n.HERE/'native.py')


def test_actual_fixed_pool_trainer_matches_native_receipt_and_controller(tmp_path):
    """Exercise the real new trainer through the native service's orchestration.

    Only model loading/export and CUDA counters are replaced with tiny CPU fixtures.
    No teacher or serving GPU process is launched.
    """
    import torch
    from .test_training import make_trainer,pool,RELEASED,S2,S8
    from .experiment import validate_training
    trainer,_=make_trainer()
    service=n.TrainerService.__new__(n.TrainerService)
    service.trainer=trainer
    service.out=tmp_path/'worker';service.out.mkdir()
    support=tmp_path/'support';support.mkdir()
    released=dict(checkpoint_identity_sha256=RELEASED,checkpoint_id='F')
    service.config=dict(episode_out=str(tmp_path),released=released,reserve_bytes=0)
    service.completed=False
    responses=pool()
    for i,r in enumerate(responses):
        path=support/f'{i}.json';n.write_json(path,r)
        r['metadata_binding']=n.base.binding(path)
    def export(event):
        path=tmp_path/'checkpoints'/f'U{event}';path.mkdir(parents=True)
        return dict(path=str(path),checkpoint_identity_sha256={1:S2,2:S8}[event],
                    checkpoint_id=f'U{event}',export_s=0.01,files={})
    service._export=export
    with patch.object(torch.cuda,'reset_peak_memory_stats'), \
         patch.object(torch.cuda,'max_memory_allocated',return_value=123), \
         patch.object(torch.cuda,'max_memory_reserved',return_value=456):
        receipt=service.train({'responses':responses})
    assert validate_training(receipt,responses)
    assert receipt['accounting']['optimizer_steps']==8
    assert receipt['accounting']['response_presentations']==16
    assert [u['accounting']['optimizer_steps_this_event'] for u in receipt['updates']]==[2,6]
    assert [r['checkpoint_id'] for r in receipt['source_losses']]==['F','S2','S8']
    assert receipt['costs']['source_loss_diagnostic_s']>0
    assert receipt['costs']['peak_allocated_gpu_bytes']==123
    assert (tmp_path/'checkpoints/U1_training_state.pt').is_file()
    assert (tmp_path/'checkpoints/U2_training_state.pt').is_file()
    with pytest.raises(ValueError,match='Exactly one'):
        service.train({'responses':responses})
