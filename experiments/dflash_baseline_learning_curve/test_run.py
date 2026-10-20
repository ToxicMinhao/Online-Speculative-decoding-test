"""Metadata and storage-admission tests; never allocate a GPU."""
import json
import pytest
from . import run
from .native import campaign_storage_bound, group_storage_bound


def test_storage_accounts_for_six_retained_groups_and_reserve(tmp_path):
    model=tmp_path/'models.json'
    model.write_text(json.dumps({'released_draft':{'checkpoint_files':{'model.safetensors':{'bytes':1024}}}}))
    config=dict(model_bindings=run.file_binding(model),output_root=str(tmp_path/'native'),
        episode_storage_budget_bytes=group_storage_bound(1024),
        campaign_storage_budget_bytes=campaign_storage_bound(1024),reserve_bytes=8*2**30)
    required=campaign_storage_bound(1024)+config['reserve_bytes']
    assert run.storage_preflight(config,free_bytes=required)['total_required_free_bytes']==required
    with pytest.raises(ValueError,match='do not fit'):
        run.storage_preflight(config,free_bytes=required-1)
    config['campaign_storage_budget_bytes']=group_storage_bound(1024)
    with pytest.raises(ValueError,match='omits retained groups'):
        run.storage_preflight(config,free_bytes=required)


def test_plan_has_no_extra_qualification_group():
    result=run.plan({'groups':[{'group_id':str(i)} for i in range(6)]})
    assert result['core_response_generations']==96
    assert result['optimizer_steps']==48
    assert result['training_response_presentations']==96
    assert result['explicit_warmup_generations']==18
    assert result['source_loss_diagnostic_generations']==0
    assert result['gpu_started'] is False


def test_write_json_preserves_newlines_and_replaces_durably(tmp_path):
    path=tmp_path/'result.json'
    run.write_json(path,{'text':'def f():\n    return 1\n'})
    assert json.loads(path.read_text())['text'].endswith('    return 1\n')
    run.write_json(path,{'status':'complete'})
    assert json.loads(path.read_text())=={'status':'complete'}
    assert not path.with_name('result.json.partial').exists()
