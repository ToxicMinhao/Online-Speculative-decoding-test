"""Admission tests ensure planning/reporting cannot silently launch work."""
import json
import pytest
from . import run


def test_plan_is_one_new_arm_and_exact_budget():
    value=run.plan({'groups':[{'group_id':str(i)} for i in range(6)]})
    assert value['new_arm']=='L8' and value['reused_arms']==['F','S8']
    assert (value['optimizer_steps'],value['training_presentations'],value['native_generations'],
        value['response_kl_evaluations'])==(48,96,48,48)
    assert value['explicit_warmups']==value['optimizer_state_exports']==0
    assert value['checkpoints_to_export']==6 and not value['gpu_started']
    assert 'Both S8 and L8' in value['initialization']
    assert 'Held-out answers never train' in value['supervision']


def test_refuses_output_inside_original_evidence_before_loading(tmp_path,monkeypatch):
    root=tmp_path/'old';root.mkdir()
    monkeypatch.setattr(run,'load_evidence',lambda *_:pytest.fail('must reject before evidence load'))
    with pytest.raises(ValueError,match='separate from retained evidence'):
        run.main(['plan','--baseline-root',str(root),'--diagnostic-root',str(tmp_path/'other'),
            '--out',str(root/'new')])


def test_refuses_existing_output(tmp_path):
    with pytest.raises(ValueError,match='fresh private'):
        run.main(['plan','--out',str(tmp_path)])


def test_missing_diagnostic_root_never_runs(tmp_path):
    with pytest.raises(ValueError,match='Both retained'):
        run.main(['run','--baseline-root',str(tmp_path),'--out',str(tmp_path/'new'),
            '--allow-gpu-execution'])


def test_report_cannot_request_gpu_execution(tmp_path):
    with pytest.raises(ValueError,match='Report reads only'):
        run.main(['report','--result',str(tmp_path/'result.json'),'--out',str(tmp_path/'new'),
            '--allow-gpu-execution'])


def test_pinned_evidence_detects_changed_raw_bytes(tmp_path):
    from .evidence import checked_file,file_binding
    path=tmp_path/'RESULT.json';path.write_text('{"status":"complete"}')
    binding=file_binding(path)
    assert checked_file(path,binding)==binding
    path.write_text('{"status":"failed"}')
    with pytest.raises(ValueError,match='Original evidence changed'):checked_file(path,binding)
