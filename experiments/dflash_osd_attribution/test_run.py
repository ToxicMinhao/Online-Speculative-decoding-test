import copy
import json
from pathlib import Path
import types

import pytest

from . import run
from .protocol import SCHEMA, digest
from .test_protocol import paired


def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'source_manifest', lambda: {'fixture.py':'fixed'})
    a, f, o = paired()
    cfg = tmp_path/'original.json'; cfg.write_text('{}')
    a.update(native_config=run.binding(cfg), projections={'path':'fixed'})
    value = dict(schema=SCHEMA, campaign_root=str(tmp_path/'campaign'), anchors=[a], cases=[],
        planned_draft_block_evaluations=2, execution_claim_root=str(tmp_path/'claims'), mapping=[])
    return run.seal_plan(value), f, o


def test_sealed_plan_source_and_digest(tmp_path, monkeypatch):
    value, _, _ = plan(tmp_path, monkeypatch)
    run.validate_plan(value)
    changed = copy.deepcopy(value); changed['anchors'][0]['turn'] = 99
    with pytest.raises(ValueError, match='Plan changed'):
        run.validate_plan(changed)
    monkeypatch.setattr(run, 'source_manifest', lambda:{'fixture.py':'new'})
    with pytest.raises(ValueError, match='implementation'):
        run.validate_plan(value)


def test_scope_still_rejected_when_resealed(tmp_path, monkeypatch):
    value, _, _ = plan(tmp_path, monkeypatch)
    value.pop('plan_digest'); value['anchors'][0]['turn'] = 19
    value = run.seal_plan(value)
    with pytest.raises(ValueError, match='Non-final'):
        run.validate_plan(value)


def test_storage_budget_has_no_model_export_and_retains_reserve(monkeypatch, tmp_path):
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda _: types.SimpleNamespace(free=12*run.GIB))
    result = run.storage_preflight(tmp_path,160)
    assert result['checkpoint_copies'] == result['training_states'] == 0
    assert result['reserve_bytes'] == 8*run.GIB
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda _: types.SimpleNamespace(free=10*run.GIB))
    with pytest.raises(ValueError, match='Insufficient'):
        run.storage_preflight(tmp_path,160)


def test_probe_one_claim_per_arm_no_retries_and_preserves_config(tmp_path, monkeypatch):
    value, f, o = plan(tmp_path, monkeypatch)
    monkeypatch.setattr(run, 'preflight_anchors', lambda p,r:(p['anchors'],[]))
    calls, configs = [], []
    class Client:
        def __init__(self,config,out): configs.append(config)
        def evaluate(self,anchor,checkpoint,arm,projections=None):
            assert (tmp_path/'claims'/(digest([anchor['anchor_id'],arm])+'.json')).exists()
            calls.append(arm); return f if arm=='F' else o
        def close(self): calls.append('close')
    result = run.probe(value,tmp_path/'out',value['source_digest'],client_factory=Client,preflight=lambda *a:{})
    assert result['claimed_block_evaluations'] == result['returned_block_evaluations'] == 2
    assert calls == ['F','OSD','close']
    assert result['comparisons'][0]['decisions'][0]['capacity']['classification'] == 'excluded'
    assert json.loads((tmp_path/'out/wildchat_00/ORIGINAL_NATIVE_CONFIG.json').read_text()) == {}
    with pytest.raises(FileExistsError):
        run.probe(value,tmp_path/'other',value['source_digest'],client_factory=Client,preflight=lambda *a:{})
    assert calls == ['F','OSD','close']


def test_failed_model_call_is_claimed_and_not_retried(tmp_path, monkeypatch):
    value, _, _ = plan(tmp_path, monkeypatch)
    monkeypatch.setattr(run,'preflight_anchors',lambda p,r:(p['anchors'],[]))
    calls=[]
    class Client:
        def __init__(self,*a): pass
        def evaluate(self,*a,**kw): calls.append('failed'); raise RuntimeError('fixture failure')
        def close(self): calls.append('close')
    with pytest.raises(RuntimeError,match='stopped without retry'):
        run.probe(value,tmp_path/'out',value['source_digest'],client_factory=Client,preflight=lambda *a:{})
    result=json.loads((tmp_path/'out/DIAGNOSTIC.json').read_text())
    assert result['claimed_block_evaluations']==1 and result['returned_block_evaluations']==0
    assert result['failure']['automatic_retry'] is False and calls==['failed','close']


def test_review_digest_required_before_model_launch(tmp_path, monkeypatch):
    value,_,_=plan(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='reviewed source'):
        run.probe(value,tmp_path/'out','wrong',client_factory=lambda *a:pytest.fail('No model allowed'))
    assert not (tmp_path/'out').exists()


def test_historical_evidence_output_is_forbidden(tmp_path, monkeypatch):
    value,_,_=plan(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='original campaign'):
        run.fresh_output(tmp_path/'campaign/overwrite',value)


def test_inventory_never_follows_with_model_or_analysis(tmp_path, monkeypatch, capsys):
    value,_,_=plan(tmp_path,monkeypatch)
    value.pop('plan_digest')
    monkeypatch.setattr(run,'inventory_campaign',lambda *a:value)
    monkeypatch.setattr(run,'probe',lambda *a:pytest.fail('Inventory must not launch models'))
    monkeypatch.setattr(run,'analyze',lambda *a:pytest.fail('Inventory must not analyze results'))
    run.main(['inventory','--campaign-root',str(tmp_path/'campaign'),'--out',str(tmp_path/'inventory')])
    receipt=json.loads(capsys.readouterr().out)
    assert receipt['models_executed']==0
    assert (tmp_path/'inventory/PLAN.json').exists()


def test_combined_report_requires_same_plan_and_runs_no_models(tmp_path,monkeypatch):
    value,_,_=plan(tmp_path,monkeypatch)
    p=tmp_path/'PLAN.json'; p.write_text(json.dumps(value))
    a=tmp_path/'ANALYSIS.json'; a.write_text(json.dumps({'analysis':{'plan_digest':value['plan_digest']}}))
    d=tmp_path/'DIAGNOSTIC.json'; d.write_text(json.dumps({'plan_digest':'other'}))
    monkeypatch.setattr(run,'probe',lambda *a:pytest.fail('Report must not launch models'))
    with pytest.raises(ValueError,match='Evidence plans differ'):
        run.main(['report','--plan',str(p),'--analysis',str(a),'--diagnostic',str(d),'--out',str(tmp_path/'report')])
