import json
from pathlib import Path

import pytest

from . import run
from .test_experiment import groups_fixture


@pytest.fixture
def fake_bundle(monkeypatch):
    from . import native
    from experiments.dflash_baseline_learning_curve import native as curve
    bundle=dict(groups=groups_fixture(),runtime_config={'learning_curve_source_sha256':curve.source_pins()},
                bindings={'baseline_commit':'165e25b7a85cabfe16f7b9d642fb575b23d82ad6'})
    monkeypatch.setattr(run,'load_evidence',lambda root:bundle)
    monkeypatch.setattr(run,'_git_identity',lambda:dict(commit='diagnostic_commit',dirty=False))
    return bundle,run.digest(native.source_pins())


def test_plan_never_constructs_backend(tmp_path,monkeypatch,fake_bundle):
    from . import native
    def forbidden(*args):raise AssertionError('Native backend constructed during plan')
    monkeypatch.setattr(native,'NativeBackend',forbidden)
    out=tmp_path/'plan'
    assert run.main(['plan','--baseline-root',str(tmp_path/'baseline'),'--out',str(out)])==0
    d=json.loads((out/'PLAN.json').read_text())
    assert d['new_source_generations']==24 and d['new_response_kl_evaluations']==48 and d['optimizer_steps']==0


def test_plan_cannot_accept_execution_flag(tmp_path,fake_bundle):
    with pytest.raises(ValueError,match='Plan never'):
        run.main(['plan','--baseline-root',str(tmp_path/'baseline'),'--out',str(tmp_path/'plan'),'--allow-gpu-execution'])


@pytest.mark.parametrize('args',[[],['--allow-gpu-execution'],['--reviewed-code-sha256','wrong']])
def test_run_requires_reviewed_execution_gate(tmp_path,fake_bundle,args):
    with pytest.raises(ValueError,match='Execution requires'):
        run.main(['run','--baseline-root',str(tmp_path/'baseline'),'--out',str(tmp_path/'run')]+args)


def test_original_evidence_cannot_be_output(tmp_path,fake_bundle):
    with pytest.raises(ValueError,match='separate'):
        run.main(['plan','--baseline-root',str(tmp_path/'baseline'),'--out',str(tmp_path/'baseline/diagnostic')])


def test_dirty_checkout_refuses_gpu(tmp_path,monkeypatch,fake_bundle):
    monkeypatch.setattr(run,'_git_identity',lambda:dict(commit='x',dirty=True))
    with pytest.raises(ValueError,match='clean reviewed'):
        run.main(['run','--baseline-root',str(tmp_path/'baseline'),'--out',str(tmp_path/'run'),
                  '--allow-gpu-execution','--reviewed-code-sha256',fake_bundle[1]])


def test_readonly_report_action_does_not_load_original_or_native(tmp_path,monkeypatch):
    def forbidden(*a):raise AssertionError('No baseline needed to report existing diagnostic')
    monkeypatch.setattr(run,'load_evidence',forbidden)
    result=tmp_path/'result.json';result.write_text(json.dumps({'status':'not_executed','groups':[]}))
    assert run.main(['report','--result',str(result),'--out',str(tmp_path/'report')])==0
    assert (tmp_path/'report/REPORT.md').is_file()
