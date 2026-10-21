"""CPU-only CLI admission: preparation cannot start native work."""
import copy
import json
from pathlib import Path
import pytest
from . import run, evidence, native, storage
from .protocol import SERVING_CONTRACT


def fixture_bundle():
    groups=[dict(group_id=g,domain=d,group_payload_sha256='group-'+g,children=[])
            for d in ('gsm8k','codealpaca','sharegpt') for g in (d+'_00',d+'_01')]
    return dict(groups=groups,bindings={'original':'verified'},parent_selection={'parents':'fixed'},
                runtime_config={'original_runtime':'bound'})


def mock_admission(monkeypatch):
    bundle=fixture_bundle(); manifest={'sealed':'requests'}
    monkeypatch.setattr(evidence,'load_evidence',lambda *args:copy.deepcopy(bundle))
    monkeypatch.setattr(evidence,'bind_children',lambda b,m:copy.deepcopy(b['groups']))
    monkeypatch.setattr(native,'source_pins',lambda:{'related':'code'})
    monkeypatch.setattr(storage,'build_storage_plan',lambda gs,cfg:dict(runtime=cfg))
    monkeypatch.setattr(run,'_git_identity',lambda:dict(commit='reviewed',dirty=False))
    monkeypatch.setattr(native,'NativeBackend',lambda cfg:pytest.fail('Native model work forbidden in this test'))
    return bundle,manifest


def argv(tmp_path,action='plan'):
    return [action,'--baseline-root',str(tmp_path/'baseline'),'--diagnostic-root',str(tmp_path/'diagnostic'),
            '--out',str(tmp_path/'out')]


def plan_args(tmp_path,manifest):
    p=tmp_path/'manifest.json';p.write_text(json.dumps(manifest))
    return argv(tmp_path)+['--manifest',str(p)]


def run_args(tmp_path,bundle,manifest):
    a=plan_args(tmp_path,manifest);a[0]='run'
    p=run.plan(bundle,manifest,bundle['groups']);p['plan_sha256']=run.digest(p)
    seal=tmp_path/'plan.json';seal.write_text(json.dumps(p))
    return a+['--sealed-plan',str(seal),'--tokenizer',str(tmp_path/'tokenizer'),
              '--allow-gpu-execution','--approved-code-sha256',run.digest({'related':'code'})]


def test_plan_budget_contract():
    # Storage receives the exact serving-contract addition, not a smaller config.
    from unittest.mock import patch
    b=fixture_bundle()
    with patch.object(storage,'build_storage_plan',return_value={}) as build:
        p=run.plan(b,{'sealed':True},b['groups'])
    assert (p['groups'],p['children'],p['native_generations'])==(6,12,24)
    assert all(p[k]==0 for k in ('optimizer_steps','training_presentations','response_kl_evaluations','new_checkpoints','explicit_warmups'))
    assert p['arms']==['F','S8'] and p['serving_contract']==SERVING_CONTRACT
    assert build.call_args.args[1]['serving_contract']==SERVING_CONTRACT
    assert not p['gpu_started']


def test_plan_seals_without_model_work(tmp_path,monkeypatch):
    b,m=mock_admission(monkeypatch)
    assert run.main(plan_args(tmp_path,m))==0
    p=json.loads((tmp_path/'out/PLAN.json').read_text());h=p.pop('plan_sha256')
    assert run.digest(p)==h
    assert not (tmp_path/'out/native').exists()
    assert json.loads((tmp_path/'out/REQUEST_MANIFEST.json').read_text())==m


def test_select_only_writes_prompt_packet(tmp_path,monkeypatch):
    mock_admission(monkeypatch)
    def packet(bundle,out):
        out.mkdir();run.write_json(out/'PARENTS_PROMPT_ONLY.json',{'prompts':['only']})
    monkeypatch.setattr(evidence,'write_prompt_only_views',packet)
    assert run.main(argv(tmp_path,'select'))==0
    assert not (tmp_path/'out/native').exists()


@pytest.mark.parametrize('action',['select','seal','plan','report'])
@pytest.mark.parametrize('flag',[['--allow-gpu-execution'],['--approved-code-sha256','x'],['--native-root','/tmp/native'],['--sealed-plan','/tmp/plan']])
def test_nonrun_actions_reject_execution_flags(tmp_path,action,flag):
    with pytest.raises(ValueError,match='Only run'):
        run.main([action,'--out',str(tmp_path/'out')]+flag)
    assert not (tmp_path/'out').exists()


def test_requires_retained_parent_context_for_plan(tmp_path):
    with pytest.raises(ValueError,match='PR10'):
        run.main(['plan','--baseline-root',str(tmp_path/'base'),'--out',str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('root',['baseline','diagnostic'])
def test_protects_original_roots_before_read(tmp_path,monkeypatch,root):
    monkeypatch.setattr(evidence,'load_evidence',lambda *a:pytest.fail('Must reject first'))
    a=argv(tmp_path);a[a.index('--out')+1]=str(tmp_path/root/'new')
    with pytest.raises(ValueError,match='separate'):run.main(a)


def test_public_output_rejected(tmp_path):
    with pytest.raises(ValueError,match='separate'):
        run.main(['select','--out',str(run.HERE/'private')])


def test_seal_refuses_changed_parent_selection(tmp_path,monkeypatch):
    mock_admission(monkeypatch)
    p=tmp_path/'parents.json';p.write_text('{}')
    a=argv(tmp_path,'seal')+['--parent-selection',str(p),'--construction','unused','--tokenizer','unused']
    with pytest.raises(ValueError,match='parent selection changed'):run.main(a)
    assert not (tmp_path/'out').exists()


def test_seal_cpu_only(tmp_path,monkeypatch):
    b,m=mock_admission(monkeypatch)
    from experiments.dflash_per_turn import framing
    monkeypatch.setattr(framing,'load_pinned_tokenizer',lambda p:'cpu-tokenizer')
    monkeypatch.setattr(evidence,'seal_children',lambda bundle,con,tok:m if tok=='cpu-tokenizer' else pytest.fail('wrong tokenizer'))
    p=tmp_path/'parents.json';p.write_text(json.dumps(b['parent_selection']))
    c=tmp_path/'construction.json';c.write_text('{}')
    assert run.main(argv(tmp_path,'seal')+['--parent-selection',str(p),'--construction',str(c),'--tokenizer','cpu'])==0
    assert json.loads((tmp_path/'out/REQUEST_MANIFEST.json').read_text())==m


@pytest.mark.parametrize('case',['dirty','code','unauthorized','plan','tokenizer'])
def test_execution_rejects_unapproved_inputs(tmp_path,monkeypatch,case):
    b,m=mock_admission(monkeypatch);a=run_args(tmp_path,b,m)
    if case=='dirty':monkeypatch.setattr(run,'_git_identity',lambda:dict(commit='reviewed',dirty=True))
    elif case=='code':a[a.index('--approved-code-sha256')+1]='different'
    elif case=='unauthorized':a.remove('--allow-gpu-execution')
    elif case=='plan':(tmp_path/'plan.json').write_text('{}')
    else:
        i=a.index('--tokenizer');del a[i:i+2]
    with pytest.raises(ValueError):run.main(a)
    assert not (tmp_path/'out').exists()


def test_reencode_mismatch_stops_before_execution(tmp_path,monkeypatch):
    b,m=mock_admission(monkeypatch)
    b['groups'][0]['children']=[dict(text='exact\n whitespace',input_ids=[7])]
    monkeypatch.setattr(evidence,'load_evidence',lambda *args:copy.deepcopy(b))
    from experiments.dflash_per_turn import framing
    monkeypatch.setattr(framing,'load_pinned_tokenizer',lambda p:None)
    monkeypatch.setattr(framing,'encode_history',lambda tok,rows:[8])
    with pytest.raises(ValueError,match='exact input-token'):
        run.main(run_args(tmp_path,b,m))
    assert not (tmp_path/'out').exists()


def test_storage_failure_durable_before_workers(tmp_path,monkeypatch):
    b,m=mock_admission(monkeypatch)
    from experiments.dflash_per_turn import framing
    monkeypatch.setattr(framing,'load_pinned_tokenizer',lambda p:None)
    def fail(cfg):
        assert cfg['storage_budget']['runtime']['serving_contract']==cfg['serving_contract']
        raise ValueError('no room')
    monkeypatch.setattr(native,'storage_preflight',fail)
    with pytest.raises(ValueError,match='no room'):run.main(run_args(tmp_path,b,m))
    status=json.loads((tmp_path/'out/STORAGE_PREFLIGHT.json').read_text())
    assert status['status']=='failed' and not status['gpu_started'] and status['original_evidence_preserved']
    assert not (tmp_path/'out/native').exists()
