"""CPU-only admission checks: sealing cannot start work; execution is explicitly gated."""
import copy
import json
import pytest
from . import run
from .protocol import SERVING_CONTRACT


def bundle_fixture():
    pools=[dict(pool_id='p8_'+d,domain=d,status='prepared') for d in ('gsm8k','codealpaca','sharegpt')]
    for pool in pools:pool['pool_payload_sha256']=run.digest(pool)
    return dict(pools=pools,
        bindings=dict(pools=[{'sealed':True}],baseline_commit='baseline',diagnostic={'diagnostic_commit':'diagnostic'}),
        runtime_config=dict(learning_curve_source_sha256={'curve':'c'},diagnostic_source_sha256={'diagnostic':'d'}))


def mocked_admission(monkeypatch):
    from . import native
    from . import loss
    from experiments.dflash_baseline_learning_curve import native as curve
    from experiments.dflash_kl_native_diagnostic import native as diagnostic
    bundle=bundle_fixture()
    monkeypatch.setattr(run,'load_evidence',lambda *_:copy.deepcopy(bundle))
    monkeypatch.setattr(loss,'validate_pool_diagnostics',lambda pool:dict(status='passed'))
    monkeypatch.setattr(native,'source_pins',lambda:{'p8':'new'})
    monkeypatch.setattr(curve,'source_pins',lambda:{'curve':'c'})
    monkeypatch.setattr(diagnostic,'source_pins',lambda:{'diagnostic':'d'})
    monkeypatch.setattr(run,'_git_identity',lambda:dict(commit='reviewed',dirty=False))
    monkeypatch.setattr(native,'NativeBackend',lambda *_:pytest.fail('No model worker may start in this test'))
    return bundle,native


def args(tmp_path,action='plan'):
    return [action,'--baseline-root',str(tmp_path/'baseline'),'--diagnostic-root',str(tmp_path/'diagnostic'),
        '--out',str(tmp_path/'new')]


def approved_args(tmp_path,bundle):
    value=run.plan(bundle);value['plan_sha256']=run.digest(value)
    sealed=tmp_path/'sealed.json';sealed.write_text(json.dumps(value))
    return args(tmp_path,'run')+['--allow-gpu-execution','--approved-code-sha256',run.digest({'p8':'new'}),
        '--sealed-plan',str(sealed)],sealed


def test_plan_exact_budget_and_pool_seals():
    value=run.plan(bundle_fixture())
    assert (value['pools'],value['optimizer_steps'],value['training_presentations'],
        value['native_generations'],value['response_kl_evaluations'])==(3,24,48,48,48)
    assert value['pool_bindings']==[{'sealed':True}]
    assert value['new_arm']=='P8' and value['reused_arms']==['F','S8']
    assert value['checkpoints_to_export']==3
    assert value['optimizer_state_exports']==value['explicit_warmups']==value['intermediate_checkpoints']==0
    assert value['serving_contract']==SERVING_CONTRACT
    assert not value['gpu_started']


def test_plan_reads_only_and_writes_sealed_membership(tmp_path,monkeypatch):
    mocked_admission(monkeypatch)
    assert run.main(args(tmp_path))==0
    out=tmp_path/'new';value=json.loads((out/'PLAN.json').read_text())
    seal=value.pop('plan_sha256')
    assert run.digest(value)==seal
    assert not (out/'native').exists()
    assert json.loads((out/'SOURCE_MANIFEST.json').read_text())['git']['commit']=='reviewed'


def test_missing_original_diagnostic_binding_blocks_before_planning(tmp_path,monkeypatch):
    mocked_admission(monkeypatch)
    from . import loss
    def missing(pool):
        raise ValueError('missing original anchor RNG receipt')
    monkeypatch.setattr(loss,'validate_pool_diagnostics',missing)
    with pytest.raises(ValueError,match='original anchor RNG'):
        run.main(args(tmp_path))
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('flag',[
    ['--allow-gpu-execution'],['--approved-code-sha256','bad'],['--native-root','/tmp/p8-worker'],
    ['--sealed-plan','/tmp/p8-sealed']])
def test_plan_cannot_request_execution_options(tmp_path,monkeypatch,flag):
    mocked_admission(monkeypatch)
    with pytest.raises(ValueError,match='Plan cannot'):run.main(args(tmp_path)+flag)
    assert not (tmp_path/'new').exists()


def test_refuses_evidence_overlap_before_read(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'load_evidence',lambda *_:pytest.fail('Must reject before reading'))
    with pytest.raises(ValueError,match='separate from retained'):
        run.main(['plan','--baseline-root',str(tmp_path/'old'),'--diagnostic-root',str(tmp_path/'diag'),
            '--out',str(tmp_path/'old'/'new')])


def test_refuses_existing_and_public_output(tmp_path):
    with pytest.raises(ValueError,match='fresh private'):run.main(['plan','--out',str(tmp_path)])
    with pytest.raises(ValueError,match='public checkout'):
        run.main(['plan','--out',str(run.HERE/'private-results')])


def test_missing_diagnostic_root_never_executes(tmp_path):
    with pytest.raises(ValueError,match='Both retained'):
        run.main(['run','--baseline-root',str(tmp_path/'baseline'),'--out',str(tmp_path/'new')])


@pytest.mark.parametrize('flag',[['--allow-gpu-execution'],['--native-root','/tmp/native'],['--sealed-plan','/tmp/plan']])
def test_report_cannot_launch_or_select_models(tmp_path,flag):
    with pytest.raises(ValueError,match='Report only reads'):
        run.main(['report','--result',str(tmp_path/'result'),'--out',str(tmp_path/'new')]+flag)


@pytest.mark.parametrize('change',['dirty','hash','authorize','seal','source'])
def test_execution_rejects_changed_or_unapproved_inputs(tmp_path,monkeypatch,change):
    bundle,native=mocked_admission(monkeypatch)
    argv,sealed=approved_args(tmp_path,bundle)
    if change=='dirty':monkeypatch.setattr(run,'_git_identity',lambda:dict(commit='changed',dirty=True))
    elif change=='hash':argv[argv.index('--approved-code-sha256')+1]='bad'
    elif change=='authorize':argv.remove('--allow-gpu-execution')
    elif change=='seal':
        value=json.loads(sealed.read_text());value['pool_order'].reverse();sealed.write_text(json.dumps(value))
    else:
        bundle['runtime_config']['learning_curve_source_sha256']={'curve':'changed'}
        monkeypatch.setattr(run,'load_evidence',lambda *_:bundle)
    with pytest.raises(ValueError):run.main(argv)
    assert not (tmp_path/'new').exists()


def test_storage_failure_is_durable_and_precedes_workers(tmp_path,monkeypatch):
    bundle,native=mocked_admission(monkeypatch)
    argv,_=approved_args(tmp_path,bundle)
    def no_space(config):
        assert config['serving_contract']==SERVING_CONTRACT
        raise ValueError('insufficient capacity')
    monkeypatch.setattr(native,'storage_preflight',no_space)
    with pytest.raises(ValueError,match='insufficient capacity'):run.main(argv)
    record=json.loads((tmp_path/'new'/'STORAGE_PREFLIGHT.json').read_text())
    assert record['status']=='failed' and not record['gpu_started']
    assert record['original_evidence_preserved']
    assert not (tmp_path/'new'/'native').exists()


@pytest.mark.parametrize('destination',['baseline','new','parent','existing'])
def test_native_destination_cannot_overwrite_evidence(tmp_path,monkeypatch,destination):
    bundle,_=mocked_admission(monkeypatch)
    argv,_=approved_args(tmp_path,bundle)
    path=tmp_path/destination
    if destination=='parent':path=tmp_path
    if destination=='existing':path.mkdir()
    with pytest.raises(ValueError):run.main(argv+['--native-root',str(path)])
