import copy
import json
from pathlib import Path
import pytest
from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from . import launch, protected, preflight, startup_retry
from .contract import digest
from .runtime import LIMITS


def fixture(tmp_path):
    root=tmp_path/'run';root.mkdir()
    source=tmp_path/'old_source';source.mkdir()
    (source/'code.py').write_text('old')
    original_source=root/'SOURCE.json'
    write_json(original_source,dict(commit='a'*40,dirty=False,files={'code.py':identity(source/'code.py')}))
    plan=dict(root=str(root),source_root=str(source),source_manifest=str(original_source),
        execution_commit='a'*40,execution_source=dict(commit='a'*40,dirty=False),
        allocation_roots=[str(root),str(source)],compiler_baseline={'original':17},
        serving={'unchanged':True},limits={'unchanged':True},input_identities={'fixed':'receipt'})
    plan['binding']=digest(plan)
    path=root/'sealed/PLAN.json';write_json(path,plan)
    work=dict(limits=LIMITS,next_operation=0,attempted={k:0 for k in LIMITS},
              completed={k:0 for k in LIMITS},pending={},failures=[])
    clock=dict(first_model_load_epoch=1234.,completed_card_s=56.0318,active={},finished=[{'cost':56.0318}])
    write_json(root/'WORK.json',work);write_json(root/'CLOCK.json',clock)
    write_json(root/'LAUNCH.json',dict(status='failed',completed=[],preflight=dict(status='passed')))
    write_json(root/'PROTECTED_DEPENDENCIES.json',{'old':'immutable'})
    write_json(root/'qualification/A/failure.json',{'engine':'illegal access'})
    auth=root/'AUTHORIZATION_RETRY.json'
    write_json(auth,dict(status='authorized',scope=startup_retry.SCOPE))
    new=tmp_path/'new_source';new.mkdir();(new/'code.py').write_text('new')
    manifest=root/'SOURCE_RETRY.json'
    write_json(manifest,dict(commit='b'*40,dirty=False,files={'code.py':identity(new/'code.py')}))
    return root,path,new,manifest,auth,plan,work,clock


def test_archives_failure_and_preserves_clock_inputs_and_baseline(tmp_path,monkeypatch):
    root,path,new,manifest,auth,old,work,clock=fixture(tmp_path)
    monkeypatch.setattr(protected,'check',lambda *a,**k: {'status':'passed'})
    before={p:identity(p) for p in (path,root/'WORK.json',root/'CLOCK.json',root/'SOURCE.json')}
    result=startup_retry.prepare(path,new,manifest,auth)
    plan=json.loads(Path(result['plan']).read_text())
    receipt=startup_retry.validate_ready(plan,result['receipt'])
    for p,pin in before.items():assert identity(p)==pin
    for key in old.keys()-{'source_root','source_manifest','execution_commit','execution_source','allocation_roots','binding'}:
        assert plan[key]==old[key]
    assert plan['allocation_roots'][:2]==old['allocation_roots']
    assert receipt['original_clock']==clock and receipt['original_work']==work
    assert not (root/'qualification').exists()
    assert not (root/'PROTECTED_DEPENDENCIES.json').exists()
    assert (root/'attempts/failed_startup_001/qualification/A/failure.json').is_file()
    assert startup_retry.read(root/'SOURCE.json')['commit']=='a'*40
    with pytest.raises(FileExistsError):startup_retry.prepare(path,new,manifest,auth)
    write_json(root/'CLOCK.json',{**clock,'completed_card_s':0})
    with pytest.raises(ValueError,match='clock or work changed'):
        startup_retry.validate_ready(plan,result['receipt'])


@pytest.mark.parametrize('change',['attempt','pending','failure','active','unauthorized','modified_plan'])
def test_rejects_ineligible_retry_before_mutation(tmp_path,monkeypatch,change):
    root,path,new,manifest,auth,plan,work,clock=fixture(tmp_path)
    monkeypatch.setattr(protected,'check',lambda *a,**k: pytest.fail('must reject before protection/mutation'))
    if change=='attempt':work['attempted']['qualification.generation']=1;write_json(root/'WORK.json',work)
    if change=='pending':work['pending']['1']={};write_json(root/'WORK.json',work)
    if change=='failure':work['failures'].append({'unknown':True});write_json(root/'WORK.json',work)
    if change=='active':clock['active']['worker']={};write_json(root/'CLOCK.json',clock)
    if change=='unauthorized':write_json(auth,dict(status='pending',scope=startup_retry.SCOPE))
    if change=='modified_plan':plan['serving']['unchanged']=False;write_json(path,plan)
    before={str(p.relative_to(root)):identity(p) for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):startup_retry.prepare(path,new,manifest,auth)
    assert before=={str(p.relative_to(root)):identity(p) for p in root.rglob('*') if p.is_file()}


def test_retry_launcher_does_not_initialize_or_reset_clock(tmp_path,monkeypatch):
    root,path,new,manifest,auth,old,work,clock=fixture(tmp_path)
    monkeypatch.setattr(protected,'check',lambda *a,**k: {'status':'passed'})
    result=startup_retry.prepare(path,new,manifest,auth)
    monkeypatch.setattr(preflight,'verify',lambda plan: {'status':'passed'})
    monkeypatch.setattr(launch,'check_compilers',lambda: {})
    monkeypatch.setattr(launch,'enforce_clock',lambda plan: clock)
    monkeypatch.setattr(launch,'initialize',lambda root: pytest.fail('must preserve the original clock/work'))
    def stop(*args,**kwargs):raise RuntimeError('fixture stops before native work')
    monkeypatch.setattr(launch,'children',stop)
    with pytest.raises(RuntimeError,match='fixture stops'):
        launch.run(result['plan'],result['receipt'])
    assert startup_retry.read(root/'CLOCK.json')==clock
    assert startup_retry.read(root/'WORK.json')==work
    assert startup_retry.read(root/'LAUNCH.json')['authorized_startup_retry']['original_clock']==clock
