"""Launcher wiring and exact qualification accounting without subprocess models."""
import json
from types import SimpleNamespace

import pytest

from online_feedback.storage import write_json
from . import launch
from .contract import ARMS
from .qualification import CASES, qualification_slots


def qualification_receipts(root):
    exact=dict(qualification_slots()['totals'],generation=64,teacher_preparation=16)
    plan=dict(root=str(root),exact_qualification_work=exact,execution_commit='fixture')
    work={'qualification.'+key:value for key,value in exact.items()}
    write_json(root/'WORK.json',dict(pending={},failures=[],attempted=work,completed=work))
    for arm in ARMS:
        write_json(root/'qualification'/arm/'DETAILS.json',dict(status='passed',
            cases=[dict(case=case,status='completed',checks='passed') for case in CASES[arm]],
            core_adam=len(CASES[arm]),core_response_forward_backward_each=2*len(CASES[arm]),
            startup_response_forward_backward_each=2,origin_restored=True,
            exact_own_resume=True,exact_native_compatibility=True))
    write_json(root/'qualification/INITIAL_SERVING.json',dict(status='passed_initial_serving',
        executions=60,cross_card_parity=True,capture_on_off_parity=True))
    write_json(root/'phases/qualification_publication_A/STATUS.json',dict(status='completed',
        result=dict(status='passed',executions=4,exact_output_and_reload_counts=True,
                    no_additional_optimizer_steps=True)))
    return plan


def test_completion_requires_exact_prespecified_controls(tmp_path):
    plan=qualification_receipts(tmp_path)
    result=launch.complete_qualification(plan)
    assert result['work']==dict(generation=64,teacher_preparation=16,adam=17,
        response_forward=40,response_backward=40,training_initialization=3)
    path=tmp_path/'qualification/C_CODE58/DETAILS.json'
    detail=json.loads(path.read_text())
    detail['cases'].pop()
    write_json(path,detail)
    with pytest.raises(ValueError,match='Training qualification'):
        launch.complete_qualification(plan)


def test_completion_cannot_redefine_the_qualification_budget(tmp_path):
    plan=qualification_receipts(tmp_path)
    plan['exact_qualification_work']['adam']=16
    with pytest.raises(ValueError,match='qualification changed'):
        launch.complete_qualification(plan)


def test_started_child_is_owned_before_process_receipt_is_written(tmp_path,monkeypatch):
    (tmp_path/'logs').mkdir()
    (tmp_path/'processes').mkdir()
    process=SimpleNamespace(pid=123,poll=lambda:None)
    stopped=[]
    monkeypatch.setattr(launch.subprocess,'Popen',lambda *a,**k:process)
    monkeypatch.setattr(launch,'child_environment',lambda *a,**k:{'PATH':'/bin'})
    monkeypatch.setattr(launch,'stop_owned',lambda p:stopped.append(p))
    real_write=launch.write_json
    count=0
    def fail_receipt(path,value):
        nonlocal count
        count+=1
        if count==1:
            raise OSError('receipt write failed')
        real_write(path,value)
    monkeypatch.setattr(launch,'write_json',fail_receipt)
    plan=dict(root=str(tmp_path),training_python='/python',source_root=str(tmp_path),zstd='/bin/zstd')
    with pytest.raises(OSError,match='receipt write failed'):
        launch.children(tmp_path/'plan.json',plan,'qualification_training',('B',),arm='C_CODE58')
    assert stopped==[process]
