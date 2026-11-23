"""Native timing field and remaining-work forecast contracts."""
import json

import pytest

from online_feedback.storage import write_json
from . import forecast as F
from .contract import ARMS
from .qualification import CASES


def write_measurements(root):
    for card in ('A','B'):
        folder=root/'precheck'/card
        work={}
        for method in ('AR','F'):
            record=folder/(method+'.json')
            write_json(record,dict(response_ids=[1,2],finish_reason='eos',
                metadata=dict(request_latency_s=2. if method=='AR' else 1.)))
            for i in range(512):
                work[f'{method}_{i}']=dict(status='valid',method=method,record_path=str(record))
            write_json(folder/method/'LIFECYCLE.json',dict(status='loaded',loading_s=3.))
        write_json(folder/'WORK.json',work)
    write_json(root/'qualification/A/FEATURES.json',{str(i):dict(receipt=dict(status='prepared',
        teacher_preparation_s=.2,feature_persistence_s=.1)) for i in range(16)})
    for arm in ARMS:
        write_json(root/'qualification'/arm/'DETAILS.json',dict(status='passed',cases=[
            dict(case=case,status='completed',checks='passed',observer_adjusted_training_s=4.)
            for case in CASES[arm]]))


def test_forecast_consumes_actual_native_timer_names_and_full_controls(tmp_path):
    write_measurements(tmp_path)
    measured=F.measurements(dict(root=str(tmp_path)))
    assert {arm:len(rows) for arm,rows in measured['native'].items()}=={'AR':1024,'F':1024}
    assert measured['loading']==3.
    assert measured['teacher']==pytest.approx(.3)
    assert measured['training']==dict.fromkeys(ARMS,4.)


def test_failed_admission_preserves_complete_forecast(tmp_path,monkeypatch):
    write_measurements(tmp_path)
    def blocked(*args):raise RuntimeError('fixed resource cap')
    monkeypatch.setattr(F,'forecast',blocked)
    with pytest.raises(RuntimeError,match='fixed resource cap'):
        F.remaining(dict(root=str(tmp_path)))
    report=json.loads((tmp_path/'forecasts/AFTER_0_ARMS.json').read_text())
    assert report['status']=='blocked'
    assert len(report['per_arm'])==3
    assert report['per_arm']['O_CODE58']['components']['native_training_s']==7424*4.
    assert report['remaining_card_s']>report['remaining_elapsed_s']


def test_forecast_rejects_nonfinite_measurements_and_wrong_arm_order(tmp_path):
    with pytest.raises(ValueError,match='Invalid measured'):
        F.measured_seconds(float('nan'),'test')
    with pytest.raises(ValueError,match='fixed-order prefix'):
        F.remaining(dict(root=str(tmp_path)),completed_arms=('C_CODE58',))


def test_safe_boundary_counts_remaining_work_and_preserves_a_resource_stop(tmp_path,monkeypatch):
    write_json(tmp_path/'forecasts/AFTER_1_ARMS.json',dict(
        per_arm={arm:dict(raw_s=36000) for arm in ARMS[1:]},
        final_evaluation_s=1000,audit_packaging_s=200))
    seen=[]
    def admit(plan,elapsed,cards):
        seen.append((elapsed,cards))
        return dict(status='passed')
    monkeypatch.setattr(F,'forecast',admit)
    result=F.at_training_boundary(dict(root=str(tmp_path)),ARMS[1],8,5000)
    full=5000*58/8
    remaining=full-5000+full
    assert seen==[(remaining+1200,2*remaining+1000)]
    assert result['status']=='admitted'
    def blocked(*args):raise RuntimeError('time cap')
    monkeypatch.setattr(F,'forecast',blocked)
    with pytest.raises(RuntimeError,match='time cap'):
        F.at_training_boundary(dict(root=str(tmp_path)),ARMS[1],16,16000)
    saved=json.loads((tmp_path/'campaign_001/arms'/ARMS[1]/'forecasts/event_016.json').read_text())
    assert saved['status']=='blocked'
