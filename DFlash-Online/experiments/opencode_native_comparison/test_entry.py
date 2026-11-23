"""Lifecycle and packaging admission checks without importing a model runtime."""
from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import pytest

from online_feedback.storage import write_json
from . import entry, preflight, serving_gate


def test_close_all_cleans_each_owned_engine_even_after_a_failure(monkeypatch):
    engines=entry.Engines({},'fixture')
    first,second=object(),object()
    engines.owned={id(first):(first,None,None),id(second):(second,None,None)}
    engines.active=first
    closed=[]
    def close(backend,output=None):
        engines.owned.pop(id(backend))
        closed.append(backend)
        if backend is first:
            raise ValueError('receipt failure')
    monkeypatch.setattr(engines,'close',close)
    with pytest.raises(RuntimeError,match='receipt failure'):
        engines.close_all()
    assert closed==[first,second] and engines.active is None and not engines.owned


def test_cleanup_failure_cannot_leave_completed_phase_status(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'experiments.persistent_prefix_withholding.qualification_entry',
        SimpleNamespace(HostMemorySample=lambda:nullcontext(SimpleNamespace(maximum_bytes=0))))
    class BrokenCleanup:
        create=close=lambda *a,**k:None
        def __init__(self,*args):pass
        def close_all(self):raise RuntimeError('fixture teardown failure')
    monkeypatch.setattr(entry,'Engines',BrokenCleanup)
    monkeypatch.setattr(entry,'boundary',lambda p:{'status':'passed'})
    monkeypatch.setattr(serving_gate,'qualify_card',lambda *a,**k:{'status':'passed'})
    manifest=tmp_path/'inputs.json'
    write_json(manifest,{})
    plan=dict(root=str(tmp_path),manifest=str(manifest),execution_commit='fixture')
    with pytest.raises(RuntimeError,match='teardown failure'):
        entry.run(plan,'qualification_serving')
    saved=json.loads((tmp_path/'phases/qualification_serving_A/STATUS.json').read_text())
    assert saved['status']=='failed'
    assert 'teardown failure' in saved['cleanup_error']
    assert saved['result']['status']=='passed'
    assert saved['wall_s']>=0


def test_zstd_round_trip_is_checked_before_model_work():
    assert preflight.verify_compact_codec()['status']=='passed'


def test_missing_zstd_is_an_explicit_preflight_blocker(monkeypatch):
    monkeypatch.setattr(preflight.shutil,'which',lambda name:None)
    with pytest.raises(RuntimeError,match='requires the existing zstd'):
        preflight.verify_compact_codec()
