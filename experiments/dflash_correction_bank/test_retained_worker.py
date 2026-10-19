"""CPU guards for bounded immutable retained-correction replay workers."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from . import retained_hooks, retained_worker as worker
from .artifacts import binding, save_snapshot


def _case():
    return dict(case_id="wildchat_00:t1:m0", turn=1, module=0,
                input_ids=[1, 2], input_ids_sha256=worker.digest([1, 2]),
                output_ids=[3], output_ids_sha256=worker.digest([3]),
                projections={"sha256": "projection"}, after_snapshot={"sha256": "snapshot"})


def _serving(tmp_path):
    case = _case()
    service = worker.RetainedServingService.__new__(worker.RetainedServingService)
    service.cases = {case["case_id"]: case}
    service.claims, service.used = tmp_path, set()
    service.config = {"released": {"checkpoint_identity_sha256": "released"}}
    body = dict(case_id=case["case_id"], turn=1, input_ids=case["input_ids"].copy(),
                projections=case["projections"], capture=False, canonical=False,
                checkpoint_id="F", arm="F", state={"arm": "F", "state_sha256": "released"})
    return service, case, body


def test_each_case_arm_is_claimed_once_before_native_execution(tmp_path, monkeypatch):
    service, case, body = _serving(tmp_path)
    def generate(_, value):
        assert len(list(tmp_path.glob("*.json"))) in (1, 3)
        assert (case["case_id"], value["arm"]) in service.used
        return {"status": "passed", "verification_calls": 1}
    native = Mock(side_effect=generate)
    monkeypatch.setattr(worker.native.ServingService, "generate", lambda self, value: native(self, value))
    first = service.generate(body)
    assert first["retained_arm"] == "F"
    with pytest.raises(ValueError, match="only once"):
        service.generate(body)
    assert native.call_count == 1
    bank = {**body, "arm": "BANK", "state": {"arm": "BANK", "snapshot": case["after_snapshot"],
                                               "state_sha256": "snapshot"}}
    result = service.generate(bank)
    assert result["routing_bypassed"] and native.call_count == 2
    assert len(list(tmp_path.glob("*.json"))) == 4


@pytest.mark.parametrize("change,reason", [
    ({"input_ids": [1, 99]}, "exact source-response input"),
    ({"turn": 2}, "exact source-response input"),
    ({"capture": True}, "without capture"),
    ({"canonical": True}, "without capture"),
    ({"checkpoint_id": "D1"}, "without capture"),
    ({"arm": "OSD"}, "Only frozen"),
    ({"projections": {"sha256": "new"}}, "projection artifact changed"),
    ({"state": {"arm": "F", "state_sha256": "other"}}, "released checkpoint"),
])
def test_wrong_input_capture_or_weights_never_reach_native(tmp_path, monkeypatch, change, reason):
    service, _, body = _serving(tmp_path)
    native = Mock(side_effect=AssertionError("Invalid generation reached native execution"))
    monkeypatch.setattr(worker.native.ServingService, "generate", native)
    with pytest.raises(ValueError, match=reason):
        service.generate({**body, **change})
    assert not service.used and not list(tmp_path.iterdir())
    native.assert_not_called()


def test_failed_native_generation_cannot_be_retried(tmp_path, monkeypatch):
    service, case, body = _serving(tmp_path)
    native = Mock(side_effect=RuntimeError("simulated native error"))
    monkeypatch.setattr(worker.native.ServingService, "generate", native)
    with pytest.raises(RuntimeError, match="native error"):
        service.generate(body)
    with pytest.raises(ValueError, match="only once"):
        service.generate(body)
    assert native.call_count == 1 and len(list(tmp_path.glob("*.json"))) == 1
    # Even a fresh in-memory object cannot overwrite the durable prior claim.
    restarted, _, _ = _serving(tmp_path)
    with pytest.raises(FileExistsError):
        restarted.generate(body)
    assert native.call_count == 1


@pytest.mark.parametrize("kind,action", [("sd", "train"), ("sd", "publish"),
                                        ("sd", "bank_feedback"), ("sd", "bank_prepare"),
                                        ("train", "train"), ("train", "generate"),
                                        ("train", "bank_feedback"), ("train", "publish")])
def test_rpc_forbids_training_publication_and_precision_worker_generation(tmp_path, monkeypatch, kind, action):
    import http.server
    config = dict(kind=kind, worker_out=str(tmp_path), workspace_root=str(tmp_path),
                  rpc_token="unit-test-only", port=12345)
    monkeypatch.setattr(worker.base, "_worker_config", lambda: config)
    monkeypatch.setattr(worker.native, "check_sources", lambda _: None)
    monkeypatch.setattr(worker.native, "check_numerical_environment", lambda: {})
    monkeypatch.setattr(worker.base, "check_dependencies", lambda _: None)
    monkeypatch.setattr(worker.base, "check_environment", lambda _: None)
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unit-test-no-device")
    service = SimpleNamespace(close=Mock())
    monkeypatch.setattr(worker, "RetainedServingService", lambda _: service)
    monkeypatch.setattr(worker, "RetainedPrecisionService", lambda _: service)
    observed = {}
    class Server:
        def __init__(self, address, handler):
            self.handler = handler
        def serve_forever(self, **kwargs):
            request = self.handler.__new__(self.handler)
            request.headers = {"X-Two-Update-Token": config["rpc_token"], "Content-Length": "2"}
            request.rfile, request.wfile = io.BytesIO(b"{}"), io.BytesIO()
            request.path = "/" + action
            request.send_response = lambda status: observed.update(http_status=status)
            request.send_header = lambda *_: None
            request.end_headers = lambda: None
            request.do_POST()
            observed["response"] = json.loads(request.wfile.getvalue())
        def server_close(self):
            observed["closed"] = True
    monkeypatch.setattr(http.server, "HTTPServer", Server)
    worker.worker_main()
    assert observed["http_status"] == 500 and observed["response"]["status"] == "failed"
    assert "outside the retained diagnostic" in observed["response"]["error"]
    service.close.assert_called_once()
    assert observed["closed"]


def _snapshot(tmp_path, module=2):
    matrices = np.zeros((4, 64, 64), dtype=np.float64)
    matrices[module] = np.eye(64)*.125
    matrices[(module+1) % 4] = np.eye(64)*9  # Would expose accidental fusion.
    state = save_snapshot({"matrices": matrices}, tmp_path/"after_turn.json")
    return dict(after_snapshot=state, module=module,
                matrix_sha256=hashlib.sha256(matrices[module].tobytes()).hexdigest()), matrices


def test_selected_matrix_is_exact_saved_singleton_copy(tmp_path):
    case, matrices = _snapshot(tmp_path)
    selected = retained_hooks.selected_matrix(case)
    np.testing.assert_array_equal(selected, matrices[case["module"]])
    selected[:] = 0
    np.testing.assert_array_equal(retained_hooks.selected_matrix(case), matrices[case["module"]])
    wrong = {**case, "matrix_sha256": "0"*64}
    with pytest.raises(RuntimeError, match="Selected matrix changed"):
        retained_hooks.selected_matrix(wrong)


def test_installed_fixed_head_applies_singleton_without_router(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    case, matrices = _snapshot(tmp_path)
    projection_path = tmp_path/"projections.npz"
    np.savez(projection_path, U=np.eye(64), V_in=np.eye(64))
    case["projections"] = binding(projection_path)
    class Draft:
        def forward_batch_generation(self, batch):
            return batch
        def _greedy_sample_from_vocab_parallel_head(self, *, hidden_states, lm_head, chunk_size=256):
            self.seen = hidden_states
            return hidden_states
    original_forward = Draft.forward_batch_generation
    class Scheduler:
        pass
    names = ("sglang", "sglang.srt", "sglang.srt.speculative", "sglang.srt.managers",
             "sglang.srt.speculative.dflash_worker", "sglang.srt.managers.scheduler")
    modules = {name: ModuleType(name) for name in names}
    modules[names[-2]].DFlashWorker = Draft
    modules[names[-1]].Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, names[0], modules[names[0]])
    for name in names[1:]:
        monkeypatch.setitem(sys.modules, name, modules[name])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    class Trace:
        def __init__(self, ids):
            self.input_ids = ids
        def finish(self, output_ids, meta):
            return {}
    entry = SimpleNamespace(RequestTrace=Trace, _active_trace=None,
                            inherited=SimpleNamespace(strict_idle=lambda _: None))
    receipt = retained_hooks.install(entry, {"retained_cases": [case], "worker_out": str(tmp_path)})
    assert receipt["routing_bypassed"] and not receipt["descriptor_hook_installed"]
    assert Draft.forward_batch_generation is original_forward
    activation = {"state": {"state_sha256": case["after_snapshot"]["sha256"],
                            "snapshot": case["after_snapshot"]}, "projections": case["projections"]}
    (tmp_path/"BANK_ACTIVATE.json").write_text(json.dumps(activation))
    Scheduler().correction_bank_activate()
    hidden = torch.ones((2, 64), dtype=torch.bfloat16)
    draft = Draft()
    for arm in ("F", "BANK", "F"):
        pending = dict(arm=arm, input_ids_sha256=retained_hooks.digest([1, 2]),
                       state_sha256=case["after_snapshot"]["sha256"])
        (tmp_path/"BANK_PENDING.json").write_text(json.dumps(pending))
        entry._active_trace = trace = entry.RequestTrace([1, 2])
        actual = draft._greedy_sample_from_vocab_parallel_head(hidden_states=hidden, lm_head=None)
        finished = trace.finish([3], {})
        if arm == "F":
            assert actual is hidden and finished["correction_bank_read"] is None
        else:
            expected = retained_hooks.corrected_features(hidden,
                torch.tensor(matrices[case["module"]], dtype=torch.float32),
                torch.eye(64), torch.eye(64))
            assert torch.equal(actual, expected)
            read = finished["correction_bank_read"]
            assert read["coefficients"] == [0., 0., 1., 0.]
            assert read["corrected_head_calls"] == 1 and read["fixed_for_entire_request"]
