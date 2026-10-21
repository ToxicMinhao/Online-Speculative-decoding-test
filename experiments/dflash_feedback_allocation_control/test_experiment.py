"""CPU controller tests with sealed pools and real P8/native pair validation.

Controller fixtures substitute only the separately tested full training-accounting
validator. They never substitute pool admission or paired measurement validation.
"""
import copy

import pytest

from . import evidence, experiment as ex
from .protocol import SERVING_CONTRACT
from .test_evidence import groups_fixture
from .test_report import pair


def pools_fixture():
    groups = groups_fixture(colliding_request_ids=True)
    for group in groups:
        for population, rows in group["reference_pairs"].items():
            for row in rows:
                for label in ("F", "S8"):
                    row[label]["kl"].update(anchor_rng_after_sha256="a" * 64,
                        role="feedback" if population == "source" else "evaluation")
    return evidence.build_pools(groups)


def training_receipt(pool):
    files = {"model.safetensors": {"bytes": 1, "sha256": ex.digest(pool["pool_id"])}}
    checkpoint = dict(checkpoint_id="P8", pool_id=pool["pool_id"], domain=pool["domain"],
        group_ids=pool["group_ids"], optimizer_steps=8, learning_rate=3e-4, files=files,
        checkpoint_identity_sha256=ex.digest(files))
    losses = {}
    for population, refs in pool["reference_pairs"].items():
        losses[population] = []
        for ref in refs:
            row = copy.deepcopy(ref["F"]["kl"])
            row.update({key: ref[key] for key in ("group_id", "request_id", "pool_id", "pool_request_id")})
            row.update(checkpoint_id="P8", checkpoint_identity_sha256=checkpoint["checkpoint_identity_sha256"],
                       role=ref["F"]["native"]["role"])
            losses[population].append(row)
    return dict(status="passed", checkpoint=checkpoint, losses=losses,
                accounting={"optimizer_steps_total": 8, "response_presentations_total": 16},
                costs={"teacher_preparation_s": 2., "training_s": 3., "export_s": 1.})


class FakeBackend:
    def __init__(self, *, outcome="null", failure=None, fail_at=1):
        self.outcome, self.failure, self.fail_at = outcome, failure, fail_at
        self.closed = False
        self.served = 0
        self.calls = []

    def start_pool(self, pool):
        self.pool = pool
        self.calls.append("start")
        self.receipt = training_receipt(pool)
        return {"status": "passed"}

    def train_pool(self):
        self.calls.append("train")
        if self.failure == "training_rpc":
            raise RuntimeError("training RPC receipt unavailable")
        if self.failure in ("training_partial", "optimizer_unknown", "kl_partial"):
            error = RuntimeError(self.failure)
            partial = copy.deepcopy(self.receipt)
            partial["status"] = "failed"
            if self.failure == "kl_partial":
                partial["losses"]["heldout"] = partial["losses"]["heldout"][:3]
                partial["failure_detail"] = dict(population="heldout", attempted_response_calls=4,
                    returned_response_rows=3, unknown_response_calls=1,
                    response_losses=partial["losses"]["heldout"])
                partial["loss_diagnostics"] = {"source": dict(population="source", attempted_response_calls=8,
                    returned_response_rows=8, unknown_response_calls=0)}
            else:
                partial["losses"] = {"source": [], "heldout": []}
                partial["accounting"].update(optimizer_steps_total=0 if self.failure == "optimizer_unknown" else 3,
                    response_presentations_total=0 if self.failure == "optimizer_unknown" else 6,
                    optimizer_steps_unknown=int(self.failure == "optimizer_unknown"),
                    attempted_response_calls=2 if self.failure == "optimizer_unknown" else 8,
                    completed_backward_presentations=2 if self.failure == "optimizer_unknown" else 7,
                    uncommitted_backward_presentations=2 if self.failure == "optimizer_unknown" else 1)
            error.diagnostic_partial = partial
            raise error
        if self.failure == "nonfinite_training":
            self.receipt["losses"]["source"][0]["loss"] = float("nan")
        return copy.deepcopy(self.receipt)

    def stop_training(self):
        self.calls.append("stop")
        return dict(status="passed", training_quiescent=self.failure != "quiescence")

    def serve_request(self, frozen, population):
        self.calls.append("serve")
        self.served += 1
        if self.failure == "native_rpc" and self.served == self.fail_at:
            raise RuntimeError("native RPC receipt unavailable")
        ref = next(row for row in self.pool["reference_pairs"][population]
                   if evidence.composite_identity(row) == evidence.composite_identity(frozen))
        row = frozen
        row.update(checkpoint_id="P8", checkpoint_identity_sha256=self.receipt["checkpoint"]["checkpoint_identity_sha256"],
            pool_id=self.pool["pool_id"], pool_request_id=ref["pool_request_id"],
            serving_contract=copy.deepcopy(SERVING_CONTRACT), feedback_capture_enabled=False)
        if self.outcome == "harmful":
            row.update(verification_calls=2, accepted_draft_tokens=0, committed_progress_per_call=1.,
                       emitted_tokens_per_verification_call=1.5)
            row["rounds"] = [dict(round_index=i, output_offset=i+1, accepted_prefix_length=0,
                accepted_draft_tokens=0, proposed_tokens=1, committed_tokens=1,
                committed_token_ids=[row["output_ids"][i+1]], first_rejection_offset=0,
                stop_clipped=False) for i in range(2)]
        if self.served == self.fail_at:
            if self.failure == "output": row["output_ids"][0] += 1
            if self.failure == "termination": row["finish_reason"] = {"type": "other_stop"}
            if self.failure == "counter": row["accepted_draft_tokens"] = 999
        return row

    def finish_pool(self):
        self.calls.append("cleanup")
        self.closed = True
        if self.failure == "cleanup":
            raise RuntimeError("cleanup failed")
        return dict(status="passed", startup_s=2., shutdown_s=1., activation_s=3., native_activation_s=1.)


@pytest.fixture
def controller(monkeypatch):
    def validate_training(receipt, pool):
        ex.require(receipt["status"] == "passed" and ex._finite(receipt), "Invalid synthetic training receipt")
        return {"status": "passed", "errors": []}
    monkeypatch.setattr(ex, "validate_training", validate_training)
    def run(*, outcome="null", failure=None, fail_at=1, fail_pool=0):
        pools = pools_fixture()
        before = copy.deepcopy(pools)
        backends, snapshots, events = [], [], []
        def factory(pool):
            backend = FakeBackend(outcome=outcome, failure=failure if len(backends) == fail_pool else None,
                                  fail_at=fail_at)
            backends.append(backend)
            return backend
        result = ex.run_campaign(pools, factory,
            snapshot=lambda value: snapshots.append(copy.deepcopy(value)),
            emit=lambda kind, value: events.append((kind, copy.deepcopy(value))))
        assert pools == before
        return result, backends, snapshots, events
    return run


@pytest.mark.parametrize("outcome", ["null", "harmful"])
def test_correct_null_or_harmful_first_pool_continues_all_three_with_fixed_budget(controller, outcome):
    result, backends, snapshots, events = controller(outcome=outcome)
    assert result["status"] == "complete" and result["unexecuted_pools"] == []
    assert (result["optimizer_steps"], result["training_presentations"], result["actual_native_generations"],
            result["actual_response_kl_evaluations"]) == (24, 48, 48, 48)
    assert result["explicit_warmups"] == result["unknown_native_work_receipts"] == 0
    assert result["first_pool_qualification"]["native_generations"] == 16
    assert result["first_pool_qualification"]["kl_evaluations"] == 16
    assert all(b.closed and b.calls[:3] == ["start", "train", "stop"] and b.served == 16 for b in backends)
    assert len([value for kind, value in events if kind == "training_raw"]) == 3
    assert len([value for kind, value in events if kind == "native_raw"]) == 48
    assert snapshots[-1] == result
    for pool in result["pools"]:
        for row in pool["source"] + pool["heldout"]:
            assert row["validation"]["status"] == "passed"
            assert row["P8"]["native"]["verification_calls"] == (2 if outcome == "harmful" else 1)


@pytest.mark.parametrize("failure", ["output", "termination", "counter"])
def test_invalid_native_receipt_is_preserved_before_gate_and_halts_remaining_pools(controller, failure):
    result, backends, snapshots, events = controller(failure=failure, fail_at=3)
    pool = result["pools"][0]
    assert result["status"] == "failed" and len(result["unexecuted_pools"]) == 2
    assert result["actual_native_generations"] == result["attempted_native_generations"] == 3
    assert result["unattempted_native_generations"] == 45 and result["unknown_native_work_receipts"] == 0
    assert len(pool["native_raw"]) == len(pool["source"]) == 3
    assert pool["source"][-1]["validation"]["status"] == "failed" and backends[0].closed
    assert any(len(s["pools"][0]["native_raw"]) == 3 and len(s["pools"][0]["source"]) == 2
               for s in snapshots if s["pools"])
    assert len([v for k, v in events if k == "native_raw"]) == 3
    assert "first_pool_qualification" not in result


def test_second_pool_failure_retains_first_qualification_and_completed_work(controller):
    result, backends, _, _ = controller(failure="counter", fail_pool=1)
    assert result["status"] == "failed" and len(result["pools"]) == 2
    assert result["pools"][0]["status"] == "complete"
    assert result["first_pool_qualification"]["status"] == "passed"
    assert result["actual_native_generations"] == 17 and result["optimizer_steps"] == 16
    assert result["unexecuted_pools"] == ["p8_sharegpt"] and all(b.closed for b in backends)


def test_quiescence_failure_prevents_any_native_work(controller):
    result, backends, _, _ = controller(failure="quiescence")
    assert result["status"] == "failed" and result["actual_native_generations"] == 0
    assert result["optimizer_steps"] == 8 and result["actual_response_kl_evaluations"] == 16
    assert backends[0].calls == ["start", "train", "stop", "cleanup"]


def test_missing_native_rpc_receipt_is_unknown_not_unattempted(controller):
    result, _, _, _ = controller(failure="native_rpc", fail_at=3)
    assert result["actual_native_generations"] == 2 and result["attempted_native_generations"] == 3
    assert result["unknown_native_work_receipts"] == 1 and result["unattempted_native_generations"] == 45
    assert len(result["pools"][0]["source"]) == 2


def test_missing_training_rpc_receipt_keeps_entire_worker_allocation_unknown(controller):
    result, backends, _, _ = controller(failure="training_rpc")
    assert result["unknown_training_work_receipts"] == result["unknown_response_kl_worker_allocations"] == 1
    assert result["optimizer_steps"] == result["actual_response_kl_evaluations"] == 0
    assert result["unattempted_optimizer_steps"] == 16 and result["unattempted_response_kl_evaluations"] == 32
    assert result["unattempted_native_generations"] == 48 and backends[0].closed


def test_known_partial_training_retains_completed_and_uncommitted_backward_work(controller):
    result, _, _, _ = controller(failure="training_partial")
    receipt = result["pools"][0]["training"]
    assert result["optimizer_steps"] == 3 and result["training_presentations"] == 6
    assert result["unattempted_optimizer_steps"] == 21 and result["unknown_training_work_receipts"] == 0
    assert result["pools"][0]["work_incomplete"] is True
    assert receipt["accounting"]["completed_backward_presentations"] == 7
    assert receipt["accounting"]["uncommitted_backward_presentations"] == 1


def test_uncertain_optimizer_step_is_not_mislabeled_unattempted(controller):
    result, _, _, _ = controller(failure="optimizer_unknown")
    assert result["optimizer_steps"] == 0 and result["unknown_optimizer_steps"] == 1
    assert result["unattempted_optimizer_steps"] == 23
    assert result["pools"][0]["training"]["accounting"]["completed_backward_presentations"] == 2


def test_partial_diagnostic_retains_known_rows_and_unknown_inflight_response(controller):
    result, _, _, _ = controller(failure="kl_partial")
    assert result["optimizer_steps"] == 8 and result["actual_response_kl_evaluations"] == 11
    assert result["unknown_response_kl_evaluations"] == 1
    assert result["unattempted_response_kl_evaluations"] == 36
    assert len(result["pools"][0]["training"]["losses"]["heldout"]) == 3
    assert result["actual_native_generations"] == 0


def test_nonfinite_training_receipt_is_preserved_and_stops_before_native(controller):
    result, backends, _, events = controller(failure="nonfinite_training")
    assert result["status"] == "failed" and result["actual_native_generations"] == 0
    raw = result["pools"][0]["training"]["losses"]["source"][0]["loss"]
    assert raw["nonfinite_float"] == "nan" and backends[0].closed
    assert events[0][0] == "training_raw"


def test_cleanup_failure_preserves_completed_measurements_but_blocks_next_pool(controller):
    result, _, _, _ = controller(failure="cleanup")
    assert result["status"] == "failed" and result["actual_native_generations"] == 16
    assert len(result["unexecuted_pools"]) == 2 and "cleanup_error" in result["pools"][0]
    assert "first_pool_qualification" not in result


def test_invalid_pool_is_rejected_before_backend_creation():
    pools = pools_fixture()
    pools[0]["support"].reverse()
    calls = []
    with pytest.raises(ValueError):
        ex.run_campaign(pools, lambda pool: calls.append(pool))
    assert calls == []


@pytest.mark.parametrize("population", ["source", "heldout"])
def test_actual_pair_validator_accepts_original_and_pool_ownership(population):
    assert ex.validate_pair(pair(population=population)) == {"status": "passed", "errors": []}


@pytest.mark.parametrize("change", ["native_pool", "native_group", "native_request", "original_s8_owner",
    "checkpoint", "pool_request", "capture", "concurrency", "cache", "tokenizer", "output", "termination",
    "diagnostic_role", "diagnostic_pool", "diagnostic_mode", "rng", "mask", "weights", "denominator", "nonfinite"])
def test_actual_pair_validator_rejects_identity_serving_or_diagnostic_changes(change):
    value = pair(population="heldout")
    native, loss = value["P8"]["native"], value["P8"]["kl"]
    if change == "native_pool": native["pool_id"] = "p8_other"
    if change == "native_group": native["group_id"] = "other"
    if change == "native_request": native["request_id"] = "other"
    if change == "original_s8_owner": value["S8"]["native"]["group_id"] = "other"
    if change == "checkpoint": value["checkpoint_bindings"]["P8"] = "e" * 64
    if change == "pool_request": value["pool_request_id"] = "local_0"
    if change == "capture": native["feedback_capture_enabled"] = True
    if change == "concurrency": native["serving_contract"]["concurrency"] = 2
    if change == "cache": native["serving_contract"]["cache_policy"] = "warm"
    if change == "tokenizer": native["serving_contract"]["tokenizer_binding"] = "other"
    if change == "output": native["output_ids"][0] += 1
    if change == "termination": native["finish_reason"] = {"type": "other"}
    if change == "diagnostic_role": loss["role"] = "feedback"
    if change == "diagnostic_pool": loss["pool_id"] = "p8_other"
    if change == "diagnostic_mode": loss["diagnostic_model_mode"] = "train"
    if change == "rng": loss["training_rng_preserved"] = False
    if change == "mask": loss["binary_loss_mask"][0][1] = 0
    if change == "weights": loss["position_weights"][0][1] = .5
    if change == "denominator": loss["normalization_denominator"] += 1
    if change == "nonfinite": loss["loss"] = float("nan")
    assert ex.validate_pair(value)["status"] == "failed"


@pytest.mark.parametrize("change", [None, "checkpoint_pool", "checkpoint_alias", "checkpoint_hash",
    "loss_count", "loss_role", "loss_owner", "anchor", "nonfinite"])
def test_training_receipt_wrapper_preserves_checkpoint_and_diagnostic_guards(monkeypatch, change):
    from . import training
    pool = pools_fixture()[0]
    receipt = training_receipt(pool)
    # The separate training suite validates actual step/RNG/exposure receipts.
    # Here isolate the controller's surrounding checkpoint and diagnostic gates.
    calls = []
    monkeypatch.setattr(training, "validate_training_accounting",
                        lambda account, owner: calls.append((account, owner)))
    if change == "checkpoint_pool": receipt["checkpoint"]["pool_id"] = "p8_other"
    if change == "checkpoint_alias": receipt["checkpoint"]["checkpoint_id"] = "S8"
    if change == "checkpoint_hash": receipt["checkpoint"]["checkpoint_identity_sha256"] = "c" * 64
    if change == "loss_count": receipt["losses"]["source"].pop()
    if change == "loss_role": receipt["losses"]["source"][0]["role"] = "evaluation"
    if change == "loss_owner": receipt["losses"]["heldout"][0]["group_id"] = "other"
    if change == "anchor": receipt["losses"]["heldout"][0]["anchor_seed"] += 1
    if change == "nonfinite": receipt["losses"]["source"][0]["loss"] = float("nan")
    if change is None:
        assert ex.validate_training(receipt, pool) == {"status": "passed", "errors": []}
        assert calls == [(receipt["accounting"], pool)]
    else:
        with pytest.raises(ValueError):
            ex.validate_training(receipt, pool)
