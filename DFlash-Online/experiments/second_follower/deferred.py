"""One disposable native twin of the first active main-training window.

The candidate window remains part of the prescribed 128 updates. A native twin
costs two additional forwards/backwards and one temporary Adam step. Main work
wrappers must exclude calls while ``trainer._second_follower_diagnostic`` is true.
Full weights/moments are copied only once, after positive exposure is known and
before that candidate optimizer step; preceding forwards cannot change them.
"""

from copy import deepcopy
from pathlib import Path
import time
import traceback

import torch

from online_feedback.kplus import KPlusRedistribution
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import write_json


LIMITS = dict(response_forwards=2, response_backwards=2, optimizer_steps=1)
SAVED_FIELDS = (
    "response_records", "window_records", "work", "_anchor_observations",
    "_presentation_schedule", "_active_event", "events_root",
)


def _helpers():
    from experiments.functional_step_geometry.qualify import (
        scientific_state, runtime_state, restore_runtime,
    )
    from experiments.prefix_guard.qualify import persist_comparison
    return scientific_state, runtime_state, restore_runtime, persist_comparison


def _coordinates(actual, reference):
    rows = []
    if actual.keys() != reference.keys():
        raise ValueError("Native twin parameter membership changed")
    for name, a in actual.items():
        b = reference[name]
        if a is None or b is None:
            if a is not b:
                raise ValueError("Native twin gradient presence changed")
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Native twin tensor shape/dtype changed")
        delta = a.float() - b.float()
        rows.append(dict(name=name, dtype=str(a.dtype), coordinates=a.numel(),
                         changed=int(torch.count_nonzero(a != b)),
                         max_abs=float(delta.abs().max()),
                         l2=float(torch.linalg.vector_norm(delta))))
    return dict(changed=sum(r["changed"] for r in rows),
                coordinates=sum(r["coordinates"] for r in rows), parameters=rows)


class DeferredFirstActiveCheck:
    """Install before the event, after conditional main-work wrappers if present."""

    def __init__(self, trainer, output):
        if trainer.core.accumulation_steps != 2:
            raise ValueError("Deferred qualification requires two-response windows")
        if trainer.core.accumulation_remainder:
            raise ValueError("Install deferred qualification at a clean boundary")
        reducer = getattr(trainer.bundle.model, "first_error_redistribution", None)
        if getattr(reducer, "contract", {}).get("name") != "kplus_second_follower_v1" or reducer.disabled:
            raise ValueError("Deferred qualification requires the enabled fixed reducer")
        if getattr(trainer, "_second_follower_diagnostic", False):
            raise ValueError("Cannot nest deferred qualification")
        self.trainer, self.output = trainer, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.done = self.closed = False
        self.window = self.pre = self.candidate_gradients = self.native_gradients = None
        self.original_train_step = trainer.core.train_step
        self.original_step = trainer.backend.optimizer.step
        self.receipt = dict(status="waiting_for_first_active_window", candidate_window_is_main=True,
                            extra_work_limits=LIMITS, windows_examined=0)
        trainer._second_follower_diagnostic = False
        trainer.core.train_step = self.train_step
        trainer.backend.optimizer.step = self.optimizer_step
        self._write()

    @property
    def diagnostic(self):
        return self.trainer._second_follower_diagnostic

    def _write(self):
        write_json(self.output / "RESULT.json", self.receipt)

    def optimizer_step(self, *args, **kwargs):
        trainer = self.trainer
        if self.diagnostic:
            self.native_gradients = {
                name: None if p.grad is None else p.grad.detach().cpu().clone()
                for name, p in trainer.bundle.draft_model.named_parameters()
            }
        elif not self.done:
            if self.window is None or len(self.window["batches"]) != 2:
                raise ValueError("Deferred check lost its two-response boundary")
            rows = trainer.response_records[-2:]
            if [r["record_id"] for r in rows] != self.window["ids"]:
                raise ValueError("Deferred check response observations changed order")
            transferred = sum(float(r["sum_metrics"]["second_follower_transferred_weight"]) for r in rows)
            if transferred > 0:
                scientific_state, _, _, _ = _helpers()
                # This is still before the actual optimizer update. Reconstruct
                # the clean prewindow state using its saved RNG/runtime/counters.
                self.pre = scientific_state(trainer)
                self.candidate_gradients = self.pre["gradients"]
                self.pre.update(rng=self.window["rng"], runtime=self.window["runtime"],
                                global_step=self.window["global_step"],
                                presentations=self.window["presentations"],
                                micro=self.window["micro"], accumulation=0,
                                gradients={name: None for name in self.candidate_gradients})
                self.receipt.update(status="candidate_active_window", transferred_weight=transferred,
                                    global_step_before=self.window["global_step"],
                                    record_ids=self.window["ids"],
                                    candidate_observations=deepcopy(rows))
                self._write()
        return self.original_step(*args, **kwargs)

    def train_step(self, batch, ctx=None):
        trainer = self.trainer
        if self.done or self.diagnostic:
            return self.original_train_step(batch, ctx)
        if trainer.core.accumulation_remainder == 0:
            if self.window is not None:
                raise ValueError("Deferred window was not cleared")
            _, runtime_state, _, _ = _helpers()
            self.window = dict(rng=capture_rng(), runtime=runtime_state(trainer.bundle.model),
                               global_step=trainer.global_step, presentations=trainer.presentations,
                               micro=trainer.core._micro, batches=[], contexts=[], ids=[])
        if self.window is None:
            raise ValueError("Deferred check entered midway through a window")
        # Native strategy only reads the TrainBatch's original CPU tensors. Keep
        # these two references alive until the boundary; no feature reload/draw.
        self.window["batches"].append(batch)
        self.window["contexts"].append(ctx)
        self.window["ids"].append(str(batch.sample_ids[0]))
        result = self.original_train_step(batch, ctx)
        if result.optimizer_stepped:
            self.receipt["windows_examined"] += 1
            if self.pre is not None:
                self._native_twin()
                self.done = True
            self.window = None
            self._write()
        # Outer event code has not yet advanced the second response's counters
        # or schedule. Return this exact object after restoring that exact state.
        return result

    def _restore_scientific(self, state):
        trainer = self.trainer
        _, _, restore_runtime, _ = _helpers()
        trainer.backend.optimizer.zero_grad()
        trainer.bundle.draft_model.load_state_dict(state["weights"], strict=True)
        trainer.backend.optimizer.load_state_dict(state["optimizer"])
        trainer.global_step, trainer.presentations = state["global_step"], state["presentations"]
        trainer.core._micro = state["micro"]
        trainer.core._ratio_totals = {}
        trainer.core._sum_totals = {}
        for name, parameter in trainer.bundle.draft_model.named_parameters():
            gradient = state["gradients"][name]
            parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
        restore_runtime(trainer.bundle.model, state["runtime"])
        restore_rng(state["rng"])

    def _native_twin(self):
        from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
        scientific_state, _, _, persist_comparison = _helpers()
        trainer = self.trainer
        post = scientific_state(trainer)
        metadata = {name: getattr(trainer, name) for name in SAVED_FIELDS}
        ratio, sums = trainer.core._ratio_totals, trainer.core._sum_totals
        reducer = trainer.bundle.model.first_error_redistribution
        costs = trainer.bundle.model.costs
        wrappers = [(trainer.core.strategy, "forward_loss"),
                    (trainer.backend, "backward"),
                    (trainer.backend.optimizer.optimizer, "step")]
        originals = [(owner, name, getattr(owner, name)) for owner, name in wrappers]
        ledger = WorkLedger(self.output / "WORK.jsonl", LIMITS)
        failure = None
        started = time.perf_counter()
        trainer._second_follower_diagnostic = True
        try:
            self._restore_scientific(self.pre)
            trainer.bundle.model.first_error_redistribution = KPlusRedistribution()
            # Preserve any required timer keys while isolating disposable writes.
            trainer.bundle.model.costs = deepcopy(costs)
            trainer.response_records, trainer.window_records, trainer.work = [], [], {}
            trainer._anchor_observations = []
            trainer._presentation_schedule = None
            trainer._active_event = "native_twin"
            trainer.events_root = self.output / "observations"
            (trainer.events_root / trainer._active_event).mkdir(parents=True)
            for (owner, name), kind in zip(wrappers, LIMITS):
                ledger.wrap(owner, name, kind)
            for batch, ctx in zip(self.window["batches"], self.window["contexts"]):
                trainer._anchor_observations.clear()
                result = self.original_train_step(batch, ctx)
                result.materialize_metrics()
                trainer.presentations += 1
                trainer.global_step += int(result.optimizer_stepped)
            if self.native_gradients is None:
                raise ValueError("Native twin did not reach its pre-Adam capture")
            candidate_rows = self.receipt["candidate_observations"]
            native_rows = deepcopy(trainer.response_records)
            self.receipt["native_observations"] = native_rows
            self.receipt["same_inputs_rng"] = persist_comparison(
                self.output / "SAME_INPUTS_RNG.json",
                dict(ids=[r["record_id"] for r in native_rows],
                     anchors=[r["sampled_anchors"] for r in native_rows], rng=capture_rng()),
                dict(ids=[r["record_id"] for r in candidate_rows],
                     anchors=[r["sampled_anchors"] for r in candidate_rows], rng=post["rng"]),
            )
            self.receipt["gradient_changes"] = _coordinates(self.candidate_gradients, self.native_gradients)
            self.receipt["weight_changes"] = _coordinates(
                post["weights"], {name: value.detach().cpu() for name, value in trainer.bundle.draft_model.state_dict().items()})
            self.receipt["native_physical_work"] = dict(trainer.work)
            require_complete_work(read_work(ledger.path, LIMITS), LIMITS)
            self.receipt["status"] = "completed"
        except BaseException as error:
            failure = error
            self.receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            # Restore even when the disposable native operation failed. Keep the
            # original failure if a cleanup action also fails; both are recorded.
            def restore():
                for owner, name, action in originals:
                    setattr(owner, name, action)
                trainer.bundle.model.first_error_redistribution = reducer
                trainer.bundle.model.costs = costs
                self._restore_scientific(post)
                for name, value in metadata.items():
                    setattr(trainer, name, value)
                trainer.core._ratio_totals, trainer.core._sum_totals = ratio, sums
                self.receipt["exact_candidate_restoration"] = persist_comparison(
                    self.output / "RESTORED_CANDIDATE.json", scientific_state(trainer), post)
            try:
                restore()
            except BaseException as cleanup_error:
                self.receipt["restoration_error"] = repr(cleanup_error)
                if failure is None:
                    self.receipt["status"] = "failed_restoration"
                    raise
                failure.add_note("Candidate restoration also failed: " + repr(cleanup_error))
            finally:
                trainer._second_follower_diagnostic = False
                self.receipt.update(extra_work=read_work(ledger.path, LIMITS),
                                    diagnostic_wall_s=time.perf_counter() - started,
                                    counter_scope="candidate post-step before outer second-response commit")
                self._write()
                self.pre = self.candidate_gradients = self.native_gradients = None

    def close(self):
        if self.closed:
            return self.receipt
        self.trainer.core.train_step = self.original_train_step
        self.trainer.backend.optimizer.step = self.original_step
        self.closed = True
        if not self.done and self.receipt["status"] == "waiting_for_first_active_window":
            self.receipt["status"] = "no_active_main_window"
            self.receipt["event_completion_not_inferred"] = True
            self.receipt["partial_window_pending"] = self.window is not None
            self.receipt["extra_work"] = dict(attempts={k: 0 for k in LIMITS},
                                               completed={k: 0 for k in LIMITS}, unknown={k: 0 for k in LIMITS})
        self.window = None
        self._write()
        return self.receipt


def install_deferred_check(trainer, output):
    """Caller enables this only when early qualification found no active pair."""
    return DeferredFirstActiveCheck(trainer, output)
