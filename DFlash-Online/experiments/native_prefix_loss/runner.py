"""One native KPLUS6→7 event with sixteen sealed P/N/E/C/H comparisons."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import traceback

from experiments.threatened_repair_replay.runner import (
    export_weights, host_admission, optimizer_delta, scalar_update_rows, termination_cleanup,
)
from experiments.threatened_repair_replay.work import Ledger, TrainingInstrumentation, WorkLimit
from experiments.ok_continuation.resume import differences
from online_feedback.storage import save_torch, write_json
from online_feedback.rng import preserve_rng
from .qualification import qualify
from .state import capture_state, compare_final_native, restore_state, tensor_change, verify_state
from .training import make_trainer, observe_source_pair


WINDOWS = tuple(range(8, 129, 8))


def read(path):
    return json.loads(Path(path).read_text())


class WindowController:
    """All branches begin at P; only the authoritative N endpoint survives."""
    def __init__(self, plan, output, ledger, *, observer=observe_source_pair,
                 exporter=export_weights, admit=host_admission):
        self.plan, self.output, self.ledger = plan, Path(output), ledger
        self.observe, self.export, self.admit = observer, exporter, admit
        self.probe_exports, self.source_rows, self.source_work = [], [], []
        self.updates, self.parity, self.windows, self.witnesses = [], [], [], []
        self.snapshot_bytes = Path(plan["origin_state"]).stat().st_size

    def publish_progress(self):
        for name, values in (("PROBE_EXPORTS", self.probe_exports), ("SOURCE_OUTCOMES", self.source_rows),
                             ("SOURCE_WORK", self.source_work), ("UPDATE_RECEIPTS", self.updates),
                             ("STATE_PARITY", self.parity), ("WINDOWS", self.windows), ("WITNESSES", self.witnesses)):
            write_json(self.output / (name+".json"), values)
        rows = scalar_update_rows(self.updates)
        fields = ["window", "state"] + sorted({key for row in rows for key in row} - {"window", "state"})
        with (self.output / "UPDATE_RECEIPTS.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)

    def _witness(self, snapshot, window, reason):
        if len(self.witnesses) >= 2 or any(row["window"] == window for row in self.witnesses):
            return
        path = self.output / "witnesses" / f"P_{window:03d}_{reason}.pt"
        payload = {key: snapshot[key] for key in ("scientific", "core", "gradients", "bookkeeping",
                                                 "schedule_state", "costs", "role_observer", "role_installed")}
        payload["runtime"] = {key: snapshot["runtime"][key] for key in ("modes", "buffers", "nonpersistent")}
        payload.update(window=window, reason=reason,
            replay_note="Complete scientific state; opaque attention caches reconstructed from sealed inputs")
        with preserve_rng():
            save_torch(path, payload)
        self.witnesses.append(dict(window=window, reason=reason, path=str(path), bytes=path.stat().st_size))

    def _export(self, weights, window, state):
        self.ledger.check()
        receipt = self.export(weights, self.plan["origin_export"], self.output / "exports" / f"window_{window:03d}_{state}")
        row = dict(window=window, state=state, reused_from=None, **receipt)
        self.probe_exports.append(row)
        return row

    def _source(self, trainer, pair, pass_index, ordinal, window, state, snapshot):
        self.ledger.check()
        directory = self.output / f"window_{window:03d}"
        rows, work = self.observe(trainer, pair, pass_index, ordinal, window=window, state=state, ledger=self.ledger)
        write_json(directory / f"SOURCE_{state}.json", dict(rows=rows, work=work))
        self.parity.append(dict(window=window, state=state, phase="source_observation",
            **verify_state(trainer, snapshot, directory / f"{state}_SOURCE_PARITY.json")))
        self.source_rows.extend(rows)
        self.source_work.extend(dict(row, window=window, state=state) for row in work)

    def __call__(self, trainer, window, pair, pass_index, ordinal):
        self.ledger.check()
        directory = self.output / f"window_{window:03d}"
        directory.mkdir()
        record = dict(window=window, status="running", current_request_ids=[row["record_id"] for row in pair])
        self.windows.append(record)
        self.admit(2*self.snapshot_bytes)
        pre = capture_state(trainer)
        native, failure = None, None
        try:
            if window in (8, 128):
                self._witness(pre, window, "fixed_boundary")
            self._export(pre["scientific"]["draft_state_dict"], window, "P")
            self._source(trainer, pair, pass_index, ordinal, window, "P", pre)
            result = trainer.run_pair(pair, pass_index, ordinal, role="N", label="N", category="native", capture_gradients=True)
            native_gradient = result["preclip_gradients"]["native"]
            native = capture_state(trainer)
            self._export(native["scientific"]["draft_state_dict"], window, "N")
            self._source(trainer, pair, pass_index, ordinal, window, "N", native)
            self.updates.append(dict(result["update"], window=window, state="N",
                displacement=tensor_change(pre["scientific"]["draft_state_dict"], native["scientific"]["draft_state_dict"])))
            # Fixed branch order; no selection or source outcome changes it.
            for state in ("E", "C", "H"):
                restore_state(trainer, pre)
                self.parity.append(dict(window=window, state=state, phase="before_branch",
                    **verify_state(trainer, pre, directory / f"{state}_PRE.json")))
                if state == "H":
                    branch = trainer.run_history_step(native_gradient, window=window, label=state)
                else:
                    branch = trainer.run_pair(pair, pass_index, ordinal, role=state, label=state,
                        category="disposable", capture_gradients=True)
                    # Same prestate/current forward, same full displayed loss,
                    # masks, anchors and denominator; only current gradients differ.
                    from .qualification import forward_receipt
                    mismatch = differences(forward_receipt(branch), forward_receipt(result), "same_state_forward")
                    write_json(directory / f"{state}_FORWARD_PARITY.json", dict(
                        status="failed" if mismatch else "passed", mismatches=mismatch[:64]))
                    if mismatch:
                        raise ValueError("Role intervention changed forward objective or sampling")
                current = capture_state(trainer)
                self.updates.append(dict(branch["update"], window=window, state=state,
                    displacement=tensor_change(pre["scientific"]["draft_state_dict"], current["scientific"]["draft_state_dict"]),
                    difference_from_native=tensor_change(native["scientific"]["draft_state_dict"], current["scientific"]["draft_state_dict"]),
                    optimizer_difference=optimizer_delta(native["scientific"]["backend"], current["scientific"]["backend"])))
                if state != "H":
                    self.updates[-1]["gradient_vs_native"] = tensor_change(native_gradient, branch["preclip_gradients"]["native"])
                self._export(current["scientific"]["draft_state_dict"], window, state)
                self._source(trainer, pair, pass_index, ordinal, window, state, current)
                del current, branch
                restore_state(trainer, native)
                self.parity.append(dict(window=window, state=state, phase="restored_N",
                    **verify_state(trainer, native, directory / f"{state}_RESTORED_N.json")))
            record.update(status="completed", branches=["P", "N", "E", "C", "H"])
            result["preclip_gradients"].clear()
            return result
        except BaseException as error:
            failure = error
            record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            write_json(directory / "FAILED.json", record)
            self._witness(pre, window, "failure")
            raise
        finally:
            from experiments.functional_step_geometry.qualify import cleanup
            cleanup(lambda: restore_state(trainer, native if native is not None else pre), failure, record,
                    "restore authoritative native state")
            cleanup(self.publish_progress, failure, record, "save completed and partial window evidence")


def price_remaining(plan, qualification, deadline):
    """Admission estimate, before any scientific intervention; never shrink scope."""
    pairs = [row["update"]["wall_s"] for row in qualification["cases"] if row["case"] != "H"]
    pair_upper = max(pairs)
    history = next(row["update"]["wall_s"] for row in qualification["cases"] if row["case"] == "H")
    # 128 N + 16 E + 16 C pairs; H uses no model forwards/backwards.
    training = 160*pair_upper + 16*history
    source = 160*float(plan.get("source_forward_estimate_s", .5))
    probes = float(plan["remaining_probe_estimate_s"])
    overhead = float(plan.get("remaining_overhead_estimate_s", 1200.))
    estimated = training + source + probes + overhead
    remaining = deadline-time.monotonic()
    return dict(status="passed" if estimated <= remaining else "blocked",
        estimated_remaining_s=estimated, remaining_cap_s=remaining, training_s=training,
        source_observation_s=source, native_probe_s=probes, state_io_loading_export_allowance_s=overhead,
        qualification_max_pair_s=pair_upper, qualification_history_s=history,
        interpretation="Conservative qualification-rate admission; fixed full panel/windows, not outcome-dependent")


def run(plan, output, deadline):
    import torch
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    native_plan = read(plan["native_plan_path"])
    if any(native_plan[name] != plan[name] for name in ("origin_state", "origin_export", "origin_config", "schedule")):
        raise ValueError("Native source plan differs from sealed prefix-loss plan")
    if tuple(plan["constants"]["windows"]) != WINDOWS:
        raise ValueError("The sixteen fixed windows changed")
    receipt = dict(status="running", execution_commit=plan["execution_commit"],
        complete_response_generations=0, probe_exports=[], source_rows=[])
    write_json(output / "RESULT.json", receipt)
    ledger = Ledger(output / "PHYSICAL_WORK.jsonl", limits={"optimizer_step":182}, deadline=deadline)
    trainer = instrumentation = controller = None
    began = time.perf_counter()
    try:
        ledger.check()
        weight_bytes = (Path(plan["origin_export"])/"model.safetensors").stat().st_size
        receipt["host_admission"] = host_admission(2*Path(plan["origin_state"]).stat().st_size + 4*weight_bytes)
        pin_retained_graph_runtime(output)
        started = time.perf_counter()
        trainer = make_trainer(native_plan, output/"trainer", ledger=ledger)
        receipt["loading_s"] = time.perf_counter()-started
        instrumentation = TrainingInstrumentation(trainer, ledger)
        with ledger.scope(category="startup"):
            receipt["startup"] = discarded_startup(trainer, plan["feature_paths"], output/"startup")
        receipt["qualification"] = qualify(trainer, plan["feature_paths"], output/"qualification")
        receipt["duration_admission"] = price_remaining(plan, receipt["qualification"], deadline)
        write_json(output/"DURATION_ADMISSION.json", receipt["duration_admission"])
        write_json(output/"QUALIFICATION_PASSED.json", receipt)
        if receipt["duration_admission"]["status"] != "passed":
            raise WorkLimit("Qualified full remaining scientific work cannot fit the occupied-card cap")
        controller = WindowController(plan, output, ledger)
        trainer.on_window = controller
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with ledger.scope(category="native"):
            result = trainer.train_event("event_007", [Path(path) for path in plan["feature_paths"]], passes=2)
        receipt["trajectory_including_diagnostics_s"] = time.perf_counter()-started
        receipt["native_event"] = result
        receipt["final_native_parity"] = compare_final_native(trainer, plan["historical_native_state"], output/"FINAL_NATIVE_PARITY.json")
        if (len(controller.windows), len(controller.probe_exports), len(controller.source_work)) != (16,80,160):
            raise ValueError("Missing scheduled windows, endpoints or source response observations")
        if any(row["status"] != "completed" for row in controller.windows):
            raise ValueError("Incomplete prescribed native window")
        summary = ledger.summary()
        if summary["completed"].get("optimizer_step") != 182:
            raise ValueError("Expected 176 main plus six qualification optimizer applications")
        receipt.update(status="completed", global_step=trainer.global_step, presentations=trainer.presentations,
            main_peak_allocated_bytes=torch.cuda.max_memory_allocated(), main_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        write_json(output/"FAILED.json", receipt)
        raise
    finally:
        if controller is not None:
            controller.publish_progress()
            receipt.update(probe_exports=controller.probe_exports, source_rows=controller.source_rows,
                           source_work=controller.source_work, windows=controller.windows, witnesses=controller.witnesses)
        receipt.update(wall_s=time.perf_counter()-began, work=ledger.summary())
        receipt["qualification"] = receipt.get("qualification", read(output/"qualification/RESULT.json")
            if (output/"qualification/RESULT.json").exists() else None)
        write_json(output/"RESULT.json", receipt)
        if instrumentation is not None:
            instrumentation.close()
        if trainer is not None:
            trainer.close()
        ledger.close()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    with termination_cleanup():
        result = run(read(args.plan), args.output, args.deadline)
    print(json.dumps(dict(status=result["status"], exports=len(result["probe_exports"])) ))
