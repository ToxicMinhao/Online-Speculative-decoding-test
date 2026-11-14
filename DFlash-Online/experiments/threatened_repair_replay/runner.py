"""One native KPLUS6→7 trajectory with sixteen isolated one-step comparisons.

This process has no serving evaluator. It exports exact P/N/U/T weights for a
later fixed-input probe worker, so held-out outcomes cannot affect selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import signal
import time
import traceback

from experiments.ok_continuation.resume import differences
from online_feedback.rng import capture_rng, preserve_rng
from online_feedback.storage import save_torch, write_json
from .analysis import WINDOWS
from .core import bind_layouts, computation_key, padded_layouts, select_replay, tag_key
from .qualification import SCIENTIFIC_FIELDS, qualify, selection_receipt
from .replay import add_replay_gradients, observe_memory, replay_denominator
from .state import capture_state, compare_final_native, copy_tree, restore_state, tensor_change, verify_state
from .training import make_trainer
from .work import Ledger, TrainingInstrumentation, WorkLimit


GIB = 1024 ** 3
HOST_RESERVE = 32 * GIB


@contextmanager
def termination_cleanup():
    """Turn the campaign's first SIGTERM into a bounded evidence-preserving stop."""
    previous = signal.getsignal(signal.SIGTERM)
    def stop(_signal, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise WorkLimit("Campaign requested a safe stop; no further scientific work")
    signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def read(path):
    return json.loads(Path(path).read_text())


def host_admission(additional_bytes=0):
    """MemAvailable already accounts for active jobs and reclaimable cache."""
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(fields["MemAvailable"].split()[0]) * 1024
    required = HOST_RESERVE + int(additional_bytes)
    receipt = dict(available_bytes=available, additional_bytes=int(additional_bytes),
                   reserve_bytes=HOST_RESERVE, required_bytes=required, status="passed" if available >= required else "blocked")
    if available < required:
        raise MemoryError(f"Host memory admission failed: {receipt}")
    return receipt


def qualification_fixture(tags, layouts, current_ids):
    """An input-only artificial selection; labels and teacher targets stay real."""
    by_slot = defaultdict(list)
    for tag in sorted(tags, key=computation_key):
        if tag["request_id"] not in set(current_ids):
            by_slot[int(tag["proposal_slot"])].append(tag)
    for slot, choices in sorted(by_slot.items()):
        left = choices[0]
        right = next((tag for tag in choices if tag["layout_id"] != left["layout_id"]), None)
        if right is not None:
            uniform, threat = [deepcopy(left)], [deepcopy(right)]
            union = padded_layouts(layouts, uniform, threat)
            return dict(selected_U=uniform, selected_T=threat, layouts=union, slot=slot,
                        kind="synthetic selection on immutable native feedback; no label or teacher modification",
                        efficacy_evidence=False)
    raise ValueError("No bounded equal-position two-layout qualification fixture is available")


def export_weights(weights, template, destination):
    """Write only exact native weights plus unchanged serving config files."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from experiments.frozen_training_repair_replay.prepare import identity

    destination, template = Path(destination), Path(template)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with preserve_rng(), safe_open(str(template / "model.safetensors"), framework="pt", device="cpu") as source:
        if set(source.keys()) != set(weights):
            raise ValueError("Disposable export keys differ from the qualified DFlash export")
        tensors = {}
        for name in weights:
            value = weights[name].detach().cpu().contiguous()
            original = source.get_slice(name)
            if value.dtype != torch.bfloat16 or list(value.shape) != original.get_shape():
                raise ValueError("Disposable export tensor dtype/shape changed: " + name)
            tensors[name] = value
        save_file(tensors, str(destination / "model.safetensors"))
        for item in template.iterdir():
            if item.is_file() and item.suffix not in (".safetensors", ".bin", ".pt") and not item.name.endswith(".index.json"):
                shutil.copy2(item, destination / item.name)
    weights_identity = identity(destination / "model.safetensors")
    return dict(checkpoint=str(destination), weights_only=True, tensors=len(weights),
                bytes=weights_identity["bytes"], weights_identity=weights_identity,
                export_s=time.perf_counter() - started)


def scalar_update_rows(updates):
    """Flatten only scalar telemetry; per-layout/private bindings stay in JSON."""
    def flatten(value, prefix=""):
        result = {}
        for name, item in value.items():
            key = f"{prefix}.{name}" if prefix else name
            if name.endswith("path") or name in ("traceback", "error"):
                continue
            if isinstance(item, dict):
                result.update(flatten(item, key))
            elif item is None or isinstance(item, (bool, int, float, str)):
                result[key] = item
        return result
    return [flatten(row) for row in updates]


def optimizer_delta(expected, actual):
    """Separate moment differences from the exact optimizer-counter requirement."""
    a = expected["optimizer"]["optimizer_state_dict"]
    b = actual["optimizer"]["optimizer_state_dict"]
    if differences(a["param_groups"], b["param_groups"], "parameter_groups") or a["state"].keys() != b["state"].keys():
        raise ValueError("Disposable optimizer options or parameter membership changed")
    result = {}
    for field in ("step", "exp_avg", "exp_avg_sq"):
        left = {str(key): value[field] for key, value in a["state"].items()}
        right = {str(key): value[field] for key, value in b["state"].items()}
        result[field] = tensor_change(left, right)
    if result["step"]["changed_entries"]:
        raise ValueError("Disposable replay changed Adam step counters")
    return result


class WindowController:
    def __init__(self, plan, output, ledger, tags, layouts, *, observer=observe_memory,
                 replayer=add_replay_gradients, exporter=export_weights, admit=host_admission):
        self.plan, self.output, self.ledger = plan, Path(output), ledger
        self.tags, self.layouts = tags, layouts
        self.memory = bind_layouts(layouts, tags)
        self.observe, self.replay, self.export, self.admit = observer, replayer, exporter, admit
        self.probe_exports, self.selection_rows, self.memory_rows = [], [], []
        self.updates, self.parity, self.windows = [], [], []
        self.witnesses, self.active_pre = [], None
        self.snapshot_bytes = Path(plan["origin_state"]).stat().st_size

    def publish_progress(self):
        write_json(self.output / "WINDOW_SELECTION.json", self.selection_rows)
        write_json(self.output / "MEMORY_OUTCOMES.json", self.memory_rows)
        write_json(self.output / "UPDATE_RECEIPTS.json", self.updates)
        rows = scalar_update_rows(self.updates)
        fields = ["window", "state"] + sorted({key for row in rows for key in row} - {"window", "state"})
        with (self.output / "UPDATE_RECEIPTS.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        write_json(self.output / "STATE_PARITY.json", self.parity)
        write_json(self.output / "PROBE_EXPORTS.json", self.probe_exports)
        write_json(self.output / "WINDOWS.json", self.windows)

    def _witness(self, snapshot, window, *, reason):
        """At most two durable witnesses; opaque compiled caches stay in RAM."""
        if len(self.witnesses) >= 2 or any(row["window"] == window for row in self.witnesses):
            return
        path = self.output / "witnesses" / f"P_{window:03d}_{reason}.pt"
        payload = dict(scientific=snapshot["scientific"], core=snapshot["core"],
            gradients=snapshot["gradients"], bookkeeping=snapshot["bookkeeping"],
            schedule=snapshot["schedule_state"], costs=snapshot["costs"],
            runtime={key:snapshot["runtime"][key] for key in ("modes", "buffers", "nonpersistent")},
            replay_note="Exact in-event tensors/RNG/cursor; native attention caches are reconstructed from sealed inputs",
            window=window, reason=reason)
        with preserve_rng():
            save_torch(path, payload)
        self.witnesses.append(dict(window=window, reason=reason, path=str(path), bytes=path.stat().st_size))
        write_json(self.output / "WITNESSES.json", self.witnesses)

    def _scan(self, trainer, layouts, window, state, expected=None):
        """Observation counters are physical ledger work, not native telemetry."""
        before_work = deepcopy(trainer.work)
        before_rng = capture_rng()
        versions = [(parameter, parameter._version, parameter.grad) for parameter in trainer.bundle.model.parameters()]
        began = time.perf_counter()
        try:
            rows, receipts = self.observe(trainer.bundle.model, layouts,
                max_length=trainer.cfg.data.max_length, ledger=self.ledger,
                context=dict(window=window, state=state, category="memory"))
        finally:
            trainer.work.clear()
            trainer.work.update(before_work)
        path = self.output / f"window_{window:03d}" / f"MEMORY_{state}.json"
        write_json(path, dict(rows=rows, work=receipts, wall_s=time.perf_counter()-began))
        if differences(before_rng, capture_rng()) or any(
                parameter._version != version or parameter.grad is not gradient
                for parameter, version, gradient in versions):
            raise ValueError("Memory observation changed parameter/gradient/RNG state")
        if expected is not None:
            self.parity.append(dict(window=window, state=state, phase="after_observation",
                **verify_state(trainer, expected, path.with_name(f"{state}_OBSERVATION_PARITY.json"))))
        return rows, str(path)

    def _export(self, trainer, weights, window, state):
        self.ledger.check()
        receipt = self.export(weights, self.plan["origin_export"], self.output / "exports" / f"window_{window:03d}_{state}")
        row = dict(window=window, state=state, reused_from=None, **receipt)
        self.probe_exports.append(row)
        return row

    def _alias(self, window, state, original, reason):
        row = dict(original, window=window, state=state, reused_from=original["state"],
                   reuse_reason=reason, export_s=0., new_physical_export=False)
        self.probe_exports.append(row)
        return row

    def _memory_outcomes(self, rows, window, state, chosen, path, *, reused_from=None):
        for row in rows:
            if tag_key(row) in chosen:
                self.memory_rows.append(dict(row, window=window, state=state,
                    execution_id=path, reused_from=reused_from))

    def __call__(self, trainer, window, pair, pass_index, ordinal):
        from .qualification import endpoint_receipt

        self.ledger.check()
        directory = self.output / f"window_{window:03d}"
        directory.mkdir()
        window_record = dict(window=window, status="running", current_request_ids=[row["record_id"] for row in pair])
        self.windows.append(window_record)
        self.admit(2 * self.snapshot_bytes)
        pre = capture_state(trainer)
        self.active_pre = pre
        native = None
        failure = None
        try:
            if window in (8, 128):
                self._witness(pre, window, reason="fixed_boundary")
            self._export(trainer, pre["scientific"]["draft_state_dict"], window, "P")
            before_rows, before_path = self._scan(trainer, self.memory, window, "P", pre)
            native_result = trainer.run_pair(pair, pass_index, ordinal, label="N", category="native", capture_gradients=True)
            native_gradient = native_result["preclip_gradients"]["native"]
            native = capture_state(trainer)
            native_export = self._export(trainer, native["scientific"]["draft_state_dict"], window, "N")
            native_rows, native_path = self._scan(trainer, self.memory, window, "N", native)
            selection = select_replay(self.tags, before_rows, native_rows,
                window=window, current_request_ids=window_record["current_request_ids"])
            # Persist the memory-only decision before either branch or any probe
            # score exists. Deferred held-out evaluation has a separate process.
            write_json(directory / "SEALED_SELECTION.json", selection)
            self.selection_rows.extend(selection["rows"])
            union = padded_layouts(self.layouts, selection["selected_U"], selection["selected_T"])
            write_json(directory / "PADDED_LAYOUTS.json", union)
            chosen = {tag_key(tag) for layout in union for tag in layout["candidate_tags"]}
            self._memory_outcomes(before_rows, window, "P", chosen, before_path)
            self._memory_outcomes(native_rows, window, "N", chosen, native_path)
            self.updates.append(dict(native_result["update"], window=window, state="N",
                displacement=tensor_change(pre["scientific"]["draft_state_dict"], native["scientific"]["draft_state_dict"])))
            if window in (8, 128):
                restore_state(trainer, pre)
                verify_state(trainer, pre, directory / "DUPLICATE_PRE.json")
                duplicate = trainer.run_pair(pair, pass_index, ordinal, label="duplicate_N", category="duplicate",
                                             reference_gradients=native_gradient)
                expected = dict(scientific={name:native["scientific"][name] for name in SCIENTIFIC_FIELDS},
                    micro=native["core"]["micro"], remainder=0, schedule=native["schedule_state"])
                mismatch = differences(endpoint_receipt(trainer), expected, "duplicate_endpoint")
                mismatch += differences(selection_receipt(duplicate), selection_receipt(native_result), "duplicate_source")
                duplicate_receipt = dict(window=window, status="passed" if not mismatch else "failed", mismatches=mismatch,
                                         update=duplicate["update"])
                write_json(directory / "NATIVE_DUPLICATE.json", duplicate_receipt)
                self.parity.append(duplicate_receipt)
                if mismatch:
                    raise ValueError("Actual-window native duplicate differs")
                restore_state(trainer, native)
                verify_state(trainer, native, directory / "DUPLICATE_RESTORED_N.json")
            if not selection["active"]:
                self._alias(window, "U", native_export, "no threats; exact native fallback")
                self._alias(window, "T", native_export, "no threats; exact native fallback")
            else:
                device = next(trainer.bundle.draft_model.parameters()).device
                denominator = replay_denominator(selection["selected_U"], device=device)
                order = ("U", "T") if (window // 8) % 2 else ("T", "U")
                prior = None
                for state in order:
                    if prior is not None and selection["identical_sets"]:
                        export, rows, path = prior
                        self._alias(window, state, export, "identical immutable selected set, padded layouts and objective")
                        self._memory_outcomes(rows, window, state, chosen, path, reused_from=export["state"])
                        continue
                    restore_state(trainer, pre)
                    self.parity.append(dict(window=window, state=state, phase="before_branch",
                        **verify_state(trainer, pre, directory / f"{state}_PRE.json")))
                    selected = selection[f"selected_{state}"]
                    def replay(current, selected=selected, state=state):
                        return self.replay(current, union, selected, ledger=self.ledger,
                            context=dict(window=window, state=state), denominator=denominator)
                    result = trainer.run_pair(pair, pass_index, ordinal, label=state, category="disposable",
                        replay=replay, reference_gradients=native_gradient)
                    current = trainer.bundle.draft_model.state_dict()
                    activation = dict(result["update"], window=window, state=state,
                        displacement=tensor_change(pre["scientific"]["draft_state_dict"], current),
                        difference_from_native=tensor_change(native["scientific"]["draft_state_dict"], current),
                        optimizer_difference=optimizer_delta(native["scientific"]["backend"],
                            {"optimizer":trainer.backend.optimizer.state_dict()}))
                    self.updates.append(activation)
                    write_json(directory / f"{state}_UPDATE.json", activation)
                    export = self._export(trainer, current, window, state)
                    rows, path = self._scan(trainer, union, window, state)
                    self._memory_outcomes(rows, window, state, chosen, path)
                    prior = export, rows, path
                    restore_state(trainer, native)
                    self.parity.append(dict(window=window, state=state, phase="restored_N",
                        **verify_state(trainer, native, directory / f"{state}_RESTORED_N.json")))
                    self.admit()
            restore_state(trainer, native)
            self.parity.append(dict(window=window, phase="complete_window",
                **verify_state(trainer, native, directory / "RESTORED_N.json")))
            window_record.update(status="completed", active=selection["active"], identical_sets=selection["identical_sets"],
                                 selected_count=len(selection["selected_T"]))
            native_result["preclip_gradients"].clear()
            return native_result
        except BaseException as error:
            failure = error
            window_record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            write_json(directory / "FAILED.json", window_record)
            if len(self.witnesses) < 2:
                self._witness(pre, window, reason="failure")
            raise
        finally:
            from experiments.functional_step_geometry.qualify import cleanup
            # Do not disguise a failed update as a completed trajectory. This
            # restoration only leaves a clean inspectable native state in RAM.
            if native is not None:
                cleanup(lambda: restore_state(trainer, native), failure, window_record, "restore native endpoint")
            else:
                cleanup(lambda: restore_state(trainer, pre), failure, window_record, "restore pre-window state")
            self.active_pre = None
            cleanup(self.publish_progress, failure, window_record, "publish partial window evidence")


def run(plan, output, deadline):
    """Entry point for the parent campaign; no automatic retry or probe launch."""
    import torch
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from .replay import qualify_memory_observer

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    native_plan = read(plan["native_plan_path"])
    if any(native_plan[name] != plan[name] for name in ("origin_state", "origin_export", "origin_config", "schedule")):
        raise ValueError("Training source plan differs from the authenticated current seal")
    tags, layouts = read(plan["memory_tags"]), read(plan["memory_layouts"])
    if (len(tags), len(layouts)) != (815, 129) or tuple(plan["constants"]["windows"]) != WINDOWS:
        raise ValueError("Training runner requires the complete original memory and fixed windows")
    receipt = dict(status="running", execution_commit=plan["execution_commit"], complete_response_generations=0,
                   probe_exports=[], selection_rows=[], memory_rows=[], expected_duration_prior="1–2 occupied GPU hours; hard cap 3")
    write_json(output / "RESULT.json", receipt)
    ledger = Ledger(output / "PHYSICAL_WORK.jsonl", limits={"optimizer_step":168}, deadline=deadline)
    trainer = instrumentation = controller = None
    began = time.perf_counter()
    try:
        ledger.check()
        recovery_bytes = Path(plan["origin_state"]).stat().st_size
        weight_bytes = (Path(plan["origin_export"]) / "model.safetensors").stat().st_size
        receipt["host_admission"] = host_admission(2*recovery_bytes + 3*weight_bytes)
        pin_retained_graph_runtime(output)
        loading = time.perf_counter()
        trainer = make_trainer(native_plan, output / "trainer", ledger=ledger)
        receipt["loading_s"] = time.perf_counter()-loading
        ledger.check()
        instrumentation = TrainingInstrumentation(trainer, ledger)
        with ledger.scope(category="startup"):
            receipt["startup"] = discarded_startup(trainer, plan["feature_paths"], output / "startup")
        first_ids = [row["record_id"] for row in trainer.schedule["records"][:2]]
        fixture = qualification_fixture(tags, layouts, first_ids)
        write_json(output / "QUALIFICATION_FIXTURE.json", fixture)
        origin = capture_state(trainer)
        memory_started = time.perf_counter()
        receipt["memory_layout_qualification"] = qualify_memory_observer(trainer.bundle.model, fixture["layouts"],
            max_length=trainer.cfg.data.max_length, ledger=ledger, context=dict(state="fixture", category="qualification"))
        receipt["memory_layout_qualification"]["wall_s"] = time.perf_counter()-memory_started
        write_json(output / "MEMORY_LAYOUT_QUALIFICATION.json", receipt["memory_layout_qualification"])
        restore_state(trainer, origin)
        verify_state(trainer, origin, output / "FIXTURE_RESTORED.json")
        del origin
        if receipt["memory_layout_qualification"]["status"] != "passed":
            raise ValueError("Independent same-layout memory observation qualification failed")
        denominator = replay_denominator(fixture["selected_U"], device=next(trainer.bundle.draft_model.parameters()).device)
        def callback(selected, *, coefficient=.1, padded=False):
            return lambda current: add_replay_gradients(current, fixture["layouts"], selected,
                coefficient=coefficient, force_padded_zero=padded, ledger=ledger,
                context=dict(category="qualification", state=ledger.context.get("state"), work_role="replay"),
                denominator=None if padded else denominator)
        callbacks = dict(disabled=None, zero=callback(fixture["selected_U"], coefficient=0),
            padded_zero=callback([], padded=True), U=callback(fixture["selected_U"]), T=callback(fixture["selected_T"]))
        receipt["qualification"] = qualify(trainer, plan["feature_paths"], output / "qualification", callbacks)
        measured = receipt["memory_layout_qualification"].get("wall_s", 0.)
        fixture_forwards = len(fixture["layouts"])
        estimated_memory = measured / fixture_forwards * (4128 + 256) if fixture_forwards else None
        receipt["duration_estimate"] = dict(memory_scoring_s=estimated_memory,
            qualification_s=receipt["qualification"]["wall_s"],
            interpretation="Fixture rate includes the independent scalar oracle and estimates memory scans only; response-length variation, native probe reconstruction and training/export add uncertainty/cost",
            prior_total_s=[3600,7200], remaining_cap_s=max(0.,deadline-time.monotonic()))
        write_json(output / "QUALIFICATION_PASSED.json", receipt)
        ledger.check()
        controller = WindowController(plan, output, ledger, tags, layouts)
        trainer.on_window = controller
        if next(trainer.bundle.draft_model.parameters()).device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        training_started = time.perf_counter()
        with ledger.scope(category="native"):
            result = trainer.train_event("event_007", [Path(path) for path in plan["feature_paths"]], passes=2)
        receipt["trajectory_including_diagnostics_s"] = time.perf_counter()-training_started
        receipt["native_event"] = result
        receipt["final_native_parity"] = compare_final_native(trainer, plan["historical_native_state"], output / "FINAL_NATIVE_PARITY.json")
        if len(controller.windows) != 16 or any(row["status"] != "completed" for row in controller.windows):
            raise ValueError("Native trajectory did not complete all 16 fixed windows")
        receipt.update(status="completed", global_step=trainer.global_step, presentations=trainer.presentations,
            probe_exports=controller.probe_exports, selection_rows=controller.selection_rows,
            memory_rows=controller.memory_rows, windows=controller.windows, witnesses=controller.witnesses)
        if next(trainer.bundle.draft_model.parameters()).device.type == "cuda":
            receipt.update(main_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           main_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        write_json(output / "FAILED.json", receipt)
        raise
    finally:
        if controller is not None:
            controller.publish_progress()
            receipt.update(probe_exports=controller.probe_exports, selection_rows=controller.selection_rows,
                           memory_rows=controller.memory_rows, windows=controller.windows)
        receipt.update(wall_s=time.perf_counter()-began, work=ledger.summary(),
            raw_paths={name:str(output / name) for name in ("WINDOW_SELECTION.json", "MEMORY_OUTCOMES.json",
                       "UPDATE_RECEIPTS.json", "STATE_PARITY.json", "PHYSICAL_WORK.jsonl", "PROBE_EXPORTS.json")})
        parity = dict(status="passed" if receipt["status"] == "completed" else "incomplete_or_failed",
            initial_recovery=read(output / "trainer/RESUME_CHECK.json") if (output / "trainer/RESUME_CHECK.json").exists() else None,
            memory_layout_qualification=receipt.get("memory_layout_qualification"),
            six_case_qualification=receipt.get("qualification", read(output / "qualification/RESULT.json")
                if (output / "qualification/RESULT.json").exists() else None),
            qualification_origin_restoration=read(output / "qualification/RESTORED_ORIGIN.json")
                if (output / "qualification/RESTORED_ORIGIN.json").exists() else None,
            windows=controller.parity if controller is not None else [],
            final_native=receipt.get("final_native_parity"))
        write_json(output / "STATE_PARITY.json", parity)
        write_json(output / "RESULT.json", receipt)
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
    print(json.dumps(dict(status=result["status"], exports=len(result["probe_exports"]))))
