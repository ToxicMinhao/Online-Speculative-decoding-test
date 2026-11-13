"""One exact K+6 continuation, with an optional presentation-bound margin term."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import traceback

from experiments.full_k_gsm8k import training as native
from experiments.functional_step_geometry import training as geometry
from experiments.ok_coding.storage import identity
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from experiments.role_loss_pilot import training as historical
from online_feedback.kplus import install_kplus, kplus_contract
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer
from .reference import feature_paths, read, validate_schedule

SCHEMA = "recent_repair_retention_event7_v1"
ARMS = ("NATIVE", "RECENT", "CONTROL")
START_STEP, FINAL_STEP = 768, 896


def native_view(origin):
    contract = dict(origin["algorithm_contract"])
    old = contract[historical.CONTRACT_KEY]
    fsg = contract.pop(geometry.CONTRACT_KEY)
    if fsg["arm"] != "KPLUS" or fsg["geometry"] is not None or fsg["dataset"] != "gsm8k":
        raise ValueError("Require original native KPLUS6, not a geometry candidate")
    return historical.native_contract_view(contract, old["binding"], "KPLUS", "gsm8k")


def validate_origin(origin, original_config_text, *, expected_cuda_rng_states=1):
    view = native_view(origin)
    binding = view[native.CONTRACT_KEY]["binding"]
    checked = native.validate_snapshot(dict(origin, algorithm_contract=view), arm="K",
        binding=binding, config_text=origin["config_text"],
        original_config_text=original_config_text, completed_event=6,
        expected_contract=view, expected_recipe=origin["backend"]["optimizer"]["recipe"],
        expected_cuda_rng_states=expected_cuda_rng_states)
    for number, event in enumerate(origin["events"], 1):
        if number <= 4:
            historical.validate_event_receipt(event, arm="KPLUS", dataset="gsm8k",
                                              event_number=number, require_completed=False)
        else:
            geometry.validate_event_receipt(event, arm="KPLUS", dataset="gsm8k",
                                            event_number=number, require_completed=False)
    return dict(checked, exact_KPLUS6=True, start_step=START_STEP)


def continuation_contract(origin, binding, arm):
    if arm not in ARMS or len(binding) != 64:
        raise ValueError("Unknown retention arm or missing pre-training binding")
    result = deepcopy(origin["algorithm_contract"])
    result[SCHEMA] = dict(schema=SCHEMA, arm=arm, binding=binding,
        origin="exact_complete_KPLUS6", start_step=768, final_step=896,
        inherited_horizon=7424, seed_reset=False, reference_coefficient=0.10,
        parent_margin_fraction=0.50, parent_margin_cap=1.0,
        native_loss=kplus_contract(), initialization="own_complete_parent_state")
    return result


def schedule_adapter(schedule, features):
    """Translate read-only schedule field names, never substitute anchor draws."""
    rows = validate_schedule(schedule, [r["record_id"] for r in features])
    return dict(status="completed", passes=2, response_presentations=256, features=features,
        records=[dict(record_id=r["record_id"], **{"pass": r["pass_index"]},
                      sampled_anchors=r["sampled_anchors"]) for r in rows])


def assignment_index(payload, arm):
    rows = [] if arm == "NATIVE" else payload["arms"][arm]
    index = {i: [] for i in range(256)}
    for row in rows:
        translated = dict(row, parent_margin=row.get("parent_margin", row.get("margin")),
                          position_weight=row.get("position_weight", row.get("weight")))
        index[int(row["presentation_id"])].append(translated)
    return index


def validate_event(receipt, arm, *, complete=True, status=None):
    checked = native.validate_event_receipt(receipt, arm="K", event_number=7,
        require_completed=complete, trainer_status=status)
    if receipt.get("method") != arm or receipt.get("kplus_loss") != kplus_contract():
        raise ValueError("Event-seven arm or native K+ contract changed")
    for row in receipt["records"]:
        observation = row["response_observation"]
        values = observation["sum_metrics"]
        if values["kplus_response_count"] != 1 or not all(math.isfinite(float(x)) for x in values.values()):
            raise ValueError("Nonfinite or duplicated per-response objective accounting")
        if observation["record_id"] != row["record_id"] or observation["sampled_anchors"] != row["sampled_anchors"]:
            raise ValueError("Native response observations escaped their presentation")
    return dict(checked, arm=arm)


class RecentRetentionTrainer(native.FullKTrainer):
    """Use the established TrainerCore; this class binds evidence and boundaries."""
    def __init__(self, config_path, output_root, *, arm, binding, origin_state,
                 origin_export, original_config_text, schedule, assignments=None, expected_cuda_rng_states=1):
        import torch
        self.arm, self.binding = arm, binding
        self.origin_path, self.origin_export = Path(origin_state), Path(origin_export)
        self.original_config_text = original_config_text
        self.origin = torch.load(origin_state, map_location="cpu", weights_only=False, mmap=True)
        validate_origin(self.origin, original_config_text, expected_cuda_rng_states=expected_cuda_rng_states)
        _export_equal(self.origin, origin_export)
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.completed_event = 6
        self.response_records, self.window_records, self.work, self._work_handles = [], [], {}, []
        self.retention = None
        self.schedule = schedule
        self.assignments = assignment_index(assignments or {"arms": {}}, arm)
        self.presentation_ordinal = None
        self.capture_first_firing = arm != "NATIVE"
        self.scientific_ledger = None
        self.firing_capture = None
        self._pending_capture = None
        self._pending_gradient = None
        self._replaying = False
        self.diagnostic_capture_s = 0.0
        SpecForgeEventTrainer.__init__(self, config_path, output_root, resume_state=origin_state,
            objective="osd_first_error", reduction="response_mean", ranking_lambda=0.)

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        native.validate_config_text(self.config_text, "K", self.original_config_text)

    def _build_runtime(self):
        SpecForgeEventTrainer._build_runtime(self)
        install_kplus(self.bundle.model)
        source = dict(self.origin["algorithm_contract"])
        source.pop(historical.CONTRACT_KEY)
        source.pop(geometry.CONTRACT_KEY)
        if differences(source, self.contract):
            raise ValueError("Constructed native runtime differs from exact parent configuration")
        self.contract = continuation_contract(self.origin, self.binding, self.arm)
        self._install_observations()
        ordinary_forward, ordinary_step = self._native_forward, self._native_step

        def finite_forward(*args, **kwargs):
            import torch
            value = ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(value.loss).all()):
                raise ValueError("Nonfinite response loss")
            return value

        def finite_step():
            from experiments.adaflash_arepo.qualify import finite_gradient_check
            from experiments.prefix_guard.qualify import named_gradients
            parameters = {n:p for n,p in self.bundle.draft_model.named_parameters() if p.requires_grad}
            finite_gradient_check(parameters, self.output_root, self.presentations)
            if self._replaying or (self._pending_capture is not None and self.window_fires()):
                self._pending_gradient = named_gradients(self)
            return ordinary_step()

        self._native_forward, self._native_step = finite_forward, finite_step

    def _restore(self, path):
        state = dict(self.origin, config_text=self.config_text, run_id=self.cfg.run_id,
                     algorithm_contract=self.contract)
        self.bundle.draft_model.load_state_dict(state["draft_state_dict"], strict=True)
        self.backend.load_state_dict(deepcopy(state["backend"]))
        self.global_step, self.presentations = state["global_step"], state["presentations"]
        self.events = deepcopy(state["events"])
        self.core._micro = self.presentations
        restore_rng(state["rng"])
        receipt = native.verify_live(self, state)
        receipt.update(origin_state=str(self.origin_path), origin_export=str(self.origin_export),
            source_step=768, historical_receipts_unchanged=True,
            changes=["additive retention contract", "run/output metadata"])
        write_json(self.output_root / "RESUME_CHECK.json", receipt)

    def install_retention(self, *, enabled=True, coefficient=.10):
        from online_feedback.recent_repair_retention import install_recent_repair_retention
        self.retention = install_recent_repair_retention(self.bundle.model,
            enabled=enabled, coefficient=coefficient)
        return self.retention

    def _batch(self, raw):
        if self.retention is not None and self.presentation_ordinal is not None:
            row = self.schedule["records"][self.presentation_ordinal]
            if raw["record_id"] != row["record_id"]:
                raise ValueError("Retained assignment request identity differs")
            anchors = row["sampled_anchors"][0]
            self.retention.set_presentation(self.assignments[self.presentation_ordinal],
                request_id=raw["record_id"], presentation_id=self.presentation_ordinal,
                prompt_length=raw["prompt_length"], expected_anchors=anchors["positions"],
                expected_keep=anchors["keep_mask"])
        return SpecForgeEventTrainer._batch(self, raw)

    def window_fires(self):
        return any(row["sum_metrics"].get("retention_firing_count", 0) > 0
                   for row in self.response_records[-2:])

    def _prepare_event(self, event_id, feature_paths, passes):
        if len(self.events) != 6 or event_id != "event_007":
            raise ValueError("Only one exact event-seven continuation is authorized")
        if (self.arm != "NATIVE") != (self.retention is not None):
            raise ValueError("Declared arm and installed objective extension differ")
        return native.FullKTrainer._prepare_event(self, event_id, feature_paths, passes)

    def _train_passes(self, features, passes):
        import torch
        from specforge.training.strategies.base import StepContext
        from experiments.prefix_ranking.qualify import cpu_copy
        from experiments.functional_step_geometry.qualify import runtime_state
        rows = []
        for pass_index in range(passes):
            order = torch.randperm(len(features)).tolist()
            for index in order:
                ordinal = len(rows)
                self.presentation_ordinal = ordinal
                feature = features[index]
                self._presentation_schedule.begin_presentation(pass_index, feature["record_id"])
                if ordinal % 2 == 0 and self.capture_first_firing and self.firing_capture is None:
                    capture_began = time.perf_counter()
                    self._pending_capture = dict(
                        scientific=cpu_copy(native.live_state(self)),
                        runtime=runtime_state(self.bundle.model), ordinal=ordinal,
                    )
                    self.diagnostic_capture_s += time.perf_counter() - capture_began
                raw = torch.load(feature["path"], map_location="cpu", weights_only=True)
                self._anchor_observations.clear()
                began = time.perf_counter()
                result = self.core.train_step(self._batch(raw), StepContext(global_step=self.global_step,
                    total_steps=self.horizon, collect_detailed_metrics=True))
                if ordinal < 2 and self.response_records[-1]["sum_metrics"].get("retention_hinge_numerator", 0) != 0:
                    raise ValueError("Unchanged parent has nonzero natural retention hinge")
                self._presentation_schedule.finish_presentation()
                metrics = result.materialize_metrics()
                elapsed = time.perf_counter() - began
                self.presentations += 1
                self.global_step += int(result.optimizer_stepped)
                if not all(math.isfinite(float(x)) for x in metrics.values()):
                    raise ValueError("Nonfinite ordinary or auxiliary metric")
                row = dict(record_id=feature["record_id"], **{"pass":pass_index},
                    optimizer_step_after=self.global_step, optimizer_stepped=result.optimizer_stepped,
                    learning_rate=self.backend.optimizer.get_learning_rate() if result.optimizer_stepped else None,
                    metrics=metrics, training_s=elapsed, sampled_anchors=deepcopy(self._anchor_observations),
                    metrics_scope="accumulation_window" if result.optimizer_stepped else "presentation",
                    response_observation=self.response_records[-1], prompt_tokens=feature["prompt_length"],
                    response_tokens=feature["response_tokens"], feature_path=feature["path"])
                rows.append(row)
                if result.optimizer_stepped and self._pending_capture is not None:
                    if self.window_fires():
                        capture_began = time.perf_counter()
                        directory = self.output_root / "first_firing"
                        directory.mkdir()
                        self.firing_capture = self._pending_capture
                        self.firing_capture.update(
                            feature_paths=[r["feature_path"] for r in rows[-2:]],
                            endpoint=cpu_copy(native.live_state(self)),
                            gradients=self._pending_gradient,
                            observations=deepcopy(self.response_records[-2:]),
                        )
                        save_torch(directory / "PRESTATE.pt", self.firing_capture["scientific"])
                        save_torch(directory / "PRODUCTION_ENDPOINT.pt", {
                            k:self.firing_capture[k] for k in ("endpoint", "gradients", "observations")})
                        write_json(directory / "IDENTITY.json", dict(
                            ordinal=self.firing_capture["ordinal"],
                            feature_paths=self.firing_capture["feature_paths"],
                            global_step=self.global_step, natural_firing=True))
                        self.diagnostic_capture_s += time.perf_counter() - capture_began
                    self._pending_capture = self._pending_gradient = None
        self.presentation_ordinal = None
        if len(rows) != 256 or len(self.window_records) != 128:
            raise ValueError("Scientific event accounting differs")
        return rows

    def _checkpoint_state(self, event):
        event.update(method=self.arm, dataset="gsm8k", kplus_loss=kplus_contract(),
                     retention_contract=self.contract[SCHEMA])
        validate_event(event, self.arm, complete=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)

    def train_event(self, event_id, features, passes=2):
        metadata = self._prepare_event(event_id, features, passes)
        schedule = schedule_adapter(self.schedule, metadata)
        result = SpecForgeEventTrainer.train_event(self, event_id, features, passes,
                                                   presentation_schedule=schedule)
        validate_event(result, self.arm, status=self.status())
        return result

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch
        with preserve_rng():
            state = torch.load(self.latest, map_location="cpu", weights_only=False, mmap=True)
            if differences(state["events"][:6], self.origin["events"]):
                raise ValueError("Historical event receipts changed")
            if differences(state["algorithm_contract"], continuation_contract(self.origin,self.binding,self.arm)):
                raise ValueError("Event-seven scientific contract changed")
            view = native_view(self.origin)
            receipt = native.validate_snapshot(dict(state, algorithm_contract=view), arm="K",
                binding=view[native.CONTRACT_KEY]["binding"], config_text=self.config_text,
                original_config_text=self.original_config_text, completed_event=7,
                expected_contract=view, expected_recipe=self.backend.optimizer.recipe,
                expected_cuda_rng_states=self.expected_cuda_rng_states)
            validate_event(state["events"][6], self.arm, complete=False)
            if export_path is not None:
                _export_equal(state, export_path)
            receipt["live"] = native.verify_live(self, state)
            receipt["export_exact"] = export_path is not None
        if receipt_path is not None:
            write_json(receipt_path, receipt)
        return receipt


def make_trainer(plan, output, arm):
    assignments = read(plan["assignments"])
    return RecentRetentionTrainer(plan["configs"][arm], output, arm=arm,
        binding=plan["binding"], origin_state=plan["origin_state"],
        origin_export=plan["origin_export"], original_config_text=Path(plan["origin_config"]).read_text(),
        schedule=read(plan["schedule"]), assignments=assignments)


def restore_disposable(trainer, state, runtime):
    from experiments.functional_step_geometry.qualify import restore_runtime
    trainer.bundle.draft_model.load_state_dict(state["draft_state_dict"], strict=True)
    trainer.backend.load_state_dict(deepcopy(state["backend"]))
    trainer.backend.optimizer.zero_grad()
    trainer.global_step, trainer.presentations = state["global_step"], state["presentations"]
    trainer.core._micro = trainer.presentations
    trainer.core._sum_totals.clear()
    trainer.core._ratio_totals.clear()
    trainer.events = deepcopy(state["events"])
    restore_runtime(trainer.bundle.model, runtime)
    restore_rng(state["rng"])


def difference_norm(left, right):
    """Bounded CPU FP64 reduction; no full-model FP64 duplicate."""
    import torch
    squared, left_squared, count = 0., 0., 0
    for name, a in left.items():
        b = right[name]
        if a is None or b is None:
            if a is not b:
                raise ValueError("Gradient/moment membership changed")
            continue
        aa, bb = a.reshape(-1), b.reshape(-1)
        for start in range(0, aa.numel(), 1 << 20):
            x, y = aa[start:start+(1 << 20)].double(), bb[start:start+(1 << 20)].double()
            squared += float((x-y).square().sum())
            left_squared += float(x.square().sum())
            count += int((x != y).sum())
    return dict(left_norm=math.sqrt(left_squared), difference_norm=math.sqrt(squared), different_entries=count)


def first_firing_replays(trainer, output):
    """Compare one already observed window, then restore the final scientific state."""
    import torch
    from specforge.training.strategies.base import StepContext
    from experiments.prefix_ranking.qualify import cpu_copy
    from experiments.functional_step_geometry.qualify import runtime_state
    from experiments.prefix_guard.qualify import persist_comparison
    capture = trainer.firing_capture
    output = Path(output)
    if capture is None:
        value = dict(status="not_observed", natural_firing=False,
                     response_forwards=0, response_backwards=0, temporary_optimizer_steps=0)
        write_json(output / "RESULT.json", value)
        return value
    began = time.perf_counter()
    final_state, final_runtime = cpu_copy(native.live_state(trainer)), runtime_state(trainer.bundle.model)
    final_costs = deepcopy(trainer.bundle.model.costs)
    final_binding = trainer.retention.presentation
    final_anchors = deepcopy(trainer._anchor_observations)
    bookkeeping = (trainer.response_records, trainer.window_records, trainer.work,
                   trainer._active_event, trainer._presentation_schedule)
    receipt = dict(status="running", natural_firing=True, response_forwards=0,
                   response_backwards=0, temporary_optimizer_steps=0)
    endpoints = {}
    from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
    limits = dict(response_forwards=4, response_backwards=4, optimizer_steps=2)
    output.mkdir(parents=True, exist_ok=True)
    ledger = WorkLedger(output / "WORK.jsonl", limits)
    wrappers = []
    for owner,name,kind in ((trainer.core.strategy,"forward_loss","response_forwards"),
                            (trainer.backend,"backward","response_backwards"),
                            (trainer.backend.optimizer.optimizer,"step","optimizer_steps")):
        wrappers.append((owner,name,getattr(owner,name)))
        ledger.wrap(owner,name,kind)
    before_coefficient = trainer.retention.coefficient
    failure = None
    try:
        trainer._replaying = True
        trainer._presentation_schedule = None
        for label, coefficient in (("ACTIVE", .10), ("OFF", 0.)):
            restore_disposable(trainer, capture["scientific"], capture["runtime"])
            trainer.retention.coefficient = coefficient
            trainer.response_records, trainer.window_records, trainer.work = [], [], {}
            trainer._active_event = "first_firing_" + label
            (trainer.events_root / trainer._active_event).mkdir()
            trainer.bundle.model.begin_event()
            for local, feature in enumerate(capture["feature_paths"]):
                trainer.presentation_ordinal = capture["ordinal"] + local
                raw = torch.load(feature, map_location="cpu", weights_only=True)
                trainer._anchor_observations.clear()
                result = trainer.core.train_step(trainer._batch(raw), StepContext(
                    global_step=trainer.global_step,total_steps=trainer.horizon,collect_detailed_metrics=True))
                result.materialize_metrics()
                receipt["response_forwards"] += 1
                receipt["response_backwards"] += 1
                receipt["temporary_optimizer_steps"] += int(result.optimizer_stepped)
                trainer.presentations += 1
                trainer.global_step += int(result.optimizer_stepped)
            state = cpu_copy(native.live_state(trainer))
            endpoints[label] = dict(state=state, gradients=trainer._pending_gradient)
            save_torch(output / f"{label}_ENDPOINT.pt", endpoints[label])
            if label == "ACTIVE":
                receipt["active_exact"] = persist_comparison(output / "ACTIVE_EXACT.json",
                    dict(state=state, gradients=trainer._pending_gradient),
                    dict(state=capture["endpoint"], gradients=capture["gradients"]))
            receipt[label.lower()+"_physical_work"] = dict(trainer.work)
        receipt["preclip_gradients"] = difference_norm(endpoints["OFF"]["gradients"],endpoints["ACTIVE"]["gradients"])
        receipt["bf16_weights"] = difference_norm(endpoints["OFF"]["state"]["draft_state_dict"],endpoints["ACTIVE"]["state"]["draft_state_dict"])
        receipt["moments"] = {}
        for moment in ("exp_avg", "exp_avg_sq"):
            values = [{str(k):v[moment] for k,v in endpoints[a]["state"]["backend"]["optimizer"]["optimizer_state_dict"]["state"].items()} for a in ("OFF","ACTIVE")]
            receipt["moments"][moment] = difference_norm(*values)
        require_complete_work(read_work(ledger.path,limits),limits)
        receipt["status"] = "passed"
    except BaseException as error:
        failure = error
        receipt.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        write_json(output / "RESULT.json", receipt)
        raise
    finally:
        for owner,name,original in wrappers:
            setattr(owner,name,original)
        receipt["work"] = read_work(ledger.path,limits)
        def restore_final():
            trainer.retention.coefficient = before_coefficient
            restore_disposable(trainer, final_state, final_runtime)
            (trainer.response_records,trainer.window_records,trainer.work,
             trainer._active_event,trainer._presentation_schedule) = bookkeeping
            trainer._replaying = False
            trainer.presentation_ordinal = None
            trainer._pending_gradient = None
            trainer.bundle.model.costs = final_costs
            trainer.retention.presentation = final_binding
            trainer._anchor_observations = final_anchors
            receipt["final_exact_restoration"] = persist_comparison(
                output / "FINAL_EXACT_RESTORATION.json", native.live_state(trainer), final_state)
        from experiments.functional_step_geometry.qualify import cleanup
        try:
            cleanup(restore_final, failure, receipt, "restore scientific state after first firing replay")
        except BaseException as error:
            receipt.update(status="failed",error=repr(error),traceback=traceback.format_exc())
            raise
        finally:
            receipt["wall_s"] = time.perf_counter() - began
            write_json(output / "RESULT.json", receipt)
    return receipt


def execute(plan_path, arm, output):
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.functional_step_geometry.qualify import cleanup
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.full_k_gsm8k.costs import begin_event_profile, finish_event_profile
    plan, output = read(plan_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = feature_paths(plan)
    progress = dict(status="running", arm=arm, execution_commit=plan["execution_commit"],
                    scientific_response_forwards=0, scientific_adam_steps=0, automatic_retry=False)
    trainer = None
    failure = None
    began = time.perf_counter()
    try:
        pin_retained_graph_runtime(output)
        loading = time.perf_counter()
        trainer = make_trainer(plan, output / "trainer", arm)
        progress["loading_s"] = time.perf_counter() - loading
        progress["startup"] = discarded_startup(trainer, paths, output / "startup")
        if arm != "NATIVE":
            trainer.install_retention()
        from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
        limits = dict(response_forwards=256, response_backwards=256, optimizer_steps=128)
        ledger = WorkLedger(output / "WORK.jsonl", limits)
        wrapped = []
        for owner, name, kind in ((trainer.core.strategy,"forward_loss","response_forwards"),
                                  (trainer.backend,"backward","response_backwards"),
                                  (trainer.backend.optimizer.optimizer,"step","optimizer_steps")):
            wrapped.append((owner,name,getattr(owner,name)))
            ledger.wrap(owner,name,kind)
        profile = begin_event_profile()
        try:
            result = trainer.train_event("event_007", paths)
        finally:
            write_json(output / "RESOURCE_COSTS.json", finish_event_profile(profile))
            for owner,name,original in wrapped:
                setattr(owner,name,original)
            progress["work"] = read_work(ledger.path, limits)
        require_complete_work(progress["work"], limits)
        progress.update(scientific_response_forwards=256, scientific_adam_steps=128)
        result["diagnostic_capture_s"] = trainer.diagnostic_capture_s
        result["first_firing"] = first_firing_replays(trainer, output / "first_firing_replay")
        result["boundary"] = trainer.verify_boundary(result["checkpoint_path"], output / "BOUNDARY_CHECK.json")
        result.update(weights_identity=identity(Path(result["checkpoint_path"])/"model.safetensors"),
                      state_identity=identity(trainer.latest), state_path=str(trainer.latest))
        write_json(output / "RESULT.json", result)
        progress.update(status="completed", final_global_step=896)
        return result
    except BaseException as error:
        failure = error
        progress.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            progress.update(global_step=trainer.global_step,presentations=trainer.presentations,
                            physical_work=dict(trainer.work))
            cleanup(trainer.close,failure,progress,"trainer close")
        progress["wall_s"] = time.perf_counter()-began
        write_json(output / "STATUS.json",progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",required=True)
    parser.add_argument("--arm",choices=ARMS,required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    execute(args.plan,args.arm,args.output)
