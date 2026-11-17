"""Exact PR51 E5 control replays with live comparison and no duplicate exports.

All sampling, forwards, backwards, accumulation and Adam work stays in the
original MathQAFixedLRTrainer. The adapter observes its existing receipts and
replaces only final tensor publication with a comparison to retained boundaries.
"""
from copy import deepcopy
from pathlib import Path
import resource
import time
import traceback

from experiments.full_k_gsm8k import training as native
from experiments.mathqa_fixed_lr_transfer.training import MathQAFixedLRTrainer
from experiments.mathqa_fixed_lr_transfer import state as historical_state
from online_feedback.storage import write_json
from .runtime import WorkLedger, read

SCHEMA = "pr52_e5_control_replay_v1"


class ControlReplayTrainer(MathQAFixedLRTrainer):
    """Original native/S trainer, with read-only observers and compact publication."""

    def __init__(self, *args, comparator, **kwargs):
        self.comparator = comparator
        self._compare_live = False
        self._replay_features = {}
        self._checked_presentations = 0
        self.comparison_s = 0.
        super().__init__(*args, **kwargs)
        ordinary_step = self.core.train_step

        def observed_train_step(batch, context):
            result = ordinary_step(batch, context)
            if not self._compare_live:
                return result
            original_metrics = result.materialize_metrics
            observed = False
            def checked_metrics():
                nonlocal observed
                metrics = original_metrics()
                if not observed:
                    observed = True
                    # Return the result to its ordinary cached method after the
                    # first materialization; do not retain a closure/tensor cycle.
                    del result.materialize_metrics
                    ordinal = self._checked_presentations
                    record_id = str(batch.sample_ids[0])
                    feature = self._replay_features[record_id]
                    anchors = deepcopy(self._anchor_observations)
                    row = dict(record_id=record_id, **{"pass":ordinal//128},
                        optimizer_step_after=self.global_step+int(result.optimizer_stepped),
                        optimizer_stepped=result.optimizer_stepped,
                        learning_rate=self.backend.optimizer.get_learning_rate() if result.optimizer_stepped else None,
                        metrics=metrics, training_s=0., sampled_anchors=anchors,
                        metrics_scope="accumulation_window" if result.optimizer_stepped else "presentation",
                        response_observation=self.response_records[-1],
                        prompt_tokens=feature["prompt_length"], response_tokens=feature["response_tokens"],
                        active_anchors=sum(sum(draw["keep_mask"][0]) for draw in anchors))
                    self._compare("observe_presentation", row)
                    self._checked_presentations += 1
                return metrics
            result.materialize_metrics = checked_metrics
            return result
        self.core.train_step = observed_train_step

    def _compare(self, method, *args):
        began = time.perf_counter()
        try:
            return getattr(self.comparator, method)(*args)
        finally:
            self.comparison_s += time.perf_counter()-began

    def _append_observation(self, name, value):
        # Persist the original row before comparing; a mismatch stops before
        # the following response without hiding the just-completed work.
        super()._append_observation(name, value)
        if self._compare_live:
            if name == "RESPONSE_OBSERVATIONS.jsonl":
                self._compare("observe_response", value)
            elif name == "WINDOW_OBSERVATIONS.jsonl":
                self._compare("observe_window", value)

    def replay_event(self, feature_paths):
        """Run only E5; the original _train_passes owns every training operation."""
        event_id, passes = "event_005", 2
        began = time.perf_counter()
        features = self._prepare_event(event_id, feature_paths, passes)
        self._replay_features = {row["record_id"]:row for row in features}
        event_dir = self.events_root/event_id
        event_dir.mkdir()
        start_step = self.global_step
        write_json(event_dir/"PENDING.json", dict(event_id=event_id,features=features,
                   passes=passes,start_step=start_step,publication="compact_comparison_only"))
        self._compare_live = True
        try:
            self.bundle.model.begin_event()
            try:
                rows = self._train_passes(features, passes)
            finally:
                objective_costs = self.bundle.model.end_event()
            if self.core.accumulation_remainder or self._checked_presentations != 256:
                raise ValueError("Replay must finish all original presentations at an empty accumulation boundary")
            event = dict(event_id=event_id,features=features,passes=passes,
                response_presentations=len(rows),start_step=start_step,end_step=self.global_step,
                optimizer_steps=self.global_step-start_step,reduction=self.reduction,
                objective=self.objective,objective_work=objective_costs,
                gradient_denominator="responses per accumulation window",records=rows)
            names = {"available_position_occurrences", "selected_position_occurrences",
                "available_position_weight", "selected_position_weight", "no_error_blocks", "empty_blocks",
                "main_loss_numerator", "loss_denominator", "response_presentations", "supervised_position_exposure",
                *(f"first_error_position_{offset}" for offset in range(1,self.bundle.model.block_size))}
            windows = [row for row in rows if row["optimizer_stepped"]]
            event["selection_totals"] = {name:sum(row["metrics"][name] for row in windows) for name in sorted(names)}
            event["selection_totals_source"] = "optimizer_boundary_accumulation_windows_once"
            # The original hook supplies and validates PR51 arm/dataset/loss
            # identity. It constructs a live state, without serializing tensors.
            checkpoint = self._checkpoint_state(event)
            self.events.append(event)
            self.completed_event = 5
            self._check_native_path()
            self._check_actual_lr()
            boundary = native.verify_live(self, checkpoint)
            historical_state.validate_snapshot(checkpoint, **self.validation_arguments(),completed_event=5)
            receipt = dict(event,status="completed",counters=dict(global_step=self.global_step,
                presentations=self.presentations,completed_events=5),
                costs=dict(training_s=sum(row["training_s"] for row in rows),checkpoint_save_s=0.,
                           export_s=0.,event_wall_s_before_commit=time.perf_counter()-began))
            historical_state.validate_event_receipt(receipt,arm=self.arm,binding=self.binding,
                event_number=5,trainer_status=self.status())
            comparison = dict(event=deepcopy(self._compare("compare_event", receipt)),
                              endpoint=deepcopy(self._compare("compare_endpoint", checkpoint, self)))
            receipt.update(publication="compact_comparison_only",checkpoint_files_written=0,
                export_files_written=0,comparison=comparison,boundary=boundary,
                comparison_s=self.comparison_s)
            write_json(event_dir/"RESULT.json", receipt)
            return receipt
        except BaseException as error:
            self.failed = True
            write_json(event_dir/"FAILED.json", dict(error=repr(error),global_step=self.global_step,
                presentations=self.presentations,checked_presentations=self._checked_presentations,
                automatic_retry=False,checkpoint_files_written=0,export_files_written=0))
            raise
        finally:
            self._compare_live = False
            self._presentation_schedule = None


def _work_delta(before, after):
    return {phase:{key:after[phase][key]-before[phase][key] for key in after[phase]}
            for phase in ("attempted","completed")}


def _json_lines(path):
    import json
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run_replay(plan, arm):
    """One original E5 replay; failed or incomplete attempts cannot be repeated."""
    import torch
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from .replay_comparison import ReplayComparator
    from .replay_gate import validate_replay_prerequisite

    preflight_started = time.perf_counter()
    prerequisite = validate_replay_prerequisite(plan, arm)
    prerequisite_validation_s = time.perf_counter()-preflight_started
    features = prerequisite["feature_paths"]
    base = Path(plan["root"])/"prerequisite"/f"replay_{arm}5"
    base.mkdir(parents=True,exist_ok=True)  # Entry owns STATUS; never replace RESULT.
    destination = base/"RESULT.json"
    if destination.exists():
        raise FileExistsError("Preserve the previous control replay attempt")
    ledger = WorkLedger(plan["root"],f"prerequisite_replay_{arm}5")
    before = read(Path(plan["root"])/"WORK.json")
    saved = plan["retained_endpoints"][arm+"5"]
    endpoint = saved["endpoint"]
    event_dir = Path(saved["event_receipt_path"]).parent
    result = dict(schema=SCHEMA,status="running",arm=arm,event=5,binding=plan["binding"],
        source_binding=plan["pr51_binding"],execution_commit=plan["execution_commit"],
        prerequisite=prerequisite,prerequisite_validation_s=prerequisite_validation_s,historical_endpoint=endpoint,
        historical_feature_identity_verified=False,checkpoint_files_written=0,export_files_written=0,
        started_epoch=time.time())
    write_json(destination,result)
    trainer, failure = None, None
    try:
        reference = torch.load(endpoint["state"],map_location="cpu",weights_only=False,mmap=True)
        comparator = ReplayComparator(reference,read(saved["event_receipt_path"]),
            _json_lines(event_dir/"RESPONSE_OBSERVATIONS.jsonl"),
            _json_lines(event_dir/"WINDOW_OBSERVATIONS.jsonl"),arm=arm,output_dir=base/"comparison")
        result["runtime_compatibility"] = pin_retained_graph_runtime(base)
        began = time.perf_counter()
        trainer = ledger.call({"training_initialization":1},"load",ControlReplayTrainer,
            plan["replay_configs"][arm],base/"trainer",arm=arm,binding=plan["pr51_binding"],
            origin_state=plan["origin_state"],origin_export=plan["origin_export"],
            original_config_text=Path(plan["origin_config"]).read_text(),completed_event=4,
            comparator=comparator)
        result["loading_s"] = time.perf_counter()-began
        ledger.wrap(trainer.strategy,"forward_loss","response_forward")
        ledger.wrap(trainer.backend,"backward","response_backward")
        ledger.wrap(trainer.backend.optimizer.optimizer,"step","adam")
        result["startup"] = discarded_startup(trainer,features,base/"startup")
        began = time.perf_counter()
        receipt = trainer.replay_event(features)
        result.update(status="passed",event_wall_s=time.perf_counter()-began,
            comparison=receipt["comparison"],boundary=receipt["boundary"],costs=receipt["costs"],
            comparison_s=receipt["comparison_s"],physical_work=dict(trainer.work),
            counts=dict(optimizer_steps=receipt["optimizer_steps"],response_presentations=receipt["response_presentations"],
                start_step=receipt["start_step"],end_step=trainer.global_step,end_presentations=trainer.presentations))
        actual = _work_delta(before,read(Path(plan["root"])/"WORK.json"))
        expected = dict(adam=128,response_forward=258,response_backward=258,training_initialization=1)
        for phase, counts in actual.items():
            if any(counts[key] != expected.get(key,0) for key in counts):
                raise ValueError("Replay physical accounting differs: "+repr(actual))
        if trainer.latest.exists() or trainer.pending_state.exists() or (trainer.events_root/"event_005/export").exists():
            raise ValueError("Control replay unexpectedly published duplicate tensor artifacts")
    except BaseException as error:
        failure = error
        result.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        if trainer is not None:
            trainer.failed = True
        raise
    finally:
        result["work"] = _work_delta(before,read(Path(plan["root"])/"WORK.json"))
        result["wall_s"] = time.time()-result["started_epoch"]
        result["host_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        result["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0
        result["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_initialized() else 0
        write_json(destination,result)
        if trainer is not None:
            try:
                trainer.close()
            except BaseException as error:
                result.update(status="failed",cleanup_error=repr(error))
                write_json(destination,result)
                if failure is None:
                    raise
                failure.add_note("Replay cleanup also failed: "+repr(error))
    return result
