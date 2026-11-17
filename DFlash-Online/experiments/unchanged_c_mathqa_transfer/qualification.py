"""Six disposable updates per card: two native paths, zero cap, C and own reload.

The active rule is unchanged PR50. Inputs and native first-error/support terms
are checked independently of its changed numerator. Every failure is saved
before raising; origin and own temporary state are restored without aliasing.
"""
from pathlib import Path
import resource
import time
import traceback

from experiments.ok_continuation.resume import differences
from online_feedback.storage import write_json
from .runtime import WorkLedger
from .state import NATIVE_LR, validate_allocation_observation

CASES = ("pr51_native", "native", "cap_zero", "active", "continued_next_pair", "reloaded_next_pair")
SCIENTIFIC_FIELDS = ("forward", "preclip", "clip", "endpoint", "allocation", "block_inputs")


def _assert_equal(actual, expected, label):
    mismatch = differences(actual, expected)
    if mismatch:
        raise ValueError(label + ": " + str(mismatch[:16]))


def _native_forward_identity(rows):
    """Do not compare added diagnostics as if they were baseline loss fields."""
    def native(values):
        return {key:value for key,value in values.items() if not key.startswith("prefix_allocation_")}
    return [dict(record_id=row["record_id"], anchors=row["sampled_anchors"],
        loss=row["response_observation"]["loss"],
        sums=native(row["response_observation"]["sum_metrics"]),
        ratios=row["response_observation"]["ratio_metrics"], metrics=native(row["metrics"])) for row in rows]


def compare_training_cards(left, right):
    """Strict cross-card scientific comparison; a failure is reported for the caller."""
    errors = []
    for name, value in (("A", left), ("B", right)):
        if value.get("status") != "passed" or tuple(row["case"] for row in value.get("cases", [])) != CASES:
            errors.append(name + ": incomplete same-card qualification")
    if not errors:
        for label in ("binding", "execution_commit", "origin_before", "origin_after"):
            if differences(left[label], right[label]):
                errors.append("Cross-card " + label)
        for a, b in zip(left["cases"], right["cases"]):
            for key in SCIENTIFIC_FIELDS:
                if differences(a[key], b[key]):
                    errors.append(a["case"] + ": cross-card " + key)
    return dict(status="failed" if errors else "passed", errors=errors,
                exact_cross_card=not errors, extra_model_work=0,
                same_card_checks_passed=left.get("status") == right.get("status") == "passed")


def run_training_qualification(plan, device):
    import torch
    from experiments.functional_step_geometry.qualify import runtime_state
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.persistent_prefix_withholding.qualification import (
        scientific_identity, tensor_identity, tree_identity,
    )
    from online_feedback.rng import preserve_rng
    from online_feedback.training import SpecForgeEventTrainer
    from specforge.export.to_hf import export_to_hf
    from .training import MathQACandidateTrainer
    from .task_runner import feature_pool

    slot = device if device in ("A", "B") else next(
        (key for key, value in plan["devices"].items() if value == device), None)
    if slot not in ("A", "B"):
        raise ValueError("Qualification requires one sealed A/B physical GPU")
    base = Path(plan["root"]) / "qualification" / ("training_" + slot)
    base.mkdir(parents=True, exist_ok=True)  # Outer entry owns this directory and STATUS.json.
    destination = base / "DETAILS.json"
    if destination.exists():
        raise FileExistsError("Never overwrite an earlier qualification attempt")
    ledger = WorkLedger(plan["root"], "qualification_training_" + slot)
    report = dict(status="running", slot=slot, physical_gpu_uuid=plan["devices"][slot],
        binding=plan["binding"], execution_commit=plan["execution_commit"], cases=[],
        planned_adam=6, planned_core_forward_backward_each=12, startup_separate=True,
        started_epoch=time.time(), operation_counts={}, phase="loading")
    trainer, failure = None, None
    write_json(destination, report)

    def source_identity(current):
        source = current.origin
        return dict(weights={name:tensor_identity(value) for name,value in source["draft_state_dict"].items()},
            optimizer=tree_identity(source["backend"]["optimizer"]), rng=tree_identity(source["rng"]),
            global_step=source["global_step"], presentations=source["presentations"])

    def counted(kind, label, function, *args, **kwargs):
        # The local receipt describes work actually entered after the global
        # reservation. Global pending work remains visible if that call fails.
        phase = report["phase"]
        counts = report["operation_counts"].setdefault(phase, {})
        def execute():
            counts[kind + "_attempted"] = counts.get(kind + "_attempted", 0) + 1
            write_json(base/"WORK.json", report["operation_counts"])
            value = function(*args, **kwargs)
            counts[kind + "_completed"] = counts.get(kind + "_completed", 0) + 1
            write_json(base/"WORK.json", report["operation_counts"])
            return value
        return ledger.call({kind:1}, label, execute)

    def wrap(owner, name, kind):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            return counted(kind, name, original, *args, **kwargs)
        setattr(owner, name, wrapped)

    try:
        features_paths = feature_pool(plan, "C", 5)
        report["runtime_compatibility"] = pin_retained_graph_runtime(base)
        began = time.perf_counter()
        trainer = counted("training_initialization", "load", MathQACandidateTrainer,
            plan["configs"]["C"], base/"trainer", arm="C", binding=plan["binding"],
            origin_state=plan["origin_state"], origin_export=plan["origin_export"],
            original_config_text=Path(plan["origin_config"]).read_text(),
            completed_event=4, qualification_only=True)
        report["loading_s"] = time.perf_counter()-began
        wrap(trainer.strategy, "forward_loss", "response_forward")
        wrap(trainer.backend, "backward", "response_backward")
        wrap(trainer.backend.optimizer.optimizer, "step", "adam")
        began = time.perf_counter()
        report["origin_before"] = source_identity(trainer)
        expected_origin = scientific_identity(trainer)
        report["initial_identity_s"] = time.perf_counter()-began
        if (expected_origin["global_step"],expected_origin["presentations"],expected_origin["micro"],
                expected_origin["accumulation_remainder"]) != (512,1024,1024,0):
            raise ValueError("Qualification did not restore exact KPLUS4 counters")
        write_json(destination, report)
        report["phase"] = "startup"
        report["startup"] = discarded_startup(trainer, features_paths, base/"startup")
        report["startup_s"] = report["startup"]["wall_s"]
        _assert_equal(scientific_identity(trainer), expected_origin, "Startup exact restoration")
        runtime = runtime_state(trainer.bundle.model)
        features = SpecForgeEventTrainer._prepare_event(trainer,"qualification_features",features_paths,2)

        def reset(cap, label, *, draw_order=True):
            receipt = trainer.reset_qualification(relative_cap=cap, runtime=runtime)
            trainer._active_event = label
            (trainer.events_root/label).mkdir(exist_ok=False)
            actual = scientific_identity(trainer)
            _assert_equal(actual, expected_origin, "Exact initial scientific state")
            _assert_equal(runtime_state(trainer.bundle.model), runtime, "Initial modes/buffers/caches")
            trainer.bundle.model.begin_event()
            return (torch.randperm(len(features)).tolist() if draw_order else None), receipt

        def run_pair(label, pair, ordinal):
            row = dict(case=label, status="attempted", pair=[x["record_id"] for x in pair],
                       actual_learning_rate=NATIVE_LR,checks="pending")
            report["phase"] = label
            report["cases"].append(row)
            write_json(destination, report)
            observer_s = 0.
            def observe(current):
                nonlocal observer_s
                began = time.perf_counter()
                row["preclip"] = {name:tensor_identity(parameter.grad)
                    for name,parameter in current.bundle.draft_model.named_parameters() if parameter.requires_grad}
                observer_s += time.perf_counter()-began
                write_json(destination, report)  # Available even if the following optimizer fails.
            block_observer_s = 0.
            in_original_forward = False
            row["block_inputs"] = []
            def observe_block(proposals, labels, weights, selection):
                nonlocal block_observer_s
                if not in_original_forward:
                    return
                began = time.perf_counter()
                row["block_inputs"].append({name:tensor_identity(value) for name,value in (
                    ("proposals",proposals),("labels",labels),("weights",weights),("selection",selection))})
                block_observer_s += time.perf_counter()-began
            model = trainer.bundle.model
            had_observer = hasattr(model, "block_observer")
            previous_observer = getattr(model, "block_observer", None)
            if previous_observer is not None:
                raise ValueError("Disposable block observation hook is already occupied")
            forward = trainer.strategy.forward_loss
            def original_forward(*args, **kwargs):
                nonlocal in_original_forward
                in_original_forward = True
                try:
                    return forward(*args, **kwargs)
                finally:
                    in_original_forward = False
            trainer.strategy.forward_loss = original_forward
            model.block_observer = observe_block
            trainer.step_observer = observe
            began = time.perf_counter()
            try:
                rows = trainer.run_pair(pair,0,ordinal)
            finally:
                trainer.step_observer = None
                trainer.strategy.forward_loss = forward
                if had_observer:
                    model.block_observer = previous_observer
                else:
                    delattr(model, "block_observer")
            wall_s = time.perf_counter()-began
            row.update(status="completed",wall_s=wall_s,preclip_observer_s=observer_s,
                update_wall_excluding_preclip_observer_s=wall_s-observer_s,
                block_observer_s=block_observer_s,
                pair_wall_excluding_observers_s=wall_s-observer_s-block_observer_s,
                forward=_native_forward_identity(rows),clip=trainer.window_records[-1]["gradient_norm"],
                allocation=[row["response_observation"]["prefix_allocation"] for row in rows],
                endpoint=scientific_identity(trainer))
            write_json(destination, report)  # Persist the measured endpoint before checking it.
            return row

        baseline = None
        for label, cap in (("pr51_native",None),("native",None),("cap_zero",0.),("active",.25)):
            order, migration = reset(cap,label)
            pair = [features[index] for index in order[:2]]
            wrapper = trainer._native_forward
            if label == "pr51_native":
                trainer._native_forward = trainer.pr51_native_forward
            try:
                row = run_pair(label,pair,0)
            finally:
                trainer._native_forward = wrapper
            row["migration"] = migration
            if not row["block_inputs"]:
                raise ValueError("No original-forward block input evidence")
            if baseline is None:
                baseline = row
            elif label in ("native", "cap_zero"):
                for key in ("forward","preclip","clip","endpoint","block_inputs"):
                    _assert_equal(row[key],baseline[key],"PR51 " + label + " equivalence " + key)
                if cap == 0:
                    for summary in row["allocation"]:
                        if summary["changed_positions"] != 0 or summary["weight_moved"] != 0:
                            raise ValueError("Zero cap changed native coefficients")
                row["native_equivalence"] = "exact"
            else:
                _assert_equal(row["block_inputs"],baseline["block_inputs"],"Active native block inputs")
                for actual, expected in zip(row["forward"],baseline["forward"]):
                    native_sums = lambda values: {key:value for key,value in values.items() if key != "main_loss_numerator"}
                    _assert_equal(native_sums(actual["sums"]),native_sums(expected["sums"]),
                                  "Active native denominator/support/first-error terms")
                for summary in row["allocation"]:
                    validate_allocation_observation({"prefix_allocation":summary})
                movement = sum(item["changed_positions"] for item in row["allocation"])
                changed = [name for name,value in row["endpoint"]["weights"].items()
                           if value != baseline["endpoint"]["weights"][name]]
                row.update(changed_coefficient_occurrences=movement, changed_parameter_tensors=len(changed),
                           unchanged_native_inputs_and_first_error_terms=True)
                if movement <= 0 or not changed:
                    raise ValueError("Fixed qualification batch exercised no actual C intervention; no search authorized")
            row["checks"] = "passed"
            write_json(destination, report)

        report["phase"] = "snapshot_and_export"
        began = time.perf_counter()
        snapshot = base/"active_state.pt"
        report["snapshot"] = trainer.save_qualification(snapshot)
        saved_runtime = runtime_state(trainer.bundle.model)
        export = base/"active_export"
        config = Path(trainer.cfg.model.draft_model_config)
        if config.is_dir():
            config /= "config.json"
        with preserve_rng():
            export_to_hf(str(snapshot),str(config),str(export),
                embedding_source=trainer.cfg.model.target_model_path,
                embedding_key=trainer.cfg.model.embedding_key)
        report.update(export=str(export),snapshot_path=str(snapshot),persistence_export_s=time.perf_counter()-began)
        write_json(destination, report)
        pair = [features[index] for index in order[2:4]]
        continued = run_pair("continued_next_pair",pair,2)
        report["restore"] = trainer.restore_qualification(snapshot,runtime=saved_runtime)
        replay = run_pair("reloaded_next_pair",pair,2)
        for key in SCIENTIFIC_FIELDS:
            _assert_equal(replay[key],continued[key],"Serialized own-state next update " + key)
        continued["checks"] = replay["checks"] = "passed"
        reset(.25,"restored_origin",draw_order=False)
        _assert_equal(scientific_identity(trainer),expected_origin,"Final original-state restoration")
        report["origin_after"] = source_identity(trainer)
        _assert_equal(report["origin_after"],report["origin_before"],"Retained origin immutability")
        report.update(status="passed",optimizer_applications=6,response_presentations=12,
            startup_forward_backward_each=2,exact_native_wrapper=True,exact_native_cap0=True,
            actual_intervention=True,exact_own_resume=True,origin_restored=True)
    except BaseException as error:
        failure = error
        report.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        if trainer is not None:
            trainer.failed = True
        raise
    finally:
        if trainer is not None:
            trainer.step_observer = None
            report["physical_work"] = dict(trainer.work)
        report["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0
        report["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_initialized() else 0
        report["host_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report["wall_s"] = time.time()-report["started_epoch"]
        write_json(destination,report)
        if trainer is not None:
            try:
                trainer.close()
            except BaseException as error:
                report["cleanup_error"] = repr(error)
                if failure is None:
                    report.update(status="failed",error=repr(error))
                write_json(destination,report)
                if failure is None:
                    raise
                failure.add_note("Qualification cleanup also failed: " + repr(error))
    return report
