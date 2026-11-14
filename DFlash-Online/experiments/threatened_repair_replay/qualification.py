"""Six disposable steps from one exact first-window state, then discard them.

The controller constructs the prescribed bounded synthetic replay fixture and
the five callbacks. This module owns native-gradient/endpoint parity, state
restoration and failure receipts; it never launches a second qualification.
"""
from pathlib import Path
import time
import traceback

from experiments.full_k_gsm8k.training import live_state
from experiments.ok_continuation.resume import differences
from experiments.recent_repair_retention.training import schedule_adapter
from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.storage import write_json
from .state import capture_state, copy_tree, restore_state, tensor_change, verify_state


CASES = ("N", "disabled", "zero", "padded_zero", "U", "T")
NEUTRAL = ("disabled", "zero", "padded_zero")
SCIENTIFIC_FIELDS = ("draft_state_dict", "backend", "rng", "global_step", "presentations",
                     "strategy", "world_size", "reduction", "accumulation_steps", "horizon")


def selection_receipt(bundle):
    """Keep scientific row fields; ignore event names and wall-clock diagnostics."""
    rows = []
    for row in bundle["rows"]:
        observed = row["response_observation"]
        rows.append(dict(record_id=row["record_id"], pass_index=row["pass"],
            sampled_anchors=row["sampled_anchors"], loss=observed["loss"],
            ratio_metrics=observed["ratio_metrics"], sum_metrics=observed["sum_metrics"],
            metrics=row["metrics"], optimizer_step_after=row["optimizer_step_after"],
            optimizer_stepped=row["optimizer_stepped"]))
    return rows


def endpoint_receipt(trainer):
    """The returned tensors may be live; copy only the one reference endpoint."""
    current = live_state(trainer)
    return dict(scientific={name: current[name] for name in SCIENTIFIC_FIELDS},
        micro=trainer.core._micro, remainder=trainer.core.accumulation_remainder,
        schedule=vars(trainer._presentation_schedule))


def qualify(trainer, feature_paths, output, replay_callbacks):
    """Run after qualified startup, restore KPLUS6, return compact parity evidence.

    ``replay_callbacks`` has disabled/zero/padded_zero/U/T callables accepted by
    run_pair. U/T use a declared synthetic fixture and the same padded layouts;
    their callback receipts carry independent sparse-row/mask correctness tests.
    A callback is allowed to be None only for the disabled case.
    """
    import torch
    from experiments.functional_step_geometry.qualify import cleanup

    if set(replay_callbacks) != set(CASES[1:]):
        raise ValueError("Qualification requires exactly the five declared replay cases")
    if any(replay_callbacks[name] is None for name in CASES[2:]):
        raise ValueError("Zero, padded-zero and active qualification need their real replay callbacks")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    origin = capture_state(trainer)
    on_cuda = next(trainer.bundle.draft_model.parameters()).device.type == "cuda"
    started, failure = time.perf_counter(), None
    report = dict(status="running", cases=[], optimizer_steps=0, native_response_presentations=0,
                  complete_response_generations=0, startup="separately counted before this function",
                  counter_scope="completed run_pair receipts; monotonic ledger also retains failed/partial physical work")
    native_endpoint = native_gradients = None
    try:
        features = trainer._prepare_event("event_007", feature_paths, 2)
        trainer._active_event = "qualification_prepared"
        trainer._presentation_schedule = PresentationSchedule(
            schedule_adapter(trainer.schedule, features), [row["record_id"] for row in features], 2,
            features=features)
        trainer.bundle.model.begin_event()
        order = torch.randperm(len(features)).tolist()
        pair = [features[index] for index in order[:2]]
        pre = capture_state(trainer)
        # Preparation changes no weights or moments. Share these immutable CPU
        # leaves so qualification retains only P plus one native endpoint.
        for name in ("draft_state_dict", "backend"):
            if differences(origin["scientific"][name], pre["scientific"][name]):
                raise ValueError("Qualification preparation mutated weights or optimizer")
            origin["scientific"][name] = pre["scientific"][name]
        for name in CASES:
            case_dir = output / name
            case_dir.mkdir()
            restore_state(trainer, pre)
            verify_state(trainer, pre, case_dir / "PRE_STATE.json")
            if on_cuda:
                torch.cuda.reset_peak_memory_stats()
            case = dict(case=name, status="running")
            report["cases"].append(case)
            write_json(output / "RESULT.json", report)
            result = trainer.run_pair(pair, 0, 0, label=name, category="qualification",
                replay=None if name == "N" else replay_callbacks[name],
                capture_gradients=name == "N", reference_gradients=native_gradients)
            report["optimizer_steps"] += result["update"]["optimizer_applications"]
            report["native_response_presentations"] += len(result["rows"])
            case.update(update=result["update"], selection=selection_receipt(result))
            endpoint = endpoint_receipt(trainer)
            if name == "N":
                native_endpoint = copy_tree(endpoint)
                native_gradients = result["preclip_gradients"]["native"]
                native_selection = case["selection"]
                mismatch = []
            else:
                mismatch = differences(case["selection"], native_selection, "native_selection")
                if name in NEUTRAL:
                    mismatch += differences(endpoint, native_endpoint, "neutral_endpoint")
                    if result["update"]["gradient_change"]["difference_norm"] != 0:
                        mismatch.append("neutral_preclip_gradient")
                else:
                    # Active replay may alter clip norm and Adam state. Ordinary
                    # source loss/exposure and RNG/schedule still must agree.
                    mismatch = differences(
                        [{**row, "metrics": {key:value for key,value in row["metrics"].items()
                                            if key != "grad_norm"}} for row in case["selection"]],
                        [{**row, "metrics": {key:value for key,value in row["metrics"].items()
                                            if key != "grad_norm"}} for row in native_selection], "native_selection")
                    for field in ("rng", "global_step", "presentations"):
                        mismatch += differences(endpoint["scientific"][field],
                            native_endpoint["scientific"][field], field)
                    for field in ("micro", "remainder", "schedule"):
                        mismatch += differences(endpoint[field], native_endpoint[field], field)
                    if result["update"]["gradient_change"]["difference_norm"] <= 0:
                        mismatch.append("active_fixture_added_gradient_is_zero")
                    case["weights_vs_native"] = tensor_change(
                        native_endpoint["scientific"]["draft_state_dict"],
                        endpoint["scientific"]["draft_state_dict"])
            case.update(status="passed" if not mismatch else "failed", mismatches=mismatch[:64])
            if on_cuda:
                case.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(case_dir / "RESULT.json", case)
            write_json(output / "RESULT.json", report)
            if mismatch:
                raise ValueError(f"Threat replay qualification {name} differs: {mismatch[:8]}")
            del result, endpoint
        if report["optimizer_steps"] != 6 or report["native_response_presentations"] != 12:
            raise ValueError("Qualification physical step/presentation accounting differs")
        report["status"] = "passed"
    except BaseException as error:
        failure = error
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        write_json(output / "FAILED.json", report)
        raise
    finally:
        # A failed branch's physical ledger is never restored. Its PAIR_FAILED
        # receipt records any partially executed optimizer work before this.
        cleanup(lambda: restore_state(trainer, origin), failure, report, "restore qualification origin")
        cleanup(lambda: verify_state(trainer, origin, output / "RESTORED_ORIGIN.json"),
                failure, report, "verify qualification origin")
        report["wall_s"] = time.perf_counter() - started
        cleanup(lambda: write_json(output / "RESULT.json", report), failure, report, "save qualification result")
    return report
