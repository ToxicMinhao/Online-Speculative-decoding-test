"""Seven disposable initialization/active-ranking checks; never run a campaign.

The one ordinary warmup is discarded. Six fresh runtimes restore the retained
post-construction, pre-sampling RNG. Exact gates remain exact and stop on the
first failed boundary after its evidence has been persisted.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import torch

from experiments.prefix_ranking.qualify import (
    RankingObserver, WorkLog, compare_detached_blocks, cpu_copy, override, run_window,
)
from online_feedback.qualify import assert_state_equal
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer

TRIALS = (("warmup", "osd", 0.), ("Oa", "osd", 0.), ("Ob", "osd", 0.),
          ("U0", "osd_uniform", 0.), ("P0", "osd_prefix", 0.),
          ("U1", "osd_uniform", 1.), ("P1", "osd_prefix", 1.))
GRADIENT_STAGES = ("after_backward_0", "after_backward_1", "pre_optimizer")
PLAN = dict(temporary_optimizer_steps=7, response_presentations=14,
            current_training_forwards=14, reference_training_forwards=4,
            detached_comparison_forwards=4, backwards=14, generations=0,
            main_optimizer_steps=0, teacher_model_forwards=0)
DEFAULT_DIAGNOSTIC = Path(__file__).resolve().parents[1] / "dflash_prefix_ranking_gpu_diagnostic_20261026/runner.py"
_helpers = None


def load_helpers(path=DEFAULT_DIAGNOSTIC):
    global _helpers
    spec = importlib.util.spec_from_file_location("controlled_qualification_diagnostic_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _helpers = module
    return module


def helpers():
    return _helpers if _helpers is not None else load_helpers()


def scientific_state(trainer):
    return helpers().scientific_state(trainer)


def storage_budget(weight_bytes):
    """All seven trials retained; no deletion/reclamation is assumed.

    Each trial: weights at initialization, three gradients, final weights plus
    two Adam moments = 7W. Add 4W failure evidence and 3W atomic-write overlap,
    2GiB work/metadata slack, and a separate 8GiB untouched reserve.
    """
    if type(weight_bytes) is not int or weight_bytes <= 0:
        raise ValueError("Positive integral draft bytes required")
    return dict(draft_bytes=weight_bytes, retained_bytes=49 * weight_bytes,
                failure_bytes=4 * weight_bytes, atomic_overlap_bytes=3 * weight_bytes,
                work_bytes=2 * 2**30, reserve_bytes=8 * 2**30,
                required_bytes=56 * weight_bytes + 10 * 2**30)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_states_compact(left, right):
    """Exact native-dtype equality first; FP64 statistics only for mismatches."""
    lt, lm = helpers().flatten_state(left)
    rt, rm = helpers().flatten_state(right)
    missing_left, missing_right = sorted(rt.keys() - lt.keys()), sorted(lt.keys() - rt.keys())
    differences = {}
    equal_tensors = coordinates = 0
    for name in sorted(lt.keys() & rt.keys()):
        a, b = lt[name], rt[name]
        coordinates += a.numel()
        if a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b):
            equal_tensors += 1
        else:
            differences[name] = helpers().tensor_difference(a, b)
    metadata = [dict(name=name, left=repr(lm.get(name)), right=repr(rm.get(name)))
                for name in sorted(lm.keys() | rm.keys())
                if name not in lm or name not in rm or type(lm[name]) is not type(rm[name]) or lm[name] != rm[name]]
    # Flattening alone loses empty containers and list/tuple distinctions.
    # Retain those contracts without serializing every equal tensor's metadata.
    def containers(value, prefix=""):
        result = {}
        if isinstance(value, dict):
            result[prefix] = ("dict", tuple(sorted((type(k).__name__, repr(k)) for k in value)))
            children = value.items()
        elif isinstance(value, (list, tuple)):
            result[prefix] = (type(value).__name__, len(value))
            children = enumerate(value)
        else:
            return result
        for key, child in children:
            result.update(containers(child, f"{prefix}.{key}" if prefix else str(key)))
        return result
    lc, rc = containers(left), containers(right)
    metadata.extend(dict(name=name, left=repr(lc.get(name)), right=repr(rc.get(name)), kind="container")
                    for name in sorted(lc.keys() | rc.keys()) if lc.get(name) != rc.get(name))
    return dict(exact_equal=not (missing_left or missing_right or differences or metadata),
                missing_left=missing_left, missing_right=missing_right,
                metadata_differences=metadata, tensor_differences=differences,
                equal_tensors=equal_tensors, compared_coordinates=coordinates)


def assert_finite(state):
    tensors, metadata = helpers().flatten_state(state)
    for name, value in tensors.items():
        if value.is_floating_point() or value.is_complex():
            flat = value.reshape(-1)
            for start in range(0, flat.numel(), 262144):
                if not torch.isfinite(flat[start:start + 262144]).all():
                    raise AssertionError(f"Nonfinite state: {name}")
    for name, value in metadata.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise AssertionError(f"Nonfinite metadata: {name}")


def core_state(trainer):
    return cpu_copy(dict(micro=trainer.core._micro,
                         ratio_totals=trainer.core._ratio_totals,
                         sum_totals=trainer.core._sum_totals))


def snapshot(trainer):
    return dict(**scientific_state(trainer), core=core_state(trainer),
                parameter_layout=helpers().parameter_layout(trainer))


def persist_comparison(path, left, right):
    result = compare_states_compact(left, right)
    write_json(path, result)
    if not result["exact_equal"]:
        raise AssertionError(f"Exact state gate failed; evidence: {path}")
    return result


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def frozen_tables(model):
    """Version/layout guard for shared frozen tables; does not copy target weights."""
    result = {}
    for prefix in ("embed_tokens", "lm_head"):
        module = getattr(model, prefix)
        for name, value in list(module.named_parameters()) + list(module.named_buffers()):
            if value.requires_grad or value.grad is not None:
                raise AssertionError(f"Target table is trainable or has gradients: {prefix}.{name}")
            result[f"{prefix}.{name}"] = dict(version=value._version, pointer=value.data_ptr(),
                                              shape=list(value.shape), dtype=str(value.dtype),
                                              stride=list(value.stride()))
    return result


def validate_fresh(trainer, pinned, trial_root):
    # A new construction owns new optimizer/counters/gradient storage. Restoring
    # the RNG happens after construction, never via a post-build seed shortcut.
    if trainer.cfg.training.seed != 17:
        raise AssertionError("The retained event seed must remain 17")
    if trainer.bundle.model.reference is not None:
        raise AssertionError("Fresh trainer unexpectedly contains a reference")
    restore_rng(pinned["rng"])
    initial = snapshot(trainer)
    save_torch(trial_root / "initial.pt", initial)
    assert_finite(initial)
    persist_comparison(trial_root / "initial_vs_pinned.json",
                       {key: initial[key] for key in pinned}, pinned)
    assert_state_equal(core_state(trainer), dict(micro=0, ratio_totals={}, sum_totals={}), "fresh_core")
    assert_state_equal((trainer.global_step, trainer.presentations, trainer.core.accumulation_remainder),
                       (0, 0, 0), "fresh_counters")
    if trainer.events or trainer._anchor_observations or trainer._presentation_schedule is not None:
        raise AssertionError("Fresh trainer contains event or sampling state")
    if any(p.grad is not None for p in trainer.bundle.draft_model.parameters()):
        raise AssertionError("Fresh trainer contains gradients")
    if trainer.backend.optimizer.state_dict()["optimizer_state_dict"]["state"]:
        raise AssertionError("Fresh Adam contains moments")
    return initial


@contextmanager
def check_full_shuffle(ordinary, log):
    original = torch.randperm
    calls = []

    def observed(n, *args, **kwargs):
        order = original(n, *args, **kwargs)
        # Observe only the actual leading shuffle. Restore the native function
        # before any model forward or compiler work sees the torch namespace.
        torch.randperm = original
        ids = [ordinary["features"][i]["record_id"] for i in order.tolist()] if n == 128 else []
        expected = [row["record_id"] for row in ordinary["records"][:128]]
        log.record("full_first_pass_shuffle", n=n, record_ids=ids, exact_equal=ids == expected)
        calls.append(n)
        assert_state_equal(ids, expected, "original_full_128_shuffle")
        return order

    with override(torch, "randperm", observed):
        yield
    assert_state_equal(calls, [128], "one_original_shuffle")


@contextmanager
def capture_gradients(trainer, trial_root, log, trial, baseline_root=None):
    original_backward, original_step = trainer.backend.backward, trainer.backend.optimizer.step

    def persist(stage):
        started = time.perf_counter()
        path = trial_root / f"{stage}.pt"
        state = dict(gradients=helpers().named_gradients(trainer), rng=capture_rng(),
                     core=core_state(trainer), global_step=trainer.global_step,
                     presentations=trainer.presentations,
                     parameter_layout=helpers().parameter_layout(trainer))
        save_torch(path, state)
        row = dict(stage=stage, path=str(path), bytes=path.stat().st_size,
                   persistence_s=time.perf_counter() - started)
        trial["gradient_snapshots"].append(row)
        log.record("gradient_snapshot", **row)
        assert_finite(state)
        if baseline_root is not None:
            persist_comparison(trial_root / f"{stage}_vs_Oa.json", state,
                               load_state(baseline_root / f"{stage}.pt"))

    def backward(*args, **kwargs):
        if log.counts["backward_attempted"] >= PLAN["backwards"]:
            raise AssertionError("Backward budget exhausted")
        with log.work("backward"):
            result = original_backward(*args, **kwargs)
        persist(f"after_backward_{log.presentation}")
        return result

    def step(*args, **kwargs):
        if log.counts["temporary_optimizer_step_attempted"] >= PLAN["temporary_optimizer_steps"]:
            raise AssertionError("Optimizer budget exhausted")
        persist("pre_optimizer")
        with log.work("temporary_optimizer_step"):
            return original_step(*args, **kwargs)

    with override(trainer.backend, "backward", backward), override(trainer.backend.optimizer, "step", step):
        yield


def validate_reference(trainer, pinned):
    model = trainer.bundle.model
    reference = model.reference.model.draft_model
    assert_state_equal(reference.state_dict(), pinned["draft"], "reference_released_weights")
    current = dict(trainer.bundle.draft_model.named_parameters())
    for name, value in reference.named_parameters():
        if value.requires_grad or value.grad is not None or value.data_ptr() == current[name].data_ptr():
            raise AssertionError("Reference aliases current parameters or has gradients")
    if any(module.training for module in reference.modules()):
        raise AssertionError("Reference is not in eval mode")


def run_qualification(config, ordinary_receipt_path, pinned_initial_path, output_root,
                      diagnostic_runner=None):
    if diagnostic_runner is not None:
        load_helpers(diagnostic_runner)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    ordinary = json.loads(Path(ordinary_receipt_path).read_text())
    if len(ordinary["features"]) != 128 or len(ordinary["records"]) != 256:
        raise ValueError("Use the complete retained 128-response, 256-presentation O event")
    pinned = load_state(pinned_initial_path)
    required = {"draft", "optimizer", "rng", "global_step", "presentations", "accumulation_remainder", "training_modes"}
    if set(pinned) != required:
        raise ValueError("Pin must be the original diagnostic initial.pt scientific-state schema")
    assert_finite(pinned)
    weight_bytes = sum(value.numel() * value.element_size() for value in pinned["draft"].values())
    budget = storage_budget(weight_bytes)
    budget["free_bytes_at_admission"] = shutil.disk_usage(root).free
    if budget["free_bytes_at_admission"] < budget["required_bytes"]:
        write_json(root / "STORAGE_REJECTED.json", budget)
        raise OSError("Insufficient peak storage including retained failure evidence and reserve")
    summary = dict(status="running", plan=PLAN, trials=[], storage=budget,
                   config=str(config), ordinary_receipt=str(ordinary_receipt_path),
                   pinned_initial=str(pinned_initial_path), pinned_initial_sha256=sha256(pinned_initial_path),
                   config_sha256=sha256(config), ordinary_receipt_sha256=sha256(ordinary_receipt_path),
                   diagnostic_runner=str(diagnostic_runner or DEFAULT_DIAGNOSTIC),
                   diagnostic_runner_sha256=sha256(diagnostic_runner or DEFAULT_DIAGNOSTIC),
                   baseline_tolerance="exact equality", generations=0, main_optimizer_steps=0,
                   schedule_scope="Full retained 256-row receipt validated; full first-pass 128 permutation checked; only first two anchor draws executed per trial",
                   historical_O_limit="No equivalence claim for retained O's entire 128-step trajectory",
                   snapshot_note="CPU persistence synchronizes gradients; costs are qualification-only")
    write_json(root / "PLAN.json", summary)
    log = WorkLog(root)
    trainer = None
    started = time.perf_counter()
    try:
        for label, objective, ranking_lambda in TRIALS:
            trial_root = root / label
            log.trial, log.presentation = label, None
            trial = dict(label=label, objective=objective, ranking_lambda=ranking_lambda,
                         status="running", records=[], gradient_snapshots=[])
            summary["trials"].append(trial)
            write_json(root / "PROGRESS.json", summary)
            if torch.cuda.is_initialized():
                torch.cuda.reset_peak_memory_stats()
            load_started = time.perf_counter()
            trainer = SpecForgeEventTrainer(config, trial_root / "trainer", objective=objective,
                                           ranking_lambda=ranking_lambda)
            trial["loading_s"] = time.perf_counter() - load_started
            trial["loading_peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
            trial["loading_peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
            trial["runtime"] = helpers().runtime_flags()
            trial["optimizer_recipe"] = trainer.backend.optimizer.recipe
            initial = validate_fresh(trainer, pinned, trial_root)
            del initial
            baseline_root = root / "Oa" if label in {"Ob", "U0", "P0"} else None
            if baseline_root is not None:
                persist_comparison(trial_root / "initial_vs_Oa.json", load_state(trial_root / "initial.pt"),
                                   load_state(baseline_root / "initial.pt"))
            model = trainer.bundle.model
            target_before = frozen_tables(model)
            before_rng = capture_rng()
            before_modes = {name: module.training for name, module in model.named_modules()}
            model.begin_event()
            assert_state_equal(capture_rng(), before_rng, "reference_setup_rng")
            assert_state_equal({name: module.training for name, module in model.named_modules()}, before_modes,
                               "reference_setup_modes")
            if ranking_lambda:
                if model.reference is None or not model.ranking_enabled:
                    raise AssertionError("Active ranking reference is absent")
                validate_reference(trainer, pinned)
            elif model.reference is not None or model.ranking_enabled:
                raise AssertionError("Lambda zero entered the reference/ranking branch")
            if label == "U1":
                by_id = {f["record_id"]: f for f in ordinary["features"]}
                compare_detached_blocks(trainer, [by_id[r["record_id"]] for r in ordinary["records"][:2]],
                                        ordinary["records"][:2], log)
            observer = RankingObserver(model, log)
            with observer.installed(), capture_gradients(trainer, trial_root, log, trial, baseline_root), \
                    check_full_shuffle(ordinary, log):
                for row in run_window(trainer, ordinary, log):
                    trial["records"].append(row)
                    write_json(root / "PROGRESS.json", summary)
                    if ranking_lambda:
                        observer.check_saved_gradients()
            # Save final state before every post-window gate, including active
            # reference assertions and historical scalar-metric comparisons.
            final = snapshot(trainer)
            save_torch(trial_root / "final.pt", final)
            trial["prefix_checks"] = observer.prefix_checks
            trial["ranking_gradient_checks"] = observer.gradient_checks
            trial["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
            trial["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
            write_json(trial_root / "TRIAL.json", trial)
            assert_finite(final)
            assert_state_equal(frozen_tables(model), target_before, "target_tables_version_layout")
            trial["frozen_target_version_layout_unchanged"] = True
            if label == "Oa":
                assert_state_equal([r["metrics"] for r in trial["records"]],
                                   [r["metrics"] for r in ordinary["records"][:2]], "historical_first_window_metrics")
            if baseline_root is not None:
                persist_comparison(trial_root / "final_vs_Oa.json", final, load_state(baseline_root / "final.pt"))
                oa = next(t for t in summary["trials"] if t["label"] == "Oa")
                strip = lambda rows: [{k: v for k, v in r.items() if k != "training_s"} for r in rows]
                persist_comparison(trial_root / "records_vs_Oa.json", strip(trial["records"]), strip(oa["records"]))
                trial["exact_baseline_compatibility"] = True
            if ranking_lambda:
                validate_reference(trainer, pinned)
                assert_state_equal((len(observer.prefix_checks), len(observer.gradient_checks)), (2, 2),
                                   "active_observer_coverage")
                expected = load_state(root / "Oa/final.pt")
                persist_comparison(trial_root / "rng_vs_Oa.json", final["rng"], expected["rng"])
                del expected
                trial["reference_rng_isolation"] = True
            trial["objective_work"] = model.end_event()
            trial["status"] = "discarded_warmup" if label == "warmup" else "passed"
            del final, observer, model
            trainer.close()
            trainer = None
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            write_json(trial_root / "TRIAL.json", trial)
            write_json(root / "PROGRESS.json", summary)
        expected = dict(current_training_forward_completed=14, reference_training_forward_completed=4,
                        detached_comparison_forward_completed=4, temporary_optimizer_step_completed=7,
                        response_presentations=14, backward_completed=14)
        for key, value in expected.items():
            assert_state_equal(log.counts[key], value, "work_budget." + key)
        summary["status"] = "passed"
        summary["main_campaign_started"] = False
        return summary
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        if summary["trials"]:
            trial["status"] = "failed"
            trial["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
            trial["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
            if trainer is not None:
                try:
                    failure = dict(state=snapshot(trainer), gradients=helpers().named_gradients(trainer))
                    save_torch(trial_root / "failure_state.pt", failure)
                    del failure
                    trial["failure_state"] = str(trial_root / "failure_state.pt")
                except BaseException as capture_error:
                    trial["failure_capture_error"] = repr(capture_error)
            write_json(trial_root / "TRIAL.json", trial)
        raise
    finally:
        summary["wall_s"] = time.perf_counter() - started
        summary["work_counts"] = dict(log.counts)
        summary["free_bytes_after"] = shutil.disk_usage(root).free
        write_json(root / "RESULT.json", summary)
        if trainer is not None:
            trainer.failed = True
            trainer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ordinary-receipt", required=True)
    parser.add_argument("--pinned-initial", required=True)
    parser.add_argument("--diagnostic-runner", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_qualification(args.config, args.ordinary_receipt, args.pinned_initial, args.output,
                      diagnostic_runner=args.diagnostic_runner)


if __name__ == "__main__":
    main()
