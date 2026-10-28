"""Seven discarded O/C/K/A qualification windows; no exports or generation.

Exact all-selected comparisons use the retained startup/RNG machinery. Natural
masks are checked on the existing training forwards; only eight detached logit
rows per presentation survive until their prediction gradients are checked.
"""
from contextlib import contextmanager
import gc
import importlib.util
import json
from pathlib import Path
import shutil
import time
import traceback

import torch
import torch.nn.functional as F

from experiments.first_error.campaign import QUALIFICATION_GATES
from experiments.prefix_ranking.qualify import WorkLog, override, run_window
from online_feedback import objective as objective_module
from online_feedback.first_error import CE_OBJECTIVES, FIRST_ERROR_OBJECTIVES, NEW_OBJECTIVES
from online_feedback.qualify import assert_state_equal
from online_feedback.rng import capture_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer


TRIALS = (
    ("warmup", "osd", False, None), ("O", "osd", False, None),
    ("C", "osd_ce", False, None),
    ("K_all", "osd_first_error", True, "O"),
    ("A_all", "osd_ce_first_error", True, "C"),
    ("K_natural", "osd_first_error", False, None),
    ("A_natural", "osd_ce_first_error", False, None),
)
PLAN = dict(temporary_optimizer_steps=7, response_presentations=14,
            current_training_forwards=14, backwards=14,
            detached_logit_gradient_calls=30, reference_training_forwards=0,
            detached_comparison_forwards=0, teacher_model_forwards=0,
            generations=0, main_optimizer_steps=0, captured_logit_rows_per_presentation=8)
DEFAULT_CONTROLLED = Path(__file__).resolve().parents[1] / "dflash_prefix_ranking_controlled_qualification_20261026/runner.py"
_controlled = None


def controlled(path=None, diagnostic_runner=None):
    global _controlled
    if path is not None or _controlled is None:
        spec = importlib.util.spec_from_file_location("first_error_controlled", path or DEFAULT_CONTROLLED)
        _controlled = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_controlled)
    if diagnostic_runner is not None:
        _controlled.load_helpers(diagnostic_runner)
    return _controlled


def storage_budget(weight_bytes):
    """Two 7W comparators plus one 7W trial, failure/atomic slack and reserve.

    Initial snapshots are retired after exact equality to the retained pin;
    completed comparator pairs are retired after all exact gates pass. Counting
    those initial copies anyway leaves a conservative bound for rolling writes.
    Main training starts only after every remaining qualification tensor retires.
    """
    if type(weight_bytes) is not int or weight_bytes <= 0:
        raise ValueError("Positive integral draft bytes required")
    return dict(draft_bytes=weight_bytes, rolling_trial_bytes=21 * weight_bytes,
                failure_bytes=4 * weight_bytes, atomic_overlap_bytes=3 * weight_bytes,
                work_bytes=2 * 2**30, reserve_bytes=8 * 2**30,
                required_bytes=28 * weight_bytes + 10 * 2**30,
                retained_qualification_tensors_after_success=0)


def retire_tensors(paths, root, *, reason, evidence):
    """Retire only this run's passed temporary states, with durable hash receipts."""
    root = Path(root).resolve()
    evidence = [Path(path).resolve() for path in evidence]
    if not evidence or any(not path.is_file() for path in evidence):
        raise AssertionError("Compact qualification evidence must exist before retirement")
    records = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if root not in path.parents or path.suffix != ".pt" or "failure" in path.name:
            raise AssertionError("Only run-local nonfailure temporary tensors may retire")
        if not path.is_file():
            continue
        records.append(dict(path=str(path), bytes=path.stat().st_size,
                            sha256=controlled().sha256(path), reason=reason,
                            retained_evidence=[str(item) for item in evidence],
                            retained_evidence_sha256={str(item): controlled().sha256(item) for item in evidence},
                            lost_capability="Direct replay/resume from this temporary tensor; compact exact checks retained"))
    if not records:
        return []
    receipt_path = root / "RECLAMATION.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"entries": []}
    batches = receipt.setdefault("batches", [])
    batch = dict(batch=len(batches), paths=[record["path"] for record in records],
                 free_bytes_before=shutil.disk_usage(root).free, status="planned")
    for record in records:
        record["batch"] = batch["batch"]
    batches.append(batch)
    receipt["entries"].extend(records)
    write_json(receipt_path, receipt)
    for record in records:
        Path(record["path"]).unlink()
        record["removed"] = True
    batch.update(free_bytes_after=shutil.disk_usage(root).free, status="completed")
    write_json(receipt_path, receipt)
    return records


def prepare_main(trainer, pinned_initial, root):
    """Fresh released C/K/A and full pinned RNG; qualification learning is discarded."""
    if trainer.objective not in NEW_OBJECTIVES or trainer.bundle.model.ranking_enabled:
        raise AssertionError("Main training requires fresh C/K/A without an auxiliary loss")
    if trainer.bundle.model.reference is not None:
        raise AssertionError("C/K/A must never construct a reference")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    initial = controlled().validate_fresh(trainer, controlled().load_state(pinned_initial), root)
    del initial
    return dict(exact_pinned_initial_state=True, full_rng_restored_after_construction=True,
                state_path=str(root / "initial.pt"), wall_s=time.perf_counter() - started,
                reference_created=False, inherited_warmup_state=False)


def save_failure(trainer, path):
    save_torch(path, dict(state=controlled().snapshot(trainer),
                          gradients=controlled().helpers().named_gradients(trainer)))


def first_error_oracle(proposals, labels, weights, eos_ids):
    """Independent CPU loop includes a first mismatch, then closes that block."""
    shape = labels.shape
    selected = torch.zeros_like(labels, dtype=torch.bool, device="cpu").reshape(-1, shape[-1])
    first_errors, empty, no_error = [0] * (shape[-1] - 1), 0, 0
    rows = zip(proposals.detach().cpu().reshape(-1, shape[-1]).tolist(),
               labels.detach().cpu().reshape(-1, shape[-1]).tolist(),
               weights.detach().cpu().reshape(-1, shape[-1]).tolist())
    for index, (prediction, canonical, weight) in enumerate(rows):
        count, failed = 0, False
        if canonical[0] not in eos_ids:
            for position in range(1, shape[-1]):
                if weight[position] <= 0:
                    break
                selected[index, position] = True
                count += 1
                if prediction[position] != canonical[position]:
                    first_errors[position - 1] += 1
                    failed = True
                    break
                if canonical[position] in eos_ids:
                    break
        empty += int(count == 0)
        no_error += int(count > 0 and not failed)
    return selected.reshape(shape).to(labels.device), (no_error, empty, first_errors)


def expected_statistics(proposals, labels, weights, selection, eos_ids):
    _, (no_error, empty, histogram) = first_error_oracle(proposals, labels, weights, eos_ids)
    counts = weights.new_tensor([no_error, empty, *histogram], dtype=torch.float32)
    return torch.cat((torch.stack(((weights > 0).sum().float(),
                                  ((weights > 0) & selection).sum().float(), weights.sum(),
                                  (weights * selection).sum())), counts))


def token_rows(logits, labels, teacher_probability=None):
    """Qualified native-dtype probability arithmetic; CE uses direct indexing."""
    log_probability = torch.log_softmax(logits, dim=-1)
    if teacher_probability is not None:
        return F.kl_div(log_probability, teacher_probability, reduction="none").sum(-1)
    flat = log_probability.reshape(-1, log_probability.shape[-1])
    return -flat[torch.arange(labels.numel(), device=labels.device), labels.reshape(-1)].reshape_as(labels)


def sample_indices(weights, selection, limit=8):
    """Include both selected and excluded rows when present, without random draws."""
    selected = ((weights > 0) & selection).reshape(-1)
    accepted = selected.nonzero().flatten()
    excluded = (~selected).nonzero().flatten()
    indices = torch.cat((accepted[:limit // 2], excluded[:limit // 2]))
    if len(indices) < min(limit, weights.numel()):
        unused = torch.ones(weights.numel(), dtype=torch.bool, device=weights.device)
        unused[indices] = False
        indices = torch.cat((indices, unused.nonzero().flatten()[:limit - len(indices)]))
    return indices


def check_prediction_gradients(sample, log):
    """Inspect actual prediction gradients and three bounded detached-logit VJPs."""
    if "actual_gradient" not in sample:
        raise AssertionError("The captured current logits did not receive a backward hook")
    logits, labels, weights = sample["logits"], sample["labels"], sample["weights"]
    selection, teacher = sample["selection"], sample.get("teacher_probability")
    excluded = (weights * selection) == 0
    actual = sample["actual_gradient"]
    assert_state_equal(actual[excluded], torch.zeros_like(actual[excluded]), "native_excluded_prediction_gradient")

    def gradient(selected, use_selection):
        if log.counts["detached_logit_gradient_attempted"] >= PLAN["detached_logit_gradient_calls"]:
            raise AssertionError("Detached-logit gradient budget exhausted")
        values = logits.detach().clone().requires_grad_(True)
        rows = token_rows(values, labels, teacher)
        weighted = weights * selected if use_selection else weights
        with log.work("detached_logit_gradient"):
            result, = torch.autograd.grad((rows * weighted).sum() / (weighted.sum() + 1e-6), values)
        return result

    selected_gradient = gradient(selection, True)
    full_gradient = gradient(torch.ones_like(selection), False)
    all_selected_gradient = gradient(torch.ones_like(selection), True)
    assert_state_equal(selected_gradient[excluded], torch.zeros_like(selected_gradient[excluded]),
                       "detached_excluded_prediction_gradient")
    assert_state_equal(full_gradient, all_selected_gradient, "detached_all_selected_gradient")
    if not torch.isfinite(actual).all() or not torch.isfinite(selected_gradient).all():
        raise AssertionError("Nonfinite bounded prediction gradient")
    return dict(rows=len(labels), vocabulary=logits.shape[-1], dtype=str(logits.dtype),
                actual_excluded_rows=int(excluded.sum()), actual_excluded_gradient_zero=True,
                detached_excluded_gradient_zero=True, all_selected_gradient_exact=True,
                detached_autograd_calls=3)


class FirstErrorObserver:
    """Observe original forwards once; separately count checkpoint recomputation."""
    def __init__(self, model, log, *, force_all=False):
        self.model, self.log, self.force_all = model, log, force_all
        self.observing = False
        self.chunk_terms, self.samples = [], []
        self.loss_checks, self.gradient_checks = [], []

    @contextmanager
    def installed(self):
        model, log = self.model, self.log
        original_forward, original_current, original_chunk = model.forward, model._forward_draft_blocks, model._chunk_loss
        original_mask = objective_module.first_error_mask
        handles = [model.draft_model.register_forward_pre_hook(
            lambda *args: log.counts.update(["current_draft_forwards_observed"]))]
        handles.append(model.lm_head.register_forward_pre_hook(
            lambda *args: log.counts.update(["vocabulary_head_forwards_including_recompute"])))
        for layer in model.draft_model.layers:
            handles.append(layer.register_forward_pre_hook(
                lambda *args: log.counts.update(["current_layer_forwards_including_recompute"])))

        def current(*args, **kwargs):
            if log.counts["current_training_forward_attempted"] >= PLAN["current_training_forwards"]:
                raise AssertionError("Current-forward budget exhausted")
            with log.work("current_training_forward"):
                return original_current(*args, **kwargs)

        def select(proposals, labels, weights, eos_token_ids=()):
            actual = (torch.ones_like(labels, dtype=torch.bool) if self.force_all else
                      original_mask(proposals, labels, weights, eos_token_ids))
            if self.observing:
                expected = (torch.ones_like(labels, dtype=torch.bool) if self.force_all else
                            first_error_oracle(proposals, labels, weights, eos_token_ids)[0])
                assert_state_equal(actual, expected, "independent_current_first_error_mask")
                if actual.requires_grad or actual.grad_fn is not None:
                    raise AssertionError("First-error mask must be detached")
            return actual

        def chunk(*args, **kwargs):
            log.counts["objective_chunk_calls_including_recompute"] += 1
            if not self.observing or model.objective not in NEW_OBJECTIVES:
                return original_chunk(*args, **kwargs)
            hidden, _, labels, weights, teacher_hidden = args[:5]
            captured = {}

            def capture(module, inputs, output):
                logits = output.detach().to(model.loss_dtype)
                if "predictions" not in captured:
                    predictions = logits.argmax(-1)
                    selection = (first_error_oracle(predictions, labels, weights, model.eos_token_ids)[0]
                                 if model.objective in FIRST_ERROR_OBJECTIVES and not self.force_all
                                 else torch.ones_like(labels, dtype=torch.bool))
                    captured.update(predictions=predictions, selection=selection)
                    if model.objective in CE_OBJECTIVES:
                        captured["rows"] = token_rows(logits, labels)
                    else:
                        captured["log_probability"] = F.log_softmax(logits, dim=-1)
                    if not self.samples:
                        indices = sample_indices(weights, selection)
                        sample = dict(logits=logits.reshape(-1, logits.shape[-1])[indices].clone(),
                                      labels=labels.reshape(-1)[indices].detach().clone(),
                                      weights=weights.reshape(-1)[indices].detach().clone(),
                                      selection=selection.reshape(-1)[indices].detach().clone())
                        self.samples.append(sample)
                        captured["sample_indices"] = indices
                        def prediction_gradient(gradient):
                            sample["actual_gradient"] = gradient.detach().reshape(-1, gradient.shape[-1])[indices].clone()
                        output.register_hook(prediction_gradient)
                elif "rows" not in captured:
                    probability = F.softmax(logits, dim=-1)
                    captured["rows"] = F.kl_div(captured.pop("log_probability"), probability,
                                                 reduction="none").sum(-1)
                    if "sample_indices" in captured:
                        self.samples[-1]["teacher_probability"] = probability.reshape(-1, probability.shape[-1])[
                            captured["sample_indices"]].clone()
                else:
                    raise AssertionError("Unexpected vocabulary projection in a new objective chunk")

            handle = model.lm_head.register_forward_hook(capture)
            try:
                terms = original_chunk(*args, **kwargs)
            finally:
                handle.remove()
            selection, prediction = captured["selection"], captured["predictions"]
            selected_weights = weights * selection
            expected = ((captured["rows"] * selected_weights).sum(), selected_weights.sum(),
                        ((prediction == labels) & (selected_weights > 0)).sum().float(),
                        (selected_weights > 0).sum().float(),
                        expected_statistics(prediction, labels, weights, selection, model.eos_token_ids))
            assert_state_equal(tuple(term.detach() for term in terms), expected, "independent_chunk_loss_counts")
            self.chunk_terms.append(expected)
            return terms

        def forward(*args, **kwargs):
            self.observing, self.chunk_terms = True, []
            try:
                result = original_forward(*args, **kwargs)
            finally:
                self.observing = False
            if model.objective in NEW_OBJECTIVES:
                total = self.chunk_terms[0]
                for terms in self.chunk_terms[1:]:
                    total = tuple(left + right for left, right in zip(total, terms))
                loss, accuracy, metrics = result
                denominator = total[1] + 1e-6
                assert_state_equal(loss.detach(), total[0] / denominator, "response_weighted_mean")
                assert_state_equal(accuracy, total[2] / total[3].clamp_min(1), "selected_accuracy")
                names = ("available_position_occurrences", "selected_position_occurrences",
                         "available_position_weight", "selected_position_weight", "no_error_blocks", "empty_blocks",
                         *(f"first_error_position_{offset}" for offset in range(1, model.block_size)))
                sums = dict(zip(names, total[4]))
                sums.update(main_loss_numerator=total[0], loss_denominator=denominator,
                            supervised_position_exposure=total[3])
                for name, expected in sums.items():
                    assert_state_equal(metrics["sum_metrics"][name], expected, "logging." + name)
                assert_state_equal(metrics["loss_terms"][1], denominator, "one_response_epsilon")
                check = dict(presentation=log.presentation, chunks=len(self.chunk_terms),
                             available=int(total[4][0]), selected=int(total[4][1]),
                             first_error_histogram=total[4][6:].cpu().tolist(),
                             exact_loss=True, exact_mask=True, exact_logging=True,
                             probability_dtype=str(model.loss_dtype), forced_all_selected=self.force_all)
                self.loss_checks.append(check)
                log.record("first_error_loss_check", **check)
                self.chunk_terms = []
            return result

        try:
            with override(model, "forward", forward), override(model, "_forward_draft_blocks", current), \
                    override(model, "_chunk_loss", chunk), override(objective_module, "first_error_mask", select):
                yield self
        finally:
            for handle in handles:
                handle.remove()

    def check_saved_gradients(self):
        if self.model.objective not in NEW_OBJECTIVES:
            return
        if len(self.samples) != 1:
            raise AssertionError("Exactly one bounded sample required per presentation")
        check = dict(presentation=self.log.presentation, **check_prediction_gradients(self.samples.pop(), self.log))
        self.gradient_checks.append(check)
        self.log.record("prediction_gradient_check", **check)


def common_core(core):
    """Keep scientific common metrics; extra diagnostics are checked independently."""
    ratio_names = {"loss", "objective_loss", "author_weighted_objective"}
    sum_names = {"supervised_position_exposure", "response_presentations"}
    return {**core,
            "ratio_totals": {key: value for key, value in core["ratio_totals"].items() if key in ratio_names},
            "sum_totals": {key: value for key, value in core["sum_totals"].items() if key in sum_names}}


def comparable_state(state):
    return {**state, "core": common_core(state["core"])}


def common_records(rows):
    metrics = {"loss", "objective_loss", "author_weighted_objective", "accuracy", "accuracy_denom",
               "supervised_position_exposure", "response_presentations", "grad_norm"}
    return [{**{key: value for key, value in row.items() if key not in {"training_s", "metrics"}},
             "metrics": {key: value for key, value in row["metrics"].items() if key in metrics}} for row in rows]


@contextmanager
def capture_gradients(trainer, trial_root, log, trial, baseline_root=None):
    driver = controlled()
    original_backward, original_step = trainer.backend.backward, trainer.backend.optimizer.step

    def persist(stage):
        state = dict(gradients=driver.helpers().named_gradients(trainer), rng=capture_rng(),
                     core=driver.core_state(trainer), global_step=trainer.global_step,
                     presentations=trainer.presentations,
                     parameter_layout=driver.helpers().parameter_layout(trainer))
        path = trial_root / f"{stage}.pt"
        if path.exists():
            raise AssertionError("Qualification snapshot would be overwritten")
        save_torch(path, state)
        trial["gradient_snapshots"].append(dict(stage=stage, path=str(path), bytes=path.stat().st_size))
        driver.assert_finite(state)
        if baseline_root is not None:
            driver.persist_comparison(trial_root / f"{stage}_vs_baseline.json", comparable_state(state),
                                      comparable_state(driver.load_state(baseline_root / f"{stage}.pt")))

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


def run_qualification(config, ordinary_receipt, pinned_initial, root, *,
                      controlled_runner=DEFAULT_CONTROLLED, diagnostic_runner=None):
    driver = controlled(controlled_runner, diagnostic_runner)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    ordinary = json.loads(Path(ordinary_receipt).read_text())
    if len(ordinary["features"]) != 128 or len(ordinary["records"]) != 256:
        raise ValueError("Use the complete retained 128-response, 256-presentation O event")
    pinned = driver.load_state(pinned_initial)
    required = {"draft", "optimizer", "rng", "global_step", "presentations", "accumulation_remainder", "training_modes"}
    if set(pinned) != required:
        raise ValueError("Pin must be the retained diagnostic initial.pt scientific-state schema")
    driver.assert_finite(pinned)
    budget = storage_budget(sum(value.numel() * value.element_size() for value in pinned["draft"].values()))
    budget["free_bytes_at_admission"] = shutil.disk_usage(root).free
    if budget["free_bytes_at_admission"] < budget["required_bytes"]:
        write_json(root / "STORAGE_REJECTED.json", budget)
        raise OSError("Insufficient rolling qualification storage including failure evidence and reserve")
    summary = dict(scope="first_error_OCKA", status="running", plan=PLAN, trials=[], storage=budget,
                   config=str(config), ordinary_receipt=str(ordinary_receipt), pinned_initial=str(pinned_initial),
                   input_sha256={str(path): driver.sha256(path) for path in (
                       config, ordinary_receipt, pinned_initial, controlled_runner,
                       diagnostic_runner or driver.DEFAULT_DIAGNOSTIC, Path(__file__))},
                   checks={name: False for name in QUALIFICATION_GATES}, baseline_tolerance="exact equality",
                   generations=0, main_optimizer_steps=0, main_campaign_started=False,
                   historical_O_limit="No equality claim for retained O's full historical 128-step trajectory",
                   synthetic_checks="All-selected masks only in two explicitly labeled disposable windows")
    write_json(root / "PLAN.json", summary)
    log, trainer, ordinary_rng = WorkLog(root), None, None
    started = time.perf_counter()
    try:
        for label, objective, force_all, baseline in TRIALS:
            trial_started = time.perf_counter()
            counts_before = dict(log.counts)
            trial_root = root / label
            log.trial, log.presentation = label, None
            trial = dict(label=label, objective=objective, force_all=force_all, baseline=baseline,
                         status="running", records=[], gradient_snapshots=[])
            summary["trials"].append(trial)
            write_json(root / "PROGRESS.json", summary)
            if torch.cuda.is_initialized():
                torch.cuda.reset_peak_memory_stats()
            load_started = time.perf_counter()
            trainer = SpecForgeEventTrainer(config, trial_root / "trainer", objective=objective,
                                           ranking_lambda=0.0, reduction="response_mean")
            trial["loading_s"] = time.perf_counter() - load_started
            trial["runtime"] = driver.helpers().runtime_flags()
            trial["optimizer_recipe"] = trainer.backend.optimizer.recipe
            initial = driver.validate_fresh(trainer, pinned, trial_root)
            del initial
            retire_tensors([trial_root / "initial.pt"], root, reason="Exact equality to retained initial pin passed",
                           evidence=[trial_root / "initial_vs_pinned.json", Path(pinned_initial)])
            model = trainer.bundle.model
            if model.reference is not None or model.ranking_enabled or model.loss_dtype != torch.bfloat16:
                raise AssertionError("Qualification requires BF16 probability arithmetic without reference/auxiliary")
            target_before, rng_before = driver.frozen_tables(model), capture_rng()
            model.begin_event()
            assert_state_equal(capture_rng(), rng_before, "event_begin_rng")
            if model.reference is not None:
                raise AssertionError("Unexpected qualification reference")
            observer = FirstErrorObserver(model, log, force_all=force_all)
            baseline_root = root / baseline if baseline else None
            with observer.installed(), capture_gradients(trainer, trial_root, log, trial, baseline_root), \
                    driver.check_full_shuffle(ordinary, log):
                for row in run_window(trainer, ordinary, log):
                    trial["records"].append(row)
                    observer.check_saved_gradients()
                    write_json(root / "PROGRESS.json", summary)
            final = driver.snapshot(trainer)
            save_torch(trial_root / "final.pt", final)
            driver.assert_finite(final)
            assert_state_equal(driver.frozen_tables(model), target_before, "frozen_target_tables")
            trial["loss_checks"], trial["gradient_checks"] = observer.loss_checks, observer.gradient_checks
            if objective in NEW_OBJECTIVES:
                assert_state_equal((len(observer.loss_checks), len(observer.gradient_checks)), (2, 2), "observer_coverage")
            if label == "O":
                assert_state_equal([row["metrics"] for row in trial["records"]],
                                   [row["metrics"] for row in ordinary["records"][:2]], "retained_O_first_window_metrics")
                ordinary_rng = final["rng"]
            elif ordinary_rng is not None:
                driver.persist_comparison(trial_root / "rng_vs_O.json", final["rng"], ordinary_rng)
            if baseline_root is not None:
                driver.persist_comparison(trial_root / "final_vs_baseline.json", comparable_state(final),
                                          comparable_state(driver.load_state(baseline_root / "final.pt")))
                baseline_trial = next(item for item in summary["trials"] if item["label"] == baseline)
                driver.persist_comparison(trial_root / "records_vs_baseline.json", common_records(trial["records"]),
                                          common_records(baseline_trial["records"]))
                trial["exact_baseline_compatibility"] = True
            trial["objective_work"] = model.end_event()
            if objective in CE_OBJECTIVES and trial["objective_work"]["teacher_head_s"] != 0:
                raise AssertionError("Canonical CE performed teacher projection work")
            trial["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
            trial["peak_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
            trial["work_counts"] = {name: value - counts_before.get(name, 0) for name, value in log.counts.items()}
            trial["wall_s_before_cleanup"] = time.perf_counter() - trial_started
            trial["status"] = "discarded_warmup" if label == "warmup" else "passed"
            write_json(trial_root / "TRIAL.json", trial)
            del final, observer, model
            trainer.close()
            trainer = None
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            if label not in {"O", "C"}:
                evidence = [trial_root / "TRIAL.json"]
                if baseline is not None:
                    evidence.extend(trial_root.glob("*_vs_baseline.json"))
                retire_tensors(trial_root.glob("*.pt"), root, reason="Discarded qualification trial passed all checks",
                               evidence=evidence)
            if baseline is not None:
                retire_tensors(baseline_root.glob("*.pt"), root, reason="Exact paired comparison passed; comparator no longer needed",
                               evidence=[trial_root / "TRIAL.json", trial_root / "final_vs_baseline.json",
                                         trial_root / "records_vs_baseline.json"])
            write_json(root / "PROGRESS.json", summary)
        expected = dict(current_training_forward_completed=14, current_draft_forwards_observed=14,
                        reference_training_forward_completed=0, reference_draft_forwards_observed=0,
                        detached_comparison_forward_completed=0, temporary_optimizer_step_completed=7,
                        response_presentations=14, backward_completed=14, detached_logit_gradient_completed=30)
        for name, value in expected.items():
            assert_state_equal(log.counts[name], value, "work_budget." + name)
            if name.endswith("_completed") and value:
                assert_state_equal(log.counts[name.replace("_completed", "_attempted")], value,
                                   "attempted_work_budget." + name)
        if list(root.rglob("*.pt")):
            raise AssertionError("Successful qualification left bulky temporary tensors before main training")
        summary.update(status="passed", checks={name: True for name in QUALIFICATION_GATES},
                       all_temporary_tensors_retired=True)
        return summary
    except BaseException as error:
        summary.update(status="failed", error=dict(type=type(error).__name__, message=str(error),
                                                   traceback=traceback.format_exc()))
        if summary["trials"] and (trainer is not None or trial["status"] == "running"):
            trial["status"] = "failed"
            if trainer is not None:
                try:
                    save_failure(trainer, trial_root / "failure_state.pt")
                    trial["failure_state"] = str(trial_root / "failure_state.pt")
                except BaseException as capture_error:
                    trial["failure_capture_error"] = repr(capture_error)
            write_json(trial_root / "TRIAL.json", trial)
        raise
    finally:
        summary.update(wall_s=time.perf_counter() - started, work_counts=dict(log.counts),
                       free_bytes_after=shutil.disk_usage(root).free)
        write_json(root / "RESULT.json", summary)
        if trainer is not None:
            trainer.failed = True
            trainer.close()
