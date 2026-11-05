"""Five disposable R qualification windows; no campaign, export or generation.

Reuse the retained ordinary discarded warm-up and post-construction full RNG
pin. Exact failures retain their evidence and stop before subsequent trials.
"""
import argparse
from contextlib import contextmanager
import gc
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time
import traceback
import weakref

import torch
import torch.nn.functional as F

from experiments.prefix_ranking.qualify import (
    WorkLog, compare_detached_blocks, override, prefix_oracle, run_window,
)
from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.qualify import assert_state_equal
from online_feedback.ranking import allocation_coefficients, reference_margin_penalty, rival_softplus
from online_feedback.rng import capture_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer

TRIALS = (("warmup", "osd", 0.), ("Oa", "osd", 0.),
          ("R0", "osd_reference_margin", 0.), ("R1", "osd_reference_margin", 1.))
PLAN = dict(temporary_optimizer_steps=5, response_presentations=10,
            current_training_forwards=10, reference_training_forwards=4,
            detached_comparison_forwards=4, backwards=10,
            detached_logit_gradient_calls=20, generations=0,
            main_optimizer_steps=0, teacher_model_forwards=0)
DEFAULT_CONTROLLED = Path(__file__).resolve().parents[1] / "dflash_prefix_ranking_controlled_qualification_20261026/runner.py"
_controlled = None


def controlled(path=None, diagnostic_runner=None):
    global _controlled
    if path is not None or _controlled is None:
        spec = importlib.util.spec_from_file_location("r_controlled_qualification", path or DEFAULT_CONTROLLED)
        _controlled = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_controlled)
    if diagnostic_runner is not None:
        _controlled.load_helpers(diagnostic_runner)
    return _controlled


def storage_budget(weight_bytes):
    """Three 7W trials + one 10W R trial + failure/atomic overlap/slack/reserve."""
    if type(weight_bytes) is not int or weight_bytes <= 0:
        raise ValueError("Positive integral draft bytes required")
    return dict(draft_bytes=weight_bytes, retained_bytes=31 * weight_bytes,
                failure_bytes=4 * weight_bytes, atomic_overlap_bytes=3 * weight_bytes,
                work_bytes=2 * 2**30, reserve_bytes=8 * 2**30,
                required_bytes=38 * weight_bytes + 10 * 2**30)


def prepare_main(trainer, pinned_initial_path, output_root):
    """Validate a fresh R runtime and restore the qualified full RNG; no warm-up."""
    if trainer.objective != "osd_reference_margin" or trainer.ranking_lambda != 1.0:
        raise AssertionError("The main runtime must be fresh R with lambda one")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    initial = controlled().validate_fresh(trainer, controlled().load_state(pinned_initial_path), root)
    del initial
    return dict(exact_pinned_initial_state=True, full_rng_restored_after_construction=True,
                state_path=str(root / "initial.pt"), wall_s=time.perf_counter() - started,
                reference_created=False, inherited_warmup_state=False)


def failure_snapshot(trainer):
    return dict(state=controlled().snapshot(trainer),
                gradients=controlled().helpers().named_gradients(trainer))


def save_failure(trainer, path):
    """Persist the caller's failed scientific state before runtime cleanup."""
    save_torch(path, failure_snapshot(trainer))


@contextmanager
def capture_r_gradients(trainer, trial_root, log, trial):
    """Keep both R1 pre-optimizer snapshots under distinct, permanent names."""
    driver = controlled()
    original_backward, original_step = trainer.backend.backward, trainer.backend.optimizer.step

    def persist(stage):
        path = trial_root / (stage + ".pt")
        if path.exists():
            raise AssertionError("Qualification snapshot would be overwritten")
        state = dict(gradients=driver.helpers().named_gradients(trainer), rng=capture_rng(),
                     core=driver.core_state(trainer), global_step=trainer.global_step,
                     presentations=trainer.presentations,
                     parameter_layout=driver.helpers().parameter_layout(trainer))
        save_torch(path, state)
        row = dict(stage=stage, path=str(path), bytes=path.stat().st_size)
        trial["gradient_snapshots"].append(row)
        log.record("gradient_snapshot", **row)
        driver.assert_finite(state)

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
        persist(f"pre_optimizer_step_{trainer.global_step + 1}")
        with log.work("temporary_optimizer_step"):
            return original_step(*args, **kwargs)

    with override(trainer.backend, "backward", backward), override(trainer.backend.optimizer, "step", step):
        yield


def run_two_windows(trainer, ordinary, log):
    """Extend the existing first-window loop to four rows of its one full draw."""
    from specforge.training.strategies.base import StepContext

    features = trainer._prepare_event("disposable_qualification", [row["path"] for row in ordinary["features"]], 2)
    schedule = PresentationSchedule(ordinary, [row["record_id"] for row in features], 2, features=features)
    trainer._presentation_schedule = schedule
    order = torch.randperm(len(features)).tolist()
    try:
        for presentation, index in enumerate(order[:4]):
            feature = features[index]
            log.presentation = presentation
            schedule.begin_presentation(0, feature["record_id"])
            raw = torch.load(feature["path"], map_location="cpu", weights_only=True)
            trainer._anchor_observations.clear()
            started = time.perf_counter()
            result = trainer.core.train_step(trainer._batch(raw), StepContext(
                global_step=trainer.global_step, total_steps=trainer.horizon, collect_detailed_metrics=True))
            schedule.finish_presentation()
            metrics = result.materialize_metrics()
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise AssertionError("Nonfinite qualification metric")
            trainer.presentations += 1
            trainer.global_step += int(result.optimizer_stepped)
            log.counts["response_presentations"] += 1
            row = dict(record_id=feature["record_id"], **{"pass": 0}, metrics=metrics,
                       sampled_anchors=list(trainer._anchor_observations),
                       optimizer_stepped=result.optimizer_stepped,
                       optimizer_step_after=trainer.global_step, training_s=time.perf_counter() - started)
            log.record("presentation_complete", row=row, counters=dict(log.counts))
            yield row
        assert_state_equal((trainer.global_step, trainer.presentations, trainer.core.accumulation_remainder),
                           (2, 4, 0), "R_two_window_counters")
    finally:
        trainer._presentation_schedule = None


def softplus_oracle(logits, labels):
    """Independent max-value oracle, retaining no vocabulary tensor in its result."""
    competing = logits.detach().clone()
    competing.scatter_(-1, labels.unsqueeze(-1), -torch.inf)
    strongest = competing.max(-1).values.float()
    desired = logits.detach().gather(-1, labels.unsqueeze(-1)).squeeze(-1).float()
    return F.softplus(strongest - desired)


def check_reference_gradients(logits, labels, reference_values, log):
    """Five bounded detached-logit autograd calls; no model/head calls or updates."""
    def gradients(loss, inputs, retain_graph=False):
        if log.counts["detached_logit_gradient_attempted"] >= PLAN["detached_logit_gradient_calls"]:
            raise AssertionError("Detached-logit gradient budget exhausted")
        with log.work("detached_logit_gradient"):
            return torch.autograd.grad(loss, inputs, retain_graph=retain_graph, allow_unused=True)

    values = logits.detach().clone().requires_grad_(True)
    reference = reference_values.detach().clone().requires_grad_(True)
    current = rival_softplus(values, labels)
    assert_state_equal(current.detach(), softplus_oracle(values, labels), "current_own_rival")
    penalty = reference_margin_penalty(current, reference)
    actual, reference_gradient = gradients(penalty.sum(), (values, reference), retain_graph=True)
    original_p, = gradients(current.sum(), (values,))
    active = current.detach() > reference.detach()
    expected = torch.where(active.unsqueeze(-1), original_p, torch.zeros_like(original_p))
    assert_state_equal(actual, expected, "R_active_exact_P_gradient")
    if reference_gradient is not None:
        raise AssertionError("Reference received an auxiliary gradient")

    equal_values = logits.detach().clone().requires_grad_(True)
    equal_current = rival_softplus(equal_values, labels)
    equal_penalty = reference_margin_penalty(equal_current, equal_current.detach())
    equal_gradient, = gradients(equal_penalty.sum(), (equal_values,))
    assert_state_equal(equal_penalty.detach(), torch.zeros_like(equal_penalty), "equal_margin_value")
    assert_state_equal(equal_gradient, torch.zeros_like(equal_values), "equal_margin_gradient")

    # These declared synthetic logits exercise changed rival IDs, a worsened
    # margin, equality and improvement even if this initial window has no active R.
    fixture = logits.new_tensor([[4., 3., -2.], [4., -3., 3.75],
                                 [3., 2., 1.], [4., 0., 2.]]).requires_grad_(True)
    fixture_reference = logits.new_tensor([[4., -3., 3.5], [4., 3., -2.],
                                           [3., 2., 1.], [3., 2., 1.]])
    fixture_labels = labels.new_zeros(4)
    fixture_current = rival_softplus(fixture, fixture_labels)
    fixture_reference_values = rival_softplus(fixture_reference, fixture_labels)
    assert_state_equal(fixture_reference_values, softplus_oracle(fixture_reference, fixture_labels),
                       "synthetic_reference_own_rival")
    fixture_penalty = reference_margin_penalty(fixture_current, fixture_reference_values)
    fixture_gradient, = gradients(fixture_penalty.sum(), (fixture,), retain_graph=True)
    fixture_p, = gradients(fixture_current.sum(), (fixture,))
    expected_active = torch.tensor([False, True, False, False], device=logits.device)
    assert_state_equal(fixture_penalty > 0, expected_active, "synthetic_changed_rival_activation")
    assert_state_equal(fixture_gradient, torch.where(expected_active[:, None], fixture_p,
                                                    torch.zeros_like(fixture_p)), "synthetic_R_gradient")
    if not torch.isfinite(actual).all() or not torch.isfinite(penalty).all():
        raise AssertionError("Nonfinite reference-margin gradient or value")
    return dict(rows=len(labels), vocabulary=logits.shape[-1],
                projected_dtype=str(logits.dtype), difference_dtype=str(current.dtype),
                observed_active_rows=int(active.sum()), exact_original_P_gradient=True,
                reference_detached=True, equality_value_and_gradient_zero=True,
                synthetic_rows=4, synthetic_changed_rivals_checked=True,
                detached_autograd_calls=5, tolerance="exact equality")


class ReferenceMarginObserver:
    """Observe existing forwards once; detached samples never retain activations."""
    def __init__(self, model, log):
        self.model, self.log = model, log
        self.prefix_checks, self.gradient_checks = [], []
        self.gradient_samples, self.reference_activations = [], []
        self.observing = False
        self.chunk_terms = []
        self.expected_prefix = None

    @contextmanager
    def installed(self):
        model, log = self.model, self.log
        original_forward, original_current = model.forward, model._forward_draft_blocks
        original_reference, original_chunk = model._reference_feedback, model._chunk_loss
        handles = []
        handles.append(model.draft_model.register_forward_pre_hook(
            lambda *args: log.counts.update(["current_draft_forwards_observed"])))
        for layer in model.draft_model.layers:
            handles.append(layer.register_forward_pre_hook(
                lambda *args: log.counts.update(["current_layer_forwards_including_recompute"])))
        handles.append(model.lm_head.register_forward_pre_hook(
            lambda *args: log.counts.update(["vocabulary_head_forwards_including_recompute"])))
        if model.reference is not None:
            handles.append(model.reference.model.draft_model.register_forward_pre_hook(
                lambda *args: log.counts.update(["reference_draft_forwards_observed"])))
            handles.append(model.reference.model.draft_model.register_forward_hook(
                lambda module, inputs, output: self.reference_activations.append(weakref.ref(output))))
            for layer in model.reference.model.draft_model.layers:
                handles.append(layer.register_forward_pre_hook(
                    lambda *args: log.counts.update(["reference_layer_forwards"])))

        def current(*args, **kwargs):
            if any(value() is not None for value in self.reference_activations):
                raise AssertionError("Reference activations survived into current autograd")
            if log.counts["current_training_forward_attempted"] >= PLAN["current_training_forwards"]:
                raise AssertionError("Current-forward budget exhausted")
            with log.work("current_training_forward"):
                return original_current(*args, **kwargs)

        def reference(*args):
            labels, weights = args[-2:]
            proposals, expected_values = [], []
            before_rng = capture_rng()
            before_modes = {name: module.training for name, module in model.named_modules()}
            reference_forwards_before = log.counts["reference_draft_forwards_observed"]
            offset = 0

            def capture(module, inputs, output):
                nonlocal offset
                width = output.shape[1]
                proposals.append(output.detach().argmax(-1))
                expected_values.append(softplus_oracle(output, labels[:, offset:offset + width]))
                offset += width

            if log.counts["reference_training_forward_attempted"] >= PLAN["reference_training_forwards"]:
                raise AssertionError("Reference-forward budget exhausted")
            handle = model.lm_head.register_forward_hook(capture)
            try:
                with log.work("reference_training_forward"):
                    result = original_reference(*args)
            finally:
                handle.remove()
            actual_prefix, reference_values, reference_correct = result
            predicted = torch.cat(proposals, dim=1)
            expected_prefix = prefix_oracle(predicted, labels, weights, model.eos_token_ids).to(labels.device)
            assert_state_equal(actual_prefix, expected_prefix, "independent_reference_prefix")
            assert_state_equal(reference_values, torch.cat(expected_values, dim=1), "reference_own_rival_values")
            assert_state_equal(reference_correct, (predicted == labels) & (weights > 0), "reference_correct_flags")
            assert_state_equal(log.counts["reference_draft_forwards_observed"] - reference_forwards_before,
                               1, "one_existing_reference_draft_forward")
            assert_state_equal(capture_rng(), before_rng, "reference_forward_rng")
            assert_state_equal({name: module.training for name, module in model.named_modules()},
                               before_modes, "reference_forward_modes")
            if reference_values.dtype != torch.float32 or reference_values.requires_grad:
                raise AssertionError("Reference softplus must be detached FP32")
            selective, rho = allocation_coefficients(actual_prefix, weights, "osd_reference_margin")
            expected_selective, expected_rho = allocation_coefficients(actual_prefix, weights, "osd_prefix")
            assert_state_equal((selective, rho), (expected_selective, expected_rho), "R_uses_P_allocation")
            self.expected_prefix = expected_prefix
            check = dict(presentation=log.presentation, blocks=labels.shape[1],
                         protected_positions=int(expected_prefix.sum()),
                         supervised_positions=int((weights > 0).sum()),
                         reference_correct_positions=int(reference_correct.sum()),
                         prefix_equal=True, reference_own_rivals_equal=True,
                         reference_rng_equal=True, reference_head_chunks=len(proposals))
            self.prefix_checks.append(check)
            log.record("reference_margin_prefix_check", **check)
            return result

        def chunk(*args, **kwargs):
            if not self.observing or not model.ranking_enabled:
                return original_chunk(*args, **kwargs)
            hidden, _, labels, weights, teacher, coefficients, reference_values, reference_correct = args
            captured = {}

            def capture(module, inputs, output):
                if "current" not in captured:
                    logits = output.detach().to(model.loss_dtype)
                    captured["current"] = softplus_oracle(logits, labels)
                    captured["correct"] = logits.argmax(-1) == labels
                    captured["log_probs"] = F.log_softmax(logits, dim=-1)
                    if not self.gradient_samples:
                        valid = (weights.reshape(-1) > 0).nonzero().flatten()[:8]
                        if len(valid):
                            self.gradient_samples.append((output.detach().reshape(-1, output.shape[-1])[valid].clone(),
                                                          labels.reshape(-1)[valid].detach().clone(),
                                                          reference_values.reshape(-1)[valid].detach().clone()))
                elif "kl" not in captured:
                    probability = F.softmax(output.detach().to(model.loss_dtype), dim=-1)
                    captured["kl"] = F.kl_div(captured.pop("log_probs"), probability, reduction="none").sum(-1)
                else:
                    raise AssertionError("Unexpected extra head forward in an R chunk")

            handle = model.lm_head.register_forward_hook(capture)
            try:
                terms = original_chunk(*args, **kwargs)
            finally:
                handle.remove()
            with torch.no_grad():
                difference = captured["current"] - reference_values
                penalty = torch.where(difference > 0, difference, torch.zeros_like(difference))
                kl_sum = (captured["kl"] * weights).sum()
                auxiliary = (weights * coefficients * penalty).sum()
                eligible = weights > 0
                reference_correct = reference_correct & eligible
                expected = (kl_sum + model.ranking_lambda * auxiliary, weights.sum(),
                            (captured["correct"] & eligible).sum().float(), eligible.sum().float(),
                            kl_sum, auxiliary,
                            ((coefficients > 0) & eligible & (penalty > 0)).sum().float(),
                            (reference_correct & ~captured["correct"]).sum().float(),
                            reference_correct.sum().float())
                assert_state_equal(tuple(value.detach() for value in terms), expected, "independent_R_chunk_terms")
                self.chunk_terms.append(expected)
            return terms

        def forward(*args, **kwargs):
            self.observing, self.chunk_terms = True, []
            try:
                result = original_forward(*args, **kwargs)
            finally:
                self.observing = False
            if model.ranking_enabled:
                totals = self.chunk_terms[0]
                for terms in self.chunk_terms[1:]:
                    totals = tuple(left + right for left, right in zip(totals, terms))
                loss, _, metrics = result
                denominator = totals[1] + 1e-6
                assert_state_equal(loss.detach(), totals[0] / denominator, "R_response_denominator")
                expected_sums = dict(kl_loss_numerator=totals[4],
                                     auxiliary_loss_numerator=model.ranking_lambda * totals[5],
                                     loss_denominator=denominator, protected_active_positions=totals[6],
                                     protected_positions=self.expected_prefix.sum().float(),
                                     reference_correct_positions=totals[8],
                                     reference_correct_current_incorrect_positions=totals[7])
                for name, expected in expected_sums.items():
                    assert_state_equal(metrics["sum_metrics"][name], expected, "R_logging." + name)
                expected_ratios = dict(forward_kl=(totals[4], denominator),
                                       auxiliary_loss=(model.ranking_lambda * totals[5], denominator),
                                       protected_activation_rate=(totals[6], expected_sums["protected_positions"]),
                                       reference_correct_to_incorrect_rate=(totals[7], totals[8]))
                for name, expected in expected_ratios.items():
                    assert_state_equal(metrics["ratio_metrics"][name], expected, "R_logging." + name)
                log.record("reference_margin_loss_logging_check", chunks=len(self.chunk_terms),
                           exact_chunk_terms=True, exact_denominator=True, exact_logging=True)
                self.chunk_terms = []
            return result

        try:
            with override(model, "forward", forward), override(model, "_forward_draft_blocks", current), \
                    override(model, "_reference_feedback", reference), override(model, "_chunk_loss", chunk):
                yield self
        finally:
            for handle in handles:
                handle.remove()

    def check_saved_gradients(self):
        if len(self.gradient_samples) != 1:
            raise AssertionError("One bounded actual-logit sample required per active presentation")
        logits, labels, reference_values = self.gradient_samples.pop()
        check = dict(presentation=self.log.presentation,
                     **check_reference_gradients(logits, labels, reference_values, self.log))
        self.gradient_checks.append(check)
        self.log.record("reference_margin_gradient_check", **check)


def run_qualification(config, ordinary_receipt_path, pinned_initial_path, output_root, *,
                      controlled_runner=DEFAULT_CONTROLLED, diagnostic_runner=None):
    driver = controlled(controlled_runner, diagnostic_runner)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    ordinary = json.loads(Path(ordinary_receipt_path).read_text())
    if len(ordinary["features"]) != 128 or len(ordinary["records"]) != 256:
        raise ValueError("Use the complete retained 128-response, 256-presentation O event")
    pinned = driver.load_state(pinned_initial_path)
    required = {"draft", "optimizer", "rng", "global_step", "presentations", "accumulation_remainder", "training_modes"}
    if set(pinned) != required:
        raise ValueError("Pin must be the retained diagnostic initial.pt scientific-state schema")
    driver.assert_finite(pinned)
    budget = storage_budget(sum(value.numel() * value.element_size() for value in pinned["draft"].values()))
    budget["free_bytes_at_admission"] = shutil.disk_usage(root).free
    if budget["free_bytes_at_admission"] < budget["required_bytes"]:
        write_json(root / "STORAGE_REJECTED.json", budget)
        raise OSError("Insufficient peak storage including failure evidence and reserve")
    summary = dict(status="running", plan=PLAN, trials=[], storage=budget,
                   config=str(config), ordinary_receipt=str(ordinary_receipt_path),
                   pinned_initial=str(pinned_initial_path),
                   input_sha256={str(path): driver.sha256(path) for path in (
                       config, ordinary_receipt_path, pinned_initial_path, controlled_runner,
                       diagnostic_runner or driver.DEFAULT_DIAGNOSTIC, Path(__file__))},
                   baseline_tolerance="exact equality", generations=0, main_optimizer_steps=0,
                   historical_O_limit="No equivalence claim for retained O's entire 128-step trajectory",
                   synthetic_checks="Detached logits only; no synthetic model training or criterion on gains")
    write_json(root / "PLAN.json", summary)
    log, trainer = WorkLog(root), None
    started = time.perf_counter()
    try:
        # Reuse gradient persistence while enforcing the smaller R work budget.
        with override(driver, "PLAN", PLAN):
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
                trial["runtime"] = driver.helpers().runtime_flags()
                trial["optimizer_recipe"] = trainer.backend.optimizer.recipe
                initial = driver.validate_fresh(trainer, pinned, trial_root)
                del initial
                baseline_root = root / "Oa" if label == "R0" else None
                if baseline_root is not None:
                    driver.persist_comparison(trial_root / "initial_vs_Oa.json", driver.load_state(trial_root / "initial.pt"),
                                              driver.load_state(baseline_root / "initial.pt"))
                model = trainer.bundle.model
                target_before = driver.frozen_tables(model)
                before_rng = capture_rng()
                before_modes = {name: module.training for name, module in model.named_modules()}
                model.begin_event()
                assert_state_equal(capture_rng(), before_rng, "reference_setup_rng")
                assert_state_equal({name: module.training for name, module in model.named_modules()}, before_modes,
                                   "reference_setup_modes")
                if ranking_lambda:
                    if model.reference is None or not model.ranking_enabled:
                        raise AssertionError("Active R reference is absent")
                    driver.validate_reference(trainer, pinned)
                    by_id = {row["record_id"]: row for row in ordinary["features"]}
                    compare_detached_blocks(trainer, [by_id[row["record_id"]] for row in ordinary["records"][:2]],
                                            ordinary["records"][:2], log)
                elif model.reference is not None or model.ranking_enabled:
                    raise AssertionError("Lambda zero entered the reference/ranking branch")
                observer = ReferenceMarginObserver(model, log)
                capture = (capture_r_gradients(trainer, trial_root, log, trial) if label == "R1"
                           else driver.capture_gradients(trainer, trial_root, log, trial, baseline_root))
                window = run_two_windows if label == "R1" else run_window
                with observer.installed(), capture, \
                        driver.check_full_shuffle(ordinary, log):
                    for row in window(trainer, ordinary, log):
                        trial["records"].append(row)
                        write_json(root / "PROGRESS.json", summary)
                        if ranking_lambda:
                            observer.check_saved_gradients()
                            if len(trial["records"]) == 2:
                                driver.persist_comparison(trial_root / "first_window_rng_vs_Oa.json", capture_rng(),
                                                          driver.load_state(root / "Oa/final.pt")["rng"])
                final = driver.snapshot(trainer)
                save_torch(trial_root / "final.pt", final)
                trial["prefix_checks"], trial["ranking_gradient_checks"] = observer.prefix_checks, observer.gradient_checks
                trial["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
                trial["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
                write_json(trial_root / "TRIAL.json", trial)
                driver.assert_finite(final)
                assert_state_equal(driver.frozen_tables(model), target_before, "target_tables_version_layout")
                trial["frozen_target_version_layout_unchanged"] = True
                if label == "Oa":
                    assert_state_equal([row["metrics"] for row in trial["records"]],
                                       [row["metrics"] for row in ordinary["records"][:2]], "historical_first_window_metrics")
                if baseline_root is not None:
                    driver.persist_comparison(trial_root / "final_vs_Oa.json", final,
                                              driver.load_state(baseline_root / "final.pt"))
                    oa = next(item for item in summary["trials"] if item["label"] == "Oa")
                    strip = lambda rows: [{key: value for key, value in row.items() if key != "training_s"} for row in rows]
                    driver.persist_comparison(trial_root / "records_vs_Oa.json", strip(trial["records"]), strip(oa["records"]))
                    trial["exact_baseline_compatibility"] = True
                if ranking_lambda:
                    driver.validate_reference(trainer, pinned)
                    assert_state_equal((len(observer.prefix_checks), len(observer.gradient_checks)), (4, 4), "R_observer_coverage")
                    trial["reference_rng_isolation"] = True
                    trial["post_update_presentations"] = 2
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
        expected = dict(current_training_forward_completed=10, reference_training_forward_completed=4,
                        current_draft_forwards_observed=10, reference_draft_forwards_observed=4,
                        detached_comparison_forward_completed=4, temporary_optimizer_step_completed=5,
                        response_presentations=10, backward_completed=10, detached_logit_gradient_completed=20)
        for key, value in expected.items():
            assert_state_equal(log.counts[key], value, "work_budget." + key)
        summary.update(status="passed", main_campaign_started=False)
        return summary
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        if summary["trials"]:
            trial["status"] = "failed"
            if trainer is not None:
                try:
                    save_torch(trial_root / "failure_state.pt", failure_snapshot(trainer))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ordinary-receipt", required=True)
    parser.add_argument("--pinned-initial", required=True)
    parser.add_argument("--controlled-runner", type=Path, default=DEFAULT_CONTROLLED)
    parser.add_argument("--diagnostic-runner", type=Path)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_qualification(args.config, args.ordinary_receipt, args.pinned_initial, args.output,
                      controlled_runner=args.controlled_runner, diagnostic_runner=args.diagnostic_runner)


if __name__ == "__main__":
    main()
