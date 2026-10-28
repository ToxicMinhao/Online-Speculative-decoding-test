"""Five disposable full-size training windows; no serving or checkpoint export.

The original 128-response shuffle is consumed before the first two retained
presentations. Native state equality is exact, as in online_feedback.qualify.
The only extra draft forwards are two eval-mode current/reference comparisons.
"""
from collections import Counter
from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import time
import traceback
import weakref

import torch

from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.qualify import assert_state_equal
from online_feedback.ranking import allocation_coefficients, fixed_anchors, rival_softplus
from online_feedback.rng import capture_rng, preserve_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer

TRIALS = (("O", "osd", 0.), ("U0", "osd_uniform", 0.),
          ("P0", "osd_prefix", 0.), ("U1", "osd_uniform", 1.),
          ("P1", "osd_prefix", 1.))
PLAN = dict(current_training_forwards=10, reference_training_forwards=4,
            detached_comparison_forwards=4, temporary_optimizer_steps=5,
            response_presentations=10, generations=0, teacher_model_forwards=0)


def prefix_oracle(proposals, labels, weights, eos_ids):
    """Independent sequential rule, including a matching EOS and stopping at gaps."""
    shape = labels.shape
    result = torch.zeros_like(labels, dtype=torch.bool, device="cpu").reshape(-1, shape[-1])
    rows = zip(proposals.detach().cpu().reshape(-1, shape[-1]).tolist(),
               labels.detach().cpu().reshape(-1, shape[-1]).tolist(),
               weights.detach().cpu().reshape(-1, shape[-1]).tolist())
    for row, (proposed, canonical, weight) in enumerate(rows):
        if canonical[0] in eos_ids:
            continue
        for position in range(1, shape[-1]):
            if weight[position] <= 0 or proposed[position] != canonical[position]:
                break
            result[row, position] = True
            if canonical[position] in eos_ids:
                break
    return result.reshape(shape)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return value


class WorkLog:
    """Durable logical work boundaries; internal forwards are separately counted."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.counts = Counter()
        self.trial = None
        self.presentation = None

    def record(self, kind, **fields):
        with (self.root / "WORK.jsonl").open("a") as stream:
            row = dict(kind=kind, trial=self.trial, presentation=self.presentation)
            row.update(fields)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @contextmanager
    def work(self, kind):
        self.counts[kind + "_attempted"] += 1
        self.record(kind, status="started", counters=dict(self.counts))
        try:
            yield
        except BaseException as error:
            self.record(kind, status="failed", error=str(error), counters=dict(self.counts))
            raise
        self.counts[kind + "_completed"] += 1
        self.record(kind, status="completed", counters=dict(self.counts))


@contextmanager
def override(owner, name, replacement):
    previous = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield previous
    finally:
        setattr(owner, name, previous)


def check_rival_gradient(logits, labels):
    """Check actual projected BF16 values without another model/head forward."""
    values = logits.detach().clone().requires_grad_(True)
    loss = rival_softplus(values, labels)
    gradient, = torch.autograd.grad(loss.sum(), values)
    competitors = values.detach().double()
    competitors.scatter_(1, labels[:, None], -torch.inf)
    rivals = competitors.argmax(-1)
    margin = (values.detach().double().gather(1, rivals[:, None]) -
              values.detach().double().gather(1, labels[:, None])).squeeze(1)
    expected = torch.zeros_like(values)
    slope = torch.sigmoid(margin).to(values.dtype)
    expected.scatter_(1, rivals[:, None], slope[:, None])
    expected.scatter_(1, labels[:, None], -slope[:, None])
    # Existing dtype-default assertion tolerances, now explicit in the receipt.
    rtol = .016 if values.dtype == torch.bfloat16 else 1.3e-6
    torch.testing.assert_close(gradient, expected, rtol=rtol, atol=1e-5)
    if not torch.isfinite(loss).all() or not torch.isfinite(gradient).all():
        raise AssertionError("Nonfinite strongest-rival loss or gradient")
    return dict(rows=len(labels), vocabulary=values.shape[-1],
                projected_dtype=str(values.dtype), difference_dtype=str(loss.dtype),
                max_gradient_error=float((gradient - expected).abs().max()),
                gradient_rtol=rtol, gradient_atol=1e-5)


class RankingObserver:
    """Observe existing forwards; only detached, bounded logits survive a step."""
    def __init__(self, model, log):
        self.model, self.log = model, log
        self.prefix_checks = []
        self.gradient_samples = []
        self.gradient_checks = []
        self.reference_activations = []

    @contextmanager
    def installed(self):
        model, log = self.model, self.log
        original_prefix = model._reference_prefix
        original_chunk = model._chunk_loss
        original_current = model._forward_draft_blocks
        handles = []
        for layer in model.draft_model.layers:
            handles.append(layer.register_forward_pre_hook(
                lambda *args: log.counts.update(["current_layer_forwards_including_recompute"])))
        handles.append(model.lm_head.register_forward_pre_hook(
            lambda *args: log.counts.update(["vocabulary_head_forwards_including_recompute"])))

        if model.reference is not None:
            handles.append(model.reference.model.draft_model.register_forward_hook(
                lambda module, inputs, output: self.reference_activations.append(weakref.ref(output))))
            for layer in model.reference.model.draft_model.layers:
                handles.append(layer.register_forward_pre_hook(
                    lambda *args: log.counts.update(["reference_layer_forwards"])))

        def current(*args, **kwargs):
            if any(reference() is not None for reference in self.reference_activations):
                raise AssertionError("Reference activations survived into the current autograd forward")
            with log.work("current_training_forward"):
                return original_current(*args, **kwargs)

        def reference_prefix(*args):
            proposals = []
            before_rng = capture_rng()
            before_modes = {name: module.training for name, module in model.named_modules()}
            handle = model.lm_head.register_forward_hook(
                lambda module, inputs, output: proposals.append(output.detach().argmax(-1).cpu()))
            try:
                with log.work("reference_training_forward"):
                    result = original_prefix(*args)
            finally:
                handle.remove()
            assert_state_equal(capture_rng(), before_rng, "reference_forward_rng")
            assert_state_equal({name: module.training for name, module in model.named_modules()},
                               before_modes, "reference_forward_modes")
            labels, weights = args[-2:]
            expected = prefix_oracle(torch.cat(proposals, dim=1), labels, weights, model.eos_token_ids)
            different = (result.cpu() != expected).nonzero()
            log.record("prefix_comparison", mismatched_positions=len(different),
                       mismatch_indices=different[:16].tolist(),
                       actual_first_blocks=result.cpu()[:, :4].tolist(),
                       expected_first_blocks=expected[:, :4].tolist())
            assert_state_equal(result.cpu(), expected, "independent_reference_prefix")
            uniform, rho = allocation_coefficients(result, weights, "osd_uniform")
            selective, _ = allocation_coefficients(result, weights, "osd_prefix")
            torch.testing.assert_close((uniform * weights).sum(), (selective * weights).sum(),
                                       rtol=1.3e-6, atol=1e-5)
            check = dict(presentation=log.presentation, blocks=labels.shape[1],
                         protected_positions=int(result.sum()), rho=float(rho),
                         supervised_positions=int((weights > 0).sum()),
                         prefix_equal=True, reference_rng_equal=True)
            self.prefix_checks.append(check)
            log.record("prefix_check", **check)
            return result

        def chunk(*args, **kwargs):
            # Capture only the first current head call of a presentation. The
            # second head call is the frozen teacher; backward recomputation
            # cannot create another sample once this presentation is recorded.
            if (not model.ranking_enabled or self.gradient_samples or
                    self.gradient_checks and self.gradient_checks[-1]["presentation"] == log.presentation):
                return original_chunk(*args, **kwargs)
            labels = args[2]
            weights = args[3]
            def capture(module, inputs, output):
                if self.gradient_samples:
                    return
                valid = (weights.reshape(-1) > 0).nonzero().flatten()[:8]
                if len(valid):
                    self.gradient_samples.append((
                        output.detach().reshape(-1, output.shape[-1])[valid].clone(),
                        labels.detach().reshape(-1)[valid].clone()))
            handle = model.lm_head.register_forward_hook(capture)
            try:
                return original_chunk(*args, **kwargs)
            finally:
                handle.remove()

        try:
            with override(model, "_forward_draft_blocks", current), \
                 override(model, "_reference_prefix", reference_prefix), \
                 override(model, "_chunk_loss", chunk):
                yield self
        finally:
            for handle in handles:
                handle.remove()

    def check_saved_gradients(self):
        if self.gradient_samples:
            logits, labels = self.gradient_samples.pop()
            check = dict(presentation=self.log.presentation, **check_rival_gradient(logits, labels))
            self.gradient_checks.append(check)
            self.log.record("ranking_gradient_check", **check)


def compare_detached_blocks(trainer, feature_rows, retained_rows, log):
    """Exactly four eval/no-grad draft forwards on the two original anchor draws."""
    model = trainer.bundle.model
    device = next(model.draft_model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    rng = capture_rng()
    try:
        with preserve_rng(), torch.no_grad():
            model.eval()
            for feature, row in zip(feature_rows, retained_rows):
                raw = torch.load(feature["path"], map_location="cpu", weights_only=True)
                batch = trainer._batch(raw).tensors
                inputs = {key: batch[key].to(device) for key in ("input_ids", "hidden_states", "loss_mask")}
                sampled, = row["sampled_anchors"]
                anchors = torch.tensor(sampled["positions"], device=device)
                keep = torch.tensor(sampled["keep_mask"], dtype=torch.bool, device=device)
                with log.work("detached_comparison_forward"):
                    reference = model.reference.hidden(**inputs, anchors=anchors, keep=keep)
                with fixed_anchors(model, anchors, keep), log.work("detached_comparison_forward"):
                    current = model._forward_draft_blocks(**inputs)[2]
                assert_state_equal(current, reference, "detached_identical_full_blocks")
                log.record("detached_block_check", record_id=feature["record_id"],
                           anchors=len(sampled["positions"][0]), exact_hidden_equality=True)
                del current, reference, inputs, batch, raw
    finally:
        for module, training in modes:
            module.training = training
    assert_state_equal(capture_rng(), rng, "detached_comparison_rng")


def run_window(trainer, ordinary, log):
    """Use normal author RNG and trainer steps; stop at the first complete window."""
    from specforge.training.strategies.base import StepContext

    features = trainer._prepare_event("disposable_qualification", [f["path"] for f in ordinary["features"]], 2)
    schedule = PresentationSchedule(ordinary, [f["record_id"] for f in features], 2, features=features)
    trainer._presentation_schedule = schedule
    order = torch.randperm(len(features)).tolist()
    rows = []
    try:
        for index in order[:2]:
            feature = features[index]
            log.presentation = len(rows)
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
            rows.append(row)
            log.record("presentation_complete", row=row, counters=dict(log.counts))
            yield row
        assert_state_equal((trainer.global_step, trainer.presentations, trainer.core.accumulation_remainder),
                           (1, 2, 0), "disposable_window_counters")
    finally:
        trainer._presentation_schedule = None


def run_qualification(config, ordinary_receipt_path, output_root):
    """Return passed only after all five trials; preserve failure evidence and raise."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    ordinary = json.loads(Path(ordinary_receipt_path).read_text())
    if len(ordinary["features"]) != 128 or len(ordinary["records"]) != 256:
        raise ValueError("Qualification requires the complete retained 128-response event")
    log = WorkLog(root)
    summary = dict(status="running", plan=PLAN, trials=[], config=str(config),
                   ordinary_receipt=str(ordinary_receipt_path), baseline_tolerance="exact equality",
                   temporary_states_exported=False)
    write_json(root / "PLAN.json", summary)
    baseline_path = root / "disposable_O_state.pt"
    started = time.perf_counter()
    trainer = None
    try:
        for label, objective, ranking_lambda in TRIALS:
            log.trial, log.presentation = label, None
            trial = dict(label=label, objective=objective, ranking_lambda=ranking_lambda,
                         status="running", records=[])
            summary["trials"].append(trial)
            load_started = time.perf_counter()
            trainer = SpecForgeEventTrainer(config, root / label, objective=objective,
                                           ranking_lambda=ranking_lambda)
            trial["loading_s"] = time.perf_counter() - load_started
            torch.cuda.reset_peak_memory_stats()
            model = trainer.bundle.model
            before_rng = capture_rng()
            before_modes = {name: module.training for name, module in model.named_modules()}
            model.begin_event()
            assert_state_equal(capture_rng(), before_rng, "reference_setup_rng")
            assert_state_equal({name: module.training for name, module in model.named_modules()},
                               before_modes, "reference_setup_modes")
            reference_state = None
            if model.reference is not None:
                reference = model.reference.model.draft_model
                reference_state = cpu_copy(reference.state_dict())
                assert_state_equal(reference_state, trainer.bundle.draft_model.state_dict(), "reference_initial_weights")
                for name, value in reference.named_parameters():
                    current = dict(trainer.bundle.draft_model.named_parameters())[name]
                    if value.data_ptr() == current.data_ptr() or value.requires_grad:
                        raise AssertionError("Reference draft aliases current weights or requires gradients")
                if any(module.training for module in reference.modules()):
                    raise AssertionError("Reference is not in eval mode")
            elif ranking_lambda:
                raise AssertionError("Active ranking has no reference")
            if label == "U1":
                by_id = {f["record_id"]: f for f in ordinary["features"]}
                compare_detached_blocks(trainer, [by_id[r["record_id"]] for r in ordinary["records"][:2]],
                                        ordinary["records"][:2], log)
            observer = RankingObserver(model, log)
            original_step = trainer.backend.optimizer.step
            def optimizer_step():
                with log.work("temporary_optimizer_step"):
                    return original_step()
            with observer.installed(), override(trainer.backend.optimizer, "step", optimizer_step):
                for row in run_window(trainer, ordinary, log):
                    trial["records"].append(row)
                    if ranking_lambda:
                        observer.check_saved_gradients()
                    write_json(root / "PROGRESS.json", summary)
            if reference_state is not None:
                assert_state_equal(model.reference.model.draft_model.state_dict(), reference_state,
                                   "reference_after_optimizer_step")
                if any(p.grad is not None for p in model.reference.model.draft_model.parameters()):
                    raise AssertionError("Frozen reference received gradients")
                del reference_state, reference
            trial["prefix_checks"] = observer.prefix_checks
            trial["ranking_gradient_checks"] = observer.gradient_checks
            trial["objective_work"] = model.end_event()
            state = dict(draft=cpu_copy(trainer.bundle.draft_model.state_dict()),
                         optimizer=cpu_copy(trainer.backend.optimizer.state_dict()), rng=capture_rng(),
                         global_step=trainer.global_step, presentations=trainer.presentations,
                         metrics=[row["metrics"] for row in trial["records"]],
                         anchors=[row["sampled_anchors"] for row in trial["records"]])
            if label == "O":
                assert_state_equal(state["metrics"], [r["metrics"] for r in ordinary["records"][:2]],
                                   "historical_O_first_window_metrics")
                save_torch(baseline_path, state)
            elif not ranking_lambda:
                expected = torch.load(baseline_path, map_location="cpu", weights_only=False, mmap=True)
                assert_state_equal(state, expected, label + "_ordinary_state")
                trial["exact_baseline_compatibility"] = True
                del expected
            else:
                expected = torch.load(baseline_path, map_location="cpu", weights_only=False, mmap=True)
                assert_state_equal(state["rng"], expected["rng"], label + "_ordinary_rng")
                trial["reference_rng_isolation"] = True
                del expected
            del state
            trial["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            trial["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
            trial["status"] = "passed"
            trainer.close()
            trainer = None
            del model, observer, original_step, optimizer_step
            gc.collect()
            torch.cuda.empty_cache()
            write_json(root / "PROGRESS.json", summary)
        expected = dict(current_training_forward_completed=10, reference_training_forward_completed=4,
                        detached_comparison_forward_completed=4, temporary_optimizer_step_completed=5,
                        response_presentations=10)
        for key, count in expected.items():
            assert_state_equal(log.counts[key], count, "work_budget." + key)
        baseline_path.unlink()
        summary["temporary_states_discarded"] = True
        summary["status"] = "passed"
        return summary
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        if summary["trials"] and summary["trials"][-1]["status"] == "running":
            summary["trials"][-1]["status"] = "failed"
        raise
    finally:
        summary["work_counts"] = dict(log.counts)
        summary["wall_s"] = time.perf_counter() - started
        # Commit the original failure before potentially fallible NCCL cleanup.
        write_json(root / "RESULT.json", summary)
        if trainer is not None:
            trainer.failed = True
            trainer.close()
