"""Sequential two-view K+ training, with one native optimizer boundary.

Only the explicit installer enables this path. Sampling, complete draft blocks,
teacher/head projection, native chunk reduction and BF16 AdamW stay in use.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import torch

from specforge.training.controller import TrainerCore
from .adversarial_block import (
    FrozenKPlusChunk, FrozenKPlusResponse, apply_mask_delta, clean_delta,
    make_perturbation, replay_model_rng, uses_two_views,
)
from .first_error import valid_prefix_mask
from .ranking import fixed_anchors, matching_prefix_mask
from .rng import capture_rng


def _detach_output(output):
    return replace(
        output, loss=output.loss.detach(),
        loss_terms=tuple(value.detach() for value in output.loss_terms),
        metrics={key: value.detach() if isinstance(value, torch.Tensor) else value
                 for key, value in output.metrics.items()},
        ratio_metrics={key: tuple(value.detach() if isinstance(value, torch.Tensor) else value
                                 for value in pair)
                       for key, pair in output.ratio_metrics.items()},
        sum_metrics={key: value.detach() if isinstance(value, torch.Tensor) else value
                     for key, value in output.sum_metrics.items()},
    )


def _combine_outputs(clean, second):
    """One response observation, with its actual equal-view mean objective."""
    mean_loss = .5 * clean.loss + .5 * second.loss
    one = mean_loss.new_ones(())
    ratios = {key: tuple(.5 * left + .5 * right for left, right in
                        zip(clean.ratio_metrics[key], second.ratio_metrics[key]))
              for key in clean.ratio_metrics}
    ratios["author_weighted_objective"] = tuple(
        .5 * left + .5 * right for left, right in zip(clean.loss_terms, second.loss_terms))
    sums = dict(clean.sum_metrics)
    # Support/count metadata keeps its clean meaning. Current loss numerators
    # describe the two-view objective rather than masquerading as clean-only.
    for name in ("main_loss_numerator", "kplus_N_K", "kplus_N1", "kplus_N2", "kplus_N_new"):
        if name in sums:
            sums[name] = .5 * clean.sum_metrics[name] + .5 * second.sum_metrics[name]
    sums.update(response_presentations=one, abt_forward_views=2 * one,
                abt_backward_views=2 * one, abt_clean_loss=clean.loss,
                abt_second_loss=second.loss, abt_loss_change=second.loss-clean.loss)
    metrics = dict(clean.metrics)
    if "accuracy" in metrics:
        metrics["accuracy"] = .5 * clean.metrics["accuracy"] + .5 * second.metrics["accuracy"]
    return replace(clean, loss=mean_loss, loss_terms=(mean_loss, one),
                   metrics=metrics, ratio_metrics=ratios, sum_metrics=sums)


class AdversarialBlockRuntime:
    """Response-scoped hooks consumed by TargetObjectiveModel's native path."""

    def __init__(self, model, *, arm, rho):
        if (model.objective != "osd_first_error" or model.loss_dtype != torch.bfloat16
                or getattr(model, "first_error_redistribution", None) is None):
            raise ValueError("Two-view training requires installed native BF16 K+")
        if any(getattr(model, name, None) is not None for name in
               ("prefix_guard", "role_loss_extension", "point_loss_extension", "adversarial_block")):
            raise ValueError("Two-view training cannot combine other objective extensions")
        self.model, self.arm, self.rho = model, arm, rho
        self.active = False

    @contextmanager
    def response(self, batch, key):
        if self.active:
            raise RuntimeError("Nested adversarial-block response")
        self.active = True
        self.view, self.key = "clean", tuple(key)
        self.chunks, self.clean_proposals, self.second_proposals = {}, {}, {}
        self.chunk_labels = {}
        self.frozen = None
        self.delta = self.embeddings = self.perturbation = None
        self.radius_metrics = {}
        self.loss_mask = batch.tensors["loss_mask"]
        original_noise = self.model._create_noise_embed
        self.original_noise = original_noise
        self.model._create_noise_embed = self._noise
        self.model.adversarial_block = self
        try:
            yield self
        finally:
            self.model._create_noise_embed = original_noise
            del self.model.adversarial_block
            self.active = False
            # Do not retain a response graph or GPU perturbation between calls.
            self.delta = self.embeddings = self.perturbation = None
            self.chunks = self.chunk_labels = {}
            self.clean_proposals = self.second_proposals = {}
            self.frozen = self.loss_mask = None

    def _noise(self, input_ids, anchors, keep):
        if self.view == "clean":
            # Sampling has finished; any stochastic draft work comes after this.
            self.pre_clean_rng = capture_rng()
        embeddings = self.original_noise(input_ids, anchors, keep)
        shaped = embeddings.reshape(*anchors.shape, self.model.block_size, -1)
        if self.view == "clean":
            self.anchors, self.keep = anchors.detach().clone(), keep.detach().clone()
            labels, weights, _ = self.model._block_labels_and_weights(
                input_ids, self.loss_mask.to(input_ids.device), anchors, keep)
            self.eligible = valid_prefix_mask(labels, weights, self.model.eos_token_ids)
            self.embeddings = shaped.detach()
            self.delta = clean_delta(shaped)
            perturbation = self.delta
        else:
            if not torch.equal(shaped, self.embeddings):
                raise ValueError("Second-view frozen token embeddings changed")
            perturbation = self.perturbation
        return apply_mask_delta(shaped, perturbation, self.eligible).reshape_as(embeddings)

    def freeze_selection(self, proposals, labels, weights, selection):
        key = labels.storage_offset()
        if self.view == "clean" and key not in self.chunks:
            self.chunks[key] = FrozenKPlusChunk.capture(
                proposals, labels, weights, self.model.eos_token_ids, selection)
            self.chunk_labels[key] = labels.detach().clone()
            self.clean_proposals[key] = proposals.detach().clone()
        if key not in self.chunks or labels.shape != self.chunk_labels[key].shape:
            raise ValueError("Second-view objective chunk identity differs")
        if self.view == "second" and key not in self.second_proposals:
            if not torch.equal(labels, self.chunk_labels[key]) or not torch.equal(weights, self.chunks[key].weights):
                raise ValueError("Second-view labels or position weights changed")
            self.second_proposals[key] = proposals.detach().clone()
        return self.chunks[key].selection

    def group_terms(self, token_loss, labels):
        # Index by storage offset, never a consume-once counter: activation
        # checkpointing re-enters this call during each view's backward.
        chunk = self.chunks[labels.storage_offset()]
        weights1 = chunk.weights * chunk.priority1
        weights2 = chunk.weights * chunk.priority2
        return (weights1.sum(), weights2.sum(), (token_loss * weights1).sum(),
                (token_loss * weights2).sum(), chunk.priority1.sum().float(),
                chunk.priority2.sum().float())

    def freeze_response(self, terms, denominator):
        if self.view == "clean":
            self.frozen = FrozenKPlusResponse.capture(
                tuple(self.chunks[key] for key in sorted(self.chunks)), terms, denominator)
        else:
            current = (terms[1], denominator, terms[5], terms[6], terms[9], terms[10])
            frozen = (self.frozen.selected_weight, self.frozen.denominator,
                      self.frozen.weight1, self.frozen.weight2, self.frozen.count1, self.frozen.count2)
            if not all(torch.equal(left, right) for left, right in zip(current, frozen)):
                raise ValueError("Second-view frozen denominator or priority masses changed")
        return self.frozen.denominator

    def finish_clean(self):
        if self.frozen is None or self.delta is None or self.delta.grad is None:
            raise RuntimeError("Clean K+ objective did not reach its perturbation leaf")
        self.perturbation, self.radius_metrics = make_perturbation(
            self.embeddings, self.eligible, self.delta.grad,
            arm="ADV" if self.arm == "CLEAN2" else self.arm,
            key=self.key, rho=0. if self.arm == "CLEAN2" else self.rho)
        self.delta = None
        self.view = "second"

    @contextmanager
    def second_view(self):
        with fixed_anchors(self.model, self.anchors, self.keep), replay_model_rng(self.pre_clean_rng):
            yield

    @torch.no_grad()
    def diagnostics(self, clean_loss, second_loss):
        keys = sorted(self.chunks)
        labels = torch.cat([self.chunk_labels[key] for key in keys], dim=1)
        weights = torch.cat([self.chunks[key].weights for key in keys], dim=1)
        clean = torch.cat([self.clean_proposals[key] for key in keys], dim=1)
        second = torch.cat([self.second_proposals[key] for key in keys], dim=1)
        clean_prefix = matching_prefix_mask(clean, labels, weights, self.model.eos_token_ids)
        second_prefix = matching_prefix_mask(second, labels, weights, self.model.eos_token_ids)
        tensors = dict(anchors=self.anchors, keep=self.keep, labels=labels, weights=weights,
                       clean_proposals=clean, second_proposals=second,
                       clean_prefix=clean_prefix, second_prefix=second_prefix,
                       selection=torch.cat([self.chunks[key].selection for key in keys], dim=1),
                       priority1=torch.cat([self.chunks[key].priority1 for key in keys], dim=1),
                       priority2=torch.cat([self.chunks[key].priority2 for key in keys], dim=1),
                       eligible=self.eligible, denominator=self.frozen.denominator,
                       W1=self.frozen.weight1, W2=self.frozen.weight2,
                       **self.radius_metrics)
        snapshots = {name: value.detach().cpu().clone() for name, value in tensors.items()}
        snapshots["model_rng"] = deepcopy(self.pre_clean_rng)
        return dict(arm=self.arm, key=self.key, clean_loss=float(clean_loss),
                    second_loss=float(second_loss),
                    clean_prefix_positions=int(clean_prefix.sum()),
                    second_prefix_positions=int(second_prefix.sum()),
                    snapshots=snapshots)


class TwoViewTrainerCore(TrainerCore):
    """Use the native core's reductions and step with two sequential graphs."""

    def __init__(self, native_core, runtime, *, forward_loss, response_key,
                 observe_response=None, work_callback=None):
        if native_core.accumulation_steps != 2 or native_core.accumulation_remainder:
            raise ValueError("Install two-view core at a complete two-response boundary")
        super().__init__(native_core.strategy, native_core.backend, accumulation_steps=2)
        self._micro = native_core._micro
        self.runtime, self.forward_loss, self.response_key = runtime, forward_loss, response_key
        self.observe_response, self.work_callback = observe_response, work_callback

    def _work(self, kind, view, phase):
        if self.work_callback is not None:
            self.work_callback(kind, view, phase)

    def _view_forward(self, batch, ctx, view):
        self._work("forward", view, "started")
        output = self.forward_loss(batch, ctx)
        self._work("forward", view, "completed")
        return output

    def _view_backward(self, output, view, boundary):
        self._work("backward", view, "started")
        self.backend.backward(.5 * output.loss / self.accumulation_steps, is_boundary=boundary)
        self._work("backward", view, "completed")

    def train_step(self, batch, ctx=None):
        stepped = (self._micro + 1) % self.accumulation_steps == 0
        with self.runtime.response(batch, self.response_key(batch, ctx)) as runtime:
            clean = self._view_forward(batch, ctx, "clean")
            self._micro += 1
            self._view_backward(clean, "clean", False)
            clean_summary = _detach_output(clean)
            del clean
            runtime.finish_clean()
            with runtime.second_view():
                second = self._view_forward(batch, ctx, "second")
                self._view_backward(second, "second", stepped)
                second_summary = _detach_output(second)
                del second
            output = _combine_outputs(clean_summary, second_summary)
            diagnostic = runtime.diagnostics(clean_summary.loss, second_summary.loss)
            if self.observe_response is not None:
                self.observe_response(batch, output, diagnostic)

        ratios = dict(output.ratio_metrics, loss=output.loss_terms)
        self._accumulate_ratio_metrics(ratios)
        for name, value in output.sum_metrics.items():
            self._sum_totals[name] = self._sum_totals.get(name, 0) + torch.as_tensor(value).detach()
        grad_norm = None
        if stepped:
            denominator = self._normalize_gradients(
                self._ratio_totals["loss"][1],
                defer_check=bool(getattr(self.backend, "checks_loss_denominator", False)))
            grad_norm = (self.backend.step() if denominator is None
                         else self.backend.step(loss_denominator=denominator))
        result = self._result(output, grad_norm, stepped,
                              ratio_metrics=self._ratio_totals if stepped else ratios,
                              sum_metrics=self._sum_totals if stepped else output.sum_metrics)
        if stepped:
            self._ratio_totals, self._sum_totals = {}, {}
        return result


def install_adversarial_block_core(native_core, model, *, arm, forward_loss, response_key,
                                   observe_response=None, work_callback=None, rho=.01):
    """Return the original core directly for disabled or zero-radius attacks."""
    if not uses_two_views(arm, rho):
        return native_core
    runtime = AdversarialBlockRuntime(model, arm=arm, rho=rho)
    return TwoViewTrainerCore(native_core, runtime, forward_loss=forward_loss,
                              response_key=response_key, observe_response=observe_response,
                              work_callback=work_callback)
