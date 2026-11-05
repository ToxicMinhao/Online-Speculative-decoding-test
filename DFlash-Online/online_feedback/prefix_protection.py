"""K with a separately normalized margin loss on released-DFlash prefixes.

The sampler, K selection and BF16 optimizer remain owned by the ordinary
training path. The reference is an independent frozen copy of the released
checkpoint, never a copy of the current adapted weights.
"""
import copy
from functools import partial
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from specforge.core.chunking import checkpointed_chunk_reduce

from .first_error import first_error_mask, selection_statistics
from .objective import TargetObjectiveModel, _ReferenceBlocks, _model_options, _synchronize
from .ranking import fixed_anchors, matching_prefix_mask
from .rng import preserve_rng


PROTECTION_COEFFICIENT = 0.10
REFERENCE_FRACTION = 0.50
REFERENCE_MARGIN_CAP = 1.0
RESPONSE_EPSILON = 1e-6


def canonical_margin(logits, labels):
    """Select a native-precision rival, then subtract two FP32 gathered values."""
    with torch.no_grad():
        competitors = logits.detach().clone()
        competitors.scatter_(-1, labels.unsqueeze(-1), -torch.inf)
        rivals = competitors.argmax(-1)
    desired = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1).float()
    competing = logits.gather(-1, rivals.unsqueeze(-1)).squeeze(-1).float()
    return desired - competing


def protection_terms(margins, weights, protected, thresholds):
    """Return additive hinge terms; normalization happens once per response."""
    protected_weights = weights * protected.detach()
    violation = torch.relu(thresholds.detach() - margins)
    return (protected_weights * violation).sum(), protected_weights.sum()


def combine_response_losses(k_numerator, k_weight, protection_numerator,
                            protection_weight, coefficient=PROTECTION_COEFFICIENT):
    k_loss = k_numerator / (k_weight + RESPONSE_EPSILON)
    protection_loss = protection_numerator / (protection_weight + RESPONSE_EPSILON)
    return k_loss + coefficient * protection_loss, k_loss, protection_loss


def _value_statistics(values, selected):
    """Explicitly mark extrema undefined when the selected set is empty."""
    chosen = values.detach()[selected]
    return {
        "count": chosen.numel(),
        "sum": float(chosen.sum()),
        "min": float(chosen.min()) if chosen.numel() else None,
        "max": float(chosen.max()) if chosen.numel() else None,
    }


class ReleasedReference:
    """Plain holder: reference parameters cannot enter the trainable state tree."""

    def __init__(self, current, checkpoint):
        self.checkpoint = str(Path(checkpoint).resolve())
        weight = next(current.draft_model.parameters())
        shared_modes = (current.lm_head.training, current.embed_tokens.training)
        with preserve_rng():
            # Instantiate architecture only, independently of current parameters.
            draft = type(current.draft_model)(copy.deepcopy(current.draft_model.config))
            draft = draft.to(device=weight.device, dtype=weight.dtype)
            released = load_file(str(Path(checkpoint) / "model.safetensors"))
            draft.load_state_dict(released, strict=True)
            del released
            draft.requires_grad_(False).eval()
            self.model = _ReferenceBlocks(draft_model=draft, **_model_options(current))
            self.model.training = False
        if shared_modes != (current.lm_head.training, current.embed_tokens.training):
            raise RuntimeError("Reference construction changed shared module modes")

    def hidden(self, input_ids, hidden_states, loss_mask, anchors, keep):
        self.model.fixed_anchors = (anchors, keep)
        with preserve_rng(), torch.no_grad():
            return self.model._forward_draft_blocks(input_ids, hidden_states, loss_mask)[2]


class ReleasedPrefixObjective(TargetObjectiveModel):
    """An opt-in KP objective; historical objectives and validators stay intact."""

    def __init__(self, author_model, released_checkpoint, *,
                 coefficient=PROTECTION_COEFFICIENT, eos_token_ids=(),
                 qualification=False, forced_active=False,
                 loss_precision="bfloat16"):
        if (not qualification and (coefficient != PROTECTION_COEFFICIENT
                                   or forced_active or loss_precision != "bfloat16")):
            raise ValueError("Production KP requires its fixed constants and BF16 path")
        if coefficient not in (0.0, PROTECTION_COEFFICIENT):
            raise ValueError("Qualification permits only neutral or production strength")
        super().__init__(author_model, "osd_first_error", loss_precision=loss_precision,
                         eos_token_ids=eos_token_ids)
        self.coefficient = coefficient
        self.forced_active = forced_active
        self.qualification = qualification
        self.response_observer = None
        self.last_response_record = None
        self.work = {}
        self._counter_handles = []
        weight = next(self.draft_model.parameters())
        _synchronize(weight)
        started = time.perf_counter()
        self.released_reference = ReleasedReference(self, released_checkpoint)
        _synchronize(weight)
        self.reference_setup_s = time.perf_counter() - started
        self.reference_binding = self.released_reference.checkpoint
        self._install_work_counters()
        self.begin_event()

    def _install_work_counters(self):
        def count(name):
            def observe(_module, _inputs):
                self.work[name] = self.work.get(name, 0) + 1
            return observe
        for role, draft in (("current", self.draft_model),
                            ("reference", self.released_reference.model.draft_model)):
            self._counter_handles.append(draft.register_forward_pre_hook(
                count(role + "_draft_forwards")))
            for layer in draft.layers:
                self._counter_handles.append(layer.register_forward_pre_hook(
                    count(role + "_layer_forwards")))

    def begin_event(self):
        super().begin_event()
        self.costs["reference_setup_s"] = self.reference_setup_s
        self.costs["extra_loss_s"] = 0.0
        self.work.clear()
        self.last_response_record = None

    def end_event(self):
        return {**self.costs, "work": dict(self.work),
                "reference_binding": self.reference_binding}

    def close_counters(self):
        for handle in self._counter_handles:
            handle.remove()
        self._counter_handles.clear()

    @torch.no_grad()
    def _reference_values(self, input_ids, hidden_states, loss_mask,
                          anchors, keep, labels, weights):
        with preserve_rng():
            _synchronize(hidden_states)
            started = time.perf_counter()
            hidden = self.released_reference.hidden(
                input_ids, hidden_states.detach(), loss_mask, anchors, keep,
            ).reshape(input_ids.shape[0], anchors.shape[1], self.block_size, -1)
            _synchronize(hidden)
            self.costs["reference_forward_s"] += time.perf_counter() - started
            self.work["reference_response_forwards"] = self.work.get("reference_response_forwards", 0) + 1
            prefixes, margins = [], []
            started = time.perf_counter()
            for start in range(0, anchors.shape[1], self.objective_chunk_blocks):
                stop = start + self.objective_chunk_blocks
                logits = self.lm_head(hidden[:, start:stop])
                self.work["reference_head_calls"] = self.work.get("reference_head_calls", 0) + 1
                prefixes.append(matching_prefix_mask(
                    logits.argmax(-1), labels[:, start:stop], weights[:, start:stop],
                    self.eos_token_ids,
                ))
                margins.append(canonical_margin(logits, labels[:, start:stop]))
                del logits
            protected = torch.cat(prefixes, dim=1)
            reference_margins = torch.cat(margins, dim=1)
            thresholds = REFERENCE_FRACTION * reference_margins.clamp(max=REFERENCE_MARGIN_CAP)
            _synchronize(hidden)
            self.costs["reference_head_s"] += time.perf_counter() - started
            statistics = _value_statistics(reference_margins, protected)
            statistics["zero_margin_count"] = int(((reference_margins == 0) & protected).sum())
            return protected, thresholds.detach(), statistics

    def _chunk_loss(self, *args, **kwargs):
        # Neutral qualification uses precisely the inherited K computation.
        self.work["current_objective_chunks"] = self.work.get("current_objective_chunks", 0) + 1
        self.work["current_head_calls"] = self.work.get("current_head_calls", 0) + 1
        self.work["teacher_head_calls"] = self.work.get("teacher_head_calls", 0) + 1
        return super()._chunk_loss(*args, **kwargs)

    def _protected_chunk(self, hidden, labels, weights, teacher_hidden,
                         protected, thresholds, block_indices, *, total_blocks):
        self.work["current_objective_chunks"] = self.work.get("current_objective_chunks", 0) + 1
        self.work["current_head_calls"] = self.work.get("current_head_calls", 0) + 1
        logits = self.lm_head(hidden).to(self.loss_dtype)
        with torch.no_grad():
            _synchronize(teacher_hidden)
            started = time.perf_counter()
            teacher_logits = self.lm_head(teacher_hidden).to(self.loss_dtype)
            self.work["teacher_head_calls"] = self.work.get("teacher_head_calls", 0) + 1
            _synchronize(teacher_logits)
            self.costs["teacher_head_s"] += time.perf_counter() - started
            teacher_probability = F.softmax(teacher_logits, dim=-1)
        token_loss = F.kl_div(
            F.log_softmax(logits, dim=-1), teacher_probability, reduction="none",
        ).sum(dim=-1)
        with torch.no_grad():
            _synchronize(logits)
            started = time.perf_counter()
            proposals = logits.detach().argmax(-1)
            selection = first_error_mask(proposals, labels, weights, self.eos_token_ids)
            selection_stats = selection_statistics(proposals, labels, weights,
                                                   selection, self.eos_token_ids)
            _synchronize(selection_stats)
            self.costs["selection_s"] += time.perf_counter() - started
        selected_weights = weights * selection
        eligible = selected_weights > 0
        k_numerator = (token_loss * selected_weights).sum()
        correct = ((proposals == labels) & eligible).sum().float()
        _synchronize(logits)
        started = time.perf_counter()
        try:
            margins = canonical_margin(logits, labels)
            if self.forced_active:
                protected = weights > 0
                thresholds = margins.detach() + 1.0
            protection_numerator, protection_weight = protection_terms(
                margins, weights, protected, thresholds,
            )
        finally:
            # Non-reentrant checkpoint replay can finish partway through this
            # calculation. Its actual work still belongs in the cost record.
            _synchronize(logits)
            self.costs["extra_loss_s"] += time.perf_counter() - started
        # Place detached small per-position diagnostics in disjoint block slots.
        # The ordinary additive reducer then preserves values for extrema and
        # histograms without logging anything during checkpoint recomputation.
        shape = (hidden.shape[0], total_blocks, self.block_size)
        positions = block_indices.unsqueeze(-1).expand(-1, -1, self.block_size)
        recorded_margins = margins.detach().new_zeros(shape).scatter(1, positions, margins.detach())
        recorded_proposals = proposals.new_zeros(shape).scatter(1, positions, proposals)
        return (k_numerator, selected_weights.sum(), correct, eligible.sum().float(),
                selection_stats, protection_numerator, protection_weight,
                recorded_margins, recorded_proposals)

    def _record_response(self, record):
        self.last_response_record = record
        if self.response_observer is not None:
            self.response_observer(record)

    def forward(self, input_ids, hidden_states, loss_mask, target_last_hidden_states=None,
                max_valid_anchors=None, selector_loss_alpha=None, collect_detailed_metrics=True):
        if target_last_hidden_states is None:
            raise ValueError("KP requires the retained conditioned teacher features")
        anchors, keep = self._sample_anchor_positions(
            input_ids.shape[1], loss_mask, input_ids.device,
            max_valid_anchors=max_valid_anchors,
        )
        labels, weights, safe_indices = self._block_labels_and_weights(
            input_ids, loss_mask, anchors, keep,
        )
        protected, thresholds, reference_stats = self._reference_values(
            input_ids, hidden_states, loss_mask, anchors, keep, labels, weights,
        )
        self.work["current_response_forwards"] = self.work.get("current_response_forwards", 0) + 1
        if self.coefficient == 0:
            with fixed_anchors(self, anchors, keep):
                result = super().forward(
                    input_ids, hidden_states, loss_mask, target_last_hidden_states,
                    max_valid_anchors, selector_loss_alpha, collect_detailed_metrics,
                )
            self._record_response({"neutral_reference_exercised": True,
                                   "protection_evaluated": False,
                                   "k_loss": float(result[0].detach()),
                                   "reference_margins": reference_stats})
            return result
        with fixed_anchors(self, anchors, keep):
            _, _, output_hidden = self._forward_draft_blocks(
                input_ids, hidden_states, loss_mask, max_valid_anchors,
            )
        output_hidden = output_hidden.reshape(input_ids.shape[0], anchors.shape[1], self.block_size, -1)
        teacher_hidden = self._aligned_target_hidden(target_last_hidden_states.detach(), safe_indices)
        block_indices = torch.arange(anchors.shape[1], device=anchors.device).expand(anchors.shape[0], -1)
        terms = checkpointed_chunk_reduce(
            partial(self._protected_chunk, total_blocks=anchors.shape[1]),
            output_hidden, labels, weights, teacher_hidden,
            protected, thresholds, block_indices,
            chunk_size=self.objective_chunk_blocks, dim=1, checkpoint=True,
        )
        k_numerator, k_weight, correct, positions, selection_stats = terms[:5]
        protection_numerator, protection_weight, margins, proposals = terms[5:]
        loss, k_loss, protection_loss = combine_response_losses(
            k_numerator, k_weight, protection_numerator, protection_weight, self.coefficient,
        )
        self._finish_response(loss, k_loss, k_numerator, k_weight, protection_loss,
                              protection_numerator, protection_weight, protected,
                              thresholds, margins, proposals, labels, weights,
                              selection_stats, reference_stats)
        one = loss.detach().new_ones(())
        names = ("available_position_occurrences", "selected_position_occurrences",
                 "available_position_weight", "selected_position_weight", "no_error_blocks",
                 "empty_blocks", *(f"first_error_position_{i}" for i in range(1, self.block_size)))
        metrics = {
            "loss_terms": (loss, one), "accuracy_denom": positions.detach(),
            "ratio_metrics": {
                "objective_loss": (loss.detach(), one),
                "forward_kl": (k_numerator.detach(), (k_weight + RESPONSE_EPSILON).detach()),
                "selection_rate": (selection_stats[1], selection_stats[0]),
                "protection_loss": (protection_numerator.detach(),
                                    (protection_weight + RESPONSE_EPSILON).detach()),
            },
            "sum_metrics": {**dict(zip(names, selection_stats)),
                            "supervised_position_exposure": positions.detach(),
                            "main_loss_numerator": k_numerator.detach(),
                            "loss_denominator": (k_weight + RESPONSE_EPSILON).detach(),
                            "protection_numerator": protection_numerator.detach(),
                            "protection_weight": protection_weight.detach()},
        }
        return loss, correct / positions.clamp_min(1), metrics

    def _finish_response(self, loss, k_loss, k_numerator, k_weight, protection_loss,
                         protection_numerator, protection_weight, protected, thresholds,
                         margins, proposals, labels, weights, selection_stats, reference_stats):
        with torch.no_grad():
            if self.forced_active:
                protected = weights > 0
                thresholds = margins + 1.0
            active = protected & (thresholds > margins)
            prefix_lengths = protected.sum(-1)
            eligible_blocks = (weights > 0).any(-1)
            record = {
                "protection_evaluated": True, "qualification_forced_active": self.forced_active,
                "k_loss": float(k_loss.detach()), "k_numerator": float(k_numerator.detach()),
                "k_selected_weight": float(k_weight), "k_denominator": float(k_weight + RESPONSE_EPSILON),
                "protection_numerator": float(protection_numerator.detach()),
                "protection_weight": float(protection_weight),
                "protection_denominator": float(protection_weight + RESPONSE_EPSILON),
                "protection_loss": float(protection_loss.detach()),
                "weighted_protection_loss": float(self.coefficient * protection_loss.detach()),
                "total_loss": float(loss.detach()),
                "available_blocks": int(eligible_blocks.sum()),
                "protected_blocks": int((prefix_lengths > 0).sum()),
                "empty_protected_blocks": int((eligible_blocks & (prefix_lengths == 0)).sum()),
                "protected_positions": int(protected.sum()),
                "protected_prefix_histogram_denominator": int(eligible_blocks.sum()),
                "protected_prefix_histogram": torch.bincount(prefix_lengths[eligible_blocks],
                                                              minlength=self.block_size + 1).tolist(),
                "active_positions": int(active.sum()), "active_weight": float((weights * active).sum()),
                "incorrect_protected_positions": int((protected & (proposals != labels)).sum()),
                "wrong_tie_positions": int((protected & (proposals != labels) & (margins == 0)).sum()),
                "current_zero_margin_positions": int((protected & (margins == 0)).sum()),
                "reference_margins": reference_stats,
                "current_margins": _value_statistics(margins, protected),
                "thresholds": _value_statistics(thresholds, protected),
                "selection_statistics": selection_stats.detach().cpu().tolist(),
                "finite": bool(torch.isfinite(loss) & torch.isfinite(margins).all()
                               & torch.isfinite(thresholds).all()),
            }
        self._record_response(record)
        if not record["finite"]:
            raise FloatingPointError("Nonfinite KP loss/margin/threshold; response evidence retained")
