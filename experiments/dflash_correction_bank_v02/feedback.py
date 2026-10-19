"""Full-hidden feedback for the v0.2 correction bank.

The source sampler/conditioning are inherited unchanged from the qualified
author path. Only reserved absolute output positions are removed from its loss.
Teacher head logits keep the native, full-vocabulary GEMM in bounded row chunks;
moment reduction and the mathematical objective scan vocabulary chunks, never a
second full FP32/FP64 copy of the frozen vocabulary matrix. Stored h and mu are
FP32; mathematical checks use FP64 and explicitly measure storage roundoff.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

import numpy as np

from experiments.dflash_correction_bank.feedback import sampled_features as _author_features
from experiments.dflash_per_turn.training import author


def _array(value, dtype=np.float32):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=dtype).copy()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def reserve_memory_blocks(stream, turn, native_blocks, training_positions, *, seed=17, maximum=2):
    """Select native candidate blocks without looking at gains.

    All positions here are zero-based *response-output* positions, excluding
    the target-supplied block seed. Native blocks provide ``block_start`` and
    ``output_positions``. Hash order is SHA256 of the JSON tuple
    [seed, stream, turn, block_start]. A proposed reservation is skipped when
    it would remove every sampled source-loss position. No resampling occurs.
    """
    if seed != 17 or maximum != 2 or type(turn) is not int or turn < 1:
        raise ValueError("Memory selection requires pinned seed17, at most two blocks and a positive turn")
    if not isinstance(stream, str) or not stream:
        raise ValueError("Missing stream identity")
    train = set(int(x) for x in training_positions)
    if any(x < 0 for x in train):
        raise ValueError("Training positions must be response-relative")
    candidates, starts = [], set()
    for source in native_blocks:
        start = source.get("block_start")
        positions = source.get("output_positions", [])
        if type(start) is not int or start < 0 or start in starts:
            raise ValueError("Native memory blocks require distinct nonnegative block starts")
        starts.add(start)
        if any(type(x) is not int or x <= start or x >= start + author.BLOCK_SIZE for x in positions):
            raise ValueError("Memory positions must exclude the seed and belong to this native block")
        if positions != sorted(set(positions)):
            raise ValueError("Native block positions must be sorted and unique")
        candidates.append((_digest([seed, stream, turn, start]), source, set(positions)))
    selected, excluded, skipped = [], set(), []
    for priority, block, positions in sorted(candidates, key=lambda row: (row[0], row[1]["block_start"])):
        reason = None
        if not positions:
            reason = "no_eligible_native_positions"
        elif positions & excluded:
            reason = "overlaps_selected_memory_block"
        elif not train or not (train - excluded - positions):
            reason = "would_leave_no_sampled_training_position"
        elif len(selected) >= maximum:
            reason = "two_block_limit"
        if reason:
            skipped.append({"block_start": block["block_start"], "reason": reason})
        else:
            selected.append({**copy.deepcopy(block), "selection_sha256": priority})
            excluded.update(positions)
    return {"selected": selected, "excluded_output_positions": sorted(excluded),
            "new_routing_memory_blocks": len(selected), "skipped": skipped,
            "seed": seed, "position_convention": "zero_based_response_candidates_excluding_seed",
            "selection": "sha256_json_seed_stream_turn_block_start_ascending",
            "no_new_routing_evidence": not selected,
            "training_positions_before": len(train), "training_positions_after": len(train - excluded)}


def _empty_features(response, embed_tokens, generator):
    """Bank-only zero-exposure handling; the OSD sampler is never modified."""
    import torch
    ids, output = response.get("input_ids"), response.get("output_ids")
    author.require(isinstance(ids, list) and ids and isinstance(output, list) and len(output) < 2,
                   "Invalid short response")
    author.require(all(type(x) is int and 0 <= x < embed_tokens.weight.shape[0] for x in ids + output),
                   "Invalid exact short-response IDs")
    author.require(response.get("input_token_count") == len(ids) and response.get("output_cap") == 1024,
                   "Short-response boundaries or cap differ")
    author.require(len(ids) <= author.INPUT_LIMIT and
                   len(ids) + len(output) + author.BLOCK_SIZE - 1 <= author.CONTEXT_LIMIT,
                   "Short response exceeds the context envelope")
    for key in ("record_id", "episode_id", "target_id"):
        author.require(isinstance(response.get(key), str) and response[key], "Missing " + key)
    for key in ("target_identity_sha256", "capture_checkpoint_sha256"):
        author.require(author._sha256(response.get(key)), "Invalid " + key)
    author.require(type(response.get("turn")) is int and response["turn"] >= 1, "Invalid response turn")
    for key, values in (("input_ids_sha256", ids), ("output_ids_sha256", output)):
        if key in response:
            author.require(response[key] == author.digest(values), "Short-response token hash differs")
    rng = generator.get_state().cpu().numpy().copy()
    empty = torch.empty((0, embed_tokens.weight.shape[1]), dtype=torch.float32)
    metadata = dict(record_id=response["record_id"], turn=response["turn"], response_start=len(ids),
                    response_end_exclusive=len(ids) + len(output), response_tokens=len(output),
                    anchor_positions=[], sampled_anchor_count=0, binary_loss_mask=[], position_weights=[],
                    eligible_positions=0, unique_supervised_position_ids=[], unique_supervised_positions=0,
                    weighted_position_sum=0., zero_supervision=True, backbone_forward_calls=0,
                    teacher_forward_calls=0, no_op_reason="fewer_than_two_eligible_response_anchors",
                    fixed_objective_across_candidates_and_steps=True,
                    input_ids_sha256=author.digest(ids), output_ids_sha256=author.digest(output))
    metadata["sampling_and_weights_sha256"] = author.digest(metadata)
    return dict(h=empty, teacher_h=empty.clone(), omega=np.empty(0, np.float64), metadata=metadata,
                memory_blocks=[], reservation=reserve_memory_blocks(response["episode_id"], response["turn"], [], []),
                rng_before=rng, rng_after=rng.copy(), costs={"teacher_preparation_s": 0., "frozen_feature_s": 0.})


def sampled_features(draft, embed_tokens, teacher_features, response, *, generator,
                     native_blocks=(), mask_token_id=128002, attention_backend="flex_attention",
                     synchronize=lambda: None, final_analysis=False):
    """Extract one author sample and apply the union memory mask to all overlaps.

    ``native_blocks`` contain the exact frozen native ``h`` rows from the served
    block, alongside response-relative ``output_positions``. Their teachers are
    reconstructed once from canonical full history at absolute position minus
    one. They are not substituted with sampled training-block hidden states.
    """
    for module in (draft, embed_tokens):
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError("Bank feature extraction requires frozen eval modules")
    if isinstance(response.get("output_ids"), list) and len(response["output_ids"]) < 2:
        result = _empty_features(response, embed_tokens, generator)
        result["metadata"]["analysis_only"] = bool(final_analysis)
        return result
    retained_teacher = {}

    def capture_teacher(ids):
        hidden, teacher = teacher_features(ids)
        retained_teacher["teacher"] = teacher.detach()
        return hidden, teacher

    data = _author_features(draft, embed_tokens, capture_teacher, response, generator=generator,
                            mask_token_id=mask_token_id, attention_backend=attention_backend,
                            synchronize=synchronize, final_analysis=final_analysis)
    synchronize(); reservation_started = time.perf_counter()
    original = copy.deepcopy(data["metadata"])
    labels = np.asarray(original["label_positions"], dtype=np.int64)
    old_weights = np.asarray(original["position_weights"], dtype=np.float64)
    positive_labels = labels[old_weights > 0]
    relative = positive_labels - len(response["input_ids"])
    reservation = reserve_memory_blocks(response["episode_id"], response["turn"], native_blocks, relative.tolist())
    excluded = set(reservation["excluded_output_positions"])
    keep = np.array([int(pos) not in excluded for pos in relative], dtype=bool)
    selected_weights = old_weights[old_weights > 0][keep]
    data["h"] = data["h"][keep].float().contiguous()
    data["teacher_h"] = data["teacher_h"][keep].float().contiguous()
    data["omega"] = selected_weights / selected_weights.sum() if len(selected_weights) else selected_weights
    memory_mask = np.isin(labels - len(response["input_ids"]), sorted(excluded))
    binary = np.asarray(original["binary_loss_mask"], dtype=np.float64)
    binary[memory_mask] = 0
    weights = old_weights.copy()
    weights[memory_mask] = 0
    positions = labels[weights > 0].tolist()
    if len(positive_labels) and not positions:
        raise ValueError("Memory reservation emptied a previously usable objective")
    data["metadata"].update(binary_loss_mask=binary.tolist(), position_weights=weights.tolist(),
        eligible_positions=len(positions), unique_supervised_position_ids=sorted(set(positions)),
        unique_supervised_positions=len(set(positions)), eligible_blocks=int((binary.sum(-1) > 0).sum()),
        weighted_position_sum=float(selected_weights.sum()), zero_supervision=not positions,
        original_sampled_eligible_positions=len(positive_labels),
        original_sampling_and_weights_sha256=original["sampling_and_weights_sha256"],
        original_binary_loss_mask=original["binary_loss_mask"],
        original_position_weights=original["position_weights"],
        excluded_memory_absolute_positions=[len(response["input_ids"]) + x for x in sorted(excluded)],
        memory_blocks_reserved=len(reservation["selected"]),
        memory_exclusion_applies_to_every_overlapping_example=True)
    data["metadata"]["sampling_and_weights_sha256"] = author.digest(data["metadata"])
    memory = []
    teacher = retained_teacher.pop("teacher")
    for block in reservation["selected"]:
        output_positions = block["output_positions"]
        if any(pos >= len(response["output_ids"]) for pos in output_positions):
            raise ValueError("Reserved native position lies beyond actual canonical output")
        h = _array(block["h"])
        if h.ndim != 2 or len(h) != len(output_positions) or h.shape[1] != data["h"].shape[1]:
            raise ValueError("Reserved native hidden rows do not match block metadata")
        teacher_positions = [len(response["input_ids"]) + pos - 1 for pos in output_positions]
        teacher_h = _array(teacher[0, teacher_positions])
        omega = np.exp(-(np.array(output_positions) - block["block_start"] - 1) / author.RECIPE["loss_decay_gamma"])
        omega /= omega.sum()
        memory.append(dict(h=h, teacher_h=teacher_h, omega=omega,
            native_h=_array(block.get('full_native_h',h)),
            native_row_indices=list(block.get('native_row_indices',range(len(h)))),
            token_ids=[response["output_ids"][pos] for pos in output_positions],
            output_positions=list(output_positions), block_start=block["block_start"],
            metadata={key: value for key, value in block.items() if key not in ("h","full_native_h")} | {
                "teacher_positions": teacher_positions, "native_frozen_features": True,
                "teacher_conditioning": "full_canonical_history_position_minus_one",
                "position_weights": omega.tolist(), "response_id": response["record_id"]}))
    data["memory_blocks"] = memory
    data["reservation"] = {**reservation, "selected": [m["metadata"] for m in memory]}
    synchronize()
    data["costs"]["memory_reservation_and_masking_s"] = time.perf_counter() - reservation_started
    return data


def compress_teacher_moments(teacher_h, E, *, row_chunk=32, vocab_chunk=4096, synchronize=lambda: None):
    """Compute E.T p using native teacher logits and FP64 reduction, store FP32.

    The native head is evaluated over its original full vocabulary in <=32-row
    batches, preserving the qualified GEMM shape instead of changing native
    logits by splitting its vocabulary dimension. Only moment accumulation
    casts bounded vocabulary slices; no full high-precision E is allocated.
    """
    import torch
    if not 1 <= row_chunk <= 32 or vocab_chunk < 1:
        raise ValueError("Invalid bounded compression chunks")
    features = _array(teacher_h)
    if features.ndim != 2 or features.shape[1] != E.shape[1] or not np.isfinite(features).all():
        raise ValueError("Invalid teacher hidden features")
    moments = np.empty_like(features, dtype=np.float32)
    max_roundoff = 0.
    max_native_probability_difference = 0.
    max_native_probability_mass_error = 0.
    synchronize(); started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(features), row_chunk):
            stop = min(start + row_chunk, len(features))
            th = torch.as_tensor(features[start:stop], device=E.device, dtype=E.dtype)
            logits = torch.nn.functional.linear(th, E.detach())
            probabilities = torch.softmax(logits.double(), dim=-1)
            native_probabilities = torch.softmax(logits, dim=-1).double()
            max_native_probability_difference = max(max_native_probability_difference,
                float((native_probabilities - probabilities).abs().max()))
            max_native_probability_mass_error = max(max_native_probability_mass_error,
                float((native_probabilities.sum(-1) - 1.).abs().max()))
            mu = torch.zeros((stop - start, E.shape[1]), device=E.device, dtype=torch.float64)
            for v in range(0, E.shape[0], vocab_chunk):
                mu.add_(probabilities[:, v:v + vocab_chunk] @ E[v:v + vocab_chunk].detach().double())
            stored = mu.float()
            max_roundoff = max(max_roundoff, float((stored.double() - mu).abs().max()))
            moments[start:stop] = stored.cpu().numpy()
    synchronize()
    return moments, {"compression_s": time.perf_counter() - started,
        "native_logits_dtype": str(E.dtype), "probability_dtype": "float64",
        "moment_reduction_dtype": "float64", "stored_moment_dtype": "float32",
        "moment_storage_max_abs_error": max_roundoff,
        "author_native_probability_max_abs_difference": max_native_probability_difference,
        "author_native_probability_mass_max_error": max_native_probability_mass_error,
        "probability_precision_scope": "Normalized FP64 teacher probabilities define bank mathematics; the unchanged author OSD path keeps native probability arithmetic",
        "native_head": "full_vocabulary_native_linear_in_bounded_row_batches",
        "row_chunk": row_chunk, "vocab_chunk": vocab_chunk,
        "full_high_precision_vocabulary_copy": False}


def _factor_parts(factors, width):
    if factors is None:
        return np.empty((width, 0), np.float32), np.empty((0, width), np.float32)
    if hasattr(factors, "B") and hasattr(factors, "A"):
        B, A = _array(factors.B), _array(factors.A)
        if B.ndim != 2 or A.ndim != 2 or B.shape[0] != width or A.shape[1] != width or B.shape[1] != A.shape[0]:
            raise ValueError("Correction factor shapes differ")
        return B, A
    W = np.asarray(factors, dtype=np.float64)
    if W.shape != (width, width) or not np.isfinite(W).all():
        raise ValueError("Invalid mathematical correction matrix")
    return W, np.eye(width, dtype=np.float64)


class FullHiddenObjective:
    """Exact FP64 objective for stored FP32 full-hidden teacher moments.

    ``loss`` is a change relative to zero correction, not absolute KL. The
    independent direct check quantifies FP32 moment storage error; retention
    guarantees refer to this declared stored-moment mathematical objective.
    """
    def __init__(self, h, mu, omega, E, *, teacher_h=None, row_chunk=32,
                 vocab_chunk=4096, synchronize=lambda: None, native_h=None, native_row_indices=None):
        import torch
        self.torch = torch
        self.E_native = E.detach()
        self.device = E.device
        self.features = _array(h)
        self.mu = _array(mu)
        self.weights = np.asarray(omega, dtype=np.float64).copy()
        self.teacher_h = None if teacher_h is None else _array(teacher_h)
        self.native_h = None if native_h is None else _array(native_h)
        self.native_row_indices = (None if native_row_indices is None else
                                   np.asarray(native_row_indices,dtype=np.int64).copy())
        self.row_chunk, self.vocab_chunk = int(row_chunk), int(vocab_chunk)
        self.sync, self.calls = synchronize, []
        if not 1 <= self.row_chunk <= 32 or self.vocab_chunk < 1:
            raise ValueError("Invalid bounded objective chunks")
        if self.features.ndim != 2 or self.features.shape != self.mu.shape or self.features.shape[1] != E.shape[1]:
            raise ValueError("Full-hidden objective feature/moment shapes differ")
        if self.weights.shape != (len(self.features),) or not np.isfinite(self.weights).all():
            raise ValueError("Objective weights do not align")
        if len(self.weights) and (np.any(self.weights <= 0) or not np.isclose(self.weights.sum(), 1., atol=1e-12)):
            raise ValueError("Source weights must be positive and response normalized")
        if not np.isfinite(self.features).all() or not np.isfinite(self.mu).all():
            raise ValueError("Nonfinite compressed feedback")
        if self.teacher_h is not None and self.teacher_h.shape != self.features.shape:
            raise ValueError("Direct teacher replay is not aligned")
        if self.native_h is not None:
            if (self.native_h.ndim!=2 or self.native_h.shape[1]!=E.shape[1] or
                    self.native_h.shape[0]!=author.BLOCK_SIZE-1 or
                    not np.isfinite(self.native_h).all() or
                    self.native_row_indices is None or self.native_row_indices.shape!=(len(self.features),) or
                    np.any(self.native_row_indices<0) or np.any(self.native_row_indices>=len(self.native_h)) or
                    len(set(self.native_row_indices.tolist()))!=len(self.native_row_indices) or
                    self.native_row_indices.tolist()!=sorted(self.native_row_indices.tolist()) or
                    not np.array_equal(self.native_h[self.native_row_indices],self.features)):
                raise ValueError('Native memory must retain all nine original head rows and exact eligible indices')
        elif self.native_row_indices is not None:
            raise ValueError('Native row indices require original native features')
        # These aliases are intentionally shared with the spectral core.
        self.h, self.omega = self.features, self.weights
        self.width = self.features.shape[1]
        self._baseline_loss = None

    @classmethod
    def from_teacher(cls, h, teacher_h, omega, E, **kwargs):
        compression_kwargs={key:value for key,value in kwargs.items() if key not in ('native_h','native_row_indices')}
        mu, metadata = compress_teacher_moments(teacher_h, E, **compression_kwargs)
        result = cls(h, mu, omega, E, teacher_h=teacher_h, **kwargs)
        result.compression = metadata
        return result

    @property
    def S(self):
        return self.covariance()

    def covariance(self):
        h = self.features.astype(np.float64)
        return (h.T * self.weights) @ h

    def _parts(self, factors, dtype):
        B, A = _factor_parts(factors, self.width)
        return (self.torch.as_tensor(B, device=self.device, dtype=dtype),
                self.torch.as_tensor(A, device=self.device, dtype=dtype))

    def _rows(self):
        for start in range(0, len(self.features), self.row_chunk):
            yield slice(start, min(start + self.row_chunk, len(self.features)))

    def _log_partition(self, x):
        torch = self.torch
        result = torch.full((len(x),), -torch.inf, device=self.device, dtype=torch.float64)
        for v in range(0, self.E_native.shape[0], self.vocab_chunk):
            logits = x @ self.E_native[v:v + self.vocab_chunk].double().T
            result = torch.logaddexp(result, torch.logsumexp(logits, dim=-1))
        return result

    def _evaluate(self, factors, gradient):
        torch = self.torch
        B, A = self._parts(factors, torch.float64)
        residuals = np.empty_like(self.features, dtype=np.float64) if gradient else None
        total = 0.
        with torch.no_grad():
            for sl in self._rows():
                h = torch.as_tensor(self.features[sl], device=self.device, dtype=torch.float64)
                x = h + (h @ A.T) @ B.T
                mu = torch.as_tensor(self.mu[sl], device=self.device, dtype=torch.float64)
                logZ = self._log_partition(x)
                w = torch.as_tensor(self.weights[sl], device=self.device, dtype=torch.float64)
                total += float(((logZ - (mu * x).sum(-1)) * w).sum())
                if gradient:
                    moment = torch.zeros_like(h)
                    for v in range(0, self.E_native.shape[0], self.vocab_chunk):
                        E = self.E_native[v:v + self.vocab_chunk].double()
                        probability = torch.exp(x @ E.T - logZ[:, None])
                        moment.add_(probability @ E)
                    residuals[sl] = (moment - mu).cpu().numpy()
        return total, residuals

    def _baseline(self):
        if self._baseline_loss is None:
            self._baseline_loss = self._evaluate(None, False)[0]
        return self._baseline_loss

    def loss(self, factors):
        self.sync(); started = time.perf_counter()
        value = self._evaluate(factors, False)[0] - self._baseline()
        self.sync(); self.calls.append({"kind": "source_loss", "seconds": time.perf_counter() - started})
        return value

    def loss_gradient(self, factors):
        from .core import GradientFactors
        self.sync(); started = time.perf_counter()
        value, residuals = self._evaluate(factors, True)
        value -= self._baseline()
        gradient = GradientFactors((residuals * self.weights[:, None]).T, self.features.astype(np.float64))
        self.sync(); self.calls.append({"kind": "source_loss_gradient", "seconds": time.perf_counter() - started})
        return value, gradient

    def direct_loss_gradient(self, factors):
        """Independently recompute uncompressed teacher distributions for checks."""
        if self.teacher_h is None:
            raise ValueError("Direct validation needs retained exact teacher features")
        torch = self.torch
        B, A = self._parts(factors, torch.float64)
        total = 0.
        gradient = np.zeros((self.width, self.width), dtype=np.float64)
        self.sync(); started = time.perf_counter()
        with torch.no_grad():
            for sl in self._rows():
                h = torch.as_tensor(self.features[sl], device=self.device, dtype=torch.float64)
                x = h + (h @ A.T) @ B.T
                th = torch.as_tensor(self.teacher_h[sl], device=self.device, dtype=self.E_native.dtype)
                teacher = torch.softmax(torch.nn.functional.linear(th, self.E_native).double(), dim=-1)
                z0, z1 = self._log_partition(h), self._log_partition(x)
                residual = torch.zeros_like(h)
                linear = torch.zeros(len(h), device=self.device, dtype=torch.float64)
                for v in range(0, self.E_native.shape[0], self.vocab_chunk):
                    E = self.E_native[v:v + self.vocab_chunk].double()
                    logits, original = x @ E.T, h @ E.T
                    p = teacher[:, v:v + self.vocab_chunk]
                    linear.add_((p * (original - logits)).sum(-1))
                    residual.add_((torch.exp(logits - z1[:, None]) - p) @ E)
                w = torch.as_tensor(self.weights[sl], device=self.device, dtype=torch.float64)
                total += float(((z1 - z0 + linear) * w).sum())
                gradient += ((residual * w[:, None]).T @ h).cpu().numpy()
        self.sync(); self.calls.append({"kind": "direct_qualification", "seconds": time.perf_counter() - started})
        return total, gradient

    def mathematical_margins(self, factors, token_ids):
        """Canonical-token margin against every other vocabulary entry in FP64."""
        torch = self.torch
        ids = self._ids(token_ids)
        B, A = self._parts(factors, torch.float64)
        margins = np.empty(len(ids), np.float64)
        self.sync(); started = time.perf_counter()
        with torch.no_grad():
            for sl in self._rows():
                h = torch.as_tensor(self.features[sl], device=self.device, dtype=torch.float64)
                x = h + (h @ A.T) @ B.T
                y = torch.as_tensor(ids[sl], device=self.device)
                target = (x * self.E_native[y].double()).sum(-1)
                competitor = torch.full_like(target, -torch.inf)
                for v in range(0, self.E_native.shape[0], self.vocab_chunk):
                    logits = x @ self.E_native[v:v + self.vocab_chunk].double().T
                    local = y - v
                    valid = (local >= 0) & (local < logits.shape[1])
                    rows = torch.arange(len(x), device=self.device)[valid]
                    logits[rows, local[valid]] = -torch.inf
                    competitor = torch.maximum(competitor, logits.max(-1).values)
                margins[sl] = (target - competitor).cpu().numpy()
        self.sync(); self.calls.append({"kind": "full_vocabulary_math_margins", "seconds": time.perf_counter() - started})
        return margins

    def _ids(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.shape != (len(self.features),) or np.any(ids < 0) or np.any(ids >= self.E_native.shape[0]):
            raise ValueError("Canonical IDs must align with full-hidden rows")
        return ids

    def native_evaluate(self, factors, token_ids, block_ranges=None, *, eos_token_ids=()):
        """Actual FP32 residual -> native feature cast -> native vocabulary head.

        Each half-open range addresses candidate rows excluding the block seed.
        Correct EOS is included, then terminates matching. No routing state or
        feedback is modified by this fixed-feature counterfactual evaluation.
        """
        torch = self.torch
        ids = self._ids(token_ids)
        B0, A0 = _factor_parts(factors, self.width)
        zero = not np.any(B0) or not np.any(A0)
        B, A = self._parts(factors, torch.float32)
        predictions = np.empty(len(ids), np.int64)
        margins = np.empty(len(ids), np.float64)
        ranges = [(0, len(ids))] if block_ranges is None else list(block_ranges)
        self.sync(); started = time.perf_counter()
        head_rows=[]
        with torch.no_grad():
            source_rows=[slice(0,len(ids))] if self.native_h is not None else self._rows()
            for sl in source_rows:
                original=self.native_h if self.native_h is not None else self.features[sl]
                native = torch.as_tensor(original, device=self.device, dtype=self.E_native.dtype)
                corrected = native if zero else (native.float() + (native.float() @ A.T) @ B.T).to(native.dtype)
                logits = torch.nn.functional.linear(corrected, self.E_native)
                head_rows.append(len(native))
                if self.native_h is not None:
                    selection=torch.as_tensor(self.native_row_indices,device=self.device)
                    logits=logits[selection]
                predictions[sl] = logits.argmax(-1).cpu().numpy()
                y = torch.as_tensor(ids[sl], device=self.device)
                target = logits.gather(1, y[:, None]).squeeze(1).double()
                values = logits.double()
                values.scatter_(1, y[:, None], -torch.inf)
                margins[sl] = (target - values.max(-1).values).cpu().numpy()
        prefixes = []
        for start, stop in ranges:
            if not 0 <= start <= stop <= len(ids):
                raise ValueError("Invalid half-open native block range")
            length = 0
            for index in range(start, stop):
                if predictions[index] != ids[index]:
                    break
                length += 1
                if int(ids[index]) in eos_token_ids:
                    break
            prefixes.append(length)
        self.sync(); elapsed = time.perf_counter() - started
        self.calls.append({"kind": "native_memory_evaluation", "seconds": elapsed})
        return {"predictions": predictions, "margins": margins, "prefix_lengths": prefixes,
                "correct": predictions == ids, "block_ranges": ranges, "seconds": elapsed,
                "precision": "fp32_residual_native_feature_cast_native_full_vocabulary_head",
                "native_head_row_counts":head_rows,
                "seed_included": False}

    def precision_diagnostic(self, factors):
        """Measure correction survival independently of any drafting improvement."""
        if self.teacher_h is None:
            return {"available": False, "reason": "exact_teacher_features_not_retained"}
        torch = self.torch
        B64, A64 = self._parts(factors, torch.float64)
        B32, A32 = self._parts(factors, torch.float32)
        B, A = _factor_parts(factors, self.width)
        zero = not np.any(B) or not np.any(A)
        result = dict(available=True, positions=len(self.features), changed_positions=0,
                      changed_coordinates=0, nonzero_precast_coordinates=0,
                      reference_loss_change=0., native_loss_change=0.,
                      reference_logit_delta_squared=0., native_logit_delta_squared=0.,
                      max_logit_delta_discrepancy=0., performance_gate=False)
        self.sync(); started = time.perf_counter()
        with torch.no_grad():
            for sl in self._rows():
                native = torch.as_tensor(self.features[sl], device=self.device, dtype=self.E_native.dtype)
                h = native.double()
                delta32 = (native.float() @ A32.T) @ B32.T
                corrected = native if zero else (native.float() + delta32).to(native.dtype)
                x = h + (h @ A64.T) @ B64.T
                z0n = torch.nn.functional.linear(native, self.E_native).double()
                zn = torch.nn.functional.linear(corrected, self.E_native).double()
                th = torch.as_tensor(self.teacher_h[sl], device=self.device, dtype=self.E_native.dtype)
                p = torch.softmax(torch.nn.functional.linear(th, self.E_native).double(), -1)
                w = torch.as_tensor(self.weights[sl], device=self.device, dtype=torch.float64)
                result["native_loss_change"] += float(((p * (z0n.log_softmax(-1) - zn.log_softmax(-1))).sum(-1) * w).sum())
                reference_linear = torch.zeros(len(h), device=self.device, dtype=torch.float64)
                for v in range(0, self.E_native.shape[0], self.vocab_chunk):
                    E = self.E_native[v:v + self.vocab_chunk].double()
                    delta = (x - h) @ E.T
                    native_delta = zn[:, v:v + self.vocab_chunk] - z0n[:, v:v + self.vocab_chunk]
                    reference_linear.add_((p[:, v:v + self.vocab_chunk] * delta).sum(-1))
                    result["reference_logit_delta_squared"] += float((delta.square().sum(-1) * w).sum())
                    result["native_logit_delta_squared"] += float((native_delta.square().sum(-1) * w).sum())
                    result["max_logit_delta_discrepancy"] = max(result["max_logit_delta_discrepancy"],
                                                                float((native_delta - delta).abs().max()))
                result["reference_loss_change"] += float(((self._log_partition(x) - self._log_partition(h) - reference_linear) * w).sum())
                changed = corrected != native
                result["changed_positions"] += int(changed.any(-1).sum())
                result["changed_coordinates"] += int(changed.sum())
                result["nonzero_precast_coordinates"] += int((delta32 != 0).sum())
        denominator = result["reference_logit_delta_squared"]
        result["native_to_reference_logit_delta_norm_ratio"] = (
            float(np.sqrt(result["native_logit_delta_squared"] / denominator)) if denominator > 0 else None)
        result["scope"] = "Precision survival is not improved acceptance; retention is scoped to the stored-moment mathematical objective"
        self.sync(); result["seconds"] = time.perf_counter() - started
        self.calls.append({"kind": "qualification_native_precision", "seconds": result["seconds"]})
        return result

    def portable(self):
        return {"h": self.features.copy(), "mu": self.mu.copy(), "omega": self.weights.copy(),
                "teacher_h": None if self.teacher_h is None else self.teacher_h.copy(),
                "native_h":None if self.native_h is None else self.native_h.copy(),
                "native_row_indices":None if self.native_row_indices is None else self.native_row_indices.copy(),
                "row_chunk": self.row_chunk, "vocab_chunk": self.vocab_chunk,
                "numeric_contract": "FP32 stored full hidden features/moments; FP64 mathematical evaluation; native precision separately measured"}


FrozenResponseObjective = FullHiddenObjective
