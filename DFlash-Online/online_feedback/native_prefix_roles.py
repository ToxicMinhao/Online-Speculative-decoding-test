"""Current-prefix / first-error gradient controls for the bounded role screen.

The native K+ scalar objective is unchanged. E and C detach the excluded rows
before both K's numerator and K+'s redistribution numerators. These controls are
not decomposed backwards: each branch makes one ordinary model backward.
"""

import math

import torch

from .kplus import first_error_groups, redistribute_response


ROLES = {"N", "ALL", "E", "C"}


@torch.no_grad()
def role_masks(proposals, labels, weights, selection):
    """Partition the already-selected valid proposals; never reselect a block."""
    selected = selection.detach() & (weights.detach() > 0)
    correct = proposals.detach() == labels
    return selected & correct, selected & ~correct


def controlled_rows(token_loss, prefix, errors, role):
    """Preserve the forward value and control only the incoming row gradient."""
    if role in {"N", "ALL"}:
        return token_loss
    if role not in {"E", "C"}:
        raise ValueError(f"Unknown native-prefix role: {role}")
    active = errors if role == "E" else prefix
    return torch.where(active, token_loss, token_loss.detach())


class NativePrefixRoles:
    """One pair's compact evidence, with response/chunk occurrence deduplication.

    ``begin_response`` is called once by the outer forward. Chunk recomputation
    can call ``control_rows`` again, but never counts another presentation or
    occurrence. Row-gradient hooks are qualification evidence only and return
    the incoming gradient unchanged. Nothing here retains vocabulary tensors.
    """

    def __init__(self, role="N", *, capture_gradients=False):
        self.reset(role=role, capture_gradients=capture_gradients)

    def reset(self, *, role=None, capture_gradients=None):
        if role is not None:
            if role not in ROLES:
                raise ValueError(f"Unknown native-prefix role: {role}")
            self.role = role
        if capture_gradients is not None:
            self.capture_gradients = bool(capture_gradients)
        self.responses = []
        self.observed_chunk_calls = 0
        self._current = None

    def begin_response(self):
        self._current = {"role": self.role, "chunks": {}, "finished": False}
        self.responses.append(self._current)

    def control_rows(self, token_loss, proposals, labels, weights, selection,
                     eos_token_ids=(), *, chunk_key):
        prefix, errors = role_masks(proposals, labels, weights, selection)
        controlled = controlled_rows(token_loss, prefix, errors, self.role)
        self.observed_chunk_calls += 1
        response = self._current
        if response is None:
            raise RuntimeError("begin_response must precede role-controlled chunks")
        # A repeated storage offset is the same physical slice in this response.
        # Retain only the original forward's observations and gradient hook.
        if chunk_key not in response["chunks"]:
            with torch.no_grad():
                active = prefix | errors if self.role in {"N", "ALL"} else (
                    errors if self.role == "E" else prefix)
                group1, group2 = first_error_groups(
                    proposals, labels, weights, selection, eos_token_ids)
                active_weights = weights.detach() * active
                row = {
                    "shape": list(token_loss.shape), "dtype": str(token_loss.dtype),
                    "prefix_occurrences": int(prefix.sum()),
                    "error_occurrences": int(errors.sum()),
                    "prefix_weight": float((weights * prefix).sum()),
                    "error_weight": float((weights * errors).sum()),
                    "active_occurrences": int(active.sum()),
                    "active_weight": float(active_weights.sum()),
                    "_active_terms": tuple(value.cpu() for value in (
                        (token_loss.detach() * active_weights).sum(),
                        (token_loss.detach() * active_weights * group1).sum(),
                        (token_loss.detach() * active_weights * group2).sum(),
                    )),
                    "gradient": None,
                }
            response["chunks"][chunk_key] = row
            if self.capture_gradients and token_loss.requires_grad:
                # Capture masks, not mutable controller state: backward can run
                # after finish_response, and cannot inherit the next response.
                active_mask = active.detach()
                prefix_mask = prefix.detach()
                error_mask = errors.detach()

                def observe(gradient):
                    values = gradient.detach()
                    excluded = values[~active_mask]
                    row["gradient"] = {
                        "finite": bool(torch.isfinite(values).all()),
                        "active_nonzero": int(torch.count_nonzero(values[active_mask])),
                        "prefix_nonzero": int(torch.count_nonzero(values[prefix_mask])),
                        "error_nonzero": int(torch.count_nonzero(values[error_mask])),
                        "excluded_nonzero": int(torch.count_nonzero(excluded)),
                        "excluded_max_abs": float(excluded.abs().max()) if excluded.numel() else 0.,
                        "global_l2": float(torch.linalg.vector_norm(values.float())),
                    }

                token_loss.register_hook(observe)
        return controlled

    def finish_response(self, loss, denominator, native_metrics):
        """Report the enabled contribution using the original global reducer."""
        response = self._current
        chunks = list(response["chunks"].values())
        totals = chunks[0]["_active_terms"]
        for chunk in chunks[1:]:
            totals = tuple(left + right for left, right in zip(totals, chunk["_active_terms"]))
        native = {name: value.detach().cpu() for name, value in native_metrics.items()}
        active_numerator, _ = redistribute_response(
            totals[0], native["kplus_S"],
            (native["kplus_W1"], native["kplus_W2"], totals[1], totals[2],
             native["kplus_priority1_error_count"], native["kplus_priority2_error_count"]),
        )
        response.update({
            "native_full_objective": float(loss.detach()),
            "gradient_enabled_objective": float(active_numerator / denominator.detach().cpu()),
            "denominator": float(denominator.detach()),
            "finished": True,
        })
        for chunk in chunks:
            del chunk["_active_terms"]

    def receipt(self):
        """JSON-safe evidence after backward; no model-sized gradient retention."""
        responses = [{**{k: v for k, v in response.items() if k != "chunks"},
                      "chunks": list(response["chunks"].values())}
                     for response in self.responses]
        def serializable(value):
            # A failed forward can leave only the three compact active scalars.
            # Keep them and nonfinite observations explicit without saving a
            # model-sized tensor or letting a failed JSON write hide the cause.
            if torch.is_tensor(value):
                return serializable(float(value))
            if isinstance(value, float) and not math.isfinite(value):
                return repr(value)
            if isinstance(value, dict):
                return {key: serializable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(item) for item in value]
            return value

        return serializable({
            "role": self.role, "responses": responses,
            "unique_projection_chunks": sum(len(row["chunks"]) for row in self.responses),
            "observed_chunk_calls": self.observed_chunk_calls,
            "chunk_call_scope": "calls reaching role control; not all physical projection/recompute work",
        })


def install_native_prefix_roles(model, role="N", *, capture_gradients=False):
    """Attach a controller only to the existing qualified BF16 K+ objective."""
    if (model.objective != "osd_first_error" or model.loss_dtype != torch.bfloat16
            or getattr(model, "first_error_redistribution", None) is None):
        raise ValueError("Native-prefix controls require installed native BF16 K+")
    conflicts = ("role_loss_extension", "point_loss_extension", "prefix_guard",
                 "successor_shuffle", "recent_repair_retention", "survival_head_extension",
                 "adversarial_block", "native_prefix_roles")
    if any(getattr(model, name, None) is not None for name in conflicts):
        raise ValueError("Native-prefix controls cannot combine objective extensions")
    controller = NativePrefixRoles(role, capture_gradients=capture_gradients)
    model.native_prefix_roles = controller
    return controller
