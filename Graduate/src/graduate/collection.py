"""Collect canonical target rows while native greedy generation is running.

The prefix/label alignment follows DFlash-Online's native capture. Generation,
tokenization and target forwards belong to the runtime; this collector observes
their results and never computes a teacher replay.
"""

import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import torch

from graduate.feedback import ResponseRecord


class GenerationRequest(BaseModel):
    """The runtime renders this text once and records the resulting exact IDs."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    record_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    role: Literal["feedback", "evaluation"] = "feedback"


def read_requests(path: Path) -> list[GenerationRequest]:
    """Reject ambiguous request identities and feedback/evaluation prompt overlap."""
    requests = [GenerationRequest.model_validate(row) for row in json.loads(Path(path).read_text())]
    if not requests or len({request.record_id for request in requests}) != len(requests):
        raise ValueError("Request input must be nonempty and contain distinct record IDs.")
    roles_by_prompt = {}
    for request in requests:
        normalized = " ".join(request.prompt.split()).casefold()
        if not normalized:
            raise ValueError("Request prompts must contain text.")
        previous_role = roles_by_prompt.setdefault(normalized, request.role)
        if previous_role != request.role:
            raise ValueError("A normalized prompt appears in both feedback and evaluation inputs.")
    return requests


def _token_ids(values):
    tokens = list(values)
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Capture requires nonempty exact nonnegative integer token IDs.")
    return tokens


class FeedbackCapture:
    """Retain [rows, width] features, keeping only the committed causal prefix."""

    def __init__(self, prompt_ids, target_layer_ids, backend: str):
        self.prompt_ids = _token_ids(prompt_ids)
        self.target_layer_ids = _token_ids(target_layer_ids)
        if len(set(self.target_layer_ids)) != len(self.target_layer_ids) or not backend:
            raise ValueError("Capture requires distinct target layers and an explicit backend.")
        self.backend = backend
        self.response_ids = []
        self._covered_ids = []
        self._auxiliary = []
        self._final_hidden = []
        self._complete = False
        self._clipped = False

    def _append(self, tokens, auxiliary, final_hidden):
        rows = len(tokens)
        if (auxiliary.ndim != 2 or final_hidden.ndim != 2
                or auxiliary.shape[0] != rows or final_hidden.shape[0] != rows
                or auxiliary.shape[1] != final_hidden.shape[1] * len(self.target_layer_ids)
                or final_hidden.shape[1] == 0
                or auxiliary.dtype != torch.bfloat16 or final_hidden.dtype != torch.bfloat16):
            raise ValueError("Captured rows must be BF16 auxiliary/final feature pairs with the selected layer widths.")
        if self._final_hidden and final_hidden.shape[1] != self._final_hidden[0].shape[1]:
            raise ValueError("Target feature width changed within one response.")
        # Own CPU storage before a later runtime call can reuse its output buffers.
        auxiliary = auxiliary.detach().to("cpu", copy=True).contiguous()
        final_hidden = final_hidden.detach().to("cpu", copy=True).contiguous()
        if not torch.isfinite(auxiliary).all() or not torch.isfinite(final_hidden).all():
            raise ValueError("Committed target features contain nonfinite values.")
        self._auxiliary.append(auxiliary)
        self._final_hidden.append(final_hidden)
        self._covered_ids.extend(tokens)

    def prefill(self, first_token: int, auxiliary, final_hidden):
        """Capture the complete prompt and its first target-produced output token."""
        if self.response_ids or self._covered_ids:
            raise ValueError("Prefill must occur exactly once, before verification.")
        first = _token_ids([first_token])
        self._append(self.prompt_ids, auxiliary, final_hidden)
        self.response_ids.extend(first)

    def verify(self, candidates, posterior, committed_progress: int, auxiliary, final_hidden):
        """Observe a native verify call after its progress has been clipped at EOS/cap."""
        if not self.response_ids or self._complete or self._clipped:
            raise ValueError("Verification requires an unfinished prefilled response.")
        candidates, posterior = _token_ids(candidates), _token_ids(posterior)
        width = len(candidates)
        if (len(posterior) != width or candidates[0] != self.response_ids[-1]
                or type(committed_progress) is not int or not 1 <= committed_progress <= width
                or auxiliary.ndim != 2 or final_hidden.ndim != 2
                or auxiliary.shape[0] != width or final_hidden.shape[0] != width):
            raise ValueError("Verification input geometry or anchor differs from native progress.")
        accepted_end = next((slot for slot in range(1, width)
                             if candidates[slot] != posterior[slot - 1]), width)
        if committed_progress > accepted_end:
            raise ValueError("Verification attempted to retain a rejected candidate suffix.")
        # Each posterior row predicts the token after its input row. Only this
        # prefix has the canonical history; rejected suffix logits are trace only.
        committed = posterior[:committed_progress]
        self._append(candidates[:committed_progress], auxiliary[:committed_progress],
                     final_hidden[:committed_progress])
        self.response_ids.extend(committed)
        self._clipped = committed_progress < accepted_end

    def final_token(self, token_id: int, auxiliary, final_hidden):
        """Append the final emitted token's feature; it supplies no response label."""
        token = _token_ids([token_id])
        if (not self.response_ids or self._complete or token != self.response_ids[-1:]
                or self._covered_ids != self.prompt_ids + self.response_ids[:-1]):
            raise ValueError("Final feature must complete the exact canonical response once.")
        self._append(token, auxiliary, final_hidden)
        self._complete = True

    def save(self, record: ResponseRecord, path: Path):
        """Persist a complete immutable packet in the format consumed by Step4."""
        if (not self._complete or record.purpose != "serving"
                or list(record.prompt_ids) != self.prompt_ids
                or list(record.response_ids) != self.response_ids
                or self._covered_ids != list(record.input_ids)):
            raise ValueError("Saved response and completed native capture differ.")
        tokens = self.prompt_ids + self.response_ids
        binding = {
            "record_id": record.record_id, "prompt_length": len(self.prompt_ids),
            "input_ids": tokens, "target_argmax_ids": list(self.response_ids),
            "target_layer_ids": list(self.target_layer_ids), "label_mismatches": [],
            "capture_backend": self.backend, "native_capture_validated": True,
            "label_binding_scope": "native_prefill_and_committed_verification",
            "final_feature_logits_used_for_labels": False, "response_only_loss": True,
        }
        input_ids = torch.tensor([tokens], dtype=torch.int64)
        loss_mask = torch.zeros(input_ids.shape, dtype=torch.float32)
        loss_mask[:, len(self.prompt_ids):] = 1
        packet = {
            "input_ids": input_ids, "loss_mask": loss_mask,
            "hidden_states": torch.cat(self._auxiliary).unsqueeze(0),
            "target_last_hidden_states": torch.cat(self._final_hidden).unsqueeze(0),
            "record_id": record.record_id, "prompt_length": len(self.prompt_ids),
            "label_binding": binding,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Link only the completed temporary file; an existing packet is never replaced.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                torch.save(packet, stream)
                stream.flush()
                os.fsync(stream.fileno())
                os.link(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return binding
