"""Load full-response feedback. Sampling training blocks belongs to model.py."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from graduate.config import Configuration
from graduate.feedback import ResponseRecord


@dataclass(frozen=True)
class FeatureLayout:
    context_limit: int
    vocab_size: int
    target_hidden_size: int
    target_layer_ids: tuple[int, ...]

    @property
    def draft_feature_width(self):
        return self.target_hidden_size * len(self.target_layer_ids)


def read_feature_layout(config: Configuration) -> FeatureLayout:
    """Bind imported feature dimensions to the configured target/draft pair."""
    target = json.loads((config.models.target / "config.json").read_text())
    draft = json.loads((config.models.draft / "config.json").read_text())
    for field in ("hidden_size", "vocab_size"):
        if target[field] != draft[field]:
            raise ValueError(f"Target and draft disagree on {field}.")
    if draft["block_size"] != config.serving.block_size:
        raise ValueError("Configured block_size differs from the released draft.")
    target_layers = draft["dflash_config"]["target_layer_ids"]
    if not target_layers or any(layer < 0 or layer >= target["num_hidden_layers"] for layer in target_layers):
        raise ValueError("Draft target-layer selection lies outside the target model.")
    return FeatureLayout(config.training.context_limit, target["vocab_size"],
                         target["hidden_size"], tuple(target_layers))


@dataclass(frozen=True)
class PreparedFeedback:
    """Keep feature rows unshifted; the training model applies predecessor indexing."""

    record_id: str
    input_ids: torch.Tensor                  # [1, length], int64
    loss_mask: torch.Tensor                  # [1, length], prompt=0, response=1
    hidden_states: torch.Tensor              # [1, length, selected_layers * hidden_size]
    target_last_hidden_states: torch.Tensor  # [1, length, hidden_size]
    stop_token_ids: tuple[int, ...]


def load_feedback(record: ResponseRecord, path: Path,
                  layout: FeatureLayout) -> PreparedFeedback:
    """Read and check imported features once; keep canonical rows unshifted."""
    # Missing files/keys already raise at their source; no wrapper is needed.
    prepared = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    tokens = list(record.input_ids)
    length = len(tokens)
    prompt_length = len(record.prompt_ids)
    if not record.eligible_anchor_count:
        raise ValueError("Response has no eligible adjacent tokens; keep it as evidence, do not replace it.")
    if length > layout.context_limit:
        raise ValueError("Complete feedback exceeds the context limit; truncation is forbidden.")
    if max(tokens) >= layout.vocab_size:
        raise ValueError("Feedback contains a token outside the model vocabulary.")

    token_ids = prepared["input_ids"]
    loss_mask = prepared["loss_mask"]
    if (prepared["record_id"] != record.record_id or prepared["prompt_length"] != prompt_length
            or token_ids.dtype != torch.int64 or token_ids.shape != (1, length)
            or token_ids.tolist() != [tokens]):
        raise ValueError("Prepared token IDs or identity differ from the saved response.")
    # A wrong mask can broadcast successfully and train on prompt tokens.
    if (loss_mask.dtype != torch.float32 or loss_mask.shape != (1, length)
            or not (loss_mask[:, :prompt_length] == 0).all()
            or not (loss_mask[:, prompt_length:] == 1).all()):
        raise ValueError("Loss mask must supervise exactly the complete response, including EOS.")

    # Imported packets retain the native verifier's token/layer binding. These
    # small metadata checks do not run again inside the later training loop.
    binding = prepared["label_binding"]
    expected_binding = {
        "record_id": record.record_id, "prompt_length": prompt_length,
        "input_ids": tokens, "target_argmax_ids": list(record.response_ids),
        "target_layer_ids": list(layout.target_layer_ids), "label_mismatches": [],
        "label_binding_scope": "native_prefill_and_committed_verification",
        "native_capture_validated": True, "response_only_loss": True,
        "final_feature_logits_used_for_labels": False,
    }
    # Local capture and the retained SGLang packets share alignment, not kernels.
    if binding["capture_backend"] not in (
        "same_frozen_SGLang_native_capture", "sglang_live_capture",
    ):
        raise ValueError("Unrecognized native feature capture backend.")
    for name, expected in expected_binding.items():
        actual = binding[name]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Prepared native binding differs at {name}.")
    for name in ("input_ids", "target_argmax_ids", "target_layer_ids"):
        if any(type(token) is not int for token in binding[name]):
            raise ValueError(f"Native {name} must contain integer IDs.")

    for name, width in (("hidden_states", layout.draft_feature_width),
                        ("target_last_hidden_states", layout.target_hidden_size)):
        features = prepared[name]
        if (features.shape != (1, length, width) or features.dtype != torch.bfloat16
                or features.device.type != "cpu" or features.requires_grad):
            raise ValueError(f"{name} must be detached CPU BF16 features of shape {(1, length, width)}.")
        # NaNs do not raise by themselves and would poison all later updates.
        if not torch.isfinite(features).all():
            raise ValueError(f"{name} contains nonfinite target features.")

    return PreparedFeedback(record.record_id, token_ids, loss_mask, prepared["hidden_states"],
                            prepared["target_last_hidden_states"], record.stop_token_ids)
