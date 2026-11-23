"""Native DFlash blocks with frozen target embedding and vocabulary projection.

Reuse SpecForge's draft, anchor sampler and full-block attention. Label positions
and decay follow DFlash-Online's online_feedback/objective.py, without its modes.
"""

from dataclasses import dataclass
import json

import torch
from torch import nn
from safetensors import safe_open
from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.dflash import DFlashDraftModel

from graduate.batches import PreparedFeedback
from graduate.config import Configuration


@dataclass(frozen=True)
class DraftBlocks:
    hidden: torch.Tensor          # [1, anchors, block_size, hidden_size]
    teacher_hidden: torch.Tensor  # Same shape; each row predicts its label.
    labels: torch.Tensor          # [1, anchors, block_size], exact saved tokens
    weights: torch.Tensor         # Same shape; seed/prompt/padding=0, then decay
    anchors: torch.Tensor         # [1, anchors], positions in full prompt + response


class DraftModel(OnlineDFlashModel):
    def forward(self, feedback: PreparedFeedback) -> DraftBlocks:
        device = self.draft_model.device
        # Count adjacent supervised tokens on CPU before moving the saved batch.
        valid = (feedback.loss_mask[:, :-1] > 0.5) & (feedback.loss_mask[:, 1:] > 0.5)
        max_valid_anchors = int(valid.sum(dim=1).max().item())
        input_ids = feedback.input_ids.to(device)
        loss_mask = feedback.loss_mask.to(device)
        target_features = feedback.hidden_states.to(device)
        target_last = feedback.target_last_hidden_states.to(device)

        # This still draws native random scores when every eligible anchor fits.
        # Attention sees complete blocks; only vocabulary projection is chunked.
        anchors, keep, hidden = self._forward_draft_blocks(
            input_ids, target_features, loss_mask, max_valid_anchors,
        )
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        indices = anchors.unsqueeze(-1) + offsets
        in_bounds = indices < input_ids.shape[1]
        safe_indices = indices.clamp(max=input_ids.shape[1] - 1)
        labels = input_ids.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe_indices)
        response_mask = loss_mask.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe_indices)
        weights = keep.unsqueeze(-1).float() * in_bounds.float() * (offsets > 0).float() * response_mask
        weights = weights * torch.exp(-(offsets - 1).clamp(min=0).float() / self.loss_decay_gamma)

        # Saved target row i predicts token i+1. Shift the teacher lookup only;
        # native draft slots and canonical labels both use their own positions.
        teacher_hidden = self._aligned_target_hidden(target_last.detach(), safe_indices)
        hidden = hidden.reshape(input_ids.shape[0], anchors.shape[1], self.block_size, -1)
        return DraftBlocks(hidden, teacher_hidden, labels, weights, anchors)


def load_model(config: Configuration, device: torch.device | str) -> DraftModel:
    """Load one trainable draft and two frozen tensors; never build the 8B target."""
    draft, loading = DFlashDraftModel.from_pretrained(
        str(config.models.draft), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation=config.training.attention_backend, output_loading_info=True,
    )
    if loading["missing_keys"] or loading["unexpected_keys"]:
        raise ValueError(f"Draft checkpoint does not match the native architecture: {loading}")
    draft = draft.to(device=device, dtype=torch.bfloat16)

    # The configured Qwen3-8B checkpoint has separate embedding and head tensors.
    # Safetensors reads just these keys from their shards, not the target backbone.
    target = config.models.target
    weight_map = json.loads((target / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        with safe_open(target / weight_map[name], framework="pt", device=str(device)) as checkpoint:
            weights[name] = checkpoint.get_tensor(name).to(dtype=torch.bfloat16)
    embedding = nn.Embedding.from_pretrained(weights["model.embed_tokens.weight"], freeze=True)
    head_weight = weights["lm_head.weight"]
    # Avoid allocating/randomly initializing a second vocabulary-sized tensor.
    head = nn.Linear(head_weight.shape[1], head_weight.shape[0], bias=False,
                     device="meta", dtype=torch.bfloat16)
    head.weight = nn.Parameter(head_weight, requires_grad=False)

    model = DraftModel(
        draft_model=draft, target_lm_head=head, target_embed_tokens=embedding,
        mask_token_id=draft.mask_token_id, block_size=config.serving.block_size,
        attention_backend=config.training.attention_backend,
        num_anchors=config.training.max_anchors_per_response,
        loss_decay_gamma=config.training.loss_decay_gamma,
        objective_chunk_blocks=config.training.projection_chunk_blocks,
    )
    return model
