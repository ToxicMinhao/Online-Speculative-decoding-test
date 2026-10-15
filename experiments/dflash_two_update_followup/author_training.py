"""AdaFlash OSD training path under our session protocol.

Faithful sampler/attention/KL port of AdaFlash 507b7aba5fbccaa99a400f161e4f2794ba6a388c,
``specforge/core/dflash.py``; see AUTHOR_PATH.md for adaptations and attribution.
Torch stays lazy so metadata auditing does not require the training environment.
The frozen teacher callback receives *all* causal prompt and committed response
IDs, with no truncation. Its two results are concatenated selected-layer features
and normalized final-layer features, both [1, sequence, width].
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

AUTHOR_COMMIT = "507b7aba5fbccaa99a400f161e4f2794ba6a388c"
ARM_NAME = "AdaFlash OSD training path under our session protocol"
BLOCK_SIZE, INPUT_LIMIT, CONTEXT_LIMIT, OUTPUT_CAP = 10, 31744, 32768, 256
RECIPE = dict(seed=17, learning_rate=3e-4, betas=(0.9, 0.999), eps=1e-8,
              weight_decay=0.01, gradient_clip=1.0, passes=2,
              responses_per_step=2, optimizer_steps_per_event=8,
              num_anchors=512, loss_decay_gamma=7.0, teacher_temperature=1.0,
              hard_label_alpha=0.0, loss="forward_KL",
              normalization="weighted_slot_sum_divided_by_weight_sum_plus_1e-6",
              padding="nine_masked_right_tail_positions", block_size=BLOCK_SIZE)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)



def _has_adaptive_head(draft):
    return any(bool(getattr(draft, name, False)) for name in (
        "use_prob_head", "use_thresh_head", "use_thresh_head_subsequent", "use_thresh_head_two_model"))


def audit_response(response):
    """Preflight exact IDs and ownership before teacher work or optimizer changes.

    Author preprocessing pads the sequence before sampling; retaining nine masked
    trailing positions preserves every response anchor without a 32K padded
    teacher replay. The author's sampler requires at least two eligible anchors.
    One-token/empty responses therefore stop with a record-specific diagnostic,
    instead of being silently dropped or changing the eight-step event budget.
    """
    require(isinstance(response.get("record_id"), str) and response["record_id"], "Missing response identity")
    require(type(response.get("turn")) is int, "Invalid response turn")
    for key in ("episode_id", "target_id"):
        require(isinstance(response.get(key), str) and response[key], "Missing " + key)
    for key in ("target_identity_sha256", "capture_checkpoint_sha256"):
        require(_sha256(response.get(key)), "Missing/invalid " + key)
    ids, output = response.get("input_ids"), response.get("output_ids")
    require(isinstance(ids, list) and ids and isinstance(output, list) and
            all(type(i) is int and i >= 0 for i in ids + output), "Invalid exact input/output token IDs")
    require(type(response.get("input_token_count")) is int and response["input_token_count"] == len(ids),
            "Input boundary does not match exact full causal prompt")
    require(type(response.get("output_cap")) is int and response["output_cap"] == OUTPUT_CAP and
            len(output) <= OUTPUT_CAP, "Output cap differs from qualified 256-or-EOS protocol")
    require(len(ids) <= INPUT_LIMIT and len(ids) + len(output) + BLOCK_SIZE - 1 <= CONTEXT_LIMIT,
            "Full causal history exceeds context envelope; truncation forbidden")
    for key, values in (("input_ids_sha256", ids), ("output_ids_sha256", output)):
        if key in response:
            require(response[key] == digest(values), "Token hash differs: " + key)
    require(len(output) >= 2,
            f"Author sampler cannot train {response['record_id']}: {len(output)} eligible response anchors; "
            "requires >=2 (num_anchors=min(512,valid_count-1)); no response was skipped")
    return dict(record_id=response["record_id"], turn=response["turn"],
                response_start=len(ids), response_end_exclusive=len(ids) + len(output),
                real_sequence_length=len(ids) + len(output),
                padded_sequence_length=len(ids) + len(output) + BLOCK_SIZE - 1,
                right_padding_positions=BLOCK_SIZE - 1, eligible_anchor_count=len(output),
                sampled_anchor_count=min(RECIPE["num_anchors"], len(output) - 1),
                response_tokens=len(output), input_ids_sha256=digest(ids), output_ids_sha256=digest(output),
                sequence_sha256=digest(ids + output),
                loss_mask_sha256=digest([0] * len(ids) + [1] * len(output) + [0] * (BLOCK_SIZE - 1)))


def event_schedule(event, responses):
    require(type(event) is int and event in (1, 2), "Only update events 1 and 2 exist")
    require(len(responses) == 8 and [r.get("turn") for r in responses] == list(range(8 * event + 1, 8 * event + 9)),
            "Event requires its eight new responses in chronological order")
    require(len({r.get("record_id") for r in responses}) == 8, "Duplicate response identity")
    return [(pass_index, responses[offset:offset + 2]) for pass_index in range(2) for offset in range(0, 8, 2)]


def sample_anchor_positions(seq_len, loss_mask, *, generator=None, num_anchors=512, block_size=BLOCK_SIZE):
    """Exact author random-sort sampler, including valid_count minus one.

    A device-local generator isolates sampling from serving RNG consumption while
    advancing continuously across both chronological passes and both updates.
    Pass None for exact comparison against the author's global-RNG function.
    """
    import torch
    device, bsz = loss_mask.device, loss_mask.shape[0]
    max_anchor = max(seq_len - block_size, 0)
    valid = loss_mask[:, :max_anchor + 1] > 0.5
    valid_counts = valid.sum(dim=1)
    max_n = min(num_anchors, int(valid_counts.max().item()) - 1)
    require(max_n > 0, "Author anchor sampler requires >=2 eligible anchors")
    indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
    masked_indices = torch.where(valid, indices, torch.tensor(seq_len + 1, device=device))
    random_vals = torch.rand(bsz, max_anchor + 1, device=device, generator=generator)
    random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))
    _, sorted_idx = random_vals.sort(dim=1)
    anchors = torch.gather(masked_indices, 1, sorted_idx)[:, :max_n].sort(dim=1).values
    keep = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=max_n)
    return torch.where(keep, anchors, torch.tensor(0, dtype=torch.long, device=device)), keep


def create_attention_mask(anchors, keep, sequence_length, *, backend="sdpa", block_size=BLOCK_SIZE):
    """Author mask: context strictly before anchor; own draft block bidirectional."""
    import torch
    bsz, count = anchors.shape
    if backend == "flex_attention":
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(b, h, q, kv):
            block = q // block_size
            safe_block = block.clamp(max=count - 1)
            context = (kv < sequence_length) & (kv < anchors[b, safe_block])
            own_draft = (kv >= sequence_length) & (block == (kv - sequence_length) // block_size)
            return (context | own_draft) & keep[b, safe_block] & (block < count)

        return create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=count * block_size,
                                 KV_LEN=sequence_length + count * block_size, device=anchors.device)
    require(backend == "sdpa", "Unsupported attention backend")
    q = torch.arange(count * block_size, device=anchors.device).view(1, 1, -1, 1)
    kv = torch.arange(sequence_length + count * block_size, device=anchors.device).view(1, 1, 1, -1)
    anchor_rows = anchors.view(bsz, 1, count, 1).repeat_interleave(block_size, dim=2)
    context = (kv < sequence_length) & (kv < anchor_rows)
    own_draft = (kv >= sequence_length) & (q // block_size == (kv - sequence_length) // block_size)
    return (context | own_draft) & keep.view(bsz, 1, count, 1).repeat_interleave(block_size, dim=2)


def author_response_loss(draft, embed_tokens, lm_head, input_ids, hidden_states, teacher_hidden_states,
                         loss_mask, *, mask_token_id, generator=None, attention_backend="sdpa",
                         projection_chunk=64, checkpoint_projection=True):
    """One batched backbone forward; weighted forward KL and auditable exposure.

    Only the frozen vocabulary projection/KL is checkpointed in row chunks. Its
    backward recomputes those heads, never the draft backbone or each block. This
    bounds the vocabulary activation memory while retaining one backward per
    response and the author's global weighted denominator. Logits use the native
    head dtype, matching the author's KL (no unannounced fp32 loss conversion).
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    require(type(projection_chunk) is int and projection_chunk > 0, "Invalid projection chunk")
    require(input_ids.ndim == 2 and input_ids.shape == loss_mask.shape and input_ids.shape[0] == 1,
            "Training uses one complete response per microbatch")
    require(hidden_states.shape[:2] == input_ids.shape and teacher_hidden_states.shape[:2] == input_ids.shape,
            "Teacher feature sequence differs from the complete sequence")
    require(not hidden_states.requires_grad and not teacher_hidden_states.requires_grad, "Teacher must be frozen")
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    anchors, keep = sample_anchor_positions(seq_len, loss_mask, generator=generator)
    n = anchors.shape[1]
    offsets = torch.arange(BLOCK_SIZE, device=device).view(1, 1, -1)
    labels = anchors.unsqueeze(-1) + offsets
    safe_labels = labels.clamp(max=seq_len - 1)
    noise_ids = torch.full((bsz, n, BLOCK_SIZE), mask_token_id, dtype=torch.long, device=device)
    noise_ids[:, :, 0] = torch.where(keep, input_ids.gather(1, anchors.clamp(0, seq_len - 1)), mask_token_id)
    with torch.no_grad():
        noise_embedding = embed_tokens(noise_ids.reshape(bsz, -1))
    positions = torch.cat((torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1),
                           labels.reshape(bsz, -1)), dim=1)
    mask = create_attention_mask(anchors, keep, seq_len, backend=attention_backend)
    output = draft(position_ids=positions, noise_embedding=noise_embedding,
                   target_hidden=hidden_states, attention_mask=mask)
    if isinstance(output, tuple):
        require(len(output) == 2 and output[1] is None, "Adaptive length/probability head must be disabled")
        output = output[0]
    require(output.shape[:2] == (bsz, n * BLOCK_SIZE), "Draft output shape differs from sampled parallel blocks")
    binary = keep.unsqueeze(-1).expand(-1, -1, BLOCK_SIZE).float() * (labels < seq_len).float()
    binary = binary * (offsets > 0).float()
    binary = binary * torch.gather(loss_mask.unsqueeze(1).expand(-1, n, -1), 2, safe_labels)
    decay = torch.exp(-(offsets - 1).clamp(min=0).float() / RECIPE["loss_decay_gamma"])
    weights = binary * decay
    denominator = weights.sum() + 1e-6
    selected = (weights.reshape(-1) > 0).nonzero(as_tuple=True)[0]
    flat_output = output.reshape(-1, output.shape[-1])
    teacher_rows = (safe_labels.reshape(-1) - 1).clamp(min=0)
    # Zero-exposure singleton selections are possible for a two-token response.
    # Preserve the author's zero loss/gradient and still execute the microbatch.
    total = output.sum() * 0.0

    def chunk_loss(student_hidden, teacher_hidden, row_weights):
        student_logits = lm_head(student_hidden)
        with torch.no_grad():
            teacher_logits = lm_head(teacher_hidden.detach().to(student_logits.dtype))
            teacher_probs = F.softmax(teacher_logits, dim=-1)
        rows = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_probs, reduction="none").sum(-1)
        return (rows * row_weights).sum()

    for start in range(0, selected.numel(), projection_chunk):
        ix = selected[start:start + projection_chunk]
        args = (flat_output[ix], teacher_hidden_states[0, teacher_rows[ix]], weights.reshape(-1)[ix])
        total = total + (checkpoint(chunk_loss, *args, use_reentrant=False) if checkpoint_projection
                         else chunk_loss(*args))
    loss = total / denominator
    require(bool(torch.isfinite(loss)), "Nonfinite author forward KL")
    positions_with_loss = labels[binary.bool()].detach().cpu().tolist()
    metadata = dict(anchor_positions=anchors[0].detach().cpu().tolist(),
                    block_keep_mask=keep[0].detach().cpu().tolist(),
                    label_positions=labels[0].detach().cpu().tolist(),
                    teacher_positions=(safe_labels[0] - 1).clamp(min=0).detach().cpu().tolist(),
                    binary_loss_mask=binary[0].detach().cpu().tolist(),
                    position_weights=weights[0].detach().cpu().tolist(),
                    sampled_blocks=n, eligible_blocks=int((binary.sum(-1) > 0).sum().item()),
                    eligible_positions=int(binary.sum().item()),
                    unique_supervised_positions=len(set(positions_with_loss)),
                    unique_supervised_position_ids=sorted(set(positions_with_loss)),
                    weighted_position_sum=float(weights.sum().item()),
                    normalization_denominator=float(denominator.item()),
                    excluded_anchor_positions=n, zero_supervision=selected.numel() == 0,
                    backbone_forward_calls=1,
                    head_projection_chunks=(selected.numel() + projection_chunk - 1) // projection_chunk)
    metadata["sampling_and_weights_sha256"] = digest(metadata)
    return loss, metadata


class PersistentAuthorOSDTrainer:
    """One episode, one released draft, one persistent optimizer through U1/U2.

    Start each episode with a new instance and freshly loaded released weights.
    The callback must return causal selected-layer and normalized-final features
    for every supplied token. ``check_frozen`` can additionally check the full
    target; embedding/head frozen status and feature detachment are always checked.
    Failed mutation poisons the instance; no automatic retry or optimizer reset.
    """
    def __init__(self, draft, embed_tokens, lm_head, teacher_features, *, episode_id, target_id,
                 target_identity_sha256, released_checkpoint_sha256, mask_token_id,
                 attention_backend="flex_attention", projection_chunk=64,
                 synchronize=None, check_frozen=None):
        import torch
        require(isinstance(episode_id, str) and episode_id and isinstance(target_id, str) and target_id,
                "Episode/target identity required")
        require(_sha256(target_identity_sha256) and _sha256(released_checkpoint_sha256), "Invalid model identity")
        require(type(mask_token_id) is int and mask_token_id >= 0, "Invalid mask token ID")
        require(attention_backend in ("flex_attention", "sdpa"), "Unsupported attention backend")
        require(type(projection_chunk) is int and projection_chunk > 0, "Invalid projection chunk")
        require(not _has_adaptive_head(draft),
                "Adaptive length/probability head must be disabled")
        self.draft, self.embed_tokens, self.lm_head = draft, embed_tokens, lm_head
        self.teacher_features = teacher_features
        self.episode_id, self.target_id = episode_id, target_id
        self.target_identity_sha256 = target_identity_sha256
        self.released_checkpoint_sha256 = released_checkpoint_sha256
        self.active_checkpoint_sha256 = released_checkpoint_sha256
        self.mask_token_id, self.attention_backend, self.projection_chunk = mask_token_id, attention_backend, projection_chunk
        self.parameters = dict(draft.named_parameters())
        require(self.parameters and all(p.requires_grad for p in self.parameters.values()),
                "Every draft parameter must be trainable for full-weight adaptation")
        self._parameter_ids = tuple(id(p) for p in self.parameters.values())
        self.device = next(iter(self.parameters.values())).device
        require(all(p.device == self.device for p in self.parameters.values()), "Single-device draft required")
        require(callable(teacher_features), "Frozen teacher callback required")
        self.synchronize = synchronize or (lambda: torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None)
        self.check_frozen = check_frozen or (lambda: None)
        self.optimizer = torch.optim.AdamW(list(self.parameters.values()), lr=3e-4, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=0.01, foreach=None, fused=None)
        self._optimizer_id = id(self.optimizer)
        self.generator = torch.Generator(device=self.device).manual_seed(RECIPE["seed"])
        self.completed_events = self.optimizer_steps = self.response_presentations = 0
        self.eligible_position_presentations = self.eligible_block_presentations = 0
        self.seen_record_ids = set()
        self.awaiting_checkpoint = self.failed = False
        self._check_owner()

    def _check_owner(self):
        require(id(self.optimizer) == self._optimizer_id and tuple(id(p) for p in self.draft.parameters()) ==
                self._parameter_ids and tuple(id(p) for g in self.optimizer.param_groups for p in g["params"]) ==
                self._parameter_ids, "Draft or optimizer replaced between updates")
        require(all(p.requires_grad for p in self.parameters.values()), "Full-weight parameter set changed")
        require(not _has_adaptive_head(self.draft),
                "Adaptive length/probability head became active")
        for module in (self.embed_tokens, self.lm_head):
            require(all(not p.requires_grad and p.grad is None and p.device == self.device for p in module.parameters()),
                    "Teacher embedding/head must remain frozen on the draft device")
        require(not self.embed_tokens.training and not self.lm_head.training, "Teacher embedding/head must be in eval mode")
        group = self.optimizer.param_groups[0]
        require(len(self.optimizer.param_groups) == 1 and group["lr"] == 3e-4 and group["betas"] == (0.9, 0.999)
                and group["eps"] == 1e-8 and group["weight_decay"] == 0.01, "Qualified AdamW recipe changed")
        states = self.optimizer.state
        require((not states and self.optimizer_steps == 0) or
                (set(states) == set(self.parameters.values()) and
                 {int(s["step"].item()) for s in states.values()} == {self.optimizer_steps}),
                "Adam state inventory/counter was reset or changed")
        self.check_frozen()

    def bind_checkpoint(self, event, checkpoint_sha256):
        require(not self.failed and self.awaiting_checkpoint and event == self.completed_events,
                "Checkpoint binding is not the completed pending event")
        require(_sha256(checkpoint_sha256), "Actual saved-weight checkpoint hash required")
        self._check_owner()
        self.active_checkpoint_sha256 = checkpoint_sha256
        self.awaiting_checkpoint = False

    def _train_response(self, response, audit):
        import torch
        self.synchronize()
        started = time.perf_counter()
        ids = torch.tensor([response["input_ids"] + response["output_ids"]], dtype=torch.long, device=self.device)
        # Full causal replay includes the current response; attention to features
        # at/after an anchor is excluded by the author's block attention mask.
        with torch.no_grad():
            features = self.teacher_features(ids)
        require(isinstance(features, (tuple, list)) and len(features) == 2, "Teacher callback must return two feature tensors")
        hidden, teacher = features
        for feature in features:
            require(feature.ndim == 3 and feature.shape[:2] == ids.shape and feature.device == self.device and
                    not feature.requires_grad and bool(torch.isfinite(feature).all()),
                    "Missing/nonfinite/unfrozen complete teacher features")
        # Do not retain the original 32K feature tuple after adding masked tail
        # rows: its selected-layer tensor can exceed a GiB on the native model.
        del feature, features
        self.synchronize()
        teacher_seconds = time.perf_counter() - started
        started = time.perf_counter()
        pad = BLOCK_SIZE - 1
        ids = torch.cat((ids, ids.new_full((1, pad), self.mask_token_id)), dim=1)
        # These placeholders are never visible to any real anchor. Do not replay
        # artificial padding through the teacher or give it ownership/loss.
        hidden = torch.cat((hidden, hidden.new_zeros((1, pad, hidden.shape[-1]))), dim=1)
        teacher = torch.cat((teacher, teacher.new_zeros((1, pad, teacher.shape[-1]))), dim=1)
        loss_mask = torch.zeros_like(ids, dtype=torch.float32)
        loss_mask[:, audit["response_start"]:audit["response_end_exclusive"]] = 1
        loss, sampling = author_response_loss(self.draft, self.embed_tokens, self.lm_head, ids, hidden, teacher,
                                              loss_mask, mask_token_id=self.mask_token_id, generator=self.generator,
                                              attention_backend=self.attention_backend, projection_chunk=self.projection_chunk)
        self.synchronize()
        forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        (loss / RECIPE["responses_per_step"]).backward()
        self.synchronize()
        backward_seconds = time.perf_counter() - started
        return dict(record_id=response["record_id"], turn=response["turn"], loss=float(loss.detach()),
                    response_start=audit["response_start"], response_end_exclusive=audit["response_end_exclusive"],
                    sequence_sha256=audit["sequence_sha256"], loss_mask_sha256=audit["loss_mask_sha256"],
                    **sampling, phase_seconds=dict(teacher_preparation_s=teacher_seconds,
                                                  forward_and_loss_s=forward_seconds, backward_s=backward_seconds))

    def train_event(self, event, responses):
        import torch
        require(not self.failed, "Prior failure requires explicit recovery with a new trainer")
        require(not self.awaiting_checkpoint, "Bind actual exported checkpoint before the next update")
        require(event == self.completed_events + 1, "Training events must execute once in order")
        self._check_owner()
        started = time.perf_counter()
        schedule = event_schedule(event, responses)
        audits = []
        for response in responses:
            require(response.get("episode_id") == self.episode_id, "Response belongs to a different episode")
            require(response.get("target_id") == self.target_id and
                    response.get("target_identity_sha256") == self.target_identity_sha256, "Fixed teacher identity changed")
            require(response.get("capture_checkpoint_sha256") == self.active_checkpoint_sha256,
                    "Response was not generated by the active serving draft")
            require(response.get("record_id") not in self.seen_record_ids, "Old support cannot be re-added")
            audits.append(audit_response(response))
        audit_seconds = time.perf_counter() - started
        lookup = {a["record_id"]: a for a in audits}
        batches, order = [], []
        training_started = time.perf_counter()
        initial_steps = self.optimizer_steps
        try:
            self.draft.train()
            self.optimizer.zero_grad(set_to_none=True)
            for pass_index, pair in schedule:
                batch_started = time.perf_counter()
                values = [self._train_response(response, lookup[response["record_id"]]) for response in pair]
                self._check_owner()
                self.synchronize()
                optimizer_started = time.perf_counter()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.parameters.values()),
                        "Missing or nonfinite full-weight gradient")
                norm = torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), 1.0, error_if_nonfinite=True)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_steps += 1
                self._check_owner()
                require(all(bool(torch.isfinite(p).all()) for p in self.parameters.values()), "Nonfinite draft update")
                self.synchronize()
                optimizer_seconds = time.perf_counter() - optimizer_started
                self.response_presentations += 2
                self.eligible_position_presentations += sum(v["eligible_positions"] for v in values)
                self.eligible_block_presentations += sum(v["eligible_blocks"] for v in values)
                order.extend(v["record_id"] for v in values)
                batches.append(dict(pass_index=pass_index, optimizer_steps_after=self.optimizer_steps,
                                    presentations=values, loss=sum(v["loss"] for v in values) / 2,
                                    gradient_norm_before_clip=float(norm), optimizer_s=optimizer_seconds,
                                    wall_s=time.perf_counter() - batch_started))
        except BaseException:
            self.failed = True
            raise
        self.completed_events = event
        self.seen_record_ids.update(r["record_id"] for r in responses)
        self.awaiting_checkpoint = True
        training_seconds = time.perf_counter() - training_started
        presentations = [v for b in batches for v in b["presentations"]]
        teacher_seconds = sum(v["phase_seconds"]["teacher_preparation_s"] for v in presentations)
        return dict(event=event, episode_id=self.episode_id, arm_name=ARM_NAME, author_commit=AUTHOR_COMMIT,
                    recipe=dict(RECIPE), attention_backend=self.attention_backend, projection_chunk=self.projection_chunk,
                    capture_checkpoint_sha256=self.active_checkpoint_sha256,
                    support_turns=[r["turn"] for r in responses], response_audits=audits,
                    optimizer_steps_this_event=self.optimizer_steps - initial_steps, optimizer_steps_total=self.optimizer_steps,
                    response_presentations_this_event=len(presentations), response_presentations_total=self.response_presentations,
                    eligible_position_presentations_this_event=sum(v["eligible_positions"] for v in presentations),
                    eligible_position_presentations_total=self.eligible_position_presentations,
                    eligible_block_presentations_this_event=sum(v["eligible_blocks"] for v in presentations),
                    eligible_block_presentations_total=self.eligible_block_presentations,
                    zero_supervision_presentations=sum(v["zero_supervision"] for v in presentations),
                    response_order=order, response_order_sha256=digest(order), batches=batches,
                    costs=dict(mask_audit_s=audit_seconds, training_s=training_seconds,
                               teacher_preparation_s=teacher_seconds,
                               draft_training_and_optimizer_s=training_seconds - teacher_seconds,
                               training_event_wall_s=time.perf_counter() - started,
                               accounting="training_s includes teacher_preparation_s; do not sum those two"))

    def save_checkpoint_state(self, path, *, include_draft=True):
        """Durable persistent Adam/RNG recovery state, separate from serving export."""
        import torch
        require(not self.failed and not self.awaiting_checkpoint and self.completed_events > 0,
                "Bind saved serving weights before saving completed trainer state")
        self._check_owner()
        path = Path(path)
        require(not path.exists() and not path.is_symlink(), "Trainer state must use a fresh path")
        state = dict(schema_version="dflash_author_osd_trainer_v1", arm_name=ARM_NAME, author_commit=AUTHOR_COMMIT,
                     episode_id=self.episode_id, target_id=self.target_id, target_identity_sha256=self.target_identity_sha256,
                     released_checkpoint_sha256=self.released_checkpoint_sha256,
                     checkpoint_sha256=self.active_checkpoint_sha256, recipe=dict(RECIPE),
                     completed_events=self.completed_events, optimizer_steps=self.optimizer_steps,
                     response_presentations=self.response_presentations,
                     eligible_position_presentations=self.eligible_position_presentations,
                     eligible_block_presentations=self.eligible_block_presentations,
                     seen_record_ids=sorted(self.seen_record_ids), optimizer=self.optimizer.state_dict(),
                     anchor_rng_state=self.generator.get_state(), torch_rng_state=torch.get_rng_state(),
                     includes_draft=include_draft)
        if include_draft:
            state["draft"] = self.draft.state_dict()
        if self.device.type == "cuda":
            state["cuda_rng_state"] = torch.cuda.get_rng_state(self.device)
        with path.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=h.hexdigest(),
                    includes_draft=include_draft, checkpoint_sha256=self.active_checkpoint_sha256)
