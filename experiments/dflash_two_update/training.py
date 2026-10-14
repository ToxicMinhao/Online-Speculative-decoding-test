"""Two chronological updates of the same full DFlash draft and AdamW state.

The injected backend is the source-pinned FAST W8 ``NativeReplay`` interface;
this module never reconstructs draft hidden states or substitutes committed
answer tokens for captured verification branches. Torch is imported lazily.

Each response carries its own input boundary and capture checkpoint. Every
block carries the response identity, native terminal/cap mask, per-slot owner,
and an explicit owned eligible slot list. Historical responses are conditioning
only, including responses from the same eight-response training group.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


BLOCK_SIZE = 10
CONTEXT_LIMIT = 17408
RECIPE = dict(seed=17, learning_rate=3e-4, betas=(0.9, 0.999), eps=1e-8,
              weight_decay=0.01, gradient_clip=1.0, passes=2,
              responses_per_step=2, optimizer_steps_per_event=8,
              loss="forward_KL", normalization="mean_slots_then_blocks_then_responses")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def response_loss_mask(*, record_id, input_token_count, positions, slot_owner_record_ids,
                       eligible_mask, padding_mask=None):
    """Explicit current-response ownership, applicable to a full sequence or block.

    There is deliberately no committed-answer end bound: a verified proposal
    beyond the first rejection may still belong to this response. The native
    terminal/cap eligibility mask supplies its actual upper bound.
    """
    require(isinstance(record_id, str) and record_id, "Missing response identity")
    require(type(input_token_count) is int and input_token_count > 0, "Invalid input boundary")
    n = len(positions)
    padding = [False] * n if padding_mask is None else padding_mask
    require(len(slot_owner_record_ids) == len(eligible_mask) == len(padding) == n,
            "Loss mask shapes differ")
    require(all(type(v) is int and v >= 0 for v in positions), "Invalid token positions")
    require(all(type(v) is bool for v in list(eligible_mask) + list(padding)), "Masks must be explicit booleans")
    return [eligible and not pad and pos >= input_token_count and owner == record_id
            for pos, owner, eligible, pad in zip(positions, slot_owner_record_ids, eligible_mask, padding)]


def owned_slots(response, block):
    """Validate an original capture block, rejecting silently dropped native slots."""
    for block_key, response_key in (("response_record_id", "record_id"), ("response_turn", "turn"),
                                    ("input_token_count", "input_token_count"),
                                    ("capture_checkpoint_sha256", "capture_checkpoint_sha256"),
                                    ("target_id", "target_id"),
                                    ("target_identity_sha256", "target_identity_sha256")):
        require(block.get(block_key) == response.get(response_key), "Block provenance differs: " + block_key)
    anchor = block.get("anchor")
    require(type(anchor) is int and anchor >= response["input_token_count"] and
            block.get("context_length") == anchor, "Block crosses the current response input boundary")
    require(anchor + BLOCK_SIZE <= CONTEXT_LIMIT, "Native complete context exceeds limit; truncation forbidden")
    require(block.get("positions") == list(range(anchor, anchor + BLOCK_SIZE)), "Native positions changed")
    require(block.get("teacher_row_for_slot") == [None] + list(range(9)), "Native teacher-row alignment changed")
    mask = block.get("eligible_mask")
    require(isinstance(mask, list) and len(mask) == BLOCK_SIZE and all(type(v) is bool for v in mask)
            and mask[0] is False, "Native seed/eligibility mask invalid")
    outcome = block.get("outcome", {})
    require(type(outcome.get("finished")) is bool and type(outcome.get("output_offset")) is int
            and outcome["output_offset"] > 0 and isinstance(outcome.get("committed_token_ids"), list)
            and outcome["committed_token_ids"], "Missing native outcome boundary")
    require(anchor == response["input_token_count"] + outcome["output_offset"] - 1,
            "Capture anchor differs from original native output chronology")
    end = response["input_token_count"] + response["output_cap"]
    if outcome["finished"]:
        end = min(end, response["input_token_count"] + outcome["output_offset"] +
                  len(outcome["committed_token_ids"]))
    expected = [False] + [anchor + slot < end for slot in range(1, BLOCK_SIZE)]
    require(mask == expected, "Native terminal/cap eligibility changed")
    owners = block.get("slot_owner_record_ids")
    require(isinstance(owners, list) and len(owners) == BLOCK_SIZE, "Missing explicit slot ownership")
    owned = response_loss_mask(record_id=response["record_id"], input_token_count=response["input_token_count"],
                               positions=block["positions"], slot_owner_record_ids=owners,
                               eligible_mask=mask, padding_mask=block.get("padding_mask"))
    slots = [i for i, enabled in enumerate(owned) if enabled]
    declared = block.get("owned_eligible_slots")
    require(isinstance(declared, list) and all(type(v) is int for v in declared) and declared == slots,
            "Owned eligible slots differ from explicit loss mask")
    # A fresh native capture belongs to exactly one response. A foreign/history/
    # padded native-eligible slot is corruption, not permission to reduce exposure.
    require(slots == [i for i, enabled in enumerate(mask) if enabled],
            "Native eligible slots were dropped or assigned to another response")
    require(block.get("eligible_positions", len(slots)) == len(slots), "Block position count differs")
    return slots


def audit_response(response):
    """Audit all small metadata before any training; feature tensors stay lazy."""
    require(isinstance(response.get("record_id"), str) and response["record_id"], "Missing response record ID")
    require(type(response.get("turn")) is int, "Invalid response turn")
    require(type(response.get("input_token_count")) is int and response["input_token_count"] > 0,
            "Invalid response input boundary")
    require(type(response.get("output_cap")) is int and response["output_cap"] > 0, "Invalid output cap")
    require(_sha256(response.get("capture_checkpoint_sha256")), "Missing capture checkpoint hash")
    ids = response.get("input_ids")
    require(isinstance(ids, list) and len(ids) == response["input_token_count"] and
            all(type(v) is int and v >= 0 for v in ids), "Input token inventory/boundary differs")
    if "input_ids_sha256" in response:
        require(digest(ids) == response["input_ids_sha256"], "Input token hash differs")
    require(response["input_token_count"] + response["output_cap"] <= CONTEXT_LIMIT,
            "Full accumulated history plus output cap exceeds limit; truncation forbidden")
    blocks = response.get("blocks")
    require(isinstance(blocks, list) and blocks, "Response has no captured blocks")
    counts = dict(blocks=len(blocks), eligible_blocks=0, eligible_positions=0,
                  excluded_seed_positions=len(blocks), native_ineligible_positions=0,
                  dropped_native_eligible_positions=0)
    for index, block in enumerate(blocks):
        require(block.get("block_index") == index, "Block index gap or duplication")
        slots = owned_slots(response, block)
        counts["eligible_blocks"] += bool(slots)
        counts["eligible_positions"] += len(slots)
        counts["native_ineligible_positions"] += BLOCK_SIZE - 1 - len(slots)
    require(counts["eligible_blocks"] > 0, "Entirely ineligible response requires capture investigation")
    require(response.get("eligible_block_count") == counts["eligible_blocks"] and
            response.get("eligible_position_count") == counts["eligible_positions"],
            "Response eligible exposure differs from immutable metadata")
    return counts


def event_schedule(event, responses):
    require(type(event) is int and event in (1, 2), "Only update events 1 and 2 exist")
    turns = list(range(8 * event + 1, 8 * event + 9))
    require(len(responses) == 8 and [r.get("turn") for r in responses] == turns,
            "Event requires exactly its eight chronological new responses")
    require(len({r.get("record_id") for r in responses}) == 8, "Response IDs must be unique within event")
    return [(pass_index, responses[offset:offset + 2]) for pass_index in range(2) for offset in range(0, 8, 2)]


def forward_kl(student_logits, teacher_logits):
    """Unchanged W8 full-vocabulary KL(P_target || Q_draft), one value per slot."""
    import torch
    import torch.nn.functional as F
    require(student_logits.ndim == 2 and student_logits.shape == teacher_logits.shape and
            student_logits.shape[0] > 0 and student_logits.shape[1] > 1, "Student/teacher logit shapes differ")
    require(student_logits.device == teacher_logits.device and
            bool(torch.isfinite(student_logits).all()) and bool(torch.isfinite(teacher_logits).all()),
            "Nonfinite/wrong-device logits")
    log_q = F.log_softmax(student_logits.float(), dim=-1)
    log_p = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    value = (log_p.exp() * (log_p - log_q)).sum(-1)
    require(bool(torch.isfinite(value).all()), "Nonfinite forward KL")
    return value


def chunk_weight(chunk_positions, block_positions, response_blocks):
    require(all(type(v) is int and v > 0 for v in (chunk_positions, block_positions, response_blocks))
            and chunk_positions <= block_positions, "Invalid KL normalization denominators")
    return chunk_positions / (block_positions * response_blocks * 2)


class PersistentFullKLTrainer:
    """One episode only; create a new instance with a released draft per episode.

    ``bind_checkpoint`` registers an actual saved checkpoint identity after an
    event. The runtime must hash the saved weights; a recipe hash is insufficient.
    A failed training attempt poisons this instance to prevent silent retries.
    """
    def __init__(self, draft, backend, *, episode_id, target_id, target_identity_sha256,
                 released_checkpoint_sha256, seed=17, position_chunk=9):
        import torch
        require(seed == RECIPE["seed"], "Only the qualified seed 17 is admitted")
        require(type(position_chunk) is int and 1 <= position_chunk <= 9, "Invalid position chunk")
        require(isinstance(episode_id, str) and episode_id and isinstance(target_id, str) and target_id,
                "Episode/target identity required")
        require(_sha256(target_identity_sha256) and _sha256(released_checkpoint_sha256), "Invalid model identity")
        self.draft, self.backend = draft, backend
        self.episode_id, self.target_id = episode_id, target_id
        self.target_identity_sha256 = target_identity_sha256
        self.released_checkpoint_sha256 = released_checkpoint_sha256
        self.active_checkpoint_sha256 = released_checkpoint_sha256
        self.seed, self.position_chunk = seed, position_chunk
        self.parameters = dict(draft.named_parameters())
        require(self.parameters and all(p.requires_grad for p in self.parameters.values()),
                "Every draft parameter must be trainable: this is full-weight adaptation")
        self._parameter_ids = tuple(id(p) for p in self.parameters.values())
        require(len(set(self._parameter_ids)) == len(self._parameter_ids), "Aliased draft parameter inventory")
        require(getattr(backend, "draft", draft) is draft, "Replay uses a different draft")
        self.optimizer = torch.optim.AdamW(list(self.parameters.values()), lr=3e-4, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=0.01, foreach=None, fused=None)
        self._optimizer_id = id(self.optimizer)
        self.completed_events = self.optimizer_steps = 0
        self.response_presentations = self.eligible_position_presentations = self.eligible_block_presentations = 0
        self.seen_record_ids = set()
        self.awaiting_checkpoint = False
        self.failed = False
        self.backend.activate(target_id)
        self.backend.check_frozen()
        self.draft.train()

    def _check_owner(self):
        require(id(self.optimizer) == self._optimizer_id and tuple(id(p) for p in self.draft.parameters()) ==
                self._parameter_ids and tuple(id(p) for g in self.optimizer.param_groups for p in g["params"]) ==
                self._parameter_ids, "Draft or optimizer replaced between updates")
        require(all(p.requires_grad for p in self.parameters.values()), "Full-weight parameter set changed")
        group = self.optimizer.param_groups[0]
        require(len(self.optimizer.param_groups) == 1 and group["lr"] == 3e-4 and group["betas"] == (0.9, 0.999)
                and group["eps"] == 1e-8 and group["weight_decay"] == 0.01, "Qualified AdamW recipe changed")
        states = self.optimizer.state
        require((not states and self.optimizer_steps == 0) or
                (set(states) == set(self.parameters.values()) and
                 {int(s["step"].item()) for s in states.values()} == {self.optimizer_steps}),
                "Adam state inventory/counter was reset or changed")
        self.backend.check_frozen()

    def bind_checkpoint(self, event, checkpoint_sha256):
        require(not self.failed and self.awaiting_checkpoint and event == self.completed_events,
                "Checkpoint binding is not the completed pending event")
        require(_sha256(checkpoint_sha256), "Actual saved-weight checkpoint hash required")
        self._check_owner()
        self.active_checkpoint_sha256 = checkpoint_sha256
        self.awaiting_checkpoint = False

    def _train_response(self, response, audit):
        phases = {key: 0.0 for key in ("feature_record_read_s", "forward_and_heads_s", "loss_reduction_s", "backward_s")}
        forward_nested_phases = {}
        synchronize = getattr(self.backend, "synchronize", lambda: None)
        seen_blocks = positions = eligible_blocks = 0
        scalar = 0.0
        metadata = response["blocks"]
        iterator = iter(self.backend.iter_blocks(response))
        while True:
            started = time.perf_counter()
            try:
                block = next(iterator)
            except StopIteration:
                break
            phases["feature_record_read_s"] += time.perf_counter() - started
            require(seen_blocks < len(metadata), "Replay yielded extra blocks")
            # Compare every source metadata field, excluding lazily supplied tensors.
            for key, value in metadata[seen_blocks].items():
                require(key in block and block[key] == value, "Replay metadata changed: " + key)
            seen_blocks += 1
            slots = owned_slots(response, block)
            require(list(self.backend.eligible_slots(block)) == slots, "Replay eligibility changed")
            if not slots:
                continue
            eligible_blocks += 1
            positions += len(slots)
            for offset in range(0, len(slots), self.position_chunk):
                chosen = slots[offset:offset + self.position_chunk]
                synchronize()
                started = time.perf_counter()
                student, teacher, _hard = self.backend.forward_block(block, chosen)
                synchronize()
                phases["forward_and_heads_s"] += time.perf_counter() - started
                for name, seconds in getattr(self.backend, "last_phase_seconds", {}).items():
                    forward_nested_phases[name] = forward_nested_phases.get(name, 0.) + seconds
                require(student.requires_grad and not teacher.requires_grad and student.shape[0] == len(chosen),
                        "Replay student differentiation/teacher detachment differs")
                started = time.perf_counter()
                loss = forward_kl(student, teacher).mean()
                weight = chunk_weight(len(chosen), len(slots), audit["eligible_blocks"])
                synchronize()
                phases["loss_reduction_s"] += time.perf_counter() - started
                started = time.perf_counter()
                (loss * weight).backward()
                synchronize()
                phases["backward_s"] += time.perf_counter() - started
                scalar += float(loss.detach()) * weight * 2
                del student, teacher, _hard, loss
        require(seen_blocks == audit["blocks"] and eligible_blocks == audit["eligible_blocks"] and
                positions == audit["eligible_positions"], "Replay block/position coverage differs")
        self.backend.finish_response()
        return dict(record_id=response["record_id"], turn=response["turn"], loss=scalar,
                    eligible_blocks=eligible_blocks, eligible_positions=positions, phase_seconds=phases,
                    forward_nested_phase_seconds=forward_nested_phases)

    def train_event(self, event, responses):
        import torch
        require(not self.failed, "Prior failure requires explicit recovery with a new trainer")
        require(not self.awaiting_checkpoint, "Bind the actual exported checkpoint before next update")
        require(event == self.completed_events + 1, "Training events must execute once in order")
        self._check_owner()
        started = time.perf_counter()
        schedule = event_schedule(event, responses)
        audits = []
        for response in responses:
            require(response.get("episode_id") == self.episode_id, "Response belongs to a different episode")
            require(response.get("target_id") == self.target_id and
                    response.get("target_identity_sha256") == self.target_identity_sha256,
                    "Fixed teacher identity changed")
            require(response.get("capture_checkpoint_sha256") == self.active_checkpoint_sha256,
                    "Feedback was not captured by the active serving draft")
            require(response.get("record_id") not in self.seen_record_ids, "Old support cannot be re-added")
            audits.append(audit_response(response))
        audit_seconds = time.perf_counter() - started
        lookup = {r["record_id"]: a for r, a in zip(responses, audits)}
        batches, order = [], []
        training_started = time.perf_counter()
        try:
            self.draft.train()
            self.optimizer.zero_grad(set_to_none=True)
            for pass_index, pair in schedule:
                batch_start = time.perf_counter()
                values = [self._train_response(response, lookup[response["record_id"]]) for response in pair]
                self.backend.check_frozen()
                require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.parameters.values()),
                        "Missing or nonfinite full-weight gradient")
                norm = torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), 1.0, error_if_nonfinite=True)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_steps += 1
                self._check_owner()
                require(all(bool(torch.isfinite(p).all()) for p in self.parameters.values()), "Nonfinite draft update")
                self.response_presentations += 2
                self.eligible_position_presentations += sum(v["eligible_positions"] for v in values)
                self.eligible_block_presentations += sum(v["eligible_blocks"] for v in values)
                order.extend(v["record_id"] for v in values)
                batches.append(dict(pass_index=pass_index, optimizer_steps_after=self.optimizer_steps,
                                    presentations=values, loss=sum(v["loss"] for v in values) / 2,
                                    gradient_norm_before_clip=float(norm), wall_s=time.perf_counter() - batch_start))
        except BaseException:
            self.failed = True
            raise
        self.completed_events = event
        self.seen_record_ids.update(r["record_id"] for r in responses)
        self.awaiting_checkpoint = True
        return dict(event=event, episode_id=self.episode_id, recipe=dict(RECIPE),
                    capture_checkpoint_sha256=self.active_checkpoint_sha256,
                    support_turns=[r["turn"] for r in responses], response_audits=audits,
                    optimizer_steps_this_event=8, optimizer_steps_total=self.optimizer_steps,
                    response_presentations_this_event=16, response_presentations_total=self.response_presentations,
                    eligible_position_presentations_this_event=2 * sum(a["eligible_positions"] for a in audits),
                    eligible_position_presentations_total=self.eligible_position_presentations,
                    eligible_block_presentations_total=self.eligible_block_presentations,
                    response_order=order, response_order_sha256=digest(order), batches=batches,
                    costs=dict(mask_audit_s=audit_seconds, training_s=time.perf_counter() - training_started,
                               training_event_wall_s=time.perf_counter() - started))

    def save_checkpoint_state(self, path, *, include_draft=True):
        """Save a fresh durable state file; callers separately save serving weights.

        If ``include_draft=False``, the bound serving checkpoint is required for
        restoration. No automatic resume/retry or optimizer reset is performed.
        """
        import torch
        require(not self.failed and not self.awaiting_checkpoint and self.completed_events > 0,
                "Bind saved serving weights before saving completed trainer state")
        self._check_owner()
        path = Path(path)
        require(not path.exists() and not path.is_symlink(), "Trainer state must use a fresh path")
        state = dict(schema_version="dflash_two_update_trainer_v1", episode_id=self.episode_id,
                     target_id=self.target_id, target_identity_sha256=self.target_identity_sha256,
                     released_checkpoint_sha256=self.released_checkpoint_sha256,
                     checkpoint_sha256=self.active_checkpoint_sha256, recipe=dict(RECIPE),
                     completed_events=self.completed_events, optimizer_steps=self.optimizer_steps,
                     response_presentations=self.response_presentations,
                     eligible_position_presentations=self.eligible_position_presentations,
                     eligible_block_presentations=self.eligible_block_presentations,
                     seen_record_ids=sorted(self.seen_record_ids), optimizer=self.optimizer.state_dict(),
                     torch_rng_state=torch.get_rng_state(), includes_draft=include_draft)
        if include_draft:
            state["draft"] = self.draft.state_dict()
        if torch.cuda.is_initialized():
            state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        with path.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=h.hexdigest(),
                    includes_draft=include_draft, checkpoint_sha256=self.active_checkpoint_sha256)
