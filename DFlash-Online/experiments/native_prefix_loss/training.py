"""Disposable role updates around the unchanged native K+ continuation."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import time
import traceback

from experiments.threatened_repair_replay.training import ThreatReplayTrainer
from online_feedback.storage import write_json


class NativePrefixTrainer(ThreatReplayTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from online_feedback.native_prefix_roles import install_native_prefix_roles
        self.roles = install_native_prefix_roles(self.bundle.model)
        # Native execution has no installed extension. The all-enabled case
        # explicitly reinstalls it to qualify the narrow wrapper separately.
        self.bundle.model.native_prefix_roles = None

    def run_pair(self, *args, role="N", capture_gradients=False, **kwargs):
        self.roles.reset(role="ALL" if role == "N" else role,
                         capture_gradients=capture_gradients)
        previous = getattr(self.bundle.model, "native_prefix_roles", None)
        self.bundle.model.native_prefix_roles = None if role == "N" else self.roles
        failure = None
        try:
            result = super().run_pair(*args, capture_gradients=capture_gradients, **kwargs)
            result["role"] = self.roles.receipt()
            result["role"]["execution_role"] = role
            if role == "N":
                result["role"]["responses"] = [dict(
                    native_full_objective=row["response_observation"]["loss"],
                    gradient_enabled_objective=row["response_observation"]["loss"],
                    scope="unchanged native full backward; no role observer installed")
                    for row in result["rows"]]
            result["update"]["role"] = result["role"]
            return result
        except BaseException as error:
            failure = error
            raise
        finally:
            self.bundle.model.native_prefix_roles = previous
            # Preserve row-level evidence even when backward or the finite gate
            # fails before the inherited pair method can return a bundle.
            label = kwargs.get("label", role)
            category = kwargs.get("category", "native")
            ordinal = args[2] if len(args) > 2 else kwargs.get("ordinal", 0)
            path = self.output_root / "role_observations" / f"{category}_{ordinal // 2 + 1:03d}_{label}.json"
            try:
                write_json(path, self.roles.receipt())
            except BaseException as saving_error:
                if failure is None:
                    raise
                failure.add_note("Could not save role observations: " + repr(saving_error))

    def run_history_step(self, native_gradients, *, window, category="disposable", label="H"):
        """Apply ordinary Adam to explicit zero current gradients, with no forward.

        H is not a training continuation: presentations, sampler, RNG and core
        microbatch counter remain unchanged. Only the native optimizer and its
        global-step counter advance. The caller restores the entire N endpoint.
        """
        import torch
        from experiments.ok_continuation.resume import differences
        from online_feedback.rng import capture_rng

        parameters = {name: p for name, p in self.bundle.draft_model.named_parameters() if p.requires_grad}
        if parameters.keys() != native_gradients.keys() or self.core.accumulation_remainder:
            raise ValueError("H requires exact native gradient membership at a clean boundary")
        scope = self.ledger.scope(category=category, window=window, state=label) if self.ledger else nullcontext()
        path = self.output_root / f"{category}_history_{window:03d}_{label}.json"
        before_rng = capture_rng()
        receipt = dict(status="running", window=window, state=label, category=category,
            start_step=self.global_step, end_step=None, start_presentations=self.presentations,
            response_presentations=0, response_forwards=0, response_backwards=0,
            sampling_draws=0, optimizer_applications=0,
            native_present=[name for name, gradient in native_gradients.items() if gradient is not None],
            native_none=[name for name, gradient in native_gradients.items() if gradient is None],
            scope="disposable optimizer-history endpoint; no forward or RNG consumption")
        started = time.perf_counter()
        write_json(path, receipt)
        try:
            for name, parameter in parameters.items():
                parameter.grad = None if native_gradients[name] is None else torch.zeros_like(parameter)
            with scope:
                # Finite-gradient wrapper -> native clip -> Adam -> zero_grad.
                # Bypass the source-observation wrapper, because H has no pair.
                norm = self._native_step()
            receipt["optimizer_applications"] = 1
            self.global_step += 1
            receipt.update(end_step=self.global_step, end_presentations=self.presentations,
                           gradient_norm=float(norm), status="completed")
            mismatch = differences(before_rng, capture_rng(), "H_rng")
            receipt["rng_mismatches"] = mismatch
            if mismatch or float(norm) != 0:
                raise ValueError("Zero-current-gradient H changed RNG or had a nonzero gradient")
            return dict(rows=[], update=receipt, preclip_gradients={})
        except BaseException as error:
            receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            receipt["wall_s"] = time.perf_counter() - started
            write_json(path, receipt)


def make_trainer(plan, output, *, on_window=None, ledger=None):
    from experiments.recent_repair_retention.reference import read
    return NativePrefixTrainer(plan["configs"]["NATIVE"], output,
        binding=plan["binding"], origin_state=plan["origin_state"], origin_export=plan["origin_export"],
        original_config_text=Path(plan["origin_config"]).read_text(), schedule=read(plan["schedule"]),
        on_window=on_window, ledger=ledger)


def observe_source_pair(trainer, pair, pass_index, ordinal, *, window, state, ledger=None):
    """Two eval/no-grad full-layout source forwards, with exact state restoration.

    Full-block attention and physical BF16 vocabulary-head shapes match training.
    Original retained anchors are replayed, never sampled or selected by outcome.
    The observer does not call TrainerCore or modify its native observations.
    """
    import torch
    from experiments.lost_repair.training_probe import prepare_batch
    from experiments.threatened_repair_replay.replay import _hidden
    from online_feedback.first_error import valid_prefix_mask, first_error_mask
    from online_feedback.functional_geometry import isolated_geometry_runtime
    from online_feedback.rng import capture_rng
    from experiments.ok_continuation.resume import differences

    model = trainer.bundle.model
    if (model.block_size, model.objective_chunk_blocks, model.loss_dtype) != (16, 8, torch.bfloat16):
        raise ValueError("Source observation requires the qualified block/head/precision layout")
    before_work, before_costs = deepcopy(trainer.work), deepcopy(model.costs)
    before_rng = capture_rng()
    versions = [(p, p._version, p.grad, None if p.grad is None else p.grad._version) for p in model.parameters()]
    rows, work = [], []
    try:
        with isolated_geometry_runtime(model), torch.no_grad():
            device = next(model.parameters()).device
            for local, feature in enumerate(pair):
                binding = trainer.schedule["records"][ordinal + local]
                if binding["record_id"] != feature["record_id"] or binding["pass_index"] != pass_index:
                    raise ValueError("Source fitting pair differs from retained schedule")
                observation, = binding["sampled_anchors"]
                scope = ledger.scope(category="source_observation", window=window, state=state) if ledger else nullcontext()
                with scope, (ledger.operation("response_forward") if ledger else nullcontext()):
                    started = time.perf_counter()
                    raw = torch.load(feature["path"], map_location="cpu", weights_only=True)
                    if raw["record_id"] != feature["record_id"]:
                        raise ValueError("Source observer loaded a different request")
                    batch = prepare_batch(raw, trainer.cfg.data.max_length, device)
                    anchors = torch.tensor(observation["positions"], device=device, dtype=torch.long)
                    keep = torch.tensor(observation["keep_mask"], device=device, dtype=torch.bool)
                    labels, weights, _ = model._block_labels_and_weights(batch["input_ids"], batch["loss_mask"], anchors, keep)
                    hidden = _hidden(model, batch, anchors, keep)
                    chunks = 0
                    for begin in range(0, anchors.shape[1], 8):
                        logits = model.lm_head(hidden[:, begin:begin+8]).to(model.loss_dtype)
                        if not bool(torch.isfinite(logits).all()):
                            raise ValueError("Nonfinite source-observation logits")
                        target, weight = labels[:, begin:begin+8], weights[:, begin:begin+8]
                        proposals = logits.argmax(-1)
                        valid = valid_prefix_mask(target, weight, model.eos_token_ids)
                        selected = first_error_mask(proposals, target, weight, model.eos_token_ids)
                        prefix = (selected & (proposals == target)).sum(-1)
                        for local_anchor in range(target.shape[1]):
                            anchor = begin + local_anchor
                            available, accepted = int(valid[0, local_anchor].sum()), int(prefix[0, local_anchor])
                            correct = (proposals[0, local_anchor] == target[0, local_anchor]) & valid[0, local_anchor]
                            rows.append(dict(window=window, state=state, request_id=feature["record_id"],
                                pass_index=pass_index, presentation_id=ordinal+local, anchor_ordinal=anchor,
                                canonical_start=int(anchors[0, anchor])-int(raw["prompt_length"]),
                                accepted_prefix=accepted, valid_proposals=available,
                                first_error=accepted+1 if accepted < available else None,
                                valid=True, keep=bool(keep[0, anchor]), eligible_block=available > 0,
                                denominator_scope="all sampled anchor occurrences; eligible_block excludes zero-supervision rows",
                                correct_by_position=correct[1:].cpu().tolist(),
                                canonical_tokens=target[0, local_anchor, 1:].cpu().tolist(),
                                proposed_tokens=proposals[0, local_anchor, 1:].cpu().tolist()))
                        chunks += 1
                    del hidden, logits
                    work.append(dict(request_id=feature["record_id"], response_forwards=1, backwards=0,
                        logical_head_chunks=chunks, anchors=int(anchors.shape[1]), wall_s=time.perf_counter()-started))
    finally:
        trainer.work.clear(); trainer.work.update(before_work)
        model.costs.clear(); model.costs.update(before_costs)
    if differences(before_rng, capture_rng()) or any(p._version != version or p.grad is not gradient or
        (gradient is not None and gradient._version != grad_version) for p, version, gradient, grad_version in versions):
        raise ValueError("Source observation changed parameters, gradients or RNG")
    return rows, work
