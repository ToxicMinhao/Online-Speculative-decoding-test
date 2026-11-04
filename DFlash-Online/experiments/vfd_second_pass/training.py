"""Native K+ training with an explicit pass boundary and complete state forks.

The loss, sampler, TrainerCore, clipping and BF16 AdamW are unchanged. Only the
response order/second-pass membership and resumable experiment metadata differ.
"""

from copy import deepcopy
import math
from pathlib import Path
import time

from experiments.full_k_gsm8k.training import (
    live_state,
    verify_live,
    validate_config_text,
)
from experiments.kplus_full_gsm8k.training import validate_snapshot as validate_original
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from online_feedback.kplus import install_kplus, kplus_contract
from online_feedback.rng import capture_rng, preserve_rng, restore_rng
from online_feedback.storage import save_torch, write_json
from online_feedback.training import SpecForgeEventTrainer
from .contract import ARMS, SCHEMA, logical_steps, validate_order


def scientific_state(state):
    """The quantities a shared first pass must fork without modification."""
    return {
        key: state[key]
        for key in (
            "draft_state_dict",
            "backend",
            "rng",
            "global_step",
            "presentations",
            "events",
            "accumulation_steps",
            "horizon",
        )
    }


def check_native_boundary(state, *, expected_steps, recipe, expected_cuda_rng_states=1):
    import torch
    from experiments.ok_continuation.resume import _finite

    if (
        state["global_step"] != expected_steps
        or state["presentations"] != 2 * expected_steps
    ):
        raise ValueError("Own state counters differ from the declared pass boundary")
    if state["accumulation_steps"] != 2 or state["horizon"] != 7424:
        raise ValueError("Accumulation or inherited optimizer horizon changed")
    optimizer = state["backend"]["optimizer"]
    if differences(optimizer["recipe"], recipe):
        raise ValueError("The qualified optimizer recipe changed")
    saved = optimizer["optimizer_state_dict"]
    ids = saved["param_groups"][0]["params"]
    if set(saved["state"]) != (set(ids) if expected_steps else set()):
        raise ValueError("Adam moment membership differs")
    for (name, shape, dtype), parameter_id in zip(recipe["parameters"], ids):
        weight = state["draft_state_dict"][name]
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != tuple(shape):
            raise ValueError("Native parameter shape or precision changed: " + name)
        _finite(weight, name)
        if expected_steps:
            moments = saved["state"][parameter_id]
            if moments["step"].item() != expected_steps:
                raise ValueError("Adam counter differs: " + name)
            for key in ("exp_avg", "exp_avg_sq"):
                value = moments[key]
                if value.dtype != torch.bfloat16 or value.shape != weight.shape:
                    raise ValueError("Native Adam moment shape or precision changed")
                _finite(value, name + "." + key)
    if (
        expected_cuda_rng_states not in (0, 1)
        or len(state["rng"]["cuda"]) != expected_cuda_rng_states
    ):
        raise ValueError("Require the original single-training-GPU RNG state")


class VFDTrainer(SpecForgeEventTrainer):
    def __init__(
        self,
        config_path,
        output_root,
        *,
        original_config_text,
        expected_cuda_rng_states=1,
    ):
        if expected_cuda_rng_states not in (0, 1):
            raise ValueError(
                "Only CPU fixtures or one native training GPU are qualified"
            )
        self.expected_cuda_rng_states = expected_cuda_rng_states
        self.original_config_text = original_config_text
        self.pass_records = []
        self.forward_observations = []
        self.work = {}
        self._handles = []
        super().__init__(
            config_path,
            output_root,
            objective="osd_first_error",
            reduction="response_mean",
            ranking_lambda=0.0,
        )

    def _validate_config(self):
        super()._validate_config()
        validate_config_text(self.config_text, "K", self.original_config_text)

    def _build_runtime(self):
        super()._build_runtime()
        install_kplus(self.bundle.model)
        self.contract[SCHEMA] = dict(
            redistribution=kplus_contract(),
            shared_feedback=True,
            second_pass_only=True,
            native_anchor_sampling=True,
            order_control="FULL_native_permutations; each_executed_pass_consumes_own_native_permutation",
        )
        self._native_forward = self.author_strategy.forward_loss

        def observed_forward(batch, ctx=None):
            result = self._native_forward(batch, ctx)
            self.forward_observations.append(
                dict(
                    record_id=str(batch.sample_ids[0]),
                    loss=float(result.loss.detach()),
                    sum_metrics={
                        key: float(value.detach())
                        for key, value in result.sum_metrics.items()
                    },
                )
            )
            return result

        self.author_strategy.forward_loss = observed_forward
        for module, name in (
            (self.bundle.draft_model, "draft_forwards"),
            (self.bundle.model.lm_head, "vocabulary_projections"),
        ):

            def count(_module, _inputs, name=name):
                self.work[name] = self.work.get(name, 0) + 1

            self._handles.append(module.register_forward_pre_hook(count))

    def restore_original(self, path):
        """Authenticate the original event-zero state before installing run metadata."""
        import torch

        original = torch.load(path, map_location="cpu", weights_only=False)
        binding = original["algorithm_contract"]["kplus_full_gsm8k_v1"]["binding"]
        validate_original(
            original,
            binding=binding,
            config_text=original["config_text"],
            original_config_text=self.original_config_text,
            completed_event=0,
            expected_contract=original["algorithm_contract"],
            expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states,
        )
        self._load_native(original)
        _export_equal(original, self.cfg.model.draft_checkpoint_path)
        if differences(scientific_state(original), scientific_state(live_state(self))):
            raise ValueError("Released event-zero scientific restoration differs")
        receipt = dict(
            status="passed",
            original_state=str(path),
            global_step=0,
            fresh_optimizer=True,
            original_rng_restored=True,
            extra_model_work=0,
        )
        write_json(self.output_root / "INITIAL_RESTORATION.json", receipt)
        return receipt

    def _load_native(self, state):
        self.bundle.draft_model.load_state_dict(state["draft_state_dict"], strict=True)
        self.backend.load_state_dict(state["backend"])
        self.backend.optimizer.zero_grad()
        self.global_step, self.presentations = (
            state["global_step"],
            state["presentations"],
        )
        self.events = deepcopy(state["events"])
        self.core._micro = self.presentations
        self.core._ratio_totals.clear()
        self.core._sum_totals.clear()
        self._anchor_observations.clear()
        self.bundle.model.train()
        restore_rng(state["rng"])

    def restore_boundary(self, path, *, arm, event, shared_fork=False):
        import torch

        packet = torch.load(path, map_location="cpu", weights_only=False)
        meta, native = packet["continuation"], packet["native"]
        expected = 64 if shared_fork else logical_steps(arm, event)
        if packet["schema"] != SCHEMA or meta != dict(
            arm="SHARED" if shared_fork else arm,
            completed_event=0 if shared_fork else event,
            shared_pass1=shared_fork,
        ):
            raise ValueError("Cannot restore another policy or event boundary")
        if (
            native["config_text"] != self.config_text
            or native["algorithm_contract"] != self.contract
        ):
            raise ValueError("Own continuation configuration or objective changed")
        check_native_boundary(
            native,
            expected_steps=expected,
            recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states,
        )
        self._load_native(native)
        check = verify_live(self, native)
        receipt = dict(
            check,
            source=str(path),
            arm=arm,
            shared_first_pass_fork=shared_fork,
            metadata_relabel_only=shared_fork,
        )
        write_json(self.output_root / "RESTORATION.json", receipt)
        return receipt

    def features(self, event, paths):
        if len(paths) != 128:
            raise ValueError(
                "The shared event must contain 128 distinct feedback responses"
            )
        return SpecForgeEventTrainer._prepare_event(
            self, f"event_{event:03d}", paths, 1
        )

    def run_pass(self, features, *, event, pass_index, full_order=None, selected=None):
        """Consume the native permutation draw, then use FULL's shared order.

        FULL uses its own draw. Other arms retain their own RNG progression but
        replace only the response order with the externally supplied FULL order.
        Subsets filter that order; anchors still come from the native sampler.
        """
        started = time.perf_counter()
        import torch
        from specforge.training.strategies.base import StepContext

        ids = [feature["record_id"] for feature in features]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("A pass requires distinct nonempty response identities")
        if full_order is not None:
            validate_order(list(full_order), ids)
        if selected is not None:
            selected = list(selected)
            if (
                not selected
                or len(selected) != len(set(selected))
                or not set(selected).issubset(ids)
            ):
                raise ValueError(
                    "Selected responses must be a nonempty distinct current-event subset"
                )
        pass_length = len(ids) if selected is None else len(selected)
        if pass_length % 2 or self.core.accumulation_remainder:
            raise ValueError(
                "A pass must consist of complete two-response optimizer windows"
            )
        own_order = [ids[index] for index in torch.randperm(len(ids)).tolist()]
        order = own_order if full_order is None else list(full_order)
        if selected is not None:
            selected_ids = set(selected)
            order = [record_id for record_id in order if record_id in selected_ids]
        by_id = {feature["record_id"]: feature for feature in features}
        start_step, start_presentations = self.global_step, self.presentations
        rows = []
        self.work.clear()
        self.forward_observations.clear()
        self.bundle.model.train()
        self.bundle.model.begin_event()
        try:
            for record_id in order:
                feature = by_id[record_id]
                raw = torch.load(feature["path"], map_location="cpu", weights_only=True)
                self._anchor_observations.clear()
                before = time.perf_counter()
                result = self.core.train_step(
                    self._batch(raw),
                    StepContext(
                        global_step=self.global_step,
                        total_steps=self.horizon,
                        collect_detailed_metrics=True,
                    ),
                )
                metrics = result.materialize_metrics()
                self.presentations += 1
                if result.optimizer_stepped:
                    self.global_step += 1
                if not all(math.isfinite(float(value)) for value in metrics.values()):
                    raise ValueError("Nonfinite native VFD training metric")
                rows.append(
                    dict(
                        record_id=record_id,
                        pass_index=pass_index,
                        optimizer_stepped=result.optimizer_stepped,
                        optimizer_step_after=self.global_step,
                        training_s=time.perf_counter() - before,
                        metrics=metrics,
                        prompt_tokens=feature["prompt_length"],
                        response_tokens=feature["response_tokens"],
                        sampled_anchors=deepcopy(self._anchor_observations),
                        response_observation=self.forward_observations[-1],
                    )
                )
        finally:
            objective_costs = self.bundle.model.end_event()
        if (
            self.global_step - start_step != len(order) // 2
            or self.core.accumulation_remainder
        ):
            raise ValueError("Native pass optimizer accounting differs")
        receipt = dict(
            event=event,
            pass_index=pass_index,
            start_step=start_step,
            end_step=self.global_step,
            optimizer_steps=self.global_step - start_step,
            response_presentations=self.presentations - start_presentations,
            records=rows,
            full_order=list(full_order) if full_order is not None else own_order,
            own_permutation=own_order,
            executed_order=order,
            objective_work=objective_costs,
            work=dict(self.work),
            training_s=sum(row["training_s"] for row in rows),
            pass_wall_s=time.perf_counter() - started,
        )
        write_json(self.output_root / f"pass_{pass_index}.json", receipt)
        self.pass_records.append(receipt)
        return receipt

    def save_boundary(self, path, *, arm, event, shared_fork=False, event_record=None):
        import torch
        from experiments.ok_events_5_8.campaign import identity

        started = time.perf_counter()
        with preserve_rng():
            if event_record is not None:
                self.events.append(deepcopy(event_record))
            state = live_state(self)
            expected = 64 if shared_fork else logical_steps(arm, event)
            check_native_boundary(
                state,
                expected_steps=expected,
                recipe=self.backend.optimizer.recipe,
                expected_cuda_rng_states=self.expected_cuda_rng_states,
            )
            verify_live(self, state)
            packet = dict(
                schema=SCHEMA,
                native=state,
                continuation=dict(
                    arm="SHARED" if shared_fork else arm,
                    completed_event=0 if shared_fork else event,
                    shared_pass1=shared_fork,
                ),
            )
            path = Path(path)
            staged = path.with_name(path.name + ".candidate")
            if staged.exists():
                raise FileExistsError(
                    "Preserve the prior failed state candidate: " + str(staged)
                )
            save_torch(staged, packet)
            decoded = torch.load(staged, map_location="cpu", weights_only=False)
            if differences(packet, decoded):
                raise ValueError("Persisted complete boundary differs from live state")
            saved_identity = identity(staged)
            staged.replace(path)
        return dict(
            status="passed",
            state_path=str(path),
            global_step=self.global_step,
            presentations=self.presentations,
            identity=saved_identity,
            persistence_s=time.perf_counter() - started,
        )

    def export(self, path):
        """Use the existing exporter with a transient native state container."""
        import torch

        started = time.perf_counter()
        with preserve_rng():
            state = live_state(self)
            save_torch(self.pending_state, state)
            self._export(Path(path))
            _export_equal(state, path)
            self.pending_state.unlink()
        return dict(
            checkpoint_path=str(path),
            export_s=time.perf_counter() - started,
            export_exact=True,
            extra_model_work=0,
        )

    def close(self):
        if self.closed:
            return
        for handle in self._handles:
            handle.remove()
        self.author_strategy.forward_loss = self._native_forward
        self._handles.clear()
        super().close()
