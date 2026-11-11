"""Native event-five training; only successor assignments differ in SHUFFLED."""

import argparse
from copy import deepcopy
import math
from pathlib import Path
import time
import traceback

from experiments.full_k_gsm8k import training as native
from experiments.ok_coding.storage import identity
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from experiments.role_loss_pilot import training as historical
from experiments.survival_head.training import validate_origin
from online_feedback.kplus import install_kplus, kplus_contract
from online_feedback.rng import preserve_rng, restore_rng
from online_feedback.storage import write_json
from online_feedback.training import SpecForgeEventTrainer
from .contract import ARMS, SCHEMA, experiment_contract, expected_work

CONTRACT_KEY = SCHEMA


def continuation_contract(origin, *, binding, arm, dataset="gsm8k"):
    old = origin["algorithm_contract"][historical.CONTRACT_KEY]
    result = deepcopy(origin["algorithm_contract"])
    result[CONTRACT_KEY] = experiment_contract(
        binding, arm, dataset, origin_binding=old["binding"]
    )
    return result


def validate_event_receipt(
    receipt,
    *,
    arm,
    dataset="gsm8k",
    event_number=5,
    require_completed=True,
    trainer_status=None
):
    if (
        arm not in ARMS
        or dataset != "gsm8k"
        or event_number != 5
        or receipt.get("method") != arm
        or receipt.get("dataset") != dataset
        or receipt.get("second_follower_work") != expected_work(arm)
        or receipt.get("kplus_loss") != (kplus_contract())
    ):
        raise ValueError("Second-follower event identity, loss or work differs")
    checked = native.validate_event_receipt(
        receipt,
        arm="K",
        event_number=5,
        require_completed=require_completed,
        trainer_status=trainer_status,
    )
    for row in receipt["records"]:
        observation = row["response_observation"]
        sums = observation["sum_metrics"]
        if (
            observation["record_id"] != row["record_id"]
            or observation["pass_index"] != row["pass"]
            or not all(math.isfinite(float(value)) for value in sums.values())
        ):
            raise ValueError("Response identity or finite metrics differ")
        if arm != "K":
            mass = sums["kplus_W1"] + sums["kplus_W2"]
            contract = kplus_contract()
            if (
                sums["kplus_response_count"] != 1
                or abs(sums["kplus_first_error_weight_after"] - mass)
                > contract["mass_conservation_atol"]
                + contract["mass_conservation_rtol"] * mass
            ):
                raise ValueError("K+ response counting or conserved mass differs")
        if arm == "SECOND" and sums.get("second_follower_disabled") != 0:
            raise ValueError("Qualification-only disabled reducer entered production")
    return dict(checked, arm=arm, dataset=dataset)


def validate_snapshot(
    state,
    *,
    origin,
    binding,
    arm,
    dataset,
    config_text,
    original_config_text,
    completed_event,
    expected_recipe,
    expected_cuda_rng_states=1
):
    if type(completed_event) is not int or completed_event not in (4, 5):
        raise ValueError("Only original512 and completed640 boundaries are authorized")
    expected = continuation_contract(origin, binding=binding, arm=arm, dataset=dataset)
    if differences(state["algorithm_contract"], expected) or differences(
        state["events"][:4], origin["events"]
    ):
        raise ValueError("Scientific contract or inherited event history changed")
    old_binding = origin["algorithm_contract"][historical.CONTRACT_KEY]["binding"]
    old_native = historical.native_contract_view(
        origin["algorithm_contract"], old_binding, "KPLUS", dataset
    )
    checked = native.validate_snapshot(
        dict(state, algorithm_contract=old_native),
        arm="K",
        binding=old_binding,
        config_text=config_text,
        original_config_text=original_config_text,
        completed_event=completed_event,
        expected_contract=old_native,
        expected_recipe=expected_recipe,
        expected_cuda_rng_states=expected_cuda_rng_states,
    )
    if completed_event == 5:
        validate_event_receipt(
            state["events"][4], arm=arm, dataset=dataset, require_completed=False
        )
    return dict(
        checked,
        arm=arm,
        historical_receipts_unchanged=True,
        start_adam_step=512,
        maximum_adam_step=640,
    )


def migrate_origin(
    origin,
    *,
    binding,
    arm,
    dataset,
    config_text,
    run_id,
    original_config_text,
    expected_recipe,
    expected_cuda_rng_states=1
):
    validate_origin(
        origin,
        dataset=dataset,
        original_config_text=original_config_text,
        expected_cuda_rng_states=expected_cuda_rng_states,
    )
    if CONTRACT_KEY in origin["algorithm_contract"]:
        raise ValueError(
            "Require original PR32 state, not a second-follower continuation"
        )
    state = dict(
        origin,
        config_text=config_text,
        run_id=run_id,
        algorithm_contract=continuation_contract(
            origin, binding=binding, arm=arm, dataset=dataset
        ),
    )
    validate_snapshot(
        state,
        origin=origin,
        binding=binding,
        arm=arm,
        dataset=dataset,
        config_text=config_text,
        original_config_text=original_config_text,
        completed_event=4,
        expected_recipe=expected_recipe,
        expected_cuda_rng_states=expected_cuda_rng_states,
    )
    return state


class SecondFollowerTrainer(native.FullKTrainer):
    def __init__(
        self,
        config_path,
        output_root,
        *,
        arm,
        binding,
        original_config_text,
        origin_state,
        origin_export,
        dataset="gsm8k",
        expected_cuda_rng_states=1
    ):
        import torch

        if arm not in ARMS or dataset != "gsm8k":
            raise ValueError("Require a declared GSM8K arm")
        self.arm, self.dataset, self.binding = arm, dataset, binding
        self.original_config_text = original_config_text
        self.expected_cuda_rng_states, self.completed_event = (
            expected_cuda_rng_states,
            4,
        )
        self.origin_path, self.origin_export = Path(origin_state), Path(origin_export)
        self.origin = torch.load(
            self.origin_path, map_location="cpu", weights_only=False, mmap=True
        )
        validate_origin(
            self.origin,
            dataset=dataset,
            original_config_text=original_config_text,
            expected_cuda_rng_states=expected_cuda_rng_states,
        )
        _export_equal(self.origin, self.origin_export)
        self.response_records, self.window_records, self.work, self._work_handles = (
            [],
            [],
            {},
            [],
        )
        self.extension_installed = False
        SpecForgeEventTrainer.__init__(
            self,
            config_path,
            output_root,
            resume_state=origin_state,
            objective="osd_first_error",
            reduction="response_mean",
            ranking_lambda=0.0,
        )

    def _validate_config(self):
        SpecForgeEventTrainer._validate_config(self)
        native.validate_config_text(self.config_text, "K", self.original_config_text)

    def _build_runtime(self):
        SpecForgeEventTrainer._build_runtime(self)
        if self.arm != "K":
            install_kplus(self.bundle.model)
        source_base = dict(self.origin["algorithm_contract"])
        del source_base[historical.CONTRACT_KEY]
        if differences(self.contract, source_base):
            raise ValueError("Native runtime differs from original KPLUS4")
        self.contract = continuation_contract(
            self.origin, binding=self.binding, arm=self.arm
        )
        self._install_observations()
        ordinary_forward, ordinary_step = self._native_forward, self._native_step

        def finite_forward(*args, **kwargs):
            import torch

            result = ordinary_forward(*args, **kwargs)
            if not bool(torch.isfinite(result.loss).all()):
                raise ValueError("Nonfinite native response loss")
            return result

        def finite_step():
            from experiments.adaflash_arepo.qualify import finite_gradient_check

            parameters = {
                name: p
                for name, p in self.bundle.draft_model.named_parameters()
                if p.requires_grad
            }
            finite_gradient_check(parameters, self.output_root, self.presentations)
            return ordinary_step()

        self._native_forward, self._native_step = finite_forward, finite_step

    def install_second(self, *, disabled=False):
        from online_feedback.second_follower import install_second_follower

        if self.extension_installed:
            raise ValueError("Second-follower reducer already installed")
        self.extension_contract = install_second_follower(
            self.bundle.model, disabled=disabled
        )
        self.extension_installed = True

    def validation_arguments(self):
        return dict(
            origin=self.origin,
            binding=self.binding,
            arm=self.arm,
            dataset=self.dataset,
            config_text=self.config_text,
            original_config_text=self.original_config_text,
            expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states,
        )

    def _restore(self, path):
        state = migrate_origin(
            self.origin,
            binding=self.binding,
            arm=self.arm,
            dataset=self.dataset,
            config_text=self.config_text,
            run_id=self.cfg.run_id,
            original_config_text=self.original_config_text,
            expected_recipe=self.backend.optimizer.recipe,
            expected_cuda_rng_states=self.expected_cuda_rng_states,
        )
        self.bundle.draft_model.load_state_dict(state["draft_state_dict"], strict=True)
        self.backend.load_state_dict(deepcopy(state["backend"]))
        self.global_step, self.presentations = (
            state["global_step"],
            state["presentations"],
        )
        self.events = deepcopy(state["events"])
        self.core._micro = self.presentations
        restore_rng(state["rng"])
        receipt = native.verify_live(self, state)
        receipt.update(
            origin_state=str(self.origin_path),
            origin_export=str(self.origin_export),
            historical_receipts_unchanged=True,
            source_step=512,
            seed_reset=False,
        )
        write_json(self.output_root / "RESUME_CHECK.json", receipt)

    def save_initial(self, *args, **kwargs):
        raise ValueError(
            "Second-follower study resumes complete KPLUS4; released initialization is forbidden"
        )

    def _prepare_event(self, event_id, feature_paths, passes):
        if len(self.events) != 4:
            raise ValueError("Second-follower study stops after event five")
        if self.arm == "SECOND":
            from online_feedback.second_follower import second_follower_contract

            if (
                not self.extension_installed
                or self.extension_contract != second_follower_contract()
                or self.bundle.model.first_error_redistribution.disabled
            ):
                raise ValueError(
                    "Production requires the active, fixed second-follower rule"
                )
        return native.FullKTrainer._prepare_event(self, event_id, feature_paths, passes)

    def _train_passes(self, features, passes):
        return historical.RoleLossTrainer._train_passes(self, features, passes)

    def train_event(
        self,
        event_id,
        feature_paths,
        passes=2,
        *,
        reference_check=None,
        presentation_schedule=None
    ):
        if reference_check is not None:
            raise ValueError("No additional model probes in main training")
        ordinary_backward = self.backend.backward

        def observed_backward(*args, **kwargs):
            result = ordinary_backward(*args, **kwargs)
            self.work["response_backwards"] = self.work.get("response_backwards", 0) + 1
            return result

        self.backend.backward = observed_backward
        try:
            result = SpecForgeEventTrainer.train_event(
                self,
                event_id,
                feature_paths,
                passes,
                presentation_schedule=presentation_schedule,
            )
        finally:
            self.backend.backward = ordinary_backward
        validate_event_receipt(result, arm=self.arm, trainer_status=self.status())
        self.completed_event = 5
        return result

    def _checkpoint_state(self, event):
        work = dict(
            optimizer_steps=sum(row["optimizer_stepped"] for row in event["records"]),
            ordinary_presentations=len(event["records"]),
            response_forwards=self.work.get("response_forwards", 0),
            response_backwards=self.work.get("response_backwards", 0),
        )
        event.update(
            method=self.arm,
            dataset=self.dataset,
            second_follower_work=work,
            kplus_loss=kplus_contract(),
        )
        validate_event_receipt(event, arm=self.arm, require_completed=False)
        return SpecForgeEventTrainer._checkpoint_state(self, event)

    def verify_boundary(self, export_path=None, receipt_path=None):
        import torch

        started = time.perf_counter()
        with preserve_rng():
            state = torch.load(
                self.latest, map_location="cpu", weights_only=False, mmap=True
            )
            receipt = validate_snapshot(
                state, **self.validation_arguments(), completed_event=5
            )
            if export_path is not None:
                _export_equal(state, export_path)
                receipt["export_exact"] = True
            receipt["live"] = native.verify_live(self, state)
        receipt["wall_s"] = time.perf_counter() - started
        if receipt_path is not None:
            write_json(receipt_path, receipt)
        return receipt


def train_native_event(root):
    from experiments.functional_step_geometry.qualify import cleanup
    from experiments.full_k_gsm8k.costs import begin_event_profile, finish_event_profile
    from experiments.ok_coding.campaign import (
        WorkLedger,
        read_work,
        require_complete_work,
    )
    from experiments.successor_shuffle.campaign import shared_inputs
    from experiments.kplus_abt.campaign import schedule_bindings
    from experiments.survival_head.data import read
    from .qualification import build_trainer

    root = Path(root)
    plan = read(root / "sealed/PLAN.json")
    output = root / "training/SECOND"
    if read(root / "qualification/ADMISSION.json")["status"] != "passed":
        raise ValueError("Require completed numerical qualification")
    output.mkdir(parents=True, exist_ok=False)
    limits = dict(response_forwards=256, response_backwards=256, optimizer_steps=128)
    ledger = WorkLedger(output / "WORK.jsonl", limits)
    progress = dict(status="running")
    trainer = None
    deferred = None
    failure = None
    started = time.perf_counter()
    try:
        loading = time.perf_counter()
        trainer, paths, baseline, progress["startup"] = build_trainer(plan, output)
        progress["loading_and_startup_s"] = time.perf_counter() - loading
        trainer.install_second()

        def count_main(owner, name, category):
            original = getattr(owner, name)

            def counted(*args, **kwargs):
                if getattr(trainer, "_second_follower_diagnostic", False):
                    return original(*args, **kwargs)
                return ledger.call(category, original, *args, **kwargs)

            setattr(owner, name, counted)

        count_main(trainer.core.strategy, "forward_loss", "response_forwards")
        count_main(trainer.backend, "backward", "response_backwards")
        count_main(trainer.backend.optimizer.optimizer, "step", "optimizer_steps")
        if not read(root / "qualification/ADMISSION.json")["positive_transfer_checked"]:
            from .deferred import install_deferred_check

            deferred = install_deferred_check(
                trainer, output / "deferred_qualification"
            )
        profile = begin_event_profile()
        try:
            result = trainer.train_event(
                "event_005",
                list(paths.values()),
                passes=2,
                presentation_schedule=baseline,
            )
        finally:
            write_json(output / "RESOURCE_COSTS.json", finish_event_profile(profile))
        if deferred is not None:
            progress["deferred_qualification"] = deferred.close()
        work = read_work(ledger.path, limits)
        require_complete_work(work, limits)
        if schedule_bindings(result) != schedule_bindings(baseline):
            raise ValueError("Actual presentation/anchor trajectory changed")
        boundary = trainer.verify_boundary(
            result["checkpoint_path"], output / "BOUNDARY_CHECK.json"
        )
        result.update(
            state_path=str(trainer.latest),
            state_identity=identity(trainer.latest),
            weights_identity=identity(
                Path(result["checkpoint_path"]) / "model.safetensors"
            ),
            boundary=boundary,
            work_ledger=work,
        )
        shared_inputs(plan)
        write_json(output / "RESULT.json", result)
        progress.update(status="completed", final_step=trainer.global_step)
    except BaseException as error:
        failure = error
        if trainer is not None:
            trainer.failed = True
        progress.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        if deferred is not None:
            cleanup(deferred.close, failure, progress, "deferred hooks")
        if trainer is not None:
            cleanup(trainer.close, failure, progress, "trainer close")
        progress.update(
            wall_s=time.perf_counter() - started, work=read_work(ledger.path, limits)
        )
        write_json(output / "STATUS.json", progress)
    return result


if __name__ == "__main__":
    from experiments.role_loss_pilot.campaign import install_signal_handlers

    install_signal_handlers()
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    train_native_event(args.root)
