"""Six disposable native applications, followed by exact origin restoration."""
from pathlib import Path
import hashlib
import inspect
import time
import traceback

from experiments.full_k_gsm8k.training import live_state
from experiments.ok_continuation.resume import differences
from experiments.recent_repair_retention.training import schedule_adapter
from experiments.threatened_repair_replay.qualification import (
    endpoint_receipt, selection_receipt, SCIENTIFIC_FIELDS,
)
from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.storage import write_json
from .state import capture_state, copy_tree, restore_state, verify_state, tensor_change


CASES = ("N", "ALL", "E", "C", "H", "N_repeat")


def gradient_role_check(receipt, role):
    """A naturally supported row must actually carry its permitted gradient."""
    chunks = [chunk for response in receipt["responses"] for chunk in response["chunks"]]
    prefix = sum(chunk["prefix_occurrences"] for chunk in chunks)
    errors = sum(chunk["error_occurrences"] for chunk in chunks)
    gradients = [chunk["gradient"] for chunk in chunks]
    failures = []
    if not prefix or not errors:
        failures.append("fixed_pair_lacks_both_required_roles")
    if any(value is None or not value["finite"] for value in gradients):
        failures.append("missing_or_nonfinite_row_gradient_evidence")
    usable = [value for value in gradients if value is not None]
    if any(value["excluded_nonzero"] for value in usable):
        failures.append("excluded_row_gradient_not_zero")
    active_key = "error_nonzero" if role == "E" else "prefix_nonzero"
    if not sum(value[active_key] for value in usable):
        failures.append("active_role_has_no_nonzero_row_gradient")
    return dict(status="failed" if failures else "passed", role=role,
        prefix_occurrences=prefix, error_occurrences=errors,
        active_nonzero=sum(value[active_key] for value in usable), failures=failures)


def forward_receipt(bundle):
    rows = selection_receipt(bundle)
    # Grad norm is an optimizer result, not a forward-loss/selection equality.
    for row in rows:
        row["metrics"] = {name: value for name, value in row["metrics"].items() if name != "grad_norm"}
    return rows


def zero_adam_reference(trainer, native_gradients):
    """Direct BF16 zero-gradient recurrence, without an optimizer application.

    Tensor operations mirror the installed noncapturable native single/foreach
    AdamW arithmetic. No trainer, optimizer or model state is mutated. This
    independent numeric expectation is separately timed and source-pinned.
    """
    import torch
    import torch.optim.adam as adam_module
    from torch.optim.optimizer import _default_to_fused_or_foreach

    began = time.perf_counter()
    optimizer = trainer.backend.optimizer.optimizer
    if len(optimizer.param_groups) != 1:
        raise ValueError("H numerical reference requires the native single parameter group")
    group = optimizer.param_groups[0]
    unsupported = ("capturable", "differentiable", "fused", "amsgrad", "maximize")
    if any(group.get(name, False) for name in unsupported):
        raise ValueError("H numerical reference does not support a changed optimizer recipe")
    expected_weights = copy_tree(trainer.bundle.draft_model.state_dict())
    expected_backend = copy_tree(trainer.backend.state_dict()["optimizer"])
    stored = expected_backend["optimizer_state_dict"]
    names = {id(p): name for name, p in trainer.bundle.draft_model.named_parameters() if p.requires_grad}
    present = [(names[id(p)], p, sid) for p, sid in zip(group["params"], stored["param_groups"][0]["params"])
               if native_gradients[names[id(p)]] is not None]
    foreach = group.get("foreach")
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach([p for _, p, _ in present], False, use_fused=False)
    beta1, beta2 = group["betas"]
    lr, decay, eps = group["lr"], group["weight_decay"], group["eps"]
    with torch.no_grad():
        params = [p.detach().clone() for _, p, _ in present]
        zeros = [torch.zeros_like(p) for p in params]
        moments1 = [optimizer.state[p]["exp_avg"].detach().clone() for _, p, _ in present]
        moments2 = [optimizer.state[p]["exp_avg_sq"].detach().clone() for _, p, _ in present]
        steps = [optimizer.state[p]["step"].detach().clone() for _, p, _ in present]
        if foreach and params:
            torch._foreach_add_(steps, torch.tensor(1., device="cpu"), alpha=1.)
            if decay:
                torch._foreach_mul_(params, 1 - lr * decay)
            torch._foreach_lerp_(moments1, zeros, 1 - beta1)
            torch._foreach_mul_(moments2, beta2)
            torch._foreach_addcmul_(moments2, zeros, zeros, 1 - beta2)
            divisors = [(1 - beta2 ** float(step)) ** .5 for step in steps]
            sizes = [-(lr / (1 - beta1 ** float(step))) for step in steps]
            denominators = torch._foreach_sqrt(moments2)
            torch._foreach_div_(denominators, divisors)
            torch._foreach_add_(denominators, eps)
            torch._foreach_addcdiv_(params, moments1, denominators, sizes)
        else:
            for parameter, zero, first, second, step in zip(params, zeros, moments1, moments2, steps):
                step.add_(1)
                if decay:
                    parameter.mul_(1 - lr * decay)
                first.lerp_(zero, 1 - beta1)
                second.mul_(beta2).addcmul_(zero, zero, value=1-beta2)
                denominator = (second.sqrt() / (1-beta2**float(step))**.5).add_(eps)
                parameter.addcdiv_(first, denominator, value=-lr/(1-beta1**float(step)))
        for (name, _, sid), parameter, first, second, step in zip(present, params, moments1, moments2, steps):
            expected_weights[name] = parameter.cpu()
            stored["state"][sid].update(exp_avg=first.cpu(), exp_avg_sq=second.cpu(), step=step.cpu())
    source = Path(inspect.getsourcefile(adam_module))
    receipt = dict(kind="direct_zero_current_gradient_tensor_recurrence", optimizer_applications=0,
        model_forwards=0, backwards=0, foreach=bool(foreach), torch_version=torch.__version__,
        installed_adam_source=str(source), installed_adam_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        represented_arithmetic="native parameter/moment dtype; no master weights", parameters=len(present),
        wall_s=time.perf_counter()-began)
    return dict(draft_state_dict=expected_weights, backend={"optimizer": expected_backend}), receipt


def qualify(trainer, feature_paths, output):
    """Run exactly the prescribed first-pair cases; preserve failure evidence."""
    import torch
    from experiments.functional_step_geometry.qualify import cleanup

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    origin = capture_state(trainer)
    began, failure = time.perf_counter(), None
    report = dict(status="running", cases=[], optimizer_steps=0, native_response_presentations=0,
        response_forwards=0, response_backwards=0, complete_response_generations=0,
        startup="separately accounted; no extra qualification optimizer applications")
    native_endpoint = native_gradients = None
    on_cuda = next(trainer.bundle.draft_model.parameters()).device.type == "cuda"
    try:
        features = trainer._prepare_event("event_007", feature_paths, 2)
        trainer._active_event = "qualification_prepared"
        trainer._presentation_schedule = PresentationSchedule(schedule_adapter(trainer.schedule, features),
            [row["record_id"] for row in features], 2, features=features)
        trainer.bundle.model.begin_event()
        order = torch.randperm(len(features)).tolist()
        pair = [features[index] for index in order[:2]]
        pre = capture_state(trainer)
        report["fixed_pair"] = [row["record_id"] for row in pair]
        for name in ("draft_state_dict", "backend"):
            if differences(origin["scientific"][name], pre["scientific"][name]):
                raise ValueError("Qualification preparation mutated weights/optimizer")
            origin["scientific"][name] = pre["scientific"][name]
        for name in CASES:
            directory = output / name
            directory.mkdir()
            restore_state(trainer, pre)
            verify_state(trainer, pre, directory / "PRE_STATE.json")
            if on_cuda:
                torch.cuda.reset_peak_memory_stats()
            case = dict(case=name, status="running")
            report["cases"].append(case)
            write_json(output / "RESULT.json", report)
            if name == "H":
                expectation, case["zero_reference"] = zero_adam_reference(trainer, native_gradients)
                result = trainer.run_history_step(native_gradients, window=1, label="H", category="qualification")
            else:
                role = "N" if name in ("N", "N_repeat") else name
                result = trainer.run_pair(pair, 0, 0, role=role, label=name, category="qualification",
                    capture_gradients=True,
                    reference_gradients=native_gradients if name in ("ALL", "N_repeat") else None)
            report["optimizer_steps"] += result["update"]["optimizer_applications"]
            report["native_response_presentations"] += len(result["rows"])
            report["response_forwards"] += len(result["rows"])
            report["response_backwards"] += len(result["rows"])
            case.update(update=result["update"], selection=selection_receipt(result))
            endpoint = endpoint_receipt(trainer)
            mismatch = []
            if name == "N":
                native_endpoint = copy_tree(endpoint)
                native_gradients = result["preclip_gradients"]["native"]
                native_forward = forward_receipt(result)
                case["native_present"] = [key for key, value in native_gradients.items() if value is not None]
            elif name == "H":
                for field in ("draft_state_dict", "backend"):
                    mismatch += differences(endpoint["scientific"][field], expectation[field], "zero_reference."+field)
                mismatch += differences(endpoint["scientific"]["rng"], pre["scientific"]["rng"], "H_rng")
                if trainer.presentations != pre["scientific"]["presentations"] or trainer.core._micro != pre["core"]["micro"]:
                    mismatch.append("H_consumed_response_presentations")
                if trainer.global_step != pre["scientific"]["global_step"]+1:
                    mismatch.append("H_optimizer_counter")
                case["displacement"] = tensor_change(pre["scientific"]["draft_state_dict"], endpoint["scientific"]["draft_state_dict"])
                del expectation
            else:
                mismatch += differences(forward_receipt(result), native_forward, "native_forward")
                if name in ("ALL", "N_repeat"):
                    mismatch += differences(endpoint, native_endpoint, "native_endpoint")
                else:
                    case["role_gradient_check"] = gradient_role_check(result["role"], name)
                    mismatch += case["role_gradient_check"]["failures"]
                    for field in ("rng", "global_step", "presentations"):
                        mismatch += differences(endpoint["scientific"][field], native_endpoint["scientific"][field], field)
                    for field in ("micro", "remainder", "schedule"):
                        mismatch += differences(endpoint[field], native_endpoint[field], field)
                    case["gradient_vs_native"] = tensor_change(native_gradients, result["preclip_gradients"]["native"])
            case.update(status="failed" if mismatch else "passed", mismatches=mismatch[:64])
            if on_cuda:
                case.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(directory / "RESULT.json", case)
            write_json(output / "RESULT.json", report)
            if mismatch:
                raise ValueError(f"Native prefix-loss qualification {name} differs: {mismatch[:8]}")
            del result, endpoint
        if (report["optimizer_steps"], report["native_response_presentations"]) != (6, 10):
            raise ValueError("Qualification must use six Adam applications and ten presentations; H has none")
        report["status"] = "passed"
    except BaseException as error:
        failure = error
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        write_json(output / "FAILED.json", report)
        raise
    finally:
        cleanup(lambda: restore_state(trainer, origin), failure, report, "restore qualification origin")
        cleanup(lambda: verify_state(trainer, origin, output / "RESTORED_ORIGIN.json"), failure, report, "verify qualification origin")
        report["wall_s"] = time.perf_counter()-began
        cleanup(lambda: write_json(output / "RESULT.json", report), failure, report, "save qualification result")
    return report
