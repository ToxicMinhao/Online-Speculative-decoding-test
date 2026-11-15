"""PR46 complete-state restoration, extended only for the role observer."""
from copy import deepcopy

from experiments.ok_continuation.resume import differences
from experiments.threatened_repair_replay import state as original
from online_feedback.storage import write_json

copy_tree = original.copy_tree
named_gradients = original.named_gradients
tensor_change = original.tensor_change
compare_final_native = original.compare_final_native


def capture_state(trainer):
    value = original.capture_state(trainer)
    value["role_observer"] = deepcopy(vars(trainer.roles))
    value["role_installed"] = getattr(trainer.bundle.model, "native_prefix_roles", None) is trainer.roles
    return value


def restore_state(trainer, snapshot):
    original.restore_state(trainer, snapshot)
    vars(trainer.roles).clear()
    vars(trainer.roles).update(deepcopy(snapshot["role_observer"]))
    trainer.bundle.model.native_prefix_roles = trainer.roles if snapshot["role_installed"] else None


def verify_state(trainer, snapshot, receipt_path=None):
    # Save the expanded result even if the original checker raises.
    try:
        result = original.verify_state(trainer, snapshot)
        mismatch = differences(vars(trainer.roles), snapshot["role_observer"], "role_observer")
        if (getattr(trainer.bundle.model, "native_prefix_roles", None) is trainer.roles) != snapshot["role_installed"]:
            mismatch.append("role_observer.binding")
        result.update(status="failed" if mismatch else "passed", mismatches=mismatch)
    except BaseException as error:
        result = dict(status="failed", error=repr(error))
        if receipt_path is not None:
            write_json(receipt_path, result)
        raise
    if receipt_path is not None:
        write_json(receipt_path, result)
    if mismatch:
        raise ValueError("Role observer state differs: " + str(mismatch[:8]))
    return result
