"""Bound the new campaign's peak writes without crediting future deletion."""
import json
from pathlib import Path
import shutil

from online_feedback.storage import write_json

GIB = 2**30


def capacity_plan(weight_bytes):
    """Two initials, one completed arm, and the other arm's atomic overlap.

    Empty-Adam initials each occupy one draft-weight payload. A completed arm
    retains three weight payloads for its model/Adam state and one export. The
    active arm can have two model/Adam states plus two exports at publication.
    Full native and prepared features coexist until verified event commitment.
    """
    rows = 128 * 2272
    features = 2 * rows * 4096 * 6 * 2 + rows * (8 + 4)
    tensor_peak = 14 * weight_bytes + features
    margin_peak = (tensor_peak * 105 + 99) // 100
    allowances = dict(compact_bytes=16 * GIB, compiler_bytes=8 * GIB,
                      workspace_bytes=2 * GIB, reserve_bytes=8 * GIB)
    return {
        "weight_bytes": weight_bytes, "state_weight_equivalents": 14,
        "feature_peak_bytes": features, "tensor_peak_bytes": tensor_peak,
        "tensor_peak_with_margin_bytes": margin_peak,
        **allowances,
        "required_initial_free_bytes": margin_peak + sum(allowances.values()),
        "scope": "Both complete streams; only new artifacts; existing data consume measured free space",
    }


def admit(root, plan, *, validated_tensor_paths=(), receipt_path=None):
    """After recovery validation, deduct only currently present owned tensors.

    Callers must establish chronological scientific boundaries before passing
    recovery files here. Retired or planned-to-be-deleted files give no credit.
    """
    root = Path(root).absolute()
    if root.resolve() != root:
        raise ValueError("New run root must not resolve through a symlink")
    seen, already_present = set(), 0
    for value in validated_tensor_paths:
        path = Path(value).absolute()
        if (not path.is_relative_to(root) or path.resolve() != path or not path.is_file()):
            raise ValueError("Only validated existing tensors inside this run may receive capacity credit")
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode not in seen:
            seen.add(inode)
            already_present += stat.st_size
    compact = sum(p.stat().st_size for p in root.rglob("*")
                  if p.is_file() and not p.is_symlink() and p.suffix not in {".pt", ".safetensors"})
    remaining_tensors = max(0, plan["tensor_peak_with_margin_bytes"] - already_present)
    required = (remaining_tensors + max(0, plan["compact_bytes"] - compact)
                + plan["compiler_bytes"] + plan["workspace_bytes"] + plan["reserve_bytes"])
    free = shutil.disk_usage(root).free
    result = dict(status="passed" if free >= required and compact <= plan["compact_bytes"] else "failed",
                  free_bytes=free, required_additional_and_reserve_bytes=required,
                  validated_existing_tensor_bytes=already_present, compact_bytes=compact,
                  remaining_tensor_bound_bytes=remaining_tensors, no_future_deletion_credit=True,
                  original_full_peak_budget=plan)
    if receipt_path is not None:
        write_json(receipt_path, result)
    if result["status"] != "passed":
        raise ValueError("Current capacity does not admit the unchanged remaining-write bound")
    return result


def boundary_files(campaign_root):
    """Return only initial/final/rolling states named by committed receipts."""
    root = Path(campaign_root)
    files = []
    for arm in ("K", "K_HALF"):
        arm_root = root / "arms" / arm
        initial = arm_root / "trainer/initial_state.pt"
        if initial.exists():
            files.append(initial)
        for receipt in sorted((arm_root / "boundaries").glob("event_[0-9][0-9][0-9].json")):
            data = json.loads(receipt.read_text())
            if data["status"] != "passed":
                raise ValueError("A recorded boundary has not passed validation")
            for name in ("state", "weights"):
                path = Path(data[name])
                if path.exists():
                    files.append(path)
    return files
