"""Storage admission and narrowly scoped retirement for the persistent OSD run.

The caller owns event ordering. Retirement is allowed only after the ordinary
controller commit and an explicit, checkpoint-bound native publication check.
Nothing here retries training, repairs a failed event, or changes its evidence.
"""
import hashlib
import json
from pathlib import Path
import shutil

from online_feedback.storage import write_json


GIB = 2**30


def _inside(path, root, *, directory=False):
    """Reject absent paths, symlinks and paths outside this campaign."""
    path = Path(path).absolute()
    root = Path(root).absolute()
    if root.resolve() != root or path.resolve() != path or not path.is_relative_to(root):
        raise ValueError("Path escapes campaign or contains a symlink: " + str(path))
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError("Required campaign artifact is absent: " + str(path))
    return path


def _json(path, root):
    path = _inside(path, root)
    return json.loads(path.read_text())


def _weight_bytes(directory):
    paths = sorted(Path(directory).glob("*.safetensors"))
    if not paths or any(not path.is_file() or path.stat().st_size <= 0 for path in paths):
        raise ValueError("Nonempty draft safetensors are required")
    return sum(path.stat().st_size for path in paths)


def admit_storage(run_root, draft_path, *, max_sequence_length,
                  responses_per_event=128, hidden_size=4096, target_layer_count=5,
                  current_state=None, current_export=None,
                  compact_allowance_bytes=8 * GIB, work_allowance_bytes=2 * GIB,
                  reserve_bytes=8 * GIB, receipt_path=None):
    """Admit the remaining writes, including atomic replacement overlap.

    Two complete feature copies remain through commit. Two BF16 model/Adam
    states and two exports coexist at the next event boundary. Existing active
    files already consume filesystem space, so deduct their actual sizes from
    the additional-write requirement. Compact evidence gets a separate bounded
    cumulative allowance; it is never assumed to fit in the work allowance.
    """
    root = _inside(run_root, run_root, directory=True)
    for value in (max_sequence_length, responses_per_event, hidden_size, target_layer_count):
        if type(value) is not int or value <= 0:
            raise ValueError("Positive integer sequence, response and hidden dimensions required")
    for value in (compact_allowance_bytes, work_allowance_bytes, reserve_bytes):
        if type(value) is not int or value < 0:
            raise ValueError("Nonnegative integer storage allowances required")
    weights = _weight_bytes(draft_path)
    state_bytes = _inside(current_state, root).stat().st_size if current_state else 0
    export_bytes = 0
    if current_export:
        export = _inside(current_export, root, directory=True)
        for path in export.glob("*.safetensors"):
            _inside(path, root)
        export_bytes = _weight_bytes(export)

    # Terminal rows compensate for the last row absent from each native capture.
    rows = responses_per_event * max_sequence_length
    features = 2 * rows * hidden_size * (target_layer_count + 1) * 2
    features += rows * (8 + 4)  # Prepared int64 IDs and FP32 masks.
    state_bound = max(3 * weights, state_bytes)
    export_bound = max(weights, export_bytes)
    tensor_peak = 2 * state_bound + 2 * export_bound + features
    tensor_with_margin = (tensor_peak * 105 + 99) // 100
    already_present = state_bytes + export_bytes

    compact_present = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Campaign storage must not contain symlinks: " + str(path))
        if path.is_file() and path.suffix not in {".pt", ".safetensors"}:
            compact_present += path.stat().st_size
    remaining_compact = max(0, compact_allowance_bytes - compact_present)
    required = tensor_with_margin - already_present + remaining_compact + work_allowance_bytes + reserve_bytes
    free = shutil.disk_usage(root).free
    result = {
        "scope": "next event; two states, two exports, native and prepared features",
        "released_safetensors_bytes": weights,
        "max_sequence_length": max_sequence_length,
        "responses_per_event": responses_per_event,
        "feature_tensor_bound_bytes": features,
        "trainer_state_bound_bytes": state_bound,
        "export_bound_bytes": export_bound,
        "tensor_peak_bytes": tensor_peak,
        "tensor_peak_with_five_percent_margin_bytes": tensor_with_margin,
        "existing_current_state_bytes": state_bytes,
        "existing_current_export_bytes": export_bytes,
        "remaining_tensor_writes_bytes": tensor_with_margin - already_present,
        "compact_cumulative_allowance_bytes": compact_allowance_bytes,
        "compact_already_present_bytes": compact_present,
        "remaining_compact_allowance_bytes": remaining_compact,
        "work_allowance_bytes": work_allowance_bytes,
        "reserve_bytes": reserve_bytes,
        "required_additional_and_reserve_bytes": required,
        "free_bytes": free,
        "admitted": free >= required and compact_present <= compact_allowance_bytes,
    }
    if receipt_path is not None:
        destination = Path(receipt_path).absolute()
        if destination.resolve() != destination or not destination.is_relative_to(root):
            raise ValueError("Storage receipt must be inside campaign")
        write_json(destination, result)
    if not result["admitted"]:
        raise ValueError("Insufficient peak storage or cumulative compact evidence allowance")
    return result


def checked_retire(event_root, records, trained_receipt, previous_export, current_runroot):
    """Retire only event tensors and the preceding export's model.safetensors.

    event_root contains loop/ plus PUBLICATION_CHECK.json. The driver marker has
    status='passed', all_equal=True, event_id, and checkpoint_path. Both a full
    controller receipt and its training member are accepted as trained_receipt.
    The current state/export and all compact JSON remain untouched.
    """
    root = _inside(current_runroot, current_runroot, directory=True)
    event_root = _inside(event_root, root, directory=True)
    if (event_root / "RETIREMENT_PLAN.json").exists() or (event_root / "RETIREMENT.json").exists():
        raise ValueError("Retirement already attempted; inspect its preserved receipt")
    training = trained_receipt.get("training", trained_receipt)
    event_id = training["event_id"]
    if not isinstance(event_id, str) or Path(event_id).name != event_id or event_id in {".", ".."}:
        raise ValueError("Unsafe event identifier")
    current_export = _inside(training["checkpoint_path"], root, directory=True)
    current_state = _inside(training["full_state_path"], root)
    current_weights = _inside(current_export / "model.safetensors", root)
    if not current_state.stat().st_size or not current_weights.stat().st_size:
        raise ValueError("Current recovery state and export must be nonempty")
    controller = _json(event_root / "loop/events" / (event_id + ".json"), root)
    buffer = _json(event_root / "loop/feedback_buffer/state.json", root)
    marker = _json(event_root / "PUBLICATION_CHECK.json", root)
    if training.get("status") != "completed" or controller.get("training") != training:
        raise ValueError("Training receipt differs from committed event")
    if _json(current_export.parent / "RESULT.json", root) != training:
        raise ValueError("Saved trainer result differs from committed event")
    ids = [record["record_id"] for record in records]
    if len(set(ids)) != len(ids) or not ids or buffer.get("records") != records:
        raise ValueError("Retirement records differ from the event buffer")
    events = buffer.get("events", [])
    if (len(events) != 1 or events[0].get("event_id") != event_id
            or events[0].get("status") != "committed" or events[0].get("record_ids") != ids
            or events[0].get("receipt") != controller
            or events[0].get("checkpoint_id") != str(current_export)):
        raise ValueError("Retirement requires the exact committed buffer event")
    if (marker.get("status") != "passed" or marker.get("all_equal") is not True
            or marker.get("event_id") != event_id
            or marker.get("checkpoint_path") != str(current_export)):
        raise ValueError("Retirement requires a passed current native publication check")

    compact_paths = set(event_root.rglob("*.json"))
    compact_paths.add(current_export.parent / "RESULT.json")
    protected = {current_state, current_weights}
    candidates = []
    preparation = controller.get("preparation", [])
    if [item.get("record_id") for item in preparation] != ids:
        raise ValueError("Prepared feature identities differ from the committed records")
    for record, prepared in zip(records, preparation):
        record_id = record["record_id"]
        if not isinstance(record_id, str) or Path(record_id).name != record_id or record_id in {".", ".."}:
            raise ValueError("Unsafe record identifier")
        capture = _inside(record["metadata"]["capture"]["feature_path"], root)
        feature = _inside(prepared["feature_path"], root)
        terminal = _inside(prepared["terminal_feature"]["tensor_path"], root)
        expected = event_root / "loop/features" / (record_id + ".pt")
        if (feature != expected or terminal != expected.with_suffix(".terminal.pt")
                or capture.suffix != ".pt" or not capture.name.startswith("request_")):
            raise ValueError("Unexpected event tensor path")
        capture_json = capture.with_suffix(".json")
        raw_capture = _json(capture_json, root)
        if (raw_capture.get("prompt_ids") != record["prompt_ids"]
                or raw_capture.get("response_ids") != record["response_ids"]
                or raw_capture.get("feature_path") != str(capture)):
            raise ValueError("Native compact evidence does not bind the retired capture")
        if _json(feature.with_suffix(".json"), root) != prepared:
            raise ValueError("Prepared compact evidence differs from committed receipt")
        terminal_json = terminal.with_suffix(".json")
        if _json(terminal_json, root) != prepared["terminal_feature"]:
            raise ValueError("Terminal compact evidence differs from committed receipt")
        compact_paths.update([capture_json, feature.with_suffix(".json"), terminal_json])
        candidates.extend([(capture, "native_capture"), (feature, "prepared_feature"),
                           (terminal, "terminal_feature")])
    if previous_export is not None:
        previous = _inside(previous_export, root, directory=True)
        if previous == current_export or previous in current_export.parents:
            raise ValueError("Current export cannot be retired")
        previous_result_path = previous.parent / "RESULT.json"
        previous_result = _json(previous_result_path, root)
        if (previous_result.get("status") != "completed"
                or previous_result.get("checkpoint_path") != str(previous)
                or any(record.get("checkpoint_id") != str(previous) for record in records)):
            raise ValueError("Previous export is not the completed checkpoint that served this event")
        compact_paths.add(previous_result_path)
        compact_paths.update(previous.glob("*.json"))
        candidates.append((_inside(previous / "model.safetensors", root), "previous_export_weights"))

    paths = [path for path, _ in candidates]
    if len(set(paths)) != len(paths) or any(path in protected for path in paths):
        raise ValueError("Duplicate or protected retirement path")
    # Parse and hash all compact JSON before the first deletion.
    compact = []
    for path in sorted(compact_paths):
        path = _inside(path, root)
        content = path.read_bytes()
        json.loads(content)
        compact.append({"path": str(path), "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    entries = []
    for path, kind in candidates:
        stat = path.stat()
        if stat.st_nlink != 1:
            raise ValueError("Retirement refuses hardlinked tensor artifacts: " + str(path))
        entries.append({"path": str(path), "kind": kind, "bytes": stat.st_size,
                        "device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns})
    plan = {"status": "planned", "event_id": event_id, "files": entries,
            "compact_evidence": compact, "retained_current_state": str(current_state),
            "retained_current_export": str(current_export),
            "free_before_bytes": shutil.disk_usage(root).free,
            "reason": "Event committed, current native publication verified, compact evidence parsed and hashed; these tensors are no longer active dependencies.",
            "lost_capability": "Retired event tensors cannot be replayed; retired prior weights cannot be reloaded. They were deleted, not archived."}
    write_json(event_root / "RETIREMENT_PLAN.json", plan)
    removed = []
    try:
        for entry in entries:
            path = _inside(entry["path"], root)
            stat = path.stat()
            if (stat.st_size, stat.st_dev, stat.st_ino, stat.st_mtime_ns) != (
                    entry["bytes"], entry["device"], entry["inode"], entry["mtime_ns"]):
                raise ValueError("Artifact changed after retirement planning: " + str(path))
            path.unlink()
            removed.append(entry)
    except BaseException as error:
        write_json(event_root / "RETIREMENT.json", {
            **plan, "status": "failed", "removed": removed, "error": repr(error),
            "free_after_bytes": shutil.disk_usage(root).free})
        raise
    result = {**plan, "status": "completed", "removed": removed,
              "removed_bytes": sum(entry["bytes"] for entry in removed),
              "free_after_bytes": shutil.disk_usage(root).free}
    write_json(event_root / "RETIREMENT.json", result)
    return result
