"""Read-only custody checks for retained author checkpoint replay.

This module reads bounded compact metadata, never optimizer states or model
tensors. The native backend verifies the admitted model bytes before startup.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiment import (SCENARIOS, _sha, load_reference, require,
                        validate_author_reference)

MAX_METADATA_BYTES = 128 * 2**20


def _contained(path, root):
    require(isinstance(path, (str, Path)) and bool(str(path)), "Missing campaign receipt path")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    require(candidate != root and candidate.is_relative_to(root), "Receipt path escapes completed campaign")
    return candidate


def _read(path, expected_sha256, expected_bytes=None):
    require(_sha(expected_sha256), "A valid expected metadata SHA256 is required")
    require(path.is_file() and path.stat().st_size <= MAX_METADATA_BYTES, "Missing or oversized campaign metadata")
    if expected_bytes is not None:
        require(type(expected_bytes) is int and 0 < expected_bytes <= MAX_METADATA_BYTES and
                path.stat().st_size == expected_bytes, "Campaign metadata byte count differs")
    data = path.read_bytes()
    require(len(data) <= MAX_METADATA_BYTES and hashlib.sha256(data).hexdigest() == expected_sha256,
            "Campaign metadata SHA256 differs")
    return data


def _checked(binding, root, actual_path=None):
    require(isinstance(binding, dict), "Campaign file binding is absent")
    require(type(binding.get("bytes")) is int and 0 < binding["bytes"] <= MAX_METADATA_BYTES,
            "Campaign metadata byte count is absent or oversized")
    path = _contained(binding.get("path"), root)
    if actual_path is not None:
        require(_contained(actual_path, root) == path, "Supplied file differs from the campaign path")
    return path, _read(path, binding.get("sha256"), binding.get("bytes"))


def _receipt_paths(value, root):
    """Check even retired paths for containment without requiring their existence."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "path":
                _contained(item, root)
            elif key == "deleted":
                require(isinstance(item, list), "Malformed retired-path receipt")
                for path in item:
                    _contained(path, root)
            else:
                _receipt_paths(item, root)
    elif isinstance(value, list):
        for item in value:
            _receipt_paths(item, root)


def load_author_replay(campaign_root, manifest_sha256, state_sha256, *, episode_id,
                       reference, selection_path, streams_path, runtime_config_path):
    """Return the immutable author result and its completed-campaign custody chain."""
    root = Path(campaign_root).resolve(strict=True)
    require(root.is_dir(), "Completed campaign root is not a directory")
    manifest_path, state_path = root / "MANIFEST.json", root / "STATE.json"
    manifest = json.loads(_read(_contained(manifest_path, root), manifest_sha256))
    state = json.loads(_read(_contained(state_path, root), state_sha256))
    require(Path(manifest.get("campaign_root", "")).resolve() == root, "Manifest campaign root differs")
    require(state.get("status") == "complete" and state.get("manifest_sha256") == manifest_sha256,
            "Campaign state is incomplete or bound to another manifest")
    expected = {f"R{replica}_{scenario}" for replica in (1, 2) for scenario in SCENARIOS}
    episodes = manifest.get("episodes", [])
    require(isinstance(episodes, list) and len(episodes) == 10 and
            {item.get("episode_id") for item in episodes} == expected and episode_id in expected,
            "Replay requires the ten completed R1/R2 episodes")
    completed = state.get("episodes", {})
    require(isinstance(completed, dict) and set(completed) == expected and
            all(item.get("status") == "complete" and type(item.get("returncode")) is int and
                item["returncode"] == 0 for item in completed.values()), "Campaign episodes did not all exit successfully")
    for key, actual in (("selection", selection_path), ("streams", streams_path),
                        ("runtime_config", runtime_config_path)):
        _checked(manifest.get(key), root, actual)
    episode = next(item for item in episodes if item["episode_id"] == episode_id)
    original_path, _ = _checked(episode.get("reference"), root)
    original_binding = reference.get("source_binding", {})
    require(reference.get("episode_id") == episode_id and
            original_binding.get("file_sha256") == episode["reference"]["sha256"] and
            _contained(original_binding.get("path"), root) == original_path,
            "Replay original reference differs from campaign manifest")
    retention_binding = completed[episode_id].get("retention")
    retention_path, retention_bytes = _checked(retention_binding, root)
    retention = json.loads(retention_bytes)
    _receipt_paths(retention, root)
    require(retention.get("episode_id") == episode_id and retention.get("status") == "complete" and
            retention.get("pending") is None, "Completed retention receipt is absent or unfinished")
    result_binding = retention.get("result_binding")
    result_path, result_bytes = _checked(result_binding, root)
    require(result_path == root / "runs" / episode_id / "RESULT.json" and not result_bytes.startswith(b"\x1f\x8b"),
            "Retained result is not the saved uncompressed episode RESULT.json")
    author = load_reference(result_path, result_binding["sha256"])
    validate_author_reference(author, reference)
    for update in author["updates"]:
        checkpoint_path = _contained(update["checkpoint"]["path"], root)
        require(checkpoint_path.is_relative_to(root / "runs" / episode_id / "native"),
                "Retained author checkpoint belongs to another episode")
    provenance = {
        "schema_version": 1, "episode_id": episode_id, "campaign_root": str(root),
        "manifest_binding": {"path": str(manifest_path), "sha256": manifest_sha256},
        "state_binding": {"path": str(state_path), "sha256": state_sha256},
        "selection_binding": manifest["selection"], "streams_binding": manifest["streams"],
        "runtime_config_binding": manifest["runtime_config"], "original_reference_binding": episode["reference"],
        "retention_binding": retention_binding, "author_result_binding": result_binding,
        "checkpoint_bindings": {update["checkpoint_id"]: update["checkpoint"] for update in author["updates"]},
        "scope": ("Completed manifest/state/retention/result custody checked read-only. "
                  "Native admission rehashes retained model files before serving. "
                  "No optimizer state, training or regenerated conversation is required.")}
    return author, provenance
