"""Bind unchanged panel outputs to authenticated PR32 native evidence.

Preparation copied the original compact canonical dictionaries, which predate
PR32's prompt/runtime authentication fields. This adapter derives those fields
again from the original records; it does not generate or replace any output.
Recorded remote paths are provenance. Callers copy the returned local evidence
into the new seal instead of assuming that historical remote paths still exist.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.ok_coding.storage import identity
from experiments.o_half_matched_lr.archive import validate_record


def _read(path):
    return json.loads(Path(path).read_text())


def load_pr32_proof(prior_sealed, dataset, *, model_pin, runtime_pin):
    """Read original proof files after checking their retained seal identities.

    ``model_pin`` and ``runtime_pin`` are the current qualification's actual
    files. Their bytes must equal PR32's pinned files before its canonical
    evidence can be reused. Returned file rows support explicit new-seal copies.
    """
    if dataset not in ("gsm8k", "mathqa"):
        raise ValueError("Unknown original dataset")
    root = Path(prior_sealed)
    plan = _read(root / "PLAN.json")
    seal = _read(root / "SEAL.json")
    remote = Path(plan["root"]) / "sealed"
    files = []
    for name in ("PLAN.json", "MODEL_PIN.json", "RUNTIME_PIN.json",
                 "REFERENCE_AUDIT.json", dataset + "/CANONICALS.json",
                 dataset + "/PANEL.json"):
        local = root / name
        recorded = str(remote / name)
        observed = identity(local)
        if seal["files"].get(recorded) != observed:
            raise ValueError("Original PR32 proof identity changed: " + name)
        files.append(dict(recorded_path=recorded, local_path=str(local), identity=observed))
    files.append(dict(recorded_path=str(remote / "SEAL.json"),
                      local_path=str(root / "SEAL.json"), identity=identity(root / "SEAL.json")))
    if (identity(model_pin) != identity(root / "MODEL_PIN.json")
            or identity(runtime_pin) != identity(root / "RUNTIME_PIN.json")):
        raise ValueError("Current model/runtime pins differ from canonical provenance")
    audit = _read(root / "REFERENCE_AUDIT.json")
    if (audit.get("status") != "passed" or audit.get("extra_generations") != 0
            or audit.get("records") != 512 or len(audit["files"]) != 512):
        raise ValueError("Require the complete original native-reference audit")
    references = _read(root / dataset / "CANONICALS.json")
    panel = _read(root / dataset / "PANEL.json")["requests"]
    if len(panel) != 256 or set(references) != {row["request_id"] for row in panel}:
        raise ValueError("Original PR32 panel/reference membership changed")
    for request in panel:
        row = references[request["request_id"]]
        source = row["source_record"]
        expected = {key: source[key] for key in ("bytes", "sha256")}
        if (row.get("verified") is not True or not row.get("binding_digest")
                or row.get("prompt_ids") != request["prompt_ids"]
                or source["path"] != row["source_record_path"]
                or audit["files"].get(source["path"]) != expected
                or seal["files"].get(source["path"]) != expected):
            raise ValueError("Original canonical source binding changed: " + request["request_id"])
    return references, audit["files"], dict(
        status="passed", dataset=dataset, files=files,
        original_execution_source=plan["execution_source"],
        current_model_runtime_exact=True, new_model_work=0)


def bind_canonicals(requests, inherited, *, authenticated, source_identities,
                    mirror, origin, binding):
    """Revalidate exact native prompts, outputs, termination and full traces.

    Only authentication metadata is added. Every inherited canonical field is
    required to agree with PR32, and both input dictionaries remain unchanged.
    ``mirror`` is an explicit byte-preserving copy of ``origin``; there is no
    fallback search or substitute source record.
    """
    request_ids = [request["request_id"] for request in requests]
    if (not request_ids or len(set(request_ids)) != len(request_ids)
            or set(inherited) != set(request_ids) or set(authenticated) != set(request_ids)):
        raise ValueError("Canonical request membership differs from the sealed panel")
    if (not isinstance(binding, str) or len(binding) != 64
            or any(value not in "0123456789abcdef" for value in binding)):
        raise ValueError("Require the caller's explicit sealed canonical binding")
    result, records = {}, []
    mirror, origin = Path(mirror), Path(origin)
    for request in requests:
        key = request["request_id"]
        original, proof = inherited[key], authenticated[key]
        if any(proof.get(name) != value for name, value in original.items()):
            raise ValueError("Inherited canonical changed since PR32: " + key)
        source = proof["source_record"]
        expected = {name: source[name] for name in ("bytes", "sha256")}
        recorded = Path(original["source_record_path"])
        if (proof.get("verified") is not True or not proof.get("binding_digest")
                or proof.get("prompt_ids") != request["prompt_ids"]
                or source["path"] != str(recorded)
                or source_identities.get(str(recorded)) != expected):
            raise ValueError("Authenticated source binding differs: " + key)
        relative = recorded.relative_to(origin)
        if ".." in relative.parts:
            raise ValueError("Source path escapes its explicit mirror")
        local = mirror / relative
        payload = local.read_bytes()
        observed = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        if observed != expected:
            raise ValueError("Retained native source identity changed: " + key)
        raw = json.loads(payload)
        original_ordinal = request.get("inherited_panel_ordinal", raw["request_index"])
        errors = validate_record(raw, dict(request, panel_ordinal=original_ordinal),
                                 raw["checkpoint_id"], original,
                                 label=raw["record_id"], session=raw["session_id"])
        if errors:
            raise ValueError(f"Invalid retained canonical {key}: {errors}")
        result[key] = dict(deepcopy(original), prompt_ids=deepcopy(raw["prompt_ids"]),
                           verified=True, binding_digest=binding,
                           source_record=dict(path=str(recorded), **observed))
        records.append(dict(request_id=key, recorded_path=str(recorded),
            local_path=str(local), identity=observed, original_record_id=raw["record_id"],
            original_session_id=raw["session_id"], original_request_index=raw["request_index"],
            checkpoint_id=raw["checkpoint_id"], verified=True))
    return result, dict(status="passed", records=records, count=len(records),
        binding_digest=binding, outputs_and_termination_unchanged=True,
        prompt_ids_checked_against_native=True, full_native_traces_checked=True,
        new_generations=0, new_model_work=0)
