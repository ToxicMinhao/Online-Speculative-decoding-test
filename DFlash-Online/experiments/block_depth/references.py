"""Reuse authenticated exact-prompt canonical tokens without analyzing outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.o_half_matched_lr.archive import ArchiveReader, validate_record
from experiments.vfd_second_pass.data import reuse_pr25_references, token_digest


def prepare_references(requests, *, pr25_root, pr25_inventory, vfd_root, vfd_inventory):
    references, _, prior_audit = reuse_pr25_references(requests, pr25_root, pr25_inventory)
    reader = ArchiveReader(vfd_root)
    inventory = json.loads(Path(vfd_inventory).read_text())["files"]

    def read(relative):
        path = reader.root / relative
        value = reader.read(path)
        expected = inventory[str(relative)]
        if any(reader.files[str(path)][k] != expected[k] for k in ("bytes", "sha256")):
            raise ValueError("Retained VFD reference evidence differs from archive: " + str(relative))
        return value

    plan = read("sealed/PLAN.json")
    canonical = read("campaign_001/canonicals/CANONICALS.json")
    by_prompt = {token_digest(r["prompt_ids"]): r for r in plan["evaluation_requests"]}
    vfd_reused = []
    for request in requests:
        if request["request_id"] in references:
            continue
        old = by_prompt.get(token_digest(request["prompt_ids"]))
        if old is None:
            continue
        if any(request[key] != old[key] for key in ("prompt_ids", "stop_token_ids", "max_new_tokens")):
            raise ValueError("Retained canonical input/inference settings differ")
        reference = canonical[old["request_id"]]
        if (reference["binding_digest"] != prior_audit["binding_digest"]
                or reference["source"] != "fresh_qualified_verifier_once" or not reference["verified"]):
            raise ValueError("Additional VFD canonical has an unsupported provenance")
        relative = reference["source_record_path"].split("/production_001/", 1)[1]
        raw = read(relative)
        native_relative = raw["metadata"]["native_trace_path"].split("/production_001/", 1)[1]
        native = read(native_relative)
        errors = validate_record(raw, old, raw["checkpoint_id"], reference,
                                 label=f"canonical_{old['panel_ordinal']:04d}", session="vfd_second_pass_evaluation")
        if native != raw["metadata"]["capture"]:
            errors.append("native_sidecar_mismatch")
        if errors:
            raise ValueError("Additional VFD canonical is invalid: " + repr(errors))
        references[request["request_id"]] = dict(
            prompt_ids=request["prompt_ids"], response_ids=raw["response_ids"], finish_reason=raw["finish_reason"],
            verified=True, binding_digest=prior_audit["binding_digest"], source="audited_VFD_exact_prompt",
            source_request_id=old["request_id"],
            source_record=dict(path=str(reader.root / relative), **reader.files[str(reader.root / relative)]),
            source_native=dict(path=str(reader.root / native_relative), **reader.files[str(reader.root / native_relative)]),
        )
        vfd_reused.append(request["request_id"])
    missing = [r["request_id"] for r in requests if r["request_id"] not in references]
    return references, dict(
        status="passed", requested=len(requests), reused=len(references),
        pr25_reused=prior_audit["reused_count"], vfd_reused=len(vfd_reused),
        missing_request_ids=missing, missing_count=len(missing), maximum_new_generations=len(missing),
        target_runtime_binding=prior_audit["binding"], binding_digest=prior_audit["binding_digest"],
        pr25_audit=prior_audit, vfd_checked_files=reader.files,
        vfd_inventory=dict(path=str(vfd_inventory), sha256=hashlib.sha256(Path(vfd_inventory).read_bytes()).hexdigest()),
        model_work=0, endpoint_performance_analysis=False,
        note="Canonical output IDs/termination only; historical timing and acceptance outcomes are not analyzed or selected.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pr25-root", "pr25-inventory", "vfd-root", "vfd-inventory", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    requests = json.loads(Path(args.manifest).read_text())["manifests"]["FINAL"]["requests"]
    references, audit = prepare_references(requests, pr25_root=args.pr25_root, pr25_inventory=args.pr25_inventory,
        vfd_root=args.vfd_root, vfd_inventory=args.vfd_inventory)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("CANONICALS.json", references), ("CANONICAL_REUSE_AUDIT.json", audit)):
        path = output / name
        if path.exists():
            raise FileExistsError("Preserve existing preparation: " + str(path))
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: audit[key] for key in ("status", "requested", "reused", "pr25_reused", "vfd_reused", "missing_count", "model_work")}))


if __name__ == "__main__":
    main()
