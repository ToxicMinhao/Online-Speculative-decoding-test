"""Independent CPU recomputation of Point3's serving statistics and freeze rule.

No experiment modules are imported. This audits retained measurements, not GPU
execution or training-state authenticity. The companion work/state audit covers
those obligations. Supply --exports to verify tensor-content identities locally;
otherwise the identity-dependent decision is conditional on the supplied report.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARMS = ("C", "F_C_SUM", "F_C_MAX")
TIMES = ("request_latency_s", "native_generate_s", "cache_maintenance_s")
COUNTS = ("output_tokens", "verification_calls", "accepted_draft_tokens",
          "native_accepted_draft_tokens", "committed_progress")
CONTRASTS = (("F_C_MAX", "F_C_SUM"), ("F_C_SUM", "C"), ("F_C_MAX", "C"))
SEED, DRAWS = 20261125, 10000
# Audit arithmetic only: floating reduction order can differ by a few ulps.
# These are not new experiment tolerances or outcome-selection allowances.
AUDIT_ATOL, AUDIT_RTOL = 1e-10, 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect(rows):
    """One trajectory for counts; three observations for each timing mean."""
    require(len(rows) == 1188, "Expected 1,188 returned measurements")
    require(sum(r["phase"] == "scored" for r in rows) == 1152,
            "Expected 1,152 scored measurements")
    require(sum(r["phase"] == "warmup" for r in rows) == 36,
            "Expected 36 included warmups")
    require(len({r["physical_gpu_uuid"] for r in rows}) == 1,
            "Serving used different physical cards")
    groups = defaultdict(list)
    for row in rows:
        require(row["valid_measurement"] is True and not row["errors"],
                "Invalid evidence must not enter serving gain claims")
        require(row["method"] in ARMS and row["phase"] in ("scored", "warmup"),
                "Unexpected arm or measurement phase")
        if row["phase"] == "scored":
            groups[row["request_id"], row["method"]].append(row)
    ids = sorted({key[0] for key in groups})
    require(len(ids) == 128 and len(groups) == 384, "Incomplete paired panel")
    for arm in ARMS:
        require(len({r["checkpoint_id"] for r in rows if r["method"] == arm}) == 1,
                "An arm changed serving checkpoint")
    paired = []
    for request_id in ids:
        item = {"request_id": request_id}
        sources = set()
        for arm in ARMS:
            repeats = sorted(groups[request_id, arm], key=lambda r: r["repeat"])
            require([r["repeat"] for r in repeats] == [0, 1, 2],
                    "Missing or duplicate timing repeat")
            first = repeats[0]
            stable = COUNTS + ("finish_reason", "source_id", "checkpoint_id")
            require(all(all(r[k] == first[k] for k in stable) for r in repeats),
                    "Repeat counts, termination, or identities differ")
            sources.add(first["source_id"])
            require(all(type(first[k]) is int and first[k] >= 0 for k in COUNTS),
                    "Counts must be nonnegative integers")
            output, calls = first["output_tokens"], first["verification_calls"]
            progress = first["committed_progress"]
            require(1 <= output <= 2048 and progress == output - 1,
                    "Output/progress identity failed")
            require(first["native_accepted_draft_tokens"] == progress - calls
                    and first["accepted_draft_tokens"] <= progress,
                    "Accepted-token/count identity failed")
            require(calls > 0 or progress == first["accepted_draft_tokens"] == 0,
                    "Zero-call progress is inconsistent")
            require(first["finish_reason"] in ("eos", "length")
                    and (first["finish_reason"] != "length" or output == 2048),
                    "Termination is inconsistent")
            for r in repeats:
                require(all(type(r[k]) in (int, float) and math.isfinite(r[k])
                            and r[k] >= 0 and (k == "cache_maintenance_s" or r[k] > 0)
                            for k in TIMES), "Invalid latency")
                require(r["native_generate_s"] <= r["request_latency_s"],
                        "Native time exceeds complete-request time")
            item[arm] = {key: first[key] for key in COUNTS + ("finish_reason",)}
            item[arm]["repeat_times"] = {k: [r[k] for r in repeats] for k in TIMES}
            for key in TIMES:
                item[arm][key] = sum(item[arm]["repeat_times"][key]) / 3
        require(len(sources) == 1, "Source identity differs across arms")
        require(len({(item[a]["output_tokens"], item[a]["finish_reason"]) for a in ARMS}) == 1,
                "Paired output length or termination differs")
        paired.append(item)
    return paired


def recompute(rows):
    paired = collect(rows)
    values = {a: {k: np.array([r[a][k] for r in paired], dtype=np.float64)
                  for k in TIMES + COUNTS} for a in ARMS}
    draws = np.random.Generator(np.random.PCG64(SEED)).integers(0, 128, (DRAWS, 128))
    totals = {}
    for arm in ARMS:
        total = {k: float(np.sum(v)) for k, v in values[arm].items()}
        calls = total["verification_calls"]
        for name, field in (("native_accepted_per_call", "native_accepted_draft_tokens"),
                            ("stop_clipped_accepted_per_call", "accepted_draft_tokens"),
                            ("committed_per_call", "committed_progress")):
            total[name] = total[field] / calls if calls else None
        total["tokens_per_s"] = total["output_tokens"] / total["request_latency_s"]
        total["repeat_totals"] = {
            k: [sum(r[arm]["repeat_times"][k][repeat] for r in paired) for repeat in range(3)]
            for k in TIMES
        }
        total["termination_counts"] = {
            reason: sum(r[arm]["finish_reason"] == reason for r in paired)
            for reason in ("eos", "length")
        }
        lengths = values[arm]["output_tokens"]
        total["output_length"] = dict(min=int(min(lengths)), max=int(max(lengths)),
                                      mean=float(np.mean(lengths)), median=float(np.median(lengths)))
        totals[arm] = total
    contrasts = {}
    for candidate, reference in CONTRASTS:
        comparison = dict(candidate=candidate, reference=reference)
        for field in TIMES + ("verification_calls",):
            numerator, denominator = values[candidate][field], values[reference][field]
            name = ("verification_call_reduction_pct" if field == "verification_calls"
                    else field + "_reduction_pct")
            with np.errstate(divide="ignore", invalid="ignore"):
                samples = 100 * (1 - numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1))
                point = 100 * (1 - numerator.sum() / denominator.sum())
            if np.isfinite(samples).all() and np.isfinite(point):
                status = "descriptive_unadjusted"
                if field == "request_latency_s":
                    status = ("prespecified_primary_component" if reference == "F_C_SUM"
                              else "required_whole_method")
                result = dict(value=float(point), ci95=np.quantile(samples, [.025, .975],
                              method="linear").tolist(), status=status)
            else:
                result = dict(value=None, ci95=None, status="undefined")
            comparison[name] = result
        call_difference = values[reference]["verification_calls"] - values[candidate]["verification_calls"]
        comparison.update(wins=int(sum(call_difference > 0)), regressions=int(sum(call_difference < 0)),
                          ties=int(sum(call_difference == 0)),
                          gross_calls_saved=int(sum(x for x in call_difference if x > 0)),
                          gross_calls_added=int(-sum(x for x in call_difference if x < 0)),
                          net_calls_saved=int(sum(call_difference)))
        contrasts[candidate + "_vs_" + reference] = comparison
    ids = [r["request_id"] for r in paired]
    encoded_ids = json.dumps(ids, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False).encode()
    return dict(requests=128, paired=paired, totals=totals, contrasts=contrasts,
                bootstrap=dict(draws=DRAWS, generator="PCG64", seed=SEED, input_order=ids,
                    input_order_digest=hashlib.sha256(encoded_ids).hexdigest(),
                    draw_sha256=hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest(),
                    shared_indices_across_all_arms=True))


def select(summary, identities):
    """Apply §9 directly, using no tolerance for the outcome-selection tests."""
    require(set(identities) == set(ARMS), "Missing serving-model identity")
    for identity in identities.values():
        require(identity.get("parameters") and identity.get("inference"), "Empty serving identity")
    same_candidates = identities["F_C_SUM"] == identities["F_C_MAX"]
    totals = summary["totals"]
    candidates = ("F_C_SUM",) if same_candidates else ARMS[1:]
    eligible = [a for a in candidates if identities[a] != identities["C"]
                and totals[a]["request_latency_s"] < totals["C"]["request_latency_s"]
                and totals[a]["verification_calls"] <= totals["C"]["verification_calls"]]
    if eligible:
        selected = sorted(eligible, key=lambda a: (totals[a]["request_latency_s"],
                          totals[a]["verification_calls"], a != "F_C_SUM"))[0]
        ci = summary["contrasts"][selected + "_vs_C"]["request_latency_s_reduction_pct"]["ci95"]
        provisional = ci is None or ci[0] <= 0
    else:
        selected, ci, provisional = "C", None, False
    return dict(selected=selected, eligible=eligible,
                status="provisional_development_selection" if provisional else "development_selection",
                selected_latency_ci95=ci, candidates_have_identical_serving_models=same_candidates,
                model_identity_ignores_optimizer_state=True,
                no_demonstrated_speedup=selected == "C" or provisional,
                further_mechanism_search=False)


def parameter_identity(path):
    """CPU-only native tensor bytes; metadata differences are not model changes."""
    import torch
    from safetensors import safe_open

    result = hashlib.sha256()
    tensors = elements = 0
    with safe_open(str(path), framework="pt", device="cpu") as file:
        for name in sorted(file.keys()):
            tensor = file.get_tensor(name)
            result.update(json.dumps(dict(name=name, shape=list(tensor.shape),
                                          dtype=str(tensor.dtype)), sort_keys=True).encode())
            flat = tensor.reshape(-1)
            for position in range(0, flat.numel(), 262144):
                result.update(flat[position:position + 262144].contiguous().view(torch.uint8).numpy().tobytes())
            tensors += 1
            elements += tensor.numel()
    require(tensors > 0, "Empty serving export")
    return dict(sha256=result.hexdigest(), tensors=tensors, elements=elements)


def compare(expected, actual, path="", errors=None):
    """Compare declared numeric fields; reporting-only extra fields are allowed."""
    errors = [] if errors is None else errors
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(path + ": not an object")
        else:
            for key, value in expected.items():
                if key not in actual:
                    errors.append(path + "/" + key + ": missing")
                else:
                    compare(value, actual[key], path + "/" + key, errors)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            errors.append(path + ": list length/type mismatch")
        else:
            for index, (left, right) in enumerate(zip(expected, actual)):
                compare(left, right, path + "/" + str(index), errors)
    elif isinstance(expected, float):
        if type(actual) not in (int, float) or not math.isclose(expected, actual,
                                                              abs_tol=AUDIT_ATOL, rel_tol=AUDIT_RTOL):
            errors.append(f"{path}: expected {expected!r}, recorded {actual!r}")
    elif expected != actual:
        errors.append(f"{path}: expected {expected!r}, recorded {actual!r}")
    return errors


def audit(rows, summary, exports=None, identity_receipt=None, execution_commit=None):
    recomputed = recompute(rows)
    errors = compare(recomputed, summary)
    identities = summary["serving_identities"]
    verified_exports = {}
    if exports is not None:
        require(set(exports) == set(ARMS), "Supply all three exports")
        for arm in ARMS:
            verified_exports[arm] = parameter_identity(exports[arm])
            errors += compare(verified_exports[arm], summary["parameter_identities"][arm],
                              "parameter_identities/" + arm)
    identity_authentication = None
    if identity_receipt is not None:
        require(identity_receipt["status"] == "passed" and identity_receipt["gpu_work"] == 0,
                "External export authentication did not pass as CPU-only work")
        require(identity_receipt["experimental_modules_imported"] is False
                and identity_receipt["weights_loaded_into_framework"] is False,
                "Expected independent header/raw-byte authentication")
        require(identity_receipt["source"]["commit"] == execution_commit,
                "External export authentication covers a different execution")
        actual = identity_receipt["tensor_content_identities"]
        require(set(actual) == set(ARMS), "External receipt is missing endpoint identities")
        for arm in ARMS:
            errors += compare(actual[arm], summary["parameter_identities"][arm],
                              "externally_authenticated_parameters/" + arm)
        verified_exports = actual
        identity_authentication = dict(
            source_commit=execution_commit, status="passed", gpu_work=0,
            method="Independent safetensors header and raw-byte parser; no framework model load",
            scope="Authenticates stored endpoints; does not independently re-execute training",
        )
    for arm in ARMS:
        require(identities[arm]["parameters"] == summary["parameter_identities"][arm]["sha256"],
                "Decision identity differs from reported exported parameters")
    decision = select(recomputed, identities)
    errors += compare(decision, summary["freeze"], "freeze")
    return dict(status="passed" if not errors else "failed", errors=errors,
                verification="Independent CPU arithmetic and freeze-rule recomputation",
                experiment_modules_imported=False, gpu_work=0,
                numerical_audit_tolerance=dict(absolute=AUDIT_ATOL, relative=AUDIT_RTOL),
                interpretation="Fixed endpoints and reused development panel; no new GPU replication",
                identity_status=("independently_verified_cpu_postflight" if identity_receipt else
                                 "independently_verified_tensor_bytes" if exports else "conditional_on_supplied_identities"),
                identity_authentication=identity_authentication,
                authentication_scope="Full event, canonical outputs and raw receipts require the companion work/state audit",
                verified_exports=verified_exports, requests=128,
                bootstrap=recomputed["bootstrap"], totals=recomputed["totals"],
                contrasts=recomputed["contrasts"], freeze=decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exports", type=Path,
                        help="JSON mapping C/F_C_SUM/F_C_MAX to local model.safetensors paths")
    parser.add_argument("--identity-receipt", type=Path,
                        help="Retained independent CPU header/raw-byte postflight receipt")
    args = parser.parse_args()
    try:
        rows = [json.loads(line) for line in (args.evaluation / "MEASUREMENTS.jsonl").read_text().splitlines()]
        summary = json.loads((args.evaluation / "SUMMARY.json").read_text())
        exports = json.loads(args.exports.read_text()) if args.exports else None
        identity_receipt = json.loads(args.identity_receipt.read_text()) if args.identity_receipt else None
        protocol = json.loads((args.evaluation / "PROTOCOL.json").read_text())
        result = audit(rows, summary, exports, identity_receipt, protocol["execution_commit"])
        if args.identity_receipt:
            result["identity_authentication"]["receipt_sha256"] = hashlib.sha256(args.identity_receipt.read_bytes()).hexdigest()
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = dict(status="failed", error=repr(error), gpu_work=0,
                      scope="CPU retained-evidence audit; no model execution")
    result["auditor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], output=str(args.output),
                          errors=len(result.get("errors", [])))))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
