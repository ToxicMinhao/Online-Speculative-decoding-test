"""Pure CPU scoring and shared validity gates for matched canonical starts.

These are local draft observations, never counterfactual target verification.
No model, tensor framework, optimizer, or experiment is started on import.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

ARMS = ("F", "S8", "L8")
ADAPTED_ARMS = ("S8", "L8")
POPULATIONS = ("heldout", "source")
BLOCK_SIZE = 10
PROPOSAL_COUNT = 9
MAX_ANCHORS_PER_REQUEST = 8
MAX_LOCATIONS = 240
MAX_BLOCK_EVALUATIONS = 720
SCHEMA = "dflash_matched_prefix_diagnostic_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value: Any):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _tokens(value):
    return isinstance(value, list) and all(type(item) is int and item >= 0 for item in value)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def prefix_score(proposals, canonical, eos_ids, *, response_boundary="cap",
                 reaches_response_boundary=False):
    """Count the correct proposed prefix, INCLUDING a correctly matched EOS.

    A matching suffix exhausted at the response cap is right censored, even
    when its length is exactly nine. An EOS is an observed stopping boundary,
    never a rejection. The known seed/correction/bonus are absent from inputs.
    """
    require(_tokens(proposals) and _tokens(canonical) and _tokens(eos_ids),
            "Prefix scoring requires nonnegative integer token lists")
    require(len(canonical) <= len(proposals) <= PROPOSAL_COUNT,
            "Only available canonical proposed positions can be scored")
    require(response_boundary in ("eos", "cap", "unknown"), "Unknown response boundary")
    matched = non_eos = 0
    matched_eos = False
    first_mismatch = None
    eos = set(eos_ids)
    available = next((i + 1 for i, token in enumerate(canonical) if token in eos), len(canonical))
    for index, (proposal, token) in enumerate(zip(proposals, canonical[:available])):
        if proposal != token:
            first_mismatch = index
            break
        matched += 1
        if token in eos:
            matched_eos = True
            break
        non_eos += 1
    exhausted = first_mismatch is None and not matched_eos
    censored = bool(exhausted and reaches_response_boundary and response_boundary != "eos")
    if first_mismatch is not None:
        boundary = "mismatch"
    elif matched_eos:
        boundary = "matched_eos"
    elif censored:
        boundary = "cap_censored" if response_boundary == "cap" else "unknown_end_censored"
    elif len(canonical) < len(proposals):
        boundary = "available_canonical_end_censored"
        censored = True
    else:
        boundary = "block_end"
    return dict(correct_proposed_prefix_length=matched, non_eos_prefix_length=non_eos,
                matched_eos=matched_eos, first_mismatch_position=first_mismatch,
                boundary=boundary, censored=censored, compared_positions=available,
                available_canonical_positions=available,
                unscored_proposal_positions=len(proposals) - available,
                available_suffix_censored=bool(reaches_response_boundary and response_boundary != "eos"),
                response_boundary=response_boundary)


def anchor_boundary(anchor):
    output, eos = anchor["output_ids"], set(anchor["eos_ids"])
    boundary = anchor.get("response_boundary", "eos" if output and output[-1] in eos else "cap")
    reaches = anchor["block_output_start"] + len(anchor["canonical_next_ids"]) == len(output)
    return dict(response_boundary=boundary, reaches_response_boundary=reaches)


def _validate_row(anchor, arm, row):
    """Recompute strict row validity; malformed/failed rows return errors.

    Both the execution controller and offline reporting use this same gate.
    State hashes bind parameter/buffer identity and mutation versions; they are
    explicitly not represented as independently recomputed weight-value hashes.
    """
    errors = []

    def check(ok, message):
        if not ok:
            errors.append(message)

    if not isinstance(anchor, dict) or not isinstance(row, dict):
        return {"status": "failed", "errors": ["anchor_and_row_must_be_objects"]}
    check(arm in ARMS, "unknown_arm")
    check(row.get("status") == "passed", "native_status_not_passed")
    for key in ("anchor_id", "group_id", "domain", "population", "request_id"):
        check(key in anchor and row.get(key) == anchor.get(key), "identity_mismatch:" + key)
    check(row.get("arm") == arm, "identity_mismatch:arm")
    try:
        inputs, output = anchor["input_ids"], anchor["output_ids"]
        start, canonical = anchor["block_output_start"], anchor["canonical_next_ids"]
        require(_tokens(inputs) and inputs and _tokens(output) and output, "invalid_canonical_token_ids")
        require(type(start) is int and 1 <= start <= len(output), "invalid_block_start")
        require(_tokens(anchor["eos_ids"]), "invalid_eos_ids")
        prefix = inputs + output[:start]
        positions = list(range(len(inputs) + start, len(inputs) + start + PROPOSAL_COUNT))
        check(anchor.get("canonical_prefix_ids") == prefix, "anchor_prefix_geometry")
        check(anchor.get("canonical_prefix_sha256") == digest(prefix), "anchor_prefix_hash")
        check(anchor.get("seed_token_id") == output[start - 1], "anchor_seed_geometry")
        check(canonical == output[start:start + PROPOSAL_COUNT], "anchor_canonical_suffix_geometry")
        check(anchor.get("expected_absolute_positions") == positions, "anchor_absolute_positions")
        check(not any(t in set(anchor["eos_ids"]) for t in output[:-1]), "canonical_tokens_after_eos")
        check(anchor_boundary(anchor)["response_boundary"] in ("eos", "cap"), "invalid_response_boundary")
        if anchor_boundary(anchor)["response_boundary"] == "eos":
            check(output[-1] in set(anchor["eos_ids"]), "eos_boundary_without_eos")
        else:
            check(output[-1] not in set(anchor["eos_ids"]), "cap_boundary_with_eos")
        if "reaches_response_boundary" in anchor:
            check(anchor["reaches_response_boundary"] == anchor_boundary(anchor)["reaches_response_boundary"],
                  "anchor_boundary_geometry")
        check(row.get("canonical_prefix_ids") == prefix, "exact_prefix_mismatch")
        check(row.get("canonical_prefix_sha256") == digest(prefix), "row_prefix_hash")
        check(row.get("prefix_input_ids_sha256") == digest(prefix[:-1]), "cold_prefill_prefix_hash")
        check(row.get("seed_token_id") == output[start - 1], "seed_mismatch")
        check(row.get("reconstructed_target_seed_id") == output[start - 1], "reconstructed_target_seed_mismatch")
        check(row.get("native_absolute_proposal_positions") == positions, "native_absolute_positions")
        check(row.get("native_materialized_block_ids") == [output[start - 1]] + [128002] * PROPOSAL_COUNT,
              "native_seed_and_mask_geometry")
        expected_checkpoint = anchor["checkpoints"][arm]["checkpoint_identity_sha256"]
        check(_sha(expected_checkpoint), "invalid_checkpoint_binding")
        for key in ("checkpoint_identity_sha256", "cache_checkpoint_identity_sha256",
                    "helper_checkpoint_identity_sha256"):
            check(row.get(key) == expected_checkpoint, key + "_mismatch")
        if "checkpoint" in row:
            check(row["checkpoint"] == anchor["checkpoints"][arm], "checkpoint_binding_mismatch")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        errors.append("malformed_anchor:" + str(error))
        canonical, positions = [], []
    for flag in ("target_seed_matches", "valid_reconstruction", "cache_valid", "helper_refresh_valid",
                 "immutable_weights", "target_weights_unchanged", "draft_weights_unchanged", "eval_mode",
                 "inference_mode", "no_generated_history", "no_training", "native_hook_passed", "native_logits_finite"):
        check(row.get(flag) is True, "failed_or_missing:" + flag)
    for key in ("native_draft_block_evaluations", "draft_block_evaluations", "native_head_calls"):
        check(type(row.get(key)) is int and row[key] == 1, "not_exactly_one:" + key)
    for key in ("target_verifications", "regenerated_response_tokens", "training_steps",
                "optimizer_creations", "optimizer_loads", "nonfinite_logit_count"):
        check(type(row.get(key)) is int and row[key] == 0, "nonzero_or_missing:" + key)
    for key, expected in (("native_feature_dtype", "torch.bfloat16"), ("native_logits_dtype", "torch.bfloat16"),
                          ("inspection_reduction_dtype", "torch.float32")):
        check(row.get(key) == expected, "native_precision_mismatch:" + key)
    chunks = row.get("target_prefill_chunks")
    check(isinstance(chunks, list) and bool(chunks) and
          all(type(n) is int and n > 0 for n in chunks), "invalid_prefill_chunks")
    if isinstance(chunks, list) and all(type(n) is int for n in chunks) and _tokens(anchor.get("canonical_prefix_ids")):
        check(sum(chunks) == len(anchor["canonical_prefix_ids"]) - 1, "prefill_geometry_mismatch")
    check(row.get("state_hash_kind") == "parameter_and_buffer_identity_version_signature", "state_hash_kind")
    for model in ("target", "draft"):
        before, after = row.get(model + "_state_before_sha256"), row.get(model + "_state_after_sha256")
        check(_sha(before) and _sha(after) and before == after, model + "_state_mutated_or_missing")
    proposals = row.get("proposed_token_ids")
    check(_tokens(proposals) and len(proposals) == PROPOSAL_COUNT, "proposal_geometry")
    for key in ("proposal_ids", "native_argmax_ids"):
        check(row.get(key) == proposals and _tokens(row.get(key)), key + "_mismatch")
    if arm == "F":
        expected = anchor.get("expected_frozen_proposal_ids")
        check(_tokens(expected) and len(expected) == PROPOSAL_COUNT, "retained_frozen_geometry")
        check(proposals == expected, "all_nine_frozen_proposals_differ_from_retained")
        check(row.get("frozen_proposals_match_retained") is True, "frozen_reconstruction_flag")
    details = row.get("positions")
    check(isinstance(details, list) and len(details) == len(canonical), "position_geometry")
    if isinstance(details, list) and _tokens(proposals):
        for index, detail in enumerate(details):
            if not isinstance(detail, dict) or index >= len(canonical) or index >= len(proposals):
                errors.append("malformed_position:" + str(index))
                continue
            def poscheck(ok, message):
                check(ok, f"position_{index}:" + message)
            for key, expected in (("block_position", index), ("native_block_slot", index + 1),
                                  ("absolute_output_position", anchor.get("block_output_start", 0) + index),
                                  ("absolute_position", positions[index] if index < len(positions) else None),
                                  ("canonical_token_id", canonical[index]), ("proposal_token_id", proposals[index]),
                                  ("native_argmax_id", proposals[index])):
                poscheck(type(detail.get(key)) is int and detail[key] == expected, "geometry_or_argmax:" + key)
            for key in ("native_canonical_log_probability", "native_bf16_canonical_margin",
                        "native_canonical_logit", "native_best_other_logit"):
                poscheck(_finite(detail.get(key)), "nonfinite_or_missing:" + key)
            logp, margin = detail.get("native_canonical_log_probability"), detail.get("native_bf16_canonical_margin")
            ylogit, other = detail.get("native_canonical_logit"), detail.get("native_best_other_logit")
            if all(_finite(value) for value in (logp, margin, ylogit, other)):
                poscheck(logp <= 0, "positive_log_probability")
                poscheck(math.isclose(margin, ylogit - other, rel_tol=1e-6, abs_tol=1e-7), "margin_reduction_mismatch")
                poscheck(margin <= 0 or proposals[index] == canonical[index], "positive_margin_argmax_mismatch")
                poscheck(margin >= 0 or proposals[index] != canonical[index], "negative_margin_argmax_mismatch")
                tied = detail.get("canonical_tied_for_max")
                poscheck(type(tied) is bool and tied == (margin == 0), "canonical_tie_mismatch")
                if margin == 0:
                    poscheck(type(detail.get("native_argmax_tie_count")) is int and detail["native_argmax_tie_count"] >= 2,
                             "missing_tie_count")
            count = detail.get("native_argmax_tie_count")
            poscheck(type(count) is int and count >= 1, "invalid_native_tie_count")
            if _finite(margin) and margin > 0:
                poscheck(count == 1, "unique_canonical_max_has_ties")
            best_other = detail.get("native_best_other_id")
            poscheck(type(best_other) is int and best_other >= 0 and best_other != canonical[index], "invalid_best_other")
            if _finite(margin) and type(best_other) is int:
                if margin < 0:
                    poscheck(proposals[index] == best_other, "best_other_native_argmax_mismatch")
                elif margin == 0:
                    poscheck(proposals[index] == min(canonical[index], best_other), "tied_native_argmax_mismatch")
    errors = sorted(set(errors))
    return {"status": "failed" if errors else "passed", "errors": errors}


def validate_row(anchor, arm, row):
    """Public fail-closed gate, including malformed externally loaded JSON."""
    try:
        return _validate_row(anchor, arm, row)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, AttributeError) as error:
        return {"status": "failed", "errors": ["malformed_native_evidence:" + str(error)]}


def compare_anchor(anchor, frozen, adapted, arm):
    """Return a paired block row and paired positions from the identical start."""
    require(arm in ADAPTED_ARMS, "Only S8/F and L8/F are prescribed")
    rows = (("F", frozen), (arm, adapted))
    missing_arms = [name for name, row in rows if row is None or row == {} or
                    isinstance(row, dict) and row.get("status") in ("unattempted", "not_attempted")]
    validations = {name: {"status": "unattempted", "errors": []} if name in missing_arms else
                   validate_row(anchor, name, row) for name, row in rows}
    invalid_observed_arms = [name for name, value in validations.items() if value["status"] == "failed"]
    errors = [name + ":" + error for name, value in validations.items() for error in value["errors"]]
    valid = not missing_arms and not invalid_observed_arms
    identity = {key: anchor.get(key) for key in ("anchor_id", "group_id", "domain", "population",
                                               "request_id", "block_output_start")}
    pair = dict(identity, arm=arm, control_arm="F", valid=valid, invalid_reasons=errors,
                status="invalid" if invalid_observed_arms else "unattempted" if missing_arms else "valid",
                validations=validations, missing_arms=missing_arms, invalid_observed_arms=invalid_observed_arms)
    positions = []
    scores = {}
    if valid:
        canonical = anchor["canonical_next_ids"]
        f, a = frozen["proposed_token_ids"], adapted["proposed_token_ids"]
        scores = {name: prefix_score(row["proposed_token_ids"], canonical, anchor["eos_ids"],
                                   **anchor_boundary(anchor)) for name, row in (("F", frozen), (arm, adapted))}
        fs, ads = scores["F"], scores[arm]
        first = fs["first_mismatch_position"]
        new_earlier = [i for i in range(fs["correct_proposed_prefix_length"]) if a[i] != canonical[i]]
        fixed = first is not None and a[first] == canonical[first]
        useful = bool(fixed and not new_earlier)
        gain = ads["correct_proposed_prefix_length"] - fs["correct_proposed_prefix_length"]
        pair.update(frozen_correct_prefix_length=fs["correct_proposed_prefix_length"],
                    adapted_correct_prefix_length=ads["correct_proposed_prefix_length"], correct_prefix_gain=gain,
                    frozen_non_eos_prefix_length=fs["non_eos_prefix_length"],
                    adapted_non_eos_prefix_length=ads["non_eos_prefix_length"],
                    non_eos_prefix_gain=ads["non_eos_prefix_length"] - fs["non_eos_prefix_length"],
                    prefix_outcome="gain" if gain > 0 else "loss" if gain < 0 else "tie",
                    frozen_first_mismatch_position=first, frozen_first_error_available=first is not None,
                    frozen_first_error_fixed=bool(fixed), useful_first_error_fix=useful,
                    first_error_fix_blocked_by_earlier_error=bool(fixed and new_earlier),
                    introduced_errors_within_frozen_prefix=len(new_earlier),
                    has_introduced_earlier_error=bool(new_earlier),
                    changed_proposal_count_all_nine=sum(x != y for x, y in zip(f, a)),
                    changed_proposal_count_scored=sum(f[i] != a[i] for i in range(fs["compared_positions"])),
                    scored_positions=fs["compared_positions"],
                    frozen_prefix_censored=fs["censored"], adapted_prefix_censored=ads["censored"],
                    frozen_matched_eos=fs["matched_eos"], adapted_matched_eos=ads["matched_eos"])
        for index, token in enumerate(canonical[:fs["compared_positions"]]):
            fdetail, adetail = frozen["positions"][index], adapted["positions"][index]
            after_f = fs["first_mismatch_position"] is not None and index > fs["first_mismatch_position"]
            after_a = ads["first_mismatch_position"] is not None and index > ads["first_mismatch_position"]
            position = dict(identity, arm=arm, valid=True, block_position=index, native_block_slot=index + 1,
                absolute_output_position=anchor["block_output_start"] + index,
                absolute_position=anchor["expected_absolute_positions"][index], canonical_token_id=token,
                frozen_proposal_id=f[index], adapted_proposal_id=a[index], proposal_changed=f[index] != a[index],
                frozen_matches_canonical=f[index] == token, adapted_matches_canonical=a[index] == token,
                positional_fix=f[index] != token and a[index] == token,
                positional_new_error=f[index] == token and a[index] != token,
                at_frozen_first_mismatch=index == first,
                useful_first_error_fix=bool(useful and index == first),
                new_error_within_frozen_prefix=index in new_earlier,
                after_earlier_frozen_mismatch=after_f, after_earlier_adapted_mismatch=after_a,
                positional_only_after_earlier_mismatch=after_f or after_a,
                target_verification_decision=None,
                observation_scope="Canonical-token positional observation; no target verification")
            for prefix, detail in (("frozen", fdetail), ("adapted", adetail)):
                for field in ("native_canonical_log_probability", "native_bf16_canonical_margin", "native_argmax_id",
                              "native_argmax_tie_count", "canonical_tied_for_max"):
                    position[prefix + "_" + field] = detail[field]
            position["canonical_log_probability_change"] = adetail["native_canonical_log_probability"] - fdetail["native_canonical_log_probability"]
            position["canonical_margin_change"] = adetail["native_bf16_canonical_margin"] - fdetail["native_bf16_canonical_margin"]
            positions.append(position)
        for field in ("canonical_log_probability_change", "canonical_margin_change"):
            pair["mean_" + field] = sum(p[field] for p in positions) / len(positions) if positions else None
    else:
        for key in ("correct_prefix_gain", "non_eos_prefix_gain", "frozen_correct_prefix_length",
                    "adapted_correct_prefix_length", "useful_first_error_fix", "frozen_first_error_fixed",
                    "changed_proposal_count_all_nine", "changed_proposal_count_scored"):
            pair[key] = None
    pair.update(scores=scores, positions=positions,
                interpretation="Local fixed-start draft observations; not saved verification calls, speedup, or the location of prior KL reduction.")
    return pair, positions
