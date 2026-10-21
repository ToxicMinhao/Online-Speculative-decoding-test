"""Synthetic CPU evidence only; these tests never load a model or retained data."""
from copy import deepcopy

import pytest

from .protocol import ARMS, compare_anchor, digest, prefix_score, validate_row


def anchor_fixture(*, request_id="r", start=1, canonical=None, output=None, frozen_length=2, population="heldout"):
    if output is None:
        output = [10] * start + (list(range(20, 29)) if canonical is None else canonical)
    canonical = output[start:start + 9]
    proposals = canonical + [700] * (9 - len(canonical))
    if frozen_length < len(canonical):
        proposals[frozen_length] = 999
    prefix = [1, 2] + output[:start]
    anchor = dict(anchor_id=f"g:{request_id}:{start}", group_id="g", domain="gsm8k", population=population,
                request_id=request_id, source_control=population == "source", input_ids=[1, 2], output_ids=output,
                block_output_start=start, retained_round_index=start - 1, canonical_prefix_ids=prefix, canonical_prefix_sha256=digest(prefix),
                seed_token_id=output[start - 1], canonical_next_ids=canonical, eos_ids=[2],
                expected_frozen_proposal_ids=proposals, expected_absolute_positions=list(range(2 + start, 11 + start)),
                response_boundary="eos" if output[-1] == 2 else "cap", reaches_response_boundary=start + len(canonical) == len(output),
                checkpoints={arm: {"checkpoint_identity_sha256": digest(arm)} for arm in ARMS})
    rounds = []
    for offset in range(1, start):
        positions = list(range(len(anchor["input_ids"]) + offset - 1, len(anchor["input_ids"]) + offset + 9))
        rounds.append(dict(round_index=offset - 1, output_offset=offset,
            verification_input_token_ids=[output[offset - 1]] + [777] * 9,
            verification_input_positions=positions, draft_proposed_token_ids=[777] * 9,
            target_predicted_token_ids=[output[offset]] + [777] * 9,
            target_prediction_positions=[position + 1 for position in positions],
            commit_length=1, committed_token_ids=[output[offset]], next_seed_token_id=output[offset],
            native_acceptance_length_before_stop=0, stop_clipped=False, finished=False,
            raw_event_sha256=digest([request_id, offset])))
    reconstruction = dict(mode="recorded_target_verification_schedule",
        prefill=dict(input_ids_sha256=digest(anchor["input_ids"]), prompt_tokens=len(anchor["input_ids"]),
                     seed_token_id=output[0], event_sha256=digest([request_id, "prefill"])),
        rounds=rounds, target_verification_count=len(rounds))
    reconstruction["schedule_sha256"] = digest(reconstruction)
    anchor["reconstruction"] = reconstruction
    return anchor


def row_fixture(anchor, arm="F", *, prefix_length=None, proposals=None):
    if proposals is None:
        proposals = list(anchor["expected_frozen_proposal_ids"])
    else:
        proposals = list(proposals)
    if prefix_length is not None:
        canonical = anchor["canonical_next_ids"]
        proposals[:len(canonical)] = canonical
        if prefix_length < len(canonical):
            proposals[prefix_length] = 999
    row = {key: anchor[key] for key in ("anchor_id", "group_id", "domain", "population", "request_id",
                                       "canonical_prefix_ids", "canonical_prefix_sha256", "seed_token_id")}
    identity = anchor["checkpoints"][arm]["checkpoint_identity_sha256"]
    row.update(status="passed", arm=arm, proposed_token_ids=proposals, proposal_ids=proposals,
               native_argmax_ids=proposals, reconstructed_target_seed_id=anchor["seed_token_id"],
               checkpoint_identity_sha256=identity, cache_checkpoint_identity_sha256=identity,
               helper_checkpoint_identity_sha256=identity, native_absolute_proposal_positions=anchor["expected_absolute_positions"],
               native_materialized_block_ids=[anchor["seed_token_id"]] + [128002] * 9,
               prefix_input_ids_sha256=digest(anchor["input_ids"]), native_feature_dtype="torch.bfloat16",
               native_logits_dtype="torch.bfloat16", inspection_reduction_dtype="torch.float32", nonfinite_logit_count=0,
               state_hash_kind="parameter_and_buffer_identity_version_signature", target_state_before_sha256=digest("target"),
               target_state_after_sha256=digest("target"), draft_state_before_sha256=digest("draft"),
               draft_state_after_sha256=digest("draft"), target_prefill_chunks=[len(anchor["input_ids"])],
               frozen_proposals_match_retained=proposals == anchor["expected_frozen_proposal_ids"] if arm == "F" else None,
               positions=[])
    for flag in ("target_seed_matches", "valid_reconstruction", "cache_valid", "helper_refresh_valid", "immutable_weights",
                 "target_weights_unchanged", "draft_weights_unchanged", "eval_mode", "inference_mode", "no_generated_history",
                 "no_training", "native_hook_passed", "native_logits_finite"):
        row[flag] = True
    for key in ("native_draft_block_evaluations", "draft_block_evaluations", "native_head_calls"):
        row[key] = 1
    for key in ("scored_target_verifications", "regenerated_response_tokens", "training_steps", "optimizer_creations", "optimizer_loads"):
        row[key] = 0
    reconstruction = anchor["reconstruction"]
    row["reconstruction_schedule_sha256"] = reconstruction["schedule_sha256"]
    row["reconstructed_prefill_seed_id"] = reconstruction["prefill"]["seed_token_id"]
    for key in ("reconstruction_target_verifications", "reconstruction_target_verify_attempts", "target_verifications"):
        row[key] = reconstruction["target_verification_count"]
    row["reconstruction_rounds"] = [dict(
        {key: item[key] for key in ("round_index", "output_offset", "commit_length", "next_seed_token_id")},
        status="passed", target_predictions_match=True, committed_tokens_match=True, cache_advance_matches=True,
        target_call_attempts=1, fresh_target_predicted_token_ids=item["target_predicted_token_ids"],
        fresh_committed_token_ids=item["committed_token_ids"],
        seq_lens_before=len(anchor["input_ids"]) + item["output_offset"] - 1,
        seq_lens_after=len(anchor["input_ids"]) + item["output_offset"] - 1 + item["commit_length"])
        for item in reconstruction["rounds"]]
    for index, token in enumerate(anchor["canonical_next_ids"]):
        correct = proposals[index] == token
        row["positions"].append(dict(block_position=index, native_block_slot=index + 1,
            absolute_output_position=anchor["block_output_start"] + index,
            absolute_position=anchor["expected_absolute_positions"][index], canonical_token_id=token,
            proposal_token_id=proposals[index], native_argmax_id=proposals[index], native_canonical_log_probability=-.5 if correct else -2.,
            native_bf16_canonical_margin=1. if correct else -1., native_canonical_logit=1. if correct else 0.,
            native_best_other_logit=0. if correct else 1., native_best_other_id=500 if correct else proposals[index],
            native_argmax_tie_count=1, canonical_tied_for_max=False))
    return row


def test_matched_eos_counts_in_primary_length_but_not_non_eos():
    score = prefix_score([20, 2, 99], [20, 2], [2], response_boundary="eos", reaches_response_boundary=True)
    assert score["correct_proposed_prefix_length"] == 2
    assert score["non_eos_prefix_length"] == 1
    assert score["matched_eos"] and score["boundary"] == "matched_eos"
    assert score["first_mismatch_position"] is None and not score["censored"]
    assert score["unscored_proposal_positions"] == 1


@pytest.mark.parametrize("length", [0, 1, 8, 9])
def test_cap_matching_suffix_including_exact_full_block_is_censored(length):
    score = prefix_score(list(range(9)), list(range(length)), [], reaches_response_boundary=True)
    assert score["correct_proposed_prefix_length"] == length
    assert score["censored"] and score["boundary"] == "cap_censored"
    assert score["first_mismatch_position"] is None


def test_mismatch_is_observed_even_when_available_suffix_is_censored():
    score = prefix_score([99] * 9, [20], [], reaches_response_boundary=True)
    assert score["first_mismatch_position"] == 0 and not score["censored"]
    assert score["available_suffix_censored"] and score["boundary"] == "mismatch"


def test_regular_full_block_without_response_boundary_is_not_censored():
    score = prefix_score(list(range(9)), list(range(9)), [])
    assert score["correct_proposed_prefix_length"] == 9 and not score["censored"]
    assert score["boundary"] == "block_end"


def test_valid_row_is_strictly_accepted():
    anchor = anchor_fixture()
    assert validate_row(anchor, "F", row_fixture(anchor)) == {"status": "passed", "errors": []}


@pytest.mark.parametrize("key,bad", [
    ("arm", "L8"), ("request_id", "different"), ("group_id", "wrong"), ("population", "source"),
    ("domain", "other"), ("checkpoint_identity_sha256", "a" * 64),
    ("cache_checkpoint_identity_sha256", "a" * 64), ("helper_checkpoint_identity_sha256", "a" * 64),
    ("canonical_prefix_ids", [1]), ("canonical_prefix_sha256", "a" * 64), ("seed_token_id", 123),
    ("reconstructed_target_seed_id", 123), ("target_state_after_sha256", "a" * 64),
    ("draft_state_after_sha256", "a" * 64), ("native_absolute_proposal_positions", [1] * 9),
    ("target_prefill_chunks", [1]), ("native_argmax_ids", [1] * 9),
    ("target_verifications", 1), ("regenerated_response_tokens", 1), ("training_steps", 1),
    ("optimizer_creations", 1), ("optimizer_loads", 1), ("draft_block_evaluations", 2),
    ("native_draft_block_evaluations", 0), ("native_head_calls", 2),
    ("prefix_input_ids_sha256", "a" * 64), ("native_materialized_block_ids", [0] * 10),
    ("native_feature_dtype", "torch.float32"), ("native_logits_dtype", "torch.float32"),
    ("inspection_reduction_dtype", "torch.bfloat16"), ("nonfinite_logit_count", 1),
    ("native_logits_finite", False), ("eval_mode", False), ("inference_mode", False),
    ("immutable_weights", False), ("cache_valid", False), ("helper_refresh_valid", False),
    ("native_hook_passed", False), ("no_training", False), ("no_generated_history", False),
])
def test_shared_validator_rejects_identity_geometry_read_only_and_counter_failures(key, bad):
    anchor = anchor_fixture()
    row = row_fixture(anchor)
    row[key] = bad
    assert validate_row(anchor, "F", row)["status"] == "failed", key


@pytest.mark.parametrize("field", ["native_canonical_log_probability", "native_bf16_canonical_margin",
                                   "native_canonical_logit", "native_best_other_logit"])
def test_nonfinite_inspection_is_invalid(field):
    anchor = anchor_fixture()
    row = row_fixture(anchor)
    row["positions"][0][field] = float("nan")
    assert validate_row(anchor, "F", row)["status"] == "failed"


def test_all_nine_frozen_proposals_match_even_unscored_cap_tail():
    anchor = anchor_fixture(canonical=[20], frozen_length=9)
    row = row_fixture(anchor)
    row["proposed_token_ids"][-1] += 1
    assert "all_nine_frozen_proposals_differ_from_retained" in validate_row(anchor, "F", row)["errors"]


@pytest.mark.parametrize("correct", [True, False])
def test_margin_zero_ties_preserve_actual_native_argmax(correct):
    anchor = anchor_fixture(frozen_length=9 if correct else 0)
    if not correct:
        anchor["expected_frozen_proposal_ids"][0] = 1
    row = row_fixture(anchor)
    row["positions"][0].update(native_canonical_logit=0., native_best_other_logit=0.,
                               native_bf16_canonical_margin=0., native_argmax_tie_count=2,
                               canonical_tied_for_max=True)
    assert validate_row(anchor, "F", row)["status"] == "passed"
    row["positions"][0]["native_argmax_id"] = 123
    assert validate_row(anchor, "F", row)["status"] == "failed"


def test_impossible_tie_and_best_other_decisions_fail_validation():
    anchor = anchor_fixture()
    row = row_fixture(anchor)
    row["positions"][0]["native_argmax_tie_count"] = 2
    assert validate_row(anchor, "F", row)["status"] == "failed"
    row = row_fixture(anchor)
    row["positions"][2]["native_best_other_id"] = 123
    assert validate_row(anchor, "F", row)["status"] == "failed"


def test_useful_first_error_fix_requires_no_new_earlier_error():
    anchor = anchor_fixture(frozen_length=3)
    frozen = row_fixture(anchor)
    adapted = row_fixture(anchor, "S8", prefix_length=5)
    pair, positions = compare_anchor(anchor, frozen, adapted, "S8")
    assert pair["valid"] and pair["correct_prefix_gain"] == 2
    assert pair["useful_first_error_fix"] and positions[3]["useful_first_error_fix"]
    blocked = row_fixture(anchor, "S8", prefix_length=1)
    pair, positions = compare_anchor(anchor, frozen, blocked, "S8")
    assert pair["frozen_first_error_fixed"] and not pair["useful_first_error_fix"]
    assert pair["first_error_fix_blocked_by_earlier_error"] and pair["correct_prefix_gain"] == -2
    assert pair["introduced_errors_within_frozen_prefix"] == 1
    assert positions[3]["positional_only_after_earlier_mismatch"]
    assert all(position["target_verification_decision"] is None for position in positions)


def test_comparison_uses_same_start_and_all_tail_changes_are_separate():
    anchor = anchor_fixture(canonical=[20], frozen_length=9)
    frozen = row_fixture(anchor)
    adapted = row_fixture(anchor, "L8")
    adapted["proposed_token_ids"][-1] += 1
    pair, positions = compare_anchor(anchor, frozen, adapted, "L8")
    assert pair["valid"] and pair["correct_prefix_gain"] == 0
    assert pair["changed_proposal_count_scored"] == 0 and pair["changed_proposal_count_all_nine"] == 1
    assert pair["frozen_prefix_censored"] and len(positions) == 1
    other = anchor_fixture(start=2)
    pair, positions = compare_anchor(anchor, frozen, row_fixture(other, "L8"), "L8")
    assert not pair["valid"] and pair["correct_prefix_gain"] is None and positions == []


def test_invalid_rows_do_not_produce_effect_counts():
    anchor = anchor_fixture()
    row = row_fixture(anchor, "S8", prefix_length=9)
    row["training_steps"] = 1
    pair, positions = compare_anchor(anchor, row_fixture(anchor), row, "S8")
    assert not pair["valid"] and pair["useful_first_error_fix"] is None and not positions


@pytest.mark.parametrize("row", [None, [], {"positions": None}, {"status": "passed", "positions": [None]}])
def test_malformed_json_evidence_fails_closed(row):
    assert validate_row(anchor_fixture(), "F", row)["status"] == "failed"


def test_missing_adapted_arm_remains_unattempted():
    anchor = anchor_fixture()
    pair, positions = compare_anchor(anchor, row_fixture(anchor), {}, "S8")
    assert pair["status"] == "unattempted" and not positions
    assert pair["missing_arms"] == ["S8"] and pair["invalid_observed_arms"] == []
    assert pair["validations"]["S8"] == dict(status="unattempted", errors=[])


@pytest.mark.parametrize("status", ["passed", "failed"])
def test_invalid_observed_frozen_row_and_missing_adapted_arm_remain_distinct(status):
    anchor = anchor_fixture()
    frozen = row_fixture(anchor)
    frozen.update(status=status, native_logits_finite=False)
    pair, positions = compare_anchor(anchor, frozen, {}, "S8")
    assert pair["status"] == "invalid" and not pair["valid"] and not positions
    assert pair["missing_arms"] == ["S8"] and pair["invalid_observed_arms"] == ["F"]
    assert pair["validations"]["F"]["status"] == "failed"
    assert pair["validations"]["S8"]["status"] == "unattempted"


def test_both_missing_arms_are_unattempted_without_observed_invalid_rows():
    pair, _ = compare_anchor(anchor_fixture(), {}, {"status": "not_attempted"}, "L8")
    assert pair["status"] == "unattempted" and pair["missing_arms"] == ["F", "L8"]
    assert pair["invalid_observed_arms"] == [] and pair["invalid_reasons"] == []


def test_recorded_schedule_target_work_is_allowed_without_scoring_verification():
    anchor = anchor_fixture(start=4)
    row = row_fixture(anchor)
    assert row["target_verifications"] == row["reconstruction_target_verifications"] == 3
    assert row["native_draft_block_evaluations"] == 1 and row["scored_target_verifications"] == 0
    assert row["target_prefill_chunks"] == [2]
    assert validate_row(anchor, "F", row)["status"] == "passed"


@pytest.mark.parametrize("key,bad", [
    ("reconstructed_prefill_seed_id", 999), ("reconstruction_schedule_sha256", "a" * 64),
    ("reconstruction_target_verifications", 2),
    ("reconstruction_target_verify_attempts", 4), ("target_verifications", 0),
    ("scored_target_verifications", 1), ("target_prefill_chunks", [5]),
    ("reconstruction_rounds", []), ("prefix_input_ids_sha256", digest([1, 2, 10, 10, 10])),
])
def test_saved_schedule_accounting_must_reconcile_exactly(key, bad):
    anchor = anchor_fixture(start=4)
    row = row_fixture(anchor)
    row[key] = bad
    assert validate_row(anchor, "F", row)["status"] == "failed"


@pytest.mark.parametrize("key,bad", [
    ("status", "failed"), ("round_index", 99), ("output_offset", 99), ("commit_length", 0),
    ("next_seed_token_id", 999), ("target_predictions_match", False), ("committed_tokens_match", False),
    ("cache_advance_matches", False), ("target_call_attempts", 2), ("fresh_target_predicted_token_ids", []),
    ("fresh_committed_token_ids", []), ("seq_lens_before", 99), ("seq_lens_after", 99),
])
def test_raw_reconstruction_receipts_are_revalidated(key, bad):
    anchor = anchor_fixture(start=4)
    row = row_fixture(anchor)
    row["reconstruction_rounds"][1][key] = bad
    assert validate_row(anchor, "F", row)["status"] == "failed"


def test_legacy_cold_prefill_row_cannot_silently_pass_repaired_protocol():
    anchor = anchor_fixture(start=4)
    row = row_fixture(anchor)
    del anchor["reconstruction"]
    for key in ("reconstruction_schedule_sha256", "reconstruction_target_verifications",
                "reconstruction_target_verify_attempts", "reconstruction_rounds", "scored_target_verifications"):
        del row[key]
    row["target_verifications"] = 0
    assert validate_row(anchor, "F", row)["status"] == "failed"
