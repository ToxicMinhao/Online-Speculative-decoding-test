"""Synthetic-only offline attribution tests; no campaign analysis or model load."""
import copy
import pytest

from experiments.dflash_osd_attribution.analysis import analyze_campaign, category, cost_rows, token_hash, validated_events


def event(inputs, output, start, proposals, targets=None):
    targets = targets or output[start:start+len(proposals)]
    targets = list(targets) + [999] * (len(proposals)-len(targets))
    reject = next((i for i, (a, b) in enumerate(zip(proposals, targets)) if a != b), None)
    count = (len(proposals)+1 if reject is None else reject+1)
    committed = output[start:start+count]
    accepted = min(len(proposals) if reject is None else reject, len(committed))
    return dict(kind="verification", output_offset=start, round_index=73,
        context=dict(input_ids_sha256=token_hash(inputs), committed_output_prefix_length=start),
        verification_input_token_ids=[output[start-1]]+proposals,
        draft_proposed_token_ids=proposals,
        draft_proposed_positions=list(range(len(inputs)+start, len(inputs)+start+len(proposals))),
        target_predicted_token_ids=targets,
        target_prediction_positions=list(range(len(inputs)+start, len(inputs)+start+len(proposals))),
        rejection_boundary_proposal_index=reject, committed_token_ids=committed,
        committed_provenance=["accepted_draft"]*accepted+["target_correction"]*(len(committed)-accepted),
        stop_clipped=False)


def trace_for(inputs, output, events):
    return dict(input_token_ids=inputs, final_output_token_ids=output,
                events=[dict(kind="target_prefill", committed_token_ids=output[:1])]+events,
                reconciliation=dict(status="passed", input_ids_sha256=token_hash(inputs), output_ids_sha256=token_hash(output)))


def fixture():
    inputs, output = [100, 101], [10, 11, 12, 13, 14, 15]
    # Frozen first block rejects position2, next starts at3. OSD repairs position2
    # and starts its next block at5. The same round number intentionally appears
    # everywhere, so an accidental index join cannot satisfy these assertions.
    f = [event(inputs, output, 1, [11, 90, 91], [11, 12, 92]),
         event(inputs, output, 3, [13, 14, 15])]
    o = [event(inputs, output, 1, [11, 12, 13]), event(inputs, output, 5, [15])]
    traces = {"F": trace_for(inputs, output, f), "OSD": trace_for(inputs, output, o)}
    scores = []
    for arm, events in (("F", f), ("OSD", o)):
        for repeat in range(3):
            scores.append(dict(arm=arm, turn=1, repeat=repeat, state_sha256=arm,
                input_ids=inputs, output_ids=output, capture=False, feedback=None, capture_costs=None,
                same_generation_valid=True, input_match=True, canonical_output_match=True,
                checkpoint_identity_match=True, request_s=(3.0 if arm == "F" else 2.0)+repeat,
                verification_calls=len(events),
                accepted_draft_tokens=sum(e["committed_provenance"].count("accepted_draft") for e in events),
                verification_committed_tokens=sum(len(e["committed_token_ids"]) for e in events), trajectory=arm))
    canonical = [dict(arm=arm, turn=1, input_ids=inputs, output_ids=output, request_s=5.0,
                      deployment_activation_s=2.0, cache_maintenance_s=1.0) for arm in ("BANK", "OSD")]
    result = dict(episode_id="wildchat_00", source="WildChat", synthetic=False, qualification=False,
        status="complete", planned_requests=1, canonical=canonical, scores=scores, updates=[])
    return result, traces


def analyze(result, traces):
    return analyze_campaign([result], traces.__getitem__, lambda token: str(token), require_full_campaign=False)


def test_repeats_are_time_measurements_not_three_token_samples():
    result, traces = fixture()
    used = []
    report = analyze_campaign([result], lambda key: (used.append(key), traces[key])[1], str,
                              require_full_campaign=False)
    assert used == ["F", "OSD"]
    assert report["overall"]["frozen_request_s"] == 4.0
    assert report["overall"]["osd_request_s"] == 3.0
    assert report["overall"]["saved_request_s"] == 1.0
    assert report["overall"]["frozen_verification_calls"] == 2
    assert len(report["timing_repeats"]) == 6
    assert report["conversations"][0]["share_of_net_saved_time_percent"] == 100


def test_alignment_uses_absolute_position_and_exact_canonical_prefix():
    result, traces = fixture()
    report = analyze(result, traces)
    repairs = [d for d in report["aligned_decisions"] if d["effect"] == "repair"]
    assert [d["absolute_output_position"] for d in repairs] == [2]
    position3 = next(d for d in report["aligned_decisions"] if d["absolute_output_position"] == 3)
    assert position3["frozen_block_start"] == 3 and position3["osd_block_start"] == 1
    assert position3["same_block_start"] is False
    assert position3["effect"] is None
    assert position3["native_outcome"] == "both_correct"
    assert position3["comparison_scope"] == "different_start_native_observation"
    # Frozen's speculative position3 after its wrong token90 is not paired.
    assert report["token_coverage"][0]["f_noncanonical_or_beyond_output_proposals"] == 1
    assert report["token_coverage"][0]["matched_block_starts"] == 1
    assert report["token_coverage"][0]["unmatched_decision_positions"] == 1
    assert report["token_coverage"][0]["matched_condition_decisions"] == 2
    assert report["token_coverage"][0]["different_start_observations"] == 2


@pytest.mark.parametrize("change,reason", [
    ({"capture": True}, "capture_contaminated"),
    ({"feedback": {}}, "capture_contaminated"),
    ({"capture_costs": {"seconds": 1}}, "capture_contaminated"),
    ({"state_sha256": "different"}, "state_identity_changed_or_absent"),
    ({"output_ids": [999]}, "exact_output_ids_mismatch"),
    ({"verification_calls": 7}, "repeat_verification_calls_mismatch"),
])
def test_invalid_repeat_is_not_silently_dropped(change, reason):
    result, traces = fixture()
    result["scores"][1].update(change)
    report = analyze(result, traces)
    assert report["overall"]["valid_requests"] == 0
    assert any(reason in r for r in report["exclusions"][0]["reasons"])
    assert len(report["timing_repeats"]) == 6
    assert not report["aligned_decisions"]


def test_missing_trace_preserves_valid_latency_but_excludes_token_claims():
    result, traces = fixture()
    del traces["F"]
    report = analyze(result, traces)
    assert report["overall"]["valid_requests"] == 1
    assert report["exclusions"][0]["scope"] == "token_analysis"
    assert not report["aligned_decisions"]


def test_exact_prefix_corruption_is_not_accepted_as_same_history():
    result, traces = fixture()
    traces["F"]["events"][1]["verification_input_token_ids"][0] = 123
    report = analyze(result, traces)
    assert report["exclusions"][0]["scope"] == "token_analysis"
    assert not report["aligned_decisions"]


def test_cost_outer_timers_count_capture_training_export_only_once():
    result, _ = fixture()
    result["updates"] = [dict(arm="OSD", event=1, recurring_adaptation_s=20.0,
        costs=dict(capture_preparation_s=1.0, teacher_preparation_s=2.0, training_s=7.0,
                   draft_training_and_optimizer_s=5.0, export_s=4.0, optimizer_state_save_s=3.0))]
    costs = cost_rows(result)
    assert costs["serving_plus_adaptation_episode_s"] == 5+2+1+20
    assert costs["training_including_teacher_s"] == 7
    assert costs["teacher_preparation_s"] == 2
    assert costs["draft_training_and_optimizer_s"] == 5
    assert costs["export_s"] == 4


def test_recurrence_uses_only_prior_completed_training_not_future_or_current_updates():
    result, traces = fixture()
    result["planned_requests"] = 2
    # Turn2 uses the same fixture tokens for a simple, exact recurrence test.
    for row in list(result["scores"]):
        result["scores"].append({**copy.deepcopy(row), "turn": 2})
    for row in list(result["canonical"]):
        result["canonical"].append({**copy.deepcopy(row), "turn": 2})
    prior = next(r for r in result["canonical"] if r["arm"] == "OSD" and r["turn"] == 1)
    prior["output_ids"] = [12]  # Training arm can have a distinct trajectory.
    result["updates"] = [dict(arm="OSD", event=1, status="passed", mask_and_schedule_check=True,
                              responses=[copy.deepcopy(prior)]),
                         dict(arm="OSD", event=2, status="passed", mask_and_schedule_check=True,
                              responses=[dict(input_ids=[1], output_ids=[11])])]
    report = analyze(result, traces)
    turn1repair = next(r for r in report["aligned_decisions"] if r["turn"] == 1 and r["effect"] == "repair")
    turn2repair = next(r for r in report["aligned_decisions"] if r["turn"] == 2 and r["effect"] == "repair")
    assert turn1repair["appeared_in_prior_trained_response"] is False
    assert turn2repair["appeared_in_prior_trained_response"] is True
    assert turn2repair["prior_trained_events"] == [1]
    token11 = next(r for r in report["aligned_decisions"] if r["turn"] == 2 and r["canonical_token_id"] == 11)
    assert token11["appeared_in_prior_trained_response"] is False


def test_current_answer_repetition_is_not_prior_training():
    from experiments.dflash_osd_attribution.analysis import _aligned_decisions
    row = dict(absolute_output_position=2, canonical_prefix_valid=True, correct=True,
               block_output_start=2, block_position=0)
    aligned, _ = _aligned_decisions([row], [row], [7, 8, 7], set(), str)
    assert aligned[0]["recurrence"] == "current_response_only"
    assert aligned[0]["appeared_in_prior_trained_response"] is False


@pytest.mark.parametrize("text,expected", [(" \n", "whitespace_punctuation"), (";", "whitespace_punctuation"),
    ("def", "code_syntax"), (" ->", "code_syntax"), ("my_var", "identifiers_numbers"),
    ("12", "identifiers_numbers"), (" camelCase", "identifiers_numbers"),
    ("ordinary", "ordinary_text"), ("word;", "mixed_uncategorized")])
def test_declared_lexical_categories(text, expected):
    assert category(text) == expected


def test_fixed_campaign_scope_and_qualification_exclusion():
    result, traces = fixture()
    with pytest.raises(ValueError, match="94"):
        analyze_campaign([result], traces.__getitem__, str)
    result["qualification"] = True
    with pytest.raises(ValueError, match="Qualification"):
        analyze(result, traces)
    result["qualification"] = False
    result["episode_id"] = "personamem_persona404"
    with pytest.raises(ValueError, match="five distinct"):
        analyze(result, traces)


def test_token_denominators_preserve_uncategorized_and_unmatched():
    result, traces = fixture()
    report = analyze_campaign([result], traces.__getitem__, lambda _: "word;", require_full_campaign=False)
    matched = sum(r["paired_decision_denominator"] for r in report["pattern_summary"])
    different = sum(r["paired_observation_denominator"] for r in report["different_start_pattern_summary"])
    assert matched == len(report["matched_condition_decisions"])
    assert different == len(report["different_start_observations"])
    assert matched + different == len(report["aligned_decisions"])
    assert all(r["category"] == "mixed_uncategorized" for r in report["pattern_summary"] if r["paired_decision_denominator"])
    assert report["unmatched_positions"]


def test_native_regressions_and_negative_contributions_remain_visible():
    result, traces = fixture()
    second = copy.deepcopy(result)
    second["episode_id"] = "wildchat_01"
    for row in second["scores"]:
        if row["arm"] == "OSD":
            row["request_s"] += 1.5
    report = analyze_campaign([result, second], traces.__getitem__, str, require_full_campaign=False)
    assert report["overall"]["saved_request_s"] == 0.5
    assert report["conversations"][0]["share_of_net_saved_time_percent"] == 200
    assert report["conversations"][1]["share_of_net_saved_time_percent"] == -100
    assert report["conversations"][1]["request_time_reduction_percent"] < 0


def test_new_errors_are_separate_from_repairs_and_unpaired_decisions():
    from experiments.dflash_osd_attribution.analysis import _aligned_decisions
    first = dict(absolute_output_position=1, canonical_prefix_valid=True, correct=True,
                 block_output_start=1, block_position=0)
    second = {**first, "correct": False}
    unmatched = {**first, "absolute_output_position": 2}
    aligned, unavailable = _aligned_decisions([first, unmatched], [second], [7, 8, 9], set(), str)
    assert len(aligned) == len(unavailable) == 1
    assert aligned[0]["effect"] == "new_error"
    assert unavailable[0]["absolute_output_position"] == 2


def test_missing_pinned_tokenizer_preserves_token_id_and_native_measurements():
    result, traces = fixture()
    report = analyze_campaign([result], traces.__getitem__, lambda _: None, require_full_campaign=False)
    assert category(None) == "mixed_uncategorized"
    assert report["overall"]["valid_requests"] == 1
    assert report["overall"]["saved_request_s"] == 1.0
    assert report["aligned_decisions"]
    assert all(r["token_text"] is None and r["token_text_available"] is False
               for r in report["aligned_decisions"])
    assert all(r["category"] == "mixed_uncategorized" for r in report["aligned_decisions"])
    assert any(r["effect"] == "repair" for r in report["aligned_decisions"])
    assert all(isinstance(r["appeared_in_prior_trained_response"], bool) for r in report["aligned_decisions"])


def different_start_fixture():
    """Frozen masked slot 3 fails where OSD masked slot 1 succeeds."""
    result, _ = fixture()
    inputs, output = [100, 101], [10, 11, 12, 13, 14, 15, 16, 17]
    events = {
        "F": [event(inputs, output, 1, [11, 12, 90], [11, 12, 13]),
              event(inputs, output, 4, [14, 15, 16])],
        "OSD": [event(inputs, output, 1, [11, 91, 92], [11, 12, 93]),
                event(inputs, output, 3, [13, 14, 15]), event(inputs, output, 7, [17])],
    }
    traces = {arm: trace_for(inputs, output, rows) for arm, rows in events.items()}
    for row in result["canonical"]:
        row.update(input_ids=inputs, output_ids=output)
    for row in result["scores"]:
        native = events[row["arm"]]
        row.update(input_ids=inputs, output_ids=output, verification_calls=len(native),
            accepted_draft_tokens=sum(e["committed_provenance"].count("accepted_draft") for e in native),
            verification_committed_tokens=sum(len(e["committed_token_ids"]) for e in native))
    return result, traces


def test_different_start_success_is_not_a_matched_repair_or_pooled_recurrence():
    result, traces = different_start_fixture()
    report = analyze(result, traces)
    assert not report["exclusions"]
    row = next(r for r in report["aligned_decisions"] if r["absolute_output_position"] == 3)
    assert (row["frozen_block_start"], row["frozen_block_position"]) == (1, 2)
    assert (row["osd_block_start"], row["osd_block_position"]) == (3, 0)
    assert row["native_outcome"] == "frozen_wrong_osd_correct"
    assert row["effect"] is None and row in report["different_start_observations"]
    assert row not in report["matched_condition_decisions"]
    assert sum(r["repair"] for r in report["pattern_summary"]) == 0
    assert sum(r["new_error"] for r in report["pattern_summary"]) == 1
    assert sum(r["paired_decision_denominator"] for r in report["pattern_summary"]) == 2
    assert sum(r["paired_observation_denominator"] for r in report["different_start_pattern_summary"]) == 3
    assert sum(r["frozen_wrong_osd_correct"] for r in report["different_start_pattern_summary"]) == 1
    assert all("repair" not in r for r in report["different_start_pattern_summary"])
    assert report["token_coverage"][0]["matched_condition_decisions"] == 2
    assert report["token_coverage"][0]["different_start_observations"] == 3
    # The interpretation fix must not change observed native performance.
    assert report["overall"]["frozen_request_s"] == 4.0
    assert report["overall"]["osd_request_s"] == 3.0
    assert report["overall"]["frozen_verification_calls"] == 2
    assert report["overall"]["osd_verification_calls"] == 3
