"""Synthetic P8 arithmetic and boundary checks through the real shared validator."""
import copy
import json

import pytest

from experiments.dflash_kl_native_diagnostic.test_experiment import native_row, loss_row
from experiments.dflash_kl_native_diagnostic.experiment import SAMPLING_FIELDS, digest
from . import report
from .protocol import SERVING_CONTRACT


def pair(request_id="r0", *, population="source", group_id="gsm8k_00", domain="gsm8k",
         f_calls=5, s_calls=6, p_calls=4, f_loss=2., s_loss=.5, p_loss=1.):
    from .evidence import composite_identity
    role = "feedback" if population == "source" else "evaluation"
    value = dict(request_id=request_id, group_id=group_id, domain=domain, population=population,
                 pool_id="p8_" + domain, validation={"status": "passed", "errors": []},
                 checkpoint_bindings={"F": "f" * 64, "S8": "8" * 64, "P8": "c" * 64})
    for arm, calls, loss in (("F", f_calls, f_loss), ("S8", s_calls, s_loss), ("P8", p_calls, p_loss)):
        native = native_row(group_id, role=role, label=arm)
        native.update(request_id=request_id, domain=domain, output_ids=[20] + [21] * 23 + [128009],
                      output_tokens=25, verification_calls=calls, accepted_draft_tokens=24-calls,
                      verification_committed_tokens=24, committed_progress_per_call=24/calls,
                      emitted_tokens_per_verification_call=25/calls,
                      checkpoint_identity_sha256=value["checkpoint_bindings"][arm],
                      request_s=.001 if arm == "P8" else 1000.)
        offset, rounds = 1, []
        for i in range(calls):
            committed = 24 // calls + (i < 24 % calls)
            rounds.append(dict(round_index=i, output_offset=offset, accepted_prefix_length=committed-1,
                               accepted_draft_tokens=committed-1, proposed_tokens=9,
                               committed_tokens=committed,
                               committed_token_ids=native["output_ids"][offset:offset+committed],
                               first_rejection_offset=committed-1, stop_clipped=False))
            offset += committed
        native["rounds"] = rounds
        kl = loss_row(native, loss)
        kl.update(role=role, anchor_rng_after_sha256="a" * 64)
        value[arm] = dict(native=native, kl=kl)
    value["pool_request_id"] = json.dumps(list(composite_identity(value)), separators=(",", ":"), ensure_ascii=False)
    value["P8"]["native"].update(pool_id=value["pool_id"], pool_request_id=value["pool_request_id"],
                                serving_contract=copy.deepcopy(SERVING_CONTRACT), feedback_capture_enabled=False)
    value["P8"]["kl"].update(pool_id=value["pool_id"], pool_request_id=value["pool_request_id"],
                             original_group_id=group_id, group_id=group_id, domain=domain)
    return value


def campaign(*pairs, status="complete"):
    pools = {}
    for value in pairs:
        owner = pools.setdefault(value["pool_id"], dict(pool_id=value["pool_id"], domain=value["domain"],
            group_ids=[], status="complete", source=[], heldout=[], native_raw=[],
            training={"costs": {"teacher_preparation_s": 2., "training_s": 3.},
                      "accounting": {"optimizer_steps_total": 8, "response_presentations_total": 16},
                      "export": {"wall_s": .5, "files": ["model.safetensors"]}},
            checkpoint={"checkpoint_id": "P8", "pool_id": value["pool_id"]}, costs={"activation_s": 2.}))
        if value["group_id"] not in owner["group_ids"]:
            owner["group_ids"].append(value["group_id"])
        owner[value["population"]].append(value)
        owner["native_raw"].append(value["P8"]["native"])
    for owner in pools.values():
        owner["checkpoint"].update(domain=owner["domain"], group_ids=list(owner["group_ids"]),
                                   checkpoint_identity_sha256="c" * 64)
    return dict(status=status, pools=list(pools.values()), actual_wall_s=10., optimizer_steps=24,
                training_presentations=48, actual_native_generations=48, actual_response_kl_evaluations=48,
                explicit_warmups=0, attempted_native_generations=48, unknown_native_work_receipts=0,
                unknown_training_work_receipts=0, unattempted_native_generations=0,
                unattempted_response_kl_evaluations=0, unexecuted_pools=[])


def complete_campaign():
    return campaign(*(pair(f"r{i}", domain=domain, group_id=f"{domain}_{group:02}", population=population)
                      for domain in ("gsm8k", "codealpaca", "sharegpt") for group in range(2)
                      for population in report.POPULATIONS for i in range(4)))


def overall(summary, population="source"):
    return next(row for row in summary["pooled"] if row["scope"] == "overall" and row["population"] == population)


def sampling_hash(value):
    value["sampling_and_weights_sha256"] = digest({key: value[key] for key in SAMPLING_FIELDS})


def test_three_arm_calls_pool_as_sums_not_mean_percentages():
    data = campaign(pair("one", f_calls=4, s_calls=8, p_calls=3),
                    pair("two", f_calls=12, s_calls=8, p_calls=12))
    row = overall(report.summarize(data))
    assert (row["F_verification_calls"], row["S8_verification_calls"], row["P8_verification_calls"]) == (16, 16, 15)
    assert row["P8_vs_F_verification_call_reduction_percent"] == pytest.approx(6.25)
    assert row["P8_vs_S8_verification_call_reduction_percent"] == pytest.approx(6.25)
    assert row["P8_vs_F_native_better_requests"] == row["P8_vs_F_native_tied_requests"] == 1
    assert row["P8_vs_S8_native_worse_requests"] == 1


def test_equal_normalized_response_loss_means_not_token_weighted():
    one, two = pair("one", f_loss=1., s_loss=.5, p_loss=.8), pair("two", f_loss=9., s_loss=4., p_loss=4.2)
    for arm in report.ARMS:
        for value, repeats in ((one, 1), (two, 2)):
            loss = value[arm]["kl"]
            for key in ("anchor_positions", "block_keep_mask", "label_positions", "teacher_positions",
                        "binary_loss_mask", "position_weights"):
                loss[key] *= repeats
            for key in ("sampled_blocks", "eligible_blocks", "eligible_positions", "weighted_position_sum",
                        "excluded_anchor_positions"):
                loss[key] *= repeats
            loss["normalization_denominator"] = loss["weighted_position_sum"] + 1e-6
            sampling_hash(loss)
    row = overall(report.summarize(campaign(one, two)))
    assert row["kl_pairs"] == 2
    assert row["F_mean_kl"] == 5. and row["P8_mean_kl"] == 2.5 and row["S8_mean_kl"] == 2.25
    assert row["P8_vs_F_kl_reduction_percent"] == 50.


def test_dataset_pool_group_and_population_summaries_preserve_ownership():
    values = [pair("same", group_id="gsm8k_00"), pair("same", group_id="gsm8k_01"),
              pair("same", population="heldout", p_calls=6),
              pair("same", domain="sharegpt", group_id="sharegpt_00", p_calls=6)]
    summary = report.summarize(campaign(*values))
    assert overall(summary)["valid_requests"] == 3
    assert overall(summary)["P8_vs_F_verification_calls_saved"] == 1
    assert overall(summary, "heldout")["P8_vs_F_verification_calls_saved"] == -1
    assert {row["scope"] for row in summary["pooled"]} == {"overall", "domain", "pool", "group"}
    assert all(row["population"] in report.POPULATIONS for row in summary["pooled"])
    assert len({row["pool_request_id"] for row in summary["comparisons"]}) == 4


def test_zero_supervision_remains_valid_for_native_comparisons():
    value = pair()
    for arm in report.ARMS:
        loss = value[arm]["kl"]
        loss.update(loss=None, zero_supervision=True, excluded_from_loss_improvement=True,
                    eligible_positions=0, eligible_blocks=0, unique_supervised_positions=0,
                    unique_supervised_position_ids=[], binary_loss_mask=[[0.] * 10],
                    position_weights=[[0.] * 10], weighted_position_sum=0., normalization_denominator=1e-6)
        sampling_hash(loss)
    row = overall(report.summarize(campaign(value)))
    assert row["valid_requests"] == 1 and row["kl_pairs"] == 0 and row["zero_supervision_pairs"] == 1
    assert row["P8_vs_F_verification_calls_saved"] == 1 and row["P8_vs_F_kl_change"] is None


@pytest.mark.parametrize("mutation", ["counter", "anchor", "heldout_role", "pool", "execution", "nonfinite"])
def test_shared_validation_rejects_invalid_rows_without_losing_raw_numbers(mutation):
    value = pair(population="heldout")
    if mutation == "counter": value["P8"]["native"]["accepted_draft_tokens"] = 999
    if mutation == "anchor": value["P8"]["kl"]["anchor_seed"] += 1
    if mutation == "heldout_role": value["P8"]["kl"]["role"] = "feedback"
    if mutation == "pool": value["P8"]["native"]["pool_id"] = "p8_other"
    if mutation == "execution": value["validation"] = {"status": "failed", "errors": ["execution rejected"]}
    if mutation == "nonfinite": value["P8"]["kl"]["loss"] = float("nan")
    summary = report.summarize(campaign(value))
    row = summary["comparisons"][0]
    assert not row["valid"] and row["invalid_reasons"]
    assert row["P8_verification_calls"] == 4
    assert all(row[f"{a}_vs_{b}_verification_calls_saved"] is None for a, b in report.CONTRASTS)
    assert overall(summary, "heldout")["valid_requests"] == 0


def test_duplicate_and_wrong_container_are_invalid():
    value = pair()
    assert overall(report.summarize(campaign(value, copy.deepcopy(value))))["invalid_requests"] == 2
    data = campaign(pair())
    data["pools"][0]["domain"] = "wrong"
    assert not report.summarize(data)["comparisons"][0]["valid"]


def test_native_harm_and_ties_are_counted():
    row = overall(report.summarize(campaign(pair("w", p_calls=6), pair("t", p_calls=5))))
    assert row["P8_vs_F_native_worse_requests"] == row["P8_vs_F_native_tied_requests"] == 1
    assert row["P8_vs_F_verification_call_reduction_percent"] == pytest.approx(-10.)


def test_retained_f_s8_and_fresh_p8_are_explicit():
    for population in report.POPULATIONS:
        row = report.comparison(pair(population=population))
        assert row["valid"]
        for arm in report.ARMS:
            assert row[f"{arm}_kl_origin"] == row[f"{arm}_native_origin"] == ("fresh" if arm == "P8" else "retained")


def test_no_latency_claim_nested_costs_and_training_export_receipts_retained(tmp_path):
    data = campaign(pair())
    summary = report.write_reports(data, tmp_path)
    assert all("request_s" not in row and "request_time_reduction_percent" not in row
               for row in summary["comparisons"] + summary["pooled"])
    assert summary["costs"][0]["training_components"] == {"teacher_preparation_s": 2., "training_s": 3.}
    assert summary["costs"][0]["components"] == {"activation_s": 2.}
    assert summary["training"][0]["training"] == data["pools"][0]["training"]
    text = (tmp_path / "REPORT.md").read_text()
    assert "not a latency comparison" in text and "No latency-reduction or serving-speedup estimate" in text
    assert "fresh AdamW state" in text and "Held-out responses never enter training" in text
    assert "no extra evaluations" in text and "retuning" in text


@pytest.mark.parametrize("p_calls,expected", [(6, "reduced harm, not successful transfer"),
                                             (5, "recovery to frozen performance"),
                                             (4, "only if the source learning is non-negligible")])
def test_interpretation_distinguishes_transfer_from_reduced_harm(p_calls, expected):
    summary = report.summarize(campaign(pair(), pair("held", population="heldout", s_calls=8, p_calls=p_calls)))
    patterns = " ".join(summary["interpretations"][0]["patterns"])
    assert expected in patterns
    assert "No numerical negligibility cutoff was prespecified" in patterns
    assert "fresh held-out confirmation" in summary["interpretation_boundary"]
    if p_calls > 4: assert "supporting transfer" not in patterns
    if p_calls > 5: assert "does not rule out feedback coverage" in patterns


def test_no_source_learning_cannot_be_declared_success():
    summary = report.summarize(campaign(pair(p_loss=2.), pair("held", population="heldout", p_calls=4)))
    text = " ".join(summary["interpretations"][0]["patterns"])
    assert "useful source learning is not established" in text and "supporting transfer" not in text


@pytest.mark.parametrize("decrease", [1., 1e-12])
def test_transfer_interpretation_requires_non_negligible_learning_without_inventing_cutoff(decrease):
    summary = report.summarize(campaign(pair(p_loss=2.-decrease),
                                      pair("held", population="heldout", p_calls=4)))
    patterns = " ".join(summary["interpretations"][0]["patterns"])
    assert "Lower source KL and fewer held-out verification calls" in patterns
    assert "only if the source learning is non-negligible" in patterns
    assert "Negligible source learning would make the control inconclusive" in patterns
    assert "No numerical negligibility cutoff was prespecified" in patterns
    assert "supporting transfer on these requests" not in patterns


def test_self_consistent_foreign_pair_checkpoint_cannot_enter_pool_gains():
    data = complete_campaign()
    value = data["pools"][0]["source"][0]
    value["checkpoint_bindings"]["P8"] = "d" * 64
    for kind in ("native", "kl"):
        value["P8"][kind]["checkpoint_identity_sha256"] = "d" * 64
    # Pair consistency is necessary but cannot replace the pool checkpoint link.
    assert report.comparison(value)["valid"]
    summary = report.summarize(data)
    row = summary["comparisons"][0]
    assert summary["status"] == "partial_or_failed" and not row["valid"]
    assert "pair_P8_checkpoint_differs_from_pool_checkpoint" in row["invalid_reasons"]
    assert row["P8_verification_calls"] == 4 and row["P8_vs_F_verification_calls_saved"] is None
    assert overall(summary)["valid_requests"] == 23


@pytest.mark.parametrize("change", ["missing", "alias", "pool", "domain", "groups", "single_group", "hash"])
def test_pool_checkpoint_metadata_must_bind_genuine_p8_owner(change):
    data = campaign(pair())
    checkpoint = data["pools"][0]["checkpoint"]
    if change == "missing": del data["pools"][0]["checkpoint"]
    if change == "alias": checkpoint["checkpoint_id"] = "S8"
    if change == "pool": checkpoint["pool_id"] = "p8_other"
    if change == "domain": checkpoint["domain"] = "other"
    if change == "groups": checkpoint["group_ids"] = ["other"]
    if change == "single_group": checkpoint["group_id"] = "gsm8k_00"
    if change == "hash": checkpoint["checkpoint_identity_sha256"] = "missing"
    summary = report.summarize(data)
    row = summary["comparisons"][0]
    assert not row["valid"] and row["P8_vs_F_verification_calls_saved"] is None
    assert "missing_or_invalid_pool_P8_checkpoint_owner" in row["invalid_reasons"]


def test_partial_counts_and_unpaired_receipts_are_not_inferred_from_pairs(tmp_path):
    data = campaign(pair(), status="failed")
    data.update(attempted_native_generations=3, actual_native_generations=2, actual_response_kl_evaluations=16,
                unknown_native_work_receipts=1, unknown_training_work_receipts=0,
                unattempted_native_generations=45, unattempted_response_kl_evaluations=32,
                invalid_native_generations=1, unexecuted_pools=["p8_codealpaca", "p8_sharegpt"])
    raw = copy.deepcopy(data["pools"][0]["native_raw"][0])
    raw.update(status="failed", text="private response", output_ids=[1, 2, 3])
    data["pools"][0]["native_raw"].append(raw)
    before = copy.deepcopy(data)
    summary = report.write_reports(data, tmp_path)
    assert data == before and "private response" not in json.dumps(summary)
    assert summary["status"] == "partial_or_failed"
    assert summary["actual_native_generations"] == 2 and summary["unknown_native_work_receipts"] == 1
    assert summary["unattempted_native_generations"] == 45
    assert summary["pools"][0]["recorded_native_receipts"] == 2
    assert summary["accounting_receipts"]["invalid_native_generations"] == 1
    assert (tmp_path / "TRAINING.csv").exists()


def test_empty_and_nonfinite_results_write_valid_json(tmp_path):
    summary = report.write_reports({"pools": [], "status": "planned"}, tmp_path)
    assert summary["status"] == "not_executed"
    assert summary["planned"]["optimizer_steps"] == 24
    value = pair()
    value["P8"]["kl"]["loss"] = float("nan")
    report.write_reports(campaign(value), tmp_path)
    saved = json.loads((tmp_path / "SUMMARY.json").read_text())
    assert saved["comparisons"][0]["P8_loss"] == "NaN"
    assert saved["comparisons"][0]["P8_vs_F_kl_change"] is None


def test_complete_scope_has_three_trajectories_six_original_groups_and_all_pairs():
    summary = report.summarize(complete_campaign())
    assert summary["status"] == "complete" and summary["training_presentations"] == 48
    assert summary["recorded_adaptation_trajectories"] == 3 and summary["original_groups"] == 6
    assert overall(summary)["valid_requests"] == overall(summary, "heldout")["valid_requests"] == 24
    assert "not six independent replicates" in summary["interpretation_boundary"]


@pytest.mark.parametrize("mutation", ["steps", "unknown", "unknown_optimizer", "unknown_kl", "attempted_kl",
                                      "warmup", "pool_failed", "missing_group"])
def test_complete_status_requires_correct_scope_and_work_accounting(mutation):
    data = complete_campaign()
    if mutation == "steps": data["optimizer_steps"] = 25
    if mutation == "unknown": data["unknown_training_work_receipts"] = 1
    if mutation == "unknown_optimizer": data["unknown_optimizer_steps"] = 1
    if mutation == "unknown_kl": data["unknown_response_kl_evaluations"] = 1
    if mutation == "attempted_kl": data["attempted_response_kl_evaluations"] = 47
    if mutation == "warmup": data["explicit_warmups"] = 1
    if mutation == "pool_failed": data["pools"][-1]["status"] = "failed"
    if mutation == "missing_group": data["pools"][-1]["group_ids"].pop()
    assert report.summarize(data)["status"] == "partial_or_failed"


def test_shared_validator_failure_or_nonstandard_return_is_never_accepted(monkeypatch):
    from . import experiment
    value = pair()
    monkeypatch.setattr(experiment, "validate_pair", lambda _: {"status": "failed", "errors": ["counter invalid"]})
    assert not report.comparison(value)["valid"]
    monkeypatch.setattr(experiment, "validate_pair", lambda _: None)
    assert not report.comparison(value)["valid"]


@pytest.mark.parametrize("receipt", [None, "passed", {}, {"status": "passed", "errors": None}])
def test_missing_or_malformed_recorded_validation_never_reenters_gains(receipt):
    value = pair()
    value["validation"] = receipt
    row = report.comparison(value)
    assert not row["valid"] and row["P8_vs_F_verification_calls_saved"] is None


def test_incomplete_native_rounds_remain_reportable():
    value = pair()
    value["P8"]["native"]["rounds"] = None
    row = report.comparison(value)
    assert not row["valid"] and row["P8_rounds"] is None
