"""Small synthetic arithmetic/interpretation fixtures, with real-validator wiring."""
import copy
import json

import pytest

from . import report

REAL_VALIDATE = report._validate


def pair(request_id="r0", *, population="source", group_id="gsm8k_00", domain="gsm8k",
         f_calls=10, s_calls=12, l_calls=8, f_loss=2., s_loss=.5, l_loss=1.):
    value = dict(request_id=request_id, group_id=group_id, domain=domain, population=population,
                 validation={"status": "passed", "errors": []})
    for arm, calls, loss in (("F", f_calls, f_loss), ("S8", s_calls, s_loss), ("L8", l_calls, l_loss)):
        value[arm] = dict(native=dict(verification_calls=calls, output_tokens=100, accepted_draft_tokens=99-calls,
            verification_committed_tokens=99, initial_output_tokens=1, cap_hit=False, eos_hit=True,
            checkpoint_identity_sha256=arm, checkpoint_id=arm, input_ids=[1,2],
            request_s=1000. if arm != "L8" else .001), kl=dict(loss=loss,
            eligible_positions=10, zero_supervision=False, unique_supervised_positions=10,
            normalization_denominator=10., anchor_positions=[2,3], loss_mask_sha256="mask", weights_sha256="weights"))
    return value


def campaign(*pairs, status="complete"):
    groups = {}
    for value in pairs:
        group = groups.setdefault(value["group_id"], dict(group_id=value["group_id"], domain=value["domain"],
            status="complete", source=[], heldout=[], training={"costs":{"teacher_preparation_s":2., "training_s":3.},
            "accounting":{"steps":8,"presentations":16}}, checkpoint={"checkpoint_id":"L8"}, costs={"activation_s":2.}))
        group[value["population"]].append(value)
    return dict(status=status, groups=list(groups.values()), actual_wall_s=10., optimizer_steps=48,
                training_presentations=96, actual_native_generations=48, actual_response_kl_evaluations=48, explicit_warmups=0)


@pytest.fixture(autouse=True)
def isolated_validation(monkeypatch):
    def validate(value):
        errors=[]
        if value.get("validation") != {"status":"passed","errors":[]}:
            errors.append("recorded_failure")
        if value.get("reject_during_revalidation"):
            errors.append("recomputed_failure")
        return errors
    monkeypatch.setattr(report, "_validate", validate)


def overall(summary, population="source"):
    return next(row for row in summary["pooled"] if row["scope"]=="overall" and row["population"]==population)


def test_three_arm_calls_pool_as_sums_not_mean_percentages():
    values=[pair("one", f_calls=10,s_calls=20,l_calls=5),pair("two",f_calls=90,s_calls=80,l_calls=90)]
    row=overall(report.summarize(campaign(*values)))
    assert (row["F_verification_calls"],row["S8_verification_calls"],row["L8_verification_calls"])==(100,100,95)
    assert row["L8_vs_F_verification_call_reduction_percent"]==pytest.approx(5.)
    assert row["L8_vs_S8_verification_call_reduction_percent"]==pytest.approx(5.)
    assert row["L8_vs_F_native_better_requests"]==1 and row["L8_vs_F_native_tied_requests"]==1
    assert row["L8_vs_S8_native_worse_requests"]==1


def test_equal_normalized_response_loss_means():
    one,two=pair("one",f_loss=1.,s_loss=.5,l_loss=.8),pair("two",f_loss=9.,s_loss=4.,l_loss=4.2)
    for arm in report.ARMS:
        one[arm]["kl"]["normalization_denominator"]=1.
        two[arm]["kl"]["normalization_denominator"]=1000.
    row=overall(report.summarize(campaign(one,two)))
    assert row["F_mean_kl"]==5. and row["L8_mean_kl"]==2.5 and row["S8_mean_kl"]==2.25
    assert row["L8_vs_F_kl_reduction_percent"]==50.


def test_populations_domains_and_groups_separated():
    summary=report.summarize(campaign(pair("s"),pair("h",population="heldout",l_calls=11),
        pair("other",domain="sharegpt",group_id="sharegpt_00",l_calls=12)))
    assert overall(summary,"source")["L8_vs_F_verification_calls_saved"]==0
    assert overall(summary,"heldout")["L8_vs_F_verification_calls_saved"]==-1
    assert all(row["population"] in report.POPULATIONS for row in summary["pooled"])
    assert len([row for row in summary["pooled"] if row["scope"]=="domain"])==4


def test_zero_supervision_only_excludes_loss():
    value=pair()
    for arm in report.ARMS:
        value[arm]["kl"].update(loss=None,zero_supervision=True,eligible_positions=0)
    row=overall(report.summarize(campaign(value)))
    assert row["valid_requests"]==1 and row["kl_pairs"]==0 and row["zero_supervision_pairs"]==1
    assert row["L8_vs_F_verification_calls_saved"]==2 and row["L8_vs_F_kl_change"] is None


@pytest.mark.parametrize("recorded", [False,True])
def test_invalid_measurements_do_not_reenter_gain_calculations(recorded):
    value=pair()
    if recorded:value["validation"]={"status":"failed","errors":["output mismatch"]}
    else:value["reject_during_revalidation"]=True
    summary=report.summarize(campaign(value))
    row=summary["comparisons"][0]
    assert not row["valid"] and row["L8_loss"]==1.
    assert all(row[f"{a}_vs_{b}_verification_calls_saved"] is None for a,b in report.CONTRASTS)
    assert overall(summary)["valid_requests"]==0


def test_duplicate_and_wrong_population_are_invalid():
    value=pair()
    summary=report.summarize(campaign(value,copy.deepcopy(value)))
    assert overall(summary)["invalid_requests"]==2
    data=campaign(pair())
    data["groups"][0]["source"][0]["population"]="heldout"
    assert not report.summarize(data)["comparisons"][0]["valid"]


def test_lower_lr_harm_and_tie_remain_visible():
    row=overall(report.summarize(campaign(pair("w",l_calls=12),pair("t",l_calls=10))))
    assert row["L8_vs_F_native_worse_requests"]==1 and row["L8_vs_F_native_tied_requests"]==1
    assert row["L8_vs_F_verification_call_reduction_percent"]==pytest.approx(-10.)


def test_all_historical_and_fresh_origins_are_explicit():
    for population in report.POPULATIONS:
        row=report.comparison(pair(population=population))
        for arm in report.ARMS:
            assert row[f"{arm}_kl_origin"]==row[f"{arm}_native_origin"]==("fresh" if arm=="L8" else "retained")


def test_no_latency_claim_and_cost_receipts_remain_separate(tmp_path):
    summary=report.write_reports(campaign(pair()),tmp_path)
    assert all("request_s" not in row and "request_time_reduction_percent" not in row for row in summary["comparisons"]+summary["pooled"])
    cost=summary["costs"][0]
    assert cost["training_components"]=={"teacher_preparation_s":2.,"training_s":3.}
    assert cost["components"]=={"activation_s":2.}
    text=(tmp_path/"REPORT.md").read_text()
    assert "not a latency comparison" in text and "No latency-reduction or serving-speedup estimate" in text
    assert "fresh AdamW state" in text and "Held-out responses never enter training" in text


def test_reduced_harm_is_not_transfer():
    result=report.summarize(campaign(pair(),pair("held",population="heldout",f_calls=10,s_calls=14,l_calls=11)))
    text=" ".join(result["interpretations"][0]["patterns"])
    assert "reduced harm, not successful transfer" in text
    assert "supporting transfer" not in text


def test_recovery_to_frozen_is_not_extra_gain():
    result=report.summarize(campaign(pair(),pair("held",population="heldout",f_calls=10,s_calls=14,l_calls=10)))
    text=" ".join(result["interpretations"][0]["patterns"])
    assert "recovery to frozen performance" in text and "supporting transfer" not in text


def test_source_fit_and_fewer_heldout_calls_support_local_transfer_only():
    result=report.summarize(campaign(pair(),pair("held",population="heldout",l_calls=9)))
    text=" ".join(result["interpretations"][0]["patterns"])
    assert "supporting transfer on these requests" in text
    assert "No numerical negligibility cutoff was prespecified" in text
    assert "fresh held-out confirmation" in result["interpretation_boundary"]


def test_no_source_learning_cannot_be_declared_success():
    result=report.summarize(campaign(pair(l_loss=2.),pair("held",population="heldout",l_calls=9)))
    text=" ".join(result["interpretations"][0]["patterns"])
    assert "useful source learning is not established" in text and "supporting transfer" not in text


def test_incomplete_results_stay_partial_and_raw_not_mutated(tmp_path):
    value=pair()
    value["F"]["native"].update(text="private response",output_ids=[1,2,3])
    before=copy.deepcopy(value)
    summary=report.write_reports(campaign(value),tmp_path)
    assert value==before and "private response" not in json.dumps(summary)
    assert summary["status"]=="partial_or_failed"
    assert (tmp_path/"TRAINING.csv").exists()


def test_empty_results_and_nonfinite_are_preserved(tmp_path):
    result=report.write_reports({"groups":[],"status":"planned"},tmp_path)
    assert result["status"]=="not_executed"
    assert result["planned"]["optimizer_steps"]==48
    value=pair();value["L8"]["kl"]["loss"]=float("nan");value["reject_during_revalidation"]=True
    report.write_reports(campaign(value),tmp_path)
    saved=json.loads((tmp_path/"SUMMARY.json").read_text())
    assert saved["comparisons"][0]["L8_loss"]=="NaN"
    assert saved["comparisons"][0]["L8_vs_F_kl_change"] is None


def test_complete_campaign_requires_six_groups_all_pairs():
    values=[pair(f"{group}_{pop}_{i}",group_id=f"g{group}",population=pop)
            for group in range(6) for pop in report.POPULATIONS for i in range(4)]
    data=campaign(*values)
    result=report.summarize(data)
    assert result["status"]=="complete" and result["training_presentations"]==96
    data["groups"][-1]["status"]="failed"
    assert report.summarize(data)["status"]=="partial_or_failed"


def test_validator_failure_or_nonstandard_return_not_accepted(monkeypatch):
    # The real three-arm validator is developed independently. Verify this module
    # calls it rather than trusting a passed execution receipt.
    from . import experiment
    monkeypatch.setattr(report,"_validate",REAL_VALIDATE)
    monkeypatch.setattr(experiment,"validate_pair",lambda _: {"status":"failed","errors":["counter invalid"]})
    assert not report.comparison(pair())["valid"]
    monkeypatch.setattr(experiment,"validate_pair",lambda _: None)
    assert not report.comparison(pair())["valid"]
    monkeypatch.setattr(experiment,"validate_pair",lambda _: {"status":"passed","errors":[]})
    value=pair();value["validation"]={"status":"failed","errors":["execution rejected"]}
    assert not report.comparison(value)["valid"]


def test_complete_status_requires_correct_new_work_counts():
    values=[pair(f"{group}_{pop}_{i}",group_id=f"g{group}",population=pop)
            for group in range(6) for pop in report.POPULATIONS for i in range(4)]
    data=campaign(*values);data["optimizer_steps"]=49
    report_value=report.summarize(data)
    assert report_value["status"]=="partial_or_failed"
    assert "optimizer_steps_does_not_match_completed_scope" in report_value["accounting_errors"]


@pytest.mark.parametrize('mutation', ['none','counter','anchor','heldout_role','execution_rejection'])
def test_actual_three_arm_validator_integration(monkeypatch,mutation):
    from experiments.dflash_kl_native_diagnostic.test_experiment import pair_fixture
    value=pair_fixture('heldout')
    for arm in ('F','S8'):
        value[arm]['kl']['anchor_rng_after_sha256']='a'*64
    value['L8']=copy.deepcopy(value['F'])
    value['checkpoint_bindings']['L8']='c'*64
    for kind in ('native','kl'):
        value['L8'][kind].update(checkpoint_id='L8',checkpoint_identity_sha256='c'*64)
    value['L8']['kl']['role']='evaluation'
    if mutation=='counter':value['L8']['native']['accepted_draft_tokens']=99
    if mutation=='anchor':value['L8']['kl']['anchor_seed']+=1
    if mutation=='heldout_role':value['L8']['kl']['role']='feedback'
    if mutation=='execution_rejection':value['validation']={'status':'failed','errors':['execution failure']}
    monkeypatch.setattr(report,'_validate',REAL_VALIDATE)
    result=report.summarize(campaign(value))
    row=result['comparisons'][0]
    assert row['valid'] is (mutation=='none')
    assert row['kl_valid'] is (mutation=='none')
    assert overall(result,'heldout')['valid_requests']==(1 if mutation=='none' else 0)
