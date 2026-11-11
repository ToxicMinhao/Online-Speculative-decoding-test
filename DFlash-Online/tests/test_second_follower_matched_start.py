import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

from experiments.second_follower import matched_start as m
from experiments.longer_run_headroom.census import parent_rows
from online_feedback.storage import write_json


def fixture():
    canonical = list(range(32)) + [99]
    prompt = [500, 501]
    def block(start, accepted, *, finished=False):
        proposed = canonical[start+1:start+16]
        if accepted < len(proposed):
            proposed[accepted] = 999
        progress = accepted if finished else accepted+1
        return dict(output_start=start,anchor_token=canonical[start],draft_tokens=proposed,
                    target_argmax=canonical[start+1:start+17],positions=list(range(len(prompt)+start,len(prompt)+start+16)),
                    accepted_draft_tokens=accepted,native_accepted_draft_tokens=max(progress-1,0),
                    raw_accepted_draft_tokens_before_stop=accepted,committed_progress=progress,
                    committed_tokens=canonical[start+1:start+1+progress],finished=finished,stop_clipped=finished)
    blocks = [block(0,14),block(15,1),block(17,15,finished=True)]
    request = dict(request_id="r0",prompt_ids=prompt,max_new_tokens=2048,stop_token_ids=[99],panel_ordinal=0)
    item = dict(dataset="gsm8k",request_id="r0",selection_rank=0,request=request,
                canonical=dict(response_ids=canonical,finish_reason="eos"))
    parent = dict(record_id="r0",checkpoint_id="parent",prompt_ids=prompt,response_ids=canonical,
                  max_new_tokens=2048,stop_token_ids=[99],finish_reason="eos",
                  metadata=dict(trace=blocks,verification_calls=3))
    selected = parent_rows(parent,item)[1]
    selected["inclusion_fraction"]=64/403
    def receipt(accepted=1, *, proposals=None, finished=False):
        child=block(15,accepted,finished=finished)
        if proposals is not None:
            child["draft_tokens"]=proposals
        response=canonical[:16+child["committed_progress"]]
        return dict(status="completed",blocks=[child],response_ids=response,
                    output_matches_canonical_prefix=True,native_finished=finished,
                    original_cap=2048,original_remaining_budget=2032,cap_unchanged=True,
                    reconstruction=dict(status="passed",target_prefix_cache_exact=True,
                        neutral_draft_prefix_cache_exact=True,destination_draft_rebuilt=True,
                        source_draft_cache_transplanted=False,original_cap=2048,remaining_budget=2032))
    return item,parent,selected,receipt


def inputs_fixture():
    selected=[dict(dataset="gsm8k",request_id=f"r{i%57}",parent_round_ordinal=i+2,
                   group="SHORT" if i<64 else "LONG",opening=False,j=2) for i in range(128)]
    jobs=[dict(starts=[r for r in selected if r["request_id"]==f"r{i}"],expected_parent_rounds=44 if i<56 else 70) for i in range(57)]
    inputs=dict(schema=m.SCHEMA,selected=selected,jobs=jobs,request_clusters=[f"r{i}" for i in range(64)],
                parent_checkpoint=dict(cumulative_adam_steps=512,weights_identity=dict(sha256=m.PARENT_SHA256)),
                limits=m.LIMITS,startup_block_limit=64,census_weights=m.CENSUS_WEIGHTS,
                historical_child_outcomes_consumed=False)
    inputs["binding_sha256"]=m.digest(inputs)
    return inputs


def checkpoints():
    return {arm:dict(checkpoint_path=arm,weights_identity=dict(bytes=1,sha256=m.NATIVE_SHA256 if arm=="NATIVE" else "a"*64),
                     cumulative_adam_steps=640) for arm in m.ARMS}


def analysis_rows():
    rows=[]
    for i in range(128):
        selected=dict(request_id=f"r{i%64}",parent_round_ordinal=i+2,group="SHORT" if i<64 else "LONG",j=2)
        base=dict(status="valid",child_accepted=1,prefix_gained=0,prefix_lost=0,**{k:0 for k in m.METRICS})
        native=dict(base)
        second=dict(base,native_prefix_change=1,child_accepted=2,prefix_gained=1,repair=1)
        rows.append(dict(selected=selected,status="valid",children=dict(NATIVE=native,SECOND=second)))
    return rows


class ContractTests(unittest.TestCase):
    def test_all_sealed_starts_are_required_and_binding_detects_mutation(self):
        value=inputs_fixture()
        self.assertIs(m.validate_inputs(value),value)
        for field, mutate in [("subset",lambda v:v["selected"].pop()),
                              ("replacement",lambda v:v["selected"][0].update(request_id="other")),
                              ("duplicate",lambda v:v["selected"].__setitem__(0,copy.deepcopy(v["selected"][1]))),
                              ("old_outcomes",lambda v:v.update(historical_child_outcomes_consumed=True))]:
            with self.subTest(field=field):
                changed=copy.deepcopy(value);mutate(changed)
                changed["binding_sha256"]=m.digest({k:v for k,v in changed.items() if k!="binding_sha256"})
                with self.assertRaises(ValueError):m.validate_inputs(changed)
        value["jobs"][0]["expected_parent_rounds"]+=1
        with self.assertRaises(ValueError):m.validate_inputs(value)

    def test_control_identity_and640_step_are_mandatory(self):
        value=checkpoints()
        self.assertEqual(m.checkpoint_contract(value)["NATIVE"]["export"],"NATIVE")
        value["NATIVE"]["cumulative_adam_steps"]=768
        with self.assertRaisesRegex(ValueError,"640"):m.checkpoint_contract(value)
        value=checkpoints();value["NATIVE"]["weights_identity"]["sha256"]="b"*64
        with self.assertRaisesRegex(ValueError,"PR39"):m.checkpoint_contract(value)
        value=checkpoints();value["SECOND"]["weights_identity"]=value["NATIVE"]["weights_identity"]
        with self.assertRaisesRegex(ValueError,"Inactive"):m.checkpoint_contract(value)

    def test_expired_phase_preserves_every_pending_start_without_modelwork(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,"preflight") as preflight, patch.object(m.subprocess,"Popen") as launch:
            result=m.run(tmp,dict(matched_start=inputs_fixture()),checkpoints(),remaining_s=0)
            self.assertEqual(result["status"],"incomplete")
            self.assertEqual(result["pending"],128)
            self.assertEqual(result["work"]["gpu_seconds"],0)
            self.assertEqual(len(json.loads((Path(tmp)/"matched_start/ROWS.json").read_text())),128)
            preflight.assert_not_called();launch.assert_not_called()

    def test_negative_and_nonfinite_budget_rejected(self):
        for value in (-1,4501,float("nan"),float("inf")):
            with self.assertRaises(ValueError):m.run("unused",{},checkpoints(),remaining_s=value)

    def test_panel_compares_tokens_across_different_id_namespaces(self):
        inp=dict(development_requests=[dict(request_id=f"dev{i}",prompt_ids=[i]) for i in range(64)])
        panel=[dict(request_id=f"eval{i}",prompt_ids=[i+100]) for i in range(256)]
        self.assertEqual(m.verify_panel_disjoint(inp,panel)["rendered_prompt_token_identity_overlap"],0)
        panel[12]["prompt_ids"]=[0]
        with self.assertRaisesRegex(ValueError,"overlaps"):m.verify_panel_disjoint(inp,panel)


class ScoringTests(unittest.TestCase):
    def setUp(self):self.item,self.parent,self.s,self.receipt=fixture()
    def evaluate(self,observed,selected=None):
        return m.score_child(selected or self.s,self.parent["metadata"]["trace"][1],observed,
                             self.item["canonical"]["response_ids"],arm="SECOND",checkpoint="second640")

    def test_two_followers_survive_without_being_accepted_if_blocker_remains(self):
        r=self.evaluate(self.receipt())
        self.assertTrue(r["original_followers_intact"])
        self.assertFalse(r["repair"])
        self.assertFalse(r["first_follower_accepted"])
        self.assertFalse(r["original_followers_accepted"])

    def test_long_first_two_acceptance_threshold_is_j_plus_two(self):
        r=self.evaluate(self.receipt(2))
        self.assertTrue(r["repair"]);self.assertFalse(r["first_follower_accepted"])
        r=self.evaluate(self.receipt(3))
        self.assertTrue(r["first_follower_accepted"]);self.assertFalse(r["first_two_original_followers_accepted"])
        r=self.evaluate(self.receipt(4))
        self.assertTrue(r["first_two_original_followers_accepted"])
        self.assertEqual(r["native_prefix_change"],3)
        self.assertEqual(r["follower_scope"],"first_two_original_followers")

    def test_short_only_requires_first_original_follower(self):
        selected=dict(self.s,group="SHORT",R=1)
        r=self.evaluate(self.receipt(3),selected)
        self.assertTrue(r["original_followers_accepted"])
        self.assertIsNone(r["first_two_original_followers_accepted"])

    def test_earlier_break_does_not_count_as_repair_even_when_old_error_matches(self):
        observed=self.receipt(0)
        proposed=list(range(16,31));proposed[0]=999
        observed["blocks"][0]["draft_tokens"]=proposed
        r=self.evaluate(observed)
        self.assertTrue(r["earlier_prefix_break"])
        self.assertFalse(r["repair"])
        self.assertEqual(r["prefix_lost"],1)

    def test_canonical_old_error_match_without_native_acceptance_is_not_repair(self):
        observed=self.receipt(1)
        observed["blocks"][0]["draft_tokens"][1]=17
        r=self.evaluate(observed)
        self.assertTrue(r["canonical_repair"])
        self.assertFalse(r["repair"])

    def test_cap_kv_noncanonical_output_and_counts_fail_closed(self):
        for name, mutate in [
            ("cap",lambda r:r.update(original_cap=16)),
            ("targetKV",lambda r:r["reconstruction"].update(target_prefix_cache_exact=False)),
            ("transplant",lambda r:r["reconstruction"].update(source_draft_cache_transplanted=True)),
            ("wrongoutput",lambda r:r["response_ids"].__setitem__(-1,999)),
            ("counter",lambda r:r["blocks"][0].update(native_accepted_draft_tokens=14)),
        ]:
            with self.subTest(name=name):
                observed=self.receipt();mutate(observed)
                with self.assertRaises(ValueError):self.evaluate(observed)

    def test_eos_acceptance_is_clipped_but_includes_valid_eos(self):
        canonical=self.item["canonical"]["response_ids"][:20]
        canonical[-1]=99
        parent=self.parent["metadata"]["trace"][1]
        parent["draft_tokens"][3]=99
        selected=dict(self.s,valid_slots=4,parent_block_sha256=m.digest(parent))
        observed=self.receipt(4)
        child=observed["blocks"][0]
        child.update(draft_tokens=canonical[16:20]+[123]*11,raw_accepted_draft_tokens_before_stop=15,
                     accepted_draft_tokens=4,native_accepted_draft_tokens=3,committed_progress=4,
                     committed_tokens=canonical[16:20],finished=True,stop_clipped=True)
        observed.update(response_ids=canonical,native_finished=True)
        row=m.score_child(selected,parent,observed,canonical,arm="SECOND",checkpoint="second")
        self.assertTrue(row["first_two_original_followers_accepted"])
        self.assertEqual(row["child_accepted"],4)
        self.assertEqual(row["child_native_accepted"],3)

    def test_neutral_tail_mismatch_is_detected(self):
        p=self.parent["metadata"]["trace"][1];other=copy.deepcopy(p);other["draft_tokens"][-1]+=1
        self.assertEqual(m.neutral_parity(p,other)["status"],"failed")


class AnalysisTests(unittest.TestCase):
    def test_paired_fixed_census_weights_and_independent_cluster_arithmetic(self):
        rows=analysis_rows()
        # Unequal numbers of starts per request makes request means inappropriate.
        rows[0]["selected"]["request_id"]="r1"
        rows[64]["children"]["SECOND"].update(native_prefix_change=-3,child_accepted=-2,prefix_gained=0,prefix_lost=3)
        result=m.summarize(rows,[f"r{i}" for i in range(64)],resamples=200)
        rng=np.random.Generator(np.random.PCG64(20261113))
        sample=rng.integers(0,64,size=(200,64))
        vals=[]
        for draw in sample:
            value=0
            for group,weight in m.CENSUS_WEIGHTS.items():
                ds=[]
                for request in draw:
                    ds.extend(r["children"]["SECOND"]["native_prefix_change"] for r in rows
                              if r["selected"]["request_id"]==f"r{request}" and r["selected"]["group"]==group)
                value+=weight*sum(ds)/len(ds)
            vals.append(value)
        expected=m.CENSUS_WEIGHTS["SHORT"]+m.CENSUS_WEIGHTS["LONG"]*(60/64)
        self.assertAlmostEqual(result["census_weighted_SECOND_minus_NATIVE"]["per_later_parent_start"],expected)
        np.testing.assert_allclose(result["census_weighted_SECOND_minus_NATIVE"]["ci95"],np.percentile(vals,[2.5,97.5],method="linear"),atol=1e-15)

    def test_within_request_cancellation_still_counts_contributing_request(self):
        rows=analysis_rows()
        for row in rows:
            row["children"]["SECOND"]=copy.deepcopy(row["children"]["NATIVE"])
        rows[0]["selected"]["request_id"]="r1"
        rows[0]["children"]["SECOND"].update(native_prefix_change=1,child_accepted=2,prefix_gained=1)
        rows[1]["children"]["SECOND"].update(native_prefix_change=-1,child_accepted=0,prefix_lost=1)
        r=m.summarize(rows,[f"r{i}" for i in range(64)],resamples=20)
        metric=r["groups"]["SHORT"]["paired_SECOND_minus_NATIVE"]["native_prefix_change"]
        self.assertEqual(metric["total"],0)
        self.assertEqual(metric["contributing_requests"],1)
        self.assertEqual(r["groups"]["SHORT"]["requests_with_changed_starts"],1)

    def test_undefined_cluster_draws_are_counted_not_replaced(self):
        rows=analysis_rows()
        for row in rows:row["selected"]["request_id"]="r0" if row["selected"]["group"]=="SHORT" else "r1"
        r=m.summarize(rows,["r0","r1"],resamples=200)
        self.assertGreater(r["census_weighted_SECOND_minus_NATIVE"]["undefined_draws"],0)
        self.assertEqual(sum(r["census_weighted_SECOND_minus_NATIVE"][k] for k in ("defined_draws","undefined_draws")),200)

    def test_partial_execution_never_estimates_valid_subset(self):
        rows=analysis_rows();rows[-1].update(status="pending",children={})
        r=m.summarize(rows,[f"r{i}" for i in range(64)],resamples=20)
        self.assertEqual(r["status"],"incomplete")
        self.assertNotIn("groups",r)
        self.assertNotIn("census_weighted_SECOND_minus_NATIVE",r)


class BudgetAndWorkerTests(unittest.TestCase):
    def test_startup_and_total_block_caps_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=m.DiagnosticBudget(Path(tmp)/"work",limits=m.LIMITS,gpu_seconds=100)
            for _ in range(64):
                budget.complete(budget.attempt("draft_block",startup=True))
            with self.assertRaises(m.BudgetExceeded):budget.attempt("draft_block",startup=True)
            budget.complete(budget.attempt("draft_block",operation="native_round"))
            self.assertEqual(budget.summary()["startup_blocks"],64)
            self.assertEqual(budget.summary()["completed"]["draft_block"],65)

    def test_inherited_publications_share_model_load_category_with_initialization(self):
        from experiments.verification_transition import native_worker
        with tempfile.TemporaryDirectory() as tmp:
            budget=m.DiagnosticBudget(Path(tmp)/"work",limits=m.LIMITS,gpu_seconds=100)
            budget.complete(budget.attempt("model_load",operation="engine_initialization"))
            worker=native_worker.NativeWorker.__new__(native_worker.NativeWorker)
            worker.budget=budget
            with patch.object(native_worker.sglang_worker.NativeWorker,"publish",return_value={"passed":True}) as publish:
                for _ in range(57):
                    for checkpoint in ("parent","native640","second640"):
                        self.assertEqual(worker.publish({"checkpoint":checkpoint}),{"passed":True})
                self.assertEqual(publish.call_count,171)
            self.assertEqual(budget.summary()["completed"]["model_load"],172)
            with self.assertRaises(m.BudgetExceeded):budget.attempt("model_load")

    def test_worker_freshly_runs_both_children_from_same_handle(self):
        item,parent,selected,receipt=fixture()
        calls=[]
        class Backend:
            loading_s=0
            def __init__(self,*args,**kwargs):pass
            def native_walk(self,request,checkpoint,boundaries):
                calls.append(("walk",checkpoint,boundaries))
                return dict(record=copy.deepcopy(parent),successors={"1":dict(handle="same-handle")})
            def one_block(self,handle,checkpoint):
                calls.append(("block",checkpoint,handle["handle"]))
                return receipt(4 if checkpoint=="second" else 1)
            def release_request(self):calls.append(("release",))
            def close(self):calls.append(("close",))
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"out";output.mkdir()
            plan=dict(matched_start=dict(selected=[selected],jobs=[dict(item=item,parent=parent,starts=[selected])],
                      parent_checkpoint=dict(export="parent"),shared=dict(target_path="target",runtime=dict(python="python"))),
                      checkpoints=dict(NATIVE=dict(export="native"),SECOND=dict(export="second")))
            write_json(Path(tmp)/"plan.json",plan);write_json(Path(tmp)/"connection.json",{})
            budget=m.DiagnosticBudget(Path(tmp)/"work",limits=m.LIMITS,gpu_seconds=100)
            with patch.object(m,"connect_budget",return_value=budget):
                code=m.worker(Path(tmp)/"plan.json",output,Path(tmp)/"connection.json",backend_factory=Backend)
            self.assertEqual(code,0)
            self.assertEqual(calls[0],("walk","parent",[1]))
            self.assertEqual([r for r in calls if r[0]=="block"],[("block",a,"same-handle") for a in ("parent","native","second")])
            rows=json.loads((output/"ROWS.json").read_text())
            self.assertEqual(rows[0]["status"],"valid")
            self.assertEqual(set(rows[0]["children"]),set(m.ARMS))
            self.assertEqual(budget.summary()["completed"]["child_block"],2)

    def test_budget_stop_between_children_keeps_partial_pair_and_failed_work(self):
        item,parent,selected,receipt=fixture()
        class Backend:
            loading_s=0
            def __init__(self,*args,**kwargs):pass
            def native_walk(self,*args):
                return dict(record=copy.deepcopy(parent),successors={"1":dict(handle="h")})
            def one_block(self,handle,checkpoint):
                if checkpoint=="second":raise m.BudgetExceeded("phase deadline")
                return receipt()
            def release_request(self):raise AssertionError("stopped request cannot complete")
            def close(self):pass
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"out";output.mkdir()
            plan=dict(matched_start=dict(selected=[selected],jobs=[dict(item=item,parent=parent,starts=[selected])],
                      parent_checkpoint=dict(export="parent"),shared=dict(target_path="target",runtime=dict(python="python"))),
                      checkpoints=dict(NATIVE=dict(export="native"),SECOND=dict(export="second")))
            write_json(Path(tmp)/"plan.json",plan);write_json(Path(tmp)/"connection.json",{})
            budget=m.DiagnosticBudget(Path(tmp)/"work",limits=m.LIMITS,gpu_seconds=100)
            with patch.object(m,"connect_budget",return_value=budget):
                code=m.worker(Path(tmp)/"plan.json",output,Path(tmp)/"connection.json",backend_factory=Backend)
            self.assertEqual(code,2)
            row=json.loads((output/"ROWS.json").read_text())[0]
            self.assertEqual(row["status"],"pending")
            self.assertEqual(set(row["children"]),{"NATIVE"})
            self.assertEqual(budget.summary()["completed"]["child_block"],1)
            self.assertEqual(budget.summary()["failed"]["child_block"],1)
            self.assertEqual(budget.summary()["unknown"],{})

if __name__ == "__main__":unittest.main()
