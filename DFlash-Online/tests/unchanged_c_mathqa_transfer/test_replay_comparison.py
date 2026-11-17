"""Exact replay tests use tiny CPU tensors and synthetic scientific receipts."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml

from experiments.unchanged_c_mathqa_transfer import replay_comparison as replay


def fixture(arm="N"):
    lr = 0.0003 if arm == "N" else 0.000288
    responses, windows, presentations = [], [], []
    for index in range(256):
        stepped = index % 2 == 1
        response = dict(scope="one_original_response_forward", record_id=f"r{index % 128:03d}",
            event_id="event_005", pass_index=index // 128, presentation_ordinal=index,
            sampled_anchors=[index % 4, 5], loss=0.125,
            ratio_metrics={key: [1.0, 8.0] for key in replay.RATIO_FIELDS},
            sum_metrics={key: 1.0 for key in replay.SUM_FIELDS})
        responses.append(response)
        metrics = {key: 1.0 for key in replay.METRIC_FIELDS}
        if stepped:
            metrics["grad_norm"] = 0.25
        presentations.append(dict(record_id=response["record_id"], **{"pass":index // 128},
            optimizer_step_after=512 + (index + 1) // 2, optimizer_stepped=stepped,
            learning_rate=lr, metrics=metrics, training_s=0.5, sampled_anchors=response["sampled_anchors"],
            metrics_scope="response", response_observation=response,
            prompt_tokens=12, response_tokens=40, active_anchors=2))
        if stepped:
            windows.append(dict(scope="optimizer_window", event_id="event_005", window=(index + 1)//2,
                global_step=512+(index + 1)//2, record_ids=[responses[-2]["record_id"], response["record_id"]],
                gradient_norm=0.25, learning_rate=lr, mean_response_loss=0.125))
    event = dict(event_id="event_005", features=["/retained/e5/feature.pt"], passes=2,
        response_presentations=256, start_step=512, end_step=640, optimizer_steps=128,
        reduction="response_average", objective="osd_first_error",
        objective_work={**{key: 1.0 for key in replay.OBJECTIVE_TIMES}, "reference_operations":0,
            "work":dict(draft_forwards=256,layer_forwards=1280,objective_chunks=512,
                        vocabulary_head_calls=1024,response_forwards=256)},
        gradient_denominator=2, records=presentations, selection_totals={"available":512},
        selection_totals_source="response_observations", method=arm, dataset="mathqa",
        kplus_loss={"retained":"native"}, mathqa_fixed_lr_transfer_v1={"binding":"original"})
    state = dict(event_schema="full_k_v1", draft_state_dict={"weight":torch.tensor([1.,2.],dtype=torch.bfloat16)},
        backend={"optimizer":{"state":{0:{"step":torch.tensor(640.),"exp_avg":torch.tensor([0.25],dtype=torch.bfloat16),
            "exp_avg_sq":torch.tensor([0.5],dtype=torch.bfloat16)}},"param_groups":[{"lr":lr,"weight_decay":0.01}]}},
        rng={"torch":torch.tensor([1,2],dtype=torch.uint8),"numpy":("MT",np.array([3,4],dtype=np.uint32),0)},
        global_step=640,presentations=1280,events=[{"event_id":f"event_00{i}"} for i in range(1,5)]+[deepcopy(event)],
        strategy="specforge",run_id="historical",world_size=1,
        config_text=yaml.safe_dump(dict(run_id="historical",output_dir="/historical/output",training={"lr":lr,"seed":17})),
        algorithm_contract={"native":True},reduction="response_average",accumulation_steps=2,horizon=7424)
    event.update(status="completed", checkpoint_path="/old/checkpoint",export_path="/old/export",
        full_state_path="/old/state",costs={key:1.0 for key in replay.COST_TIMES},
        counters=dict(global_step=640,presentations=1280,completed_events=5))
    parameter=torch.tensor([1.],dtype=torch.bfloat16,requires_grad=True)
    trainer=SimpleNamespace(global_step=640,presentations=1280,
        core=SimpleNamespace(_micro=1280,accumulation_remainder=0,_ratio_totals={},_sum_totals={}),
        bundle=SimpleNamespace(model=SimpleNamespace(training=True,reference=None,ranking_enabled=False,
            objective="osd_first_error"),draft_model=SimpleNamespace(parameters=lambda:iter([parameter]))),
        _presentation_schedule=None,closed=False,failed=False,_anchor_observations=responses[-1]["sampled_anchors"],
        _check_native_path=lambda:None,_check_actual_lr=lambda:None)
    return state,event,responses,windows,trainer


class ReplayComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output=Path(self.tmp.name)/"comparison"
        self.state,self.event,self.responses,self.windows,self.trainer=fixture()

    def comparator(self):
        return replay.ReplayComparator(self.state,self.event,self.responses,self.windows,
            arm=self.event["method"],output_dir=self.output)

    def observe_all(self, compare):
        for index,response in enumerate(self.responses):
            compare.observe_response(deepcopy(response))
            if index%2:
                compare.observe_window(deepcopy(self.windows[index//2]))
            compare.observe_presentation(deepcopy(self.event["records"][index]))

    def event_ready(self):
        compare=self.comparator()
        self.observe_all(compare)
        self.assertEqual(compare.compare_event(deepcopy(self.event))["status"],"passed")
        self.assertEqual(compare.receipt["status"],"running")
        return compare

    def failure(self, compare_call, field):
        with self.assertRaisesRegex(ValueError,"Exact replay mismatch"):
            compare_call()
        receipt=json.loads((self.output/"COMPARISON.json").read_text())
        self.assertEqual(receipt["status"],"failed")
        self.assertIn(field,receipt["first_divergence"]["field"])
        self.assertFalse(receipt["historical_feature_identity_verified"])
        return receipt

    def test_complete_exact_comparison_and_declared_metadata_exclusions(self):
        compare=self.comparator()
        self.observe_all(compare)
        event=deepcopy(self.event)
        event["checkpoint_path"]="/new/output"
        event["costs"]["training_s"]=999.0
        event["objective_work"]["teacher_head_s"]=999.0
        event["records"][0]["training_s"]=999.0
        phase=compare.compare_event(event)
        self.assertTrue(phase["full_gate_pending_endpoint"])
        live=deepcopy(self.state)
        live["run_id"]="new-replay"
        config=yaml.safe_load(live["config_text"])
        config.update(run_id="new-replay",output_dir="/new/output")
        live["config_text"]=yaml.safe_dump(config)
        live["events"][-1]=event
        receipt=compare.compare_endpoint(live,self.trainer)
        self.assertEqual(receipt["status"],"passed")
        self.assertEqual(receipt["categories"],dict.fromkeys(replay.CATEGORIES,"exact"))
        self.assertEqual(receipt["exclusions"],replay.EXCLUSIONS)
        self.assertEqual(receipt["extra_optimizer_steps"],0)
        self.assertEqual(receipt["tensor_comparisons"],6)
        self.assertTrue(receipt["endpoint_compared"])
        self.assertEqual(self.state["run_id"],"historical")

    def test_response_one_ulp_difference_stops_before_counter_advance(self):
        compare=self.comparator()
        row=deepcopy(self.responses[0])
        row["loss"]=np.nextafter(row["loss"],1.0).item()
        receipt=self.failure(lambda:compare.observe_response(row),"response[0].loss")
        self.assertEqual(receipt["responses_compared"],0)

    def test_missing_required_historical_field_stops_durably(self):
        del self.event["method"]
        self.failure(lambda: replay.ReplayComparator(self.state,self.event,self.responses,self.windows,
            arm="N",output_dir=self.output),"event.fields")

    def test_missing_nested_scientific_metric_is_not_a_match(self):
        del self.responses[0]["sum_metrics"]["loss_denominator"]
        self.failure(self.comparator,"sum_metrics.fields")

    def test_unknown_live_scientific_field_is_rejected(self):
        compare=self.comparator()
        row=deepcopy(self.responses[0]);row["ignored_loss"]=1.
        self.failure(lambda:compare.observe_response(row),"response[0].fields")

    def test_window_and_natural_anchor_mismatches_are_exact(self):
        compare=self.comparator()
        row=deepcopy(self.windows[0]);row["learning_rate"]*=0.96
        self.failure(lambda:compare.observe_window(row),"window[0].learning_rate")

    def test_incomplete_event_cannot_pass(self):
        compare=self.comparator()
        self.failure(lambda:compare.compare_event(self.event),"complete_observation_counts")

    def test_input_feature_paths_remain_scientific(self):
        compare=self.comparator();self.observe_all(compare)
        event=deepcopy(self.event);event["features"]=["/different/input.pt"]
        self.failure(lambda:compare.compare_event(event),"event5.features[0]")

    def test_unknown_timing_key_is_not_silently_excluded(self):
        compare=self.comparator();self.observe_all(compare)
        event=deepcopy(self.event);event["costs"]["unknown_s"]=1.
        self.failure(lambda:compare.compare_event(event),"event.costs.fields")

    def test_tensor_mismatch_has_compact_fingerprints(self):
        compare=self.event_ready();live=deepcopy(self.state)
        live["backend"]["optimizer"]["state"][0]["step"].add_(1)
        self.failure(lambda:compare.compare_endpoint(live,self.trainer),"endpoint.backend.optimizer.state.0.step")
        rows=[json.loads(line) for line in (self.output/"TENSOR_COMPARISONS.jsonl").read_text().splitlines()]
        self.assertFalse(rows[-1]["equal"])
        self.assertIn("sha256",rows[-1]["expected"])
        self.assertNotIn("values",rows[-1]["actual"])
        self.assertEqual(self.state["backend"]["optimizer"]["state"][0]["step"].item(),640)

    def test_nonfinite_tensor_is_failure_even_if_equal(self):
        self.state["draft_state_dict"]["weight"][0]=float("nan")
        compare=self.event_ready()
        self.failure(lambda:compare.compare_endpoint(deepcopy(self.state),self.trainer),"endpoint.draft_state_dict.weight")

    def test_rng_and_ancestry_and_scientific_config_stay_required(self):
        for name in ("rng","ancestry","config"):
            with self.subTest(name=name):
                self.output=Path(self.tmp.name)/name
                compare=self.event_ready();live=deepcopy(self.state)
                if name=="rng":
                    live["rng"]["torch"][0]+=1
                    field="endpoint.rng"
                elif name=="ancestry":
                    live["events"][0]["event_id"]="substituted"
                    field="original_event_ancestry"
                else:
                    config=yaml.safe_load(live["config_text"]);config["training"]["lr"]=0.000288
                    live["config_text"]=yaml.safe_dump(config);field="scientific_config_text.training.lr"
                self.failure(lambda:compare.compare_endpoint(live,self.trainer),field)

    def test_clean_accumulation_and_unforced_sampling_required(self):
        compare=self.event_ready()
        self.trainer._presentation_schedule=["forced"]
        self.failure(lambda:compare.compare_endpoint(deepcopy(self.state),self.trainer),"sampler_accumulation_boundary.schedule_absent")

    def test_missing_optimizer_boundary_gradient_metric_rejected(self):
        del self.event["records"][1]["metrics"]["grad_norm"]
        self.failure(self.comparator,"presentation.metrics.fields")

    def test_scaled_arm_preserves_literal_lr(self):
        self.state,self.event,self.responses,self.windows,self.trainer=fixture("S")
        compare=self.event_ready()
        self.assertEqual(compare.compare_endpoint(deepcopy(self.state),self.trainer)["status"],"passed")
        self.assertEqual(self.state["backend"]["optimizer"]["param_groups"][0]["lr"],0.000288)


if __name__=="__main__":
    unittest.main()
