"""C identity, immutable ancestry and saved recovery contracts; CPU only."""
from copy import deepcopy
import io
import unittest

import torch
import yaml

from experiments.unchanged_c_mathqa_transfer import state as S
from experiments.ok_continuation.resume import differences
from online_feedback.prefix_allocation import prefix_allocation_contract
from online_feedback.rng import capture_rng
from test_mathqa_fixed_lr_transfer_training import origin, own_state as native_state, BINDING


def candidate(event=4):
    source, native, _, args = native_state("N", event)
    state, receipt = S.fork_state(source, arm="C", binding=BINDING,
        config_text=native["config_text"], run_id=source["run_id"],
        original_config_text=args["original_config_text"], expected_cuda_rng_states=0)
    if event > 4:
        state = deepcopy(state)
        for key in ("global_step", "presentations", "backend", "events"):
            state[key] = deepcopy(native[key])
        for entry in state["events"][4:]:
            entry.pop(S.baseline.CONTRACT_KEY)
            entry.update(method="C", **{S.CONTRACT_KEY:S.experiment_contract("C",BINDING)})
            for row in entry["records"]:
                row["response_observation"]["prefix_allocation"] = dict(
                    changed_positions=0, changed_blocks=0, weight_moved=0.,
                    prefix_weight_before=1., prefix_weight_after=1., mass_error_abs_sum=0.,
                    delta_numerator=0., mass_violations=0, bounds_violations=0,
                    support_violations=0, nonfinite_values=0)
    args.update(arm="C", expected_recipe=deepcopy(state["backend"]["optimizer"]["recipe"]))
    return source, state, receipt, args


class StateTests(unittest.TestCase):
    def test_fork_preserves_native_tensors_rng_and_history(self):
        source, state, receipt, args = candidate()
        before, rng = deepcopy(source), capture_rng()
        for key in ("draft_state_dict","backend","rng","events"):
            self.assertIs(source[key],state[key])
        self.assertEqual(state["algorithm_contract"][S.CONTRACT_KEY]["allocation"],prefix_allocation_contract())
        self.assertEqual(S.validate_snapshot(state,**args)["global_step"],512)
        self.assertEqual(differences(before,source),[])
        self.assertEqual(differences(rng,capture_rng()),[])
        self.assertFalse(receipt["seed_reset"])

    def test_complete_boundaries_preserve_allocation_and_serialize_readonly(self):
        for event in (4,5,6):
            _, state, _, args = candidate(event)
            before = deepcopy(state)
            checked = S.validate_snapshot(state,**args)
            self.assertEqual(checked["global_step"],128*event)
            stream=io.BytesIO();torch.save(state,stream);stream.seek(0)
            decoded=torch.load(stream,weights_only=False)
            S.validate_snapshot(decoded,**args)
            self.assertEqual(differences(before,decoded),[])
            self.assertEqual(differences(before,state),[])
            self.assertEqual(decoded["horizon"],7424)

    def test_changed_cap_rate_ancestry_or_counter_fails(self):
        _, state, _, args = candidate(5)
        mutations = [
            lambda x:x["algorithm_contract"][S.CONTRACT_KEY]["allocation"].update(relative_cap=0.),
            lambda x:x["algorithm_contract"][S.CONTRACT_KEY].update(arm="N"),
            lambda x:x["backend"]["optimizer"]["optimizer_state_dict"]["param_groups"][0].update(lr=2.88e-4),
            lambda x:x["events"][0]["records"][1].update(learning_rate=2.88e-4),
            lambda x:x.update(global_step=512), lambda x:x.update(horizon=768),
            lambda x:x["events"][4]["records"][0]["response_observation"]["prefix_allocation"].update(support_violations=1),
            lambda x:x["events"][4]["records"][0]["response_observation"]["prefix_allocation"].update(weight_moved=float("nan")),
        ]
        for mutation in mutations:
            bad=deepcopy(state);mutation(bad)
            with self.assertRaises(ValueError):S.validate_snapshot(bad,**args)
        with self.assertRaises(ValueError):S.validate_snapshot(state,**dict(args,completed_event=7))
        with self.assertRaises(ValueError):S.validate_origin(state,args["original_config_text"],expected_cuda_rng_states=0)

    def test_settings_and_original_native_guards_remain_strict(self):
        _,state,_,args=candidate()
        for section,key,value in (("training","num_anchors",32),("training","learning_rate",2.88e-4),
                                  ("training","seed",23),("training","max_steps",768),
                                  ("online_feedback","weight_decay",0.)):
            raw=yaml.safe_load(state["config_text"]);raw[section][key]=value
            with self.assertRaises(ValueError):S.validate_config_text(yaml.safe_dump(raw),"C",args["original_config_text"])
        with self.assertRaises(ValueError):S.arm_lr("N")
        with self.assertRaises(ValueError):S.baseline.validate_snapshot(state,**dict(args,arm="N"))


if __name__=="__main__":unittest.main()
