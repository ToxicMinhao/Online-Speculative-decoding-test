"""Native replay envelopes preserve exact training and stop at the first mismatch."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
import yaml

from experiments.unchanged_c_mathqa_transfer.replay import ControlReplayTrainer, _json_lines
from experiments.unchanged_c_mathqa_transfer.replay_comparison import ReplayComparator
from experiments.mathqa_fixed_lr_transfer.training import MathQAFixedLRTrainer
from experiments.mathqa_fixed_lr_transfer import state as S
from experiments.ok_continuation.resume import differences
import test_mathqa_fixed_lr_transfer_training as baseline_fixtures


class ReplayTinyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("MATHQA_REPLAY_TINY_CHILD")=="1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("MATHQA_REPLAY_TINY_CHILD")=="1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get("MATHQA_REPLAY_TINY_CHILD")=="1":
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUp(self)

    make_parent=baseline_fixtures.MathQATinyIntegrationTests.make_parent

    def subprocess_or_continue(self):
        if os.environ.get("MATHQA_REPLAY_TINY_CHILD")=="1":return False
        root=Path(__file__).resolve().parents[2]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",MATHQA_REPLAY_TINY_CHILD="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        env["PYTHONPATH"]=os.pathsep.join(str(root/p) for p in (".","tests","training","inference"))
        result=subprocess.run([sys.executable,"-m","unittest","-v",__name__+"."+type(self).__name__+"."+self._testMethodName],
            cwd=root,env=env,text=True,capture_output=True,timeout=240)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return True

    def build(self,name,arm,source,export,*,comparator=None):
        config=self.root/(name+".yaml")
        raw=yaml.safe_load(self.original_config_text)
        raw["training"]["learning_rate"]=S.arm_lr(arm)
        raw.update(run_id=name,output_dir=str(self.root/name))
        config.write_text(yaml.safe_dump(raw))
        cls=MathQAFixedLRTrainer if comparator is None else ControlReplayTrainer
        extra={} if comparator is None else {"comparator":comparator}
        trainer=cls(config,self.root/name,arm=arm,binding=baseline_fixtures.BINDING,
            origin_state=source,origin_export=export,original_config_text=self.original_config_text,
            expected_cuda_rng_states=0,**extra)
        self.addCleanup(trainer.close)
        return trainer

    def historical(self,arm,source,export):
        trainer=self.build("historical_"+arm,arm,source,export)
        receipt=trainer.train_event("event_005",self.features)
        state=deepcopy(torch.load(trainer.latest,weights_only=False))
        directory=trainer.events_root/"event_005"
        responses=_json_lines(directory/"RESPONSE_OBSERVATIONS.jsonl")
        windows=_json_lines(directory/"WINDOW_OBSERVATIONS.jsonl")
        trainer.close()
        return state,receipt,responses,windows

    def test_both_native_and_scaled_replay_exact_without_new_tensor_publication(self):
        if self.subprocess_or_continue():return
        source,export,original=self.make_parent()
        for arm in ("N","S"):
            historical=self.historical(arm,source,export)
            comparator=ReplayComparator(*historical,arm=arm,output_dir=self.root/("compare_"+arm))
            trainer=self.build("replay_"+arm,arm,source,export,comparator=comparator)
            result=trainer.replay_event(self.features)
            self.assertEqual(result["status"],"completed")
            self.assertEqual(result["optimizer_steps"],128)
            self.assertEqual(result["response_presentations"],256)
            self.assertEqual(comparator.receipt["status"],"passed")
            self.assertEqual(comparator.receipt["presentations_compared"],256)
            self.assertEqual(comparator.receipt["responses_compared"],256)
            self.assertEqual(comparator.receipt["windows_compared"],128)
            self.assertFalse(trainer.latest.exists())
            self.assertFalse(trainer.pending_state.exists())
            self.assertFalse((trainer.events_root/"event_005/export").exists())
            self.assertEqual(differences(trainer.origin,original),[])
            trainer.close()
        self.assertFalse(torch.cuda.is_initialized())

    def test_worker_charges_original_startup_and_event_without_tensor_files(self):
        if self.subprocess_or_continue():return
        from experiments.unchanged_c_mathqa_transfer import replay as R
        from experiments.unchanged_c_mathqa_transfer.runtime import initialize_work
        source,export,_=self.make_parent()
        self.historical("N",source,export)
        historical_root=self.root/"historical_N"
        run_root=self.root/"campaign";run_root.mkdir();initialize_work(run_root)
        plan=dict(root=str(run_root),binding="c"*64,pr51_binding=baseline_fixtures.BINDING,
            execution_commit="CPU-fixture",replay_configs={"N":str(self.root/"historical_N.yaml")},
            origin_state=str(source),origin_export=str(export),origin_config=str(self.config),
            retained_endpoints={"N5":dict(endpoint=dict(state=str(historical_root/"training_state.pt")),
                event_receipt_path=str(historical_root/"events/event_005/RESULT.json"))})
        original=R.ControlReplayTrainer
        def cpu_trainer(*args,**kwargs):
            return original(*args,expected_cuda_rng_states=0,**kwargs)
        with patch.object(R,"ControlReplayTrainer",side_effect=cpu_trainer), \
             patch("experiments.unchanged_c_mathqa_transfer.replay_gate.validate_replay_prerequisite",return_value={"feature_paths":[str(path) for path in self.features]}), \
             patch("experiments.prefix_guard.qualify.pin_retained_graph_runtime",return_value={"status":"CPU-fixture"}):
            result=R.run_replay(plan,"N")
        self.assertEqual(result["status"],"passed")
        self.assertEqual(result["comparison"]["event"]["status"],"passed")
        self.assertTrue(result["comparison"]["event"]["full_gate_pending_endpoint"])
        self.assertEqual(result["comparison"]["endpoint"]["status"],"passed")
        work=json.loads((run_root/"WORK.json").read_text())
        self.assertEqual(work["pending"],{})
        for key,expected in dict(adam=128,response_forward=258,response_backward=258,training_initialization=1).items():
            self.assertEqual(work["completed"][key],expected)
            self.assertEqual(work["attempted"][key],expected)
        self.assertEqual(list(run_root.rglob("*.pt")),[])
        self.assertEqual(list(run_root.rglob("*.safetensors")),[])
        self.assertFalse(torch.cuda.is_initialized())

    def test_first_response_mismatch_stops_before_backward_or_adam(self):
        if self.subprocess_or_continue():return
        source,export,_=self.make_parent()
        historical=self.historical("N",source,export)
        comparator=ReplayComparator(*historical,arm="N",output_dir=self.root/"mismatch")
        comparator.responses[0]["loss"]+=1.
        trainer=self.build("replay_mismatch","N",source,export,comparator=comparator)
        with self.assertRaisesRegex(ValueError,"response\\[0\\].loss"):
            trainer.replay_event(self.features)
        self.assertEqual((trainer.global_step,trainer.presentations),(512,1024))
        self.assertEqual(comparator.receipt["status"],"failed")
        self.assertEqual(comparator.receipt["windows_compared"],0)
        self.assertEqual(len(trainer.window_records),0)
        self.assertTrue((trainer.events_root/"event_005/FAILED.json").exists())
        self.assertTrue((self.root/"mismatch/COMPARISON.json").exists())
        self.assertFalse(trainer.latest.exists())
        self.assertFalse(torch.cuda.is_initialized())

    def test_first_presentation_metrics_mismatch_stops_before_following_forward(self):
        if self.subprocess_or_continue():return
        source,export,_=self.make_parent()
        historical=self.historical("S",source,export)
        comparator=ReplayComparator(*historical,arm="S",output_dir=self.root/"presentation_mismatch")
        # The complete reference was valid at construction; simulate a changed
        # live metrics receipt that the online observation must reject early.
        comparator.presentations[0]["metrics"]["loss"]+=1.
        trainer=self.build("replay_presentation_mismatch","S",source,export,comparator=comparator)
        with self.assertRaisesRegex(ValueError,"presentation\\[0\\].metrics.loss"):
            trainer.replay_event(self.features)
        self.assertEqual(comparator.receipt["responses_compared"],1)
        self.assertEqual(comparator.receipt["windows_compared"],0)
        self.assertEqual(trainer.global_step,512)
        self.assertEqual(len(trainer.response_records),1)
        self.assertTrue(trainer.failed)
        self.assertFalse(torch.cuda.is_initialized())


class ReplayEntryTests(unittest.TestCase):
    def test_native_training_and_checkpoint_state_hooks_are_inherited_unchanged(self):
        self.assertIs(ControlReplayTrainer._train_passes,MathQAFixedLRTrainer._train_passes)
        self.assertIs(ControlReplayTrainer._checkpoint_state,MathQAFixedLRTrainer._checkpoint_state)
        self.assertIs(ControlReplayTrainer._build_runtime,MathQAFixedLRTrainer._build_runtime)

    def test_wrong_physical_card_is_rejected_before_creating_a_worker(self):
        from experiments.unchanged_c_mathqa_transfer import replay_entry as E
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"GPU-B"}):
            with self.assertRaisesRegex(ValueError,"physical GPU A"):
                E.run({"devices":{"A":"GPU-A","B":"GPU-B"}},"N")

    def test_bad_feature_gate_stops_before_trainer_initialization(self):
        from experiments.unchanged_c_mathqa_transfer import replay as R
        with patch("experiments.unchanged_c_mathqa_transfer.replay_gate.validate_replay_prerequisite",side_effect=ValueError("changed features")), \
             patch.object(R,"ControlReplayTrainer") as construct:
            with self.assertRaisesRegex(ValueError,"changed features"):R.run_replay({},"N")
            construct.assert_not_called()


if __name__=="__main__":unittest.main()
