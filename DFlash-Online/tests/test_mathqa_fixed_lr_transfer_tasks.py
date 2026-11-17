"""CPU task integration with native-record fixtures and no model construction."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.mathqa_fixed_lr_transfer import task_runner as T, runtime as R
from test_full_k_gsm8k_evaluation import FullBackend
from test_mathqa_fixed_lr_transfer_evaluation import panel, origin_references


class Backend(FullBackend):
    loading_s = .01
    worker = SimpleNamespace(pid=12345)

    def prepare_feedback(self, record, path):
        path.write_bytes(b"CPU feature fixture, not a tensor")
        return dict(status="prepared", feature_path=str(path), record_id=record["record_id"])


class TaskTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.plan=dict(root=str(self.root),binding="sealed",execution_commit="a"*40,
                       origin_export="/KPLUS4",origin_state="/KPLUS4/state.pt",
                       origin_weights_identity={"sha256":"origin"},
                       data_panel=str(self.root/"PANEL.json"),warmups=str(self.root/"WARMUPS.json"),
                       references=str(self.root/"CANONICALS.json"),warmup_references=str(self.root/"WARMUP_CANONICALS.json"),
                       storage={"temporary_feature_bytes":100},devices={"A":"GPU-fixed","B":"GPU-other"})
        self.inputs=panel()
        for i,row in enumerate(self.inputs):row["request_index"]=i
        self.warmups=deepcopy(self.inputs[:2])
        for i,row in enumerate(self.warmups):
            row.update(request_id=f"warm{i}",source_id=f"mathqa:test:{9000+i}",prompt_ids=[1,8000+i])
        write_json(self.plan["data_panel"],{"requests":self.inputs})
        write_json(self.plan["warmups"],{"requests":self.warmups})
        for event in (5,6):
            requests=[]
            for i in range(128):
                ordinal=(event-1)*128+i
                requests.append(dict(request_id=f"train_{ordinal:05d}",request_index=ordinal+1,
                    prompt_ids=[1,10000+ordinal],stop_token_ids=[151645],
                    max_new_tokens=2048,source_id=f"mathqa:train:{ordinal}"))
            name=f"event{event}_inputs";self.plan[name]=str(self.root/(name+".json"))
            write_json(self.plan[name],{"requests":requests})
        R.initialize_work(self.root)

    def base(self,task):
        path=self.root/"tasks"/task;path.mkdir(parents=True);return path

    def save_endpoint(self,arm,event):
        endpoint=dict(export=f"/{arm}{event}",state=f"/{arm}{event}/state.pt",
            weights_identity={"sha256":arm+str(event)},state_identity={"sha256":"state"},
            global_step=event*128,presentations=event*256)
        write_json(self.root/"tasks"/f"{arm}{event}"/"RESULT.json",
            dict(status="completed",arm=arm,event=event,binding="sealed",
                 execution_commit="a"*40,endpoint=endpoint))
        return endpoint

    def fake_backend(self,checkpoint,inputs=None):
        backend=Backend(inputs or self.inputs);backend.checkpoint_id=checkpoint;return backend

    def test_full_origin_reference_task_and_actual_consumer(self):
        backend=self.fake_backend("/KPLUS4");base=self.base("references")
        with patch.object(T,"create_backend",return_value=backend),patch.object(T,"close_backend"),patch.object(T,"boundary",return_value={}):
            result=T.references_task(self.plan,base,R.WorkLedger(self.root,"references"),"GPU-fixed")
        self.assertEqual(result["reference_count"],258)
        self.assertEqual(len(backend.executions),258)
        self.assertTrue(all(x[0]=="/KPLUS4" and x[2]["purpose"]=="evaluation" for x in backend.executions))
        self.assertEqual(R.read(self.root/"WORK.json")["completed"]["canonical_generation"],258)
        refs=R.read(self.plan["references"]);warm=R.read(self.plan["warmup_references"])
        from experiments.mathqa_fixed_lr_transfer.evaluation import validate_inputs
        validate_inputs(self.inputs,refs,self.warmups,warm,"/KPLUS4")
        self.assertTrue(all(Path(r["source_record_path"]).exists() for r in [*refs.values(),*warm.values()]))
        self.assertEqual(result["new_teacher_features"],0)

    def test_bad_reference_preserves_returned_record_and_stops(self):
        backend=self.fake_backend("/WRONG");base=self.base("references")
        with patch.object(T,"create_backend",return_value=backend),patch.object(T,"close_backend"),patch.object(T,"boundary",return_value={}):
            with self.assertRaisesRegex(ValueError,"identity"):
                T.references_task(self.plan,base,R.WorkLedger(self.root,"references"),"GPU-fixed")
        self.assertEqual(len(backend.executions),1)
        self.assertTrue((base/"records"/"eval0.json").exists())
        self.assertEqual(R.read(self.root/"WORK.json")["completed"]["canonical_generation"],1)
        self.assertFalse(Path(self.plan["references"]).exists())

    def collect(self,task,checkpoint):
        event=5 if task=="common5" else 6
        inputs=R.read(self.plan[f"event{event}_inputs"])["requests"]
        backend=self.fake_backend(checkpoint,{r["request_index"]:r for r in inputs})
        original=backend.generate
        order=[]
        def generate(*args,**kwargs):
            result=original(*args,**kwargs)
            result["metadata"]["feedback_capture_in_timed_request"]=True
            write_json(kwargs["save_to"],result)
            return result
        backend.generate=generate
        def live(record,prepared,*args):
            self.assertTrue(Path(prepared["feature_path"]).exists());order.append((record["record_id"],"live"))
        def retire(record,prepared,*args):
            self.assertEqual(order[-1],(record["record_id"],"live"));order.append((record["record_id"],"retire"))
        base=self.base(task)
        with patch.object(T,"create_backend",return_value=backend),patch.object(T,"close_backend"),patch.object(T,"boundary",return_value={}),patch.object(T,"check_retirement_live_use",side_effect=live),patch("experiments.kplus_full_gsm8k.storage.retire_redundant_features",side_effect=retire):
            result=T.collect_task(self.plan,task,base,R.WorkLedger(self.root,task),"GPU-fixed")
        self.assertEqual(len(order),256)
        self.assertEqual(result["generated_responses"],128)
        self.assertEqual(result["teacher_preparations"],128)
        return result,backend

    def test_common_e5_is_same_pool_for_both_arms_and_retirement_follows_preparation(self):
        result,backend=self.collect("common5","/KPLUS4")
        self.assertEqual(T.feature_pool(self.plan,"N",5),T.feature_pool(self.plan,"S",5))
        self.assertEqual(result["own_checkpoint"],"/KPLUS4")
        self.assertEqual(len(backend.executions),128)
        work=R.read(self.root/"WORK.json")
        self.assertEqual(work["completed"]["feedback_generation"],128)
        self.assertEqual(work["completed"]["teacher_preparation"],128)

    def test_unprepared_packet_stops_before_any_retirement(self):
        inputs=R.read(self.plan["event5_inputs"])["requests"]
        backend=self.fake_backend("/KPLUS4",{r["request_index"]:r for r in inputs})
        base=self.base("common5")
        with patch.object(T,"create_backend",return_value=backend),patch.object(T,"close_backend"),patch.object(T,"boundary",return_value={}),patch("experiments.full_k_gsm8k.campaign.online_record_check"),patch.object(backend,"prepare_feedback",return_value={"status":"skipped"}),patch.object(T,"check_retirement_live_use") as live,patch("experiments.kplus_full_gsm8k.storage.retire_redundant_features") as retire:
            with self.assertRaisesRegex(ValueError,"eligible"):
                T.collect_task(self.plan,"common5",base,R.WorkLedger(self.root,"common5"),"GPU-fixed")
        live.assert_not_called();retire.assert_not_called()
        self.assertEqual(R.read(base/"FEATURES.json")["status"],"failed")
        self.assertEqual(len(backend.executions),1)

    def test_retirement_rejects_nonowned_task_before_process_inspection(self):
        base=self.root/"tasks/older_unique_features";base.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,"owned collection"):
            T.check_retirement_live_use({"record_id":"train_00512"},{},self.root,base,12345)
        receipt=R.read(base/"preparation_retirement/train_00512.LIVE_USE.json")
        self.assertEqual(receipt["status"],"failed")
        self.assertEqual(receipt["targets"],[])

    def test_e6_must_be_own_arm_features_and_identity(self):
        self.save_endpoint("N",5);self.save_endpoint("S",5)
        result,backend=self.collect("S_collect6","/S5")
        self.assertEqual(len(T.feature_pool(self.plan,"S",6)),128)
        self.assertTrue(all(x[0]=="/S5" for x in backend.executions))
        result["own_checkpoint"]="/N5";result["own_weights_identity"]={"sha256":"N5"}
        write_json(self.root/"tasks/S_collect6/FEATURES.json",result)
        with self.assertRaisesRegex(ValueError,"arm-own"):
            T.feature_pool(self.plan,"S",6)

    def test_incomplete_endpoint_or_wrong_counters_rejected_before_backend(self):
        endpoint=self.save_endpoint("N",5);endpoint["global_step"]=639
        path=self.root/"tasks/N5/RESULT.json";receipt=R.read(path);receipt["endpoint"]=endpoint;write_json(path,receipt)
        with self.assertRaisesRegex(ValueError,"cumulative"):T.endpoint(self.plan,"N",5)

    def seal_fixture_references(self):
        from experiments.bounded_prefix_allocation.prepare import identity
        write_json(self.plan["references"],origin_references(self.inputs))
        write_json(self.plan["warmup_references"],origin_references(self.warmups))
        write_json(self.root/"references/SEAL.json",dict(status="passed",binding=self.plan["binding"],
            execution_commit=self.plan["execution_commit"],checkpoint=self.plan["origin_export"],
            checkpoint_identity=self.plan["origin_weights_identity"],reference_count=258,
            files={self.plan[k]:identity(self.plan[k]) for k in ("references","warmup_references")}))

    def test_reference_file_mutation_stops_before_backend(self):
        self.save_endpoint("N",6);self.save_endpoint("S",6);self.seal_fixture_references()
        references=R.read(self.plan["references"]);references["eval0"]["response_ids"][0]+=1
        write_json(self.plan["references"],references)
        with patch.object(T,"create_backend") as backend:
            with self.assertRaisesRegex(ValueError,"changed after sealing"):
                T.final_task(self.plan,self.base("final"),R.WorkLedger(self.root,"final"),"GPU-fixed")
        backend.assert_not_called()
        self.assertEqual(R.read(self.root/"WORK.json")["attempted"]["scored_generation"],0)

    def test_full_final_task_charges_real_two_arm_schedule(self):
        self.save_endpoint("N",6);self.save_endpoint("S",6)
        self.seal_fixture_references()
        backend=self.fake_backend("/N6");base=self.base("final")
        with patch.object(T,"create_backend",return_value=backend),patch.object(T,"close_backend"),patch.object(T,"boundary",return_value={}),patch("experiments.kplus_full_gsm8k.preflight.timing_barrier"),patch("experiments.mathqa_fixed_lr_transfer.analysis.write_report",return_value={"status":"completed"}):
            result=T.final_task(self.plan,base,R.WorkLedger(self.root,"final"),"GPU-fixed")
        work=R.read(self.root/"WORK.json")
        self.assertEqual((work["completed"]["scored_generation"],work["completed"]["warmup_generation"]),(1536,48))
        self.assertEqual(len(backend.executions),1584)
        self.assertEqual(result["evaluation"]["completed_valid_generations"],1584)
        self.assertEqual(set(backend.switches),{"/N6","/S6"})
        self.assertEqual(len(backend.switches),24)
        self.assertEqual(work["pending"],{})

    def test_failed_work_stays_reserved_and_caps_cannot_be_reset(self):
        ledger=R.WorkLedger(self.root,"fixture")
        with self.assertRaisesRegex(RuntimeError,"fixture"):
            ledger.call({"adam":1},"bad",lambda:(_ for _ in ()).throw(RuntimeError("fixture")))
        work=R.read(self.root/"WORK.json")
        self.assertEqual((work["attempted"]["adam"],work["completed"]["adam"]),(1,0))
        self.assertEqual(len(work["pending"]),1)
        with self.assertRaises(FileExistsError):R.initialize_work(self.root)
        with self.assertRaisesRegex(RuntimeError,"cap"):
            ledger.call({"adam":528},"over",lambda:None)

if __name__=="__main__":unittest.main()
