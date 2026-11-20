"""CPU-only ledger, canonical integration and bounded-run failure regressions."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json, unittest
from unittest.mock import patch

from experiments.fresh_mathqa_endpoints import runtime as R, entry as E, preflight as P
from experiments.fresh_mathqa_endpoints import evaluation as V
from online_feedback.storage import write_json
from test_full_k_gsm8k_evaluation import FullBackend, requests


class CanonicalBackend(FullBackend):
    def generate(self, *args, **kwargs):
        value=super().generate(*args, **kwargs)
        capture=value['metadata']['capture']
        capture.update(status='passed',label_binding_scope='native_prefill_and_committed_verification',
            target_argmax_ids=value['response_ids'],
            covered_token_ids=value['prompt_ids']+value['response_ids'][:-1],
            covered_rows=len(value['prompt_ids'])+len(value['response_ids'])-1)
        return value


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        R.initialize(self.root)

    def test_failed_generation_remains_reserved_and_cannot_reset(self):
        ledger=R.WorkLedger(self.root,'qualification')
        def fail():raise RuntimeError('lost generation')
        with self.assertRaisesRegex(RuntimeError,'lost generation'):
            ledger.call('qualification_generation','first',fail)
        work=R.read(self.root/'WORK.json')
        self.assertEqual(work['attempted']['qualification_generation'],1)
        self.assertEqual(work['completed']['qualification_generation'],0)
        self.assertEqual(len(work['pending']),1)
        with self.assertRaises(FileExistsError):R.initialize(self.root)

    def test_every_attempt_counts_against_original_limit(self):
        ledger=R.WorkLedger(self.root,'qualification')
        for i in range(24):self.assertEqual(ledger.call('qualification_generation',str(i),lambda:7),7)
        with self.assertRaisesRegex(RuntimeError,'budget'):ledger.call('qualification_generation','25',lambda:7)
        self.assertEqual(R.read(self.root/'WORK.json')['completed']['qualification_generation'],24)

    def test_storage_rejects_oversized_peak_and_reserves_remaining_writes(self):
        plan=dict(root=str(self.root),allocation_roots=[str(self.root)],projected_peak_bytes=30_000_000_001)
        with self.assertRaises(ValueError):R.storage(plan)
        plan.update(projected_peak_bytes=20_000_000_000,compiler_baseline=dict(directories=['/cache'],allocated_bytes=0))
        with patch.object(R,'allocated_bytes',return_value=5_000_000_000), patch.object(R,'compiler_snapshot',return_value=dict(directories=['/cache'],allocated_bytes=0)), patch.object(R.shutil,'disk_usage',return_value=SimpleNamespace(free=410_000_000_000)):
            with self.assertRaisesRegex(RuntimeError,'Storage'):R.storage(plan,complete_forecast=True)
            self.assertEqual(R.storage(plan)['required_free_bytes'],400_000_000_000)

    def test_forecast_requires_finite_nonnegative_remaining_work(self):
        with patch.object(R,'clock_status',return_value=dict(elapsed_s=100,occupied_s=100,active={})):
            for bad in (-1,float('inf'),float('nan'),True):
                with self.assertRaises(ValueError):R.forecast({'root':str(self.root)},bad,100)
            with self.assertRaisesRegex(RuntimeError,'no longer fits'):R.forecast({'root':str(self.root)},30000,100)
            self.assertEqual(R.forecast({'root':str(self.root)},1000,2000)['conservative_elapsed_s'],2200)

    def test_schedule_reads_original_sealed_container(self):
        p=self.root/'schedule.json';write_json(p,dict(activations=[dict(activation=0)]))
        self.assertEqual(R.schedule_rows(p),[dict(activation=0)])
        write_json(p,dict(wrong=[]))
        with self.assertRaises(ValueError):R.schedule_rows(p)

    def canonical_plan(self):
        rows=requests(4)
        for i,r in enumerate(rows):r.update(source_id=f'mathqa:test:{i}',dataset='mathqa')
        panel=self.root/'panel.json';write_json(panel,dict(requests=rows))
        plan=dict(root=str(self.root),panel=str(panel),checkpoints={m:'/'+m for m in V.METHODS},
            session=V.dataset_contract('mathqa')['session'],checkpoint_identities={'O58':dict(bytes=123,sha256='a'*64)},
            canonical_provenance={k:dict(sha256=str(i)*64) for i,k in enumerate(('target_identity','tokenizer_identity','runtime_identity'),1)})
        return rows,plan

    def test_canonical_shards_pass_real_native_record_and_bound_reference_validation(self):
        rows,plan=self.canonical_plan();all_refs={}
        with patch.object(E,'boundary',return_value={}):
            for slot in ('A','B'):
                base=self.root/slot;base.mkdir()
                result=E.canonical_shard(CanonicalBackend(rows),plan,base,R.WorkLedger(self.root,slot),slot)
                self.assertEqual(result['complete_generations'],2)
                refs=R.read(base/'CANONICALS.json');self.assertTrue(set(all_refs).isdisjoint(refs));all_refs.update(refs)
        V.validate_references(rows,all_refs,reference_checkpoint='/O58',required_provenance=True,
            producer_weight_identity=plan['checkpoint_identities']['O58'],canonical_provenance=plan['canonical_provenance'])
        for request in rows:
            ref=all_refs[request['request_id']]
            for key in ('request_id','source_id','max_new_tokens','stop_token_ids'):
                self.assertEqual(ref[key],request[key])
        self.assertEqual(R.read(self.root/'WORK.json')['completed']['canonical_generation'],4)

    def test_wrong_canonical_record_identity_stops_with_raw_evidence(self):
        rows,plan=self.canonical_plan();base=self.root/'canonical';base.mkdir()
        backend=CanonicalBackend(rows,mutation=lambda r,o:r.update(session_id='wrong'))
        with patch.object(E,'boundary',return_value={}):
            with self.assertRaisesRegex(ValueError,'canonical_generation_identity'):
                E.canonical_shard(backend,plan,base,R.WorkLedger(self.root,'canonical'),'A')
        self.assertEqual(len(list(base.glob('canonical_eval0.json'))),1)
        self.assertEqual(R.read(self.root/'WORK.json')['attempted']['canonical_generation'],1)
        self.assertEqual(R.read(self.root/'WORK.json')['completed']['canonical_generation'],1)
        self.assertFalse((base/'CANONICALS.json').exists())

    def test_failure_closes_backend_before_releasing_card_lease(self):
        events=[]
        @contextmanager
        def lease(*args):
            events.append('allocated')
            try:yield
            finally:events.append('released')
        backend=SimpleNamespace(loading_s=1.)
        plan=dict(root=str(self.root),execution_commit='a'*40,devices={'A':'GPU-A'},checkpoints={'O58':'/O58'})
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'GPU-A'}), patch.object(E,'occupied_card',lease), patch.object(E,'boundary',return_value={}), patch.object(E,'create_backend',return_value=backend), patch.object(E,'qualify',side_effect=ValueError('output mismatch')), patch.object(E,'close_backend',side_effect=lambda *a:events.append('closed')):
            with self.assertRaisesRegex(ValueError,'output mismatch'):E.run(plan,'qualification','A')
        self.assertEqual(events,['allocated','closed','released'])
        self.assertEqual(R.read(self.root/'phases/qualification_A/STATUS.json')['status'],'failed')

    def test_pool_status_cannot_bypass_missing_manifest_gate(self):
        seal=dict(status='sealed_before_new_model_work',scored=1024,warmup=2,new_model_work=0,
            files={n:{} for n in ('PANEL.json','SCHEDULE.json','WARMUPS.json','WARMUP_CANONICALS.json')},input_audit=dict(identity={}))
        pool=dict(status='passed_pool_availability',counts=dict(source_rows=2985,total_excluded=1799,eligible_rows=1186,
            required_scored=1024,known_used_or_reserved_rows=1799,duplicates={},input_ineligible={}),
            historical_coverage=dict(missing_essential_manifests=['missing-old-inputs'],unknown_later_roots=[],historic_receipts_reauthenticated=True))
        plan={k:k for k in ('panel_seal','panel','schedule','warmups','warmup_canonicals','pool_audit')}
        with patch.object(P,'identity',return_value={}), patch.object(P,'read',side_effect=[seal,pool]):
            with self.assertRaisesRegex(ValueError,'coverage/count'):P.verify_panel(plan)

    def test_postflight_checks_all_three_exports_and_preserves_changed_identity(self):
        checkpoints={};identities={}
        for method in V.METHODS:
            folder=self.root/method;folder.mkdir();path=folder/'model.safetensors'
            path.write_bytes(method.encode());checkpoints[method]=str(folder);identities[method]=P.identity(path)
        plan=dict(root=str(self.root),checkpoints=checkpoints,checkpoint_identities=identities)
        self.assertEqual(P.postflight_exports(plan)['status'],'passed')
        (Path(checkpoints['AREPO58'])/'model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'postflight identity'):P.postflight_exports(plan)
        receipt=R.read(self.root/'POSTFLIGHT_EXPORT_IDENTITIES.json')
        self.assertEqual(receipt['status'],'failed')
        self.assertEqual(set(receipt['exports']),set(V.METHODS))
        self.assertFalse(receipt['exports']['AREPO58']['unchanged'])
        self.assertTrue(receipt['exports']['O58']['unchanged'])
        self.assertTrue(receipt['exports']['KPLUS_C58']['unchanged'])


if __name__=='__main__':unittest.main()
