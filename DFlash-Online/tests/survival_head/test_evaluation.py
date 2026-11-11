from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments.survival_head import evaluation as E

# Reuse a competent native-record fixture without importing another test suite.
spec = importlib.util.spec_from_file_location('native_fixture',
    Path(__file__).parents[1]/'test_full_k_gsm8k_evaluation.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inputs = fixture.requests(8)
        self.checkpoints = {m:'/'+m for m in E.METHODS}
        self.canonicals = {r['request_id']: dict(prompt_ids=r['prompt_ids'],
            response_ids=[10,11,12,13,151645], finish_reason='eos', verified=True,
            binding_digest='a'*64) for r in self.inputs}

    def run_fixture(self, backend=None):
        backend=backend or fixture.FullBackend(self.inputs)
        folder=self.root/'evaluation'
        result=E.evaluate_stage(backend, self.inputs, self.checkpoints, folder,
            canonicals=self.canonicals, dataset='gsm8k', event=5, production=False)
        audit=E.audit_stage(folder, self.inputs, self.checkpoints,
            canonicals=self.canonicals, dataset='gsm8k', event=5, production=False)
        return folder, result, audit

    def test_schedule_exact768_and_8request_cyclic_blocks_reversed(self):
        slots=E.evaluation_schedule(fixture.requests(128))
        self.assertEqual(len(slots), 768)
        self.assertEqual([s['panel_ordinal'] for s in slots[:8]], list(range(8)))
        first=[]
        for block in range(16):
            orders=[]
            for repeat in range(2):
                order=[s['method'] for s in slots if s['repeat']==repeat and s['panel_ordinal']==block*8]
                orders.append(order)
            self.assertEqual(orders[0], orders[1][::-1]); first.append(tuple(orders[0]))
        self.assertEqual(sorted(Counter(first).values()), [5,5,6])
        balance=E.schedule_balance(slots)
        self.assertEqual(balance['block_repeat_orders'], 32)
        self.assertEqual(set(balance['pair_precedence'].values()), {16})

    def test_full_raw_audit_first_round_and_qualification_inside_budget(self):
        folder, status, audit=self.run_fixture()
        self.assertEqual(status['completed_valid_generations'], 48)
        self.assertEqual(audit['errors'], [])
        first=audit['rows'][0]
        self.assertEqual(first['first_round_output_start'], 0)
        self.assertEqual(first['first_round_accepted_prefix_tokens'], 1)
        self.assertTrue(first['first_round_present'])
        gate=json.loads((folder/'QUALIFICATION.json').read_text())
        self.assertEqual((gate['status'], gate['generations'], gate['extra_generations']), ('passed',48,0))

    def test_first_round_uses_shared_seed_and_not_later_divergent_call(self):
        record=dict(prompt_ids=[1,2], response_ids=[10,11,12], finish_reason='eos', metadata=dict(trace=[
            dict(output_start=0,anchor_token=10,accepted_draft_tokens=1,raw_accepted_draft_tokens_before_stop=2),
            dict(output_start=2,anchor_token=12,accepted_draft_tokens=14,raw_accepted_draft_tokens_before_stop=14)]))
        before=E.first_round(record)
        record['metadata']['trace'][1]['accepted_draft_tokens']=0
        self.assertEqual(E.first_round(record), before)
        record['response_ids'][0]=99; record['metadata']['trace'][0]['anchor_token']=99
        self.assertNotEqual(E.first_round(record)['first_round_start_digest'], before['first_round_start_digest'])

    def test_wrong_output_preserves_invalid_and_unattempted_no_retry(self):
        def mutate(record, options): record['response_ids'][0] += 1
        with self.assertRaises(RuntimeError): self.run_fixture(fixture.FullBackend(self.inputs, mutate))
        status=json.loads((self.root/'evaluation/STATUS.json').read_text())
        self.assertEqual((status['attempted_generations'],status['invalid_generations'],status['unattempted_generations']), (1,1,47))
        self.assertTrue((self.root/'evaluation/0000_r0_KPLUS.json').is_file())

    def test_sidecar_tamper_detected_even_valid_scalar_flags(self):
        folder, _, _=self.run_fixture()
        path=folder/'0000_r0_KPLUS_NATIVE.json'
        value=json.loads(path.read_text()); value['feature_bytes']=999; path.write_text(json.dumps(value))
        audit=E.audit_stage(folder,self.inputs,self.checkpoints,canonicals=self.canonicals,
            dataset='gsm8k',event=5,production=False)
        self.assertIn('0000_r0_KPLUS: native_capture_changed', audit['errors'])

    def test_missing_arm_or_unsealed_schedule_rejected(self):
        with self.assertRaises(ValueError):
            E.validate_evaluation_inputs(self.inputs, {k:v for k,v in self.checkpoints.items() if k!='LOCAL'},
                self.canonicals,dataset='gsm8k',event=5,production=False)
        changed=E.evaluation_schedule(self.inputs); changed.reverse()
        with self.assertRaises(ValueError):
            E.validate_evaluation_inputs(self.inputs,self.checkpoints,self.canonicals,
                dataset='gsm8k',event=5,schedule=changed,production=False)

    def test_unknown_attempt_preserved(self):
        class Failed(fixture.FullBackend):
            def generate(self, *args, **kwargs): raise RuntimeError('fixture interrupted')
        with self.assertRaises(RuntimeError): self.run_fixture(Failed(self.inputs))
        status=json.loads((self.root/'evaluation/STATUS.json').read_text())
        self.assertEqual(status['unknown_generation_attempts'], 1)
        self.assertEqual(status['returned_generations'], 0)
        self.assertEqual(status['unattempted_generations'], 47)


if __name__ == '__main__': unittest.main()
