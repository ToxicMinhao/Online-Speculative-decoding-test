"""CPU checks for the genuine three-arm schedule, raw validity and fixed decision."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.unchanged_c_mathqa_transfer import analysis as A, evaluation as E
from test_full_k_gsm8k_evaluation import FullBackend, requests, harmful
from test_persistent_prefix_evaluation import references, quiet


def panel():
    result=requests(256)
    for i,row in enumerate(result):
        row['source_id']=f'mathqa:validation:{i}'
    return result


def origin_references(rows):
    result = {k:dict(v,reference_checkpoint='/KPLUS4') for k,v in references(rows).items()}
    for reference in result.values():
        payload = {k:v for k,v in reference.items() if k != 'binding_digest'}
        reference['binding_digest'] = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


class ScheduleTests(unittest.TestCase):
    def test_exact_permutations_budget_and_position_balance(self):
        schedule = E.evaluation_schedule(panel())
        self.assertEqual(len(schedule), 36)
        self.assertEqual(sum(len(row['request_ordinals']) for row in schedule), 2304)
        self.assertEqual(2 * len(schedule), 72)
        permutations = [('N','C','S'), ('N','S','C'), ('C','N','S'),
                        ('C','S','N'), ('S','N','C'), ('S','C','N')]
        observed = []
        for start in range(0, 36, 3):
            block = schedule[start:start+3]
            repeat, ordinal = block[0]['repeat'], block[0]['block_ordinal']
            order = tuple(row['method'] for row in block)
            self.assertEqual(order, permutations[(4 * repeat + ordinal) % 6])
            self.assertTrue(all(row['request_ordinals'] == list(range(64*ordinal,64*(ordinal+1))) for row in block))
            observed.append(order)
        self.assertEqual(Counter(observed), Counter({permutation:2 for permutation in permutations}))
        self.assertEqual(Counter((row['method'],row['position']) for row in schedule),
                         Counter({(method,position):4 for method in E.METHODS for position in range(3)}))

    def test_reference_must_be_separate_origin_and_warmups_disjoint(self):
        inputs=panel();warmups=deepcopy(inputs[:2])
        for i,row in enumerate(warmups):row.update(request_id=f'warm{i}',source_id=f'warm:{i}',prompt_ids=[1,8000+i])
        refs=origin_references(inputs);warmrefs=origin_references(warmups)
        E.validate_inputs(inputs,refs,warmups,warmrefs,'/KPLUS4')
        refs['eval0']['reference_checkpoint']='/S'
        with self.assertRaisesRegex(ValueError,'KPLUS4'):
            E.validate_inputs(inputs,refs,warmups,warmrefs,'/KPLUS4')
        refs=origin_references(inputs);warmups[0]['source_id']=inputs[0]['source_id']
        with self.assertRaisesRegex(ValueError,'separate fixed warmups'):
            E.validate_inputs(inputs,refs,warmups,warmrefs,'/KPLUS4')

    def test_canonical_contents_must_match_binding_digest(self):
        inputs=panel();refs=origin_references(inputs)
        refs['eval0']['source_record_path']='/substituted.json'
        with self.assertRaisesRegex(ValueError,'binding digest'):
            E.validate_references(inputs,refs,origin_checkpoint='/KPLUS4')

    def test_new_wrapper_does_not_change_prior_two_arm_contract(self):
        from experiments.mathqa_fixed_lr_transfer import evaluation as prior
        self.assertEqual(prior.METHODS, ('N', 'S'))
        self.assertEqual((prior.SCORED, prior.WARMUPS, prior.TOTAL), (1536, 48, 1584))
        self.assertEqual(len(prior.evaluation_schedule(panel())), 24)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.inputs=panel();self.refs=origin_references(self.inputs)
        self.warmups=deepcopy(self.inputs[:2])
        for i,row in enumerate(self.warmups):row.update(request_id=f'warm{i}',source_id=f'warm:{i}',prompt_ids=[1,8000+i])
        self.warmrefs=origin_references(self.warmups)

    def run_final(self,backend,folder):
        return E.evaluate_final(backend,self.inputs,{'N':'/N6','C':'/C6','S':'/S6'},self.root/folder,
            canonicals=self.refs,warmup_requests=self.warmups,warmup_canonicals=self.warmrefs,
            schedule=E.evaluation_schedule(self.inputs),origin_checkpoint='/KPLUS4',
            physical_gpu_uuid='GPU-fixed',boundary_check=quiet,execution_commit='c'*40)

    def test_complete_native_consumer_and_machine_tables(self):
        backend=FullBackend(self.inputs)
        status=self.run_final(backend,'complete')
        self.assertEqual((status['scored_generations'],status['warmup_generations']),(2304,72))
        self.assertEqual(len(backend.switches),36)
        self.assertEqual(backend.weight_checks,['/N6','/C6','/S6'])
        self.assertTrue(all(x[2]['session_id']==E.SESSION for x in backend.executions))
        root=self.root/'complete'
        result=A.write_report(root)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(set(result['method_totals']),{'N','C','S'})
        self.assertEqual(result['method_totals']['N']['verification_calls'],512)
        self.assertEqual(result['decision']['outcome'],'PRACTICAL_TIME_GAIN_NOT_SUPPORTED')
        self.assertEqual(len((root/'ALL_REPEATS.csv').read_text().splitlines()),2305)
        self.assertEqual(len((root/'WARMUPS.csv').read_text().splitlines()),73)
        self.assertEqual(len((root/'PER_REQUEST.csv').read_text().splitlines()),769)
        self.assertEqual(len((root/'TRACE_COUNTS.csv').read_text().splitlines()),4753)
        costs=json.loads((root/'EVALUATION_COSTS.json').read_text())
        self.assertEqual(costs['measured_time_by_phase']['scored']['request_latency_s'],2304)
        self.assertEqual(costs['measured_time_by_phase']['warmup']['request_latency_s'],72)
        protocol_path=root/'PROTOCOL.json'
        protocol=json.loads(protocol_path.read_text())
        protocol_path.write_text(json.dumps(dict(protocol,execution_commit='unverified')))
        self.assertIn('execution_identity',E.audit_raw_records(root)['errors'])
        protocol_path.write_text(json.dumps(protocol))
        # Independent native capture comparison rejects changed raw proposals.
        rawpath=root/'0000_r0_N.json';raw=json.loads(rawpath.read_text())
        raw['metadata']['trace'][0]['draft_tokens'][0]+=1
        rawpath.write_text(json.dumps(raw))
        self.assertEqual(A.write_report(root)['status'],'INVALID_OR_INCOMPLETE')

    def test_invalid_return_saved_before_failure(self):
        def mutation(record,options):record['response_ids'][0]+=1
        with self.assertRaisesRegex(RuntimeError,'correctness'):
            self.run_final(FullBackend(self.inputs,mutation),'invalid')
        root=self.root/'invalid';status=json.loads((root/'STATUS.json').read_text())
        self.assertEqual((status['invalid_generations'],status['unknown_generation_attempts'],status['unattempted_generations']),(1,0,2375))
        self.assertTrue((root/'warmup_a00_0_N.json').is_file())
        self.assertEqual(A.write_report(root)['status'],'INVALID_OR_INCOMPLETE')

    def test_unknown_attempt_not_retried_or_counted_unattempted(self):
        backend=FullBackend(self.inputs)
        with patch.object(backend,'generate',side_effect=RuntimeError('worker lost')):
            with self.assertRaisesRegex(RuntimeError,'worker lost'):self.run_final(backend,'unknown')
        status=json.loads((self.root/'unknown/STATUS.json').read_text())
        self.assertEqual((status['returned_generations'],status['unknown_generation_attempts'],status['unattempted_generations']),(0,1,2375))

    def test_missing_c_checkpoint_is_rejected_before_execution(self):
        backend = FullBackend(self.inputs)
        with self.assertRaisesRegex(ValueError, 'three distinct'):
            E.evaluate_final(backend, self.inputs, {'N':'/N6','S':'/S6'}, self.root/'two_arm',
                canonicals=self.refs, warmup_requests=self.warmups, warmup_canonicals=self.warmrefs,
                schedule=E.evaluation_schedule(self.inputs), origin_checkpoint='/KPLUS4',
                physical_gpu_uuid='GPU-fixed', boundary_check=quiet, execution_commit='c'*40)
        self.assertEqual(backend.executions, [])
        self.assertEqual(backend.switches, [])
        self.assertFalse((self.root/'two_arm').exists())

    def test_repeat_count_mismatch_stops_even_with_identical_tokens(self):
        def mutation(record,options):
            if options['request_id']=='0000_r1_N':harmful(record)
        with self.assertRaisesRegex(RuntimeError,'repeat'):
            self.run_final(FullBackend(self.inputs,mutation),'repeat')
        rows=[json.loads(s) for s in (self.root/'repeat/MEASUREMENTS.jsonl').read_text().splitlines()]
        self.assertFalse(rows[-1]['valid_measurement'])
        self.assertEqual(rows[-1]['label'],'0000_r1_N')

    def test_zero_call_eos_valid_and_no_historical_count_gate(self):
        def mutation(record,options):
            record['response_ids']=[151645]
            record['metadata'].update(trace=[],verification_calls=0,accepted_draft_tokens=0,committed_progress=0)
        backend=FullBackend(self.inputs,mutation)
        slot=dict(phase='scored',method='N',repeat=0,event=6,panel_ordinal=0,activation=0,block_ordinal=0,position=0,request_id='eval0')
        record=backend.generate(self.inputs[0]['prompt_ids'],request_id=E.label_for(slot),session_id=E.SESSION,
            request_index=0,purpose='evaluation',save_to=self.root/'zero.json')
        ref=dict(self.refs['eval0'],response_ids=[151645],F_counts={'verification_calls':999})
        row,errors=E.measure(record,self.inputs[0],'/F',ref,slot,'GPU-fixed')
        self.assertEqual(errors,[])
        self.assertEqual(row['native_accepted_draft_tokens'],0)


if __name__ == '__main__':
    unittest.main()
