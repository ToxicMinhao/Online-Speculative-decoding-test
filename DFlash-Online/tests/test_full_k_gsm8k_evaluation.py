"""Focused CPU fixtures; no model loading, training or generation."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.full_k_gsm8k import evaluation as E


def requests(count):
    return [dict(request_id=f'eval{index}',panel_ordinal=index,original_index=index+20000,
        prompt_ids=[1,2000+index],stop_token_ids=[151645],max_new_tokens=2048) for index in range(count)]


def harmful(record):
    record['metadata'].update(verification_calls=3,accepted_draft_tokens=1,trace=[
        dict(output_start=0,anchor_token=10,committed_progress=1,committed_tokens=[11],accepted_draft_tokens=0),
        dict(output_start=1,anchor_token=11,committed_progress=1,committed_tokens=[12],accepted_draft_tokens=0),
        dict(output_start=2,anchor_token=12,committed_progress=2,committed_tokens=[13,151645],accepted_draft_tokens=1)])


class FullBackend:
    """Small complete native record fixture without importing any trainer."""
    def __init__(self, inputs, mutation=None):
        self.requests,self.mutation=inputs,mutation
        self.checkpoint_id='/F'
        self.executions,self.switches,self.weight_checks=[],[],[]

    def publish(self, checkpoint):
        self.checkpoint_id=checkpoint
        self.switches.append(checkpoint)
        return dict(checkpoint_id=checkpoint,activation_s=.1,checkpoint_structure_valid=True,
            native_status=dict(status='passed',poisoned=None,capture_pending=False,cache_tokens=0,
                attention_backend='fa3',block_size=16,publication=dict(status='passed',
                    checkpoint_path=checkpoint,cache_after_tokens=0,helper_matches_current_layers=True,
                    helper_norm_matches_current_layers=True,fused_enabled=True)))

    def check_weights(self, checkpoint):
        self.weight_checks.append(checkpoint)
        return dict(status='passed',all_equal=True,tensors=58,source_tensor_count=58,native_tensor_count=43,
            checkpoint_path=checkpoint,helper_matches_current_layers=True,
            helper_norm_matches_current_layers=True,fused_enabled=True)

    def generate(self,prompt_ids,**options):
        self.executions.append((self.checkpoint_id,prompt_ids,options))
        request=self.requests[options['request_index']]
        record=dict(record_id=options['request_id'],session_id=options['session_id'],
            request_index=options['request_index'],checkpoint_id=self.checkpoint_id,purpose=options['purpose'],
            prompt_ids=prompt_ids,response_ids=[10,11,12,13,151645],finish_reason='eos',
            stop_token_ids=request['stop_token_ids'],max_new_tokens=2048,
            metadata=dict(verification_calls=2,accepted_draft_tokens=2,committed_progress=4,
                request_latency_s=1.,native_generate_s=.9,feedback_capture_in_timed_request=False,
                block_size=16,context_limit=4096,runtime='pinned_SGLang_FA3',
                cache_policy='idle_full_flush_before_every_request_and_publication',
                trace=[dict(output_start=0,anchor_token=10,committed_progress=2,committed_tokens=[11,12],accepted_draft_tokens=1),
                       dict(output_start=2,anchor_token=12,committed_progress=2,committed_tokens=[13,151645],accepted_draft_tokens=1)],
                capture=dict(collect_features=False,feature_bytes=0,native_capture_validated=True,cache_maintenance_s=.1)))
        if self.mutation:
            self.mutation(record,options)
        for block in record['metadata']['trace']:
            accepted=block['accepted_draft_tokens']
            target=block['committed_tokens']+[0]*(16-len(block['committed_tokens']))
            proposed=target[:15]
            proposed[accepted]+=100
            start=len(prompt_ids)+block['output_start']
            block.update(draft_tokens=proposed,target_argmax=target,verify_size=16,
                positions=list(range(start,start+16)),native_accepted_draft_tokens=block['committed_progress']-1,
                raw_accepted_draft_tokens_before_stop=accepted,stop_clipped=block['committed_progress']<accepted+1)
        capture=record['metadata']['capture']
        capture.update(prompt_ids=prompt_ids,response_ids=record['response_ids'],blocks=deepcopy(record['metadata']['trace']))
        path=Path(options['save_to'])
        native_path=path.with_name(path.stem+'_NATIVE.json')
        record['metadata']['native_trace_path']=str(native_path)
        write_json(native_path,capture)
        write_json(path,record)
        return record


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.inputs = requests(8)
        self.canonicals = {r['request_id']:dict(response_ids=[10,11,12,13,151645],finish_reason='eos',
            source_record_path='/original/'+r['request_id']+'.json') for r in self.inputs}
        for value in self.canonicals.values():
            value.update(F_checkpoint='/F', F_counts=dict(verification_calls=2,
                accepted_draft_tokens=2, committed_progress=4, output_tokens=5))
        self.checkpoints = {'F':'/F','O':'/O58','K':'/K58','K_HALF':'/KH58'}

    def tiny(self):
        return patch.multiple(E, PANEL_SIZE=8, REQUEST_BLOCK=2, INTERMEDIATE_SIZE=4)

    def test_final_budget_and_exact_block_position_precedence_balance(self):
        slots = E.final_schedule(requests(1024))
        balance = E.schedule_balance(slots)
        self.assertEqual(len(slots),12288)
        self.assertEqual(balance['block_repeat_orders'],48)
        self.assertEqual(balance['method_positions'],{m:[12]*4 for m in E.METHODS})
        self.assertEqual(set(balance['pairwise_precedence'].values()),{24})
        self.assertEqual([r['panel_ordinal'] for r in slots[:64]],list(range(64)))
        self.assertEqual([r['method'] for r in slots[:256:64]],list(E.method_order(0,0)))
        self.assertEqual(slots[768]['panel_ordinal'],64)
        intermediate = [s for arm in E.ARMS for event in E.INTERMEDIATE_EVENTS
                        for s in E.intermediate_schedule(requests(1024),arm,event)]
        self.assertEqual(len(intermediate),768)
        self.assertEqual(len(intermediate)+len(slots)+14946+8,28010)

    def test_final_immutable_reference_three_repeats_and_raw_audit(self):
        original = deepcopy(self.canonicals)
        backend = FullBackend(self.inputs)
        with self.tiny():
            result = E.evaluate_final(backend,self.inputs,self.checkpoints,self.root/'final',canonicals=self.canonicals)
            audit = E.audit_stage(self.root/'final',self.inputs,self.checkpoints,
                canonicals=self.canonicals,phase='final')
        self.assertEqual(result['completed_valid_generations'],96)
        self.assertEqual(result['unknown_generation_attempts'],0)
        self.assertEqual(self.canonicals,original)
        self.assertEqual(audit['errors'],[])
        self.assertEqual(len(audit['rows']),96)
        self.assertEqual(len(backend.executions),96)

    def test_intermediate_uses_first_requests_once_without_efficacy_gate(self):
        def mutation(record, options):
            harmful(record)
        backend = FullBackend(self.inputs,mutation)
        with self.tiny():
            result = E.evaluate_intermediate(backend,self.inputs,'/K14',self.root/'mid',
                arm='K',event=14,canonicals=self.canonicals)
            audit = E.audit_stage(self.root/'mid',self.inputs,{'K':'/K14'},
                arm='K',event=14,phase='intermediate',canonicals=self.canonicals)
        self.assertEqual(result['completed_valid_generations'],4)
        self.assertEqual(audit['errors'],[])
        self.assertFalse(audit['latency_comparison'])
        self.assertEqual([r['panel_ordinal'] for r in audit['rows']],list(range(4)))
        self.assertTrue(all(r['verification_calls'] > 2 for r in audit['rows']))

    def test_wrong_output_stops_with_raw_invalid_and_unattempted_records(self):
        def mutation(record, options):
            record['response_ids'][0] += 1
        backend = FullBackend(self.inputs,mutation)
        with self.tiny(),self.assertRaisesRegex(RuntimeError,'correctness'):
            E.evaluate_final(backend,self.inputs,self.checkpoints,self.root/'failed',canonicals=self.canonicals)
        status = json.loads((self.root/'failed/STATUS.json').read_text())
        self.assertEqual((status['attempted_generations'],status['returned_generations'],status['invalid_generations']), (1,1,1))
        self.assertEqual(status['unattempted_generations'],95)
        self.assertTrue((self.root/'failed/0000_r0_F.json').exists())
        self.assertEqual(len((self.root/'failed/MEASUREMENTS.jsonl').read_text().splitlines()),1)

    def test_unknown_attempt_and_no_automatic_retry(self):
        backend = FullBackend(self.inputs)
        with self.tiny(),patch.object(backend,'generate',side_effect=RuntimeError('native failure')):
            with self.assertRaisesRegex(RuntimeError,'native failure'):
                E.evaluate_final(backend,self.inputs,self.checkpoints,self.root/'failed',canonicals=self.canonicals)
            with self.assertRaises(FileExistsError):
                E.evaluate_final(backend,self.inputs,self.checkpoints,self.root/'failed',canonicals=self.canonicals)
        status = json.loads((self.root/'failed/STATUS.json').read_text())
        self.assertEqual(status['unknown_generation_attempts'],1)
        self.assertEqual(status['returned_generations'],0)
        self.assertEqual(status['unattempted_generations'],95)

    def test_report_revalidates_native_proposals_not_saved_valid_flag(self):
        backend = FullBackend(self.inputs)
        with self.tiny():
            E.evaluate_final(backend,self.inputs,self.checkpoints,self.root/'final',canonicals=self.canonicals)
            path = self.root/'final/0000_r0_F.json'
            raw = json.loads(path.read_text())
            raw['metadata']['trace'][0]['draft_tokens'][0] += 101
            path.write_text(json.dumps(raw))
            audit = E.audit_stage(self.root/'final',self.inputs,self.checkpoints,
                canonicals=self.canonicals,phase='final')
        self.assertEqual(audit['status'],'invalid_or_incomplete')
        self.assertTrue(any('0000_r0_F' in error for error in audit['errors']))

    def test_repeat_inconsistency_stops_without_historical_O_count_gate(self):
        def mutation(record, options):
            if options['request_id'] == '0000_r1_O':
                harmful(record)
        backend = FullBackend(self.inputs,mutation)
        with self.tiny(),self.assertRaisesRegex(RuntimeError,'repeat_count_mismatch'):
            E.evaluate_final(backend,self.inputs,self.checkpoints,self.root/'failed',canonicals=self.canonicals)
        self.assertTrue((self.root/'failed/0000_r1_O.json').exists())


if __name__ == '__main__':
    unittest.main()
