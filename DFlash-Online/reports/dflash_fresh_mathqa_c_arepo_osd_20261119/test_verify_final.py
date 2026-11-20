"""Independent auditor's numerical and malformed-evidence CPU fixtures."""
import hashlib,json
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import verify_final as V


def rows_fixture():
    requests=[dict(request_id='r10',source_id='mathqa:test:1',panel_ordinal=0),dict(request_id='r2',source_id='mathqa:test:2',panel_ordinal=1)]
    times={'O58':[3.,20.],'AREPO58':[2.,10.],'KPLUS_C58':[1.,8.]}
    calls={'O58':[3,9],'AREPO58':[2,8],'KPLUS_C58':[1,7]}
    rows=[]
    for method in V.ARMS:
        for r in requests:
            i=r['panel_ordinal'];c=calls[method][i]
            for repeat in range(3):
                rows.append(dict(r,method=method,repeat=repeat,verification_calls=c,native_accepted_draft_tokens=15-c,accepted_draft_tokens=16-c,committed_progress=15,output_tokens=16,request_latency_s=times[method][i]*(repeat+1),native_generate_s=times[method][i]*(repeat+1)-.1,cache_maintenance_s=.1,finish_reason='eos',cap_hit=False,checkpoint_id='/'+method,physical_gpu_uuid='GPU-one'))
    return requests,rows


def raw_fixture():
    request=dict(request_id='fixed',source_id='mathqa:test:9',panel_ordinal=0,prompt_ids=[1,2],stop_token_ids=[151645,151643],max_new_tokens=2048)
    response=[10,151645];target=[151645]+list(range(20,35));draft=target[:15];draft[3]+=100
    block=dict(output_start=0,anchor_token=10,committed_progress=1,committed_tokens=[151645],accepted_draft_tokens=1,native_accepted_draft_tokens=0,raw_accepted_draft_tokens_before_stop=3,stop_clipped=True,draft_tokens=draft,target_argmax=target,verify_size=16,positions=list(range(2,18)))
    capture=dict(status='passed',collect_features=False,feature_bytes=0,native_capture_validated=True,prompt_ids=[1,2],response_ids=response,blocks=[block],cache_maintenance_s=.1,
        label_binding_scope='native_prefill_and_committed_verification',target_argmax_ids=response,
        covered_token_ids=[1,2,10],covered_rows=3)
    record=dict(record_id='label',session_id=V.SESSION,request_index=0,checkpoint_id='/O58',purpose='evaluation',prompt_ids=[1,2],response_ids=response,finish_reason='eos',stop_token_ids=[151645,151643],max_new_tokens=2048,metadata=dict(block_size=16,context_limit=4096,runtime='pinned_SGLang_FA3',cache_policy='idle_full_flush_before_every_request_and_publication',feedback_capture_in_timed_request=False,trace=[block],capture=capture,verification_calls=1,accepted_draft_tokens=1,committed_progress=1,request_latency_s=1.,native_generate_s=.8))
    reference=dict(response_ids=response,finish_reason='eos')
    return request,reference,record


class AuditTests(unittest.TestCase):
    def test_averages_repeats_within_requests_and_ratio_of_sums(self):
        requests,rows=rows_fixture();paired=V.reduce_rows(rows,requests)
        self.assertEqual([r['request_id'] for r in paired],['r10','r2'])
        self.assertEqual(paired[0]['KPLUS_C58']['request_latency_s'],2.)
        result=V.statistics(paired,draws=64,chunk=7)
        effect=result['comparisons']['KPLUS_C58_vs_AREPO58']
        self.assertAlmostEqual(effect['time_reduction']['point'],.25)
        self.assertNotAlmostEqual(effect['time_reduction']['point'],.35)
        self.assertEqual(effect['net_calls_saved'],2)
        self.assertEqual(effect['requests_fewer_calls'],2)
        self.assertNotEqual(effect['clipped_acceptance']['candidate_tokens_per_call'],result['method_totals']['KPLUS_C58']['native_accepted_tokens_per_call'])

    def test_chunked_rng_draw_hash_and_percentiles_match_literal_draws(self):
        requests,rows=rows_fixture();paired=V.reduce_rows(rows,requests)
        result=V.statistics(paired,draws=127,chunk=11)
        ix=np.random.Generator(np.random.PCG64(20261119)).integers(0,2,size=(127,2))
        self.assertEqual(result['bootstrap']['draw_matrix_sha256'],hashlib.sha256(ix.astype('<i8').tobytes()).hexdigest())
        candidates=np.array([2.,16.]);references=np.array([4.,20.])
        literal=1-candidates[ix].sum(1)/references[ix].sum(1)
        np.testing.assert_allclose(result['comparisons']['KPLUS_C58_vs_AREPO58']['time_reduction']['ci95'],np.quantile(literal,[.025,.975],method='linear'),rtol=0,atol=0)
        self.assertEqual(V.statistics(paired,draws=127,chunk=127)['bootstrap'],result['bootstrap'])

    def test_zero_native_denominator_stays_undefined_not_success(self):
        requests,rows=rows_fixture()
        for row in rows:
            row.update(verification_calls=15,native_accepted_draft_tokens=0,accepted_draft_tokens=0)
        result=V.statistics(V.reduce_rows(rows,requests),draws=31)
        effect=result['comparisons']['KPLUS_C58_vs_AREPO58']['acceptance_improvement']
        self.assertEqual(effect,dict(point=None,ci95=None,undefined_draws=31))
        self.assertEqual(V.evidence(effect),'DIRECTION_UNRESOLVED')

    def test_default_bootstrap_has_exact_prespecified_draws_seed_and_order(self):
        requests,rows=rows_fixture();paired=V.reduce_rows(rows,requests)
        result=V.statistics(paired)
        indices=np.random.Generator(np.random.PCG64(20261119)).integers(0,2,size=(10000,2))
        self.assertEqual(result['bootstrap']['draws'],10000)
        self.assertEqual(result['bootstrap']['seed'],20261119)
        self.assertEqual(result['bootstrap']['draw_matrix_sha256'],hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest())
        with self.assertRaisesRegex(V.AuditFailure,'lexical request order'):V.statistics(list(reversed(paired)))

    def test_missing_duplicate_or_inconsistent_repeats_rejected(self):
        requests,rows=rows_fixture()
        with self.assertRaises(V.AuditFailure):V.reduce_rows(rows[:-1],requests)
        corrupted=deepcopy(rows);corrupted[1]['repeat']=0
        with self.assertRaisesRegex(V.AuditFailure,'duplicate timing'):V.reduce_rows(corrupted,requests)
        corrupted=deepcopy(rows);corrupted[1]['verification_calls']+=1
        with self.assertRaisesRegex(V.AuditFailure,'count identity'):V.reduce_rows(corrupted,requests)

    def test_native_and_stop_clipped_acceptance_are_distinct(self):
        request,reference,record=raw_fixture()
        result=V.trace_record(record,request,reference,'/O58','label',V.SESSION)
        self.assertEqual(result['native_accepted_draft_tokens'],0)
        self.assertEqual(result['accepted_draft_tokens'],1)
        self.assertEqual(result['verification_calls'],1)

    def test_tampered_canonical_checkpoint_termination_and_native_trace_rejected(self):
        request,reference,record=raw_fixture()
        for mutate in [lambda r:r.update(checkpoint_id='/wrong'),lambda r:r.update(finish_reason='length'),lambda r:r['metadata']['trace'][0].update(committed_tokens=[999]),lambda r:r.update(session_id='wrong'),lambda r:r['metadata']['capture'].update(covered_rows=4),lambda r:r['metadata']['capture'].update(target_argmax_ids=[999])]:
            bad=deepcopy(record);mutate(bad)
            with self.assertRaises(V.AuditFailure):V.trace_record(bad,request,reference,'/O58','label',V.SESSION)
        with self.assertRaisesRegex(V.AuditFailure,'canonical output'):
            V.trace_record(record,request,dict(response_ids=[10,20,151645],finish_reason='eos'),'/O58','label',V.SESSION)

    def test_sign_boundaries_do_not_claim_equivalence(self):
        for bounds,label in [([0,.1],'DIRECTION_UNRESOLVED'),([-.1,0],'DIRECTION_UNRESOLVED'),([.001,.1],'SUPPORTED_GAIN'),([-.1,-.001],'SUPPORTED_LOSS'),(None,'DIRECTION_UNRESOLVED')]:
            self.assertEqual(V.evidence(dict(ci95=bounds)),label)

    def test_active_or_incomplete_run_rejected_before_heavy_reads(self):
        audit=V.Audit('/unused')
        with patch.object(audit,'json',side_effect=[{},dict(status='running'),dict(active={}),dict(status='completed')]),patch.object(V,'statistics',side_effect=AssertionError('must not bootstrap')):
            with self.assertRaisesRegex(V.AuditFailure,'incomplete/active'):audit.run()

    def test_exact_identity_rejects_boolean_integer_substitution(self):
        with self.assertRaises(V.AuditFailure):V.equal(1,True)
        with self.assertRaises(V.AuditFailure):V.equal(False,0)

if __name__=='__main__':unittest.main()
