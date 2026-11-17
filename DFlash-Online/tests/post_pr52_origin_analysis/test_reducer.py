from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.post_pr52_origin_analysis import reducer as R


def check(condition, name):
    if not condition:raise ValueError(name)


def fixture(response=None, blocks=None, finish='eos'):
    response=[1,9] if response is None else response
    blocks=[dict(draft_tokens=[9,3],target_argmax=[9,3,4],verify_size=3,committed_progress=1,
        output_start=0,anchor_token=1,positions=[2,3,4],committed_tokens=[9],
        raw_accepted_draft_tokens_before_stop=2,accepted_draft_tokens=1,native_accepted_draft_tokens=0,
        stop_clipped=True,finished=True)] if blocks is None else blocks
    req=dict(prompt_ids=[7,8],panel_ordinal=0,stop_token_ids=[9])
    canonical=dict(prompt_ids=req['prompt_ids'],response_ids=response,finish_reason=finish)
    capture=dict(collect_features=False,feature_bytes=0,native_capture_validated=True,blocks=blocks,
        prompt_ids=req['prompt_ids'],response_ids=response)
    record=dict(prompt_ids=req['prompt_ids'],response_ids=response,record_id='r0',request_index=0,session_id='session',
        purpose='evaluation',checkpoint_id='origin',max_new_tokens=2048,stop_token_ids=[9],finish_reason=finish,
        metadata=dict(**R.SETTINGS,capture=capture,trace=blocks,feedback_capture_in_timed_request=False,
        verification_calls=len(blocks),accepted_draft_tokens=sum(b['accepted_draft_tokens'] for b in blocks),
        committed_progress=len(response)-1))
    return record,capture,req,canonical


def count(values):
    return R.count_native(*values,checkpoint='origin',session='session',label='r0',check=check)


def paired(*,zero=False):
    rows=[]
    for i in range(256):
        row=dict(source_id=f'source{i:04d}',request_id=f'req{i}',panel_ordinal=i)
        for arm,calls in [('KPLUS4',12),('N',10),('C',9),('S',11)]:
            progress=0 if zero or i==0 else 30
            v=calls if progress else 0
            row[arm]=dict(verification_calls=v,native_accepted_draft_tokens=progress-v,
                accepted_draft_tokens=progress-v,committed_progress=progress,output_tokens=progress+1,
                finish_reason='eos',cap_hit=False)
        rows.append(row)
    return rows


class OriginReducerTests(unittest.TestCase):
    def test_stop_clipping_matching_is_not_native(self):
        result=count(fixture())
        self.assertEqual(result['native_accepted_draft_tokens'],0)
        self.assertEqual(result['accepted_draft_tokens'],1)
        self.assertEqual(result['committed_progress'],1)

    def test_first_token_eos_is_zero_call_not_discarded(self):
        result=count(fixture([9],[]))
        self.assertEqual([result[k] for k in R.COUNTS],[0,0,0,0,1])

    def test_length_cap_retains_last_partial_block(self):
        blocks=[];position=0
        while position<2047:
            c=min(16,2047-position)
            blocks.append(dict(draft_tokens=[1]*15,target_argmax=[1]*16,verify_size=16,committed_progress=c,
                output_start=position,anchor_token=1,positions=list(range(2+position,18+position)),
                committed_tokens=[1]*c,raw_accepted_draft_tokens_before_stop=15,accepted_draft_tokens=min(15,c),
                native_accepted_draft_tokens=c-1,stop_clipped=c<16,finished=position+c==2047))
            position+=c
        result=count(fixture([1]*2048,blocks,'length'))
        self.assertEqual(result['output_tokens'],2048)
        self.assertEqual(result['committed_progress'],2047)
        self.assertEqual(result['native_accepted_draft_tokens'],2047-len(blocks))
        self.assertTrue(result['cap_hit'])

    def test_wrong_source_or_output_or_native_counter_fails(self):
        for mode in ('input','output','checkpoint','counter'):
            with self.subTest(mode=mode):
                values=fixture()
                if mode=='input':values[2]['prompt_ids']=[1,2]
                elif mode=='output':values[3]['response_ids']=[1,8]
                elif mode=='checkpoint':values[0]['checkpoint_id']='other'
                else:values[0]['metadata']['trace'][0]['native_accepted_draft_tokens']=1
                with self.assertRaises(ValueError):count(values)

    def test_gap_at_seed_and_prior_eos_fail(self):
        values=fixture();values[0]['metadata']['trace'][0]['output_start']=1
        with self.assertRaisesRegex(ValueError,'seed'):count(values)
        with self.assertRaisesRegex(ValueError,'no_prior_eos'):count(fixture([9,9],[]))

    def test_reference_latency_not_a_required_or_used_field(self):
        values=fixture()
        values[0]['metadata'].update(request_latency_s=float('nan'),native_generate_s=-1)
        self.assertEqual(count(values)['verification_calls'],1)

    def test_complete_paired_pooled_bootstrap_and_original_seeds(self):
        rows=paired();results=[R.summarize(rows,('N','C','S'),seed) for seed in (1711,1717)]
        for result in results:
            self.assertEqual(result['method_totals']['N']['zero_call_requests'],1)
            self.assertAlmostEqual(result['comparisons']['N_vs_KPLUS4']['acceptance_gain']['point'],1/3)
            self.assertAlmostEqual(result['comparisons']['N_vs_KPLUS4']['call_reduction']['point'],1/6)
            self.assertEqual(result['comparisons']['N_vs_KPLUS4']['net_calls_saved'],510)
            self.assertEqual(result['comparisons']['N_vs_KPLUS4']['requests_improved'],255)
            self.assertEqual(result['comparisons']['N_vs_KPLUS4']['requests_tied'],1)
            self.assertFalse(result['reference_latency_used'])
        self.assertNotEqual(results[0]['bootstrap']['draw_matrix_sha256'],results[1]['bootstrap']['draw_matrix_sha256'])
        self.assertEqual(R.summarize(list(reversed(rows)),('N','C','S'),1717),results[1])

    def test_undefined_ratios_remain_visible(self):
        result=R.summarize(paired(zero=True),('N','S'),1711)
        gain=result['comparisons']['N_vs_KPLUS4']['acceptance_gain']
        self.assertEqual(gain,dict(point=None,ci95=None,undefined_draws=10000))
        self.assertIsNone(result['method_totals']['KPLUS4']['native_accepted_proposals_per_call'])
        json.dumps(result,allow_nan=False)

    def test_incomplete_or_duplicate_panels_fail(self):
        with self.assertRaises(ValueError):R.summarize(paired()[:-1],('N','S'),1711)
        rows=paired();rows[1]['source_id']=rows[0]['source_id']
        with self.assertRaises(ValueError):R.summarize(rows,('N','S'),1711)

    def test_manifest_hash_required(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'sealed').mkdir()
            payload=json.dumps({'root':'/original'}).encode();(root/'sealed/PLAN.json').write_bytes(payload)
            (root/'RETRIEVAL_MANIFEST.json').write_text(json.dumps({'sealed/PLAN.json':R.identity(payload)}))
            evidence=R.Evidence(root)
            self.assertEqual(evidence.plan['root'],'/original')
            (root/'sealed/PLAN.json').write_text('{"root":"/substituted"}')
            with self.assertRaisesRegex(ValueError,'source_identity'):R.Evidence(root)

    def test_missing_records_block_contrast_without_favorable_subset(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'compatibility.json').write_text(json.dumps({'panels':{'PR51':{},'PR52':{}}}))
            with patch.object(R,'load_panel',side_effect=FileNotFoundError('missing full source records')):
                result=R.run('p51','p52',path/'compatibility.json',path/'result')
            self.assertEqual(result['status'],'BLOCKED_RECORD_AVAILABILITY')
            self.assertTrue(all(not p['available_contrasts'] for p in result['panels'].values()))
            self.assertEqual(list((path/'result').glob('*SUMMARY.json')),[])


if __name__=='__main__':unittest.main()
