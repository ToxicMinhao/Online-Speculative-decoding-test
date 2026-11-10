from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.verification_transition.budget import WorkBudget
from experiments.verification_transition.campaign import compare_walk, run_campaign, check_crossed
from test_verification_transition_analysis import walk


class FakeBackend:
    """CPU protocol fixture only; this cannot qualify a native constructor."""
    def __init__(self, original):
        self.original=original;self.suffixes=[];self.walks=[];self.releases=0

    def native_walk(self, request, checkpoint, capture_rounds):
        self.walks.append(request['request_id'])
        return dict(record=deepcopy(self.original), successors={str(i):i for i in capture_rounds})

    def continue_from(self, boundary, checkpoint, kind):
        self.suffixes.append((boundary,checkpoint,kind))
        record=deepcopy(self.original)
        record['metadata']['trace']=record['metadata']['trace'][boundary:]
        record['metadata']['verification_calls']=len(record['metadata']['trace'])
        return dict(record=record,verification_calls=len(record['metadata']['trace']),
                    reconstruction=dict(status='passed'))

    def release_request(self):self.releases+=1


def fixture(original):
    requests=[dict(dataset=d,request_id=f'{d}{i}',selection_rank=i,
                   request={'request_id':f'{d}{i}'},canonical={'response_ids':original['response_ids'],
                                                             'finish_reason':'eos'})
              for i in range(64) for d in ('gsm8k','mathqa')]
    return dict(requests=requests,checkpoints={d:{k:{'export':d+k} for k in ('KPLUS4','KPLUS6')}
                for d in ('gsm8k','mathqa')},execution_commit='cpu-fixture')


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.record=walk([3,3,3]);self.record['stop_token_ids']=[109]

    def test_neutral_qualification_included_and_dataset_order_preserved(self):
        backend=FakeBackend(self.record);budget=WorkBudget(self.root/'WORK.jsonl')
        with patch('experiments.verification_transition.campaign.load_originals',
                   return_value={'KPLUS4':self.record,'KPLUS6':self.record}):
            result=run_campaign(fixture(self.record),backend,budget,self.root/'out')
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['completed_requests'],128)
        self.assertEqual(budget.summary()['completed'],{'original_completion':256,'neutral_suffix':8})
        self.assertEqual(backend.walks[:4],['gsm8k0','gsm8k0','mathqa0','mathqa0'])
        self.assertEqual(backend.releases,128)

    def test_no_nonterminal_boundary_stops_without_substitution(self):
        record=walk([3]);record['stop_token_ids']=[103]
        backend=FakeBackend(record);budget=WorkBudget(self.root/'WORK.jsonl')
        with patch('experiments.verification_transition.campaign.load_originals',
                   return_value={'KPLUS4':record,'KPLUS6':record}):
            result=run_campaign(fixture(record),backend,budget,self.root/'out')
        self.assertEqual(result['status'],'technically_incomplete')
        self.assertEqual(backend.walks,[])
        self.assertIn('No nonterminal',result['failure']['message'])

    def test_changed_proposal_is_failure_even_when_final_answer_matches(self):
        changed=deepcopy(self.record);changed['metadata']['trace'][0]['draft_tokens'][9]=-66
        with self.assertRaisesRegex(ValueError,'draft_tokens'):compare_walk(changed,self.record)

    def test_native_position_change_is_failure(self):
        changed = deepcopy(self.record)
        changed['metadata']['trace'][0]['positions'][0] += 1
        with self.assertRaisesRegex(ValueError, 'positions'):
            compare_walk(changed, self.record)

    def test_suffix_cannot_reset_or_extend_original_budget(self):
        record=deepcopy(self.record);record['metadata']['trace']=record['metadata']['trace'][1:]
        record['metadata']['verification_calls']=2;record['max_new_tokens']=2047
        result=dict(record=record,verification_calls=2)
        with self.assertRaises(ValueError):check_crossed(result,self.record,self.record,1)

    def test_budget_stop_preserves_all_selected_rows_and_no_retries(self):
        backend=FakeBackend(self.record)
        budget=WorkBudget(self.root/'WORK.jsonl',limits={'original_completion':1,'neutral_suffix':8,'crossed_suffix':256,'draft_block':32768})
        with patch('experiments.verification_transition.campaign.load_originals',
                   return_value={'KPLUS4':self.record,'KPLUS6':self.record}):
            result=run_campaign(fixture(self.record),backend,budget,self.root/'out')
        self.assertEqual(result['status'],'incomplete');self.assertEqual(len(backend.walks),1)
        import json
        self.assertEqual(len(json.loads((self.root/'out/ROWS.json').read_text())),128)
        self.assertIn('TECHNICALLY INCOMPLETE',(self.root/'out/report/REPORT.md').read_text())


if __name__=='__main__':unittest.main()
