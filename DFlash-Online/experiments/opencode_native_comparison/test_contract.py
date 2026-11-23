import copy
import unittest
from .contract import (event_requests, validate_requests, PRODUCTION_WORK,
                       QUALIFICATION_CEILINGS)
from .schedule import evaluation_schedule, schedule_summary


class ContractTests(unittest.TestCase):
    def test_production_counts_exclude_gsm_tail(self):
        generations = sum(v for k, v in PRODUCTION_WORK.items() if k.endswith('generation'))
        self.assertEqual(generations, 39932)
        self.assertEqual(generations+QUALIFICATION_CEILINGS['generation'], 39996)
        self.assertEqual(PRODUCTION_WORK['adam'], 3*58*128)
        manifest = dict(train_requests=list(range(7424)))
        self.assertEqual(event_requests(manifest, 58), list(range(7296, 7424)))
        with self.assertRaises(ValueError):
            event_requests(manifest, 59)

    def test_all_pairs_have_balanced_precedence(self):
        requests = [dict(request_id=str(i)) for i in range(1024)]
        rows = evaluation_schedule(requests)
        result = schedule_summary(rows)
        self.assertEqual(result['warmup_generations'], 240)
        self.assertEqual(rows[0]['method'], 'O_CODE58')
        self.assertEqual(rows[5]['method'], 'AR')
        with self.assertRaises(ValueError):
            schedule_summary(rows[:-1])
        duplicate = copy.deepcopy(rows)
        duplicate[0]['request_ids'][1] = duplicate[0]['request_ids'][0]
        with self.assertRaises(ValueError):
            schedule_summary(duplicate)

    def test_shared_context_eligibility_keeps_whole_prompt(self):
        request = dict(request_id='a', request_index=0, panel_ordinal=0,
                       prompt_ids=[1]*2048, max_new_tokens=2048,
                       stop_token_ids=[151645, 151643])
        validate_requests([request], count=1, role='evaluation')
        with self.assertRaises(ValueError):
            validate_requests([dict(request, prompt_ids=[1]*2049)], count=1, role='evaluation')
        with self.assertRaises(ValueError):
            validate_requests([dict(request, output='unused')], count=1, role='evaluation')


if __name__ == '__main__':
    unittest.main()
