import unittest
from audit_work import ledger_counts
import diagnostics


class FinalDiagnosticTests(unittest.TestCase):
    def test_mathqa_seals_distinct_sampling_domain(self):
        records=[{'record_id':str(i)} for i in range(256)]
        gsm=diagnostics.expected_binding(records,5,65,'gsm8k')
        math=diagnostics.expected_binding(records,5,65,'mathqa')
        self.assertEqual(math['dataset'],'mathqa')
        self.assertEqual(math['diagnostic_rng_key'],[20261108,1,5,65])
        self.assertEqual(math['fitting_ids'],gsm['fitting_ids'])
        self.assertNotEqual(math['diagnostic_rng_key'],gsm['diagnostic_rng_key'])

    def test_ledger_preserves_attempted_failure_and_unknown(self):
        rows=[{'kind':'additional_forward','status':'attempted'},
              {'kind':'additional_forward','status':'failed'},
              {'kind':'additional_backward','status':'attempted'}]
        result=ledger_counts(rows)
        self.assertEqual(result['additional_forward','failed'],1)
        self.assertEqual(result['additional_backward','attempted'],1)
        self.assertEqual(result['additional_backward','completed'],0)

    def test_ledger_rejects_adam_and_duplicate_completion(self):
        cases=[[{'kind':'adam','status':'attempted'}],
               [{'kind':'additional_forward','status':'completed'}],
               [{'kind':'other','status':'attempted'}],
               [{'kind':'additional_forward','status':'attempted'},
                {'kind':'additional_backward','status':'attempted'}]]
        for rows in cases:
            with self.assertRaises(ValueError):ledger_counts(rows)

    def test_ledger_forty_bound_includes_failed_attempts(self):
        pair=[{'kind':'additional_forward','status':'attempted'},
              {'kind':'additional_forward','status':'failed'}]
        self.assertEqual(ledger_counts(pair*40)['additional_forward','attempted'],40)
        with self.assertRaises(ValueError):ledger_counts(pair*41)


if __name__=='__main__':unittest.main()
