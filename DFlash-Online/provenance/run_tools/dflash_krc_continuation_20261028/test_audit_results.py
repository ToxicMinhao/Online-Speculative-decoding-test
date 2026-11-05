"""Small arithmetic/provenance fixtures; no model execution or raw changes."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import audit_results as audit


class AuditTests(unittest.TestCase):
    def test_counts_once_three_repeat_mean_and_signed_regressions(self):
        requests=[dict(request_id='a',panel_ordinal=0),dict(request_id='b',panel_ordinal=1)]
        rows=[]
        for request,counts in zip(requests,((10,9),(10,12))):
            for repeat in range(3):
                for arm,calls in zip(audit.ARMS,counts):
                    rows.append(dict(request_id=request['request_id'],method=arm,repeat=repeat,
                        verification_calls=calls,accepted_draft_tokens=20,committed_progress=30,
                        output_tokens=30,request_latency_s=1+repeat,native_generate_s=.5+repeat,
                        cache_maintenance_s=.1))
        totals,comparison,_=audit.pool(rows,requests)
        self.assertEqual(totals['K2']['verification_calls'],20)
        self.assertEqual(totals['KRC2']['verification_calls'],21)
        self.assertEqual(totals['K2']['summed_mean_request_latency_s'],4)
        self.assertAlmostEqual(comparison['call_reduction_pct'],-5)
        self.assertEqual(comparison['call_request_directions'],dict(improved=1,regressed=1,tied=0))
        self.assertEqual((comparison['call_gross_saved'],comparison['call_gross_extra']),(1,2))
        with self.assertRaises(ValueError):
            audit.pool(rows[:-1],requests)

    def test_relative_advantage_can_grow_while_krc_itself_regresses(self):
        requests=[dict(request_id='a')]
        values={('a','K2'):dict(verification_calls=120),('a','KRC2'):dict(verification_calls=100)}
        history=dict(K1={'a':100},KRC1={'a':90})
        result=audit.count_history(values,requests,history)
        self.assertGreater(result['relative_advantage_change_percentage_points'],0)
        self.assertLess(result['comparisons']['KRC2_vs_KRC1']['call_reduction_pct'],0)

    def test_orders_balance_each_repeat_without_extra_slots(self):
        requests=[dict(request_id=str(i),panel_ordinal=i) for i in range(64)]
        slots=audit.slots(requests)
        self.assertEqual(len(slots),384)
        for repeat in range(3):
            starts=[arm for i,(_,rep,arm) in enumerate(slots) if i%2==0 and rep==repeat]
            self.assertEqual(starts.count('K2'),32)
            self.assertEqual(starts.count('KRC2'),32)

    def test_additional_history_is_descriptive_not_a_k2_gate(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            rows=[dict(request_id='a',method=method,repeat=0,verification_calls=calls)
                  for method,calls in (('K1',20),('KRC',18))]
            (root/'MEASUREMENTS.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            panel=dict(name='additional',historical_counts=dict(K1={'a':20},KRC1={'a':18}),
                       expected_control_counts=None)
            issues=[]
            audit.audit_historical_inputs(panel,root,issues)
            self.assertEqual(issues,[])
            panel['expected_control_counts']={'a':20}
            audit.audit_historical_inputs(panel,root,issues)
            self.assertEqual(issues,['additional: no historical K2 or K1 count gate'])


if __name__=='__main__':
    unittest.main()
