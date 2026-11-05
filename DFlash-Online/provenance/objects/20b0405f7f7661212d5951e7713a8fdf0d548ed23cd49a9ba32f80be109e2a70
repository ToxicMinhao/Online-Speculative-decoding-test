"""Parser/accounting tests; no experimental model or source changes."""
import unittest
from pathlib import Path
import audit_results as audit


class AuditAccountingTests(unittest.TestCase):
    def test_counts_once_timing_mean_and_signed_primary(self):
        requests=[dict(request_id='a',panel_ordinal=0),dict(request_id='b',panel_ordinal=1)]
        rows=[]
        for rid,kr,krc in [('a',8,10),('b',15,10)]:
            for repeat in range(3):
                for arm,calls in [('KR',kr),('KRC',krc)]:
                    rows.append(dict(request_id=rid,method=arm,repeat=repeat,
                        verification_calls=calls,accepted_draft_tokens=20,committed_progress=30,output_tokens=31,
                        request_latency_s=(repeat+1)*(2 if arm=='KR' else 3),
                        native_generate_s=1.,cache_maintenance_s=.1))
        totals,comparison,_=audit.pool(rows,requests)
        self.assertEqual(totals['KR']['verification_calls'],23)
        self.assertEqual(totals['KRC']['verification_calls'],20)
        self.assertEqual(totals['KR']['summed_mean_request_latency_s'],8.)
        self.assertAlmostEqual(comparison['call_reduction_pct'],-15.)
        self.assertEqual(comparison['call_request_directions'],dict(improved=1,regressed=1,tied=0))
        self.assertEqual(comparison['call_gross_saved'],2)
        self.assertEqual(comparison['call_gross_extra'],5)
        with self.assertRaises(ValueError):audit.pool(rows[:-1],requests)

    def test_balanced_order_with_no_extra_slots(self):
        requests=[dict(request_id=str(i),panel_ordinal=i) for i in range(64)]
        slots=audit.slots(requests)
        self.assertEqual(len(slots),384)
        self.assertEqual(len(set(slots)),384)
        for repeat in range(3):
            first=[arm for i,(_,r,arm) in enumerate(slots) if i%2==0 and r==repeat]
            self.assertEqual(first.count('KR'),32)
            self.assertEqual(first.count('KRC'),32)

    def test_padded_anchors_and_canonical_author_mapping(self):
        row=dict(sampled_anchors=[dict(positions=[[7,8,9]],keep_mask=[[True,False,True]])])
        self.assertEqual(list(audit.active_anchors(row)),[7,9])
        source='/data/sources/minhao/DFlash-Online/training/specforge/__init__.py'
        local=audit.mapped(source,audit.HERE/'raw')
        self.assertTrue(local and local.is_file())
        self.assertTrue(str(local).startswith('/home/minhao2/DFlash-Research/training/'))


if __name__=='__main__':unittest.main()
