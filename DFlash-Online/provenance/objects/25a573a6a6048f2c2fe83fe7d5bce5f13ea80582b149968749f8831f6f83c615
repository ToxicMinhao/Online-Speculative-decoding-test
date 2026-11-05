"""Independent arithmetic and count-provenance fixtures; no model execution."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import audit_results as audit


class AuditTests(unittest.TestCase):
    def test_primary_favors_correction_and_counts_each_request_once(self):
        requests = [dict(request_id='a', panel_ordinal=0),dict(request_id='b', panel_ordinal=1)]
        rows = []
        for request, calls in zip(requests, ((10,9),(10,12))):
            for repeat in range(3):
                for arm,count in zip(audit.ARMS,calls):
                    rows.append(dict(request_id=request['request_id'],method=arm,repeat=repeat,
                        verification_calls=count,accepted_draft_tokens=20,committed_progress=30,
                        output_tokens=30,request_latency_s=1+repeat,native_generate_s=.5+repeat,
                        cache_maintenance_s=.1))
        totals, comparison, _ = audit.pool(rows,requests)
        self.assertEqual(totals['KRC2_OFF']['verification_calls'],20)
        self.assertEqual(totals['KRC2']['verification_calls'],21)
        self.assertEqual(totals['KRC2']['summed_mean_request_latency_s'],4)
        self.assertAlmostEqual(comparison['call_reduction_pct'],-5)
        self.assertEqual(comparison['call_request_directions'],dict(improved=1,regressed=1,tied=0))
        self.assertEqual((comparison['call_gross_saved'],comparison['call_gross_extra']),(1,2))
        with self.assertRaises(ValueError): audit.pool(rows[:-1],requests)

    def test_off_learning_and_correction_advantage_are_distinct(self):
        requests=[dict(request_id='a')]
        values={('a','KRC2_OFF'):dict(verification_calls=90),('a','KRC2'):dict(verification_calls=95)}
        history=dict(KRC1={'a':100},K2={'a':92})
        result=audit.count_history(values,requests,history)['comparisons']
        self.assertAlmostEqual(result['KRC2_OFF_vs_KRC1']['call_reduction_pct'],10)
        self.assertGreater(result['KRC2_OFF_vs_K2']['call_reduction_pct'],0)
        self.assertEqual(result['KRC2_vs_KRC1']['candidate_calls'],95)

    def test_exact_balanced_budget(self):
        requests=[dict(request_id=str(i),panel_ordinal=i) for i in range(64)]
        slots=audit.slots(requests)
        self.assertEqual(len(slots),384)
        for repeat in range(3):
            starts=[arm for i,(_,rep,arm) in enumerate(slots) if i%2==0 and rep==repeat]
            self.assertEqual(starts.count('KRC2_OFF'),32)
            self.assertEqual(starts.count('KRC2'),32)

    def test_both_panels_bind_krc2_not_k2_or_krc1(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            retained=root/'krc1';retained.mkdir()
            (retained/'MEASUREMENTS.jsonl').write_text(json.dumps(dict(request_id='a',method='KRC',repeat=0,verification_calls=20)))
            prior=root/'dflash_krc_continuation_20261028/raw/campaign_001/evaluation'
            for name in ('development','additional'):
                folder=prior/name;folder.mkdir(parents=True)
                (folder/'MEASUREMENTS.jsonl').write_text('\n'.join(json.dumps(dict(request_id='a',method=arm,repeat=0,verification_calls=calls)) for arm,calls in (('K2',19),('KRC2',18))))
                panel=dict(name=name,historical_counts=dict(KRC1={'a':20},K2={'a':19}),expected_control_counts={'a':18})
                with patch.object(audit,'HERE',root/'newrun'):
                    issues=[];audit.audit_historical_inputs(panel,retained,issues)
                    self.assertEqual(issues,[])
                    panel['expected_control_counts']={'a':19}
                    audit.audit_historical_inputs(panel,retained,issues)
                    self.assertEqual(issues,[name+': retained KRC2 counts'])

if __name__=='__main__': unittest.main()
