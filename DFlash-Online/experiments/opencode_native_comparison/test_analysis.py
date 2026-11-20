from copy import deepcopy
import unittest

from .contract import METHODS
from .analysis import reduce_requests, summarize


def fixture():
    rows=[]
    for request_id,calls in [('b',0),('a',10)]:
        output=1 if calls==0 else 41
        for method in METHODS:
            n = None if method=='AR' else max(0,calls-(1 if method=='C_CODE58' and calls else 0))
            for repeat in range(3):
                row=dict(request_id=request_id,source_id=request_id,method=method,
                    phase='scored',repeat=repeat,valid_measurement=True,errors=[],
                    output_tokens=output,verification_calls=n,
                    accepted_draft_tokens=None if n is None else output-1-n,
                    native_accepted_draft_tokens=None if n is None else output-1-n,
                    committed_progress=None if n is None else output-1,
                    finish_reason='eos',checkpoint_id=method,physical_gpu_uuid='cardA',
                    request_latency_s=(2 if method=='AR' else 1)*(repeat+1),
                    native_generate_s=(1 if method=='AR' else .5)*(repeat+1),
                    cache_maintenance_s=0.)
                rows.append(row)
    return rows


class AnalysisTests(unittest.TestCase):
    def test_repeats_are_not_samples_and_zero_calls_retained(self):
        report=summarize(fixture(),expected_requests=2)
        self.assertEqual(report['requests'],2)
        self.assertEqual(report['bootstrap']['input_order'],['a','b'])
        self.assertEqual(report['totals']['O_CODE58']['verification_calls'],10)
        self.assertEqual(report['totals']['O_CODE58']['output_tokens'],42)
        self.assertEqual(report['totals']['AR']['verification_calls'],None)
        self.assertEqual(report['totals']['O_CODE58']['request_latency_s'],4)
        self.assertEqual(report['contrasts']['O_CODE58_vs_AR']['absolute_speedup']['value'],2)
        comparison=report['contrasts']['C_CODE58_vs_AREPO_CODE58']
        self.assertEqual((comparison['wins'],comparison['ties'],comparison['gross_calls_saved']),(1,1,1))

    def test_incomplete_duplicate_tampered_or_invalid_rejected(self):
        for mutation in ('missing','duplicate','invalid','repeat','ar_counts','gpu'):
            rows=deepcopy(fixture())
            if mutation=='missing': rows.pop()
            elif mutation=='duplicate': rows.append(deepcopy(rows[-1]))
            elif mutation=='invalid': rows[0]['valid_measurement']=False
            elif mutation=='repeat': rows[0]['output_tokens']=2
            elif mutation=='ar_counts':
                for row in rows:
                    if row['method']=='AR': row['verification_calls']=0
            else: rows[0]['physical_gpu_uuid']='cardB'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                reduce_requests(rows,expected_requests=2)


if __name__=='__main__':
    unittest.main()
