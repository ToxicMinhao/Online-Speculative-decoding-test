"""Exercise the actual new arm names through execution, raw audit and reduction."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from experiments.bounded_prefix_allocation import analysis as A
from experiments.persistent_prefix_withholding import evaluation as E
from test_persistent_prefix_evaluation import inputs, permutations, references, quiet
from test_full_k_gsm8k_evaluation import FullBackend


class CandidateEvaluationTests(unittest.TestCase):
    def test_complete_real_consumer_paths_and_distinct_candidate(self):
        requests=inputs(); warmups=deepcopy(requests[:2]); perm=permutations()
        for i,row in enumerate(warmups):
            row.update(request_id='warm'+str(i),source_id='old:'+str(i),prompt_ids=[1,9000+i])
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'evaluation'
            backend=FullBackend(requests)
            result=E.evaluate_final(backend,requests,{m:'/'+m for m in A.METHODS},root,
                canonicals=references(requests),warmup_requests=warmups,warmup_canonicals=references(warmups),
                schedule=E.evaluation_schedule(requests,perm,methods=A.METHODS),within_block_permutations=perm,
                physical_gpu_uuid='GPU-fixed',boundary_check=quiet,execution_commit='c'*40,
                methods=A.METHODS,session='bounded_prefix_allocation_event8')
            self.assertEqual(result['attempted_generations'],1224)
            audit=E.audit_raw_records(root)
            self.assertEqual(audit['status'],'passed',audit['errors'][:3])
            report=A.write_report(root)
            self.assertEqual(report['status'],'completed')
            self.assertEqual(set(report['method_totals']),set(('N','C','S')))
            self.assertEqual(report['method_totals']['C']['verification_calls'],256)
            self.assertEqual(report['decision']['outcome'],'CLOSE_FIXED_CANDIDATE_AT_THIS_SCREEN')
            self.assertEqual(len((root/'ALL_REPEATS.csv').read_text().splitlines()),1153)
            self.assertEqual(len((root/'PER_REQUEST.csv').read_text().splitlines()),385)
            self.assertTrue(all(x[2]['session_id']=='bounded_prefix_allocation_event8' for x in backend.executions))


if __name__=='__main__':unittest.main()
