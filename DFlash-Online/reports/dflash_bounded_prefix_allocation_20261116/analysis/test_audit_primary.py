"""Synthetic full-panel fixtures only; no model, feedback or real result read."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from test_full_k_gsm8k_evaluation import FullBackend
from test_persistent_prefix_evaluation import inputs,permutations,references,quiet
from experiments.persistent_prefix_withholding.evaluation import evaluate_final,evaluation_schedule
from experiments.bounded_prefix_allocation.analysis import write_report,summarize

spec=importlib.util.spec_from_file_location('independent_primary',Path(__file__).with_name('audit_primary.py'))
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


class IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        rows=inputs();p=permutations();refs=references(rows);warm=deepcopy(rows[:2])
        for i,row in enumerate(warm):row.update(source_id=f'old:{i}',request_id=f'warm{i}',prompt_ids=[1,9000+i])
        warmrefs=references(warm);checkpoints={a:'/'+a for a in A.ARMS}
        final=cls.root/'tasks/final/evaluation'
        evaluate_final(FullBackend(rows),rows,checkpoints,final,canonicals=refs,warmup_requests=warm,
            warmup_canonicals=warmrefs,schedule=evaluation_schedule(rows,p,methods=A.ARMS),within_block_permutations=p,
            physical_gpu_uuid='GPU-fixed',boundary_check=quiet,execution_commit='a'*40,methods=A.ARMS,session='bounded_prefix_allocation_event8')
        write_report(final)
        def save(rel,value):
            path=cls.root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        save('sealed/PANEL.json',dict(requests=rows,within_block_permutations=p))
        save('references/CANONICALS.json',refs);save('sealed/WARMUPS.json',dict(requests=warm));save('sealed/WARMUP_REFERENCES.json',warmrefs)
        save('tasks/C8/RESULT.json',dict(status='completed',endpoint=dict(global_step=1024,presentations=2048,export='/C')))
        save('sealed/PLAN.json',dict(root=str(cls.root),execution_commit='a'*40,devices={'A':'GPU-fixed'},
            controls={a:dict(export='/'+a) for a in ('N','S')},
            data_panel=str(cls.root/'sealed/PANEL.json'),references=str(cls.root/'references/CANONICALS.json'),
            warmups=str(cls.root/'sealed/WARMUPS.json'),warmup_references=str(cls.root/'sealed/WARMUP_REFERENCES.json')))
        cls.manifest={str(p.relative_to(cls.root)):A.identity(p.read_bytes()) for p in cls.root.rglob('*') if p.is_file()}
        (cls.root/'DELIVERY_MANIFEST.json').write_text(json.dumps(cls.manifest))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def run_audit(self):
        with tempfile.NamedTemporaryFile() as output:return A.run(self.root,output.name)

    def mutation(self,relative,change,callback):
        path=self.root/relative;raw=path.read_bytes();original_manifest=dict(self.manifest)
        try:
            value=json.loads(raw);change(value);path.write_text(json.dumps(value))
            self.manifest[relative]=A.identity(path.read_bytes())
            (self.root/'DELIVERY_MANIFEST.json').write_text(json.dumps(self.manifest))
            callback(self.run_audit())
        finally:
            path.write_bytes(raw);self.manifest.clear();self.manifest.update(original_manifest)
            (self.root/'DELIVERY_MANIFEST.json').write_text(json.dumps(self.manifest))

    def test_actual_retrieval_manifest_schema(self):
        manifest=self.root/'DELIVERY_MANIFEST.json';retrieval=self.root/'RETRIEVAL_MANIFEST.json'
        manifest.rename(retrieval)
        try:
            result=self.run_audit()
            self.assertEqual(result['status'],'passed',result['errors'][:3])
            self.assertEqual(result['input_manifest'],'RETRIEVAL_MANIFEST.json')
        finally:retrieval.rename(manifest)

    def test_complete_raw_and_bootstrap_agree(self):
        result=self.run_audit()
        self.assertEqual(result['status'],'passed',result['errors'][:3])
        self.assertEqual(result['errors'],[])
        self.assertEqual(result['raw_generations'],1224)
        self.assertEqual(result['scored_generations'],1152)
        self.assertEqual(result['warmup_generations'],72)
        self.assertEqual(result['independent']['method_totals']['C']['verification_calls'],256)
        self.assertEqual(result['independent']['comparisons']['C_vs_N']['time_reduction']['point'],0.)
        self.assertEqual(result['decision'],'CLOSE_FIXED_CANDIDATE_AT_THIS_SCREEN')


    def test_nondegenerate_statistics_match_paired_production(self):
        records=[json.loads(line) for line in (self.root/'tasks/final/evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
        for row in records:
            if row['phase']!='scored':continue
            i=row['panel_ordinal'];arm=row['method'];rep=row['repeat']
            # Mixed directions and lengths ensure nonzero interval width and W/L/T.
            base=20+i%11
            change=((-2,-1,0,1,2)[i%5] if arm=='C' else (-1 if i%3==0 else 0) if arm=='S' else 0)
            calls=base+change;progress=80+i%31;native=progress-calls
            q,r=divmod(native,calls)
            hist={str(q):calls-r}
            if r:hist[str(q+1)]=r
            row.update(verification_calls=calls,committed_progress=progress,output_tokens=progress+1,
                native_accepted_draft_tokens=native,accepted_draft_tokens=native,
                native_acceptance_histogram=hist,request_latency_s=calls*.02+rep*.001+(i%7)*.0003,
                native_generate_s=calls*.018+rep*.001,cache_maintenance_s=.01+(i%3)*.002)
        independent=A.statistics(records);production=summarize(records)
        audit=A.Audit(self.root)
        for key in ('method_totals','comparisons','repeat_totals','paired_requests'):
            A.compare_tree(audit,production[key],independent[key],key)
        self.assertEqual(audit.errors,[])
        self.assertEqual(independent['draw_matrix_sha256'],production['bootstrap']['draw_matrix_sha256'])
        c=independent['comparisons']['C_vs_N']
        self.assertEqual((c['requests_fewer_calls'],c['requests_more_calls'],c['request_call_ties']),(52,50,26))
        self.assertEqual((c['gross_calls_saved'],c['gross_calls_added']),(78,75))
        self.assertLess(c['call_reduction']['ci95'][0],c['call_reduction']['ci95'][1])

    def test_matched_eos_is_separate_from_native_acceptance(self):
        raw=json.loads((self.root/'tasks/final/evaluation/0000_r0_C.json').read_text())
        meta=raw['metadata'];block=deepcopy(meta['trace'][0])
        target=raw['response_ids'][1:]+[0]*12
        block.update(target_argmax=target,draft_tokens=target[:15],committed_progress=4,
            committed_tokens=raw['response_ids'][1:],raw_accepted_draft_tokens_before_stop=15,
            accepted_draft_tokens=4,native_accepted_draft_tokens=3,stop_clipped=True,finished=True)
        meta.update(trace=[block],verification_calls=1,accepted_draft_tokens=4,committed_progress=4)
        meta['capture']['blocks']=[deepcopy(block)]
        protocol=json.loads((self.root/'tasks/final/evaluation/PROTOCOL.json').read_text())
        request=json.loads((self.root/'sealed/PANEL.json').read_text())['requests'][0]
        canonical=json.loads((self.root/'references/CANONICALS.json').read_text())[request['request_id']]
        slot=dict(phase='scored',activation=0,repeat=0,block_ordinal=0,position=0,method='C',panel_ordinal=0)
        audit=A.Audit(self.root)
        counted=A.raw_counts(audit,raw,request,canonical,slot,protocol,deepcopy(meta['capture']))
        self.assertEqual(audit.errors,[])
        self.assertEqual((counted['verification_calls'],counted['native_accepted_draft_tokens'],counted['accepted_draft_tokens'],counted['committed_progress']),(1,3,4,4))

    def test_native_decision_corruption_is_not_accepted_by_updated_manifest(self):
        def change(row):row['metadata']['trace'][0]['draft_tokens'][0]+=1
        def check(result):
            self.assertEqual(result['status'],'INVALID_OR_INCOMPLETE')
            self.assertNotIn('independent',result)
            self.assertTrue(result['errors'])
        self.mutation('tasks/final/evaluation/0000_r0_C.json',change,check)

    def test_primary_summary_disagreement_is_detected(self):
        def change(summary):summary['comparisons']['C_vs_N']['time_reduction']['point']=.1
        def check(result):
            self.assertEqual(result['status'],'AUDIT_DISAGREEMENT')
            self.assertIn('summary:comparisons.C_vs_N.time_reduction.point',result['errors'])
        self.mutation('tasks/final/evaluation/SUMMARY.json',change,check)

    def test_incomplete_status_never_yields_effects(self):
        def change(status):status['status']='failed';status['scored_generations']=1151
        def check(result):
            self.assertEqual(result['status'],'INVALID_OR_INCOMPLETE')
            self.assertNotIn('independent',result)
        self.mutation('tasks/final/evaluation/STATUS.json',change,check)

    def test_missing_native_record_preserves_failure(self):
        path=self.root/'tasks/final/evaluation/0000_r0_C_NATIVE.json';raw=path.read_bytes()
        try:
            path.unlink();result=self.run_audit()
            self.assertEqual(result['status'],'INVALID_OR_INCOMPLETE')
            self.assertNotIn('independent',result)
        finally:path.write_bytes(raw)


if __name__=='__main__':unittest.main()
