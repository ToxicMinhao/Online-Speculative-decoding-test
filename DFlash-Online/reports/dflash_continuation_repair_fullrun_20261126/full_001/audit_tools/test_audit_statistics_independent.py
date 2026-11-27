import sys
import unittest
import hashlib
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
study=Path(__file__).resolve().parent
code=study/'implementation_checkout/DFlash-Online'
sys.path[:0]=[str(code),str(code/'tests')]
from audit_statistics_independent import reduce_independently,compare,audit,digest
from test_continuation_repair_fullrun_serving import rows_fixture
from experiments.continuation_repair_fullrun.analysis import summarize,selection_decision

class IndependentArithmeticTests(unittest.TestCase):
    def test_same_draw_stream_and_results(self):
        rows=rows_fixture();actual=summarize(rows,expected_requests=4)
        actual['decision']=selection_decision(actual,complete=True)
        expected=reduce_independently(rows,n=4)
        self.assertEqual(expected['draw_sha256'],actual['bootstrap']['draw_sha256'])
        self.assertEqual(compare({k:expected[k] for k in ('requests','totals','contrasts','decision')},actual),127)
    def test_corrupt_scalar_and_missing_repeat_rejected(self):
        with self.assertRaises(ValueError):compare({'value':.1},{'value':.2})
        with self.assertRaises(ValueError):reduce_independently(rows_fixture()[:-1],n=4)


class StatisticalBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'evaluation').mkdir()
        self.external={}
        def external(name,value):
            path=self.root/name;payload=json.dumps(value).encode();path.write_bytes(payload)
            locator='/sealed-source/'+name;self.external[locator]=str(path)
            return locator,dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        source,source_id=external('SOURCE.json',dict(commit='abc',clean=True))
        panel=[dict(request_id=f'r{i}') for i in range(1024)]
        warmups=[dict(request_id=f'w{i}') for i in range(2)]
        inputs,input_id=external('INPUTS.json',dict(evaluation_requests=panel,warmup_requests=warmups))
        self.plan=dict(root='/original/run',execution_commit='abc',source_manifest=source,
                       source_identity=source_id,evaluation_inputs=inputs,input_identities={inputs:input_id})
        self.plan['binding']=digest(self.plan)
        self.rows=[dict(phase='scored',request_id=row['request_id']) for row in panel for _ in range(12)]
        self.rows += [dict(phase='warmup',request_id=row['request_id']) for row in warmups for _ in range(48)]
        (self.root/'evaluation/MEASUREMENTS.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        self.reduced=dict(requests=1024,totals={},contrasts={},decision={},draw_sha256='draws')
        self.final={k:v for k,v in self.reduced.items() if k!='draw_sha256'}
        self.final.update(execution_commit='abc',bootstrap=dict(draw_sha256='draws'))
        self.write_final()
    def write_final(self):
        (self.root/'FINAL_ANALYSIS.json').write_text(json.dumps(self.final))
    def test_bound_result_and_actual_input_identities(self):
        with patch('audit_statistics_independent.reduce_independently',return_value=self.reduced):
            result,inputs=audit(self.plan,self.root,self.external)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['plan_binding'],self.plan['binding'])
        self.assertEqual(result['execution_commit'],'abc')
        self.assertEqual(len(inputs),4)
        self.assertEqual(result['input_manifest_digest'],digest(inputs))
        self.assertIn('arithmetic only',result['scope'])
    def test_changed_plan_source_and_analysis_rejected(self):
        changed=dict(self.plan,execution_commit='wrong')
        self.assertEqual(audit(changed,self.root,self.external)[0]['status'],'failed')
        source=Path(self.external[self.plan['source_manifest']]);original=source.read_bytes()
        source.write_bytes(original+b' ')
        self.assertEqual(audit(self.plan,self.root,self.external)[0]['status'],'failed')
        source.write_bytes(original);self.final['execution_commit']='wrong';self.write_final()
        self.assertEqual(audit(self.plan,self.root,self.external)[0]['status'],'failed')
    def test_missing_mapping_and_changed_panel_rejected(self):
        self.assertEqual(audit(self.plan,self.root)[0]['status'],'incomplete')
        self.rows[0]['request_id']='not-sealed'
        (self.root/'evaluation/MEASUREMENTS.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        self.assertEqual(audit(self.plan,self.root,self.external)[0]['status'],'failed')

if __name__=='__main__':unittest.main()
