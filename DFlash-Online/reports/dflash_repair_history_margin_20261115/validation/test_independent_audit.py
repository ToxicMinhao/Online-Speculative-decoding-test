"""Synthetic fixture for the separate reduction; no real native outcomes."""
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.repair_history_margin.analysis import build_pre_k6, analyze_k6

spec=importlib.util.spec_from_file_location('independent_audit',Path(__file__).with_name('independent_audit.py'))
audit_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit_module)


class AuditTests(unittest.TestCase):
    def test_original_v1_numeric_depth_hash_is_explicit_and_immutable(self):
        body=dict(schema='repair_history_margin.pre_k6.v1',coverage=dict(unmatched_depth_counts={2:3,10:4}))
        fingerprint=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        sealed=json.loads(json.dumps(dict(body,payload_sha256=fingerprint)))
        before=deepcopy(sealed)
        validation=audit_module.verify_original_v1_payload(sealed)
        self.assertFalse(validation['plain_string_key_hash_passed'])
        self.assertTrue(validation['typed_v1_hash_passed'])
        self.assertEqual(sealed,before)
        sealed['coverage']['unmatched_depth_counts']['2']=7
        self.assertFalse(audit_module.verify_original_v1_payload(sealed)['typed_v1_hash_passed'])

    def test_separate_native_reduction_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);current=root/'requests'/'request';current.mkdir(parents=True)
            def write(path,value):path.write_text(json.dumps(value))
            response=[9,1,2,3,4,5,6,151645]
            blocks=[]
            for start in (0,2):
                blocks.append(dict(output_start=start,anchor_token=response[start],draft_tokens=[response[start+1]]+[99]*14,
                    target_argmax=[response[start+1]]+[99]*15,positions=list(range(start+1,start+17)),accepted_draft_tokens=1,
                    native_accepted_draft_tokens=1,committed_progress=2,committed_tokens=response[start+1:start+3],finished=False,stop_clipped=False))
            parent=dict(response_ids=response,stop_token_ids=[151645,151643],metadata=dict(trace=blocks))
            write(current/'PARENT.json',parent)
            starts=[];margins={};outcomes=[]
            for ordinal,block in enumerate(blocks):
                start=block['output_start'];n4=deepcopy(block);n6=deepcopy(block)
                if ordinal==0:
                    for modified in (n4,n6):modified.update(draft_tokens=[99]*15,accepted_draft_tokens=0,native_accepted_draft_tokens=0,committed_progress=1,committed_tokens=[response[start+1]])
                for arm,value in (('K5',block),('K4',n4)):write(current/f'{ordinal:04d}_{arm}.json',dict(blocks=[value]))
                write(current/f'K6_{start:05d}.json',dict(blocks=[n6]))
                value=.125*(ordinal+1)
                margins[str(start)]=[dict(source_id='source',start_offset=start,slot=1,canonical_id=response[start+1],chosen_id=response[start+1],canonical_score=value,competitor_score=0.,competitor_id=0,margin=value,native_dtype='torch.bfloat16',qa_hash=audit_module.compact_hash([20261115,'source',start,1]))]
                starts.append(dict(source_id='source',start_offset=start,a4=n4['accepted_draft_tokens'],a5=1,valid_proposal_count=7-start,k4_first_error=True,margins=[dict(slot=1,margin=value)]))
                outcomes.append(dict(source_id='source',start_offset=start,a6=n6['accepted_draft_tokens'],valid_proposal_count=7-start,k6_first_error=True,direct_correct={str(j):p==y for j,(p,y) in enumerate(zip(n6['draft_tokens'],response[start+1:]),1)}))
            write(current/'MARGINS.json',dict(starts=margins))
            seal=build_pre_k6(starts);write(root/'PRE_K6.json',seal)
            write(root/'PLAN.json',dict(requests=[dict(source_id='source',request_id='request')]))
            write(root/'OUTCOMES.json',outcomes);write(root/'ANALYSIS.json',analyze_k6(seal,outcomes))
            samples=[]
            for rows in margins.values():
                for row in rows:
                    scores=torch.zeros(8,dtype=torch.bfloat16);scores[row['canonical_id']]=row['canonical_score']
                    samples.append(dict(row,native_scores=scores))
            samples.sort(key=lambda row:row['qa_hash'])
            qa_path=root/'QA.pt';torch.save(dict(rows=samples),qa_path)
            result=audit_module.audit(root,qa_path)
            self.assertEqual(result['failures'],[])
            self.assertEqual(result['primary_delta'],1.)
            self.assertEqual(result['margin_full_vocabulary_qa'],'checked')
            wrong=deepcopy(samples);wrong[0]['canonical_score']=.5
            torch.save(dict(rows=wrong),qa_path)
            self.assertEqual(audit_module.audit(root,qa_path)['status'],'failed')
            torch.save(dict(rows=samples),qa_path)
            altered=json.loads((root/'ANALYSIS.json').read_text());altered['primary']['delta']=0.
            write(root/'ANALYSIS.json',altered)
            self.assertEqual(audit_module.audit(root)['status'],'failed')


if __name__=='__main__': unittest.main()
