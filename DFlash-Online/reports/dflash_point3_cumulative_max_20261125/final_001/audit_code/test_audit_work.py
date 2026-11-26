"""Portable, synthetic CPU tests; no private records or model runtime needed."""
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

spec = importlib.util.spec_from_file_location('audit_work',Path(__file__).with_name('audit_work.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def readout_fixture():
    margins, minima, multiplicities = [2.,2.,-1.,1.], [0,0,2,2], [2,0,2,0]
    positions = [dict(position=i+1,margin=m,bottleneck_position=minima[i]+1,
        bottleneck_multiplicity=multiplicities[i],tied_bottleneck_prefix=i==1,
        assessment_contribution_fp32=(4-i)*math.log1p(math.exp(-m))/math.log(2),
        max_assessment_contribution_fp32=math.log1p(math.exp(-margins[minima[i]]))/math.log(2))
        for i,m in enumerate(margins)]
    summed=math.fsum(p['assessment_contribution_fp32'] for p in positions)
    maximized=math.fsum(p['max_assessment_contribution_fp32'] for p in positions)
    block=dict(origin=dict(m=4,K=4,e=3,labels=[0,1,1,1,1],prefix=[False,True,True,False,False]),
        proposals=[0,1,1,2,1],positions=positions,assessment_sum_fp64=summed,
        max_assessment_sum_fp64=maximized,metrics=dict(matching_prefix_length=2,
        fixed_K_shortfall=2,raw_repair=0,useful_repair=0,damaged_prefix_positions=0,
        damaged_prefix_blocks=0,surviving_followers=1,reachable_followers=0))
    response=dict(blocks=[block],nonempty_blocks=1,
        assessments=dict(SUM=dict(fp64=summed),MAX=dict(fp64=maximized)))
    empty=dict(blocks=[],nonempty_blocks=0,assessments=dict(SUM=dict(fp64=0.),MAX=dict(fp64=0.)))
    return dict(status='passed',state_preserved=True,optimizer_steps=0,backwards=0,
        teacher_forwards=0,responses=[response,empty],
        assessments=dict(SUM=dict(fp64=summed/2),MAX=dict(fp64=maximized/2)))


def numeric_fixture():
    fp32=lambda value:struct.unpack('f',struct.pack('f',value))[0]
    pools,rows=[],[]
    for count in (13,15):
        beta=[fp32(math.exp(-j/7)) for j in range(count)]
        scores=[fp32((j+1)*.001) for j in range(count)]
        offsets=[x-min(scores) for x in scores]
        mean=math.fsum(x*w for x,w in zip(offsets,beta))/math.fsum(beta)
        centered=[x-mean for x in offsets]; maximum=max(abs(x) for x in centered)
        coefficients=[fp32(b*(1+.25*c/maximum)) for b,c in zip(beta,centered)]
        tolerance=[2e-7+2e-6*abs(b) for b in beta]
        alignment=math.fsum((a-b)*c for a,b,c in zip(coefficients,beta,centered))
        allowance=math.fsum(t*abs(c) for t,c in zip(tolerance,centered))
        pools.append(dict(beta=beta,reference_coefficients_fp32=coefficients,
            reference_scores_fp32=scores,positions=list(range(1,count+1)),
            fast_coefficients_fp32=[coefficients,coefficients],
            gate_b=dict(per_position_bound=[.025*b+t for b,t in zip(beta,tolerance)]),
            gate_c=dict(reference_alignment=alignment,tolerance=allowance,
                        required_alignment=.9*alignment-allowance)))
        rows.extend(dict(reference_score_fp64=s,primary_dot_fp64=-2.5*s,denominator=1.25) for s in scores)
    invariants=[[dict(passed=True,checks=dict(fixture=True),failures=[]) for _ in range(2)] for _ in range(2)]
    gates=dict(status='passed',reference_backward_count=28,old_hidden_state_gate='failed_retained_diagnostic',
        repeat_exact=dict(scores=True,allocations=True),selection=dict(selected_before_scores=True),
        selected_pool_receipts=pools,invariants=invariants)
    return gates,dict(rows=rows)


def raw_fixture():
    trace=[dict(output_start=0,anchor_token=10,committed_progress=2,committed_tokens=[20,30],
        draft_tokens=[20,99],target_argmax=[20,30,70],raw_accepted_draft_tokens_before_stop=1,
        accepted_draft_tokens=1,native_accepted_draft_tokens=1),
        dict(output_start=2,anchor_token=30,committed_progress=1,committed_tokens=[40],
        draft_tokens=[40,99],target_argmax=[40,70,80],raw_accepted_draft_tokens_before_stop=1,
        accepted_draft_tokens=1,native_accepted_draft_tokens=0)]
    reference=dict(verified=True,reference_kind='frozen_DFlash',prompt_ids=[1,2],
        response_ids=[10,20,30,40],finish_reason='eos',stop_token_ids=[40],max_new_tokens=2048)
    record={k:deepcopy(v) for k,v in reference.items() if k not in ('verified','reference_kind')}
    record.update(checkpoint_id='fixture-checkpoint',metadata=dict(block_size=16,context_limit=4096,
        feedback_capture_in_timed_request=False,runtime='pinned_SGLang_FA3',
        cache_policy='idle_full_flush_before_every_request_and_publication',trace=trace,
        verification_calls=2,request_latency_s=1.,native_generate_s=.8,capture=dict(cache_maintenance_s=.1)))
    row=dict(checkpoint_id='fixture-checkpoint',output_tokens=4,prompt_tokens=2,
        verification_calls=2,accepted_draft_tokens=2,native_accepted_draft_tokens=1,
        committed_progress=3,request_latency_s=1.,native_generate_s=.8,cache_maintenance_s=.1)
    return row,record,reference


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'CONTRACT.json').write_text('{}')
        self.audit=module.Audit(self.root)

    def numeric_files(self):
        path=self.root/'qualification/native/max_numerics';path.mkdir(parents=True)
        gates,rows=numeric_fixture()
        (path/'QUALIFICATION.json').write_text(json.dumps(gates))
        (path/'REFERENCE_PROGRESS.json').write_text(json.dumps(rows))
        return path

    def test_max_ties_and_empty_response_average(self):
        self.assertEqual(self.audit.readout(readout_fixture()),1)

    def test_wrong_tie_branch_and_multiplicity_are_rejected(self):
        for key,value in (('bottleneck_position',2),('bottleneck_multiplicity',1)):
            row=readout_fixture();row['responses'][0]['blocks'][0]['positions'][0][key]=value
            with self.assertRaises(AssertionError):self.audit.readout(row)

    def test_empty_response_cannot_renormalize_other_response(self):
        row=readout_fixture();row['assessments']['MAX']['fp64']*=2
        with self.assertRaisesRegex(AssertionError,'equal response average'):self.audit.readout(row)

    def test_coefficient_and_alignment_gates(self):
        self.numeric_files();self.audit.numerical_gates()

    def test_incorrect_oracle_normalization_is_rejected(self):
        path=self.numeric_files()/'REFERENCE_PROGRESS.json';row=json.loads(path.read_text())
        row['rows'][0]['reference_score_fp64']*=2;path.write_text(json.dumps(row))
        with self.assertRaisesRegex(AssertionError,'normalization exactly once'):self.audit.numerical_gates()

    def test_missing_evidence_is_incomplete(self):
        self.audit.section('missing',lambda:self.audit.load('absent.json'))
        self.assertEqual(self.audit.sections['missing']['status'],'incomplete')

    def test_relative_root_is_normalized(self):
        audit=module.Audit(os.path.relpath(self.root,Path.cwd()))
        self.assertEqual(audit.load(audit.root/'CONTRACT.json'),{})

    def test_scientific_identity_excludes_only_timing_and_event_label(self):
        case=dict(rows=[dict(training_s=1.,response_observation=dict(event_id='old',loss=.2))],
            gradient_identity='gradient',endpoint='endpoint',native_preclip_norm=1.,adam_completed=1)
        changed=deepcopy(case);changed['rows'][0]['training_s']=2.
        changed['rows'][0]['response_observation']['event_id']='new'
        self.assertEqual(module.scientific_case(case),module.scientific_case(changed))
        changed['rows'][0]['response_observation']['loss']=.3
        self.assertNotEqual(module.scientific_case(case),module.scientific_case(changed))

    def test_raw_tokens_and_verified_prefix(self):
        row,record,reference=raw_fixture();self.audit.raw_record(row,record,reference)
        record['response_ids'][1]+=1
        with self.assertRaisesRegex(AssertionError,'raw canonical response_ids'):
            self.audit.raw_record(row,record,reference)


if __name__=='__main__':unittest.main()
