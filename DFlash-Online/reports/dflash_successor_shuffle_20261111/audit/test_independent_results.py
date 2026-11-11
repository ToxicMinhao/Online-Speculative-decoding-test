"""Synthetic CPU checks only: these are not measurements of the study models."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest

import numpy as np
import independent_results as audit


def repeats():
    rows = []
    for ordinal, arm, repeat, block, position in audit.schedule(8):
        base = 6+ordinal
        calls = {'K':base,'TRUE':base+(1 if ordinal%2 == 0 else -1),'SHUFFLED':base+2}[arm]
        latency = calls*(1+0.1*repeat)+ordinal/10
        rows.append(dict(panel_ordinal=ordinal,request_id='request_'+str(ordinal),method=arm,repeat=repeat,
            block_ordinal=block,position=position,seed=17,event=5,dataset='gsm8k',phase='final',
            verification_calls=calls,accepted_draft_tokens=99-calls,native_accepted_draft_tokens=99-calls,
            committed_progress=99,output_tokens=100,request_latency_s=latency,
            native_generate_s=latency*.9,cache_maintenance_s=latency*.1,
            checkpoint_id='/checkpoints/'+arm,finish_reason='eos',cap_hit=False,
            first_round_present=True,first_round_start_digest=str(ordinal)*64,
            first_round_output_start=0,first_round_accepted_prefix_tokens=2,first_round_raw_prefix_tokens=2,
            valid_measurement=True,validation_status='valid',errors=[],structural_errors=[]))
    return rows


def manipulation(response_id='synthetic'):
    # Include mixed bits, single-class groups, a singleton, terminal error, and an empty block.
    blocks = [dict(anchor_offset=16*i,first_error_position=pos,eligible_successor=pos<15,
                   true_priority=bit,assigned_priority=bit,donor_anchor_offset=16*i)
              for i,(pos,bit) in enumerate([(3,2),(3,1),(3,2),(3,1),(7,2),(7,2),(8,1),(15,1)])]
    blocks.append(dict(anchor_offset=128,first_error_position=None,eligible_successor=False,
                       true_priority=None,assigned_priority=None,donor_anchor_offset=None))
    strata = []
    for pos in (3,7,8):
        chosen = [b for b in blocks if b['first_error_position'] == pos]
        donors = sorted(chosen,key=lambda b:(hashlib.sha256(
            ('[20261111,%s,%d,%d]'%(json.dumps(response_id),pos,b['anchor_offset'])).encode()).digest(),b['anchor_offset']))
        for receiver,donor in zip(chosen,donors):
            receiver['assigned_priority'] = donor['true_priority']
            receiver['donor_anchor_offset'] = donor['anchor_offset']
        n,k = len(chosen),sum(b['true_priority'] == 2 for b in chosen)
        strata.append(dict(position=pos,n=n,k=k,recipients=[b['anchor_offset'] for b in chosen],
            donors=[b['anchor_offset'] for b in donors],true_high_count=k,assigned_high_count=k,
            flipped_bits=sum(b['true_priority'] != b['assigned_priority'] for b in chosen),
            uniform_expected_flips=2*k*(n-k)/n))
    row = dict(schema='kplus_successor_shuffle_v1',status='passed',seed=20261111,identity=False,
        response_id=response_id,blocks=blocks,anchor_offsets=[b['anchor_offset'] for b in blocks],strata=strata,
        class_counts_exact=True,mass_preserved=True,selection_preserved=True,denominator_preserved=True,
        eligible_errors=7,ineligible_errors=1,mixed_strata=1,flipped_bits=sum(s['flipped_bits'] for s in strata),
        uniform_expected_flips=2.0,projection_calls=2,projection_completed=2,projection_s=.123456789,
        assignment_s=.00123456789,reassigned_weight=1.0,coefficient_l1_weight=2.0,
        selected_weight=5.0,denominator=5.000001,true_loss=.3,assigned_loss=.3125)
    for key in ('W1','W2','coefficient1','coefficient2','error_mass'):
        row['true_'+key] = row['assigned_'+key] = 1.0
    sums = {'shuffle_response_count':1,'kplus_S':5.0,'kplus_first_error_weight_after':1.0}
    for key in (*audit.MANIP_TOTALS,'coefficient_l1_weight','true_loss','assigned_loss'):
        sums['shuffle_'+key] = struct.unpack('f',struct.pack('f',row[key]))[0]
    for key in ('W1','W2','coefficient1','coefficient2'): sums['kplus_'+key] = 1.0
    return dict(record_id=response_id,loss=.3125,sum_metrics=sums,successor_shuffle=row)


class AuditTests(unittest.TestCase):
    def test_raw_pairing_counts_and_manual_bootstrap(self):
        result = audit.compute([audit.normalize(r) for r in repeats()],8)
        effect = result['comparisons']['TRUE_vs_SHUFFLED']['metrics']['verification_calls']
        self.assertEqual(result['timing_rows'],48)
        self.assertEqual(effect['candidate_sum'],76)
        self.assertEqual(effect['reference_sum'],92)
        self.assertEqual(effect['point_pct'],100*(1-76/92))
        # Separate draw-at-a-time calculation and explicit linear order-statistic interpolation.
        rng = np.random.Generator(np.random.PCG64(20261111))
        draws = []
        for _ in range(10000):
            sample = rng.integers(0,8,size=8).tolist()
            candidate = sum(6+i+(1 if i%2 == 0 else -1) for i in sample)
            reference = sum(8+i for i in sample)
            draws.append(100*(1-candidate/reference))
        draws.sort()
        for probability,actual in zip((.025,.975),effect['ci95_pct']):
            position = probability*9999
            lower = int(position)
            expected = draws[lower]+(position-lower)*(draws[lower+1]-draws[lower])
            self.assertEqual(actual,expected)
        self.assertNotAlmostEqual(effect['point_pct'],sum(100*(1-(6+i+(1 if i%2==0 else -1))/(8+i)) for i in range(8))/8)
        self.assertEqual(result['classification']['outcome'],'informative_placement_supported')
        self.assertAlmostEqual(result['paired_requests'][0]['K']['request_latency_s'],6.3)

    def test_incomplete_duplicate_and_order_rejected(self):
        raw = repeats()
        for rows in (raw[:-1],raw+[raw[0]],raw[1:]+raw[:1],raw[:-1]+[raw[0]]):
            with self.assertRaises(AssertionError): audit.compute(rows,8)

    def test_changed_repeat_rejected(self):
        raw = repeats(); raw[-1]['verification_calls'] += 1
        with self.assertRaises(AssertionError): audit.compute(raw,8)

    def test_seed_eos_is_included(self):
        raw = repeats()
        for row in raw:
            if row['panel_ordinal'] == 0:
                row.update(verification_calls=0,accepted_draft_tokens=0,native_accepted_draft_tokens=0,
                    committed_progress=0,output_tokens=1,first_round_present=False,
                    first_round_start_digest=None,first_round_output_start=None,
                    first_round_accepted_prefix_tokens=0,first_round_raw_prefix_tokens=0)
        result = audit.compute([audit.normalize(r) for r in raw],8)
        self.assertEqual(result['request_count'],8)
        self.assertEqual(result['comparisons']['TRUE_vs_K']['first_round']['seed_eos_requests'],1)

    def test_invalid_measurement_rejected(self):
        for key,value in (('valid_measurement',False),('errors',['bad']),('request_latency_s',float('nan'))):
            row = repeats()[0]; row[key] = value
            with self.assertRaises(AssertionError): audit.normalize(row)

    def test_publication_count(self):
        self.assertEqual(len(audit.publication_methods(audit.schedule(256))),161)
        self.assertEqual(len(audit.publication_methods(audit.schedule(8))),5)

    def test_hash_map_and_recorded_fp32_metrics(self):
        row = manipulation()
        self.assertEqual(audit.check_manipulation(row)['eligible_errors'],7)
        changed = copy.deepcopy(row)
        changed['successor_shuffle']['strata'][0]['donors'].reverse()
        with self.assertRaises(AssertionError): audit.check_manipulation(changed)
        changed = copy.deepcopy(row)
        changed['sum_metrics']['shuffle_projection_s'] = row['successor_shuffle']['projection_s']
        with self.assertRaises(AssertionError): audit.check_manipulation(changed)

    def test_partial_work_never_zero_imputed(self):
        rows = [dict(kind='optimizer_steps',phase='started',ordinal=1)]
        with self.assertRaises(AssertionError): audit.check_ledger(rows,{'optimizer_steps':1})
        rows.append(dict(kind='optimizer_steps',phase='completed',ordinal=1))
        self.assertEqual(audit.check_ledger(rows,{'optimizer_steps':1})['unknown'],{'optimizer_steps':0})

    def test_full_training_counts_and_permuted_second_pass(self):
        records = []
        order = list(range(128))+list(reversed(range(128)))
        for i,request in enumerate(order):
            observation = manipulation('response_'+str(request))
            observation.update(pass_index=i//128,presentation_ordinal=i,sampled_anchors=[{'positions':[i,i+1]}])
            records.append(dict(record_id=observation['record_id'],pass_=i//128,
                optimizer_step_after=512+(i+1)//2,optimizer_stepped=bool(i%2),
                learning_rate=.0003 if i%2 else None,sampled_anchors=observation['sampled_anchors'],
                response_observation=observation))
            records[-1]['pass'] = records[-1].pop('pass_')
        expected = dict(optimizer_steps=128,response_forwards=256,response_backwards=256)
        work = []
        for kind,count in expected.items():
            for ordinal in range(1,count+1):
                work.extend(dict(kind=kind,ordinal=ordinal,phase=phase) for phase in ('started','completed'))
        work_summary = audit.check_ledger(work,expected)
        baseline = dict(features=['synthetic_packet'],records=records)
        results = {arm:dict(status='completed',method=arm,dataset='gsm8k',event_id='event_005',
            start_step=512,end_step=640,optimizer_steps=128,response_presentations=256,passes=2,
            successor_work=dict(ordinary_presentations=256,**expected),records=records,
            features=baseline['features'],weights_identity={'sha256':arm},work_ledger=work_summary)
            for arm in ('K','SHUFFLED')}
        rows = [r['response_observation']['successor_shuffle'] for r in records]
        summary = dict(status='passed',presentations=256,rows=rows,final_export_changed=True,
                       labels_changed=True,all_invariants_passed=True,actuation='active')
        summary.update({k:sum(r[k] for r in rows) for k in audit.MANIP_TOTALS})
        checkpoints = {a:dict(weights_identity={'sha256':a}) for a in audit.ARMS}
        ledgers = {a:work for a in ('K','SHUFFLED')}
        result = audit.check_training(results,baseline,summary,checkpoints,ledgers)
        self.assertEqual(result['main_optimizer_steps'],256)
        self.assertEqual(result['main_presentations'],512)
        self.assertEqual(result['TRUE_new_main_steps'],0)
        results['K']['optimizer_steps'] = 256
        with self.assertRaises(AssertionError): audit.check_training(results,baseline,summary,checkpoints,ledgers)

    def test_zero_cache_is_undefined_and_wrong_panel_fails(self):
        raw = repeats()
        for row in raw: row['cache_maintenance_s'] = 0
        result = audit.compute(raw,8)
        effect = result['comparisons']['TRUE_vs_SHUFFLED']['metrics']['cache_maintenance_s']
        self.assertIsNone(effect['point_pct'])
        self.assertIsNone(effect['ci95_pct'])
        self.assertEqual(effect['undefined_bootstrap_draws'],10000)
        with self.assertRaises(AssertionError): audit.compute(repeats(),7)

    def test_missing_input_writes_failure_without_results(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'AUDIT.json'
            process = subprocess.run([sys.executable,str(Path(audit.__file__)), '--root',folder,
                '--baseline',folder+'/absent_baseline.json','--output',str(output)],capture_output=True,text=True)
            self.assertEqual(process.returncode,1)
            result = json.loads(output.read_text())
            self.assertEqual(result['status'],'failed')
            self.assertFalse(result['inference_permitted'])
            self.assertNotIn('results',result)
            self.assertIn('FileNotFoundError',result['error'])


if __name__ == '__main__': unittest.main()
