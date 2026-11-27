import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from collections import Counter

from experiments.continuation_repair_fullrun import serving as s
from experiments.continuation_repair_fullrun import analysis as a


def requests(n=1024,offset=0):
    return [dict(request_id=f'r{i+offset:04d}',source_id=f's{i+offset}',request_index=i,panel_ordinal=i,
        prompt_ids=[i+offset+1],stop_token_ids=[151645,151643],max_new_tokens=2048) for i in range(n)]


def rows_fixture(n=4):
    rows=[]
    for i in range(n):
        for method in s.METHODS:
            for repeat in range(3):
                calls=3+(i%2)
                rows.append(dict(request_id=f'r{i}',source_id=f's{i}',method=method,phase='scored',repeat=repeat,
                    checkpoint_id='/'+method,physical_gpu_uuid='A',valid_measurement=True,errors=[],
                    request_latency_s=1.+i*.2+(method=='TAIL')*.0005,native_generate_s=.9+i*.2,
                    cache_maintenance_s=.001,output_tokens=10,verification_calls=calls,
                    accepted_draft_tokens=9-calls,native_accepted_draft_tokens=9-calls,committed_progress=9,finish_reason='eos'))
    return rows


class ScheduleTests(unittest.TestCase):
    def test_balanced_schedule_and_counts(self):
        entries=s.schedule(requests());balance=s.schedule_summary(entries)
        self.assertEqual(balance['activations'],48);self.assertEqual(balance['scored_generations'],12288)
        self.assertEqual(balance['warmup_generations'],96)
        slots=[slot for entry in entries for slot,_ in s.slots_for(entry,requests(),requests(2,2000))]
        self.assertEqual(len(slots),12384)
        positions=Counter((e['method'],e['position']) for e in entries)
        self.assertEqual(set(positions.values()),{3})
    def test_order_tampering_and_wrong_panel_rejected(self):
        entries=s.schedule(requests());entries[0]['method']='F'
        with self.assertRaises(ValueError):s.schedule_summary(entries)
        with self.assertRaises(ValueError):s.schedule(requests(1023))
    def test_complete_accounting_exact_slot_coverage(self):
        panel,warm=requests(),requests(2,2000)
        rows=[dict(slot,request_id=request['request_id']) for entry in s.schedule(panel) for slot,request in s.slots_for(entry,panel,warm)]
        self.assertEqual(a.complete_accounting(rows)['valid'],12384)
        with self.assertRaises(ValueError):a.complete_accounting(rows[:-1])
        rows[2]['request_id']=rows[3]['request_id']
        with self.assertRaises(ValueError):a.complete_accounting(rows)


class AnalysisTests(unittest.TestCase):
    def test_shared_bootstrap_sign_and_secondary_scope(self):
        result=a.summarize(rows_fixture(),expected_requests=4)
        d=result['contrasts']['TAIL_vs_BASE']['request_latency_s_degradation_pct']
        self.assertGreater(d['value'],0)
        self.assertLess(d['value'],.1)
        self.assertEqual(result['bootstrap']['seed'],20261126)
        self.assertEqual(result['bootstrap']['draws'],10000)
        self.assertEqual(set(result['contrasts']),{'TAIL_vs_BASE','BASE_vs_AREPO','TAIL_vs_AREPO','BASE_vs_F','TAIL_vs_F'})
        self.assertEqual(result['totals']['BASE']['verification_calls'],14)
        self.assertEqual(result['totals']['BASE']['output_tokens'],40)
    def test_practical_ci_and_speed_rules_are_separate(self):
        def decision(point,upper,**kwargs):
            report={'contrasts':{'TAIL_vs_BASE':{'request_latency_s_degradation_pct':dict(value=point,ci95=[-.2,upper])}}}
            return a.selection_decision(report,complete=True,**kwargs)
        result=decision(.1,.2)
        self.assertEqual(result['selected'],'TAIL');self.assertTrue(result['practical_rule_met'])
        self.assertFalse(result['degradation_bound_supported']);self.assertFalse(result['speed_improvement_supported'])
        self.assertEqual(decision(.10000000001,.11)['selected'],'BASE')
        self.assertTrue(decision(.09,.1)['degradation_bound_supported'])
        self.assertFalse(decision(-.1,0)['speed_improvement_supported'])
        self.assertTrue(decision(-.1,-1e-9)['speed_improvement_supported'])
        self.assertFalse(decision(0,.1,identical_weights=True)['mechanism_benefit_established'])
        self.assertEqual(decision(0,.2,identical_weights=True)['selected'],'TAIL')
    def test_invalid_or_changed_repeat_stops(self):
        rows=rows_fixture();rows[0]['verification_calls']+=1
        with self.assertRaises(ValueError):a.summarize(rows,expected_requests=4)
        rows=rows_fixture();rows[0]['valid_measurement']=False
        with self.assertRaises(ValueError):a.summarize(rows,expected_requests=4)
    def test_no_call_count_adoption_gate(self):
        report=a.summarize(rows_fixture(),expected_requests=4)
        report['contrasts']['TAIL_vs_BASE']['verification_call_reduction_pct']['value']=-1
        result=a.selection_decision(report,complete=True)
        self.assertEqual(result['selected'],'TAIL');self.assertFalse(result['call_count_is_adoption_gate'])


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'evaluation'
        self.panel,self.warm=requests(),requests(2,2000)
        self.plan=dict(execution_commit='abc',devices={'A':'GPU-A'},canonical_provenance={})
        self.refs={r['request_id']:{} for r in self.panel};self.wrefs={r['request_id']:{} for r in self.warm}
        self.checkpoints={m:'/'+m for m in s.METHODS}
    def tearDown(self):self.temp.cleanup()
    def execute(self,**overrides):
        kwargs=dict(activate=lambda *args:None,boundary=lambda slot:None,ledger=None)
        kwargs.update(overrides)
        return s.execute(self.plan,self.root,self.panel,self.warm,self.refs,self.wrefs,self.checkpoints,**kwargs)
    def seed_protocol(self):
        with patch.object(s,'check_reference'),self.assertRaisesRegex(RuntimeError,'before activation'):
            self.execute(boundary=lambda slot:(_ for _ in ()).throw(RuntimeError('before activation')))
    def test_pending_failure_never_reexecutes(self):
        self.seed_protocol();status=s.read(self.root/'STATUS.json');status['pending']={'slot':'unknown'}
        s.write_json(self.root/'STATUS.json',status)
        with patch.object(s,'check_reference'),self.assertRaisesRegex(ValueError,'reconciliation'):
            self.execute(resume=True)
    def test_resume_only_last_two_slots_and_two_retry_warmups(self):
        self.seed_protocol()
        planned=[(slot,r) for e in s.schedule(self.panel) for slot,r in s.slots_for(e,self.panel,self.warm)]
        def measured(record,request,checkpoint,canonical,slot):
            return dict(slot,request_id=request['request_id'],checkpoint_id=checkpoint,source_id=request['source_id'],
                output_tokens=10,verification_calls=3,accepted_draft_tokens=6,native_accepted_draft_tokens=6,committed_progress=9,
                finish_reason='eos',request_latency_s=1.,native_generate_s=.9,cache_maintenance_s=.001),[]
        retained=[]
        for slot,request in planned[:-2]:
            row,_=measured({},request,self.checkpoints[slot['method']],{},slot)
            row.update(physical_gpu_uuid='GPU-A',valid_measurement=True,errors=[],raw_path=str(self.root/'raw'/(s.label_for(slot)+'.json')))
            retained.append(row)
        (self.root/'MEASUREMENTS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in retained))
        status=s.read(self.root/'STATUS.json');status.update(attempted=len(retained),returned=len(retained),completed_valid=len(retained))
        s.write_json(self.root/'STATUS.json',status)
        class Backend:
            def generate(self,*args,**kwargs):return {}
        class Ledger:
            def __init__(self):self.calls=Counter()
            def call(self,counts,label,function,*args,**kwargs):
                self.calls.update(counts);return function(*args,**kwargs)
        ledger=Ledger();oldread=s.read
        def read(path):return {} if Path(path).parent.name=='raw' else oldread(path)
        def activate(*args,**kwargs):
            s.append_json(self.root/'ACTIVATIONS.jsonl',{'publication':{}})
            return Backend()
        with patch.object(s,'check_reference'),patch.object(s,'measure',side_effect=measured),patch.object(s,'read',side_effect=read),patch.object(s,'_activation',side_effect=activate):
            result=self.execute(resume=True,ledger=ledger)
        self.assertEqual(result['completed_valid'],12384)
        self.assertEqual(ledger.calls,Counter(scored_generation=2,resume_warmup_generation=2))
        self.assertEqual(len((self.root/'MEASUREMENTS.jsonl').read_text().splitlines()),12384)
        self.assertEqual(len((self.root/'RETRY_MEASUREMENTS.jsonl').read_text().splitlines()),2)
    def test_existing_unbound_measurements_rejected(self):
        self.root.mkdir();(self.root/'MEASUREMENTS.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'historical protocol'):self.execute(resume=True)
    def test_malformed_return_is_preserved_invalid_not_unknown(self):
        class Backend:
            def generate(self,*args,**kwargs):return {'incomplete':'native return'}
        class Ledger:
            def call(self,counts,label,function,*args,**kwargs):return function(*args,**kwargs)
        with patch.object(s,'check_reference'),patch.object(s,'_activation',return_value=Backend()),patch.object(s,'measure',side_effect=KeyError('metadata')):
            with self.assertRaisesRegex(ValueError,'malformed_returned_record'):
                self.execute(ledger=Ledger())
        status=s.read(self.root/'STATUS.json')
        self.assertEqual((status['attempted'],status['returned'],status['invalid'],status['unknown']),(1,1,1,0))
        row=json.loads((self.root/'MEASUREMENTS.jsonl').read_text())
        self.assertFalse(row['valid_measurement']);self.assertTrue(Path(row['raw_path']).exists())


class CanonicalQualificationTests(unittest.TestCase):
    def test_loading_in_canonical_phase_is_not_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for phase,loading,activation in [('canonicals',7.,2.),('evaluation',0.,3.)]:
                (root/phase).mkdir()
                s.append_json(root/phase/'ACTIVATIONS.jsonl',dict(loading_s=loading,publication={'activation_s':activation}))
            s.write_json(root/'canonicals/raw/one.json',dict(metadata=dict(request_latency_s=4.,native_generate_s=3.9,capture={'cache_maintenance_s':.1})))
            result=s.cost_totals(root,[dict(phase='warmup',request_latency_s=2.),dict(phase='scored',request_latency_s=5.)])
        self.assertEqual(result['loading_s'],7.)
        self.assertEqual(result['canonical_activation_s'],2.);self.assertEqual(result['activation_s'],3.)
        self.assertEqual(result['canonical_request_latency_s'],4.);self.assertEqual(result['scored_request_latency_s'],5.)
        self.assertEqual(result['actual_canonical_activations'],1);self.assertEqual(result['resume_warmup_generations'],0)
    def test_qualification_exact_eight_and_native_F_first(self):
        class Backend:
            def generate(self,*args,**kwargs):return dict(response_ids=[5])
        class Ledger:
            def __init__(self):self.calls=Counter()
            def call(self,counts,label,function,*args,**kwargs):
                self.calls.update(counts);return function(*args,**kwargs)
        ledger=Ledger();methods=[]
        def activate(callback,checkpoint,method,slot,root):methods.append(method);return Backend()
        def measure(*args):
            slot=args[-1]
            return dict(slot,output_tokens=2,verification_calls=1,accepted_draft_tokens=0,
                        native_accepted_draft_tokens=0,committed_progress=1,finish_reason='eos'),[]
        with tempfile.TemporaryDirectory() as directory,patch.object(s,'_activation',side_effect=activate),patch.object(s,'measure',side_effect=measure),patch.object(s,'make_reference',return_value={'native':'F'}):
            result=s.qualify_serving({},Path(directory)/'qual',requests(1)[0],{m:'/'+m for m in s.METHODS},activate=None,boundary=lambda slot:None,ledger=ledger)
        self.assertEqual(ledger.calls,Counter(generation=8))
        self.assertEqual(methods,['F','BASE','TAIL','AREPO']*2)
        self.assertEqual(result['status'],'passed');self.assertFalse(result['target_only_AR'])
    def test_uncertain_canonical_is_not_replayed(self):
        class Backend:
            def generate(self,*args,**kwargs):raise RuntimeError('native interrupted')
        class Ledger:
            def __init__(self):self.calls=0
            def call(self,counts,label,function,*args,**kwargs):self.calls+=1;return function(*args,**kwargs)
        ledger=Ledger();plan=dict(execution_commit='abc',devices={'A':'gpu'},canonical_provenance={})
        with tempfile.TemporaryDirectory() as directory,patch.object(s,'_activation',return_value=Backend()):
            root=Path(directory)/'canonical'
            with self.assertRaisesRegex(RuntimeError,'native interrupted'):
                s.execute_canonicals(plan,root,requests(),'/F',activate=None,boundary=lambda slot:None,ledger=ledger)
            status=s.read(root/'STATUS.json')
            self.assertEqual(status['attempted'],1);self.assertEqual(status['unknown'],1)
            with self.assertRaisesRegex(ValueError,'not be regenerated'):
                s.execute_canonicals(plan,root,requests(),'/F',activate=None,boundary=lambda slot:None,ledger=ledger,resume=True)
        self.assertEqual(ledger.calls,1)
    def test_malformed_canonical_return_is_preserved_invalid(self):
        class Backend:
            def generate(self,*args,**kwargs):return {'incomplete':'native return'}
        class Ledger:
            def call(self,counts,label,function,*args,**kwargs):return function(*args,**kwargs)
        plan=dict(execution_commit='abc',devices={'A':'gpu'},canonical_provenance={})
        with tempfile.TemporaryDirectory() as directory,patch.object(s,'_activation',return_value=Backend()):
            root=Path(directory)/'canonical'
            with self.assertRaisesRegex(ValueError,'malformed_returned_record'):
                s.execute_canonicals(plan,root,requests(),'/F',activate=None,boundary=lambda slot:None,ledger=Ledger())
            status=s.read(root/'STATUS.json')
            self.assertEqual((status['attempted'],status['returned'],status['invalid'],status['unknown']),(1,1,1,0))
            row=json.loads((root/'VALIDITY.jsonl').read_text())
            self.assertFalse(row['valid_measurement']);self.assertTrue(Path(row['raw_path']).exists())


if __name__=='__main__':unittest.main()
