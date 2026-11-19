"""Independent arithmetic fixtures for paired full-trajectory endpoint reporting."""
import hashlib,unittest
from unittest.mock import patch
import numpy as np
from experiments.full_c_gsm8k import analysis as A, evaluation as E

def rows(dataset):
 result=[];contract=E.dataset_contract(dataset)
 for i in range(1024):
  for repeat in range(3):
   for method in E.METHODS:
    calls=15+i%7-({'O58':0,'KPLUS58':1,'KPLUS_C58':2}[method]);native=99-calls;q,r=divmod(native,calls)
    histogram={str(q):calls-r}
    if r:histogram[str(q+1)]=r
    result.append(dict(phase='scored',**contract,panel_ordinal=i,block_ordinal=i//64,request_id=f'eval{i}',
     source_id=f'{dataset}:test:{1023-i}',method=method,repeat=repeat,checkpoint_id='/'+method,physical_gpu_uuid='GPU-fixed',
     verification_calls=calls,native_accepted_draft_tokens=native,accepted_draft_tokens=native-1,
     committed_progress=99,output_tokens=100,native_acceptance_histogram=histogram,finish_reason='eos',cap_hit=False,
     validation_status='valid',valid_measurement=True,errors=[],request_latency_s=float(calls+repeat),
     native_generate_s=calls+repeat-.1,cache_maintenance_s=.1))
 return result

class AnalysisTests(unittest.TestCase):
 def test_exact10000_draws_seeds_lexical_request_order_all_three_ratios(self):
  for dataset,seed in [('gsm8k',20261118),('mathqa',20261119)]:
   data=rows(dataset);result=A.summarize(data,dataset=dataset)
   draws=np.random.Generator(np.random.PCG64(seed)).integers(0,1024,size=(10000,1024))
   self.assertEqual(result['bootstrap']['draw_matrix_sha256'],hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest())
   paired=result['paired_requests'];self.assertEqual([x['request_id']for x in paired],sorted({r['request_id']for r in data}))
   self.assertNotEqual([x['source_id']for x in paired],sorted({r['source_id']for r in data}))
   for candidate,reference in A.CONTRASTS:
    effect=result['comparisons'][candidate+'_vs_'+reference]
    c=np.array([r[candidate]['verification_calls']for r in paired]);r=np.array([x[reference]['verification_calls']for x in paired])
    self.assertEqual(effect['call_reduction']['point'],1-c.sum()/r.sum())
    expected=1-c[draws].sum(1)/r[draws].sum(1)
    np.testing.assert_array_equal(effect['call_reduction']['ci95'],np.quantile(expected,[.025,.975],method='linear'))
    cn=99-c;rn=99-r
    self.assertEqual(effect['acceptance_improvement']['point'],(cn.sum()/c.sum())/(rn.sum()/r.sum())-1)
    clipped=effect['clipped_acceptance'];self.assertEqual(clipped['candidate_tokens_per_call'],(cn-1).sum()/c.sum())
    self.assertNotEqual(clipped['candidate_tokens_per_call'],cn.sum()/c.sum())
   self.assertEqual(result['comparisons']['KPLUS_C58_vs_KPLUS58']['net_calls_saved'],1024)
   self.assertNotIn('decision',result);self.assertEqual(len(result['block_views']),48)
 def test_zero_calls_all_draws_undefined_without_epsilon(self):
  data=rows('mathqa')
  for row in data:row.update(verification_calls=0,native_accepted_draft_tokens=0,accepted_draft_tokens=0,
   committed_progress=0,output_tokens=1,native_acceptance_histogram={})
  with patch.object(A,'BOOTSTRAP_DRAWS',64):result=A.summarize(data,dataset='mathqa')
  for effect in result['comparisons'].values():
   self.assertEqual(effect['call_reduction'],dict(point=None,ci95=None,undefined_draws=64))
   self.assertEqual(effect['clipped_acceptance']['relative_change'],dict(point=None,ci95=None,undefined_draws=64))
  self.assertEqual(result['method_totals']['O58']['zero_call_requests'],1024)
 def test_caps_and_pooled_request_mean_latency(self):
  data=rows('gsm8k')
  for row in data:
   if row['panel_ordinal']==0:row.update(verification_calls=1024,native_accepted_draft_tokens=1023,accepted_draft_tokens=1023,
    committed_progress=2047,output_tokens=2048,native_acceptance_histogram={'0':1,'1':1023},finish_reason='length',cap_hit=True)
   if row['method']=='KPLUS_C58':row['request_latency_s']*=.1 if row['panel_ordinal']==0 else 2
  with patch.object(A,'BOOTSTRAP_DRAWS',64):result=A.summarize(data,dataset='gsm8k')
  self.assertEqual(result['method_totals']['KPLUS_C58']['cap_hit_requests'],1)
  totals=result['method_totals'];effect=result['comparisons']['KPLUS_C58_vs_KPLUS58']
  self.assertEqual(effect['time_reduction']['point'],1-totals['KPLUS_C58']['request_latency_s']/totals['KPLUS58']['request_latency_s'])
  self.assertEqual(effect['evidence']['time_reduction'],'SUPPORTED_LOSS')
 def test_three_way_evidence_no_minimum_or_equivalence(self):
  for bounds,want in [([1e-12,2e-12],'SUPPORTED_GAIN'),([-2e-12,-1e-12],'SUPPORTED_LOSS'),([-.1,.1],'DIRECTION_UNRESOLVED'),([0,.1],'DIRECTION_UNRESOLVED'),(None,'DIRECTION_UNRESOLVED')]:
   self.assertEqual(A.evidence_label({'ci95':bounds}),want)
 def test_missing_repeat_and_native_accounting_rejected(self):
  data=rows('gsm8k')
  with self.assertRaisesRegex(ValueError,'exactly three'):A.reduce_requests(data[1:],dataset='gsm8k')
  for row in data:
   if row['panel_ordinal']==0:row['native_accepted_draft_tokens']+=1
  with self.assertRaisesRegex(ValueError,'identities'):A.reduce_requests(data,dataset='gsm8k')
  with self.assertRaisesRegex(ValueError,'Only valid'):A.reduce_requests(rows('mathqa'),dataset='gsm8k')

if __name__=='__main__':unittest.main()
