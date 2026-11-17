from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.c_paired_confirmation import preflight as P


class Tokens:
    def __init__(self,values):self.values=values
    def __getitem__(self,index):return self
    def tolist(self):return self.values


def packet_fixture():
    request=dict(request_id='r',prompt_ids=[1,2])
    capture=dict(record_id='r',checkpoint_id='origin',prompt_ids=[1,2],response_ids=[3,4])
    prep=dict(record_id='r',input_ids=[1,2,3,4],target_argmax_ids=[3,4],native_capture_validated=True,label_mismatches=[])
    packet=dict(record_id='r',prompt_length=2,input_ids=Tokens([1,2,3,4]))
    return packet,request,prep,capture


class Tests(unittest.TestCase):
    def test_runtime_alias_preserves_exact_value_checks(self):
        a={'serving':{'python_version':'3.12.14','torch':'2.9.1'}}
        b={'serving':{'python':'3.12.14','torch':'2.9.1'}}
        self.assertTrue(P.same_runtime_contract(a,b))
        self.assertIn('python_version',a['serving'])
        b['serving']['python']='3.12.15'
        self.assertFalse(P.same_runtime_contract(a,b))
        a['serving']['python']='3.12.15'
        with self.assertRaisesRegex(ValueError,'Contradictory'):P.same_runtime_contract(a,b)

    def test_common_packet_bound_to_original_native_producer(self):
        args=packet_fixture();P.packet_input_binding(*args,'origin')
        for field,value in [('checkpoint_id','another_arm'),('response_ids',[3,5]),('prompt_ids',[2,1])]:
            x=list(deepcopy(args));x[3][field]=value
            with self.assertRaisesRegex(ValueError,'producer/target'):P.packet_input_binding(*x,'origin')

    def test_feature_swapping_cannot_be_hidden_by_record_id(self):
        args=list(packet_fixture());args[0]['input_ids']=Tokens([1,2,3,5])
        with self.assertRaisesRegex(ValueError,'binding'):P.packet_input_binding(*args,'origin')

    def test_gsm_pool_cardinality_checked_before_packet_load(self):
        plan={'datasets':{'gsm8k':{'common_features':[],'common_feature_paths':[]}}}
        with self.assertRaisesRegex(ValueError,'cardinality'):P.inspect_common_gsm8k(plan,{'gsm8k':{'first_inputs':{'requests':[]}}})

    def test_mathqa_reuses_exact_typed_functional_gate_without_claiming_old_bytes(self):
        sub=dict(origin_state='state',origin_export='export',common_e5={'feature_paths':['f']},common_feature_paths=['f'],
                 historical_plan={'path':'old'},current_feature_manifest={'path':'features'},functional_replay={'path':'gate'})
        old=dict(origin_state='state',origin_export='export',common_e5=sub['common_e5'],runtime_contract={'x':1},
                 prerequisite={'feature_manifest':sub['current_feature_manifest']},event5_inputs='inputs')
        plan=dict(datasets={'mathqa':sub},runtime_contract={'x':1});inputs={'mathqa':{'first_inputs':{'requests':[{'request_id':'r'}]}}}
        with patch.object(P,'pinned',side_effect=[old,{'kind':'FUNCTIONAL_E5_REPLAY_QUALIFIED'}]),patch.object(P,'validate_main_receipt',return_value={'status':'passed'}) as gate,patch.object(P,'require_current_features',return_value={'status':'passed'}) as current,patch.object(P,'read',return_value={'requests':[{'request_id':'r'}]}):
            result=P.inspect_common_mathqa(plan,inputs)
        gate.assert_called_once();current.assert_called_once_with(old)
        self.assertIs(result['historical_feature_identity_verified'],False)

    def test_mathqa_wrong_current_pool_fails_before_gate(self):
        sub=dict(origin_state='state',origin_export='export',common_e5={},common_feature_paths=[],historical_plan={})
        with patch.object(P,'pinned',return_value={'origin_state':'wrong'}),patch.object(P,'validate_main_receipt') as gate:
            with self.assertRaisesRegex(ValueError,'exact functionally qualified'):P.inspect_common_mathqa({'datasets':{'mathqa':sub}}, {})
        gate.assert_not_called()

    def test_origin_content_tampering_rejected_before_mmap(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'state').write_text('a');(d/'export').mkdir();(d/'export/model.safetensors').write_text('b');(d/'config').write_text('c')
            sub=dict(origin_state=str(d/'state'),origin_state_identity=P.identity(d/'state'),origin_export=str(d/'export'),
                origin_weights_identity=P.identity(d/'export/model.safetensors'),origin_config=str(d/'config'),original_config={'identity':P.identity(d/'config')})
            P.verify_origin_paths(sub);(d/'state').write_text('changed')
            with self.assertRaisesRegex(ValueError,'content changed'):P.verify_origin_paths(sub)

    def test_visible_cuda_fails_with_durable_zero_model_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'PLAN.json';p.write_text(json.dumps({'root':d,'execution_commit':'fixture'}))
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch.object(P,'inspect_states') as state:
                with self.assertRaisesRegex(ValueError,'CUDA_VISIBLE_DEVICES'):P.inspect(p)
            result=json.loads((Path(d)/'CPU_PREFLIGHT.json').read_text())
        state.assert_not_called();self.assertEqual(result['status'],'failed');self.assertEqual(result['model_work'],0)


if __name__=='__main__':unittest.main()
