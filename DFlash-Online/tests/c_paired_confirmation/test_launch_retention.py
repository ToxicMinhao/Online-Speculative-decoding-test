"""CPU integration gates and exact derived-export retirement boundaries."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.c_paired_confirmation import launch as L, retention as R
from experiments.ok_coding.storage import identity


class InfrastructureTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.plan=dict(root=str(self.root),binding='bound',execution_commit='c'*40,target='/target',
            devices={'A':'GPU-A','B':'GPU-B'},storage={'retain_final_exports':False},
            datasets={d:{'origin_export':'/'+d+'/origin'} for d in ('gsm8k','mathqa')})

    def qualifications(self):
        for d in ('gsm8k','mathqa'):
            write_json(self.root/'qualification'/f'{d}_training_A'/'STATUS.json',dict(status='passed',
                result={'cpu_export_qualification':{'status':'passed'},'cpu_export':{'status':'passed'},
                        'draft_config':'/qualified/config.json','embedding_source':'/target',
                        'embedding_key':'model.embed_tokens.weight'}))
            for s in ('A','B'):
                write_json(self.root/'qualification'/f'{d}_serving_{s}'/'STATUS.json',dict(status='passed',
                    result={'qualification':{'signatures':{'a':[1,2],'b':[3]}},'checkpoint_identity':{'sha256':d}}))

    def test_qualification_gate_accepts_actual_schema_then_rejects_cross_card_difference(self):
        self.qualifications()
        gate=L.qualification_gate(self.plan)
        self.assertEqual(gate['status'],'passed');self.assertEqual(set(gate['datasets']),{'gsm8k','mathqa'})
        path=self.root/'qualification/mathqa_serving_B/STATUS.json';row=R.read(path)
        row['result']['qualification']['signatures']['a'].append(4);write_json(path,row)
        with self.assertRaisesRegex(ValueError,'cross-card'):L.qualification_gate(self.plan)

    def prepare_endpoints(self):
        self.qualifications();states=[]
        for d in ('gsm8k','mathqa'):
            for arm in ('N','C'):
                for stage in ('first','second'):
                    task=arm+'_'+stage;base=self.root/'tasks'/f'{d}_{task}';base.mkdir(parents=True)
                    state=base/'state.pt';state.write_bytes((d+task).encode());states.append(state)
                    export=base/'export';export.mkdir();(export/'model.safetensors').write_bytes(b'weights')
                    (export/'config.json').write_text('{}')
                    saved=dict(state=str(state),state_identity=identity(state),export=str(export),weights_identity=identity(export/'model.safetensors'))
                    write_json(base/'RESULT.json',dict(status='completed',endpoint=saved))
                    write_json(base/'STATUS.json',dict(status='completed'))
                write_json(self.root/'tasks'/f'{d}_{arm}_collect'/'STATUS.json',dict(status='completed'))
        return states

    def test_exact_export_manifest_preserves_all_eight_recoveries_and_feature_banks(self):
        states=self.prepare_endpoints()
        banks=[]
        for d in ('gsm8k','mathqa'):
            for arm in ('N','C'):
                p=self.root/'tasks'/f'{d}_{arm}_collect'/'features.pt';p.write_bytes(b'unique');banks.append(p)
        captured=[]
        def unlink(plan,paths,**kwargs):captured.extend(paths);return {'status':'completed'}
        with patch('experiments.c_paired_confirmation.export.verify_derived_export',return_value={'status':'passed'}),patch.object(R,'unlink_manifest',side_effect=unlink):
            for d in ('gsm8k','mathqa'):
                for arm in ('N','C'):R.retire_endpoint_export(self.plan,d,arm+'_first')
            for arm in ('N','C'):R.retire_endpoint_export(self.plan,'gsm8k',arm+'_second')
            with self.assertRaisesRegex(ValueError,'GSM final'):R.retire_endpoint_export(self.plan,'mathqa','N_second')
        self.assertEqual(len(captured),12)
        self.assertTrue(all(Path(p).parent.name=='export' for p in captured))
        self.assertTrue(all(p.is_file() for p in states+banks))
        self.assertFalse(set(map(str,states+banks)).intersection(captured))

    def test_retirement_waits_for_real_consumers_and_qualified_cpu_reconstruction(self):
        self.prepare_endpoints()
        write_json(self.root/'tasks/gsm8k_N_second/STATUS.json',{'status':'running'})
        with patch('experiments.c_paired_confirmation.export.verify_derived_export') as verify:
            with self.assertRaisesRegex(ValueError,'consumer'):R.retire_endpoint_export(self.plan,'gsm8k','N_first')
            verify.assert_not_called()
        write_json(self.root/'tasks/gsm8k_N_second/STATUS.json',{'status':'completed'})
        write_json(self.root/'qualification/gsm8k_training_A/STATUS.json',{'status':'failed'})
        with patch.object(R,'unlink_manifest') as unlink:
            with self.assertRaisesRegex(ValueError,'qualified'):R.retire_endpoint_export(self.plan,'gsm8k','N_first')
            unlink.assert_not_called()

    def test_busy_live_or_foreign_retirement_cannot_unlink(self):
        p=self.root/'new.pt';p.write_bytes(b'x');plan=self.plan
        options=dict(receipt_path=self.root/'receipt.json',reason='test',retained=[],verification={})
        with patch.object(R,'clock_status',return_value={'active':{'worker':{}}}):
            with self.assertRaisesRegex(RuntimeError,'workers'):R.unlink_manifest(plan,[p],**options)
        with patch.object(R,'clock_status',return_value={'active':{}}):
            with self.assertRaisesRegex(ValueError,'newly-created'):R.unlink_manifest(plan,[self.root.parent/'foreign.pt'],**options)
            process=SimpleNamespace(pid=12,children=lambda **kw:[],open_files=lambda:[SimpleNamespace(path=str(p))],memory_maps=lambda **kw:[])
            with patch('psutil.Process',return_value=process):
                with self.assertRaisesRegex(RuntimeError,'still live'):R.unlink_manifest(plan,[p],**options)
        self.assertTrue(p.exists())


if __name__=='__main__':unittest.main()
