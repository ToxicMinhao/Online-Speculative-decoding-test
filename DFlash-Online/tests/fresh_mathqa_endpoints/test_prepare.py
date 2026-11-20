"""No model work: authenticate package assembly and refuse incomplete delivery."""
from pathlib import Path
import json
import tempfile
import unittest

from experiments.fresh_mathqa_endpoints.prepare import prepare


class PrepareTests(unittest.TestCase):
    def fixture(self, root):
        task = root/'task'
        sealed = task/'panel_audit/sealed'
        sealed.mkdir(parents=True)
        (task/'closeout_audit').mkdir()
        for name in ('PANEL.json','SCHEDULE.json','SEAL.json','WARMUPS.json'):
            (sealed/name).write_text('{}')
        (sealed/'WARMUP_CANONICALS.json').write_text(json.dumps({
            'fixed':{'reference_checkpoint':'/retained/warmup-producer'}}))
        (task/'panel_audit/POOL_AUDIT.json').write_text('{}')
        (task/'closeout_audit/CLOSEOUT.json').write_text('{"status":"passed"}')
        rows = [dict(role=role,path=f'/models/{role}/{name}',
                     expected_identity=dict(bytes=100,sha256=char*64))
                for role,name,char in [('O58_selected_endpoint','model.safetensors','a'),
                                       ('AREPO58','model.safetensors','b'),
                                       ('C58','model.safetensors','c'),
                                       ('released_target','tokenizer.json','d')]]
        (task/'EXPECTED_INPUT_IDENTITIES.json').write_text(json.dumps(rows))
        old = root/'dflash_full_c_gsm8k_20261118/package_001/sealed'
        old.mkdir(parents=True)
        (old/'RUNTIME_PIN.json').write_text('{"runtime":"fixture"}')
        baseline = root/'compiler.json'
        baseline.write_text('{"directories":{},"allocated_bytes":0}')
        return task, baseline

    def test_fixed_endpoint_package_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task, baseline = self.fixture(root)
            package = root/'package'
            args = (task,package,'/data/sources/minhao/runs/new/execution_001',
                    '/data/sources/minhao/new_source/DFlash-Online','e'*40,baseline)
            plan = prepare(*args)
            self.assertEqual(set(plan['checkpoints']),{'O58','AREPO58','KPLUS_C58'})
            self.assertEqual(plan['reference_checkpoints']['scored'],plan['checkpoints']['O58'])
            self.assertEqual(plan['optimizer_steps'],0)
            self.assertEqual(plan['limits']['elapsed_s'],28800)
            self.assertEqual(plan['limits']['occupied_card_s'],43200)
            self.assertEqual(sum(plan['storage_components_bytes'].values()),plan['projected_peak_bytes'])
            self.assertEqual((package/'sealed/SEAL.json').read_bytes(),(task/'panel_audit/sealed/SEAL.json').read_bytes())
            self.assertEqual(set(plan['canonical_provenance']),{'target_identity','tokenizer_identity','runtime_identity'})
            with self.assertRaises(FileExistsError):
                prepare(*args)

    def test_missing_closeout_stops_before_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task, baseline = self.fixture(root)
            (task/'closeout_audit/CLOSEOUT.json').write_text('{"status":"pending"}')
            with self.assertRaises(ValueError):
                prepare(task,root/'package','/run','/source','e'*40,baseline)
            self.assertFalse((root/'package').exists())


if __name__ == '__main__':
    unittest.main()
