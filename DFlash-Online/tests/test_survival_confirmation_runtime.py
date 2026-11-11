"""Serving-only orchestration boundaries; no GPU or optimizer work."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from experiments.survival_confirmation import runtime as R

class RuntimeTests(unittest.TestCase):
    def test_cleanup_failure_preserves_durable_counters(self):
        class Broken:
            def close(self): raise RuntimeError('close failed')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'STATUS.json';receipt=dict(status='failed',attempted_generations=1)
            original=ValueError('primary')
            R.finish_worker(Broken(),receipt,path,time.perf_counter(),original)
            saved=json.loads(path.read_text())
            self.assertEqual(saved['attempted_generations'],1)
            self.assertEqual(saved['status'],'failed');self.assertIn('close failed',saved['cleanup_error'])
            with self.assertRaisesRegex(RuntimeError,'close failed'):
                R.finish_worker(Broken(),receipt,path,time.perf_counter())
            self.assertTrue(path.exists())
    def test_observed_physical_card_must_match_pin(self):
        plan=dict(serving_gpu=0,serving_gpu_identity='GPU-zero')
        with patch.object(R.subprocess,'check_output',return_value='0, GPU-zero\n1, GPU-one\n'):
            self.assertEqual(R.physical_gpu(plan),'GPU-zero')
        with patch.object(R.subprocess,'check_output',return_value='0, GPU-other\n'):
            with self.assertRaisesRegex(ValueError,'changed'): R.physical_gpu(plan)
    def test_no_missing_refs_never_constructs_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'panel.json').write_text(json.dumps({'requests':[{'request_id':'one'}]}))
            (root/'reuse.json').write_text(json.dumps({'one':{'verified':True}}))
            plan=dict(root=str(root),panel_manifest=str(root/'panel.json'),reused_references=str(root/'reuse.json'),missing_reference_ids=[])
            with patch('online_feedback.sglang_backend.SGLangDFlashBackend') as backend:
                result=R.prepare_references(plan)
            backend.assert_not_called()
            self.assertEqual(result['planned_generations'],0)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['reused'],1)
            self.assertEqual(result['returned_without_validation'],0)
    def test_run_stops_preflight_failure_without_reference_or_serving_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'sealed').mkdir()
            (root/'sealed/PLAN.json').write_text(json.dumps(dict(execution_source=dict(commit='a'*40))))
            with patch.object(R,'preflight',side_effect=ValueError('bad pin')),patch.object(R,'prepare_references') as refs,patch.object(R,'evaluate') as evaluate:
                with self.assertRaisesRegex(ValueError,'bad pin'):R.run(root)
            refs.assert_not_called();evaluate.assert_not_called()
            state=json.loads((root/'STATUS.json').read_text());self.assertEqual(state['status'],'failed')
            self.assertEqual(state['optimizer_steps'],0)
            with self.assertRaises(FileExistsError):R.run(root)

if __name__=='__main__':unittest.main()
