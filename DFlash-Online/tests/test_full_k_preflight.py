"""Original-evidence reconstruction is CPU-only and fails on native drift."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from online_feedback.storage import write_json
from experiments.full_k_gsm8k import preflight as P
from experiments.o_half_matched_lr.archive import ArchiveReader
from experiments.ok_events_5_8.storage import identity
from test_full_k_gsm8k_evaluation import FullBackend, requests, harmful


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.folder = self.root / 'campaign_001/evaluation'
        self.folder.mkdir(parents=True)
        self.inputs = requests(2)
        self.manifest = {'eval_requests': [dict(row, request_index=ordinal + 1)
            for ordinal, row in enumerate(self.inputs)]}
        self.checkpoints = {'F': '/F', 'O': '/O58'}
        backend = FullBackend(self.inputs)
        publications = []
        for ordinal, request in enumerate(self.inputs):
            for repeat in range(3):
                for method in ('F', 'O'):
                    publications.append(backend.publish(self.checkpoints[method]))
                    label = f'{ordinal:04d}_r{repeat}_{method}'
                    record = backend.generate(request['prompt_ids'], request_id=label,
                        session_id='heldout', request_index=ordinal, purpose='evaluation',
                        max_new_tokens=2048, save_to=self.folder / (label + '.json'))
                write_json(self.folder / f'{ordinal:04d}_r{repeat}_comparison.json', dict(
                    request_id=request['request_id'], repeat=repeat,
                    order=['F','O'] if (ordinal + repeat) % 2 == 0 else ['O','F'],
                    valid=True, errors=[], canonical_response_ids=record['response_ids']))
        write_json(self.folder / 'BENCHMARK_ACTIVATIONS.json', publications)
        write_json(self.folder / 'SUMMARY.json', dict(distinct_requests=2, valid_pairs=6, invalid_pairs=0,
            F={'verification_calls':12}, O={'verification_calls':12}))

    def audit(self):
        return P.audit_evaluation(ArchiveReader(self.root), self.manifest, self.checkpoints)

    def mutate(self, name, function):
        path = self.folder / name
        value = json.loads(path.read_text())
        function(value)
        write_json(path, value)

    def test_all_original_repeats_full_native_sources_and_first_F_canonicals(self):
        reader = ArchiveReader(self.root)
        result = P.audit_evaluation(reader, self.manifest, self.checkpoints)
        self.assertEqual(result['raw_generations'], 12)
        self.assertEqual(result['native_blocks'], 24)
        self.assertEqual(result['call_totals'], {'F':4, 'O':4})
        self.assertEqual([r['original_index'] for r in result['requests']], [1,2])
        self.assertEqual([r['panel_ordinal'] for r in result['requests']], [0,1])
        self.assertEqual(len(reader.files), 32)
        self.assertEqual(result['canonicals']['eval0']['F_counts']['verification_calls'], 2)
        self.assertFalse(result['latency_reused'])

    def test_later_repeat_canonical_mismatch_is_rejected(self):
        self.mutate('0001_r2_F.json', lambda row: row['response_ids'].__setitem__(0, 999))
        with self.assertRaises(ValueError):
            self.audit()

    def test_termination_mismatch_is_rejected(self):
        self.mutate('0000_r0_O.json', lambda row: row.update(finish_reason='length'))
        with self.assertRaisesRegex(ValueError, 'finish_reason|termination'):
            self.audit()

    def test_native_file_mismatch_is_rejected_even_when_inline_record_looks_valid(self):
        self.mutate('0000_r0_F_NATIVE.json', lambda row: row.update(collect_features=True))
        with self.assertRaisesRegex(ValueError, 'native_source_differs'):
            self.audit()

    def test_valid_but_different_repeat_counts_are_rejected(self):
        backend = FullBackend(self.inputs, lambda record, _: harmful(record))
        backend.publish('/O58')
        backend.generate(self.inputs[0]['prompt_ids'], request_id='0000_r1_O',
            session_id='heldout',request_index=0,purpose='evaluation',max_new_tokens=2048,
            save_to=self.folder/'0000_r1_O.json')
        with self.assertRaisesRegex(ValueError, 'counts differ across repeats'):
            self.audit()

    def test_false_saved_pair_or_reordered_identity_is_rejected(self):
        self.mutate('0000_r1_comparison.json', lambda row: row.update(valid=False))
        with self.assertRaisesRegex(ValueError, 'pair validity or schedule'):
            self.audit()

    def test_original_seal_allows_only_reviewed_files_with_historical_bytes(self):
        code = self.root / 'code'
        (code/'online_feedback').mkdir(parents=True)
        (code/'provenance/objects').mkdir(parents=True)
        path = code/'online_feedback/objective.py'
        path.write_text('old')
        old = identity(path)['sha256']
        (code/'provenance/objects'/old).write_text('old')
        path.write_text('new reviewed K branch')
        result = P.original_source_check(code, {'source_sha256': {'online_feedback/objective.py': old}})
        self.assertEqual(len(result),1)
        (code/'provenance/objects'/old).write_text('unrelated bytes')
        with self.assertRaisesRegex(ValueError,'historical implementation'):
            P.original_source_check(code, {'source_sha256': {'online_feedback/objective.py': old}})

    def test_unreviewed_runtime_source_change_is_rejected(self):
        path=self.root/'runtime.py'
        path.write_text('unchanged')
        old=identity(path)['sha256']
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Unexpected original'):
            P.original_source_check(self.root, {'source_sha256': {'runtime.py': old}})

    def test_inventory_failure_is_durable_and_does_not_start_model_work(self):
        output=self.root/'audit'
        with self.assertRaises(FileNotFoundError):
            P.inventory(self.root,self.root/'pins',output)
        receipt=json.loads((output/'REUSE_AUDIT.json').read_text())
        self.assertEqual(receipt['status'],'failed')
        self.assertEqual(receipt['model_work'],0)
        self.assertIn('FileNotFoundError',receipt['error'])

    def test_seal_keeps_original_bytes_and_complete_balanced_schedules(self):
        from experiments.full_k_gsm8k import seal as S
        from experiments.osd_persistent.data import build_manifest
        from test_osd_persistent_data import PersistentDataTests
        from test_ok_coding_training import configuration
        prepared=self.root/'prepared';prepared.mkdir()
        manifest=build_manifest(*PersistentDataTests.full_rows(),cap_policy='fixed2048',
                                train_context_limit=2272,serving_context_limit=4096)
        write_json(prepared/'ORIGINAL_MANIFEST.json',manifest)
        (prepared/'ORIGINAL_CONFIG.yaml').write_text(configuration(17))
        write_json(prepared/'CANONICALS.json',{'fixture':'No generated reference'})
        inputs=[dict(r,panel_ordinal=i,original_index=r['request_index'],stop_token_ids=[151645])
                for i,r in enumerate(manifest['eval_requests'])]
        audit=dict(status='passed',model_work=0,raw_generations=6144,
            canonical_identity=identity(prepared/'CANONICALS.json'),
            original_config='/original/config',original_manifest='/original/manifest',
            files={'/original/config':identity(prepared/'ORIGINAL_CONFIG.yaml'),
                   '/original/manifest':identity(prepared/'ORIGINAL_MANIFEST.json'),
                   '/F/model.safetensors':{'bytes':2097259104,'sha256':'0'*64}},
            checkpoints={'F':'/F','O':'/O58'},requests=inputs,manifest_binding=P.stable_digest(manifest),
            original_run='/original',target='/target',draft='/F',training_python='/trainpython',worker_python='/nativepython')
        write_json(prepared/'REUSE_AUDIT.json',audit)
        pins=self.root/'pins';pins.mkdir()
        for name in ('MODEL_PIN.json','RUNTIME_PIN.json','TRAINING_PIN.json'):
            write_json(pins/name,{})
        source=self.root/'source';(source/'experiments/full_k_gsm8k').mkdir(parents=True)
        args=SimpleNamespace(code=str(source),inventory=str(prepared/'REUSE_AUDIT.json'),pins=str(pins),
            output=str(self.root/'sealedrun'),remote_root='/approved/run',remote_code='/approved/source')
        def git(command,**options):
            return {'rev-parse':'a'*40+'\n','status':'','ls-files':b''}[command[1]]
        with patch.object(S.subprocess,'check_output',side_effect=git):
            result=S.seal(args)
        folder=self.root/'sealedrun/sealed'
        plan=json.loads((folder/'PLAN.json').read_text())
        self.assertEqual((folder/'ORIGINAL_MANIFEST.json').read_bytes(),(prepared/'ORIGINAL_MANIFEST.json').read_bytes())
        self.assertEqual(len(plan['evaluation_schedule']),12288)
        self.assertEqual(sum(map(len,plan['intermediate_schedules'].values())),768)
        self.assertEqual(plan['checkpoints']['K'],'/approved/run/campaign_001/arms/K/trainer/events/event_058/export')
        self.assertEqual(plan['budget']['total_generations'],28010)
        self.assertFalse(plan['automatic_retry'])
        self.assertEqual(result['status'],'sealed_before_model_work')


if __name__ == '__main__':
    unittest.main()
