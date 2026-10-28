"""CPU fixtures for immutable parent inputs and late-event preparation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.krc_events_3_4.test_evaluate import panel

SPEC = importlib.util.spec_from_file_location('krc34_prepare', Path(__file__).with_name('prepare_remote.py'))
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prepare.write(path, value)
    return path


def text_file(path, value='retained fixture'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def measurements(path, requests, totals):
    rows = []
    for arm, total in totals.items():
        for index, request in enumerate(requests):
            calls = total // len(requests) + int(index < total % len(requests))
            rows.extend(dict(request_id=request['request_id'], method=arm, repeat=repeat,
                             valid_measurement=True, verification_calls=calls) for repeat in range(3))
    text_file(path, ''.join(json.dumps(row) + '\n' for row in rows))


def event_fixture(base, number, checkpoint):
    event_id = f'event_{number:03d}'
    features, response_paths = [], []
    for offset in range(128):
        index = (number - 1) * 128 + offset
        rid = f'train_{index:05d}'
        capped = index == 399
        prompt = [index + 10000] * (183 if capped else 3)
        response = [10] * 2048 if capped else [10, 99]
        path = text_file(base / 'features' / (rid + '.pt'))
        features.append(dict(record_id=rid, length=len(prompt) + len(response),
            prompt_length=len(prompt), response_tokens=len(response),
            adjacent_response_positions=len(response) - 1, path=str(path)))
        response_paths.append(dump(base / 'responses' / (rid + '.json'), dict(
            record_id=rid, prompt_ids=prompt, response_ids=response,
            finish_reason='length' if capped else 'eos', checkpoint_id=checkpoint,
            session_id=event_id, max_new_tokens=2048, stop_token_ids=[99])))
    rows = []
    for index in range(256):
        rows.append(dict(record_id=features[index % 128]['record_id'],
            **{'pass': index // 128}, optimizer_step_after=(number - 1) * 128 + (index + 1) // 2,
            optimizer_stepped=bool(index % 2),
            sampled_anchors=[dict(positions=[[1]], keep_mask=[[True]])]))
    receipt = dict(event_id=event_id, objective='osd', passes=2, response_presentations=256,
        optimizer_steps=128, start_step=(number - 1) * 128, end_step=number * 128,
        reduction='response_mean', status='completed', features=features, records=rows)
    pool = dict(event_id=event_id, checkpoint=checkpoint, immutable=True, response_generations=128,
        record_ids=[row['record_id'] for row in features], features=[row['path'] for row in features],
        identities={row['path']: prepare.identity(row['path']) for row in features})
    return receipt, pool, response_paths


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.runs = self.root / 'runs'
        self.runs.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(prepare, 'SOURCE', self.source).start()
        patch.object(prepare, 'RUNS', self.runs).start()

    def installation(self, root):
        name = 'experiments/krc_events_3_4/prep_fixture_addition.py'
        staged = text_file(root / 'code' / name, '# exact additive fixture\n')
        additions = {name: prepare.identity(staged)}
        dump(root / 'ADDITIONS.json', additions)
        source = dict(execution_git_commit='a' * 40, dirty_files=[],
            repository='https://example.test/evidence', branch='codex/test',
            working_source=str(self.source), source_subdirectory='DFlash-Online', additions=additions)
        dump(root / 'EXECUTION_SOURCE.json', source)
        return name, source

    def parent(self):
        parent = self.runs / 'parent'
        text_file(parent / 'EXIT_CODE', '0\n')
        dump(parent / 'POSTFLIGHT.json', {'status': 'passed'})
        dump(parent / 'campaign_001/STATUS.json', {'status': 'completed'})
        code = text_file(self.source / 'qualified.py')
        model = text_file(self.root / 'model/model.safetensors')
        seal = dict(files={str(path): prepare.identity(path) for path in (code, model)},
                    exact_file_sets={str(model.parent): [model.name]})
        dump(parent / 'sealed/SEAL.json', seal)
        return parent, code, model

    def test_retained_count_maps_require_exact_three_valid_integer_repeats(self):
        requests = panel(count=2)['requests']
        path = self.root / 'measurements.jsonl'
        measurements(path, requests, {'O': 5})
        self.assertEqual(sum(prepare.retained_counts(path, 'O', requests).values()), 5)
        original = path.read_text()
        for changed in ('duplicate', 'pending', 'boolean_count', 'inconsistent'):
            rows = [json.loads(line) for line in original.splitlines()]
            if changed == 'duplicate':
                rows[1]['repeat'] = rows[0]['repeat']
            elif changed == 'pending':
                rows[0]['valid_measurement'] = None
            elif changed == 'boolean_count':
                rows[0]['verification_calls'] = True
            else:
                rows[0]['verification_calls'] += 1
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'Invalid retained'):
                prepare.retained_counts(path, 'O', requests)

    def test_parent_scientific_changes_missing_pins_and_extra_export_files_fail(self):
        parent, code, model = self.parent()
        prepare.verify_parent(parent, {})
        original = code.read_text()
        code.write_text('changed arithmetic')
        with self.assertRaisesRegex(ValueError, 'dependency changed'):
            prepare.verify_parent(parent, {})
        code.write_text(original)
        text_file(model.parent / 'extra.json')
        with self.assertRaisesRegex(ValueError, 'file set changed'):
            prepare.verify_parent(parent, {})
        (model.parent / 'extra.json').unlink()
        code.unlink()
        with self.assertRaises(FileNotFoundError):
            prepare.verify_parent(parent, {})

    def test_only_explicit_publication_document_mapping_can_change(self):
        parent, code, _ = self.parent()
        doc = text_file(self.source / 'README.md', 'before')
        original = text_file(self.root / 'retained-readme.md', 'before')
        seal = prepare.read(parent / 'sealed/SEAL.json')
        seal['files'][str(doc)] = prepare.identity(doc)
        dump(parent / 'sealed/SEAL.json', seal)
        doc.write_text('published documentation')
        with self.assertRaisesRegex(ValueError, 'dependency changed'):
            prepare.verify_parent(parent, {})
        binding = dict(original_identity=prepare.identity(original), execution_identity=prepare.identity(doc),
                       retained_original=str(original))
        _, _, used = prepare.verify_parent(parent, {str(doc): binding})
        self.assertEqual(len(used), 1)
        code_old = text_file(self.root / 'old-code.py', code.read_text())
        code.write_text('new scientific source')
        code_binding = dict(original_identity=prepare.identity(code_old), execution_identity=prepare.identity(code),
                            retained_original=str(code_old))
        with self.assertRaisesRegex(ValueError, 'dependency changed'):
            prepare.verify_parent(parent, {str(doc): binding, str(code): code_binding})

    def test_exact_additive_installation_and_commit_binding(self):
        root = self.root / 'new'
        name, source = self.installation(root)
        self.assertEqual(prepare.validate_installation(root, source), source['additions'])
        for change in (dict(dirty_files=['modified.py']), dict(execution_git_commit='short'), dict(additions={})):
            with self.assertRaisesRegex(ValueError, 'committed execution'):
                prepare.validate_installation(root, {**source, **change})
        existing = text_file(self.source / name)
        with self.assertRaisesRegex(ValueError, 'additive installation'):
            prepare.validate_installation(root, source)
        existing.unlink()
        text_file(root / 'code/unlisted.py')
        with self.assertRaisesRegex(ValueError, 'file list'):
            prepare.validate_installation(root, source)

    def test_pool_retains_all_features_capped_record_and_excludes_eval_prompts(self):
        ordinary, pool, responses = event_fixture(self.root / 'event004', 4, '/O3')
        retained_k = copy.deepcopy(ordinary)
        panels = [panel(count=2), panel('additional', 2)]
        features, capped = prepare.validate_pool(pool, ordinary, retained_k, responses, panels, 4)
        self.assertEqual(features, pool['features'])
        self.assertEqual(capped[0]['record_id'], 'train_00399')
        with self.assertRaisesRegex(ValueError, 'response count'):
            prepare.validate_pool(pool, ordinary, retained_k, responses[:-1], panels, 4)
        changed = copy.deepcopy(retained_k)
        changed['features'][0]['path'] = 'substitution.pt'
        with self.assertRaisesRegex(ValueError, 'features'):
            prepare.validate_pool(pool, ordinary, changed, responses, panels, 4)
        row = prepare.read(responses[0])
        row['prompt_ids'] = panels[0]['requests'][0]['prompt_ids']
        dump(responses[0], row)
        with self.assertRaisesRegex(ValueError, 'overlaps training'):
            prepare.validate_pool(pool, ordinary, retained_k, responses, panels, 4)

    def complete_fixture(self):
        root = self.runs / 'new'
        self.installation(root)
        parent = self.runs / 'dflash_krc_continuation_20261028'
        continuation = self.runs / 'dflash_ok_continuation_20261027'
        panels = [panel(), panel('additional')]
        for value in panels:
            value['provenance'] = dict(canonical=str(text_file(parent / (value['name'] + '_canonical.json'), '{}')),
                                      historical_counts=str(text_file(parent / (value['name'] + '_historical.json'), '{}')))
        target = text_file(self.root / 'target/model.safetensors').parent
        draft = text_file(self.root / 'draft/model.safetensors').parent
        initial = self.runs / 'ordinary_one/campaign_001/trainer/events/event_001/RESULT.json'
        dump(initial, {})
        measurements(initial.parents[3] / 'evaluation/MEASUREMENTS.jsonl', panels[0]['requests'], {'O': 3075})
        dump(continuation / 'sealed/PLAN.json', dict(initial={'O': {'receipt': str(initial)}}))
        previous_export = text_file(continuation / 'campaign_001/O/events/event_002/export/model.safetensors').parent
        ordinary2_path = dump(continuation / 'campaign_001/O/events/event_002/RESULT.json',
                              {'checkpoint_path': str(previous_export)})
        krc2, _, _ = event_fixture(parent / 'event2-fixture', 2, '/O1')
        state = text_file(parent / 'receipt_selected_state.pt')
        checkpoint = text_file(parent / 'receipt_selected_export/model.safetensors').parent
        krc2.update(objective='osd_first_error', full_state_path=str(state), checkpoint_path=str(checkpoint))
        dump(parent / 'campaign_001/KRC2/events/event_002/RESULT.json', krc2)
        for number in (3, 4):
            event = f'event_{number:03d}'
            base = continuation / 'campaign_001' / event
            ordinary, pool, _ = event_fixture(base, number, str(previous_export))
            o_export = text_file(continuation / f'campaign_001/O/events/{event}/export/model.safetensors').parent
            k_export = text_file(continuation / f'campaign_001/K/events/{event}/export/model.safetensors').parent
            ordinary['checkpoint_path'] = str(o_export)
            retained_k = copy.deepcopy(ordinary)
            retained_k.update(objective='osd_first_error', checkpoint_path=str(k_export))
            dump(continuation / f'campaign_001/O/events/{event}/RESULT.json', ordinary)
            dump(continuation / f'campaign_001/K/events/{event}/RESULT.json', retained_k)
            dump(base / 'POOL.json', pool)
            dump(base / 'COSTS.json', {'historical': True})
            previous_export = o_export
        for value in panels:
            measurements(parent / 'campaign_001/evaluation' / value['name'] / 'MEASUREMENTS.jsonl',
                         value['requests'], {'K2': 2973, 'KRC2': 2967})
        measurements(continuation / 'campaign_001/evaluation_002/MEASUREMENTS.jsonl', panels[0]['requests'], {'O': 3059})
        measurements(continuation / 'campaign_001/evaluation_004/MEASUREMENTS.jsonl', panels[0]['requests'], {'O': 2994, 'K': 2942})
        startup_state = text_file(parent / 'pinned_initial.pt')
        old = dict(startup_ordinary_receipt=str(initial), pinned_initial=str(startup_state),
            event2_ordinary_receipt=str(ordinary2_path), target=str(target), draft=str(draft),
            worker_python='/qualified/python', panels=panels)
        dump(parent / 'sealed/PLAN.json', old)
        for name in ('MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
            dump(parent / 'sealed' / name, {})
        text_file(parent / 'sealed/config.yaml', 'max_steps: 7424\n')
        text_file(parent / 'EXIT_CODE', '0\n')
        dump(parent / 'POSTFLIGHT.json', {'status': 'passed'})
        dump(parent / 'campaign_001/STATUS.json', {'status': 'completed'})
        parent_files = [path for path in parent.rglob('*') if path.is_file()]
        parent_files.extend([target / 'model.safetensors', draft / 'model.safetensors'])
        dump(parent / 'sealed/SEAL.json', dict(
            files={str(path): prepare.identity(path) for path in parent_files},
            exact_file_sets={str(target): ['model.safetensors'], str(draft): ['model.safetensors']}))
        return root, parent, state, checkpoint

    def test_complete_preparation_seals_receipt_state_all256_responses_and_exact_exports(self):
        root, parent, state, checkpoint = self.complete_fixture()
        result = prepare.prepare(root)
        self.assertEqual(result['retained_responses'], 256)
        self.assertEqual(result['development_control_calls'], {'O4': 2994, 'K4': 2942})
        plan = prepare.read(root / 'sealed/PLAN.json')
        seal = prepare.read(root / 'sealed/SEAL.json')
        self.assertEqual(plan['krc2_state'], str(state))
        self.assertEqual(plan['krc2_checkpoint'], str(checkpoint))
        self.assertEqual(plan['main']['optimizer_steps'], 256)
        self.assertEqual(plan['evaluation']['generations'], 1152)
        self.assertEqual(plan['panels'][1]['expected_control_counts'], None)
        self.assertNotIn('O1', plan['panels'][1]['historical_counts'])
        self.assertNotIn('O2', plan['panels'][1]['historical_counts'])
        old = prepare.read(parent / 'sealed/PLAN.json')
        for value, prior in zip(plan['panels'], old['panels']):
            self.assertEqual(value['requests'], prior['requests'])
        for event in plan['events']:
            pool = prepare.read(event['pool'])
            self.assertEqual(event['feature_paths'], pool['features'])
            schedule = prepare.read(event['reference_schedule'])
            self.assertEqual(len(schedule['windows']), 128)
        self.assertTrue(seal['sealed_before_training'])
        self.assertEqual(seal['execution_source']['execution_git_commit'], 'a' * 40)
        self.assertEqual(seal['exact_file_sets'][str(checkpoint)], ['model.safetensors'])
        self.assertIn(str(state), seal['files'])
        self.assertTrue(all(prepare.identity(path) == expected for path, expected in seal['files'].items()))
        new_responses = [name for name in seal['files'] if '/dflash_ok_continuation_20261027/campaign_001/event_' in name
                         and '/responses/' in name]
        self.assertEqual(len(new_responses), 256)
        self.assertFalse((root / 'campaign_001').exists())
        self.assertEqual(plan['capped_responses_preserved']['event_004'][0]['record_id'], 'train_00399')


if __name__ == '__main__':
    unittest.main()
