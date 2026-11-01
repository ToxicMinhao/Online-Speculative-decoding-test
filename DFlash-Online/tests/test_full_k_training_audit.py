"""CPU-only raw receipt fixtures for the full-stream training audit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from online_feedback.storage import write_json
from experiments.full_k_gsm8k import audit as A
from experiments.o_half_matched_lr.archive import ArchiveReader
from test_full_k_gsm8k_evaluation import FullBackend, requests as evaluation_requests


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def observations(event=1):
    rows, responses, windows = [], [], []
    additive = dict(available_position_occurrences=3., selected_position_occurrences=2.,
        available_position_weight=3., selected_position_weight=2., no_error_blocks=0.,
        empty_blocks=0., main_loss_numerator=1., loss_denominator=2.000001,
        supervised_position_exposure=2., **{f'first_error_position_{i}': float(i == 2) for i in range(1, 16)})
    for index in range(256):
        draw = [dict(positions=[[3]], keep_mask=[[True]])]
        observed = dict(scope='one_original_response_forward', record_id=f'r{index % 128}',
            event_id=f'event_{event:03d}', pass_index=index // 128, presentation_ordinal=index,
            sampled_anchors=draw, loss=.5, ratio_metrics={'objective_loss': [1., 2.000001]},
            sum_metrics=deepcopy(additive))
        stepped = index % 2 == 1
        metrics = {key: value * (2 if stepped else 1) for key, value in additive.items()}
        metrics['response_presentations'] = 2. if stepped else 1.
        rows.append(dict(record_id=observed['record_id'], pass_=index // 128,
            optimizer_step_after=(event - 1) * 128 + (index + 1) // 2,
            optimizer_stepped=stepped, learning_rate=3e-4 if stepped else None,
            sampled_anchors=deepcopy(draw), metrics=metrics, response_observation=deepcopy(observed), training_s=.1))
        rows[-1]['pass'] = rows[-1].pop('pass_')
        responses.append(observed)
        if stepped:
            windows.append(dict(scope='optimizer_window', event_id=f'event_{event:03d}',
                window=(index + 1) // 2, global_step=(event - 1) * 128 + (index + 1) // 2,
                record_ids=[responses[-2]['record_id'], responses[-1]['record_id']],
                gradient_norm=1., learning_rate=3e-4, mean_response_loss=.5))
    totals = {key: sum(row['metrics'][key] for row in rows if row['optimizer_stepped']) for key in rows[0]['metrics']}
    result = dict(event_id=f'event_{event:03d}', status='completed', objective='osd_first_error',
        passes=2, response_presentations=256, optimizer_steps=128, start_step=(event - 1) * 128,
        end_step=event * 128, reduction='response_mean', records=rows,
        selection_totals=totals, selection_totals_source='optimizer_boundary_accumulation_windows_once',
        objective_work=dict(reference_operations=0, teacher_head_s=1., selection_s=.5,
            work=dict(response_forwards=256, draft_forwards=256, layer_forwards=2560,
                      vocabulary_head_calls=512, objective_chunks=512)))
    return result, responses, windows


def work_rows():
    rows = []
    counts = dict.fromkeys(A.ARM_WORK, 0)
    def call(kind):
        counts[kind] += 1
        rows.extend(dict(kind=kind, ordinal=counts[kind], phase=phase) for phase in ('started', 'completed'))
    for event in range(1, 59):
        for _ in range(128):
            call('online_generations')
        if event == 1:
            call('correctness_generations')
        for _ in range(128):
            call('terminal_preparation_forwards')
        for _ in range(128):
            for _ in range(2):
                call('response_forwards')
                call('response_backwards')
            call('optimizer_steps')
        if event == 1:
            for _ in range(3):
                call('correctness_generations')
        if event in (14, 29, 44):
            for _ in range(128):
                call('intermediate_generations')
    for _ in range(49):
        call('online_generations')
    return rows


class TrainingAuditTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_exact_work_counts_and_causal_full_horizon(self):
        path = self.root / 'WORK.jsonl'
        write_lines(path, work_rows())
        result = A.audit_work(path)
        self.assertEqual(result['completed'], A.ARM_WORK)
        self.assertEqual(sum(result['unknown'].values()), 0)

    def test_same_totals_wrong_feedback_order_fails(self):
        rows = work_rows()
        first_prepare = next(i for i, r in enumerate(rows) if r['kind'] == 'terminal_preparation_forwards')
        entry = rows[first_prepare:first_prepare + 2]
        del rows[first_prepare:first_prepare + 2]
        rows[0:0] = entry
        path = self.root / 'WORK.jsonl'
        write_lines(path, rows)
        with self.assertRaisesRegex(ValueError, 'complete own feedback'):
            A.audit_work(path)

    def test_unknown_optimizer_work_stays_unknown_and_fails(self):
        rows = work_rows()
        rows = [r for r in rows if not (r['kind'] == 'optimizer_steps' and r['ordinal'] == 7424 and r['phase'] == 'completed')]
        path = self.root / 'WORK.jsonl'
        write_lines(path, rows)
        work = A.read_work(path, A.ARM_WORK)
        self.assertEqual(work['unknown']['optimizer_steps'], 1)
        with self.assertRaises(ValueError):
            A.audit_work(path)

    def test_response_exposure_uses_boundaries_once(self):
        result, responses, windows = observations()
        audit = A.audit_response_observations(result, responses, windows)
        self.assertEqual(audit['selection_totals']['selected_position_occurrences'], 512)
        self.assertEqual(audit['sampled_anchor_occurrences'], 256)
        self.assertEqual(audit['actual_native_work']['objective_chunks'], 512)
        result['selection_totals']['selected_position_occurrences'] = 768
        with self.assertRaisesRegex(ValueError, 'double-counted'):
            A.audit_response_observations(result, responses, windows)

    def test_observation_identity_and_premerge_content_must_match(self):
        result, responses, windows = observations()
        responses[0]['sampled_anchors'][0]['positions'][0][0] = 4
        with self.assertRaisesRegex(ValueError, 'identity/anchors'):
            A.audit_response_observations(result, responses, windows)
        result, responses, windows = observations()
        windows[0]['record_ids'].reverse()
        with self.assertRaisesRegex(ValueError, 'window identity'):
            A.audit_response_observations(result, responses, windows)

    def native(self, capture=False):
        request = dict(evaluation_requests(1)[0], request_index=0)
        path = self.root / 'response.json'
        backend = FullBackend([request])
        record = backend.generate(request['prompt_ids'], request_id=request['request_id'],
            request_index=0, session_id='gsm8k_train' if capture else 'gsm8k_tail',
            purpose='serving' if capture else 'evaluation', save_to=path)
        record['metadata']['feedback_capture_in_timed_request'] = capture
        native = record['metadata']['capture']
        native.update(collect_features=capture, covered_token_ids=record['prompt_ids'] + record['response_ids'][:-1],
            target_argmax_ids=record['response_ids'], label_binding_scope='native_prefill_and_committed_verification',
            feature_path=str(self.root / 'request_001.pt'), capture_host_s=.01, feature_persistence_s=.01)
        write_json(record['metadata']['native_trace_path'], native)
        write_json(path, record)
        return request, record, path

    def test_raw_trace_and_counts_are_checked_not_status_only(self):
        request, record, path = self.native(capture=True)
        reader = ArchiveReader(self.root)
        A.audit_native(reader, path, request, '/F', capture=True, session='gsm8k_train')
        record['metadata']['verification_calls'] += 1
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, 'scalar count'):
            A.audit_native(reader, path, request, '/F', capture=True)

    def test_teacher_mask_and_terminal_position_bound_to_same_response(self):
        _, record, _ = self.native(capture=True)
        combined = record['prompt_ids'] + record['response_ids']
        terminal = dict(input_ids=combined, positions=[len(combined) - 1],
            final_feature_logits_used_for_labels=False, tensor_path=str(self.root / 'feature.terminal.pt'))
        prepared = dict(record_id=record['record_id'], prompt_length=2, response_length=5,
            input_ids=combined, target_argmax_ids=record['response_ids'], label_mismatches=[],
            target_layer_ids=[1, 8, 15, 22, 29], feature_shape=[1, 7, 20480],
            capture_backend='same_frozen_SGLang_native_capture', native_capture_validated=True,
            label_binding_scope='native_prefill_and_committed_verification', final_feature_logits_used_for_labels=False,
            response_only_loss=True, terminal_feature=terminal, status='prepared',
            feature_path=str(self.root / 'feature.pt'), teacher_preparation_s=.1,
            feature_persistence_s=.01, supervised_positions=5)
        feature = dict(record_id=record['record_id'], path=prepared['feature_path'], length=7,
            prompt_length=2, response_tokens=5, adjacent_response_positions=4)
        write_json(self.root / 'feature.json', prepared)
        write_json(self.root / 'feature.terminal.json', terminal)
        A.audit_preparation(ArchiveReader(self.root), prepared, feature, record)
        prepared['response_only_loss'] = False
        write_json(self.root / 'feature.json', prepared)
        with self.assertRaisesRegex(ValueError, 'response-only'):
            A.audit_preparation(ArchiveReader(self.root), prepared, feature, record)

    def test_retirement_hashes_and_exact_file_set(self):
        _, record, _ = self.native(capture=True)
        root = self.root / 'event'
        result = dict(event_id='event_001', full_state_path='/run/state.pt', checkpoint_path='/run/export')
        prepared = dict(feature_path=str(self.root / 'feature.pt'),
            terminal_feature=dict(tensor_path=str(self.root / 'terminal.pt')))
        write_json(root / 'loop/events/event_001.json', {'preparation': [prepared]})
        compact = self.root / 'compact.json'
        write_json(compact, {'retained': True})
        entries = [dict(path=record['metadata']['capture']['feature_path'], kind='native_capture', bytes=2),
            dict(path=prepared['feature_path'], kind='prepared_feature', bytes=2),
            dict(path=prepared['terminal_feature']['tensor_path'], kind='terminal_feature', bytes=2)]
        plan = dict(status='planned', event_id='event_001', files=entries,
            compact_evidence=[dict(path=str(compact), bytes=compact.stat().st_size,
                sha256=hashlib.sha256(compact.read_bytes()).hexdigest())],
            retained_current_state=result['full_state_path'], retained_current_export=result['checkpoint_path'])
        write_json(root / 'RETIREMENT_PLAN.json', plan)
        write_json(root / 'RETIREMENT.json', dict(plan, status='completed', removed=entries, removed_bytes=6))
        checked = A.audit_retirement(ArchiveReader(self.root), root, result, [record], None)
        self.assertEqual(checked['files'], 3)
        write_json(compact, {'retained': False})
        with self.assertRaisesRegex(ValueError, 'changed since retirement'):
            A.audit_retirement(ArchiveReader(self.root), root, result, [record], None)

    def test_resource_timers_are_nested_and_work_free(self):
        row = dict(training_peak_allocated_bytes=100, training_peak_reserved_bytes=200,
            training_compiler_timers={'compile': dict(calls=2, durations_s=[.1, .2], total_s=.1 + .2)},
            extra_forwards=0, extra_backwards=0, extra_optimizer_steps=0)
        self.assertEqual(A.audit_resources(row), row)
        row['extra_forwards'] = 1
        with self.assertRaisesRegex(ValueError, 'introduced model work'):
            A.audit_resources(row)

    def test_complete_event_links_native_records_buffer_loss_publication_and_retirement(self):
        base = self.root / 'campaign_001/arms/K'
        event_root = base / 'events/event_001'
        trainer_root = base / 'trainer/events/event_001'
        export = str(trainer_root / 'export')
        state_path = str(base / 'trainer/training_state.pt')
        inputs = [dict(item, request_id=f'r{i}', request_index=i)
                  for i, item in enumerate(evaluation_requests(128))]
        backend = FullBackend(inputs)
        records, prepared_rows, features = [], [], []
        for index, request in enumerate(inputs):
            raw_path = event_root / 'loop/serving' / (request['request_id'] + '.json')
            record = backend.generate(request['prompt_ids'], request_id=request['request_id'],
                request_index=index, session_id='gsm8k_train', purpose='serving', save_to=raw_path)
            native_path = base / 'native_attempt_001/native' / f'request_{index+1:06d}.json'
            record['metadata']['native_trace_path'] = str(native_path)
            record['metadata']['feedback_capture_in_timed_request'] = True
            capture = record['metadata']['capture']
            capture.update(collect_features=True, feature_path=str(native_path.with_suffix('.pt')),
                covered_token_ids=record['prompt_ids'] + record['response_ids'][:-1],
                target_argmax_ids=record['response_ids'],
                label_binding_scope='native_prefill_and_committed_verification',
                capture_host_s=.01, feature_persistence_s=.02)
            write_json(native_path, capture)
            write_json(raw_path, record)
            write_json(raw_path.with_suffix('.costs.json'), {'serve_call_s': 1.1})
            feature_path = event_root / 'loop/features' / (request['request_id'] + '.pt')
            combined = record['prompt_ids'] + record['response_ids']
            terminal = dict(input_ids=combined, positions=[6], final_feature_logits_used_for_labels=False,
                            tensor_path=str(feature_path.with_suffix('.terminal.pt')))
            prepared = dict(record_id=request['request_id'], prompt_length=2, response_length=5,
                input_ids=combined, target_argmax_ids=record['response_ids'], label_mismatches=[],
                feature_shape=[1, 7, 20480], target_layer_ids=[1, 8, 15, 22, 29],
                capture_backend='same_frozen_SGLang_native_capture', native_capture_validated=True,
                label_binding_scope='native_prefill_and_committed_verification',
                final_feature_logits_used_for_labels=False, response_only_loss=True,
                terminal_feature=terminal, status='prepared', supervised_positions=5,
                feature_path=str(feature_path), teacher_preparation_s=.1, feature_persistence_s=.01)
            write_json(feature_path.with_suffix('.json'), prepared)
            write_json(feature_path.with_suffix('.terminal.json'), terminal)
            prepared_rows.append(prepared)
            features.append(dict(record_id=request['request_id'], path=str(feature_path), length=7,
                prompt_length=2, response_tokens=5, adjacent_response_positions=4))
            records.append(record)
        result, responses, windows = observations()
        result.update(features=features, checkpoint_path=export, export_path=export, full_state_path=state_path,
            costs=dict(training_s=sum(row['training_s'] for row in result['records']),
                       checkpoint_save_s=.2, export_s=.3, event_wall_s_before_commit=30.))
        write_json(trainer_root / 'RESULT.json', result)
        write_json(trainer_root / 'PENDING.json', {key: result[key] for key in ('event_id', 'features', 'passes', 'start_step')})
        write_lines(trainer_root / 'RESPONSE_OBSERVATIONS.jsonl', responses)
        write_lines(trainer_root / 'WINDOW_OBSERVATIONS.jsonl', windows)
        controller = dict(event_id='event_001', preparation=prepared_rows, training=result,
                          activation=backend.publish(export))
        write_json(event_root / 'loop/events/event_001.json', controller)
        write_json(event_root / 'loop/events/event_001.costs.json', {'update_call_s': 50.})
        write_json(event_root / 'loop/feedback_buffer/state.json', dict(schema_version=2, capacity=128,
            records=records, events=[dict(event_id='event_001', status='committed',
                record_ids=[r['request_id'] for r in inputs], checkpoint_id=export,
                receipt=controller, error=None)]))
        publication = dict(backend.check_weights(export), event_id='event_001', trainer_status={'global_step': 128})
        write_json(event_root / 'PUBLICATION_CHECK.json', publication)
        validation = dict(status='passed', arm='K', completed_event=1, global_step=128,
            presentations=256, adam_state_entries=58, cuda_rng_states=1, extra_model_passes=0,
            extra_optimizer_steps=0, export_exact=True,
            live=dict(status='passed', exact_resume=True, mismatches=[], global_step=128,
                      presentations=256, no_rng_reset=True))
        marker = dict(status='passed', arm='K', event=1, scientific_check=validation,
            weights=export + '/model.safetensors', state=str(trainer_root / 'training_state.pt'),
            state_identity={'bytes': 100, 'sha256': '1'*64}, weights_identity={'bytes': 100, 'sha256': '2'*64})
        write_json(base / 'boundaries/event_001.json', marker)
        write_json(base / 'boundaries/event_001_validation.json', validation)
        resource = dict(training_peak_allocated_bytes=100, training_peak_reserved_bytes=200,
            training_compiler_timers={}, extra_forwards=0, extra_backwards=0, extra_optimizer_steps=0)
        write_json(event_root / 'RESOURCE_COSTS.json', resource)
        compact = [dict(path=str(path), bytes=path.stat().st_size,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in event_root.rglob('*.json')]
        retired_files = []
        for record, prepared in zip(records, prepared_rows):
            retired_files.extend(dict(path=path, kind=kind, bytes=100) for path, kind in (
                (record['metadata']['capture']['feature_path'], 'native_capture'),
                (prepared['feature_path'], 'prepared_feature'),
                (prepared['terminal_feature']['tensor_path'], 'terminal_feature')))
        retirement = dict(status='planned', event_id='event_001', files=retired_files,
            compact_evidence=compact, retained_current_state=state_path, retained_current_export=export)
        write_json(event_root / 'RETIREMENT_PLAN.json', retirement)
        write_json(event_root / 'RETIREMENT.json', dict(retirement, status='completed',
            removed=retired_files, removed_bytes=38400))
        plan = dict(draft='/F', checkpoints={'K': str(base / 'trainer/events/event_058/export')})
        checked = A._audit_event(ArchiveReader(self.root), base, plan, 'K', 1, inputs, None)
        self.assertEqual(checked['raw_online_generations'], 128)
        self.assertEqual(checked['retirement']['files'], 384)
        self.assertEqual(checked['exposure']['selection_totals']['selected_position_occurrences'], 512)
        self.assertEqual(checked['resources'], resource)
        self.assertAlmostEqual(checked['costs']['controller_serving_s'], 140.8)
        self.assertEqual(checked['costs']['blocking_update_s'], 50.)
        # A self-consistent status/result label cannot hide a corrupt original token.
        first = event_root / 'loop/serving/r0.json'
        wrong = json.loads(first.read_text())
        wrong['response_ids'][1] += 1
        write_json(first, wrong)
        with self.assertRaisesRegex(RuntimeError, 'committed tokens'):
            A._audit_event(ArchiveReader(self.root), base, plan, 'K', 1, inputs, None)

    def test_incomplete_run_preserves_failure_report_before_raising(self):
        with self.assertRaisesRegex(ValueError, 'preserved'):
            A.audit_training(self.root)
        receipt = json.loads((self.root / 'campaign_001/TRAINING_AUDIT.json').read_text())
        self.assertEqual(receipt['status'], 'invalid_or_incomplete')
        self.assertTrue(receipt['errors'])
        self.assertEqual(receipt['extra_model_work'], 0)

    def test_hard_killed_worker_preserves_partial_audit_and_unavailable_clocks(self):
        sealed = self.root / 'sealed'
        write_json(sealed / 'ORIGINAL_MANIFEST.json', {'train_requests': []})
        write_json(sealed / 'CANONICALS.json', {})
        # Deliberately incomplete: the audit must report missing scientific
        # evidence without requiring a fabricated complete-run fixture.
        write_json(sealed / 'PLAN.json', dict(manifest=str(sealed / 'ORIGINAL_MANIFEST.json'),
            canonicals=str(sealed / 'CANONICALS.json'), checkpoints={
                arm: str(self.root / 'campaign_001/arms' / arm / 'trainer/events/event_058/export')
                for arm in A.ARMS}, evaluation_requests=[], binding='a'*64, draft='/F'))
        write_json(self.root / 'campaign_001/arms/K/STATUS.json', dict(status='running',
            segments=[dict(resume=False, backend_loading_s=1.2, trainer_loading_s=2.3)], event_costs=[]))
        result = A.audit_training(self.root, raise_on_error=False)
        self.assertEqual(result['status'], 'invalid_or_incomplete')
        self.assertEqual(result['arms']['K']['costs']['loading_s'], 3.5)
        for name in ('online_wall_s', 'initial_validation_s', 'publication_validation_s', 'retention_s'):
            self.assertIsNone(result['arms']['K']['costs'][name])
            self.assertIn(name, result['arms']['K']['unavailable_costs'])
            self.assertTrue(any('unavailable cost ' + name in error for error in result['errors']))
        self.assertEqual(result['arms']['K']['actual_work']['completed']['optimizer_steps'], 0)
        saved = json.loads((self.root / 'campaign_001/TRAINING_AUDIT.json').read_text())
        self.assertEqual(saved, result)
        with self.assertRaisesRegex(ValueError, 'preserved'):
            A.audit_training(self.root)

    def test_missing_clock_is_not_a_measured_zero(self):
        self.assertEqual(A.sum_recorded_costs([0.], 'clock'), 0.)
        with self.assertRaisesRegex(ValueError, 'No recorded clock'):
            A.sum_recorded_costs([], 'clock')


if __name__ == '__main__':
    unittest.main()
