"""CPU fixtures for report-only decisions, publication privacy and failure gates."""
import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('build_final_report.py')
SPEC = importlib.util.spec_from_file_location('fullrun_report', SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def primary(point, lower=-.2, upper=.2):
    return {'contrasts': {'TAIL_vs_BASE': {'request_latency_s_degradation_pct':
            dict(value=point, ci95=[lower, upper])}}}


def event_fixture(arm='TAIL', event=1):
    components = dict(denominator=2., effective_FKL_numerator=2., effective_PAIR_numerator=1.,
        effective_total_numerator=3., effective_FKL=1., effective_PAIR=.5, effective_total=1.5,
        reconciliation_error=0., reconciliation_tolerance=1e-5)
    continuation = dict(continuation_active_response=True, continuation_changed_high_positions=2,
        continuation_moved_weight=.01, continuation_high_count=2, continuation_low_count=1,
        continuation_kappa=.2, continuation_fk_correction=.1, continuation_pair_correction=.02)
    records = []
    for pass_index in (0, 1):
        for ordinal in range(128):
            observation = dict(record_id=f'PRIVATE_SOURCE_{ordinal}', pass_index=pass_index,
                               effective_components=copy.deepcopy(components))
            if arm == 'TAIL':
                observation['continuation_repair'] = copy.deepcopy(continuation)
            records.append(dict(response_observation=observation))
    result = dict(method=arm, event_id=f'event_{event:03d}', status='completed',
        response_presentations=256, optimizer_steps=128, start_step=(event-1)*128,
        end_step=event*128, records=records)
    resource = dict(training_peak_allocated_bytes=100, training_peak_reserved_bytes=200)
    feedback_costs = [dict(serve_call_s=1.2, buffer_persistence_s=.1,
        generation_validation_and_recording_s=1.1, native_request_latency_s=1.) for _ in range(128)]
    feedback = [dict(response_ids=[7, 8, 9], metadata=dict(verification_calls=1,
        accepted_draft_tokens=1, committed_progress=2, trace=[dict(native_accepted_draft_tokens=1)],
        native_generate_s=.8, capture=dict(cache_maintenance_s=.1))) for _ in range(128)]
    costs = {key: 1. for key in ('training_s', 'checkpoint_save_s', 'export_s', 'update_call_s',
        'teacher_preparation_s', 'feature_persistence_s', 'retirement_s', 'compact_persistence_s')}
    costs['activation'] = dict(activation_s=1.)
    return result, resource, feedback_costs, feedback, costs


def complete_fixture(root):
    def save(path, value):
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        report.write(destination, value)
        return destination
    plan = dict(execution_commit='1'*40, root='/remote/immutable/private-run', source_identity=dict(bytes=1, sha256='2'*64))
    names = ('original_config', 'runtime_pin', 'training_pin', 'model_pin', 'manifest',
             'evaluation_inputs', 'data_seal', 'exposure_inventory', 'witness_selection', 'warmup_canonicals')
    plan.update({name: '/private/' + name for name in names})
    plan.update(configs={arm: '/private/' + arm + '.yaml' for arm in report.ARMS},
        formula_code_identities={}, initialization='released_fresh_independently_per_arm',
        original_training_order_digest='a'*64, serving={}, limits={})
    plan['sealed_files'] = {plan[name]: dict(bytes=1, sha256='a'*64) for name in names}
    plan['sealed_files'].update({path: dict(bytes=1, sha256='a'*64) for path in plan['configs'].values()})
    plan['binding'] = report.digest(plan)
    plan_path = save('PLAN.json', plan)
    completed = {'production.'+key: value for key, value in report.WORK.items()}
    completed.update({'production.activation': 49, 'production.witness_response_forward': 72,
                      'qualification.adam': 4, 'qualification.generation': 8})
    save('WORK.json', dict(attempted=completed, completed=completed, pending={}, failures=[],
        qualification_limits={'adam': 8, 'generation': 16}, production_targets=report.WORK))
    save('CLOCK.json', dict(active={}, completed_card_s=10000., first_model_load_epoch=1., finished=[{}]))
    save('STATUS.json', dict(status='completed', clock=dict(elapsed_s=20000.)))
    save('qualification/QUALIFICATION.json', dict(status='passed', execution_commit=plan['execution_commit']))
    for arm in report.ARMS:
        save(f'campaign_001/arms/{arm}/STATUS.json', dict(status='completed', events_completed=58,
            global_step=7424, presentations=14848, binding=plan['binding'], execution_commit=plan['execution_commit'],
            event_costs=[dict(event=e, costs=event_fixture(arm, e)[-1]) for e in range(1, 59)],
            cleanup_errors=[], worker_wall_s=5000., backend_loading_s=3.))
        save(f'campaign_001/arms/{arm}/startup_000/RESULT.json', dict(status='completed', wall_s=2.))
    save('canonicals/STATUS.json', dict(status='completed', completed=1024, invalid=0, unknown=0, unattempted=0))
    save('evaluation/STATUS.json', dict(status='completed', completed_valid=12384, invalid=0, unknown=0, unattempted=0))
    for folder, count in (('canonicals', 1), ('evaluation', 48)):
        (root / folder / 'ACTIVATIONS.jsonl').write_text('{}\n' * count)
    paired, rows = [], []
    sample = dict(request_latency_s=1., native_generate_s=.8, cache_maintenance_s=.1,
        output_tokens=3, verification_calls=1, accepted_draft_tokens=1,
        native_accepted_draft_tokens=1, committed_progress=2, finish_reason='eos',
        repeat_times={key: [v]*3 for key, v in zip(report.TIMES, (1., .8, .1))})
    for ordinal in range(1024):
        item = {'request_id': f'PRIVATE_EVAL_{ordinal}'}
        item.update({method: copy.deepcopy(sample) for method in report.METHODS})
        paired.append(item)
        for method in report.METHODS:
            for repeat in range(3):
                rows.append(dict(request_id=item['request_id'], phase='scored', method=method,
                    activation=0, repeat=repeat, group=ordinal//256, valid_measurement=True,
                    **{key: sample[key] for key in (*report.TIMES, *report.COUNTS, 'finish_reason')}))
    for activation in range(48):
        for ordinal in range(2):
            rows.append(dict(request_id=f'PRIVATE_WARMUP_{ordinal}', phase='warmup', method='F',
                activation=activation, repeat=activation//16, group=(activation//4)%4, valid_measurement=True,
                **{key: sample[key] for key in (*report.TIMES, *report.COUNTS, 'finish_reason')}))
    (root / 'evaluation/MEASUREMENTS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    totals = {method: dict(**{key: sample[key]*1024 for key in (*report.TIMES, *report.COUNTS)},
        repeat_totals={key: [sample[key]*1024]*3 for key in report.TIMES}) for method in report.METHODS}
    contrasts = {}
    for name in report.CONTRASTS:
        candidate, baseline = name.split('_vs_')
        value = dict(candidate=candidate, baseline=baseline, wins=0, regressions=0, ties=1024,
            gross_calls_saved=0, gross_calls_added=0, net_calls_saved=0)
        value.update({key: dict(value=0., ci95=[-.2, .2]) for key in
            (*[key+'_degradation_pct' for key in report.TIMES], 'verification_call_reduction_pct')})
        contrasts[name] = value
    final = dict(execution_commit=plan['execution_commit'], requests=1024, paired=paired, totals=totals,
        contrasts=contrasts, bootstrap=dict(draws=10000, generator='PCG64', seed=20261126,
            percentiles='linear 2.5/97.5', shared_indices_across_endpoints_metrics=True,
            input_order=[r['request_id'] for r in paired]),
        parameter_identities={m: dict(sha256=str(i)*64) for i,m in enumerate(report.METHODS)},
        serving_costs=dict(loading_s=2., activation_s=48., actual_scoring_phase_activations=48))
    final['decision'] = dict(report.decision(final), status='completed', identical_serving_weights=False,
                             identical_native_trajectories=False)
    save('FINAL_ANALYSIS.json', final)
    audit_paths = {}
    for name in ('serving_audit', 'work_audit', 'resource_audit', 'postflight'):
        audit_paths[name] = str(save(name+'.json', dict(status='passed', plan_binding=plan['binding'],
            execution_commit=plan['execution_commit'])))
    for endpoint in report.EXPECTED_WITNESSES:
        blocks = [] if endpoint == 'released' else [dict(Y=2, raw_repair=True, useful_repair=True,
            reachable_origin_followers=1, lost_prefix=0, damaged_prefix_positions=0, prefix_damaged=False)]
        save(f'witness/{endpoint}/STATUS.json', dict(status='completed', training_state_unchanged=True,
            state=dict(status='passed'), responses=[{}]*8, paired_blocks=blocks,
            event=0 if endpoint=='released' else int(endpoint.split('_')[1]),
            work=dict(draft_forwards_completed=8), wall_s=1.))
    return argparse.Namespace(plan=str(plan_path), raw_root=str(root), output=str(root/'public'),
        cleanup=None, zstd='/home/minhao2/miniconda3/bin/zstd', **audit_paths)


class ReportTests(unittest.TestCase):
    def test_point_rule_does_not_require_significance(self):
        result = report.decision(primary(.09772))
        self.assertEqual(result['selected'], 'TAIL')
        self.assertFalse(result['degradation_bound_supported'])
        self.assertFalse(result['speed_improvement_supported'])

    def test_unrounded_point_above_threshold_selects_base(self):
        self.assertEqual(report.decision(primary(.1000000000001))['selected'], 'BASE')

    def test_threshold_and_zero_boundary_are_distinct(self):
        result = report.decision(primary(.1, -.1, .1))
        self.assertTrue(result['practical_rule_met'])
        self.assertTrue(result['degradation_bound_supported'])
        self.assertFalse(report.decision(primary(-.1, -.2, 0.))['speed_improvement_supported'])
        self.assertTrue(report.decision(primary(-.1, -.2, -1e-12))['speed_improvement_supported'])

    def test_nonfinite_is_not_a_result(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            report.decision(primary(float('nan')))

    def test_event_counts_original_presentations_once(self):
        args = event_fixture()
        activity, costs = report.summarize_event('TAIL', 1, *args)
        self.assertEqual(activity['component_checks'], 256)
        self.assertEqual(activity['continuation_changed_high_positions'], 512)
        self.assertEqual(activity['effective_total_mean'], 1.5)
        self.assertEqual(costs['feedback_serve_call_s'], 153.6)
        self.assertEqual(activity['feedback_verification_calls'], 128)

    def test_duplicate_presentation_and_component_error_block(self):
        args = event_fixture()
        args[0]['records'][1] = args[0]['records'][0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            report.summarize_event('TAIL', 1, *args)
        args = event_fixture()
        args[0]['records'][0]['response_observation']['effective_components']['effective_total_numerator'] += 1.
        with self.assertRaisesRegex(ValueError, 'reconcile'):
            report.summarize_event('TAIL', 1, *args)

    def test_unicode_plan_digest_matches_native_serialization(self):
        import hashlib
        encoded = json.dumps({'text': '数学'}, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
        self.assertEqual(report.digest({'text': '数学'}), hashlib.sha256(encoded).hexdigest())

    def test_full_report_fixture_and_private_data_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = complete_fixture(root)
            def archive(_root, arm, event, _zstd):
                result, resource, costs, feedback, _ = event_fixture(arm, event)
                summary = dict(arm=arm, event=event, archive_bytes=10, archive_sha256='a'*64,
                    original_member_count=3, removed_loose_files=3, removed_loose_allocated_bytes=30, wall_s=.1)
                return result, resource, costs, feedback, summary
            with patch.object(report, 'archived_event', side_effect=archive):
                output = report.build(args)
            self.assertEqual(report.read(output/'BUILD_STATUS.json')['status'], 'completed')
            self.assertIn('bound remains unresolved', (output/'REPORT.md').read_text())
            self.assertIn('Execution commit: `'+'1'*40+'`', (output/'REPORT.md').read_text())
            for path in output.iterdir():
                text = path.read_text()
                self.assertNotIn('PRIVATE_', text, path.name)
                self.assertNotIn('response_ids', text, path.name)
                self.assertNotIn('sampled_anchors', text, path.name)
            self.assertEqual(len((output/'EVENT_TRAINING.csv').read_text().splitlines()), 117)
            self.assertEqual(len((output/'ALL_REPEATS.csv').read_text().splitlines()), 12289)

    def test_missing_final_state_leaves_only_blocked_build(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = complete_fixture(root)
            (root/'campaign_001/arms/TAIL/STATUS.json').unlink()
            with self.assertRaises(FileNotFoundError):
                report.build(args)
            self.assertEqual(report.read(Path(args.output)/'BUILD_STATUS.json')['status'], 'blocked')
            self.assertFalse((Path(args.output)/'REPORT.md').exists())

    def test_stale_audit_cannot_publish_report(self):
        with tempfile.TemporaryDirectory() as folder:
            args = complete_fixture(Path(folder))
            receipt = report.read(args.work_audit)
            receipt['plan_binding'] = '0'*64
            report.write(args.work_audit, receipt)
            with self.assertRaisesRegex(ValueError, 'actual plan'):
                report.build(args)
            self.assertFalse((Path(args.output)/'REPORT.md').exists())

    def test_activation_total_includes_canonical(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = complete_fixture(root)
            report.completed_evidence(root, report.read(args.plan))
            work = report.read(root/'WORK.json')
            work['completed']['production.activation'] = 48
            report.write(root/'WORK.json', work)
            with self.assertRaisesRegex(ValueError, 'activation'):
                report.completed_evidence(root, report.read(args.plan))

    def test_witness_identity_failure_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            complete_fixture(root)
            path = root/'witness/TAIL_058/STATUS.json'
            row = report.read(path)
            row['paired_blocks'][0]['Y'] += 1
            report.write(path, row)
            with self.assertRaisesRegex(ValueError, 'decomposition'):
                report.witness_tables(root)

    def test_archive_reader_authenticates_exact_members(self):
        zstd = '/home/minhao2/miniconda3/bin/zstd'
        if not Path(zstd).exists():
            self.skipTest('zstd is unavailable')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, resource, costs, feedback, _ = event_fixture()
            prefix = 'campaign_001/arms/TAIL'
            members = {f'{prefix}/trainer/events/event_001/RESULT.json': result,
                       f'{prefix}/events/event_001/RESOURCE_COSTS.json': resource}
            for i in range(128):
                members[f'{prefix}/events/event_001/loop/serving/{i}.json'] = feedback[i]
                members[f'{prefix}/events/event_001/loop/serving/{i}.costs.json'] = costs[i]
            archive = root/'compact/TAIL/event_001.tar.zst'
            archive.parent.mkdir(parents=True)
            stream = io.BytesIO()
            index = dict(arm='TAIL', event=1, members={})
            with tarfile.open(fileobj=stream, mode='w') as tar:
                for name, value in members.items():
                    data = json.dumps(value).encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                    index['members'][name] = dict(bytes=len(data), sha256=report.hashlib.sha256(data).hexdigest())
            compressed = subprocess.run([zstd, '-q', '-c'], input=stream.getvalue(), capture_output=True, check=True).stdout
            archive.write_bytes(compressed)
            index.update(archive_bytes=len(compressed), archive_sha256=report.sha256(archive))
            index['binding_digest'] = report.digest(index)
            report.write(str(archive)+'.index.json', index)
            report.write(str(archive)+'.retirement.json', dict(status='completed', removed_paths=[], removed_allocated_bytes=0, wall_s=1.))
            recovered = report.archived_event(root, 'TAIL', 1, zstd)
            self.assertEqual(len(recovered[2]), 128)
            self.assertEqual(recovered[0], result)
            archive.write_bytes(compressed[:-1] + bytes([compressed[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'archive identity'):
                report.archived_event(root, 'TAIL', 1, zstd)


if __name__ == '__main__':
    unittest.main()
