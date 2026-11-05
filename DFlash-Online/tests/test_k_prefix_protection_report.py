"""Audit complete synthetic native records, reject tampering, and keep losses response-scoped."""
import json
from pathlib import Path
import tempfile
import unittest

from online_feedback.storage import write_json
from experiments.k_prefix_protection_v1 import report as R
from experiments.k_prefix_protection_v1.contract import (
    ARMS, CELLS, METHODS, NEW_BUDGET, QUALIFICATION_BUDGET, ORIGINAL_COMMIT,
)
from experiments.k_prefix_protection_v1.prepare import UPSTREAM_BINDING, evaluation_schedule
from experiments.ok_continuation.test_resume import event as training_event
from experiments.ok_coding import campaign as old_campaign
from experiments.osd_persistent.run import score_record
from test_ok_coding_campaign import requests
from test_ok_coding_evaluation import Backend
from test_ok_coding_report import complete_work
from test_osd_one_event import publication


def protection_record(loss=1.):
    return dict(protection_evaluated=True, finite=True, qualification_forced_active=False,
        k_loss=loss, k_numerator=loss * 3, k_selected_weight=3., k_denominator=3.000001,
        protection_numerator=0., protection_weight=2., protection_denominator=2.000001,
        protection_loss=0., weighted_protection_loss=0., total_loss=loss,
        available_blocks=2, protected_blocks=1, empty_protected_blocks=1,
        protected_positions=2, protected_prefix_histogram_denominator=2,
        protected_prefix_histogram=[1, 0, 1], active_positions=0, active_weight=0.,
        incorrect_protected_positions=0, wrong_tie_positions=0, current_zero_margin_positions=0,
        reference_margins=dict(count=2, sum=2., min=1., max=1., zero_margin_count=0),
        current_margins=dict(count=2, sum=2., min=1., max=1.),
        thresholds=dict(count=2, sum=1., min=.5, max=.5), selection_statistics=[2, 1])


def training_receipt(event, arm):
    result = training_event(event, 'osd_first_error')
    for i, row in enumerate(result['records']):
        loss = 1. if i % 2 == 0 else 3.
        row.update(training_s=.01, learning_rate=.0003 if arm == 'KP' else .00015,
                   metrics={'grad_norm': .5, 'loss': 999.},
                   response_observation={'record_id': row['record_id'],
                     'scope': 'one_original_response_forward', 'loss': loss,
                     'ratio_metrics': {'loss': [loss, 1.]},
                     'sum_metrics': {'available_position_occurrences': 3.,
                                     'selected_position_occurrences': 2.}})
        if arm == 'KP':
            row['response_observation']['prefix_protection'] = protection_record(loss)
    return result


def fixture(root):
    inputs = requests()
    fixed = dict(new_budget=dict(NEW_BUDGET), qualification_budget=dict(QUALIFICATION_BUDGET),
        training_binding={'upstream_data_binding': UPSTREAM_BINDING}, original_execution_commit=ORIGINAL_COMMIT,
        training_seeds=[17, 29], stop_step=512, inherited_horizon=7424, no_automatic_retry=True,
        evaluation_requests=inputs, evaluation_schedule=evaluation_schedule(inputs),
        configs={str(seed): {arm: '/config' for arm in ARMS} for seed in (17, 29)},
        pools={str(seed): {str(event): {} for event in range(1, 5)} for seed in (17, 29)},
        original_k4={str(seed): {'state': '/state', 'export': f'/K{seed}'} for seed in (17, 29)},
        original_schedules={str(seed): {} for seed in (17, 29)},
        serving=dict(block_size=16, max_new_tokens=2048, context_limit=4096, concurrency=1,
                     temperature=0, capture=False, precision='bfloat16', attention_backend='fa3'),
        execution_source={'commit': 'a' * 40, 'dirty': False}, draft='/F')
    out = root / 'campaign_001'
    for seed in (17, 29):
        for event in range(1, 5):
            original_path = root / 'original' / f'{seed}_{event}.json'
            write_json(original_path, training_event(event, 'osd'))
            fixed['original_schedules'][str(seed)][str(event)] = str(original_path)
            for arm in ARMS:
                audit = out / f'seed_{seed}' / f'resume_{event:03d}_{arm}'
                limits = dict(forwards=256, backwards=256, optimizer_steps=128)
                if arm == 'KP':
                    limits['reference_forwards'] = 256
                complete_work(audit / 'WORK.jsonl', limits)
                startup = dict(pid=seed*100+event*2+(arm=='KP'), cuda_visible_devices='1', device_uuid='same',
                    cuda_initialized_on_entry=False, warmup_model_work=0,
                    caches={name: {'path': str(audit/name), 'initially_empty': True} for name in old_campaign.CACHE_VARIABLES})
                write_json(audit/'WORKER.json', dict(status='completed', additional_qualification_steps=0,
                    work=old_campaign.read_work(audit/'WORK.jsonl', limits), startup=startup))
                write_json(audit/'CLOSED.json', dict(closed=True, failed=False, accumulation_remainder=0, global_step=event*128, presentations=event*256))
                write_json(audit/'LIVE.json', {'exact_resume': True})
                write_json(audit/'LIVE_BOUNDARY.json', {'exact_resume': True})
                write_json(audit/'SAVED_ENDPOINT.json', dict(status='passed', export_exact=True, global_step=event*128))
                write_json(audit/'MIGRATION.json', dict(status='passed', source_weights_rng_moments_counters_history_exact=True))
                write_json(audit/'COSTS.json', dict(training_s=2., response_observations=256,
                    objective_work={'work': {'reference_response_forwards': 256 if arm=='KP' else 0}}))
                endpoint = out/f'seed_{seed}'/arm/'events'/f'event_{event:03d}'
                result = training_receipt(event, arm)
                result['export_path'] = str(endpoint/'export')
                write_json(endpoint/'RESULT.json', result)
    write_json(root/'sealed/PLAN.json', fixed)
    limits = {name: value for name, value in QUALIFICATION_BUDGET.items() if name != 'generations'}
    from experiments.k_prefix_protection_v1.qualification import CASES, case_limits, qualification_work
    for index, (name, _options, references) in enumerate(CASES):
        folder = root/'qualification_001'/name
        complete_work(folder/'WORK.jsonl', case_limits(references))
        write_json(folder/'CASE.json', dict(status='completed',
            work=old_campaign.read_work(folder/'WORK.jsonl', case_limits(references)),
            serialized_restored_exact=True, frozen_changes=[], global_step=513, presentations=1026,
            startup=dict(pid=100+index, device_uuid='same')))
    write_json(root/'qualification_001/RESULT.json', dict(status='passed', budget=limits,
        work=qualification_work(root)))
    write_json(root/'qualification_001/Q2/VERSUS_Q1.json', dict(weights={'exact': True},
        gradients={'exact': True}, optimizer_exact=True, state_differences=[], losses_exact=True,
        anchors_exact=True, response_metrics_exact=True))
    write_json(root/'qualification_001/Q4/FORCED_ACTIVE.json', dict(gradients={'exact': False}, moments={'exact': False}))
    canonicals = {}
    for seed, event in CELLS:
        folder = out/f'seed_{seed}'/f'event_{event:03d}'/'evaluation'
        folder.mkdir(parents=True)
        checkpoints = {'F': '/F', 'K': f'/K{seed}', **{
            arm: str(out/f'seed_{seed}'/arm/'events'/f'event_{event:03d}'/'export') for arm in ARMS}}
        backend = Backend(inputs)
        schedule = evaluation_schedule(inputs, seed=seed)
        rows = []
        for slot in schedule:
            ordinal, repeat, method = slot['panel_ordinal'], slot['repeat'], slot['method']
            request = inputs[ordinal]
            label = f'{ordinal:04d}_r{repeat}_{method}'
            path = folder/(label+'.json')
            backend.checkpoint_id = checkpoints[method]
            record = backend.generate(request['prompt_ids'], max_new_tokens=2048,
                request_id=label, request_index=ordinal, session_id='k_prefix_protection_v1_evaluation',
                purpose='evaluation', save_to=path)
            row = score_record(record, request_id=request['request_id'], repeat=repeat, method=method)
            row.update(**slot, label=label, record_path=str(path), finish_reason=record['finish_reason'],
                cache_maintenance_s=.1, structural_errors=[], errors=[], valid_measurement=True, validation_status='valid')
            rows.append(row)
            if method == 'F' and request['request_id'] not in canonicals:
                canonicals[request['request_id']] = dict(response_ids=record['response_ids'],
                    finish_reason=record['finish_reason'], source_record_path=str(path), source_method='F',
                    source_repeat=0, source_cell=[17, 4], F_checkpoint='/F',
                    F_counts={name: row[name] for name in R.COUNT_FIELDS})
        (folder/'MEASUREMENTS.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        write_json(folder/'PROTOCOL.json', dict(schedule=schedule, checkpoints=checkpoints))
        write_json(folder/'QUALIFICATION.json', dict(status='passed', requests=8, generations=96,
            methods_with_native_equality=sorted(METHODS), performance_win_required=False, extra_generations=0))
        write_json(folder/'STATUS.json', dict(status='completed', qualification='passed',
            attempted_generations=1536, returned_generations=1536, completed_valid_generations=1536,
            invalid_generations=0, pending_canonical_generations=0, unknown_generation_attempts=0, unattempted_generations=0))
        activations=[]
        for method in METHODS:
            write_json(folder/('WEIGHTS_'+method+'.json'), {'native_reply': backend.check_weights(checkpoints[method]), 'errors': []})
            activations.append({'method': method, **publication(checkpoints[method]), 'errors': []})
        (folder/'ACTIVATIONS.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in activations))
    write_json(out/'EVALUATION_CANONICAL.json', canonicals)
    from experiments.k_prefix_protection_v1.campaign import reconcile
    write_json(out/'STATUS.json', {'status': 'completed', **reconcile(root)})
    return fixed


class ReportTests(unittest.TestCase):
    def test_complete3072_raw_audit_and_invalid_measurement_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fixture(root)
            result=R.write_campaign_report(root, root/'report', execution_commit='a'*40)
            self.assertEqual(result['errors'], [])
            self.assertEqual(result['status'], 'valid')
            self.assertEqual(result['analysis']['request_count'], 128)
            self.assertEqual(len((root/'report/ALL_REPEATS.csv').read_text().splitlines()), 3073)
            self.assertEqual(len((root/'report/TRAINING_RESPONSES.csv').read_text().splitlines()), 4097)
            self.assertEqual(len((root/'report/TRAINING_WINDOWS.csv').read_text().splitlines()), 2049)
            path=root/'campaign_001/seed_29/event_004/evaluation/0000_r0_KP.json'
            record=json.loads(path.read_text()); record['metadata']['request_latency_s']=2.
            write_json(path, record)
            changed=R.write_campaign_report(root, root/'changed')
            self.assertEqual(changed['status'], 'invalid_or_incomplete')
            self.assertIsNone(changed['analysis'])
            self.assertIn('raw_measurement_disagreement', ' '.join(changed['errors']))
            self.assertEqual(len((root/'changed/PAIRED_REQUESTS.csv').read_text().splitlines()), 129)

    def test_response_mean_uses_premerge_values_and_windows_once(self):
        result=training_receipt(1,'KP')
        responses, windows=R.response_observations(result, seed=17, event=1, arm='KP')
        aggregate=R.summarize_observations(responses)
        self.assertEqual(aggregate['mean_total_response_loss'],2.)
        self.assertEqual(aggregate['mean_response_k_loss'],2.)
        self.assertEqual(aggregate['sum_metrics']['selected_position_occurrences'],512)
        self.assertEqual(len(windows),128)
        self.assertEqual(aggregate['protected_positions'],512)
        self.assertEqual(aggregate['protected_prefix_histogram'],[256,0,256])
        self.assertTrue(all(row['scope']=='optimizer_window' for row in windows))

    def test_invalid_training_event_does_not_enter_protection_aggregates(self):
        result=training_receipt(1,'KP')
        responses,windows=R.response_observations(result,seed=17,event=1,arm='KP')
        aggregate=R.aggregate_training([dict(valid=False,response_records=responses),
                                       dict(valid=True,response_records=responses)])
        self.assertEqual(aggregate['17/KP/1']['responses'],256)
        result['records'][0]['response_observation']['prefix_protection']['finite']=False
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            R.response_observations(result,seed=17,event=1,arm='KP')

    def test_accumulation_scope_and_reference_override_fail(self):
        result=training_receipt(1,'KP')
        result['records'][1]['response_observation']['scope']='accumulation_window'
        with self.assertRaisesRegex(ValueError,'scope'):
            R.response_observations(result,seed=17,event=1,arm='KP')
        result=training_receipt(1,'KP')
        result['records'][0]['response_observation']['prefix_protection']['qualification_forced_active']=True
        with self.assertRaisesRegex(ValueError,'forced-active'):
            R.response_observations(result,seed=17,event=1,arm='KP')


if __name__=='__main__':
    unittest.main()
