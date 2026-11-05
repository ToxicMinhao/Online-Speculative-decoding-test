"""Count provenance, ordinary-response accounting and failure-safe report tests."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.o_half_matched_lr import report as R
from experiments.o_half_matched_lr import statistics as S
from experiments.ok_continuation.test_resume import event as training_event


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def fixture(count=4):
    requests=[dict(request_id=f'r{i}',panel_ordinal=i) for i in range(count)]
    archive=dict(status='passed',source_root='/archive',source_commit='a'*40,panel_binding='binding',
        requests=requests,canonicals={r['request_id']:dict(response_ids=list(range(120)),finish_reason='eos') for r in requests},
        checkpoints={str(seed):{m:'/'+m+str(seed) for m in R.ARCHIVED_METHODS} for seed in S.SEEDS},
        counts={str(seed):{m:{r['request_id']:dict(verification_calls=20+r['panel_ordinal'],
            accepted_draft_tokens=100,committed_progress=119,output_tokens=120) for r in requests}
            for m in R.ARCHIVED_METHODS} for seed in S.SEEDS})
    new=[]
    for seed in S.SEEDS:
        for method in R.NEW_METHODS:
            for r in requests:
                for repeat in (0,1):
                    new.append(dict(seed=seed,event=4,phase='final',**r,method=method,repeat=repeat,
                        verification_calls=21+r['panel_ordinal'],accepted_draft_tokens=99,committed_progress=119,
                        output_tokens=120,cap_hit=False,finish_reason='eos',valid_measurement=True,
                        validation_status='valid',errors=[],checkpoint_id='/'+method+str(seed),
                        record_path=f'/new/{seed}/{method}/{r["request_id"]}/{repeat}.json',
                        request_latency_s=1.0,native_generate_s=.9,cache_maintenance_s=.1))
    return archive,new


def response_event(event=1):
    result=training_event(event,'osd')
    for ordinal,row in enumerate(result['records']):
        loss=1. if ordinal%2==0 else 3.
        row.update(training_s=.01,learning_rate=1.5e-4 if row['optimizer_stepped'] else None,
            metrics=dict(loss=999.,grad_norm=.5),
            response_observation=dict(record_id=row['record_id'],scope='one_original_response_forward',
                loss=loss,ratio_metrics={'objective_loss':[loss*5,5]},
                sum_metrics={'supervised_position_exposure':7.}))
    return result


class MatchedLRReportTests(unittest.TestCase):
    def test_join_uses_only_counts_with_explicit_old_new_provenance(self):
        archive,new=fixture()
        joined=R.join_counts(archive,new)
        self.assertEqual(len(joined),8)
        for row in joined:
            for method in R.METHODS:
                self.assertFalse(any(name in row[method] for name in R.TIMINGS))
                self.assertEqual(row[method]['verified_repeats'],3 if method in R.ARCHIVED_METHODS else 2)
                self.assertEqual(row[method]['provenance'],
                    'archived_PR21_repeat_0' if method in R.ARCHIVED_METHODS else 'new_final_repeat_0')
        self.assertEqual(S.summarize(joined)['request_count'],4)

    def test_join_rejects_bridge_missing_duplicate_invalid_counts_and_wrong_canonical(self):
        archive,new=fixture()
        for mutation in ('bridge','missing','duplicate','invalid','count','ordinal','canonical'):
            a,n=copy.deepcopy(archive),copy.deepcopy(new)
            if mutation=='bridge':n[0]['phase']='bridge'
            elif mutation=='missing':n.pop()
            elif mutation=='duplicate':n.append(n[0])
            elif mutation=='invalid':n[0]['valid_measurement']=False
            elif mutation=='count':n[1]['verification_calls']+=1
            elif mutation=='ordinal':n[0]['panel_ordinal']=3
            else:a['canonicals']['r0']['response_ids'].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):R.join_counts(a,n)

    def test_ordinary_response_metrics_do_not_require_K_selection(self):
        receipt=response_event()
        responses,windows=R.response_observations(receipt,seed=17,event=1)
        self.assertEqual((len(responses),len(windows)),(256,128))
        self.assertNotIn('prefix_protection',responses[0])
        self.assertNotIn('selected_position_occurrences',responses[0]['sum_metrics'])
        aggregates=R.aggregate_training([dict(valid=True,response_records=responses)])
        self.assertEqual(aggregates['17/1']['mean_normalized_response_loss'],2.)
        self.assertEqual(aggregates['17/1']['summed_exposure']['supervised_position_exposure'],1792.)
        self.assertEqual(aggregates['17/1/pass_0']['responses'],128)
        receipt['records'][0]['response_observation']['prefix_protection']={}
        with self.assertRaisesRegex(ValueError,'scope'):R.response_observations(receipt,seed=17,event=1)

    def test_evaluation_accounting_retains_phase_and_no_old_timing_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'campaign_001/bridge_seed_17';folder.mkdir(parents=True)
            write_json(folder/'STATUS.json',dict(seed=17,phase='bridge',status='failed',attempted_generations=1))
            (folder/'MEASUREMENTS.jsonl').write_text(json.dumps(dict(request_latency_s=2.,native_generate_s=1.,cache_maintenance_s=.1))+'\n')
            rows,costs=R.evaluation_accounting(root)
            self.assertEqual(len(rows),1)
            self.assertEqual(costs[0]['returned_timing_totals']['request_latency_s'],2.)
            self.assertEqual(costs[0]['phase'],'bridge')
            self.assertFalse(costs[0]['latency_comparison_supported'])
            self.assertEqual(costs[-1]['status'],'unattempted')

    def test_costs_keep_historical_arms_unmatched(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan={'original_run':str(root/'old'),'pr21_run':str(root/'pr21')}
            for name,methods in [('old',('O','K')),('pr21',('K_HALF',))]:
                folder=root/name/'final_report';folder.mkdir(parents=True)
                rows=[dict(seed=s,event=e,arm=m,training_s=1.,event_wall_s=2.,
                    loading_initialization_resume_validation_s=3.,checkpoint_save_s=4.,export_s=5.)
                    for s in S.SEEDS for e in range(1,5) for m in methods]
                R.write_csv(folder/'TRAINING_COSTS.csv',rows)
            events=[dict(seed=17,event=1,valid=True,costs=dict(training_s=10.,event_wall_s=12.,
                peak_allocated_bytes=123,objective_work={'teacher_head_s':2.,'scope':'nested'}))]
            summary=R.cost_summary(events,plan)
            self.assertEqual(summary['current_O_HALF']['17']['training_s'],10.)
            self.assertEqual(summary['historical_unmatched']['17/O']['training_s'],4.)
            self.assertEqual(summary['historical_unmatched']['29/K_HALF']['status'],'historical_unmatched')
            self.assertIn('Not supported',summary['comparative_latency'])
            self.assertFalse(any('reduction' in name for name in summary))

    def test_render_is_count_only_and_labels_reused_panel_and_E(self):
        a,n=fixture();analysis=S.summarize(R.join_counts(a,n))
        summary=dict(analysis=analysis,interpretation=analysis['interpretation'],
            execution_source={'commit':'a'*40},errors=[],costs={'current_O_HALF':{}})
        text=R.render_report(summary)
        self.assertIn('not fresh confirmation',text)
        self.assertIn('No comparative serving latency',text)
        self.assertIn('common F-call denominator',text)
        self.assertNotIn('Request-time reduction',text)
        self.assertIn('O_HALF/K_HALF',text)
        self.assertIn('256 seed/request outcomes',text)
        summary.update(analysis=None,interpretation=S.interpret({}, {},valid=False),errors=['failed trace'])
        text=R.render_report(summary)
        self.assertIn('Decision E',text)
        self.assertIn('failed trace',text)
        self.assertNotIn('Paired 95% interval',text)

    def test_complete_report_with_real_cpu_evaluator_and_worker_receipts(self):
        from test_o_half_evaluation import FullBackend, fixture_archive
        from test_ok_coding_campaign import requests
        from test_ok_coding_report import complete_work
        from experiments.o_half_matched_lr import evaluation as E
        from experiments.o_half_matched_lr import campaign as C
        from experiments.ok_coding import campaign as original
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);inputs=requests();archive=fixture_archive(inputs)
            archive.update(source_root='/archive',source_commit='b'*40,panel_binding='binding')
            plan=dict(execution_source={'commit':'a'*40},original_schedules={'17':{},'29':{}},
                original_run=str(root/'old'),pr21_run=str(root/'pr21'),original_endpoints={})
            for seed in S.SEEDS:
                plan['original_endpoints'][str(seed)]={'O':{'export':f'/O{seed}'}}
                for event in range(1,5):
                    schedule=root/'original'/f'{seed}_{event}.json'
                    write_json(schedule,training_event(event,'osd'))
                    plan['original_schedules'][str(seed)][str(event)]=str(schedule)
                    base=root/'campaign_001'/f'seed_{seed}'
                    folder=base/f'resume_{event:03d}_O_HALF'
                    limits=dict(forwards=256,backwards=256,optimizer_steps=128)
                    complete_work(folder/'WORK.jsonl',limits)
                    startup=dict(pid=seed*100+event,cuda_visible_devices='1',device_uuid='same',
                        cuda_initialized_on_entry=False,warmup_model_work=0,
                        caches={key:{'path':str(folder/key),'initially_empty':True} for key in original.CACHE_VARIABLES})
                    write_json(folder/'WORKER.json',dict(status='completed',additional_qualification_steps=0,
                        startup=startup,worker_wall_s=4.,work=original.read_work(folder/'WORK.jsonl',limits)))
                    write_json(folder/'EXEC.json',dict(returncode=0,wall_s=5.))
                    write_json(folder/'LIVE.json',{'exact_resume':True})
                    write_json(folder/'LIVE_BOUNDARY.json',{'exact_resume':True})
                    write_json(folder/'SAVED_ENDPOINT.json',dict(status='passed',export_exact=True,global_step=event*128))
                    write_json(folder/'CLOSED.json',dict(closed=True,failed=False,accumulation_remainder=0,
                        global_step=event*128,presentations=event*256))
                    write_json(folder/'MIGRATION.json',dict(status='passed',source_weights_rng_moments_counters_history_exact=True))
                    write_json(folder/'STATE_IDENTITIES.json',{})
                    write_json(folder/'COSTS.json',dict(training_s=2.56,event_wall_s=3.,response_observations=256,
                        reference_operations=0,objective_work={'work':{'response_forwards':256},'teacher_head_s':1.}))
                    receipt=response_event(event);receipt['export_path']=f'/OH{seed}'
                    write_json(base/'O_HALF/events'/f'event_{event:03d}'/'RESULT.json',receipt)
                E.evaluate_bridge(FullBackend(inputs),inputs,archive['checkpoints'][str(seed)],
                    root/'campaign_001'/f'bridge_seed_{seed}',seed=seed,archive=archive)
                E.evaluate_final(FullBackend(inputs),inputs,{'O':f'/O{seed}','O_HALF':f'/OH{seed}'},
                    root/'campaign_001'/f'seed_{seed}'/'event_004/evaluation',seed=seed,archive=archive)
            write_json(root/'sealed/PLAN.json',plan)
            write_json(root/'campaign_001/STATUS.json',{'status':'completed',**C.reconcile(root)})
            write_json(root/'campaign_001/COSTS.json',{'native_loading_s':123.})
            with patch('experiments.o_half_matched_lr.contract.validate_plan'),patch.object(R,'audit_archive_reuse',return_value=archive):
                summary=R.write_campaign_report(root,root/'report')
            self.assertEqual(summary['errors'],[])
            self.assertEqual(summary['analysis']['request_count'],128)
            self.assertEqual(summary['interpretation']['branch'],'B')
            self.assertFalse(summary['latency_comparison_supported'])
            self.assertEqual(summary['campaign_status']['evaluation_valid'],1072)
            self.assertEqual(len(csv_rows(root/'report/ALL_NEW_EVALUATIONS.csv')),1072)
            self.assertEqual(len(csv_rows(root/'report/PAIRED_REQUESTS.csv')),256)
            self.assertEqual(len(csv_rows(root/'report/TRAINING_RESPONSES.csv')),2048)
            self.assertEqual(json.loads((root/'report/EVALUATION_COSTS.json').read_text())['campaign_operational_costs']['native_loading_s'],123.)
            # Reporting must independently reject a record that still claims valid.
            path=root/'campaign_001/seed_17/event_004/evaluation/0000_r0_O.json'
            bad=json.loads(path.read_text());bad['metadata']['trace'][0]['draft_tokens'][0]+=9
            write_json(path,bad)
            with patch('experiments.o_half_matched_lr.contract.validate_plan'),patch.object(R,'audit_archive_reuse',return_value=archive):
                invalid=R.write_campaign_report(root,root/'invalid_report')
            self.assertEqual(invalid['interpretation']['branch'],'E')
            self.assertIsNone(invalid['analysis'])
            self.assertEqual(len(csv_rows(root/'invalid_report/ALL_NEW_EVALUATIONS.csv')),1072)


if __name__=='__main__':unittest.main()
