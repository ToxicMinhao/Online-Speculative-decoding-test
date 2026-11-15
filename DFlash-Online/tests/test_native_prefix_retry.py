"""Approved retry carries authenticated failed work without restarting its cap."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.native_prefix_loss import campaign, prepare, probe_client
from experiments.threatened_repair_replay.work import read_ledger


def write_ledger(path, *, steps=5, category='qualification'):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [dict(serial=i, kind='optimizer_step', category=category, phase=phase)
               for i in range(1, steps+1) for phase in ('started', 'completed')]
    records.append(dict(phase='ledger_closed', physical={}))
    path.write_text(''.join(json.dumps(row)+'\n' for row in records))


def prior_fixture(root, *, steps=5, category='qualification', seconds=203.0700637307018):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ledger = root/'training/PHYSICAL_WORK.jsonl'
    write_ledger(ledger, steps=steps, category=category)
    original = prepare.attempt_accounting()
    budget = dict(attempted={}, completed={}, failed={}, unknown={},
                  gpu_seconds=seconds, gpu_seconds_limit=10800.)
    status = dict(status='failed', phase='training_qualification_and_native_trajectory',
        execution_commit='fdccdf5482f2125f212476547522068f953c564a', budget=budget,
        budget_accounting=original, cumulative_gpu_seconds=seconds)
    costs = dict(native_budget=budget, training=read_ledger(ledger), prior_attempt=None,
        budget_accounting=original, cumulative=dict(completed_adam_steps=steps,
        attempted_adam_steps=steps, gpu_seconds=seconds, adam_limit=182, gpu_seconds_limit=10800))
    paths = dict(status=root/'STATUS.json', work_costs=root/'WORK_AND_COSTS.json', training_work=ledger)
    prepare.write(paths['status'], status)
    prepare.write(paths['work_costs'], costs)
    prior = dict(approved_cumulative_adam_limit=187)
    for name,path in paths.items():
        prior[name+'_path'] = str(path)
        prior[name+'_identity'] = prepare.identity(path)
    return prior


def retry_plan(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prior = prior_fixture(root.parent/'execution_001')
    prepare.write(root/'parent.json', {'metadata':{'trace':[1,2,3]}})
    prepare.write(root/'sealed/PROBE_KEYS.json', [])
    (root/'sealed/PROBE_KEYS.csv').write_text('request_id,canonical_start,round_ordinal\n')
    plan = dict(root=str(root), execution_commit='repaired-fixture', serving_gpu=0,
        prior_attempt=prior, budget_accounting=prepare.attempt_accounting(prior),
        probe_requests=[{'request_id':'r'}], parent_records={'r':str(root/'parent.json')},
        probe_keys=str(root/'sealed/PROBE_KEYS.json'))
    prepare.write(root/'sealed/PLAN.json', plan)
    return plan


class RetryAccountingTests(unittest.TestCase):
    def test_first_attempt_preserves_scientific_budget(self):
        accounting = prepare.attempt_accounting()
        self.assertEqual(accounting['current_attempt_adam_limit'],182)
        self.assertEqual(accounting['cumulative_adam_limit'],182)
        self.assertEqual(accounting['current_attempt_gpu_seconds_limit'],10800.)
        self.assertEqual(prepare.CONSTANTS['maximum_adam'],182)

    def test_authenticated_receipts_carry_exact_counts_and_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            prior = prior_fixture(Path(folder)/'execution_001')
            accounting = prepare.attempt_accounting(prior)
            self.assertEqual(accounting['prior_adam_steps'],5)
            self.assertEqual(accounting['prior_gpu_seconds'],203.0700637307018)
            self.assertEqual(accounting['current_attempt_adam_limit'],182)
            self.assertEqual(accounting['cumulative_adam_limit'],187)
            self.assertEqual(accounting['current_attempt_gpu_seconds_limit'],10800.-203.0700637307018)

    def test_all_three_evidence_files_need_existing_unchanged_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('status','work_costs','training_work'):
                for mutation in ('missing_pin','changed_bytes'):
                    with self.subTest(name=name,mutation=mutation):
                        prior = prior_fixture(Path(folder)/'execution_001')
                        if mutation == 'missing_pin':
                            prior.pop(name+'_identity')
                        else:
                            Path(prior[name+'_path']).write_text('{}')
                        with self.assertRaisesRegex(ValueError,'Missing retained identity|Retained artifact changed'):
                            prepare.attempt_accounting(prior)

    def test_wrong_work_or_clock_cannot_be_imported(self):
        with tempfile.TemporaryDirectory() as folder:
            for kwargs in ({'steps':4},{'steps':6},{'category':'native'},
                           {'category':'startup'},{'seconds':float('nan')},
                           {'seconds':float('inf')},{'seconds':10800}):
                with self.subTest(kwargs=kwargs):
                    # Fixtures deliberately retain JSON nonfinite values so the
                    # receipt validator, rather than the writer, rejects them.
                    if 'seconds' in kwargs and kwargs['seconds'] != 10800:
                        prior = prior_fixture(Path(folder)/'execution_001')
                        for stem in ('status','work_costs'):
                            path=Path(prior[stem+'_path']); value=prepare.read(path)
                            key='budget' if stem=='status' else 'native_budget'
                            value[key]['gpu_seconds']=kwargs['seconds']
                            path.write_text(json.dumps(value))
                            prior[stem+'_identity']=prepare.identity(path)
                    else:
                        prior = prior_fixture(Path(folder)/'execution_001',**kwargs)
                    with self.assertRaises(ValueError):
                        prepare.attempt_accounting(prior)

    def test_cross_receipt_disagreement_wrong_source_and_unknown_work_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            for mutation in ('cost_counts','status_time','source','unknown','completed','cap'):
                with self.subTest(mutation=mutation):
                    prior=prior_fixture(Path(folder)/'execution_001')
                    if mutation=='cap':
                        prior['approved_cumulative_adam_limit']=188
                    elif mutation=='unknown':
                        path=Path(prior['training_work_path'])
                        lines=path.read_text().splitlines(); del lines[-2]
                        path.write_text('\n'.join(lines)+'\n')
                        prior['training_work_identity']=prepare.identity(path)
                    else:
                        stem='work_costs' if mutation=='cost_counts' else 'status'
                        path=Path(prior[stem+'_path']); value=prepare.read(path)
                        if mutation=='cost_counts':value['training']['completed']['optimizer_step']=4
                        elif mutation=='status_time':value['budget']['gpu_seconds']=0.
                        elif mutation=='source':value['execution_commit']='unrelated-source'
                        else:value['status']='completed'
                        path.write_text(json.dumps(value));prior[stem+'_identity']=prepare.identity(path)
                    with self.assertRaises(ValueError):prepare.attempt_accounting(prior)

    def test_retry_accounting_cannot_be_reset_or_relocated(self):
        with tempfile.TemporaryDirectory() as folder:
            plan=retry_plan(Path(folder)/'execution_002')
            for change in ('remove_prior','reset_clock','reset_steps','different_destination'):
                changed=deepcopy(plan)
                if change=='remove_prior':changed.pop('prior_attempt')
                elif change=='reset_clock':changed['budget_accounting']['current_attempt_gpu_seconds_limit']=10800.
                elif change=='reset_steps':changed['budget_accounting']['prior_adam_steps']=0
                else:changed['root']=str(Path(folder)/'execution_003')
                with self.subTest(change=change),self.assertRaises(ValueError):
                    campaign.checked_accounting(changed)

    def test_prepare_seals_prior_identities_at_the_new_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); old=root/'retained_pr46'; old.mkdir()
            prepare.write(old/'PROBE_KEYS.json',[]);(old/'PROBE_KEYS.csv').write_text('request_id\n')
            export=root/'export';export.mkdir();(export/'model.safetensors').write_bytes(b'weights')
            state=root/'state.pt';state.write_bytes(b'state')
            previous=dict(execution_commit='d51718a7fa993a1494e593d15a8539090227d8b8',
                input_identities={},source_manifest='old_source.json',
                immutable_files={name:prepare.identity(old/name) for name in ('PROBE_KEYS.json','PROBE_KEYS.csv')},
                feature_paths=list(range(128)),probe_requests=list(range(64)),
                origin_export=str(export),origin_state=str(state))
            prepare.write(old/'PLAN.json',previous)
            source=root/'SOURCE.json';source.write_text('{}')
            prior=prior_fixture(root/'execution_001')
            spec=dict(output=str(root/'execution_002/sealed'),previous_plan=str(old/'PLAN.json'),
                previous_plan_identity=prepare.identity(old/'PLAN.json'),
                source_manifest=str(source),source_manifest_identity=prepare.identity(source),
                execution_commit='new-source',prior_attempt=prior)
            with patch.object(prepare,'authenticate_source',return_value={'source_path':'fixture-checkout'}):
                sealed=prepare.prepare(spec)
            self.assertEqual(sealed['budget_accounting']['cumulative_adam_limit'],187)
            self.assertEqual(sealed['prior_attempt'],prior)
            for stem in ('status','work_costs','training_work'):
                self.assertEqual(sealed['input_identities'][prior[stem+'_path']],prior[stem+'_identity'])
            self.assertEqual(sealed['constants'],prepare.CONSTANTS)

    def test_campaign_alarm_and_child_deadline_use_remaining_clock(self):
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder)/'execution_002'; plan=retry_plan(root)
            service=SimpleNamespace(connection='fixture',close=Mock())
            stack.enter_context(patch.object(campaign,'BudgetService',return_value=service))
            stack.enter_context(patch.object(campaign,'preflight',return_value={}))
            stack.enter_context(patch.object(probe_client,'qualify',return_value={}))
            stack.enter_context(patch('experiments.native_prefix_loss.probe_client.run',return_value={}))
            training=stack.enter_context(patch.object(campaign,'launch_training',return_value={'probe_exports':[]}))
            stack.enter_context(patch.object(campaign,'report',return_value={'status':'completed'}))
            stack.enter_context(patch.object(campaign.time,'monotonic',return_value=100.))
            with patch.object(campaign.signal,'setitimer') as alarm:
                result=campaign.run(root/'sealed/PLAN.json')
            remaining=plan['budget_accounting']['current_attempt_gpu_seconds_limit']
            self.assertEqual(training.call_args.args[1],100.+remaining)
            alarm.assert_any_call(campaign.signal.ITIMER_REAL,remaining)
            self.assertEqual(result['budget']['gpu_seconds_limit'],remaining)
            self.assertGreaterEqual(result['cumulative_gpu_seconds'],203.0700637307018)
            service.close.assert_called_once()

    def test_report_keeps_prior_and_current_work_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'execution_002';plan=retry_plan(root)
            write_ledger(root/'training/PHYSICAL_WORK.jsonl',steps=4)
            clock=[0.]
            budget=campaign.CampaignBudget(root/'work.jsonl',clock=lambda:clock[0])
            budget.start_gpu(1);clock[0]=10.;budget.end_gpu()
            result=campaign.report(plan,dict(status='failed',error='fixture stop',wall_s=10.),budget)
            costs=prepare.read(root/'WORK_AND_COSTS.json')
            self.assertEqual(costs['cumulative']['completed_adam_steps'],9)
            self.assertEqual(costs['cumulative']['gpu_seconds'],203.0700637307018+10.)
            self.assertEqual(costs['cumulative']['adam_limit'],187)
            self.assertEqual(costs['budget_accounting']['current_attempt_adam_limit'],182)
            self.assertEqual(costs['training']['completed']['optimizer_step'],4)
            self.assertEqual(result['status'],'incomplete')


if __name__=='__main__':unittest.main()
