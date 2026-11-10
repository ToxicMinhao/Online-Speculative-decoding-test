"""The authorized retry reuses verified packets and retains failed work costs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.kplus_abt import contract, prepare_feedback, report, retry
from experiments.ok_coding.storage import identity


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def work(counts):
    return dict(attempts=counts, completed=counts, unknown={key:0 for key in counts})


def ledger(path, counts):
    rows = [dict(kind=kind, phase=phase, ordinal=i)
            for kind,count in counts.items() for i in range(1,count+1)
            for phase in ('started','completed')]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def fixture(base):
    old_root, output, evidence, code = [base/name for name in ('old','new','evidence','code')]
    evidence.mkdir(); (code/'online_feedback').mkdir(parents=True)
    (code/'online_feedback/sglang_backend.py').write_text('# unchanged backend\n')
    old = dict(schema=contract.SCHEMA, root=str(old_root), binding='a'*64,
        execution_source=dict(commit='b'*40,source_path=str(base/'old_source')),
        arms=list(contract.ARMS), events=list(contract.EVENTS), rho=contract.RHO,
        budgets=contract.budgets(), analysis=contract.ANALYSIS, target='/target',draft='/draft',
        worker_python='/worker',training_python='/trainer',training_gpu='1',datasets={})
    new = deepcopy(old)
    new.update(root=str(output),binding='c'*64,
        execution_source=dict(commit='d'*40,source_path=str(code)),
        shared_feedback_preparation=str(output/'qualification_feedback/PREPARATION.json'))
    pins = ('ORIGINAL_CONFIG.yaml','MODEL_PIN.json','RUNTIME_PIN.json','TRAINING_PIN.json')
    files = {}
    for name in pins:
        write(output/'sealed'/name, {'unchanged':name})
        files[str(old_root/'sealed'/name)] = identity(output/'sealed'/name)
    preparation = dict(status='completed',datasets={},shared_feedback_generations=4,
        teacher_preparations=4,extra_qualification_generations=0,main_updates=0,
        execution_commit=old['execution_source']['commit'],wall_s=7.)
    for dataset in contract.DATASETS:
        requests = [dict(request_id=f'{dataset}_{i}',event_id=f'event_{5+i//128:03d}',
            dataset=dataset,prompt_ids=[i],request_index=i,max_new_tokens=2048,collect_feedback=True)
            for i in range(256)]
        manifest = dict(dataset=dataset,start_ordinal=512,stop_ordinal=768,train_requests=requests)
        write(output/'sealed'/dataset/'TRAINING.json',manifest)
        write(output/'sealed'/dataset/'PANEL.json',{'unchanged':dataset})
        for name in ('TRAINING.json','PANEL.json'):
            files[str(old_root/'sealed'/dataset/name)] = identity(output/'sealed'/dataset/name)
        item = dict(origin_export='/original/'+dataset,
            origin_state='/state/'+dataset,origin_state_identity={'bytes':1,'sha256':'e'*64},
            training_manifest=str(output/'sealed'/dataset/'TRAINING.json'),evaluation_schedule=[])
        new['datasets'][dataset] = item
        old['datasets'][dataset] = dict(item,training_manifest=str(old_root/'sealed'/dataset/'TRAINING.json'))
        rows = []
        for request in requests[:2]:
            record = old_root/'qualification_feedback'/dataset/(request['request_id']+'.json')
            feature = record.with_suffix('.pt')
            write(record,{'record_id':request['request_id']})
            feature.write_bytes(b'valid retained fixture bytes')
            rows.append(dict(request_id=request['request_id'],status='completed',
                record_path=str(record),record_identity=identity(record),
                feature_path=str(feature),feature_identity=identity(feature)))
        preparation['datasets'][dataset] = rows
    write(evidence/'PLAN.json',old)
    files[str(old_root/'sealed/PLAN.json')] = identity(evidence/'PLAN.json')
    files[str(base/'old_source/online_feedback/sglang_backend.py')] = identity(code/'online_feedback/sglang_backend.py')
    write(evidence/'SEAL.json',dict(binding=old['binding'],execution_source=old['execution_source'],files=files))
    write(evidence/'PREPARATION.json',preparation)
    case = dict(response_forwards=2,response_backwards=2,optimizer_steps=1)
    startup = dict(response_forwards=2,response_backwards=2,optimizer_steps=0)
    write(evidence/'RESULT.json',dict(status='failed',dataset='gsm8k',case='KPLUS',
        binding=old['binding'],execution_commit=old['execution_source']['commit'],
        work=work(case),startup=dict(work=work(startup),wall_s=2.),wall_s=5.))
    write(evidence/'STATUS.json',dict(status='failed',completed=[],
        attempted=dict(dataset='gsm8k',case='KPLUS',kind='disposable_update'),
        binding=old['binding'],execution_commit=old['execution_source']['commit'],wall_s=15.))
    ledger(evidence/'WORK.jsonl',case);ledger(evidence/'STARTUP_WORK.jsonl',startup)
    (evidence/'EXIT_CODE').write_text('1\n')
    (evidence/'ADDITIONAL_STATUS.txt').write_text('DISPOSABLE_ENDPOINT_NOT_SAVED\nMAIN_PILOT_NOT_STARTED\n')
    write(evidence/'AUTHORIZATION.json',{'user_authorization':'Fix the checker and rerun bounded qualification.'})
    return evidence,output,new,code


def install(evidence, output, plan, code):
    plan['prior_attempt'], files = retry.seal_prior_attempt(evidence,output,plan,code)
    write(output/'sealed/PLAN.json',plan)
    return files


class RetryTests(unittest.TestCase):
    def test_reuse_does_not_initialize_backend_or_change_old_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence,output,plan,code = fixture(Path(temp))
            before = {str(path):identity(path) for path in (Path(temp)/'old').rglob('*') if path.is_file()}
            files = install(evidence,output,plan,code)
            with patch.object(prepare_feedback,'SGLangDFlashBackend') as backend, patch(
                    'experiments.full_k_gsm8k.campaign.online_record_check') as validate:
                result = prepare_feedback.prepare(output)
            backend.assert_not_called();self.assertEqual(validate.call_count,4)
            self.assertEqual(result['shared_feedback_generations'],0)
            self.assertEqual(result['teacher_preparations'],0)
            self.assertEqual(result['reused_feedback_packets'],4)
            self.assertEqual(len(files),8)
            self.assertEqual(before,{name:identity(name) for name in before})
            self.assertEqual(plan['budgets'],contract.budgets())
            self.assertEqual(plan['prior_attempt']['counts']['optimizer_steps'],1)
            self.assertEqual(plan['prior_attempt']['counts']['response_forwards'],4)
            self.assertEqual(plan['prior_attempt']['retry_qualification_budget']['optimizer_steps'],12)
            with self.assertRaises(FileExistsError): prepare_feedback.prepare(output)

    def test_changed_packet_and_request_fail_without_new_model_work(self):
        for mutation in ('feature','request'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                evidence,output,plan,code = fixture(Path(temp));install(evidence,output,plan,code)
                if mutation == 'feature':
                    row = json.loads((evidence/'PREPARATION.json').read_text())['datasets']['gsm8k'][0]
                    Path(row['feature_path']).write_bytes(b'changed')
                else:
                    path = output/'sealed/gsm8k/TRAINING.json'
                    data=json.loads(path.read_text());data['train_requests'][0]['request_id']='other';write(path,data)
                with patch.object(prepare_feedback,'SGLangDFlashBackend') as backend:
                    with self.assertRaises(ValueError): prepare_feedback.prepare(output)
                backend.assert_not_called()
                self.assertEqual(json.loads((output/'qualification_feedback/PREPARATION.json').read_text())['status'],'failed')

    def test_changed_runtime_model_origin_source_or_ledger_is_rejected(self):
        for mutation in ('pin','origin','source','ledger','startup','authorization'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                evidence,output,plan,code = fixture(Path(temp))
                if mutation=='pin':write(output/'sealed/MODEL_PIN.json',{'changed':True})
                elif mutation=='origin':plan['datasets']['gsm8k']['origin_state']='/different'
                elif mutation=='source':(code/'online_feedback/sglang_backend.py').write_text('# changed')
                elif mutation=='ledger':(evidence/'WORK.jsonl').write_text('')
                elif mutation=='startup':(evidence/'STARTUP_WORK.jsonl').write_text('')
                else:write(evidence/'AUTHORIZATION.json',{'user_authorization':''})
                with self.assertRaises(ValueError):install(evidence,output,plan,code)

    def test_report_separates_prior_cost_and_detects_raw_receipt_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence,output,plan,code = fixture(Path(temp));install(evidence,output,plan,code)
            observed = report.prior_attempt(report.Evidence(output,plan))
            self.assertEqual(observed['counts']['response_forwards'],4)
            self.assertEqual(observed['counts']['main_optimizer_steps'],0)
            self.assertEqual(observed['costs']['qualification_wall_s'],15.)
            summary=report.build(output,Path(temp)/'report')
            self.assertEqual(summary['prior_attempt']['counts']['feedback_generations'],4)
            self.assertIn('4 retained plus 508 newly collected',(Path(temp)/'report/REPORT.md').read_text())
            (output/'sealed/prior_attempt/WORK.jsonl').write_text('')
            observed=report.prior_attempt(report.Evidence(output,plan))
            self.assertEqual(observed['status'],'invalid_prior_evidence')
            self.assertIsNone(observed['counts'])


if __name__ == '__main__':
    unittest.main()
