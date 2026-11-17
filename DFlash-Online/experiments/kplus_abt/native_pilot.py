"""Native ABT workers: one immutable pool, one complete state per arm-event.

Adapted from the qualified FSG orchestration without changing its source.
Qualification and production are separate; no automatic retries or warmup answers.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.ok_coding.storage import identity
from experiments.kw_v1.evaluate import append_json
from experiments.functional_step_geometry.native_pilot import discarded_startup
from .contract import ARMS, DATASETS, EVENTS
from .data import seal_shared_packets, verify_shared_packets, feedback_requests, read
from .campaign import validate_update, require_admission


def training_work(arm):
    if arm not in ARMS:
        raise ValueError('Unknown ABT arm')
    views = 256 if arm == 'KPLUS' else 512
    return dict(response_forwards=views,response_backwards=views,optimizer_steps=128)


class ViewWorkCallback:
    """Record real view starts/completions through the existing durable ledger."""
    def __init__(self, ledger):
        self.ledger=ledger
        self.pending={}

    def __call__(self, kind, view, phase, **metadata):
        fields={'forward':'response_forwards','backward':'response_backwards'}
        if kind not in fields or view not in ('clean','second') or phase not in ('started','completed'):
            raise ValueError('Unknown native view work callback')
        field=fields[kind]; key=(field,view)
        ledger=self.ledger
        if phase=='started':
            if key in self.pending or ledger.attempts[field]>=ledger.limits[field]:
                raise RuntimeError('View work already pending or over budget')
            ordinal=ledger.attempts[field]+1
            append_json(ledger.path,dict(kind=field,phase='started',ordinal=ordinal,view=view))
            ledger.attempts[field]=ordinal
            self.pending[key]=ordinal
        else:
            if key not in self.pending:
                raise ValueError('View completion lacks its preceding start')
            ordinal=self.pending.pop(key)
            append_json(ledger.path,dict(kind=field,phase='completed',ordinal=ordinal,view=view))
            ledger.completed[field]+=1

def require_capacity(plan, receipt_path):
    from .storage import check_current_capacity
    receipt = check_current_capacity(plan['root'], plan['main_storage'])
    write_json(receipt_path, receipt)
    if receipt['status'] != 'passed':
        raise ValueError('ABT resource admission failed before additional work')
    return receipt


def event_root(root, dataset, event, arm=None):
    base = Path(root) / 'campaign_001' / dataset / f'event_{event:03d}'
    return base / arm if arm else base


def result_path(root, dataset, event, arm):
    return event_root(root, dataset, event, arm) / 'RESULT.json'


def retained_packets(plan, dataset, requests):
    """Reuse the two original qualification responses without recollecting them."""
    preparation = read(plan['shared_feedback_preparation'])
    rows = preparation['datasets'][dataset]
    if (preparation['status'] != 'completed' or len(rows) != 2
            or [row['request_id'] for row in rows] != [r['request_id'] for r in requests[:2]]):
        raise ValueError('The four qualifying packets must be the prescribed event-five prefix')
    for row in rows:
        if row['status'] != 'completed':
            raise ValueError('Retained qualification feedback is incomplete')
        for field in ('record', 'feature'):
            if identity(row[field + '_path']) != row[field + '_identity']:
                raise ValueError('Retained shared packet changed')
    return deepcopy(rows)


def collect_shared(plan, dataset, event, producer=None):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.full_k_gsm8k.campaign import online_record_check
    from experiments.ok_events_5_8.campaign import publish_checked
    from experiments.functional_step_geometry.qualify import cleanup

    root = Path(plan['root'])
    output = event_root(root, dataset, event) / 'shared'
    output.mkdir(parents=True, exist_ok=False)
    require_capacity(plan, output/'STORAGE_ADMISSION.json')
    requests = feedback_requests(plan, dataset, event)
    checkpoint = (plan['datasets'][dataset]['origin_export'] if event == 5
                  else producer['checkpoint_path'])
    rows = retained_packets(plan, dataset, requests) if event == 5 else []
    progress = dict(status='running', dataset=dataset, event=event, producer=f'KPLUS{event-1}',
        reused_feedback=len(rows), records=rows, automatic_retry=False)
    ledger = WorkLedger(output / 'WORK.jsonl',
        dict(feedback_generations=128-len(rows), teacher_preparations=128-len(rows)))
    backend = None
    failure = None
    started = time.perf_counter()
    try:
        for row, request in zip(rows, requests):
            online_record_check(read(row['record_path']), request, checkpoint, capture=True)
        backend = SGLangDFlashBackend(plan['target'], checkpoint,
            worker_python=plan['worker_python'], output_root=output / 'native',
            serving_gpu='0', context_limit=4096, block_size=16)
        progress['loading_s'] = backend.loading_s
        progress['publication'] = publish_checked(backend, checkpoint, output/'PUBLICATION.json')
        for request in requests[len(rows):]:
            record_path = output / 'records' / (request['request_id']+'.json')
            row = dict(request_id=request['request_id'], record_path=str(record_path),
                       status='generation_attempted')
            rows.append(row)
            write_json(output/'COLLECTION.json', progress)
            record = ledger.call('feedback_generations', backend.generate,
                request['prompt_ids'], max_new_tokens=2048,
                request_id=request['request_id'], session_id=dataset+'_abt_shared',
                request_index=request['request_index'], purpose='serving', save_to=record_path)
            online_record_check(record, request, checkpoint, capture=True)
            feature = output / 'loop/features' / (request['request_id']+'.pt')
            row.update(status='teacher_preparation_attempted', feature_path=str(feature))
            write_json(output/'COLLECTION.json', progress)
            prepared = ledger.call('teacher_preparations', backend.prepare_feedback, record, feature)
            if prepared.get('eligible') is False or prepared.get('status') != 'prepared':
                raise ValueError('Ineligible feedback is retained; no replacement is permitted')
            from experiments.kplus_full_gsm8k.storage import retire_redundant_features
            retirement = retire_redundant_features(record, prepared, root, output)
            row.update(status='completed', preparation=prepared, preparation_retirement=retirement, feature_identity=identity(feature),
                record_identity=identity(record_path), output_tokens=len(record['response_ids']),
                finish_reason=record['finish_reason'])
            write_json(output/'COLLECTION.json', progress)
        pool = seal_shared_packets([row['feature_path'] for row in rows],
            [row['record_path'] for row in rows], dataset=dataset, event=event,
            output=output/'POOL.json')
        require_complete_work(read_work(output/'WORK.jsonl', ledger.limits), ledger.limits)
        progress.update(status='completed', pool=str(output/'POOL.json'))
    except BaseException as error:
        failure = error
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            cleanup(backend.close, failure, progress, 'shared serving backend close')
        progress.update(wall_s=time.perf_counter()-started,
                        work=read_work(output/'WORK.jsonl', ledger.limits))
        cleanup(lambda: write_json(output/'COLLECTION.json', progress), failure, progress, 'collection receipt')
    return pool


def train_native_event(root, dataset, event, arm):
    """Complete native continuation; all additional views are physically charged."""
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.full_k_gsm8k.costs import begin_event_profile, finish_event_profile
    from experiments.functional_step_geometry.qualify import cleanup
    from .training import AdversarialBlockTrainer
    from . import diagnostics

    root=Path(root);plan=read(root/'sealed/PLAN.json')
    require_admission(plan)
    if arm not in ARMS or dataset not in DATASETS or event not in EVENTS:
        raise ValueError('Require the sealed ABT arm/dataset/event')
    output=event_root(root,dataset,event,arm)
    output.mkdir(parents=True,exist_ok=False)
    require_capacity(plan,output/'STORAGE_ADMISSION.json')
    pool=read(event_root(root,dataset,event)/'shared/POOL.json')
    verify_shared_packets(pool)
    features=[Path(row['feature']['path']) for row in pool['records']]
    item=plan['datasets'][dataset]
    previous=read(result_path(root,dataset,event-1,arm)) if event==6 else None
    baseline=read(result_path(root,dataset,event,'KPLUS')) if arm!='KPLUS' else None
    limits=training_work(arm); ledger=WorkLedger(output/'WORK.jsonl',limits)
    trainer=None;failure=None;started=time.perf_counter()
    progress=dict(status='running',dataset=dataset,event=event,arm=arm,automatic_retry=False)
    try:
        pin_retained_graph_runtime(output)
        loading=time.perf_counter()
        trainer=AdversarialBlockTrainer(item['configs'][arm],output/'trainer',arm=arm,
            dataset=dataset,binding=plan['binding'],original_config_text=Path(plan['original_config']).read_text(),
            origin_state=item['origin_state'],origin_export=item['origin_export'],
            resume_state=None if previous is None else previous['state_path'],completed_event=event-1)
        progress['loading_s']=time.perf_counter()-loading
        progress['startup']=discarded_startup(trainer,features,output/'startup')
        callback=None
        if arm!='KPLUS':
            callback=ViewWorkCallback(ledger)
            trainer.install_two_views(enabled=True,rho=plan['rho'],work_callback=callback,response_observer=None)
        else:
            ledger.wrap(trainer.core.strategy,'forward_loss','response_forwards')
            ledger.wrap(trainer.backend,'backward','response_backwards')
        ledger.wrap(trainer.backend.optimizer.optimizer,'step','optimizer_steps')
        diagnostic_results=diagnostics.install(trainer,features,output/'diagnostics',event=event)
        profile=begin_event_profile()
        try:
            result=trainer.train_event(f'event_{event:03d}',features,passes=2,presentation_schedule=baseline)
        finally:
            write_json(output/'RESOURCE_COSTS.json',finish_event_profile(profile))
        work=read_work(output/'WORK.jsonl',limits)
        require_complete_work(work,limits)
        if callback is not None and callback.pending:
            raise ValueError('Unfinished native view callback')
        result['abt_work']=dict(work['completed'],failed=0,unknown=sum(work['unknown'].values()))
        result['diagnostic_windows']=diagnostic_results
        diagnostic_s=sum(row['wall_s'] for row in diagnostic_results)
        result['costs'].update(diagnostic_observer_s=diagnostic_s,
            training_excluding_diagnostic_observer_s=result['costs']['training_s']-diagnostic_s,
            timing_scope='training_s includes both native views and fixed post-update diagnostics; nested timers are not additive')
        validate_update(result,dataset=dataset,arm=arm,event=event,baseline=baseline)
        boundary=trainer.verify_boundary(export_path=result['checkpoint_path'],receipt_path=output/'BOUNDARY_CHECK.json')
        result.update(state_path=str(trainer.latest),state_identity=identity(trainer.latest),
            weights_identity=identity(Path(result['checkpoint_path'])/'model.safetensors'),boundary=boundary)
        verify_shared_packets(pool)
        write_json(output/'RESULT.json',result)
        if previous is not None:
            from .storage import retire_previous_boundary
            progress['retirement']=retire_previous_boundary(root,previous,result,output/'PREVIOUS_RETIREMENT.json')
        progress.update(status='completed',result=str(output/'RESULT.json'),
                        final_step=trainer.global_step,final_presentations=trainer.presentations)
    except BaseException as error:
        failure=error
        if trainer is not None: trainer.failed=True
        progress.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            cleanup(trainer.close,failure,progress,'training close')
        progress.update(wall_s=time.perf_counter()-started,work=read_work(output/'WORK.jsonl',limits))
        cleanup(lambda:write_json(output/'STATUS.json',progress),failure,progress,'training receipt')
    return result


def launch_training(plan, dataset, event, arm, pool, previous=None):
    """Wait for a real native worker; failed jobs never receive automatic retries."""
    from experiments.role_loss_pilot.campaign import install_signal_handlers
    install_signal_handlers()
    root = Path(plan['root'])
    output = event_root(root, dataset, event)
    command = [plan['training_python'], '-u', '-m',
               'experiments.kplus_abt.native_pilot', str(root),
               '--train', dataset, str(event), arm]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['training_gpu']))
    receipt = dict(command=command, dataset=dataset, event=event, arm=arm, automatic_retry=False)
    started = time.perf_counter()
    require_capacity(plan, output/(arm+'_STORAGE_ADMISSION.json'))
    with (output/(arm+'.log')).open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env=environment, cwd=plan['execution_source']['source_path'])
        receipt['pid'] = process.pid
        write_json(output/(arm+'_EXEC.json'), receipt)
        try:
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            receipt.update(returncode=process.returncode, wall_s=time.perf_counter()-started)
            write_json(output/(arm+'_EXEC.json'), receipt)
    if code:
        raise RuntimeError('Native training failed; no retry: '+str(result_path(root,dataset,event,arm)))
    return read(result_path(root,dataset,event,arm))


def evaluate_dataset(plan, dataset, final):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.kplus_full_gsm8k.preflight import timing_barrier
    from .evaluation import evaluate_stage, audit_stage
    from experiments.functional_step_geometry.qualify import cleanup

    root = Path(plan['root'])
    item = plan['datasets'][dataset]
    output = root/'campaign_001/evaluation'/dataset
    checkpoints = {arm: final[arm]['checkpoint_path'] for arm in ARMS}
    requests = read(item['panel_manifest'])['requests']
    canonicals = read(item['references'])
    schedule = item['evaluation_schedule']
    backend = None
    failure = None
    receipt = dict(status='running')
    started = time.perf_counter()
    try:
        timing_barrier()
        backend = SGLangDFlashBackend(plan['target'], checkpoints['KPLUS'],
            worker_python=plan['worker_python'], output_root=output.parent/(dataset+'_native'),
            serving_gpu='0', context_limit=4096, block_size=16)
        receipt['loading_s'] = backend.loading_s
        def boundary():
            from .storage import check_current_capacity
            storage = check_current_capacity(root, plan['main_storage'])
            return dict(storage=storage, quiescence=timing_barrier(backend.worker.pid))
        evaluate_stage(backend, requests, checkpoints, output, canonicals=canonicals,
            dataset=dataset, event=6, schedule=schedule, boundary_check=boundary)
        for arm in ARMS:
            if identity(Path(checkpoints[arm])/'model.safetensors') != final[arm]['weights_identity']:
                raise ValueError('Final checkpoint changed during paired serving')
        audit = audit_stage(output, requests, checkpoints, canonicals=canonicals,
            dataset=dataset, event=6, schedule=schedule, evidence_root=root)
        write_json(output/'AUDIT.json', audit)
        if audit['status'] != 'passed':
            raise ValueError('Endpoint raw-evidence audit failed')
        from .statistics import summarize
        write_json(output/'SUMMARY.json', summarize(audit['rows'], dataset=dataset, event=6))
        receipt['status'] = 'completed'
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        if backend is not None:
            cleanup(backend.close, failure, receipt, 'evaluation close')
        receipt['wall_s'] = time.perf_counter()-started
        cleanup(lambda: write_json(output.parent/(dataset+'_WORKER.json'),receipt),failure,receipt,'evaluation receipt')
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--train',nargs=3,metavar=('DATASET','EVENT','ARM'))
    args=parser.parse_args()
    if args.train:
        dataset,event,arm=args.train
        train_native_event(args.root,dataset,int(event),arm)
    else:
        from .campaign import run
        from .storage import retire_shared_pool
        run(args.root,collect_shared=collect_shared,train_arm=launch_training,
            evaluate_dataset=evaluate_dataset,retire_shared=retire_shared_pool)


if __name__=='__main__':
    main()
