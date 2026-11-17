"""Native workers, shared qualified trainer, and capture-disabled endpoint replay."""
import argparse
from pathlib import Path
import os
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.ok_coding.storage import identity
from experiments.functional_step_geometry.native_pilot import discarded_startup
from experiments.functional_step_geometry.qualify import cleanup
from .contract import ARMS
from .data import read, verify_shared_packets
from .storage import require_capacity
from .campaign import validate_update, require_admission, shared_inputs


def event_root(root, arm=None):
    base = Path(root)/'campaign_001/gsm8k/event_005'
    return base/arm if arm else base


def train_native_event(root, arm):
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.full_k_gsm8k.costs import begin_event_profile, finish_event_profile
    from .training import SurvivalTrainer
    root = Path(root); plan = read(root/'sealed/PLAN.json'); require_admission(plan)
    if arm not in ARMS:
        raise ValueError('Unknown fixed arm')
    output = event_root(root, arm); output.mkdir(parents=True, exist_ok=False)
    require_capacity(plan, output/'STORAGE_ADMISSION.json')
    pool, baseline = shared_inputs(plan)
    features = [Path(row['feature']['path']) for row in pool['records']]
    limits = dict(response_forwards=256, response_backwards=256, optimizer_steps=128,
                  head_optimizer_steps=0 if arm == 'KPLUS' else 128)
    ledger = WorkLedger(output/'WORK.jsonl', limits)
    progress = dict(status='running', arm=arm, dataset='gsm8k', event=5, automatic_retry=False)
    trainer = None; failure = None; started = time.perf_counter()
    try:
        pin_retained_graph_runtime(output)
        loading = time.perf_counter()
        trainer = SurvivalTrainer(plan['configs'][arm], output/'trainer', arm=arm,
            binding=plan['binding'], original_config_text=Path(plan['original_config']).read_text(),
            origin_state=plan['origin_state'], origin_export=plan['origin_export'])
        progress['loading_s'] = time.perf_counter()-loading
        progress['startup'] = discarded_startup(trainer, features, output/'startup')
        trainer.install_head()
        extension = getattr(trainer.bundle.model, 'survival_head_extension', None)
        if extension is not None:
            ledger.wrap(extension.optimizer.optimizer, 'step', 'head_optimizer_steps')
        ledger.wrap(trainer.core.strategy, 'forward_loss', 'response_forwards')
        ledger.wrap(trainer.backend, 'backward', 'response_backwards')
        ledger.wrap(trainer.backend.optimizer.optimizer, 'step', 'optimizer_steps')
        profile = begin_event_profile()
        try:
            result = trainer.train_event('event_005', features, passes=2, presentation_schedule=baseline)
        finally:
            write_json(output/'RESOURCE_COSTS.json', finish_event_profile(profile))
        work = read_work(ledger.path, limits); require_complete_work(work, limits)
        if any(result['survival_work'][k] != v for k,v in work['completed'].items()):
            raise ValueError('Native and external work accounting disagree')
        result['work_ledger'] = work
        result['head_optimizer_steps'] = 0 if extension is None else extension.steps
        validate_update(result, arm, baseline)
        boundary = trainer.verify_boundary(export_path=result['checkpoint_path'], receipt_path=output/'BOUNDARY_CHECK.json')
        result.update(state_path=str(trainer.latest), state_identity=identity(trainer.latest),
            weights_identity=identity(Path(result['checkpoint_path'])/'model.safetensors'), boundary=boundary)
        verify_shared_packets(pool)
        write_json(output/'RESULT.json', result)
        progress.update(status='completed', final_step=trainer.global_step,
            final_presentations=trainer.presentations, result=str(output/'RESULT.json'))
    except BaseException as error:
        failure = error
        if trainer is not None: trainer.failed = True
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            cleanup(trainer.close, failure, progress, 'trainer close')
        progress.update(wall_s=time.perf_counter()-started, work=read_work(ledger.path, limits))
        cleanup(lambda:write_json(output/'STATUS.json', progress), failure, progress, 'training receipt')
    return result


def launch_training(plan, arm):
    base = event_root(plan['root']); base.mkdir(parents=True, exist_ok=True)
    command = [plan['training_python'], '-u', '-m', 'experiments.survival_head.native_pilot',
               plan['root'], '--train', arm]
    receipt = dict(command=command, arm=arm, physical_gpu=plan['training_gpu_identity'])
    started = time.perf_counter()
    with (base/(arm+'.log')).open('x') as stream:
        process = subprocess.Popen(command, cwd=plan['execution_source']['source_path'],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['training_gpu'])), stdout=stream, stderr=subprocess.STDOUT)
        receipt['pid'] = process.pid; write_json(base/(arm+'_EXEC.json'), receipt)
        try:
            code = process.wait()
        finally:
            if process.poll() is None: process.terminate(); process.wait()
            receipt.update(returncode=process.returncode, wall_s=time.perf_counter()-started)
            write_json(base/(arm+'_EXEC.json'), receipt)
    if code:
        raise RuntimeError('Native training failed; no retry: '+arm)
    return read(base/arm/'RESULT.json')


def evaluate(plan, final):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.kplus_full_gsm8k.preflight import timing_barrier
    from .evaluation import evaluate_stage, audit_stage
    root = Path(plan['root']); output = root/'campaign_001/evaluation/gsm8k'
    checkpoints = {arm:final[arm]['checkpoint_path'] for arm in ARMS}
    requests = read(plan['panel_manifest'])['requests']; canonicals = read(plan['references'])
    backend = None; failure = None; started = time.perf_counter()
    receipt = dict(status='running', physical_gpu=plan['serving_gpu_identity'])
    try:
        require_admission(plan)
        require_capacity(plan, root/'EVALUATION_STORAGE_ADMISSION.json')
        for arm in ARMS:
            if identity(Path(checkpoints[arm])/'model.safetensors') != final[arm]['weights_identity']:
                raise ValueError('Checkpoint changed before native serving')
        timing_barrier()
        backend = SGLangDFlashBackend(plan['target'], checkpoints['KPLUS'],
            worker_python=plan['worker_python'], output_root=output.parent/'gsm8k_native',
            serving_gpu=str(plan['serving_gpu']), context_limit=4096, block_size=16)
        receipt['loading_s'] = backend.loading_s
        def boundary():
            return dict(storage=require_capacity(plan, root/'CURRENT_STORAGE.json'),
                quiescence=timing_barrier(backend.worker.pid), physical_gpu=plan['serving_gpu_identity'])
        evaluate_stage(backend, requests, checkpoints, output, canonicals=canonicals,
            dataset='gsm8k', event=5, schedule=plan['evaluation_schedule'], boundary_check=boundary)
        for arm in ARMS:
            if identity(Path(checkpoints[arm])/'model.safetensors') != final[arm]['weights_identity']:
                raise ValueError('Checkpoint changed during serving')
        audit = audit_stage(output, requests, checkpoints, canonicals=canonicals,
            dataset='gsm8k', event=5, schedule=plan['evaluation_schedule'], evidence_root=root)
        write_json(output/'AUDIT.json', audit)
        if audit['status'] != 'passed': raise ValueError('Native evidence audit failed')
        receipt['status'] = 'completed'
    except BaseException as error:
        failure = error; receipt.update(status='failed', error=repr(error)); raise
    finally:
        if backend is not None: cleanup(backend.close, failure, receipt, 'evaluation close')
        receipt['wall_s'] = time.perf_counter()-started
        cleanup(lambda:write_json(output.parent/'gsm8k_WORKER.json', receipt), failure, receipt, 'evaluation receipt')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root'); parser.add_argument('--train', choices=ARMS)
    args = parser.parse_args()
    if args.train: train_native_event(args.root, args.train)
    else:
        from .campaign import run
        run(args.root, train_arm=launch_training, evaluate=evaluate)
