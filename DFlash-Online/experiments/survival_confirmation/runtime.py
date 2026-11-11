"""Frozen-checkpoint confirmation: authenticate, finish references, time, audit, stop.

No trainer, optimizer, feedback capture, or auxiliary head is constructed here.
"""
import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.survival_head.contract import ARMS, ANALYSIS, SERVING
from experiments.survival_head.data import read
from experiments.survival_head.evaluation import SESSION
from experiments.o_half_matched_lr.archive import validate_record


def finish_worker(backend, receipt, path, started, failure=None):
    """Persist counters before cleanup, retaining the original execution failure."""
    receipt['wall_s'] = time.perf_counter()-started
    write_json(path, receipt)
    cleanup_error = None
    if backend is not None:
        try:
            backend.close()
        except BaseException as error:
            cleanup_error = error
            receipt.update(status='failed', cleanup_error=repr(error))
    receipt['wall_s'] = time.perf_counter()-started
    write_json(path, receipt)
    if cleanup_error is not None and failure is None:
        raise cleanup_error


def physical_gpu(plan):
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping = {int(row.split(',')[0]):row.split(',')[1].strip() for row in rows.splitlines()}
    observed = mapping[plan['serving_gpu']]
    if observed != plan['serving_gpu_identity']:
        raise ValueError('Physical serving GPU changed')
    return observed


def capacity(plan):
    free = shutil.disk_usage(plan['root']).free
    required = plan['storage']['reserve_bytes'] + plan['storage']['maximum_additional_bytes']
    value = dict(status='passed' if free >= required else 'failed', free_bytes=free,
                 required_bytes=required)
    if value['status'] != 'passed':
        raise RuntimeError('Insufficient confirmation storage: '+repr(value))
    return value


def verify_exports(plan):
    checked = {}
    for arm in ARMS:
        item = plan['checkpoints'][arm]
        result = read(item['result_path'])
        if (identity(item['result_path']) != item['result_identity']
                or result['method'] != arm or result['event_id'] != 'event_005'
                or result['start_step'] != 512 or result['end_step'] != 640
                or result['status'] != 'completed'
                or result['checkpoint_path'] != item['checkpoint_path']
                or result['weights_identity'] != item['weights_identity']
                or result['boundary'].get('export_exact') is not True):
            raise ValueError('Retained event-five result changed: '+arm)
        path = Path(item['checkpoint_path'])
        observed = {p.name:identity(p) for p in path.iterdir() if p.is_file()}
        if observed != item['export_files'] or observed['model.safetensors'] != item['weights_identity']:
            raise ValueError('Frozen export changed: '+arm)
        config = read(path/'config.json')
        if config['num_hidden_layers'] != 5 or config['block_size'] != 16:
            raise ValueError('Require exact five-layer head-free draft export')
        checked[arm] = observed
    return checked


def preflight(root, *, postflight=False):
    root = Path(root); plan = read(root/'sealed/PLAN.json')
    receipt = dict(status='running', model_work=0)
    start = time.perf_counter()
    try:
        seal = read(root/'sealed/SEAL.json')
        code = Path(__file__).resolve().parents[2]
        if (plan['root'] != str(root.resolve()) or plan['execution_source']['source_path'] != str(code)
                or plan['analysis'] != ANALYSIS or plan['serving'] != SERVING
                or plan['panel_size'] < 136 or plan['panel_size'] > 768 or plan['panel_size'] % 8
                or plan['budget']['timed_completions'] != 6*plan['panel_size']
                or plan['budget']['optimizer_steps'] != 0 or plan['automatic_retry'] is not False):
            raise ValueError('Sealed confirmation contract changed')
        for path, expected in seal['files'].items():
            if identity(path) != expected: raise ValueError('Sealed input changed: '+path)
        source = read(plan['source_manifest'])
        if source['commit'] != plan['execution_source']['commit'] or source['dirty'] is not False:
            raise ValueError('Execution-source receipt differs')
        for name, expected in source['files'].items():
            if identity(code/name) != expected: raise ValueError('Execution code changed: '+name)
        receipt['sealed_files'] = len(seal['files'])
        receipt['source_files'] = len(source['files'])
        receipt['checkpoints'] = verify_exports(plan)
        from experiments.osd_components.prepare import verify_files, verify_native_runtime
        pin = read(root/'sealed/MODEL_PIN.json')
        verify_files(plan['target'], pin['target']['files'], exact_files=True)
        receipt['runtime'] = verify_native_runtime(plan['worker_python'], read(root/'sealed/RUNTIME_PIN.json'))
        receipt['execution_commit'] = plan['execution_source']['commit']
        if not postflight:
            from experiments.kplus_full_gsm8k.preflight import timing_barrier
            receipt['quiet'] = timing_barrier()
            receipt['capacity'] = capacity(plan)
            observed = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
            mapping = {int(line.split(',')[0]):line.split(',')[1].strip() for line in observed.splitlines()}
            if mapping[plan['serving_gpu']] != plan['serving_gpu_identity']:
                raise ValueError('Physical serving GPU changed')
            receipt['physical_gpu'] = mapping[plan['serving_gpu']]
        receipt['status'] = 'passed'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error)); raise
    finally:
        receipt['wall_s'] = time.perf_counter()-start
        write_json(root/('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'),receipt)
    return receipt


def prepare_references(plan):
    """Bind missing canonical tokens before timing; never replace selected inputs."""
    from online_feedback.sglang_backend import SGLangDFlashBackend
    root = Path(plan['root'])/'references'; root.mkdir(exist_ok=False)
    requests = read(plan['panel_manifest'])['requests']
    references = read(plan['reused_references'])
    missing = [r for r in requests if r['request_id'] not in references]
    if [r['request_id'] for r in missing] != plan['missing_reference_ids']:
        raise ValueError('Reference inventory changed after selection')
    status = dict(status='running', requested=len(requests), reused=len(references),
        planned_generations=len(missing), attempted_generations=0, returned_generations=0,
        valid_generations=0, invalid_generations=0, optimizer_steps=0, feedback_collections=0,
        automatic_retry=False, loading_s=0., generation_wall_s=0.)
    backend = None; failure = None; started = time.perf_counter()
    try:
        if missing:
            from experiments.kplus_full_gsm8k.preflight import timing_barrier
            capacity(plan); timing_barrier()
            backend = SGLangDFlashBackend(plan['target'],plan['checkpoints']['KPLUS']['checkpoint_path'],
                worker_python=plan['worker_python'],output_root=root/'native',
                serving_gpu=str(plan['serving_gpu']),context_limit=4096,block_size=16)
            status['loading_s'] = backend.loading_s
        for request in missing:
            capacity(plan); timing_barrier(backend.worker.pid)
            label = 'canonical_'+str(request['panel_ordinal']).zfill(4)
            path = root/(label+'.json')
            status['attempted_generations'] += 1; write_json(root/'STATUS.json',status)
            generation_start = time.perf_counter()
            try:
                raw = backend.generate(request['prompt_ids'],max_new_tokens=2048,request_id=label,
                    session_id=SESSION,request_index=request['panel_ordinal'],purpose='evaluation',save_to=path)
            finally:
                status['generation_wall_s'] += time.perf_counter()-generation_start
            status['returned_generations'] += 1
            reference = dict(prompt_ids=raw['prompt_ids'],response_ids=raw['response_ids'],
                finish_reason=raw['finish_reason'],verified=True,binding_digest=plan['canonical_binding'],
                source='fresh_pinned_native_verifier_before_timing',source_record_path=str(path),
                source_record_identity=identity(path))
            errors = validate_record(raw,request,raw['checkpoint_id'],reference,label=label,session=SESSION)
            native = read(raw['metadata']['native_trace_path'])
            if native != raw['metadata']['capture']: errors.append('native_sidecar_mismatch')
            write_json(root/(label+'_VALIDATION.json'),dict(status='invalid' if errors else 'valid',errors=errors))
            if errors:
                status['invalid_generations'] += 1
                raise ValueError('Invalid canonical reference: '+repr(errors))
            status['valid_generations'] += 1
            references[request['request_id']] = reference
            write_json(root/'PARTIAL_CANONICAL.json',references)
            write_json(root/'STATUS.json',status)
        if set(references) != {r['request_id'] for r in requests}:
            raise ValueError('Incomplete canonical binding')
        write_json(root/'CANONICAL.json',references)
        status.update(status='completed',canonical_identity=identity(root/'CANONICAL.json'))
    except BaseException as error:
        failure = error
        status.update(status='failed',error=repr(error)); raise
    finally:
        status.update(wall_s=time.perf_counter()-started,
            unattempted_generations=len(missing)-status['attempted_generations'],
            unknown_generations=status['attempted_generations']-status['returned_generations'],
            returned_without_validation=status['returned_generations']-status['valid_generations']-status['invalid_generations'])
        finish_worker(backend,status,root/'STATUS.json',started,failure)
    return status


def evaluate(plan):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from experiments.kplus_full_gsm8k.preflight import timing_barrier
    from .evaluation import evaluate_stage,audit_stage
    root = Path(plan['root']); output=root/'evaluation'
    checkpoints={arm:plan['checkpoints'][arm]['checkpoint_path'] for arm in ARMS}
    requests=read(plan['panel_manifest'])['requests']; canonicals=read(root/'references/CANONICAL.json')
    receipt=dict(status='running',physical_gpu=plan['serving_gpu_identity'])
    backend=None; failure=None; started=time.perf_counter()
    try:
        capacity(plan); timing_barrier(); verify_exports(plan)
        backend=SGLangDFlashBackend(plan['target'],checkpoints['KPLUS'],worker_python=plan['worker_python'],
            output_root=root/'evaluation_native',serving_gpu=str(plan['serving_gpu']),context_limit=4096,block_size=16)
        receipt['loading_s']=backend.loading_s
        def boundary():
            return dict(storage=capacity(plan),quiescence=timing_barrier(backend.worker.pid),
                        physical_gpu=physical_gpu(plan))
        args=dict(canonicals=canonicals,sealed_n=plan['panel_size'],schedule=plan['evaluation_schedule'],
                  physical_gpu=plan['serving_gpu_identity'])
        evaluate_stage(backend,requests,checkpoints,output,boundary_check=boundary,**args)
        receipt['checkpoint_postcheck']=verify_exports(plan)
        audit=audit_stage(output,requests,checkpoints,evidence_root=root,**args)
        write_json(output/'AUDIT.json',audit)
        if audit['status']!='passed': raise ValueError('Raw endpoint audit failed')
        receipt['status']='completed'
    except BaseException as error:
        failure=error
        receipt.update(status='failed',error=repr(error)); raise
    finally:
        finish_worker(backend,receipt,output/'WORKER.json',started,failure)
    return receipt


def run(root):
    root=Path(root).resolve(); plan=read(root/'sealed/PLAN.json')
    if (root/'STATUS.json').exists(): raise FileExistsError('Preserve prior execution; no automatic retry')
    started=time.perf_counter(); status=dict(status='running',phase='preflight',automatic_retry=False,
        execution_commit=plan['execution_source']['commit'],optimizer_steps=0)
    write_json(root/'STATUS.json',status); resources={}
    try:
        resources['preflight']=preflight(root)
        status['phase']='canonical_references'; write_json(root/'STATUS.json',status)
        resources['references']=prepare_references(plan)
        status['phase']='timed_evaluation'; write_json(root/'STATUS.json',status)
        resources['evaluation']=evaluate(plan)
        status['phase']='postflight'; write_json(root/'STATUS.json',status)
        resources['postflight']=preflight(root,postflight=True)
        status.update(status='completed',phase='completed')
    except BaseException as error:
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        status['wall_s']=time.perf_counter()-started
        write_json(root/'RESOURCES.json',resources); write_json(root/'STATUS.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root')
    args=parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt('Supervisor signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    run(args.root)
