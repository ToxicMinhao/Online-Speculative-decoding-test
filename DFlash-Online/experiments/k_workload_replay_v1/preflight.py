"""Verify pins and actual capacity without initializing a model or CUDA."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from online_feedback.storage import write_json
from experiments.ok_coding.preflight import verify_seal
from experiments.osd_components.prepare import verify_files, verify_native_runtime
from .contract import ARMS, BUDGET, final_schedule
from .storage import admit, capacity_plan, check_growth
from .training import validate_config_text
from .seal import validate_plan


def read(path):
    return json.loads(Path(path).read_text())


def timing_barrier(backend_pid):
    """Do not benchmark alongside training or artifact transfers."""
    parents, transfers = {}, []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            status = (path / 'status').read_text()
            ppid = int(next(row.split()[1] for row in status.splitlines() if row.startswith('PPid:')))
            command = (path / 'comm').read_text().strip()
            parents[int(path.name)] = ppid
            if command in ('scp', 'rsync', 'sftp-server') and path.stat().st_uid == os.getuid():
                transfers.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    foreign = []
    gpu_pids = [int(line.strip()) for line in output.splitlines() if line.strip()]
    for pid in gpu_pids:
        parent = pid
        seen = set()
        while parent and parent not in seen and parent != backend_pid:
            seen.add(parent)
            parent = parents.get(parent, 0)
        if parent != backend_pid:
            foreign.append(pid)
    receipt = dict(status='passed' if not foreign and not transfers else 'failed',
        serving_owner_pid=backend_pid, gpu_pids=gpu_pids, other_gpu_pids=foreign,
        artifact_transfer_pids=transfers, checked_at_epoch=time.time())
    if foreign or transfers:
        raise RuntimeError('Timed serving requires training/transfer quiescence: ' + repr(receipt))
    return receipt


def verify(root, *, postflight=False):
    import torch
    import transformers
    from experiments.kw_v1.preflight import hardware_admission
    root = Path(root)
    result, started = dict(status='running', model_work=0, postflight=postflight), time.perf_counter()
    try:
        result['seal'] = verify_seal(root)
        plan = read(root / 'sealed/PLAN.json')
        result['plan'] = validate_plan(plan)
        inputs = read(plan['requests'])
        if plan['budget'] != BUDGET or set(plan['configs']) != set(ARMS):
            raise ValueError('Sealed scope is not the complete fixed K/U/R experiment')
        if plan['evaluation_schedule'] != final_schedule(inputs['evaluation']):
            raise ValueError('Final order changed after sealing')
        original = Path(plan['original_config']).read_text()
        for arm in ARMS:
            validate_config_text(Path(plan['configs'][arm]).read_text(), arm, original)
        pins = read(root / 'sealed/MODEL_PIN.json')
        for model in ('target', 'draft'):
            verify_files(plan[model], pins[model]['files'], exact_files=True)
        result['native_runtime'] = verify_native_runtime(plan['worker_python'], read(root / 'sealed/RUNTIME_PIN.json'))
        actual = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__)
        if actual != read(root / 'sealed/TRAINING_PIN.json')['environment'] or torch.cuda.is_initialized():
            raise ValueError('Runtime changed or CPU preflight initialized CUDA')
        result['training_runtime'] = actual
        rebuilt = capacity_plan(inputs['training'], plan['storage']['weight_bytes'],
                                compiler_baseline=plan['storage']['compiler_baseline'])
        if any(plan['storage'].get(key) != value for key, value in rebuilt.items()):
            raise ValueError('Storage geometry differs from the sealed protocol')
        if postflight:
            result['storage'] = check_growth(root, plan['storage'])
        else:
            result['hardware'] = hardware_admission()
            memory = {k: int(v.split()[0]) * 1024 for k, v in
                      (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
            result['host_available_bytes'] = memory['MemAvailable']
            if memory['MemAvailable'] < 80 * 2**30:
                raise ValueError('Require at least 80 GiB available host memory')
            result['storage'] = admit(root, plan['storage'], root / 'STORAGE_ADMISSION.json')
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['wall_s'] = time.perf_counter() - started
        write_json(root / ('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--postflight', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, postflight=args.postflight), indent=2))
