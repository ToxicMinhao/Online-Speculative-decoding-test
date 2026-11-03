"""Authenticate retained controls, initialization and capacity without model work."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

from online_feedback.storage import write_json
from experiments.ok_coding.preflight import verify_seal
from experiments.ok_coding.storage import identity
from experiments.osd_components.prepare import verify_files, verify_native_runtime
from experiments.kw_v1.preflight import hardware_admission
from .contract import validate_plan
from .storage import admit, capacity_plan, check_growth
from .training import validate_config_text


def timing_barrier(owner_pid=None):
    """Reject concurrent training/transfers; this check is outside request time."""
    processes = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    transfers = []
    training = []
    for line in processes.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        command = fields[1]
        executable = Path(command.split()[0]).name
        if executable in {'scp', 'rsync', 'sftp', 'sftp-server'}:
            transfers.append(line.strip())
        if executable.startswith('python') and any(
                name in command.split() for name in ('--train', '--training', 'train.py')):
            training.append(line.strip())
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'], text=True)
    pids = [int(value.strip()) for value in active.splitlines() if value.strip().isdigit()]
    # The native worker can own a child engine process. Its own process tree is
    # allowed; any other GPU process is an execution blocker.
    allowed = set()
    if owner_pid is not None:
        allowed.add(int(owner_pid))
        tree = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
        pairs = [tuple(map(int, line.split())) for line in tree.splitlines() if line.strip()]
        changed = True
        while changed:
            before = len(allowed)
            allowed.update(pid for pid, parent in pairs if parent in allowed)
            changed = len(allowed) != before
    foreign = sorted(set(pids) - allowed)
    if transfers or training or foreign:
        raise ValueError(f'Timed serving is not quiescent: transfers={transfers}, training={training}, foreign_gpu={foreign}')
    return dict(status='passed', transfers=[], training=[], foreign_gpu_processes=[], allowed_worker_pids=sorted(allowed))


def verify(root, *, postflight=False):
    import torch
    import transformers
    from experiments.full_k_gsm8k.training import validate_snapshot
    from experiments.ok_coding.training import _export_equal
    root = Path(root)
    result, started = dict(status='running', model_work=0, postflight=postflight), time.perf_counter()
    try:
        result['seal'] = verify_seal(root)
        plan = json.loads((root / 'sealed/PLAN.json').read_text())
        result['plan'] = validate_plan(plan)
        audit = json.loads((root / 'sealed/CONTROL_AUDIT.json').read_text())
        if audit['status'] != 'passed' or audit['model_work'] != 0:
            raise ValueError('Retained-control audit is incomplete')
        for path, expected in audit['files'].items():
            if identity(path) != expected:
                raise ValueError('Retained control/input changed: ' + path)
        for name in ('ORIGINAL_MANIFEST.json', 'ORIGINAL_CONFIG.yaml', 'CANONICALS.json',
                     'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
            original = str(Path(audit['original_root']) / 'sealed' / name)
            if identity(root / 'sealed' / name) != audit['files'][original]:
                raise ValueError('Copied input differs from authenticated original: ' + name)
        if identity(plan['baseline_counts']) != audit['baseline_counts_identity']:
            raise ValueError('Retained per-request K58 baseline changed')
        if identity(Path(plan['references']['K']) / 'model.safetensors') != audit['control_identity']:
            raise ValueError('Retained K58 export changed')
        original_config = Path(plan['original_config']).read_text()
        result['config'] = validate_config_text(Path(plan['configs']['KPLUS']).read_text(), original_config)
        pins = json.loads((root / 'sealed/MODEL_PIN.json').read_text())
        for model in ('target', 'draft'):
            verify_files(plan[model], pins[model]['files'], exact_files=True)
        result['native_runtime'] = verify_native_runtime(plan['worker_python'],
            json.loads((root / 'sealed/RUNTIME_PIN.json').read_text()))
        actual = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__)
        if actual != json.loads((root / 'sealed/TRAINING_PIN.json').read_text())['environment'] or torch.cuda.is_initialized():
            raise ValueError('Training runtime differs or CPU preflight initialized CUDA')
        result['training_runtime'] = actual
        if plan['storage'] != capacity_plan((Path(plan['draft']) / 'model.safetensors').stat().st_size):
            raise ValueError('Sealed capacity calculation changed')
        if not postflight:
            result['hardware'] = hardware_admission()
            memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            result['host_available_bytes'] = int(memory['MemAvailable'].split()[0]) * 1024
            if result['host_available_bytes'] < 80 * 2**30:
                raise ValueError('Need80GiB available host memory for complete-state comparisons')
            result['storage'] = admit(root, plan['storage'], receipt_path=root / 'STORAGE_ADMISSION.json')
            old = json.loads(Path(plan['original_full_k_plan']).read_text())
            state = torch.load(plan['original_initial_state'], map_location='cpu', weights_only=False)
            result['original_initial_state'] = validate_snapshot(state, arm='K', binding=old['binding'],
                config_text=Path(old['configs']['K']).read_text(),
                original_config_text=Path(old['original_config']).read_text(),
                completed_event=0, expected_contract=state['algorithm_contract'],
                expected_recipe=state['backend']['optimizer']['recipe'], expected_cuda_rng_states=1)
            _export_equal(state, Path(plan['draft']))
            result['released_weight_equality'] = True
            del state
        else:
            result['storage'] = check_growth(root, plan['storage'])
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['wall_s'] = time.perf_counter() - started
        write_json(root / ('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--postflight', action='store_true')
    args = parser.parse_args()
    result = verify(args.root, postflight=args.postflight)
    print(json.dumps({key: result[key] for key in ('status', 'wall_s')}))


if __name__ == '__main__':
    main()
