"""Read-only source/runtime validation and peak disk/host admission."""
import json
from pathlib import Path
import platform
import shutil
import sys
import time

from online_feedback.storage import write_json
from .prepare import identity


def storage_budget(root, draft_bytes, *, available_host_bytes):
    # One newly saved complete state (3W), one export (W), plus exporter
    # temporary W. Event2's regenerated tensors are retired only after exact
    # scientific equality, complete compact retention and original re-verification.
    # All candidate snapshots/gradients stay in RAM; no tensor retrieval staging.
    budget = dict(draft_bytes=draft_bytes, tensor_peak_bytes=int(5 * draft_bytes * 1.10),
        compiler_staging_logs_bytes=4 * 2**30, reserve_bytes=8 * 2**30,
        free_bytes=shutil.disk_usage(root).free, host_available_bytes=available_host_bytes,
        host_required_bytes=80 * 2**30,
        basis='one replay state3W + exportW + exporter temporaryW;qualified event2 duplicate retired before4;10% slack;4GiB compact/cache allowance;8GiB reserve')
    budget['required_bytes'] = sum(budget[key] for key in ('tensor_peak_bytes',
        'compiler_staging_logs_bytes', 'reserve_bytes'))
    budget['admitted'] = (budget['free_bytes'] >= budget['required_bytes']
                          and budget['host_available_bytes'] >= budget['host_required_bytes'])
    return budget


def verify(root, postflight=False):
    from online_feedback.qualify_sglang import check_recipe
    from experiments.osd_components.prepare import verify_files, verify_native_runtime
    from .protocol import validate_plan
    root = Path(root)
    started = time.perf_counter()
    seal = json.loads((root / 'sealed/SEAL.json').read_text())
    for path, expected in seal['files'].items():
        if identity(path) != expected:
            raise ValueError('Sealed input changed: ' + path)
    for folder, names in seal['exact_file_sets'].items():
        if sorted(path.name for path in Path(folder).iterdir() if path.is_file()) != names:
            raise ValueError('Retained file set changed: ' + folder)
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    validate_plan(plan)
    model_pin = json.loads((root / 'sealed/MODEL_PIN.json').read_text())
    for name in ('target', 'draft'):
        verify_files(plan[name], model_pin[name]['files'], exact_files=True)
    recipe = check_recipe(plan['config'])
    runtime = verify_native_runtime(plan['worker_python'], json.loads((root / 'sealed/RUNTIME_PIN.json').read_text()))
    import torch
    import transformers
    training = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__)
    if training != json.loads((root / 'sealed/TRAINING_PIN.json').read_text())['environment']:
        raise ValueError('Qualified training environment differs')
    result = dict(status='passed', files_checked=len(seal['files']), runtime=runtime,
        training_runtime=training, recipe=recipe, cuda_initialized=torch.cuda.is_initialized())
    if not postflight:
        if (root / 'campaign_001').exists():
            raise ValueError('Fresh campaign directory required; no automatic retry')
        weight_bytes = sum(path.stat().st_size for path in Path(plan['draft']).glob('*.safetensors'))
        meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(meminfo['MemAvailable'].split()[0]) * 1024
        budget = storage_budget(root, weight_bytes, available_host_bytes=available)
        write_json(root / 'STORAGE_ADMISSION.json', budget)
        if not budget['admitted']:
            raise RuntimeError('Peak disk/host admission failed; retain the reserve')
        result['storage'] = budget
    result['wall_s'] = time.perf_counter() - started
    write_json(root / ('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'), result)
    return result


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1], postflight='--postflight' in sys.argv[2:])))
