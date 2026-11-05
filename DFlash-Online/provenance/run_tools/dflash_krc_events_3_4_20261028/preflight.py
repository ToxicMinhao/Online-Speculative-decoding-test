"""Check source/artifact integrity, runtime and actual peak storage before the KRC events3/4 campaign."""
import json
from pathlib import Path
import shutil
import platform
import sys
import time

from online_feedback.storage import write_json
from online_feedback.qualify_sglang import check_recipe
from experiments.osd_components.prepare import verify_files, verify_native_runtime
from prepare_remote import identity


def verify(root, postflight=False):
    root = Path(root)
    started = time.perf_counter()
    seal = json.loads((root / 'sealed/SEAL.json').read_text())
    for path, expected in seal['files'].items():
        if identity(path) != expected:
            raise ValueError('Sealed input changed: ' + path)
    for folder, expected in seal['exact_file_sets'].items():
        if sorted(p.name for p in Path(folder).iterdir() if p.is_file()) != expected:
            raise ValueError('Retained file set changed: ' + folder)
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    model_pin = json.loads((root / 'sealed/MODEL_PIN.json').read_text())
    for name in ('target', 'draft'):
        verify_files(plan[name], model_pin[name]['files'], exact_files=True)
    recipe = check_recipe(plan['config'])
    runtime = verify_native_runtime(plan['worker_python'],
        json.loads((root / 'sealed/RUNTIME_PIN.json').read_text()))
    import torch
    import transformers
    training_pin = json.loads((root / 'sealed/TRAINING_PIN.json').read_text())['environment']
    training_actual = dict(python=platform.python_version(), torch=torch.__version__,
                           transformers=transformers.__version__)
    if training_actual != training_pin:
        raise ValueError('Qualified training environment changed')
    result = dict(status='passed', files_checked=len(seal['files']), runtime=runtime,
                  training_runtime=dict(torch=torch.__version__, transformers=transformers.__version__,
                                        python=sys.version), recipe=recipe)
    if not postflight:
        weight_bytes = sum(p.stat().st_size for p in Path(plan['draft']).glob('*.safetensors'))
        # Peak: KRC3 and KRC4 full states (6W total), both exports (2W),
        # and an exporter temporary copy (W). Atomic latest is hardlinked to
        # each retained state; replacement never creates a third full payload.
        # Add 10% serialization slack,4GiB compiler/compact retrieval allowance
        # and8GiB reserve. No full tensor retrieval staging occurs on A800.
        budget = dict(draft_bytes=weight_bytes, tensor_peak_bytes=int(9 * weight_bytes * 1.10),
                      compiler_staging_logs_bytes=4 * 2**30, reserve_bytes=8 * 2**30,
                      free_bytes=shutil.disk_usage(root).free)
        budget['required_bytes'] = sum(budget[k] for k in ('tensor_peak_bytes',
            'compiler_staging_logs_bytes', 'reserve_bytes'))
        budget['basis'] = 'two immutable states6W + two exports2W + exporter temporaryW;10% slack;compact-only retrieval'
        if (root / 'campaign_001').exists():
            raise ValueError('Fresh campaign directory required; replacement overlap not admitted')
        budget['admitted'] = budget['free_bytes'] >= budget['required_bytes']
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        budget['host_available_bytes'] = int(memory['MemAvailable'].split()[0]) * 1024
        budget['host_required_bytes'] = 72 * 2**30
        budget['admitted'] &= budget['host_available_bytes'] >= budget['host_required_bytes']
        write_json(root / 'STORAGE_ADMISSION.json', budget)
        if not budget['admitted']:
            raise RuntimeError('Peak disk/host storage admission failed')
        result['storage'] = budget
    result['wall_s'] = time.perf_counter() - started
    write_json(root / ('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'), result)
    return result


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1], postflight='--postflight' in sys.argv[2:])))
