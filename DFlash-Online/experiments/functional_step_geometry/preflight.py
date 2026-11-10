"""Authenticate qualified sources and complete KPLUS4 states before GPU work."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import time

from experiments.kplus_full_gsm8k.preflight import timing_barrier
from experiments.kw_v1.preflight import hardware_admission
from experiments.ok_coding.storage import identity
from experiments.osd_components.prepare import verify_files, verify_native_runtime
from experiments.ok_coding.training import _export_equal
from online_feedback.storage import write_json
from .training import validate_origin


QUALIFICATION_ADDITIONAL_BYTES = 32 * 2**30
RESERVE_BYTES = 8 * 2**30


def verify(root, *, postflight=False):
    import torch
    import transformers

    root = Path(root)
    started = time.perf_counter()
    receipt = dict(status='running', postflight=postflight, model_work=0)
    try:
        plan = json.loads((root / 'sealed/PLAN.json').read_text())
        seal = json.loads((root / 'sealed/SEAL.json').read_text())
        checked = 0
        for name, expected in seal['files'].items():
            if identity(name) != expected:
                raise ValueError('Sealed source/input differs: ' + name)
            checked += 1
        receipt['sealed_files_checked'] = checked
        receipt['execution_commit'] = plan['execution_source']['commit']
        pins = json.loads((root / 'sealed/MODEL_PIN.json').read_text())
        for name in ('target', 'draft'):
            verify_files(plan[name], pins[name]['files'], exact_files=True)
        receipt['native_runtime'] = verify_native_runtime(plan['worker_python'],
            json.loads((root / 'sealed/RUNTIME_PIN.json').read_text()))
        environment = dict(python=platform.python_version(), torch=torch.__version__,
                           transformers=transformers.__version__)
        expected = json.loads((root / 'sealed/TRAINING_PIN.json').read_text())['environment']
        if environment != expected or torch.cuda.is_initialized():
            raise ValueError('Qualified training environment differs or initialized CUDA early')
        receipt['training_runtime'] = environment
        original = Path(plan['original_config']).read_text()
        receipt['origins'] = {}
        for dataset, item in plan['datasets'].items():
            for key in ('origin_state', 'origin_weights'):
                if identity(item[key]) != item[key + '_identity']:
                    raise ValueError('Retained KPLUS4 identity differs: ' + dataset + '/' + key)
            state = torch.load(item['origin_state'], map_location='cpu',
                               weights_only=False, mmap=True)
            validation = validate_origin(state, dataset=dataset, original_config_text=original)
            _export_equal(state, item['origin_export'])
            receipt['origins'][dataset] = dict(validation, export_exact=True)
            del state
        free = shutil.disk_usage(root).free
        receipt['storage'] = dict(free_bytes=free, reserve_bytes=RESERVE_BYTES,
            additional_writes_bytes=QUALIFICATION_ADDITIONAL_BYTES,
            required_bytes=QUALIFICATION_ADDITIONAL_BYTES + RESERVE_BYTES,
            scope='bounded_qualification_only_not_full_pilot')
        if not postflight:
            if free < QUALIFICATION_ADDITIONAL_BYTES + RESERVE_BYTES:
                raise ValueError('Insufficient qualification storage including reserve')
            receipt['hardware'] = hardware_admission()
            receipt['quiescence'] = timing_barrier()
            memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            available = int(memory['MemAvailable'].split()[0]) * 1024
            receipt['host_available_bytes'] = available
            if available < 80 * 2**30:
                raise ValueError('Decoded qualification comparisons require 80 GiB available host memory')
        receipt['status'] = 'passed'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        receipt['wall_s'] = time.perf_counter() - started
        write_json(root / ('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'), receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--postflight', action='store_true')
    args = parser.parse_args()
    result = verify(args.root, postflight=args.postflight)
    print(json.dumps({key: result[key] for key in ('status', 'wall_s')}))
