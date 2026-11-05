"""Authenticate the serving-only experiment and reserve its full disk budget."""
import argparse
import json
from pathlib import Path
import shutil
import time

from online_feedback.storage import write_json
from experiments.ok_coding.preflight import verify_seal
from experiments.ok_coding.storage import identity
from experiments.osd_components.prepare import verify_files, verify_native_runtime
from experiments.kw_v1.preflight import hardware_admission
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from .contract import BUDGET, METHODS, validate_plan

GIB = 2 ** 30


def capacity_plan():
    # Bound the original raw/native schemas at the full cap and one token of
    # progress per verification. Keep all three trace representations intact.
    return dict(raw_native_bytes=BUDGET['total_generations'] * 5327030,
        plan_reports_logs_bytes=2 * GIB, initialization_cache_bytes=4 * GIB,
        temporary_staging_bytes=GIB, reserve_bytes=8 * GIB,
        peak_required_bytes=BUDGET['total_generations'] * 5327030 + 15 * GIB,
        offload_assumed=False, new_checkpoint_bytes=0,
        source='sealed/STORAGE_SCHEMA_ESTIMATE.json')


def check_space(root, *, initial=False):
    free = shutil.disk_usage(root).free
    required = capacity_plan()['peak_required_bytes'] if initial else 8 * GIB
    receipt = dict(status='passed' if free >= required else 'blocked',
                   free_bytes=free, required_bytes=required, initial_admission=initial)
    if free < required:
        raise RuntimeError('Storage budget/reserve unavailable: ' + repr(receipt))
    return receipt


def verify_exports(plan, audit):
    if audit['status'] != 'passed' or audit['model_work'] != 0 or audit['checkpoints'] != plan['checkpoints']:
        raise ValueError('Retained export authentication is incomplete or changed')
    result = {}
    for arm in METHODS:
        export = audit['exports'][arm]
        if export['path'] != plan['checkpoints'][arm]:
            raise ValueError('Export receipt/path mismatch: ' + arm)
        verify_files(export['path'], export['files'], exact_files=True)
        result[arm] = dict(path=export['path'], files=export['files'], unchanged=True)
    for path, expected in audit['verified_files'].items():
        if identity(path) != expected:
            raise ValueError('Original identity/receipt changed: ' + path)
    return result


def verify(root, *, postflight=False):
    root = Path(root)
    started = time.perf_counter()
    result = dict(status='running', postflight=postflight, model_work=0)
    try:
        result['seal'] = verify_seal(root)
        plan = json.loads((root / 'sealed/PLAN.json').read_text())
        result['plan'] = validate_plan(plan)
        if plan['storage'] != capacity_plan():
            raise ValueError('Sealed peak storage calculation changed')
        audit = json.loads((root / 'sealed/EXPORT_AUDIT.json').read_text())
        result['exports'] = verify_exports(plan, audit)
        pins = json.loads((root / 'sealed/MODEL_PIN.json').read_text())
        for model in ('target', 'draft'):
            verify_files(plan[model], pins[model]['files'], exact_files=True)
        result['runtime'] = verify_native_runtime(plan['worker_python'],
            json.loads((root / 'sealed/RUNTIME_PIN.json').read_text()))
        if not postflight:
            result['hardware'] = hardware_admission()
            result['quiescence'] = timing_barrier()
            mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            result['available_host_bytes'] = int(mem['MemAvailable'].split()[0]) * 1024
            if result['available_host_bytes'] < 32 * GIB:
                raise RuntimeError('Insufficient available host memory for unchanged native loading')
        result['storage'] = check_space(root, initial=not postflight)
        result['status'] = 'passed'
        return result
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['wall_s'] = time.perf_counter() - started
        write_json(root / ('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--postflight', action='store_true')
    args = parser.parse_args()
    result = verify(args.root, postflight=args.postflight)
    print(json.dumps(dict(status=result['status'], wall_s=result['wall_s'])))
