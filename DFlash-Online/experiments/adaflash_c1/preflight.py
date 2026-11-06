"""Check exact retained artifacts, runtime and full-capacity admission on CPU."""
import argparse
import json
from pathlib import Path
import platform
import time

from online_feedback.storage import write_json
from online_feedback.adaflash_opd import adaflash_opd_contract
from experiments.ok_coding.preflight import verify_seal
from experiments.ok_coding.storage import identity
from experiments.osd_components.prepare import verify_files, verify_native_runtime
from experiments.kw_v1.preflight import hardware_admission
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from .contract import validate_plan
from .storage import admit, capacity_plan, check_growth
from .training import validate_config_text


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
        exports = json.loads((root / 'sealed/EXPORT_AUDIT.json').read_text())
        if (audit['status'] != 'passed' or exports['status'] != 'passed'
                or audit['model_work'] != 0 or exports['model_work'] != 0):
            raise ValueError('Retained control evidence is incomplete')
        for name in ('ORIGINAL_MANIFEST.json', 'ORIGINAL_CONFIG.yaml', 'CANONICALS.json',
                     'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
            original = str(Path(audit['original_root']) / 'sealed' / name)
            if identity(root / 'sealed' / name) != audit['files'][original]:
                raise ValueError('Copied original input changed: ' + name)
        result['retained_exports'] = {}
        for arm, export_path in plan['references'].items():
            export = exports['exports'][arm]
            if export['path'] != export_path:
                raise ValueError('Retained export path changed: ' + arm)
            verify_files(export_path, export['files'], exact_files=True)
            result['retained_exports'][arm] = dict(path=export_path, unchanged=True)
        if (plan['loss'] != adaflash_opd_contract()
                or json.loads((root / 'sealed/LOSS_CONTRACT.json').read_text()) != plan['loss']):
            raise ValueError('Paper-defined loss/precision contract changed')
        original_config = Path(plan['original_config']).read_text()
        result['config'] = validate_config_text(Path(plan['configs']['A']).read_text(), original_config)
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
            result['quiescence'] = timing_barrier()
            memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            result['host_available_bytes'] = int(memory['MemAvailable'].split()[0]) * 1024
            if result['host_available_bytes'] < 80 * 2**30:
                raise ValueError('Need 80 GiB available host memory for complete-state comparisons')
            result['storage'] = admit(root, plan['storage'], receipt_path=root / 'STORAGE_ADMISSION.json')
            if identity(plan['original_initial_state']) != audit['original_initial_identity']:
                raise ValueError('Original seed-17 initialization artifact changed')
            old = json.loads(Path(plan['original_full_k_plan']).read_text())
            state = torch.load(plan['original_initial_state'], map_location='cpu', weights_only=False)
            result['original_initial_state'] = validate_snapshot(state, arm='K', binding=old['binding'],
                config_text=Path(old['configs']['K']).read_text(), original_config_text=original_config,
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
