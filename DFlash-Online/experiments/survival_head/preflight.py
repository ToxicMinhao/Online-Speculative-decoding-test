"""Authenticate source/origin/runtime and admit work only after priority PR36."""
import argparse
from copy import deepcopy
from pathlib import Path
import platform
import subprocess
import time

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .contract import ARMS, ANALYSIS, SCHEMA, budgets
from .data import read, verify_panel, feedback_requests
from .storage import capacity_plan, require_capacity


def replay_runtime_pin(original):
    """Add the reviewed replay dependency without rewriting historical pins."""
    pinned = deepcopy(original)
    name = 'managers/schedule_batch.py'
    expected = 'e72161a7245e882eef40612a728fd6f72963498666a8caa4fa5b8b8d45413199'
    previous = pinned['native_runtime_sha256'].get(name)
    if previous is not None and previous != expected:
        raise ValueError('Historical and supplemental replay runtime pins disagree')
    pinned['native_runtime_sha256'][name] = expected
    return pinned


def verify_seal(root):
    """Authenticate the actual executing tree and every immutable plan input."""
    from .seal import authenticate_source

    root = Path(root).resolve()
    plan = read(root/'sealed/PLAN.json')
    seal = read(root/'sealed/SEAL.json')
    code = Path(__file__).resolve().parents[2]
    if (plan['root'] != str(root) or plan.get('schema') != SCHEMA
            or plan['budgets'] != budgets() or plan['arms'] != list(ARMS)
            or plan.get('analysis') != ANALYSIS or plan.get('automatic_retry') is not False
            or plan.get('dataset') != 'gsm8k' or plan.get('event') != 5
            or set(plan['configs']) != set(ARMS)
            or seal['binding'] != plan['binding']
            or seal.get('execution_source') != plan['execution_source']
            or Path(plan['execution_source']['source_path']).resolve() != code):
        raise ValueError('Sealed design/source mismatch')
    required = {str(root/'sealed'/name) for name in ('PLAN.json', 'ORIGINAL_CONFIG.yaml',
        'MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json', 'SOURCE_PANEL.json')}
    required.update(plan[key] for key in ('training_manifest', 'panel_manifest', 'references',
                                         'prior_pool', 'original_schedule'))
    required.update(plan['configs'].values())
    if not required.issubset(seal['files']):
        raise ValueError('Seal omits the plan or a required scientific input')
    source_files = {str(Path(path).relative_to(code)):expected for path, expected in seal['files'].items()
                    if Path(path).is_relative_to(code)}
    authenticate_source(dict(plan['execution_source'], status='verified', files=source_files),
                        code, verify_bytes=False)
    feature_sizes = [row['feature']['bytes'] for row in read(plan['prior_pool'])['records']]
    if (len(feature_sizes) != 128 or any(type(size) is not int or size <= 0 for size in feature_sizes)
            or plan['main_storage'] != capacity_plan(plan['origin_weights_identity']['bytes'],
                plan['origin_state_identity']['bytes'], feature_bytes=sum(feature_sizes),
                max_feature_bytes=max(feature_sizes))):
        raise ValueError('Sealed peak storage budget differs')
    for path, expected in seal['files'].items():
        if identity(path) != expected:
            raise ValueError('Sealed source or input changed: '+path)
    return plan, len(seal['files'])


def priority_finished(plan):
    root = Path(plan['priority_root'])
    if not (root/'EXIT_CODE').is_file():
        raise ValueError('PR36 still owns the GPU window')
    if ((root/'EXIT_CODE').read_text().strip() != '0'
            or read(root/'campaign_001/STATUS.json')['status'] != 'completed'):
        raise ValueError('PR36 did not finish successfully; do not conceal its blocker with another run')
    return dict(status='passed', exit_code=0, campaign_status='completed', priority_root=str(root))


def verify(root, *, scope='qualification', postflight=False):
    if scope not in ('qualification', 'pilot'):
        raise ValueError('Unknown survival admission scope')
    root = Path(root); started = time.perf_counter()
    receipt = dict(status='running', scope=scope, postflight=postflight, model_work=0)
    try:
        plan, count = verify_seal(root)
        receipt.update(binding=plan['binding'], execution_commit=plan['execution_source']['commit'])
        receipt['sealed_files_checked'] = count
        receipt['panel'] = verify_panel(read(plan['panel_manifest']), read(root/'sealed/SOURCE_PANEL.json'), feedback_requests(plan))
        receipt['priority'] = priority_finished(plan)
        import torch
        import transformers
        from experiments.osd_components.prepare import verify_files, verify_native_runtime
        from experiments.ok_coding.training import _export_equal
        from .training import validate_origin
        pin = read(root/'sealed/MODEL_PIN.json')
        for name in ('target','draft'): verify_files(plan[name], pin[name]['files'], exact_files=True)
        receipt['runtime'] = verify_native_runtime(plan['worker_python'],
            replay_runtime_pin(read(root/'sealed/RUNTIME_PIN.json')))
        actual = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__)
        if actual != read(root/'sealed/TRAINING_PIN.json')['environment'] or torch.cuda.is_initialized():
            raise ValueError('Training environment changed or CPU preflight initialized CUDA')
        receipt['training_runtime'] = actual
        for field, key in (('origin_state','origin_state_identity'),('origin_weights','origin_weights_identity')):
            if identity(plan[field]) != plan[key]: raise ValueError('Original KPLUS4 changed')
        state = torch.load(plan['origin_state'], map_location='cpu', weights_only=False, mmap=True)
        receipt['complete_origin'] = validate_origin(state, dataset='gsm8k', original_config_text=Path(plan['original_config']).read_text())
        _export_equal(state, plan['origin_export']); del state
        if not postflight:
            from experiments.kplus_full_gsm8k.preflight import timing_barrier
            receipt['quiet'] = timing_barrier()
            gpus = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'], text=True)
            mapping = {int(line.split(',')[0]):line.split(',')[1].strip() for line in gpus.splitlines()}
            if any(mapping[plan[k+'_gpu']] != plan[k+'_gpu_identity'] for k in ('training','serving')):
                raise ValueError('Physical GPU identity changed')
            receipt['physical_gpus'] = mapping
            receipt['storage'] = require_capacity(plan, root/'STORAGE_ADMISSION.json')
            mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            if int(mem['MemAvailable'].split()[0])*1024 < 80*2**30:
                raise ValueError('Need80GiB available host RAM for endpoint comparisons')
            if scope == 'pilot':
                from .campaign import require_admission
                receipt['qualification'] = require_admission(plan)
        receipt['status'] = 'passed'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error)); raise
    finally:
        receipt['wall_s'] = time.perf_counter()-started
        name = ('POSTFLIGHT_' if postflight else 'PREFLIGHT_')+scope.upper()+'.json'
        write_json(root/name, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root'); parser.add_argument('--scope', choices=('qualification','pilot'), default='qualification')
    parser.add_argument('--postflight', action='store_true')
    args = parser.parse_args(); verify(args.root, scope=args.scope, postflight=args.postflight)
