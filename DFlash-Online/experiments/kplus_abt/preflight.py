"""Verify actual ABT source, runtime, complete origins and phase storage."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import time

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .contract import budgets, ARMS, DATASETS, EVENTS, RHO
from .storage import check_current_capacity


def verify_seal(root):
    root=Path(root)
    plan=json.loads((root/'sealed/PLAN.json').read_text())
    seal=json.loads((root/'sealed/SEAL.json').read_text())
    if (plan['root'] != str(root) or seal['binding'] != plan['binding']
            or seal['execution_source'] != plan['execution_source']
            or plan['budgets'] != budgets() or plan['arms'] != list(ARMS)
            or set(plan['datasets']) != set(DATASETS) or plan['events'] != list(EVENTS)
            or plan['rho'] != RHO or plan.get('automatic_retry') is not False):
        raise ValueError('Sealed ABT source/design identity changed')
    if Path(__file__).resolve().parents[2] != Path(plan['execution_source']['source_path']).resolve():
        raise ValueError('The executing ABT module is not from the sealed source directory')
    for name,expected in seal['files'].items():
        if identity(name) != expected:
            raise ValueError('Sealed source/input differs: '+name)
    return plan,len(seal['files'])


def verify(root, *, scope='qualification', postflight=False):
    if scope not in ('qualification','pilot'):
        raise ValueError('Unknown ABT admission scope')
    root=Path(root);started=time.perf_counter()
    receipt=dict(status='running',scope=scope,postflight=postflight,model_work=0)
    try:
        plan,checked=verify_seal(root)
        receipt.update(sealed_files_checked=checked,execution_commit=plan['execution_source']['commit'])
        import torch
        import transformers
        from experiments.osd_components.prepare import verify_files,verify_native_runtime
        from experiments.ok_coding.training import _export_equal
        from .training import validate_origin
        pins=json.loads((root/'sealed/MODEL_PIN.json').read_text())
        for name in ('target','draft'):
            verify_files(plan[name],pins[name]['files'],exact_files=True)
        receipt['native_runtime']=verify_native_runtime(plan['worker_python'],
            json.loads((root/'sealed/RUNTIME_PIN.json').read_text()))
        environment=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__)
        expected=json.loads((root/'sealed/TRAINING_PIN.json').read_text())['environment']
        if environment != expected or torch.cuda.is_initialized():
            raise ValueError('Qualified training environment differs or CUDA initialized early')
        receipt['training_runtime']=environment
        original=Path(plan['original_config']).read_text()
        receipt['origins']={}
        for dataset,item in plan['datasets'].items():
            state=torch.load(item['origin_state'],map_location='cpu',weights_only=False,mmap=True)
            validation=validate_origin(state,dataset=dataset,original_config_text=original)
            _export_equal(state,item['origin_export'])
            receipt['origins'][dataset]=dict(validation,export_exact=True)
            del state
        if scope=='qualification':
            storage=plan['qualification_storage']
            free=shutil.disk_usage(root).free
            receipt['storage']=dict(storage,free_bytes=free,
                status='passed' if free>=storage['required_initial_free_bytes'] else 'failed')
        else:
            receipt['storage']=check_current_capacity(root,plan['main_storage'])
        if not postflight:
            if receipt['storage']['status'] != 'passed':
                raise ValueError('Insufficient actual storage for declared phase peak and reserve')
            from experiments.kplus_full_gsm8k.preflight import timing_barrier
            from experiments.kw_v1.preflight import hardware_admission
            receipt['hardware']=hardware_admission()
            receipt['quiescence']=timing_barrier()
            memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            available=int(memory['MemAvailable'].split()[0])*1024
            receipt['host_available_bytes']=available
            if available<80*2**30:
                raise ValueError('Complete-state qualification/comparisons require 80 GiB available host memory')
            if scope=='pilot':
                from .campaign import require_admission
                receipt['pilot_admission']=require_admission(plan)
        receipt['status']='passed'
    except BaseException as error:
        receipt.update(status='failed',error=repr(error))
        raise
    finally:
        receipt['wall_s']=time.perf_counter()-started
        name=('POSTFLIGHT_' if postflight else 'PREFLIGHT_')+scope.upper()+'.json'
        write_json(root/name,receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root');parser.add_argument('--scope',choices=('qualification','pilot'),default='qualification')
    parser.add_argument('--postflight',action='store_true')
    args=parser.parse_args()
    result=verify(args.root,scope=args.scope,postflight=args.postflight)
    print(json.dumps({key:result[key] for key in ('status','scope','wall_s')}))
