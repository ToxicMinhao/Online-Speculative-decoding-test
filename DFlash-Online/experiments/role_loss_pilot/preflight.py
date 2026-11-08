"""CPU-only exact source/runtime/input/state and complete peak admission."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import time

from online_feedback.storage import write_json
from experiments.ok_coding.preflight import verify_seal
from experiments.osd_components.prepare import verify_files, verify_native_runtime
from experiments.kw_v1.preflight import hardware_admission
from experiments.kplus_full_gsm8k.preflight import timing_barrier
from .contract import ARMS, DATASETS, validate_plan
from .storage import admit,capacity_plan
from .training import validate_config_text


def verify(root,*,postflight=False,qualification_only=False):
    import torch
    import transformers
    from experiments.full_k_gsm8k.training import validate_snapshot,CONTRACT_KEY
    from experiments.ok_coding.training import _export_equal
    root=Path(root);started=time.perf_counter()
    result=dict(status='running',model_work=0,postflight=postflight,qualification_only=qualification_only)
    try:
        result['seal']=verify_seal(root)
        plan=json.loads((root/'sealed/PLAN.json').read_text());result['plan']=validate_plan(plan)
        original_config=Path(plan['original_config']).read_text()
        for dataset in DATASETS:
            for arm in ARMS:
                validate_config_text(Path(plan['datasets'][dataset]['configs'][arm]).read_text(),original_config)
        pins=json.loads((root/'sealed/MODEL_PIN.json').read_text())
        for model in ('target','draft'):
            verify_files(plan[model],pins[model]['files'],exact_files=True)
        result['native_runtime']=verify_native_runtime(plan['worker_python'],
            json.loads((root/'sealed/RUNTIME_PIN.json').read_text()))
        actual=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__)
        if actual!=json.loads((root/'sealed/TRAINING_PIN.json').read_text())['environment'] or torch.cuda.is_initialized():
            raise ValueError('Training runtime differs or CPU preflight initialized CUDA')
        result['training_runtime']=actual
        if plan['storage']!=capacity_plan((Path(plan['draft'])/'model.safetensors').stat().st_size):
            raise ValueError('Sealed phase peak storage calculation changed')
        if not postflight:
            result['hardware']=hardware_admission();result['quiescence']=timing_barrier()
            memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            result['host_available_bytes']=int(memory['MemAvailable'].split()[0])*1024
            if result['host_available_bytes']<80*2**30:
                raise ValueError('Need 80 GiB available host memory for decoded-state comparisons')
            if qualification_only:
                free=shutil.disk_usage(root).free
                result['storage']=dict(free_bytes=free,required_bytes=24*2**30,
                    additional_writes_bytes=16*2**30,reserve_bytes=8*2**30,scope='qualification_only')
                if free<24*2**30: raise ValueError('Insufficient qualification storage and reserve')
            else:
                result['storage']=admit(root,plan['storage'],receipt_path=root/'STORAGE_ADMISSION.json')
            state=torch.load(plan['original_initial_state'],map_location='cpu',weights_only=False)
            source_contract=state['algorithm_contract'][CONTRACT_KEY]
            result['original_initial_state']=validate_snapshot(state,arm='K',binding=source_contract['binding'],
                config_text=state['config_text'],original_config_text=original_config,completed_event=0,
                expected_contract=state['algorithm_contract'],expected_recipe=state['backend']['optimizer']['recipe'],
                expected_cuda_rng_states=1)
            _export_equal(state,Path(plan['draft']));result['released_weight_equality']=True
            del state
        else:
            result['storage']=admit(root,plan['storage'])
        result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        result['wall_s']=time.perf_counter()-started
        filename='POSTFLIGHT.json' if postflight else 'QUALIFICATION_PREFLIGHT.json' if qualification_only else 'PREFLIGHT.json'
        write_json(root/filename,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root')
    parser.add_argument('--postflight',action='store_true');parser.add_argument('--qualification-only',action='store_true')
    args=parser.parse_args();result=verify(args.root,postflight=args.postflight,qualification_only=args.qualification_only)
    print(json.dumps({k:result[k] for k in ('status','wall_s')}))


if __name__=='__main__':main()
