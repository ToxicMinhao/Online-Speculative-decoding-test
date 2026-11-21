"""Authenticate actual paths, native dependencies and full CODE input bounds on CPU."""
import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import time

from online_feedback.storage import write_json
from experiments.bounded_prefix_allocation.preflight import verify_source
from experiments.osd_components.prepare import verify_native_runtime
from experiments.ok_coding.storage import identity
from experiments.persistent_prefix_withholding.launch import check_compilers
from .contract import digest, validate_manifest, SERVING, RESOURCE_LIMITS, ARMS, OUTPUT_POLICY
from .runtime import read, storage


def verify_compact_codec(executable=None):
    """Completed-event retention needs the installed lossless codec, before CUDA."""
    executable=shutil.which(executable or 'zstd')
    if executable is None:
        raise RuntimeError('Lossless event retention requires the existing zstd executable')
    payload=b'CODE compact retention preflight\x00\xff\n'
    encoded=subprocess.run([executable,'-q','-3','-c'],input=payload,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
    restored=subprocess.run([executable,'-q','-d','-c'],input=encoded,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
    if restored!=payload:
        raise ValueError('Lossless compact codec round trip changed bytes')
    return dict(executable=executable,status='passed',round_trip_bytes=len(payload),
        version=subprocess.check_output([executable,'--version'],text=True).strip())


def verify(plan):
    import torch
    import transformers
    from .configuration import validate_config_text
    from .budget_resume import effective_limits
    from experiments.full_k_gsm8k.training import validate_snapshot
    from experiments.ok_coding.training import _export_equal
    began=time.perf_counter()
    result=dict(status='running',execution_commit=plan['execution_commit'],native_model_work=0)
    try:
        if plan['binding']!=digest({k:v for k,v in plan.items() if k!='binding'}):
            raise ValueError('The execution plan changed after sealing')
        if plan.get('output_policy') != OUTPUT_POLICY:
            raise ValueError('Explicitly seal the authorized AR/speculative output policy')
        if plan['serving']!=SERVING or plan['limits']!=effective_limits(plan) or plan['arms']!=list(ARMS):
            raise ValueError('The scientific/serving/resource contract changed')
        result['source']=verify_source(plan)
        for path,pin in plan['input_identities'].items():
            if identity(path)!=pin:
                raise ValueError('Sealed execution input changed: '+path)
        for entry in plan['authenticated_files']:
            if identity(entry['path'])!=entry['expected_identity']:
                raise ValueError('Required model/runtime/witness changed: '+entry['path'])
        manifest=read(plan['manifest'])
        seal=read(plan['data_seal'])
        for name,pin in seal['files'].items():
            if Path(name).name!=name or identity(Path(plan['data_seal']).parent/name)!=pin:
                raise ValueError('Input seal member changed: '+name)
        if seal['manifest_content_sha256']!=digest(manifest) or not seal['all_outputs_unobserved']:
            raise ValueError('Sealed panel identity or pre-output selection changed')
        result['data']=validate_manifest(manifest)
        original=Path(plan['original_config']).read_text()
        result['configs']={arm:validate_config_text(Path(plan['configs'][arm]).read_text(),original) for arm in ARMS}
        witness=torch.load(plan['initial_witness'],map_location='cpu',weights_only=False,mmap=True)
        old=witness['algorithm_contract']['full_k_gsm8k_v1']
        result['historical_initial']=validate_snapshot(witness,arm='K',binding=old['binding'],
            config_text=witness['config_text'],original_config_text=original,completed_event=0,
            expected_contract=witness['algorithm_contract'],expected_recipe=witness['backend']['optimizer']['recipe'])
        _export_equal(witness,plan['draft'])
        del witness
        result['native_runtime']=verify_native_runtime(plan['worker_python'],read(plan['runtime_pin']))
        result['compilers']=check_compilers()
        result['compact_codec']=verify_compact_codec(plan['zstd'])
        current=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__)
        if current!=read(plan['training_pin'])['environment'] or torch.cuda.is_initialized():
            raise ValueError('Training environment changed or CPU preflight initialized CUDA')
        result['training_environment']=current
        result['storage']=storage(plan)
        result['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
        actual={line.split(',')[0].strip() for line in result['gpu_inventory'].splitlines() if line.strip()}
        if actual!=set(plan['devices'].values()):
            raise ValueError('Physical GPU identities changed')
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if active.strip():
            raise RuntimeError('Initial qualification needs both assigned cards idle')
        from experiments.kw_v1.preflight import hardware_admission
        result['hardware']=hardware_admission()
        memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        result['available_host_bytes']=int(memory['MemAvailable'].split()[0])*1024
        if result['available_host_bytes']<80*2**30:
            raise RuntimeError('Complete-state qualification needs80GiB available host RAM')
        from .protected import capture_authenticated
        result['protected_dependencies']=capture_authenticated(plan)
        result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        result['wall_s']=time.perf_counter()-began
        write_json(Path(plan['root'])/'CPU_PREFLIGHT.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    args=parser.parse_args()
    verify(read(args.plan))
