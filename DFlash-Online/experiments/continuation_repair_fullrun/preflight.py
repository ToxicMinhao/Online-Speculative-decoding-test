"""Authenticate complete inputs and the fixed native environment before model work."""
from pathlib import Path
import platform
import subprocess
import time

from experiments.frontier_preservation_pilot.qualify import verify_source
from experiments.opencode_native_comparison.preflight import verify_compact_codec
from experiments.osd_components.prepare import verify_native_runtime
from experiments.ok_coding.storage import identity
from experiments.persistent_prefix_withholding.launch import check_compilers
from .contract import ARMS, SERVING, INPUT_IDENTITY, digest, validate_inputs
from .configuration import validate_config_text
from .runtime import LIMITS,read,write_json,storage


def verify(plan):
    import torch
    import transformers
    root=Path(plan['root']);start=time.perf_counter()
    result=dict(status='running',execution_commit=plan['execution_commit'],model_forwards=0)
    try:
        if plan['binding']!=digest({k:v for k,v in plan.items() if k!='binding'}):
            raise ValueError('Execution plan changed after sealing')
        if plan['arms']!=list(ARMS) or plan['serving']!=SERVING:
            raise ValueError('Full-run scientific contract changed')
        if any(plan['limits'][key]!=value for key,value in LIMITS.items()):
            raise ValueError('Resource ceilings changed')
        verify_source(plan)
        result['source']='all staged source files authenticated'
        if identity(plan['manifest'])!=INPUT_IDENTITY:
            raise ValueError('Original PR56 training-input bytes changed')
        files={}
        for path,pin in plan['input_identities'].items():
            if identity(path)!=pin:raise ValueError('Sealed input changed: '+path)
            files[path]=pin
        for row in plan['authenticated_files']:
            if identity(row['path'])!=row['expected_identity']:
                raise ValueError('Required model/runtime/state changed: '+row['path'])
            files[row['path']]=row['expected_identity']
        for method,endpoint in plan['retained_endpoints'].items():
            weight=str(Path(endpoint['checkpoint'])/'model.safetensors')
            if identity(weight)!=endpoint['checkpoint_identity']:
                raise ValueError('Retained endpoint changed: '+method)
            files[weight]=endpoint['checkpoint_identity']
            for path,pin in endpoint['companion_files'].items():
                if identity(path)!=pin:raise ValueError('Endpoint configuration changed')
                files[path]=pin
        result['inputs']=validate_inputs(read(plan['manifest']),read(plan['evaluation_inputs']))
        result['configs']={arm:validate_config_text(Path(plan['configs'][arm]).read_text(),
            Path(plan['original_config']).read_text()) for arm in ARMS}
        result['native_runtime']=verify_native_runtime(plan['worker_python'],read(plan['runtime_pin']))
        result['compilers']=check_compilers()
        result['codec']=verify_compact_codec(plan['zstd'])
        result['training_environment']=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__)
        if result['training_environment']!=read(plan['training_pin'])['environment'] or torch.cuda.is_initialized():
            raise ValueError('CPU preflight environment changed or CUDA was initialized')
        result['storage']=storage(plan)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if active.strip():raise RuntimeError('Both GPUs must be idle for initial admission')
        memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        result['available_host_bytes']=int(memory['MemAvailable'].split()[0])*1024
        if result['available_host_bytes']<80*2**30:
            raise RuntimeError('Complete native state checks require80GiB available host memory')
        protection=dict(status='sealed',files=[dict(path=path,**pin,mtime_ns=Path(path).stat().st_mtime_ns)
            for path,pin in sorted(files.items())])
        destination=root/'PROTECTED_INPUTS.json'
        if destination.exists() and read(destination)!=protection:
            raise ValueError('Existing protected-dependency seal changed')
        write_json(destination,protection)
        result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        result['wall_s']=time.perf_counter()-start
        write_json(root/'CPU_PREFLIGHT.json',result)
    return result
