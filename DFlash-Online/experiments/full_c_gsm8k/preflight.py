"""CPU authentication of the complete staged experiment before GPU allocation."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

from online_feedback.storage import write_json
from experiments.bounded_prefix_allocation.preflight import verify_source
from experiments.ok_coding.storage import identity
from experiments.osd_components.prepare import verify_native_runtime
from .contract import validate_plan
from .storage import admit


def verify(plan):
    import torch
    import transformers
    from experiments.kplus_full_gsm8k.training import validate_config_text
    from experiments.full_k_gsm8k.training import validate_snapshot
    from experiments.ok_coding.training import _export_equal
    from online_feedback.training import validate_feature, validate_target_binding
    began = time.perf_counter()
    result = dict(status='running', model_work=0, execution_commit=plan['execution_commit'])
    try:
        result['source'] = verify_source(plan)
        for path, expected in plan['input_identities'].items():
            if identity(path) != expected:
                raise ValueError('Sealed input changed: '+path)
        for item in plan['protected_dependencies']['rows']:
            path = Path(item['path'])
            if item['kind'] == 'file' and identity(path) != item['expected_identity']:
                raise ValueError('Required historical dependency changed: '+str(path))
            if item['kind'] != 'file' and not path.is_dir():
                raise FileNotFoundError(path)
        result['plan'] = validate_plan(plan)
        original = Path(plan['original_config']).read_text()
        validate_config_text(Path(plan['config_path']).read_text(), original)
        witness = torch.load(plan['initial_witness'], map_location='cpu', weights_only=False, mmap=True)
        old_contract = witness['algorithm_contract']['full_k_gsm8k_v1']
        result['initial_witness'] = validate_snapshot(witness, arm='K', binding=old_contract['binding'],
            config_text=witness['config_text'], original_config_text=original, completed_event=0,
            expected_contract=witness['algorithm_contract'], expected_recipe=witness['backend']['optimizer']['recipe'])
        _export_equal(witness, plan['draft'])
        del witness
        result['features'] = []
        for path in plan['qualification_feature_paths']:
            packet = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            info = validate_feature(packet,max_length=2272,hidden_width=20480,vocab_size=151936)
            validate_target_binding(packet)
            if packet['hidden_states'].dtype != torch.bfloat16 or packet['target_last_hidden_states'].dtype != torch.bfloat16:
                raise ValueError('Qualification feature precision changed')
            result['features'].append(info)
            del packet
        result['native_runtime'] = verify_native_runtime(plan['worker_python'],
            json.loads(Path(plan['runtime_pin']).read_text()))
        environment = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__)
        if environment != json.loads(Path(plan['training_pin']).read_text())['environment'] or torch.cuda.is_initialized():
            raise ValueError('Training runtime changed or CPU preflight initialized CUDA')
        result['training_environment'] = environment
        result['storage'] = admit(Path(plan['root']),plan['storage'])
        memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        result['available_ram_bytes'] = int(memory['MemAvailable'].split()[0])*1024
        if result['available_ram_bytes'] < 80*2**30:
            raise RuntimeError('Complete-state CPU checking requires 80 GiB available RAM')
        result['gpu_inventory'] = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
        actual_uuids={line.split(',')[0].strip() for line in result['gpu_inventory'].splitlines() if line.strip()}
        if actual_uuids!=set(plan['devices'].values()):
            raise RuntimeError('Physical GPU inventory differs from the two pinned A800s')
        from experiments.kw_v1.preflight import hardware_admission
        result['hardware_admission']=hardware_admission()
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if active.strip():
            raise RuntimeError('GPU admission requires both assigned cards to be quiet')
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        result['wall_s'] = time.perf_counter()-began
        write_json(Path(plan['root'])/'CPU_PREFLIGHT.json', result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    verify(json.loads(Path(parser.parse_args().plan).read_text()))
