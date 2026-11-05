"""Read-only O/K5--8 source, own-state, runtime and phase-capacity admission.

No model is constructed, no CUDA context is initialized and no forward,
backward, generation or optimizer step is performed by this preflight.
"""
import gc
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from online_feedback.storage import write_json
from .prepare import identity
from .storage import GIB, FEATURE_ROW_BYTES, required_bytes


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_seal(root):
    root = Path(root)
    seal = json.loads((root/'sealed/SEAL.json').read_text())
    required = {str(root/'sealed'/name) for name in
        ('PLAN.json','config.yaml','MODEL_PIN.json','RUNTIME_PIN.json','TRAINING_PIN.json','REMOTE_INPUT_IDENTITIES.json')}
    require(required <= set(seal['files']), 'Seal omits a required plan/config/runtime/source identity artifact')
    for path, expected in seal['files'].items():
        require(identity(path) == expected, 'Sealed input changed: ' + path)
    for folder, names in seal['exact_file_sets'].items():
        require(sorted(path.name for path in Path(folder).iterdir() if path.is_file()) == names,
                'Sealed file set changed: ' + folder)
    originals = json.loads((root/'sealed/REMOTE_INPUT_IDENTITIES.json').read_text())
    for path, expected in originals.items():
        require(identity(path) == expected, 'Original own-state/export/receipt identity changed: ' + path)
    return seal, originals


def history_draws(events):
    """Compare shared sample/anchor history without conflating arm objectives."""
    return [dict(event_id=event['event_id'],
        features=[feature['record_id'] for feature in event['features']],
        records=[{key:row[key] for key in ('record_id','pass','sampled_anchors')} for row in event['records']])
        for event in events]


def verify_initial_states(plan, originals, *, expected_cuda_rng_states=1):
    import torch
    from .resume import OBJECTIVES, differences, load_snapshot, validate_event_receipt
    require(not torch.cuda.is_initialized(), 'Preflight must start without a CUDA context')
    reference_rng = reference_history = None
    results = {}
    config = Path(plan['config']).read_text()
    new_ids = {request['request_id'] for window in plan['windows'] for request in window['requests']}
    for arm, objective in OBJECTIVES.items():
        source = plan['initial'][arm]
        require(originals.get(source['state']) == source['state_identity'], 'Own complete state identity is not sealed: ' + arm)
        needed = [source['receipt'], *[str(path) for path in Path(source['export']).iterdir() if path.is_file()]]
        require(all(path in originals for path in needed), 'Own original export/receipt identity is not sealed: ' + arm)
        original_receipt = json.loads(Path(source['receipt']).read_text())
        require(original_receipt.get('full_state_path') == source['state']
                and original_receipt.get('export_path') == source['export'], 'Original receipt is bound to a different own endpoint: ' + arm)
        validate_event_receipt(original_receipt, event_number=4, objective=objective)
        state, check = load_snapshot(source['state'], source['export'], objective=objective,
            config_text=config, completed_event=4, expected_cuda_rng_states=expected_cuda_rng_states)
        draws = history_draws(state['events'])
        used = {feature['record_id'] for event in state['events'] for feature in event['features']}
        require(len(used) == 512 and not used & new_ids, 'New feedback overlaps the complete previous training history')
        if arm == 'O':
            reference_rng, reference_history = state['rng'], draws
        else:
            require(not differences(reference_rng,state['rng']), 'Own O4/K4 RNG payloads differ')
            require(not differences(reference_history,draws), 'Own O4/K4 normal response/anchor histories differ')
        check.update(state_identity=originals[source['state']], original_receipt=source['receipt'],
                     prior_distinct_training_records=len(used))
        results[arm] = check
        del state
        gc.collect()
        require(not torch.cuda.is_initialized(), 'CPU state validation initialized CUDA')
    return dict(status='passed', arms=results, final_O_K_rng_exact=True,
        normal_history_exact=True, new_training_disjoint=True, model_passes=0, optimizer_steps=0)


def gpu_admission(gpu_csv, process_csv):
    processes = [line.strip() for line in process_csv.splitlines() if line.strip()]
    require(not processes, 'GPU compute processes are active')
    rows = []
    for line in gpu_csv.splitlines():
        if not line.strip():
            continue
        index, name, total, used = [item.strip() for item in line.split(',')]
        row = dict(index=int(index),name=name,total_mib=int(total),used_mib=int(used))
        require('A800' in name and row['total_mib'] >= 80*1024-2048
                and row['used_mib'] <= 1024 and row['total_mib']-row['used_mib'] >= 72*1024,
                'Require idle qualified A800 capacity: ' + line)
        rows.append(row)
    require([row['index'] for row in rows] == [0,1], 'Require the qualified two-A800 environment')
    return dict(status='passed',gpus=rows,active_compute_processes=processes,cuda_context_created=False)


def hardware_admission():
    gpu = subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=30)
    processes = subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=30)
    return gpu_admission(gpu.stdout,processes.stdout)


def storage_admission(plan, *, free_bytes, available_host_bytes):
    settings = plan['storage']
    require(settings['reserve_bytes'] == 8*GIB and settings['offload_timeout_s'] > 0
            and settings.get('offload_destination'), 'Require reserve and verified-offload settings')
    first = plan['windows'][0]
    rows = sum(len(row['prompt_ids'])+row['max_new_tokens'] for row in first['requests'])
    payload = rows*FEATURE_ROW_BYTES
    required = max(required_bytes(payload),settings.get('initial_required_bytes',0))
    result = dict(scope='First collection phase; later preparation and each training phase re-admit actual free space.',
        free_bytes=free_bytes,required_bytes=required,maximum_capture_rows=rows,capture_payload_bytes=payload,
        host_available_bytes=available_host_bytes,host_required_bytes=80*GIB,reserve_bytes=8*GIB,
        future_phase_guards=['complete prepared-copy overlap','3W pending own Adam state + W export + W workspace',
            'both closed trainers and required evaluation before offload','verified offload before local retirement'],
        no_assumption_all_future_phases_fit=True)
    result['admitted'] = free_bytes >= required and available_host_bytes >= result['host_required_bytes']
    return result


def verify(root, postflight=False):
    root = Path(root).resolve()
    started = time.perf_counter()
    result = dict(status='running',stage='source_seals',postflight=postflight,
                  model_passes=0,optimizer_steps=0,generations=0)
    try:
        if not postflight:
            require(not (root/'campaign_001').exists(), 'Fresh campaign directory required; no automatic retry')
        seal, originals = verify_seal(root)
        result['files_checked'] = len(seal['files'])
        result['original_files_checked'] = len(originals)
        plan = json.loads((root/'sealed/PLAN.json').read_text())
        from .campaign import validate_plan
        result['plan'] = validate_plan(plan)
        if not postflight:
            result['stage'] = 'hardware_and_phase_capacity'
            result['hardware'] = hardware_admission()
            meminfo = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            storage = storage_admission(plan,free_bytes=shutil.disk_usage(root).free,
                available_host_bytes=int(meminfo['MemAvailable'].split()[0])*1024)
            result['storage'] = storage
            write_json(root/'STORAGE_ADMISSION.json',storage)
            require(storage['admitted'], 'Initial phase disk/host admission failed')
        result['stage'] = 'model_runtime_configuration'
        from online_feedback.qualify_sglang import check_recipe
        from experiments.osd_components.prepare import verify_files, verify_native_runtime
        pins = json.loads((root/'sealed/MODEL_PIN.json').read_text())
        for name in ('target','draft'):
            verify_files(plan[name],pins[name]['files'],exact_files=True)
        result['recipe'] = check_recipe(plan['config'])
        result['native_runtime'] = verify_native_runtime(plan['worker_python'],
            json.loads((root/'sealed/RUNTIME_PIN.json').read_text()))
        import torch
        import transformers
        actual = dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__)
        require(actual == json.loads((root/'sealed/TRAINING_PIN.json').read_text())['environment'],
                'Qualified training environment changed')
        result['training_runtime'] = actual
        require(not torch.cuda.is_initialized(), 'Runtime metadata verification initialized CUDA')
        result['stage'] = 'complete_own_O4_K4_states'
        result['initial_states'] = verify_initial_states(plan,originals)
        result.update(status='passed',stage='finished',cuda_initialized=torch.cuda.is_initialized())
        return result
    except BaseException as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        result['wall_s'] = time.perf_counter()-started
        write_json(root/('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'),result)


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1],postflight='--postflight' in sys.argv[2:])))
