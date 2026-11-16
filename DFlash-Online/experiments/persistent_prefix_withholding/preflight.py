"""CPU source, complete-state, ancestry and peak-storage admission before CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from online_feedback.storage import write_json
from experiments.ok_coding.training import _export_equal
from experiments.ok_continuation.resume import differences
from .state import validate_legacy_origin, fork_state, validate_config_text, validate_snapshot
from .resources import storage_plan, FREE_RESERVE_BYTES


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def runtime_inventory(plan):
    """Read actual package versions, installed verifier sources and GPU IDs."""
    import os
    from .launch import check_compilers
    compilers = check_compilers()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    script = ("import importlib.metadata as m,json,platform,torch,transformers; "
              "print(json.dumps(dict(versions=dict(python=platform.python_version(),"
              "torch=torch.__version__,transformers=transformers.__version__),"
              "package_metadata=dict(torch=m.version('torch'),transformers=m.version('transformers')),"
              "cuda_version=torch.version.cuda,cuda_initialized=torch.cuda.is_initialized())))")
    versions = {}
    for name, key in (('training', 'training_python'), ('serving', 'worker_python')):
        observed = json.loads(subprocess.check_output(
            [plan[key], '-c', script], env=environment, text=True))
        check_environment(observed, plan['runtime_contract'][name], name)
        versions[name] = observed
    runtime = Path(plan['worker_python']).parent.parent/'lib/python3.12/site-packages/sglang/srt'
    expected_sources = json.loads(Path(plan['installed_runtime_identities']).read_text())
    observed_sources = {}
    for name in sorted(set(expected_sources) | set(plan['runtime_contract']['native_runtime_sha256'])):
        actual = identity(runtime/name)
        if name in expected_sources and actual != expected_sources[name]:
            raise ValueError('Installed first-token/graph source differs: '+name)
        expected_hash = plan['runtime_contract']['native_runtime_sha256'].get(name)
        if expected_hash and actual['sha256'] != expected_hash:
            raise ValueError('Installed qualified runtime differs: '+name)
        observed_sources[str(runtime/name)] = actual
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free',
                                  '--format=csv,noheader,nounits'], text=True)
    cards = []
    for line in raw.splitlines():
        index, uuid, used, free = [part.strip() for part in line.split(',')]
        cards.append(dict(index=int(index), uuid=uuid, used_mib=int(used), free_mib=int(free)))
    if set(plan['devices'].values()) != {row['uuid'] for row in cards}:
        raise ValueError('Physical GPU identities differ from the sealed two-card plan')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                        '--format=csv,noheader,nounits'], text=True)
    if processes.strip():
        raise RuntimeError('A GPU is occupied before qualification: '+processes)
    available = next(int(line.split()[1])*1024 for line in
                     Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    # Only one disposable trainer runs before its working set is measured.
    # Production pair admission uses measured peaks, not this historical bound.
    initial_required = (80+32)*2**30
    if available < initial_required:
        raise RuntimeError('Insufficient host RAM for first bounded qualification')
    return dict(versions=versions, installed_sources=observed_sources, physical_cards=cards,
                gpu_processes=processes, available_host_bytes=available,
                first_single_trainer_required_bytes=initial_required, qualified_compilers=compilers)


def check_environment(observed, expected, name):
    """Compare imported versions with the same imported-version historical pin.

    Distribution metadata may omit PyTorch's CUDA local-version suffix. Keep
    it in the receipt, but do not compare it to a different version surface.
    """
    versions = {('python' if key == 'python_version' else key): value
                for key, value in expected.items() if key != 'cuda'}
    if observed['versions'] != versions:
        raise ValueError('Pinned environment differs: '+name)
    if 'cuda' in expected and observed['cuda_version'] != expected['cuda']:
        raise ValueError('Pinned CUDA runtime differs: '+name)
    if observed['cuda_initialized']:
        raise ValueError('CPU environment inspection initialized CUDA')


def inspect(root):
    import torch
    root = Path(root)
    plan = json.loads((root/'sealed/PLAN.json').read_text())
    began = time.perf_counter()
    result = dict(status='running', execution_commit=plan['execution_commit'],
                  cuda_initialized=False, model_work=0, states={}, inputs={})
    try:
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU preflight must precede all CUDA initialization')
        result['runtime'] = runtime_inventory(plan)
        inventory = []
        for path, expected in plan['input_identities'].items():
            actual = identity(path)
            if actual != expected:
                raise ValueError('Required immutable dependency changed: '+path)
            result['inputs'][path] = actual
            stat = Path(path).stat()
            inventory.append(dict(path=path, resolved=str(Path(path).resolve()),
                uid=stat.st_uid, gid=stat.st_gid, device=stat.st_dev, inode=stat.st_ino,
                links=stat.st_nlink, allocated_bytes=stat.st_blocks*512, **actual))
        write_json(root/'INPUT_OWNERSHIP.json', inventory)
        source_manifest = json.loads(Path(plan['source_manifest']).read_text())
        if source_manifest['commit'] != plan['execution_commit'] or source_manifest['dirty']:
            raise ValueError('Execution source must identify its clean published checkout')
        for name, expected in source_manifest['files'].items():
            path = Path(plan['source_root'])/name
            if expected.get('kind') == 'symlink':
                if (not path.is_symlink() or path.readlink().as_posix() != expected['target']
                        or not path.resolve().is_relative_to(Path(plan['source_root']).resolve())):
                    raise ValueError('Staged internal source link differs: '+name)
                continue
            if identity(path) != expected:
                raise ValueError('Staged code differs: '+name)
        original_config = Path(plan['origin_config']).read_text()
        expected_ancestry = json.loads(Path(plan['expected_ancestry']).read_text())['requests']
        history_six = None
        for name, number, key, export_key in (
            ('K6',6,'origin_state','origin_export'),
            ('N7',7,'native_n7_state','native_n7_export')):
            state = torch.load(plan[key],map_location='cpu',weights_only=False,mmap=True)
            receipt = validate_legacy_origin(state, original_config, number,
                                              expected_cuda_rng_states=1)
            actual_ids = [f['record_id'] for event in state['events'] for f in event['features']]
            expected_ids = [r['request_id'] for r in expected_ancestry[:number*128]]
            if actual_ids != expected_ids:
                raise ValueError('Actual recovery ancestry differs: '+name)
            _export_equal(state,plan[export_key])
            if name == 'K6':
                history_six = state['events']
            elif differences(history_six, state['events'][:6]):
                raise ValueError('Original N7 changed K6 historical receipts')
            receipt.update(export_exact=True, actual_ancestry_records=len(actual_ids))
            result['states'][name] = receipt
            if name == 'K6':
                for arm in ('E','S'):
                    cfg = Path(plan['configs'][arm]).read_text()
                    import yaml
                    validate_config_text(cfg,arm,original_config)
                    migrated, changes = fork_state(state,arm=arm,binding=plan['binding'],
                        config_text=cfg,run_id=yaml.safe_load(cfg)['run_id'])
                    validate_snapshot(migrated, arm=arm, binding=plan['binding'],
                        config_text=cfg, original_config_text=original_config, completed_event=6,
                        expected_contract=migrated['algorithm_contract'],
                        expected_recipe=migrated['backend']['optimizer']['recipe'],
                        expected_cuda_rng_states=1)
                    # Validate unchanged model/Adam values and RNG without creating
                    # another full model-sized checkpoint on disk.
                    for field in ('draft_state_dict','rng','events','global_step','presentations'):
                        if differences(state[field],migrated[field]):
                            raise ValueError('Fork modified protected state: '+field)
                    if differences(state['backend']['optimizer']['optimizer_state_dict']['state'],
                                   migrated['backend']['optimizer']['optimizer_state_dict']['state']):
                        raise ValueError('Fork modified Adam history')
                    write_json(root/('MIGRATION_'+arm+'.json'), changes)
                    del migrated
            del state
        parity = json.loads(Path(plan['n7_parity_receipt']).read_text())
        if parity.get('status') != 'passed' or parity.get('weights_optimizer_counters_rng_exact') is not True:
            raise ValueError('Original N7 exact parity evidence is not passed')
        inputs = json.loads(Path(plan['event8_inputs']).read_text())['requests']
        config = json.loads((Path(plan['origin_export'])/'config.json').read_text())
        measured = storage_plan([len(r['prompt_ids']) for r in inputs],
            weight_bytes=Path(plan['origin_export']+'/model.safetensors').stat().st_size,
            recovery_bytes=max(Path(plan['origin_state']).stat().st_size,
                               Path(plan['native_n7_state']).stat().st_size),
            hidden_size=config['hidden_size'], captured_layers=len(config['dflash_config']['target_layer_ids']),
            prior_allocation_bytes=plan['prior_allocation_bytes'])
        if measured != plan['storage']:
            raise ValueError('Storage calculation differs from sealed physical schema')
        free = shutil.disk_usage(root).free
        result['storage'] = dict(measured, actual_free_bytes=free)
        if measured['status'] != 'passed' or free < measured['required_initial_free_bytes']:
            raise ValueError('Complete peak storage does not fit shared caps')
        result['cuda_initialized'] = torch.cuda.is_initialized()
        if result['cuda_initialized']:
            raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        result['wall_s'] = time.perf_counter()-began
        write_json(root/'CPU_PREFLIGHT.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    args=parser.parse_args()
    print(json.dumps(inspect(args.root)))


if __name__ == '__main__':
    main()
