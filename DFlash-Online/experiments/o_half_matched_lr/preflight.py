"""CPU-only reuse audit, including decoded O/K/K_HALF endpoints and raw traces."""
import argparse
import gc
import json
from pathlib import Path
import time

from online_feedback.storage import write_json
from experiments.ok_events_5_8.storage import identity
from experiments.ok_coding.preflight import verify_seal
from experiments.k_prefix_protection_v1 import preflight as kp_preflight


def read(path):
    return json.loads(Path(path).read_text())


def inventory(original, previous, output):
    import torch
    from experiments.ok_coding.training import load_snapshot, _export_equal
    from experiments.k_prefix_protection_v1.training_worker import validate_own
    from experiments.k_prefix_protection_v1.campaign import completed_job
    from .archive import audit_archive

    original, previous, output = map(Path, (original, previous, output))
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', model_work=0, original_run=str(original), pr21_run=str(previous))
    started = time.perf_counter()
    try:
        # This qualified audit checks the eight pools, all original event receipts,
        # initial K/final K decoded states and original native evidence on CPU.
        base = kp_preflight.inventory(original, output / 'original_reuse')
        receipt.update(base)
        receipt.update(status='running', pr21_run=str(previous))
        old_plan = read(original / 'sealed/PLAN.json')
        kp_plan = read(previous / 'sealed/PLAN.json')
        if kp_plan['execution_source']['commit'] != 'd10521a1aa7d256bb75b37a6aae07586ae7cc8b3':
            raise ValueError('PR21 execution identity differs')
        receipt['pr21_source'] = verify_seal(previous)
        endpoints, algorithms, recipes, initials = {}, {}, {}, {}
        for seed in (17, 29):
            key = str(seed)
            original_base = original / 'campaign_001' / f'seed_{seed}'
            initials[key] = str(original_base / 'O/initial_state.pt')
            endpoints[key] = {}
            for arm in ('O', 'K'):
                endpoint = original_base / arm / 'events/event_004'
                endpoints[key][arm] = dict(state=str(endpoint / 'training_state.pt'), export=str(endpoint / 'export'))
            half = previous / 'campaign_001' / f'seed_{seed}/K_HALF/events/event_004'
            endpoints[key]['K_HALF'] = dict(state=str(half / 'training_state.pt'), export=str(half / 'export'))
            for event, state_path, export in ((0, initials[key], old_plan['draft']),
                    (4, endpoints[key]['O']['state'], endpoints[key]['O']['export'])):
                state, check = load_snapshot(state_path, export, arm='O', seed=seed,
                    campaign_contract=base['original_campaign_contracts'][key],
                    config_text=Path(old_plan['configs'][key]).read_text(), completed_event=event)
                receipt.setdefault('o_states', {})[f'{seed}/{event}'] = check
                if event == 0:
                    algorithms[key] = state['algorithm_contract']
                    recipes[key] = state['backend']['optimizer']['recipe']
                del state
                gc.collect()
                receipt['files'][state_path] = identity(state_path)
                for path in Path(export).iterdir():
                    if path.is_file():
                        receipt['files'][str(path)] = identity(path)
            for event in range(1, 5):
                completed_job(previous, kp_plan, seed, 'K_HALF', event, decode=False)
                audit = previous / 'campaign_001' / f'seed_{seed}/resume_{event:03d}_K_HALF'
                for name in ('WORKER.json', 'EXEC.json', 'WORK.jsonl', 'CLOSED.json',
                             'SAVED_ENDPOINT.json', 'LIVE_BOUNDARY.json', 'STATE_IDENTITIES.json'):
                    receipt['files'][str(audit / name)] = identity(audit / name)
                result = previous / 'campaign_001' / f'seed_{seed}/K_HALF/events/event_{event:03d}/RESULT.json'
                receipt['files'][str(result)] = identity(result)
            state = torch.load(half / 'training_state.pt', map_location='cpu', weights_only=False)
            check = validate_own(state, kp_plan, kp_plan['configs'][key]['K_HALF'], 'K_HALF', seed, 4)
            _export_equal(state, half / 'export')
            receipt.setdefault('half_states', {})[key] = dict(check, export_exact=True)
            del state
            gc.collect()
            for path in [half / 'training_state.pt', *(half / 'export').iterdir()]:
                if path.is_file():
                    receipt['files'][str(path)] = identity(path)
        archive_path = output / 'ARCHIVE.json'
        archive = audit_archive(previous, output_path=archive_path)
        receipt['files'][str(archive_path)] = identity(archive_path)
        # Archive dependencies include original canonical tokens and native traces;
        # the CSV totals alone are deliberately insufficient.
        receipt['files'].update(archive['files'])
        receipt.update(original_initial_states=initials, original_endpoints=endpoints,
                       original_algorithm_contracts=algorithms, original_optimizer_recipes=recipes,
                       archive=dict(path=str(archive_path), identity=identity(archive_path)),
                       pr21_plan=str(previous / 'sealed/PLAN.json'), status='passed')
        receipt.pop('qualification', None)
        if torch.cuda.is_initialized():
            raise ValueError('CPU reuse audit initialized CUDA')
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        receipt['wall_s'] = time.perf_counter() - started
        write_json(output / 'REUSE_AUDIT.json', receipt)
    return receipt


def verify(root, *, postflight=False, receipt_path=None):
    import platform
    import torch
    import transformers
    import yaml
    from copy import deepcopy
    from experiments.kw_v1.preflight import hardware_admission
    from experiments.osd_components.prepare import verify_files, verify_native_runtime
    from .contract import validate_plan
    from .evaluation import bridge_schedule, final_schedule
    from .storage import admit
    from .training import validate_config_text

    root = Path(root)
    result = dict(status='running', model_work=0, postflight=postflight)
    started = time.perf_counter()
    try:
        result['seal'] = verify_seal(root)
        plan = read(root / 'sealed/PLAN.json')
        result['plan'] = validate_plan(plan)
        requests = plan['evaluation_requests']
        if requests != read(plan['archive']['path'])['requests']:
            raise ValueError('The immutable archived panel changed')
        if plan['serving'] != dict(block_size=16, max_new_tokens=2048, context_limit=4096,
                concurrency=1, temperature=0, capture=False, precision='bfloat16', attention_backend='fa3'):
            raise ValueError('Qualified serving configuration changed')
        if plan['bridge_schedule'] != sum((bridge_schedule(requests, seed) for seed in (17, 29)), []):
            raise ValueError('Compatibility bridge order changed')
        if plan['evaluation_schedule'] != sum((final_schedule(requests, seed) for seed in (17, 29)), []):
            raise ValueError('Final evaluation order changed')
        for path, expected in read(root / 'sealed/REUSE_AUDIT.json')['files'].items():
            if identity(path) != expected:
                raise ValueError('Reused artifact changed: ' + path)
        for seed in (17, 29):
            key = str(seed)
            path = Path(plan['configs'][key]['O_HALF'])
            raw = yaml.safe_load(path.read_text())
            expected = deepcopy(yaml.safe_load(Path(plan['original_configs'][key]).read_text()))
            expected.update(run_id=raw['run_id'], output_dir=raw['output_dir'])
            expected['training']['learning_rate'] = 1.5e-4
            if raw != expected:
                raise ValueError('Unexpected change outside O_HALF LR and output identity')
            validate_config_text(path.read_text(), seed, Path(plan['original_configs'][key]).read_text())
        pins = read(root / 'sealed/MODEL_PIN.json')
        for model in ('target', 'draft'):
            verify_files(plan[model], pins[model]['files'], exact_files=True)
        result['native_runtime'] = verify_native_runtime(plan['worker_python'], read(root / 'sealed/RUNTIME_PIN.json'))
        actual = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__)
        if actual != read(root / 'sealed/TRAINING_PIN.json')['environment'] or torch.cuda.is_initialized():
            raise ValueError('Training runtime changed or CPU preflight initialized CUDA')
        result['training_runtime'] = actual
        if not postflight:
            result['hardware'] = hardware_admission()
            result['storage'] = admit(root)
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['wall_s'] = time.perf_counter() - started
        write_json(Path(receipt_path) if receipt_path else root / ('POSTFLIGHT.json' if postflight else 'PREFLIGHT.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--previous')
    parser.add_argument('--inventory-output')
    parser.add_argument('--postflight', action='store_true')
    args = parser.parse_args()
    result = inventory(args.root, args.previous, args.inventory_output) if args.inventory_output else verify(args.root, postflight=args.postflight)
    print(json.dumps({key: result[key] for key in ('status', 'wall_s')}))


if __name__ == '__main__':
    main()
