"""CPU-only checks of the original full-stream evidence and new sealed run."""
import argparse
import json
from pathlib import Path
import shutil
import time

from online_feedback.storage import write_json
from experiments.ok_events_5_8.storage import identity
from experiments.ok_coding.preflight import verify_seal
from experiments.osd_persistent.data import validate_manifest
from experiments.osd_persistent.run import score_record
from experiments.o_half_matched_lr.archive import ArchiveReader, validate_record
from experiments.fresh_confirmation.protocol import (
    COUNT_FIELDS, measurement_errors, publication_errors, repeat_errors, weight_check_errors,
)
from .contract import ARMS, INTERMEDIATE_EVENTS, stable_digest, validate_plan


def read(path):
    return json.loads(Path(path).read_text())


def audit_evaluation(reader, manifest, checkpoints):
    """Recover canonicals from original F, checking all six native executions."""
    requests, canonicals, totals = [], {}, {method: 0 for method in ('F', 'O')}
    folder = reader.root / 'campaign_001/evaluation'
    blocks = 0
    for ordinal, original in enumerate(manifest['eval_requests']):
        first_path = folder / f'{ordinal:04d}_r0_F.json'
        first = reader.read(first_path)
        request = dict(original, panel_ordinal=ordinal, original_index=original['request_index'],
                       stop_token_ids=first['stop_token_ids'])
        canonical = dict(response_ids=first['response_ids'], finish_reason=first['finish_reason'],
            source_record_path=str(first_path), F_checkpoint=checkpoints['F'])
        groups = {method: [] for method in ('F', 'O')}
        for repeat in range(3):
            for method in ('F', 'O'):
                label = f'{ordinal:04d}_r{repeat}_{method}'
                record = first if repeat == 0 and method == 'F' else reader.read(folder / (label + '.json'))
                errors = validate_record(record, request, checkpoints[method], canonical,
                                         label=label, session='heldout')
                row = score_record(record, request_id=request['request_id'], repeat=repeat, method=method)
                row.update(finish_reason=record['finish_reason'],
                           cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'])
                errors.extend(measurement_errors(row, request, canonical))
                native = reader.read(reader.saved_path(record['metadata']['native_trace_path']))
                if native != record['metadata']['capture']:
                    errors.append('native_source_differs')
                if errors:
                    raise ValueError(label + ': ' + repr(errors))
                groups[method].append(row)
                blocks += len(record['metadata']['trace'])
            pair = reader.read(folder / f'{ordinal:04d}_r{repeat}_comparison.json')
            order = ['F', 'O'] if (ordinal + repeat) % 2 == 0 else ['O', 'F']
            if (pair['request_id'] != request['request_id'] or pair['repeat'] != repeat
                    or pair['order'] != order or pair['valid'] is not True or pair['errors']
                    or pair['canonical_response_ids'] != canonical['response_ids']):
                raise ValueError('Original pair validity or schedule differs')
        for method, rows in groups.items():
            if repeat_errors(rows, method):
                raise ValueError('Original per-request verification counts differ across repeats')
            totals[method] += rows[0]['verification_calls']
        canonical['F_counts'] = {key: groups['F'][0][key] for key in COUNT_FIELDS}
        canonicals[request['request_id']] = canonical
        requests.append(request)
    summary = reader.read(folder / 'SUMMARY.json')
    count = len(requests)
    if (summary['distinct_requests'] != count or summary['valid_pairs'] != count * 3
            or summary['invalid_pairs'] != 0
            or any(summary[m]['verification_calls'] != 3 * totals[m] for m in totals)):
        raise ValueError('Original published count totals differ from native trajectories')
    for publication in reader.read(folder / 'BENCHMARK_ACTIVATIONS.json'):
        if publication['checkpoint_id'] not in checkpoints.values() or publication_errors(
                publication, publication['checkpoint_id']):
            raise ValueError('Original evaluation publication was invalid')
    return dict(requests=requests, canonicals=canonicals, raw_generations=count * 6,
                native_blocks=blocks, call_totals=totals, latency_reused=False)


def original_source_check(code, seal):
    """Keep original bytes; allow only the reviewed additive objective/trainer files."""
    code = Path(code)
    differences = []
    for name, old_hash in seal['source_sha256'].items():
        current = identity(code / name)
        if current['sha256'] == old_hash:
            continue
        if name not in ('online_feedback/objective.py', 'online_feedback/training.py'):
            raise ValueError('Unexpected original numerical/runtime source change: ' + name)
        archived = code / 'provenance/objects' / old_hash
        if identity(archived)['sha256'] != old_hash:
            raise ValueError('Missing exact historical implementation: ' + name)
        differences.append(dict(path=name, historical=old_hash, current=current['sha256'],
                                historical_object=str(archived)))
    return differences


def inventory(original, pins, output, *, code=None):
    """Read retained artifacts without reconstructing features or initializing CUDA."""
    import yaml
    original, pins, output = map(Path, (original, pins, output))
    code = Path(code) if code else Path(__file__).resolve().parents[2]
    output.mkdir(parents=True, exist_ok=False)
    reader = ArchiveReader(original)
    receipt = dict(status='running', model_work=0, original_run=str(original), files={})
    started = time.perf_counter()
    try:
        seal = reader.read(original / 'sealed/SEAL.json')
        manifest_path, config_path = original / 'sealed/MANIFEST.json', original / 'sealed/config.yaml'
        manifest = reader.read(manifest_path)
        receipt['manifest_check'] = validate_manifest(manifest)
        for path, field in ((manifest_path, 'manifest_sha256'), (config_path, 'config_sha256')):
            reader.files[str(path)] = identity(path)
            if reader.files[str(path)]['sha256'] != seal[field]:
                raise ValueError('Original sealed input changed: ' + str(path))
        differences = original_source_check(code, seal)
        config = yaml.safe_load(config_path.read_text())
        draft, target = config['model']['draft_checkpoint_path'], config['model']['target_model_path']
        endpoint = original / 'campaign_001/trainer/events/event_058'
        result = reader.read(endpoint / 'RESULT.json')
        expected = dict(status='completed', event_id='event_058', objective='osd',
            passes=2, response_presentations=256, start_step=7296, end_step=7424, optimizer_steps=128,
            reduction='response_mean', counters=dict(global_step=7424, presentations=14848, completed_events=58))
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError('Original O58 event accounting differs')
        checkpoints = dict(F=draft, O=str(endpoint / 'export'))
        if result['export_path'] != checkpoints['O'] or result['checkpoint_path'] != checkpoints['O']:
            raise ValueError('Original O58 export identity changed')
        status = reader.read(original / 'campaign_001/STATUS.json')
        if any(status.get(k) != v for k, v in dict(status='completed', serving_completed=7473,
                updates_completed=58, optimizer_steps=7424).items()):
            raise ValueError('Original full O stream is incomplete')
        publication = reader.read(original / 'campaign_001/events/event_058/PUBLICATION_CHECK.json')
        if weight_check_errors(publication, checkpoints['O']):
            raise ValueError('Original O58 loaded/exported weight check failed')
        for directory in checkpoints.values():
            for path in Path(directory).iterdir():
                if path.is_file():
                    reader.files[str(path)] = identity(path)
        panel = audit_evaluation(reader, manifest, checkpoints)
        pin_plan = reader.read(pins / 'PLAN.json')
        for name in ('MODEL_PIN.json', 'RUNTIME_PIN.json', 'TRAINING_PIN.json'):
            reader.read(pins / name)
        if (pin_plan['target'] != target or pin_plan['draft'] != draft
                or pin_plan['training_python'] != seal['trainer_python']):
            raise ValueError('Pinned model or training runtime differs from original full O')
        shutil.copy2(manifest_path, output / 'ORIGINAL_MANIFEST.json')
        shutil.copy2(config_path, output / 'ORIGINAL_CONFIG.yaml')
        write_json(output / 'CANONICALS.json', panel.pop('canonicals'))
        receipt.update(status='passed', **panel, target=target, draft=draft, checkpoints=checkpoints,
            original_manifest=str(manifest_path), original_config=str(config_path),
            manifest_binding=stable_digest(manifest), source_differences=differences,
            worker_python=pin_plan['worker_python'], training_python=pin_plan['training_python'],
            retained_state_required=False, old_feature_reconstruction=False,
            canonical_identity=identity(output / 'CANONICALS.json'))
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        receipt.update(files=reader.files, wall_s=time.perf_counter() - started)
        write_json(output / 'REUSE_AUDIT.json', receipt)
    return receipt


def verify(root, *, postflight=False, receipt_path=None):
    import platform
    import torch
    import transformers
    from experiments.kw_v1.preflight import hardware_admission
    from experiments.osd_components.prepare import verify_files, verify_native_runtime
    from .evaluation import final_schedule, intermediate_schedule, schedule_balance
    from .storage import admit, capacity_plan
    from .training import validate_config_text

    root = Path(root)
    result, started = dict(status='running', model_work=0, postflight=postflight), time.perf_counter()
    try:
        result['seal'] = verify_seal(root)
        plan = read(root / 'sealed/PLAN.json')
        result['plan'] = validate_plan(plan)
        requests = plan['evaluation_requests']
        reuse = read(root / 'sealed/REUSE_AUDIT.json')
        if requests != reuse['requests'] or identity(plan['canonicals']) != reuse['canonical_identity']:
            raise ValueError('Original request/canonical binding changed')
        for path, expected in reuse['files'].items():
            if identity(path) != expected:
                raise ValueError('Reused scientific input changed: ' + path)
        expected_intermediates = {f'{arm}/{event}': intermediate_schedule(requests, arm, event)
            for arm in ARMS for event in INTERMEDIATE_EVENTS}
        if (plan['evaluation_schedule'] != final_schedule(requests)
                or plan['intermediate_schedules'] != expected_intermediates):
            raise ValueError('Sealed evaluation schedule changed')
        balance = schedule_balance(plan['evaluation_schedule'])
        if (any(v != [12] * 4 for v in balance['method_positions'].values())
                or any(v != 24 for v in balance['pairwise_precedence'].values())):
            raise ValueError('Final checkpoint order is not balanced')
        original_config = Path(plan['original_config']).read_text()
        for arm in ARMS:
            validate_config_text(Path(plan['configs'][arm]).read_text(), arm, original_config)
        pins = read(root / 'sealed/MODEL_PIN.json')
        for model in ('target', 'draft'):
            verify_files(plan[model], pins[model]['files'], exact_files=True)
        result['native_runtime'] = verify_native_runtime(plan['worker_python'], read(root / 'sealed/RUNTIME_PIN.json'))
        actual = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__)
        if actual != read(root / 'sealed/TRAINING_PIN.json')['environment'] or torch.cuda.is_initialized():
            raise ValueError('Training runtime changed or CPU preflight initialized CUDA')
        result['training_runtime'] = actual
        weight_bytes = (Path(plan['draft']) / 'model.safetensors').stat().st_size
        if plan['storage'] != capacity_plan(weight_bytes):
            raise ValueError('Sealed storage peak differs from actual released weight size')
        if not postflight:
            result['hardware'] = hardware_admission()
            memory = {k: int(v.split()[0]) * 1024 for k, v in
                      (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
            result['host_available_bytes'] = memory['MemAvailable']
            if memory['MemAvailable'] < 80 * 2**30:
                raise ValueError('Require at least 80 GiB currently available host memory')
            result['storage'] = admit(root, plan['storage'])
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
    parser.add_argument('--pins')
    parser.add_argument('--inventory-output')
    parser.add_argument('--code')
    parser.add_argument('--postflight', action='store_true')
    args = parser.parse_args()
    result = (inventory(args.root, args.pins, args.inventory_output, code=args.code)
        if args.inventory_output else verify(args.root, postflight=args.postflight))
    print(json.dumps({key: result[key] for key in ('status', 'wall_s')}))


if __name__ == '__main__':
    main()
