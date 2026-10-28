"""Seal the fixed panel and construct two exact hybrids, without using a GPU."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import yaml

from online_feedback.storage import write_json
from .checkpoints import build_hybrids, _file_hash
from .protocol import (ARMS, BLOCK_SIZE, CONTEXT_LIMIT, OUTPUT_CAP, REQUESTS,
                       SEED, record_errors)


def verify_source(prior_root, code_root):
    """One preparation check against the successful run's immutable source seal."""
    sealed = Path(prior_root) / 'sealed'
    seal = json.loads((sealed / 'SEAL.json').read_text())
    checks = {}
    for filename, digest in seal['source_sha256'].items():
        checks[filename] = hashlib.sha256((Path(code_root) / filename).read_bytes()).hexdigest() == digest
    for filename, key in [('MANIFEST.json', 'manifest_sha256'), ('config.yaml', 'config_sha256')]:
        checks['sealed/' + filename] = hashlib.sha256((sealed / filename).read_bytes()).hexdigest() == seal[key]
    if not all(checks.values()):
        raise ValueError('Successful-run source seal differs: ' + str([k for k, v in checks.items() if not v]))
    return checks


def verify_native_runtime(worker_python, runtime_pin):
    """Inspect installed versions/source in the existing worker environment, CPU only."""
    script = '''import hashlib, importlib.util, json, pathlib, platform
import torch, transformers
root = pathlib.Path(importlib.util.find_spec('sglang').origin).parent / 'srt'
names = json.loads(input())
print(json.dumps({'python_version': platform.python_version(), 'torch': torch.__version__,
 'transformers': transformers.__version__, 'cuda': torch.version.cuda,
 'native_runtime_sha256': {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}}))
'''
    result = subprocess.run([str(worker_python), '-c', script],
                            input=json.dumps(list(runtime_pin['native_runtime_sha256'])) + '\n',
                            text=True, capture_output=True, check=True, timeout=120)
    actual = json.loads(result.stdout.strip().splitlines()[-1])
    for key in ('python_version', 'torch', 'transformers', 'cuda', 'native_runtime_sha256'):
        if actual[key] != runtime_pin[key]:
            raise ValueError('Pinned native runtime differs: ' + key)
    return actual


def verify_files(directory, files, *, exact_files=False):
    """Check immutable artifacts once at preparation/launch, outside timed serving."""
    if exact_files:
        actual = {path.name for path in Path(directory).iterdir() if path.is_file()}
        if actual != set(files):
            raise ValueError('Pinned artifact file set changed: ' + str(directory))
    for name, expected in files.items():
        path = Path(directory) / name
        if path.stat().st_size != expected['bytes'] or _file_hash(path) != expected['sha256']:
            raise ValueError('Pinned artifact changed: ' + str(path))


def verify_artifacts(prepared, plan, code):
    """Bind files actually loaded by the runtime to the sealed construction evidence."""
    prepared = Path(prepared)
    identities = json.loads((prepared / 'ARTIFACT_IDENTITIES.json').read_text())
    for arm, identity in identities['arms'].items():
        if identity['path'] != plan['arms'][arm]:
            raise ValueError('Checkpoint path differs from construction: ' + arm)
        verify_files(identity['path'], identity['files'], exact_files=True)
    target = identities['target']
    if target['path'] != plan['target']:
        raise ValueError('Target path differs from historical pin')
    verify_files(target['path'], target['files'], exact_files=True)
    verify_files(code, identities['experiment_source'])


def select_panel(manifest, evaluation_dir, arms):
    """Select indices first; only then read their retained answers and measurements."""
    candidates = manifest['eval_requests']
    if len(candidates) != 1024:
        raise ValueError('Expected the original 1,024-request evaluation manifest')
    indices = sorted(random.Random(SEED).sample(range(len(candidates)), REQUESTS))
    panel = []
    for ordinal, index in enumerate(indices):
        source = candidates[index]
        reference_path = Path(evaluation_dir) / f'{index:04d}_r0_F.json'
        frozen = json.loads(reference_path.read_text())
        online_path = Path(evaluation_dir) / f'{index:04d}_r0_O.json'
        online = json.loads(online_path.read_text())
        request = {name: frozen[name] for name in (
            'prompt_ids', 'response_ids', 'finish_reason', 'stop_token_ids', 'max_new_tokens')}
        request.update(request_id=source['request_id'], dataset_id=source['dataset_id'],
                       original_index=index, original_request_index=source['request_index'],
                       panel_ordinal=ordinal,
                       reference_calls={'F': frozen['metadata']['verification_calls'],
                                        'O': online['metadata']['verification_calls']},
                       reference_paths={'F': str(reference_path), 'O': str(online_path)})
        if (request['prompt_ids'] != source['prompt_ids'] or source['max_new_tokens'] != OUTPUT_CAP
                or request['max_new_tokens'] != OUTPUT_CAP or source['collect_feedback'] is not False
                or len(request['prompt_ids']) + OUTPUT_CAP > CONTEXT_LIMIT):
            raise ValueError('Saved input/cap differs from the sealed request: ' + source['request_id'])
        for repeat in range(3):
            for method in ('F', 'O'):
                path = Path(evaluation_dir) / f'{index:04d}_r{repeat}_{method}.json'
                record = json.loads(path.read_text())
                errors = record_errors(record, request, method, arms[method])
                if errors:
                    raise ValueError(f'Retained baseline record does not qualify: {path}: {errors}')
        panel.append(request)
    return panel


def prepare(args):
    root = Path(args.output).absolute()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    prior = Path(args.prior_root).absolute()
    code = Path(__file__).resolve().parents[2]
    source_check = verify_source(prior, code)
    config = yaml.safe_load((prior / 'sealed/config.yaml').read_text())
    frozen = config['model']['draft_checkpoint_path']
    trained = str(prior / 'campaign_001/trainer/events/event_058/export')
    endpoint_receipt = json.loads((prior / 'campaign_001/events/event_058/PUBLICATION_CHECK.json').read_text())
    if (endpoint_receipt['checkpoint_path'] != trained or not endpoint_receipt['all_equal']
            or endpoint_receipt['trainer_status']['global_step'] != 7424):
        raise ValueError('Final checkpoint identity/accounting differs from the successful endpoint')
    model_pin = json.loads(Path(args.model_pin).read_text())
    verify_files(config['model']['target_model_path'], model_pin['target']['files'], exact_files=True)
    verify_files(frozen, model_pin['draft']['files'], exact_files=True)
    write_json(root / 'MODEL_PIN.json', model_pin)
    write_json(root / 'ENDPOINT_PUBLICATION.json', endpoint_receipt)
    manifest = json.loads((prior / 'sealed/MANIFEST.json').read_text())
    requests = select_panel(manifest, prior / 'campaign_001/evaluation', {'F': frozen, 'O': trained})
    write_json(root / 'PANEL.json', {
        'selection_seed': SEED, 'selection': 'uniform_without_replacement_then_original_order',
        'source_manifest': str(prior / 'sealed/MANIFEST.json'),
        'status': 'sealed_before_hybrid_construction_and_generation', 'requests': requests})
    pin = json.loads(Path(args.runtime_pin).read_text())
    runtime_check = verify_native_runtime(args.worker_python, pin)
    write_json(root / 'RUNTIME_CHECK.json', runtime_check)
    shutil.copy2(args.runtime_pin, root / 'RUNTIME_PIN.json')
    shutil.copy2(prior / 'sealed/config.yaml', root / 'SOURCE_TRAINING_CONFIG.yaml')
    hybrids = build_hybrids(Path(frozen), Path(trained), root / 'checkpoints')
    identities = {
        'arms': dict(hybrids['inventory']['sources']),
        'target': {'path': config['model']['target_model_path'],
                   'files': model_pin['target']['files']},
        'experiment_source': {
            str(path.relative_to(code)): {'bytes': path.stat().st_size, 'sha256': _file_hash(path)}
            for path in sorted((code / 'experiments/osd_components').iterdir())
            if path.suffix in ('.py', '.sh')},
        'endpoint_provenance': {
            'source': 'retained event_058 export; original 7424-step publication receipt',
            'historical_export_digest_available': False,
            'current_digest_scope': 'Pins the retained file now; not a historical digest comparison.',
            'required_native_check': 'F/O must reproduce historical output IDs and verification counts.'}}
    for arm in ('C-only', 'B-only'):
        artifact = hybrids['qualification']['hybrids'][arm]
        identities['arms'][arm] = {key: artifact[key] for key in ('path', 'files')}
    # Include every original released file, including architecture and tokenizer artifacts.
    identities['arms']['F']['files'] = model_pin['draft']['files']
    write_json(root / 'ARTIFACT_IDENTITIES.json', identities)
    plan = {
        'status': 'prepared_not_gpu_qualified', 'prior_root': str(prior),
        'target': config['model']['target_model_path'], 'arms': hybrids['arms'],
        'worker_python': str(args.worker_python), 'runtime_pin': str(root / 'RUNTIME_PIN.json'),
        'panel': str(root / 'PANEL.json'), 'model_arms': list(ARMS),
        'requests': REQUESTS, 'repeats': 3, 'generations': 768, 'qualification_requests': 8,
        'optimizer_steps': 0, 'feedback_collections': 0, 'extra_generation_warmups': 0,
        'block_size': BLOCK_SIZE, 'context_limit': CONTEXT_LIMIT, 'max_new_tokens': OUTPUT_CAP,
        'precision': 'bfloat16', 'temperature': 0, 'attention_backend': 'fa3',
        'concurrency': 1, 'max_wall_seconds': 7200,
        'source_checks': source_check, 'preparation_s': time.perf_counter() - started,
        'preparation_cost_note': 'CPU artifact preparation; no model generations or training.'}
    write_json(root / 'PLAN.json', plan)
    # Seal small run inputs once. No per-request hashing or model scans are added.
    inputs = ['PANEL.json', 'PLAN.json', 'RUNTIME_PIN.json', 'RUNTIME_CHECK.json',
              'SOURCE_TRAINING_CONFIG.yaml', 'MODEL_PIN.json', 'ENDPOINT_PUBLICATION.json',
              'ARTIFACT_IDENTITIES.json', 'checkpoints/PARAMETER_INVENTORY.json',
              'checkpoints/CHECKPOINT_QUALIFICATION.json', 'checkpoints/STORAGE_ADMISSION.json']
    write_json(root / 'INPUT_SEAL.json', {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in inputs})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prior-root', 'runtime-pin', 'model-pin', 'worker-python', 'output'):
        parser.add_argument('--' + name, required=True)
    plan = prepare(parser.parse_args())
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
