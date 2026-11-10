"""Seal IDs first, then bind private records without selecting by their outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .contract import CHECKPOINTS, DATASETS, interleaved_requests, scientific_config, select_requests


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def check_receipt(receipt, field='local_path'):
    actual = identity(receipt[field])
    if any(actual[key] != receipt[key] for key in ('bytes', 'sha256')):
        raise ValueError('Artifact differs from retained receipt: ' + receipt[field])


def seal(inventory_path, output):
    inventory = read(inventory_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    panels = {}
    selected = {}
    # Selection is completed and durably written before loading any outcomes.
    for dataset in DATASETS:
        receipt = inventory['datasets'][dataset]['panel']
        check_receipt(receipt)
        panels[dataset] = read(receipt['local_path'])['requests']
        selected[dataset] = select_requests(dataset, panels[dataset])
    ordered = interleaved_requests(selected)
    write(output / 'SELECTED_IDS.json', dict(
        status='sealed_before_outcome_analysis', config=scientific_config(),
        source_inventory=identity(inventory_path), selected=selected, execution_order=ordered))
    private = []
    bound_files = {}
    for dataset in DATASETS:
        item = inventory['datasets'][dataset]
        check_receipt(item['canonical'])
        canonical = read(item['canonical']['local_path'])
        requests = {r['request_id']: r for r in panels[dataset]}
        records = {r['request_id']: r for r in item['records']}
        for chosen in selected[dataset]:
            request_id = chosen['request_id']
            request = requests[request_id]
            source = records[request_id]
            # Retain only serving inputs, never benchmark answers as new data.
            serving = {key: request[key] for key in ('request_id', 'prompt_ids',
                       'max_new_tokens', 'stop_token_ids', 'panel_ordinal')}
            native = {}
            for checkpoint in CHECKPOINTS:
                receipt = source[checkpoint][0]
                check_receipt(receipt)
                record = read(receipt['local_path'])
                if (record['prompt_ids'] != serving['prompt_ids']
                        or record['response_ids'] != canonical[request_id]['response_ids']
                        or record['finish_reason'] != canonical[request_id]['finish_reason']
                        or record['checkpoint_id'] != item['checkpoints'][checkpoint]['export']):
                    raise ValueError('Original native record does not bind this request/checkpoint')
                native[checkpoint] = dict(path=receipt['source_path'], local_path=receipt['local_path'],
                                          bytes=receipt['bytes'], sha256=receipt['sha256'])
                bound_files[receipt['source_path']] = {key: receipt[key] for key in ('bytes','sha256')}
            private.append(dict(**chosen, request=serving, canonical=canonical[request_id], native=native))
    rank = {(r['dataset'],r['request_id']): index for index,r in enumerate(ordered)}
    private.sort(key=lambda row: rank[row['dataset'],row['request_id']])
    write(output / 'INPUTS.json', dict(
        config=scientific_config(), shared=inventory['shared'], requests=private,
        checkpoints={d:inventory['datasets'][d]['checkpoints'] for d in DATASETS},
        bound_native_files=bound_files, selected_ids_identity=identity(output/'SELECTED_IDS.json'),
        qualification='pending native own-successor reconstruction',
        source_live_availability='pending A800 preflight'))
    write(output / 'SEAL.json', dict(status='prepared_no_model_work',
          selected=identity(output/'SELECTED_IDS.json'), private_inputs=identity(output/'INPUTS.json'),
          requests=len(private), native_original_records=len(bound_files)))
    return read(output/'SEAL.json')


def verify_seal(inputs_path):
    """Authenticate the prepared bytes and selected IDs before any model work."""
    inputs_path = Path(inputs_path)
    seal_receipt = read(inputs_path.parent/'SEAL.json')
    selected_path = inputs_path.parent/'SELECTED_IDS.json'
    if identity(inputs_path) != seal_receipt['private_inputs']:
        raise ValueError('Prepared INPUTS bytes changed after sealing')
    selected_identity = identity(selected_path)
    if selected_identity != seal_receipt['selected']:
        raise ValueError('Selected-ID manifest changed after sealing')
    plan = read(inputs_path)
    if selected_identity != plan['selected_ids_identity']:
        raise ValueError('INPUTS refers to a different selected-ID manifest')
    selected = read(selected_path)
    fields = ('dataset','request_id','selection_rank','selection_sha256')
    rows = [{key:row[key] for key in fields} for row in plan['requests']]
    if rows != selected['execution_order']:
        raise ValueError('Execution requests differ from sealed selection')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args.inventory, args.output), indent=2))


if __name__ == '__main__':
    main()
