"""Bind a clean published implementation to the already selected private panel."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from experiments.lost_repair.prepare import source_manifest, identity, read, write

GIB = 1024**3


def prepare_run(inputs, prior_plan, output, remote_root, code=None):
    inputs, output, remote_root = Path(inputs), Path(output), Path(remote_root)
    code = Path(code or Path(__file__).resolve().parents[2])
    if output.exists():
        raise FileExistsError('Keep the earlier execution seal unchanged')
    if not remote_root.is_relative_to('/data/sources/minhao/runs'):
        raise ValueError('Explicit A800 artifact root required')
    source = source_manifest(code)
    bundle = read(inputs/'INPUTS.json')
    input_seal = read(inputs/'INPUT_SEAL.json')
    for name, expected in input_seal['files'].items():
        if identity(inputs/name) != expected:
            raise ValueError('Prepared input selection seal changed: '+name)
    prior = read(prior_plan)
    output.mkdir(parents=True)
    sealed = output/'sealed'
    sealed.mkdir()
    write(output/'SOURCE.json', source)
    pinned = {str(remote_root/'SOURCE.json'):identity(output/'SOURCE.json')}
    for path in inputs.iterdir():
        if path.is_file():
            shutil.copyfile(path, sealed/path.name)
            pinned[str(remote_root/'sealed'/path.name)] = identity(path)
    requests = []
    all_context_tokens = 0
    max_snapshot_tokens = 0
    for entry in bundle['requests']:
        row = deepcopy(entry['request'])
        canonical = deepcopy(entry['canonical'])
        canonical.update(max_new_tokens=row['max_new_tokens'], stop_token_ids=row['stop_token_ids'])
        relative = Path('sealed/requests')/row['request_id']
        destination = output/relative
        destination.mkdir(parents=True)
        write(destination/'CANONICAL.json', canonical)
        native = entry['native_K5']
        path = Path(native['local_path'])
        if identity(path) != {key:native[key] for key in ('bytes','sha256')}:
            raise ValueError('Authenticated K5 native trace changed')
        shutil.copyfile(path, destination/'K5_ARCHIVE.json')
        for name in ('CANONICAL.json','K5_ARCHIVE.json'):
            pinned[str(remote_root/relative/name)] = identity(destination/name)
        row.update(canonical_path=str(remote_root/relative/'CANONICAL.json'),
                   archived_trace_path=str(remote_root/relative/'K5_ARCHIVE.json'))
        requests.append(row)
        trace = read(path)['metadata']['trace']
        prompt_length = len(row['prompt_ids'])
        all_context_tokens += prompt_length+len(canonical['response_ids'])
        max_snapshot_tokens = max(max_snapshot_tokens,
            sum(prompt_length+block['output_start'] for block in trace))
    # Native Qwen target KV, five-layer draft KV and target-feature ingress.
    # This deliberately overcounts longest prefixes and ingested final tokens.
    bytes_per_context_token = 176*1024
    compact_bound = all_context_tokens*bytes_per_context_token
    per_request_scratch_bound = max_snapshot_tokens*bytes_per_context_token
    host_memory = dict(bytes_per_context_token=bytes_per_context_token,
        total_context_tokens=all_context_tokens, max_request_snapshot_tokens=max_snapshot_tokens,
        compact_bound_bytes=compact_bound, per_request_scratch_bound_bytes=per_request_scratch_bound,
        other_model_and_metadata_allowance_bytes=32*GIB, reserve_bytes=8*GIB,
        required_available_bytes=compact_bound+per_request_scratch_bound+40*GIB)
    # Keep original model/runtime artifacts pinned, omitting unrelated training
    # recovery states and old execution packages from this no-training probe.
    for path, expected in prior['pinned_files'].items():
        if path.startswith('/data/sources/minhao/models/'):
            pinned[path] = expected
    checkpoints = deepcopy(prior['checkpoints'])
    for checkpoint in checkpoints.values():
        pinned[str(Path(checkpoint['export'])/'config.json')] = checkpoint['config_identity']
    plan = dict(schema='repair-history-margin-native-v1', execution_commit=source['commit'],
        source_manifest_path=str(remote_root/'SOURCE.json'),
        requests=requests, panel_identity=identity(inputs/'PANEL.json'),
        input_seal_identity=identity(inputs/'INPUT_SEAL.json'),
        preparation_inputs_identity=identity(inputs/'INPUTS.json'),
        checkpoints=checkpoints, pinned_files=pinned,
        target_path=prior['target_path'], worker_python=prior['worker_python'],
        versions={'serving':prior['versions']['serving']}, runtime_contract=prior['runtime_contract'],
        serving=prior['serving'], gpu_seconds_limit=10800,
        storage=dict(artifact_allowance_bytes=8*GIB,reserve_bytes=8*GIB,required_free_bytes=16*GIB),
        host_memory=host_memory, output_root=str(remote_root/'campaign_001'),
        analysis=dict(seed=20261115,caliper=0.25,bootstrap_draws=10000,threshold_pp=5,
            minimum_matched_repaired_starts=100,minimum_matched_requests=40,maximum_absolute_smd=0.10,
            qa_hash='SHA256 compactUTF8JSON [20261115,source_id,start_offset,slot]; bottom32acceptedrows',
            matching_sealed_before_k6=True),
        scope=dict(parent_instrumented_walks=len(requests),archived_k5_rounds=bundle['traces']['rounds'],
                   optimizer_steps=0,backward_calls=0,training_forward=0,timed_endpoints=0),
        physical_gpu_index=0,physical_gpu_uuid='GPU-9986735a-3e68-d1bd-5998-9634ef24341e')
    write(sealed/'PLAN.json',plan)
    write(sealed/'SEAL.json',dict(status='sealed',execution_commit=source['commit'],
        files={str(path.relative_to(output)):identity(path) for path in sorted(output.rglob('*')) if path.is_file()}))
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','prior-plan','output','remote-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    plan=prepare_run(args.inputs,args.prior_plan,args.output,args.remote_root)
    print(json.dumps(dict(status='sealed',requests=len(plan['requests']),commit=plan['execution_commit'],
                         host_memory=plan['host_memory'])))

if __name__=='__main__':main()
