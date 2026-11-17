"""Qualify exact CPU reconstruction before retiring a derived serving export."""
from pathlib import Path
import hashlib
import json
import time

from experiments.ok_coding.training import _export_equal
from experiments.bounded_prefix_allocation.prepare import identity
from experiments.ok_continuation.resume import differences
from experiments.persistent_prefix_withholding.qualification import tensor_identity
from online_feedback.rng import preserve_rng
from online_feedback.storage import write_json


def export_manifest(path):
    """Tensor identities are independent of safetensors file layout/sharding."""
    from safetensors import safe_open
    path = Path(path)
    tensors, metadata = {}, {}
    for file in sorted(path.iterdir()):
        if file.suffix == '.safetensors':
            with safe_open(file, framework='pt', device='cpu') as source:
                for name in source.keys():
                    if name in tensors:
                        raise ValueError('Duplicate serving tensor: ' + name)
                    tensors[name] = tensor_identity(source.get_tensor(name))
        elif file.name != 'model.safetensors.index.json' and file.is_file():
            data = file.read_bytes()
            metadata[file.name] = json.loads(data) if file.suffix == '.json' else dict(
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    if not tensors or 'config.json' not in metadata:
        raise ValueError('Serving export lacks tensors or required configuration')
    return dict(tensors=tensors, metadata=metadata)


def reconstruct_export(state_path, draft_config_path, output_dir, *, embedding_source=None,
                       embedding_key='model.embed_tokens.weight'):
    """Invoke the qualified author CPU exporter without changing training RNG."""
    from specforge.export.to_hf import export_to_hf
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('Reconstruction requires a new derived-copy destination')
    config = Path(draft_config_path)
    if config.is_dir():
        config /= 'config.json'
    with preserve_rng():
        export_to_hf(str(state_path), str(config), str(output),
                     embedding_source=embedding_source, embedding_key=embedding_key)
    return str(output)


def verify_derived_export(state_path, export_path, *, reconstruction, receipt_path=None):
    """Check actual recovery tensors and pinned configuration, never just file hashes."""
    import torch
    began = time.perf_counter()
    state = torch.load(state_path, map_location='cpu', weights_only=False, mmap=True)
    _export_equal(state, export_path)
    manifest = export_manifest(export_path)
    result = dict(status='passed', complete_recovery=str(state_path), derived_export=str(export_path),
        global_step=state['global_step'], presentations=state['presentations'],
        manifest=manifest, reconstruction=reconstruction,
        weights_files={file.name:identity(file) for file in sorted(Path(export_path).glob('*.safetensors'))},
        optimizer_steps=0, gpu_inference=0, wall_s=time.perf_counter()-began)
    if receipt_path is not None:
        write_json(receipt_path, result)
    return result


def reconstruction_command(state_path, draft_config_path, output_dir, receipt_path, *,
                           embedding_source=None, embedding_key='model.embed_tokens.weight'):
    argv = ['python', '-m', 'experiments.c_paired_confirmation.export',
            '--state', str(state_path), '--draft-config', str(draft_config_path),
            '--output-dir', str(output_dir), '--receipt', str(receipt_path),
            '--embedding-key', embedding_key]
    if embedding_source is not None:
        argv.extend(['--embedding-source', str(embedding_source)])
    return dict(argv=argv, embedding_source=embedding_source, embedding_key=embedding_key)


def qualify_cpu_export(state_path, draft_config_path, expected_export, output_dir, *,
                       embedding_source=None, embedding_key='model.embed_tokens.weight', receipt_path=None):
    """Rebuild from complete recovery and compare authenticated serving metadata too."""
    began = time.perf_counter()
    command = reconstruction_command(state_path, draft_config_path, output_dir,
        receipt_path or Path(output_dir).parent/'CPU_EXPORT_RECONSTRUCTION.json',
        embedding_source=embedding_source, embedding_key=embedding_key)
    reconstruct_export(state_path, draft_config_path, output_dir,
                       embedding_source=embedding_source, embedding_key=embedding_key)
    result = verify_derived_export(state_path, output_dir, reconstruction=command)
    tensor_metadata_mismatch = differences(export_manifest(expected_export), result['manifest'])
    expected_files = {file.name:identity(file) for file in sorted(Path(expected_export).glob('*.safetensors'))}
    file_mismatch = differences(expected_files, result['weights_files'])
    mismatch = tensor_metadata_mismatch + ['weights_file_identity: '+item for item in file_mismatch]
    result.update(expected_export=str(expected_export), exact_metadata_and_tensors=not tensor_metadata_mismatch,
                  expected_weights_files=expected_files, exact_weights_file_identity=not file_mismatch,
                  differences=mismatch, wall_s=time.perf_counter()-began)
    if mismatch:
        result['status'] = 'failed'
    if receipt_path is not None:
        write_json(receipt_path, result)
    if mismatch:
        raise ValueError('CPU reconstruction differs from authenticated serving export: ' + str(mismatch[:8]))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True)
    parser.add_argument('--draft-config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--embedding-source')
    parser.add_argument('--embedding-key', default='model.embed_tokens.weight')
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    reconstruct_export(args.state, args.draft_config, args.output_dir,
        embedding_source=args.embedding_source, embedding_key=args.embedding_key)
    verify_derived_export(args.state, args.output_dir, reconstruction=reconstruction_command(
        args.state, args.draft_config, args.output_dir, args.receipt,
        embedding_source=args.embedding_source, embedding_key=args.embedding_key), receipt_path=args.receipt)


if __name__ == '__main__':
    main()
