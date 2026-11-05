"""Peak admission and bounded retirement of this run's redundant tensors.

Native captures are retired only after the equivalent prepared packet is
durable and checked. Prepared packets and predecessor checkpoints survive
until the next complete state/export has passed native publication checks.
Compact evidence, initial state and current recovery state are never retired.
"""
import json
import os
from pathlib import Path
import shutil
import time

from experiments.ok_coding.storage import identity
from experiments.osd_persistent.retention import _inside, _json
from online_feedback.storage import write_json

GIB = 2**30
CACHE_ENVIRONMENT = {
    'TORCH_HOME': '/data/sources/minhao/.cache/torch',
    'TORCH_EXTENSIONS_DIR': '/data/sources/minhao/.cache/torch_extensions',
    'TORCHINDUCTOR_CACHE_DIR': '/data/sources/minhao/.cache/torchinductor',
    'TVM_FFI_CACHE_DIR': '/data/sources/minhao/.cache/tvm-ffi',
    'TRITON_CACHE_DIR': '/data/sources/minhao/.cache/triton',
    'CUDA_CACHE_PATH': '/data/sources/minhao/.cache/cuda',
    'SGLANG_CACHE_DIR': '/data/sources/minhao/.cache/sglang',
}


def capacity_plan(weight_bytes):
    # Initial weights + two complete model/Adam states + two exports. During
    # preparation one replacement packet and its terminal row overlap the
    # 128 captured/prepared responses. No future offload receives credit.
    rows = 128 * 2272
    features = (rows + 2272 + 1) * (4096 * 6 * 2 + 12)
    tensors = 9 * weight_bytes + features
    padded = (tensors * 105 + 99) // 100
    allowances = dict(compact_bytes=8 * GIB, compiler_bytes=4 * GIB,
                      workspace_bytes=2 * GIB, reserve_bytes=8 * GIB)
    return dict(weight_bytes=weight_bytes, state_weight_equivalents=9,
                feature_peak_bytes=features, tensor_peak_bytes=tensors,
                tensor_peak_with_margin_bytes=padded, **allowances,
                required_initial_free_bytes=padded + sum(allowances.values()),
                lifecycle="incremental durable preparation; validated event retirement",
                offload_assumed=False)


def boundary_files(root):
    base = Path(root) / 'campaign_001/arms/KPLUS'
    paths = [base / 'trainer/initial_state.pt']
    for receipt in sorted((base / 'boundaries').glob('event_[0-9][0-9][0-9].json')):
        boundary = json.loads(receipt.read_text())
        if boundary['status'] != 'passed':
            raise ValueError('Cannot credit an unvalidated recovery boundary')
        paths.extend(Path(boundary[key]) for key in ('state', 'weights'))
    return [path for path in paths if path.exists()]


def allocated_bytes(paths):
    """Physical allocation once per inode, including directory metadata."""
    total, seen = 0, set()
    for path in paths:
        if path.is_symlink():
            raise ValueError('Storage accounting cannot follow a symlink: ' + str(path))
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode not in seen:
            total += stat.st_blocks * 512
            seen.add(inode)
    return total


def compiler_snapshot():
    """Read only the pinned compiler cache roots, never the whole workspace."""
    directories = {name: str(Path(os.environ.get(name, default)).absolute())
                   for name, default in CACHE_ENVIRONMENT.items()}
    flashinfer_base = Path(os.environ.get('FLASHINFER_WORKSPACE_BASE', '/data/sources/minhao'))
    directories['FLASHINFER_CACHE'] = str((flashinfer_base / '.cache/flashinfer').absolute())
    paths = []
    for value in sorted(set(directories.values())):
        directory = Path(value)
        if directory.is_symlink() or directory.resolve() != directory:
            raise ValueError('Compiler cache root contains a symlink: ' + value)
        if directory.exists():
            if not directory.is_dir():
                raise ValueError('Compiler cache root is not a directory: ' + value)
            paths.append(directory)
            paths.extend(directory.rglob('*'))
    return dict(directories=directories, allocated_bytes=allocated_bytes(paths))


def compiler_growth(root):
    current = compiler_snapshot()
    path = root / 'COMPILER_BASELINE.json'
    if path.exists():
        baseline = _json(path, root)
        if baseline['directories'] != current['directories']:
            raise ValueError('Compiler cache roots changed after initial admission')
    else:
        if any((root / 'campaign_001/arms').glob('*/WORK.jsonl')):
            raise ValueError('Compiler baseline is missing after campaign work started')
        write_json(path, current)
        baseline = current
    return dict(compiler_baseline_path=str(path), compiler_cache_directories=current['directories'],
                compiler_baseline_allocated_bytes=baseline['allocated_bytes'],
                compiler_current_allocated_bytes=current['allocated_bytes'],
                compiler_growth_bytes=max(0, current['allocated_bytes'] - baseline['allocated_bytes']))


def admit(root, plan, *, receipt_path=None):
    root = _inside(root, root, directory=True)
    compiler = compiler_growth(root)
    present = allocated_bytes(_inside(path, root) for path in boundary_files(root))
    compact_paths = [root]
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Campaign storage contains a symlink: ' + str(path))
        if path.is_dir() or path.suffix not in {'.pt', '.safetensors'}:
            compact_paths.append(path)
    compact = allocated_bytes(compact_paths)
    required = (max(0, plan['tensor_peak_with_margin_bytes'] - present)
                + max(0, plan['compact_bytes'] - compact)
                + plan['compiler_bytes'] + plan['workspace_bytes'] + plan['reserve_bytes'])
    free = shutil.disk_usage(root).free
    passed = (free >= required and compact <= plan['compact_bytes']
              and compiler['compiler_growth_bytes'] <= plan['compiler_bytes'])
    result = dict(status='passed' if passed else 'failed',
                  free_bytes=free, required_additional_and_reserve_bytes=required,
                  validated_existing_tensor_bytes=present, compact_bytes=compact,
                  accounting='st_blocks*512; unique device/inode including directory metadata',
                  no_future_deletion_credit=True, plan=plan, **compiler)
    if receipt_path is not None:
        write_json(receipt_path, result)
    if result['status'] != 'passed':
        raise ValueError('K+ peak storage admission failed')
    return result


def check_growth(root, plan):
    return admit(root, plan)


def _retire(root, receipt_path, candidates, compact_paths, *, reason, retained):
    """Write the exact plan before unlinking any file; preserve partial failure."""
    started = time.perf_counter()
    root = _inside(root, root, directory=True)
    receipt_path = Path(receipt_path)
    if receipt_path.exists() or receipt_path.with_suffix('.plan.json').exists():
        raise ValueError('Retirement already attempted; inspect the preserved receipt')
    evidence = {}
    for value in sorted(set(map(str, compact_paths))):
        path = _inside(value, root)
        json.loads(path.read_text())
        evidence[str(path)] = identity(path)
    entries = []
    for path, kind in candidates:
        path = _inside(path, root)
        stat = path.stat()
        if stat.st_nlink != 1:
            raise ValueError('Only the sole obsolete tensor link may be retired')
        entries.append(dict(path=str(path), kind=kind, bytes=stat.st_size,
                            inode=stat.st_ino, device=stat.st_dev, mtime_ns=stat.st_mtime_ns))
    if len({item['path'] for item in entries}) != len(entries):
        raise ValueError('Duplicate retirement path')
    result = dict(status='planned', files=entries, compact_evidence=evidence,
                  retained=retained, reason=reason, removed=[],
                  free_before_bytes=shutil.disk_usage(root).free,
                  lost_capability='Deleted tensors are not archived; direct tensor replay is lost.')
    write_json(receipt_path.with_suffix('.plan.json'), result)
    try:
        for entry in entries:
            path = _inside(entry['path'], root)
            stat = path.stat()
            if (stat.st_size, stat.st_ino, stat.st_dev, stat.st_mtime_ns) != tuple(
                    entry[key] for key in ('bytes', 'inode', 'device', 'mtime_ns')):
                raise ValueError('Retirement candidate changed after validation')
            path.unlink()
            result['removed'].append(entry)
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['free_after_bytes'] = shutil.disk_usage(root).free
        result['removed_bytes'] = sum(item['bytes'] for item in result['removed'])
        result['wall_s'] = time.perf_counter() - started
        write_json(receipt_path, result)
    return result


def retire_redundant_features(record, prepared_receipt, root, event_root):
    """Validate the replacement and token binding before deleting raw copies."""
    import torch
    started = time.perf_counter()
    event_root = _inside(event_root, root, directory=True)
    prepared = prepared_receipt
    feature = _inside(prepared['feature_path'], root)
    expected = event_root / 'loop/features' / (record['record_id'] + '.pt')
    capture = _inside(record['metadata']['capture']['feature_path'], root)
    terminal = _inside(prepared['terminal_feature']['tensor_path'], root)
    if feature != expected or terminal != expected.with_suffix('.terminal.pt'):
        raise ValueError('Unexpected replacement feature path')
    if (_json(feature.with_suffix('.json'), root) != prepared
            or _json(terminal.with_suffix('.json'), root) != prepared['terminal_feature']):
        raise ValueError('Prepared/terminal receipt changed')
    capture_record = _json(capture.with_suffix('.json'), root)
    combined = record['prompt_ids'] + record['response_ids']
    if (prepared['status'] != 'prepared' or prepared['record_id'] != record['record_id']
            or prepared['input_ids'] != combined or not prepared['response_only_loss']
            or capture_record['prompt_ids'] != record['prompt_ids']
            or capture_record['response_ids'] != record['response_ids']):
        raise ValueError('Durable feedback identity differs')
    packet = torch.load(feature, map_location='cpu', weights_only=False)
    native = torch.load(capture, map_location='cpu', weights_only=False)
    last = torch.load(terminal, map_location='cpu', weights_only=False)
    mask = torch.zeros((1, len(combined)), dtype=torch.float32)
    mask[:, len(record['prompt_ids']):] = 1
    if (packet['record_id'] != record['record_id']
            or packet['input_ids'].tolist() != [combined]
            or packet['prompt_length'] != len(record['prompt_ids'])
            or not torch.equal(packet['loss_mask'], mask)
            or packet['hidden_states'].shape[1] != len(combined)
            or packet['target_last_hidden_states'].shape[1] != len(combined)):
        raise ValueError('Durable feature packet differs from native response')
    if (native['input_ids'] != combined[:-1]
            or native['target_argmax_ids'] != record['response_ids']
            or last['input_ids'] != combined or last['positions'] != [len(combined) - 1]
            or last['final_feature_logits_used_for_labels'] is not False):
        raise ValueError('Decoded source features differ from the committed token binding')
    shapes = {}
    for key in ('hidden_states', 'target_last_hidden_states'):
        prepared_rows, native_rows, terminal_row = packet[key], native[key], last[key]
        if (prepared_rows.dtype != torch.bfloat16 or native_rows.dtype != torch.bfloat16
                or terminal_row.dtype != torch.bfloat16
                or not torch.equal(prepared_rows[0, :-1], native_rows)
                or not torch.equal(prepared_rows[0, -1:], terminal_row)):
            raise ValueError('Prepared tensor rows do not exactly preserve native and terminal features: ' + key)
        shapes[key] = list(prepared_rows.shape)
    equivalence = event_root / 'preparation_retirement' / (record['record_id'] + '.equivalence.json')
    write_json(equivalence, dict(status='passed', record_id=record['record_id'],
        native_capture=str(capture), terminal_feature=str(terminal), prepared_packet=str(feature),
        exact_decoded_rows=True, tensor_shapes=shapes, dtype='torch.bfloat16',
        validation_wall_s=time.perf_counter() - started, additional_model_forwards=0))
    del packet, native, last, prepared_rows, native_rows, terminal_row
    return _retire(root, event_root / 'preparation_retirement' / (record['record_id'] + '.json'),
                   [(capture, 'native_capture'), (terminal, 'terminal_feature')],
                   [capture.with_suffix('.json'), feature.with_suffix('.json'),
                    terminal.with_suffix('.json'), equivalence],
                   reason='Durable packet exactly preserves native/terminal tensor rows and token/mask binding.',
                   retained={'prepared_packet': str(feature)})


def retire_event(root, event_root, records, receipt, previous_export,
                 previous_boundary, current_boundary):
    event_root = _inside(event_root, root, directory=True)
    training = receipt['training']
    event_id = training['event_id']
    saved = _json(event_root / 'loop/events' / (event_id + '.json'), root)
    buffer = _json(event_root / 'loop/feedback_buffer/state.json', root)
    marker = _json(event_root / 'PUBLICATION_CHECK.json', root)
    if (saved != receipt or training['status'] != 'completed'
            or current_boundary['status'] != 'passed'
            or marker['status'] != 'passed' or marker['all_equal'] is not True
            or marker['event_id'] != event_id
            or marker['checkpoint_path'] != training['checkpoint_path']
            or buffer['records'] != records
            or len(buffer['events']) != 1 or buffer['events'][0]['status'] != 'committed'
            or buffer['events'][0]['receipt'] != receipt):
        raise ValueError('Retirement requires a complete validated published event')
    for field in ('state', 'weights'):
        if identity(current_boundary[field]) != current_boundary[field + '_identity']:
            raise ValueError('Current retained boundary identity changed')
    candidates = []
    if [p['record_id'] for p in receipt['preparation']] != [r['record_id'] for r in records]:
        raise ValueError('Prepared response order differs')
    for record, prepared in zip(records, receipt['preparation']):
        prior = _json(event_root / 'preparation_retirement' / (record['record_id'] + '.json'), root)
        if prior['status'] != 'completed' or prior['retained']['prepared_packet'] != prepared['feature_path']:
            raise ValueError('Incremental preparation retirement is incomplete')
        candidates.append((_inside(prepared['feature_path'], root), 'prepared_feature'))
    if previous_export is not None:
        previous = _inside(previous_export, root, directory=True)
        if (previous_boundary is None or current_boundary['event'] != previous_boundary['event'] + 1
                or Path(previous_boundary['weights']) != previous / 'model.safetensors'
                or any(r['checkpoint_id'] != str(previous) for r in records)):
            raise ValueError('Predecessor is not the checkpoint serving this event')
        candidates.extend([(Path(previous_boundary['state']), 'previous_state'),
                           (Path(previous_boundary['weights']), 'previous_export')])
    evidence = list(event_root.rglob('*.json'))
    evidence.append(Path(training['checkpoint_path']).parent / 'RESULT.json')
    return _retire(root, event_root / 'RETIREMENT.json', candidates, evidence,
                   reason='Complete successor state/export validated and native publication passed.',
                   retained={'current_state': current_boundary['state'],
                             'current_weights': current_boundary['weights'],
                             'initial_state': str(Path(root) / 'campaign_001/arms/KPLUS/trainer/initial_state.pt')})
