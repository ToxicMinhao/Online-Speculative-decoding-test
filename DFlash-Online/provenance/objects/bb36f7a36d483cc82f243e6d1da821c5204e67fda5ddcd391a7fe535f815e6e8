"""Retire one verified duplicate disposable qualification container, on CPU."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, '/data/sources/minhao/DFlash-Online')
from experiments.ok_continuation.resume import differences


def identity(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


def retire_duplicate(root):
    root = Path(root)
    trial = Path('/data/sources/minhao/runs/dflash_prefix_ranking_gpu_diagnostic_20261026/trials/O2')
    source = trial / 'trainer/training_state.pt'
    retained = trial / 'final.pt'
    # O2/initial.pt is a required startup input and is never touched.
    state = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    final = torch.load(retained, map_location='cpu', weights_only=False, mmap=True)
    pairs = {
        'draft': (state['draft_state_dict'], final['draft']),
        'optimizer': (state['backend']['optimizer'], final['optimizer']),
        'rng': (state['rng'], final['rng']),
        'global_step': (state['global_step'], final['global_step']),
        'presentations': (state['presentations'], final['presentations']),
    }
    errors = [error for name, pair in pairs.items() for error in differences(*pair, prefix=name)]
    if errors:
        raise ValueError('Disposable recovery is not redundant: ' + repr(errors[:10]))
    metadata = {key: value for key, value in state.items() if key not in ('draft_state_dict', 'backend')}
    metadata['retained_backend_metadata'] = {
        key: value for key, value in state['backend'].items() if key != 'optimizer'
    }
    metadata_path = root / 'RETAINED_O2_RECOVERY_METADATA.pt'
    torch.save(metadata, metadata_path)
    restored_metadata = torch.load(metadata_path, map_location='cpu', weights_only=False)
    if differences(metadata, restored_metadata):
        raise ValueError('Retained metadata did not round-trip exactly')
    before = shutil.disk_usage(root).free
    receipt = dict(status='verified_before_removal', timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        before_free_bytes=before, removed_path=str(source), original_identity=identity(source),
        retained_exact_tensors_optimizer_rng=str(retained), retained_identity=identity(retained),
        retained_state_metadata=str(metadata_path), metadata_identity=identity(metadata_path),
        verified_equal_fields=list(pairs),
        reason='Completed disposable qualification recovery duplicates retained final tensors, Adam and RNG exactly; all other state metadata retained separately.',
        capability_change='Original redundant serialized path retired. Equivalent recovery contents remain in final.pt plus metadata; not an archive of the original file.',
        production_checkpoints_retired=0, required_initial_state_untouched=str(trial/'initial.pt'))
    receipt_path = root / 'STORAGE_MAINTENANCE.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    source.unlink()
    receipt.update(status='completed', after_free_bytes=shutil.disk_usage(root).free)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    retire_duplicate(sys.argv[1])
