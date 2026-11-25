"""CPU-only final audit and compact retrieval of the completed native gate."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import torch

from experiments.repair_conditional_kl_pilot.resources import boundary, read
from experiments.repair_conditional_kl_pilot.training import compare_initial_witnesses
from experiments.opencode_native_comparison.serving_gate import _compare
from experiments.opencode_native_comparison.validity import validate_measurement
from online_feedback.storage import write_json

root = Path('/data/sources/minhao/runs/dflash_repair_conditional_kl_20261124/execution_001')
assert read(root / 'LAUNCH.json')['status'] == 'completed'
assert read(root / 'QUALIFICATION.json')['status'] == 'passed'
assert (root / 'EXIT_CODE.txt').read_text().strip() == '0'
plan = read(root / 'sealed/PLAN.json')
inputs = read(plan['manifest'])
references = read(root / 'sealed/QUALIFICATION_REFERENCES.json')
canonicals = read(root / 'sealed/CANONICALS.json')
witnesses = [torch.load(root / 'qualification' / arm / 'trainer/INITIAL_WITNESS.pt',
                       map_location='cpu', weights_only=False) for arm in ('BASE', 'PAIR')]
initial = compare_initial_witnesses(*witnesses)
checked = []
for arm in ('BASE', 'PAIR'):
    base = root / 'qualification_inputs' / arm
    assert read(base / 'STATUS.json')['status'] == 'passed'
    work = read(base / 'WORK.json')
    assert not work['pending'] and work['completed'] == work['attempted']
    for request in inputs['qualification_requests']:
        record = read(base / 'records' / (request['request_id'] + '.json'))
        validate_measurement(record, arm=arm)
        _compare(record, references[request['request_id']], compare_counts=True)
        checked.append(dict(arm=arm, phase='feedback', request_id=request['request_id'],
                            valid=True, verification_calls=record['metadata']['verification_calls']))
    previous = None
    for repeat in (0, 1):
        result = read(base / f'EVALUATION_{repeat}.json')
        assert result['valid'] and not result['errors']
        row = result['row']
        label = f'a{repeat:03d}_qualification_{row["request_ordinal"]:04d}_{arm}'
        record = read(base / 'records' / (label + '.json'))
        validate_measurement(record, arm=arm)
        _compare(record, canonicals[row['request_id']], compare_counts=False)
        if previous is not None:
            _compare(record, previous, compare_counts=True)
        previous = record
        checked.append(dict(arm=arm, phase='reload', request_id=row['request_id'],
                            repeat=repeat, valid=True, verification_calls=row['verification_calls']))

write_json(root / 'FINAL_CPU_AUDIT.json', dict(
    status='passed', execution_commit=plan['execution_commit'],
    exact_initialization_comparison=initial, raw_native_records=checked,
    new_model_forwards=0, new_backward_calls=0, new_optimizer_steps=0,
    cuda_initialized=torch.cuda.is_initialized(), resource=boundary(plan),
    free_bytes=shutil.disk_usage(root).free))

# Private compact receipt bundle; publication separately selects scalar fields.
selected = []
for path in root.rglob('*'):
    if not path.is_file() or path.suffix not in ('.json', '.jsonl', '.log', '.sh', '.txt', '.yaml'):
        continue
    relative = path.relative_to(root)
    if 'records' in relative.parts or 'native' in relative.parts:
        continue
    if path.name in ('INPUTS.json', 'CANONICALS.json', 'WARMUP_CANONICALS.json',
                     'QUALIFICATION_REFERENCES.json'):
        continue
    if path.stat().st_size > 4_000_000:
        continue
    selected.append(path)
manifest = {str(path.relative_to(root)): dict(bytes=path.stat().st_size,
    sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in selected}
write_json(root / 'RETRIEVAL_MANIFEST.json', manifest)
with tarfile.open(root / 'compact_qualification.tar.gz', 'w:gz') as archive:
    for path in selected + [root / 'RETRIEVAL_MANIFEST.json']:
        archive.add(path, arcname=str(path.relative_to(root)))
print(json.dumps(dict(status='passed', records=len(checked), compact_files=len(selected),
    archive_bytes=(root / 'compact_qualification.tar.gz').stat().st_size,
    archive_sha256=hashlib.sha256((root / 'compact_qualification.tar.gz').read_bytes()).hexdigest(),
    cuda_initialized=torch.cuda.is_initialized())))
