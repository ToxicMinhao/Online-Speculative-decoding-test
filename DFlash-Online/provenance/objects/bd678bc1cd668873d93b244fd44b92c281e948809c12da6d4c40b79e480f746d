"""Install only campaign additions and prepare both unsealed length policies."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

root = Path('/data/sources/minhao/runs/dflash_online_osd_rerun_20261023')
prep = root / 'preparation'
code = Path('/data/sources/minhao/DFlash-Online')
base = json.loads((prep / 'BASE_SOURCE.json').read_text())
for relative, expected in base.items():
    if hashlib.sha256((code / relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Installed core differs from qualified snapshot: ' + relative)
additions = json.loads((prep / 'SOURCE_ADDITIONS.json').read_text())
with tarfile.open(prep / 'source_additions.tar.gz') as archive:
    if set(archive.getnames()) != set(additions):
        raise RuntimeError('Source package differs from declared additions')
    for relative, expected in additions.items():
        contents = archive.extractfile(relative).read()
        if hashlib.sha256(contents).hexdigest() != expected:
            raise RuntimeError('Source package content differs: ' + relative)
        target = code / relative
        if target.exists() and target.read_bytes() != contents:
            raise RuntimeError('Preserve existing different file: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)

sys.path.insert(0, str(code))
from experiments.osd_persistent.data import prepare_manifest

archive = Path('/data/sources/minhao/EAGLE_SPEC/research/adaflash_reproduction_20260930/online_reference_001/osd/response_archive')
summaries = {}
fixed_context_limit = 4096
for policy in ('remaining', 'fixed2048'):
    training_limit = 2048 if policy == 'remaining' else fixed_context_limit
    manifest = prepare_manifest(
        archive, prep / 'TRAIN_REFERENCE.jsonl', prep / 'EVAL_REFERENCE.json',
        cap_policy=policy, train_context_limit=training_limit, serving_context_limit=4096)
    (prep / f'CANDIDATE_{policy}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = manifest['train_requests']
    fixed_context_limit = max(len(row['prompt_ids']) for row in rows) + 2048
    summaries[policy] = {
        'training_context_limit': training_limit, 'serving_context_limit': 4096,
        'train_requests': len(rows), 'heldout_requests': len(manifest['eval_requests']),
        'prompt_tokens_min': min(len(row['prompt_ids']) for row in rows),
        'prompt_tokens_max': max(len(row['prompt_ids']) for row in rows),
        'output_cap_min': min(row['max_new_tokens'] for row in rows),
        'output_cap_max': max(row['max_new_tokens'] for row in rows),
        'feedback_requests': sum(row['collect_feedback'] for row in rows),
    }
receipt = {'status': 'prepared_not_sealed', 'model_execution_started': False,
           'qualified_core_files_unchanged': len(base), 'source_additions': additions,
           'length_policies': summaries, 'pending': 'User selection of output/context limits'}
(prep / 'PREPARATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
