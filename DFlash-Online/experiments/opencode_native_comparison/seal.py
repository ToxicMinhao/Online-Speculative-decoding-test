"""Seal bounded input-only CODE reservoirs before any model execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .contract import digest, validate_manifest
from .data import (DATASET, NORMALIZATION, exclude_inputs, length_summary,
                   manifest_requests, prepare_input, reservoir_plan, select_inputs,
                   training_length_bounds)


def file_identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=sha)


def validate_source_rows(rows, receipt):
    """Require the exact authorized prefix, without replacement or further scan."""
    plan = reservoir_plan(receipt['split_order'])
    expected = {(r['pool'], r['split'], i) for r in plan['ranges']
                for i in range(r['start'], min(r['stop'], receipt['split_rows'][r['split']]))}
    actual = []
    for row in rows:
        identity = row['source_identity']
        if identity['dataset'] != DATASET or identity['revision'] != receipt['revision']:
            raise ValueError('Source dataset or immutable revision changed')
        if identity['shard'] not in receipt['shards']:
            raise ValueError('Source shard is not authenticated')
        if any(k in row for k in ('output', 'answer', 'solution')):
            raise ValueError('The source projection must contain inputs only')
        actual.append((row['pool'], identity['split'], identity['raw_index']))
    if len(rows) > 50000 or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Source rows differ from the fixed bounded reservoirs')
    return plan


def seal(source_inputs, source_receipt, history, tokenizer_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('A sealed panel cannot be overwritten')
    receipt = json.loads(Path(source_receipt).read_text())
    prior = json.loads(Path(history).read_text())
    rows = [json.loads(line) for line in Path(source_inputs).open() if line.strip()]
    plan = validate_source_rows(rows, receipt)
    if file_identity(source_inputs) != receipt['input_projection_identity']:
        raise ValueError('Input projection differs from its acquisition receipt')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    rendered = [prepare_input(row, tokenizer) for row in rows]
    eligible, excluded = exclude_inputs(rendered, prior['fingerprints'])
    selected = select_inputs(eligible)
    manifest = manifest_requests(selected)
    manifest.update(schema='opencode_input_manifest_v1', dataset=DATASET,
        source_receipt=receipt, history_receipt=file_identity(history),
        normalization=NORMALIZATION, selection_seed=20261120,
        preparation_source_files={name:file_identity(Path(__file__).parent/name)
            for name in ('data.py', 'seal.py', 'contract.py')},
        tokenizer=dict(path=str(tokenizer_path), files={name:file_identity(Path(tokenizer_path)/name)
            for name in ('tokenizer.json', 'tokenizer_config.json')}),
        prompt_contract=dict(add_generation_prompt=True, enable_thinking=False,
            original_input_whitespace=True, maximum_prompt_tokens=2048,
            response_cap=2048, context_limit=4096,
            attention_mask='all prompt positions visible; no padding or truncation'))
    audit = dict(schema='opencode_input_audit_v1', source_plan=plan,
        inspected_rows=len(rows), source_fields_read=['input'], dataset_answers_used=False,
        reservoir_counts=dict(Counter(row['pool'] for row in rows)), excluded_rows=len(excluded),
        known_history_coverage=prior['coverage'], history_files=prior['files'],
        exclusions=excluded, exclusion_reason_counts=dict(Counter(reason for row in excluded for reason in row['reasons'])),
        eligible_counts=dict(Counter(row['pool'] for row in eligible)),
        selected_counts={role:len(group) for role, group in selected.items()},
        selected_lengths={role:length_summary(group) for role,group in selected.items()},
        lengths={pool:dict(before=length_summary(r for r in rendered if r['pool']==pool),
                           after=length_summary(r for r in eligible if r['pool']==pool))
                 for pool in ('train', 'evaluation')},
        training_length_bounds=training_length_bounds(manifest['train_requests']),
        source_files={str(p):file_identity(p) for p in (source_inputs, source_receipt, history)})
    output.mkdir(parents=True)
    for name, value in [('INPUTS.json', manifest), ('DATA_AUDIT.json', audit)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n')
    seal_record = dict(schema='opencode_input_seal_v1', validation=validate_manifest(manifest),
        files={name:file_identity(output/name) for name in ('INPUTS.json', 'DATA_AUDIT.json')},
        manifest_content_sha256=digest(manifest), all_outputs_unobserved=True)
    (output/'DATA_SEAL.json').write_text(json.dumps(seal_record, indent=2)+'\n')
    return seal_record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-inputs', 'source-receipt', 'history', 'tokenizer-path', 'output'):
        parser.add_argument('--'+name, required=True)
    print(json.dumps(seal(**vars(parser.parse_args())), indent=2))
