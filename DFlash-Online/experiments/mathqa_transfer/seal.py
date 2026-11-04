"""Seal the input-only panel, authenticated exports, source and analysis once."""
import argparse
import json
from pathlib import Path
import shutil

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from .contract import ANALYSIS, BUDGET, METHODS, SCHEMA, SERVING, final_schedule, validate_plan
from .data import read_source, select_panel
from .preflight import capacity_plan


def build(root, inputs, source_manifest):
    root, inputs = Path(root), Path(inputs)
    source = json.loads(Path(source_manifest).read_text())
    if source['dirty'] or len(source['commit']) != 40:
        raise ValueError('Publish clean experiment source before sealing')
    for path, expected in source['files'].items():
        if identity(path) != expected:
            raise ValueError('Deployed source differs from published commit: ' + path)
    panel = json.loads((inputs / 'PANEL.json').read_text())
    history = json.loads((inputs / 'HISTORY_INVENTORY.json').read_text())
    audit = json.loads((inputs / 'EXPORT_AUDIT.json').read_text())
    source_rows, archive = read_source(inputs / 'MathQA.zip')
    if (panel['status'] != 'prepared' or panel['required_count'] != 1024
            or len(panel['requests']) != 1024 or history['unreadable_files']
            or panel['source']['sha256'] != archive['sha256']):
        raise ValueError('Input selection is incomplete, unreadable or bound to another source')
    # Decode the exact source fields again, never an answer or result. Verify
    # that the prepared requests are the prespecified first eligible identities.
    expected_order = panel['eligible_order'][:1024]
    if [row['source_row'] for row in panel['requests']] != expected_order:
        raise ValueError('Prepared selected order changed')
    from transformers import AutoTokenizer
    from .data import render_inputs
    tokenizer = AutoTokenizer.from_pretrained(audit['target'], local_files_only=True)
    rendered = render_inputs(source_rows, tokenizer)
    overlap = {index: [] for index in history['overlapping_source_rows']}
    repeated = select_panel(rendered, overlap)
    if repeated['eligible_order'] != panel['eligible_order']:
        raise ValueError('Eligibility/hash-ranked selection differs on pinned runtime tokenizer')
    requests = []
    for row in panel['requests']:
        current = rendered[row['source_row']]
        if any(row[key] != current[key] for key in ('prompt_ids', 'prompt_ids_sha256', 'user_message',
                                                   'rendered_prompt', 'selection_key', 'question')):
            raise ValueError('Prepared exact prompt or token identity changed')
        requests.append(dict(row, panel_ordinal=row['ordinal'], original_index=row['source_row'],
            max_new_tokens=2048, stop_token_ids=audit['control_request']['stop_token_ids'], collect_feedback=False))
    root.mkdir(parents=True, exist_ok=False)
    sealed = root / 'sealed'
    sealed.mkdir()
    names = ('HANDOFF.txt', 'PANEL.json', 'HISTORY_INVENTORY.json', 'EXPORT_AUDIT.json',
             'MODEL_PIN.json', 'RUNTIME_PIN.json', 'STORAGE_SCHEMA_ESTIMATE.json', 'MathQA.zip',
             'REMOTE_MATHQA_INVENTORY_002.json', 'PREPARATION_REVIEW.json')
    for name in names:
        shutil.copy2(inputs / name, sealed / name)
    execution_source = {key: source[key] for key in ('commit', 'dirty', 'repository', 'branch')}
    plan = dict(schema=SCHEMA, methods=list(METHODS), serving=SERVING, budget=BUDGET,
        analysis=ANALYSIS, automatic_retry=False, execution_source=execution_source,
        target=audit['target'], draft=audit['draft'], worker_python=audit['worker_python'],
        checkpoints=audit['checkpoints'], evaluation_requests=requests,
        evaluation_schedule=final_schedule(requests), control_request=audit['control_request'],
        control_canonical=audit['control_canonical'], control_counts=audit['control_counts'],
        storage=capacity_plan(), dataset=panel['source'], wrapper=panel['wrapper'],
        input_only_selection=True, source_answers_used=False,
        output_root=str(root), native_gpu='0', run_mode='serving_only')
    validation = validate_plan(plan)
    write_json(sealed / 'PLAN.json', plan)
    write_json(sealed / 'SOURCE.json', source)
    write_json(sealed / 'CONTRACT_CHECK.json', validation)
    files = {**source['files'], **{str(path): identity(path) for path in sorted(sealed.iterdir())}}
    write_json(sealed / 'SEAL.json', dict(execution_source=execution_source, files=files,
        exact_file_sets={audit['exports'][arm]['path']: sorted(audit['exports'][arm]['files']) for arm in METHODS}))
    return dict(status='sealed', requests=len(requests), eligible=panel['eligible_count'],
                generations=BUDGET['total_generations'], execution_source=execution_source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--source-manifest', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.inputs, args.source_manifest)))
