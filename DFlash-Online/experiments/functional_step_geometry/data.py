"""Continue the exact PR32 input order without looking at model outcomes."""
from copy import deepcopy
import json
from pathlib import Path

from experiments.ok_coding.storage import identity
from experiments.role_loss_pilot import data as original
from online_feedback.storage import write_json
from .contract import DATASETS, SCHEMA

PREFIX_COUNT, NEW_COUNT = 512, 256


def _input_signature(row):
    return (row['source_id'], row['prompt_ids'], row['normalized_stem'],
            row['rendered_prompt_sha256'], row['prompt_ids_sha256'])


def validate_continuation(full_stream, retained_training, panels, *, dataset):
    """Check the original prefix before taking the next prescribed 256 rows.

    Duplicates or overlap fail preparation. They never trigger outcome-dependent
    replacement or shrink the requested continuation.
    """
    if (dataset not in DATASETS or len(retained_training) != PREFIX_COUNT
            or len(full_stream) < PREFIX_COUNT + NEW_COUNT):
        raise ValueError('Require all 512 retained inputs and the reconstructed first 768')
    if [_input_signature(r) for r in full_stream[:PREFIX_COUNT]] != [
            _input_signature(r) for r in retained_training]:
        raise ValueError('Reconstructed training order does not reproduce the exact PR32 prefix')
    previous = original.overlap_index(retained_training)
    evaluation = original.overlap_index(panels)
    selected = []
    for ordinal, row in enumerate(full_stream[PREFIX_COUNT:PREFIX_COUNT+NEW_COUNT], PREFIX_COUNT):
        if row['dataset'] != dataset:
            raise ValueError('Cross-dataset feedback input')
        reasons = original.overlap_reasons(row, previous) + original.overlap_reasons(row, evaluation)
        if reasons:
            raise ValueError(f'Prescribed continuation input overlaps prior/evaluation inputs: {row["source_id"]}: {reasons}')
        if (not row['prompt_ids'] or row.get('collect_feedback') is not True
                or row.get('max_new_tokens') != 2048 or len(row['prompt_ids']) + 2048 > 2272):
            raise ValueError('Continuation preprocessing or context budget changed')
        expected_event = f'event_{ordinal//128+1:03d}'
        if row.get('event_id') != expected_event:
            raise ValueError('The original stream event binding differs')
        selected.append(deepcopy(row))
        for key, value in original.overlap_keys(row).items():
            previous[key].add(value)
    return selected


def prepare_continuation_inputs(original_gsm_manifest, retained_data_root, math_archive,
                                tokenizer, output):
    """Seal both next windows and unchanged panels, with no model execution.

    ``tokenizer`` is the caller's authenticated pinned tokenizer. The GSM8K
    source is the original full prepared manifest, not a newly shuffled dataset.
    MathQA is reconstructed with the original PR32 selection/filter implementation.
    """
    root = Path(retained_data_root)
    inherited, panels = {}, {}
    for dataset in DATASETS:
        inherited[dataset] = json.loads((root/dataset/'TRAINING.json').read_text())
        panels[dataset] = json.loads((root/dataset/'PANEL.json').read_text())
    all_panels = [row for dataset in DATASETS for row in panels[dataset]['requests']]
    gsm_source = json.loads(Path(original_gsm_manifest).read_text())
    gsm = []
    stop_ids = inherited['gsm8k']['train_requests'][0]['stop_token_ids']
    for item in gsm_source['train_requests'][:PREFIX_COUNT+NEW_COUNT]:
        row = deepcopy(item)
        row.update(original.prompt_fields(row, 'gsm8k', tokenizer), stop_token_ids=stop_ids)
        gsm.append(row)
    math_inputs, math_source = original.read_mathqa_train(math_archive)
    math, exclusions = original.select_mathqa_training(math_inputs, all_panels, tokenizer,
        inherited['mathqa']['train_requests'][0]['stop_token_ids'], count=PREFIX_COUNT+NEW_COUNT)
    selected = {}
    for dataset, stream in (('gsm8k', gsm), ('mathqa', math)):
        selected[dataset] = validate_continuation(stream, inherited[dataset]['train_requests'],
                                                 all_panels, dataset=dataset)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    for dataset in DATASETS:
        directory = destination/dataset
        directory.mkdir()
        write_json(directory/'TRAINING.json', dict(schema=SCHEMA, dataset=dataset,
            train_requests=selected[dataset], start_ordinal=512, stop_ordinal=768,
            source_answers_used=False, selection_uses_outputs=False,
            earlier_independent_project_exposure_allowed=True,
            exposure_note=('GSM8K train was used in earlier independent project campaigns; '
                           'freshness is within the resumed trajectory.' if dataset=='gsm8k'
                           else 'Earlier independent project exposure is not an exclusion criterion.'),
            source=inherited[dataset]['source'] if dataset=='gsm8k' else math_source))
        # Copy the complete retained panel/canonical records without reselecting
        # requests, rewriting tokens, or making another reference generation.
        for name in ('PANEL.json', 'CANONICALS.json'):
            (directory/name).write_bytes((root/dataset/name).read_bytes())
    write_json(destination/'MATHQA_EXCLUSIONS.json', exclusions)
    receipt = dict(schema=SCHEMA, status='sealed_inputs', model_executions=0,
        training_requests=512, events=[5, 6], unchanged_evaluation_requests=512,
        retained_prefix_checked=512, source_answers_used=False,
        input_identities={str(p): identity(p) for p in (Path(original_gsm_manifest), Path(math_archive),
            *[root/d/n for d in DATASETS for n in ('TRAINING.json','PANEL.json','CANONICALS.json')])})
    write_json(destination/'PREPARATION.json', receipt)
    return receipt


def seal_shared_packets(feature_paths, canonical_paths, *, dataset, event, output):
    """Bind shared bytes once; every arm verifies this same inventory before use."""
    if dataset not in DATASETS or event not in (5, 6) or len(feature_paths) != len(canonical_paths):
        raise ValueError('Require corresponding new canonical and feature packets')
    if len(feature_paths) != 128 or len(set(map(str, feature_paths))) != 128:
        raise ValueError('A shared event contains exactly 128 distinct feature files')
    rows = []
    for feature, canonical in zip(feature_paths, canonical_paths):
        record = json.loads(Path(canonical).read_text())
        rows.append(dict(record_id=record['record_id'],
            feature=dict(path=str(Path(feature).resolve()), **identity(feature)),
            canonical=dict(path=str(Path(canonical).resolve()), **identity(canonical))))
    if len({row['record_id'] for row in rows}) != 128:
        raise ValueError('Shared canonical packet identities are not distinct')
    inventory = dict(schema=SCHEMA, dataset=dataset, event=event,
        producer=f'KPLUS{event-1}', records=rows, shared_bytes_required=True)
    destination = Path(output)
    if destination.exists():
        raise ValueError('Shared packet inventory is immutable')
    write_json(destination, inventory)
    return inventory


def verify_shared_packets(inventory):
    for row in inventory['records']:
        for name in ('feature', 'canonical'):
            expected = row[name]
            if identity(expected['path']) != {key:expected[key] for key in ('bytes','sha256')}:
                raise ValueError('Shared feedback packet changed: '+expected['path'])
    return dict(status='passed', packets=len(inventory['records']), byte_identical=True)
