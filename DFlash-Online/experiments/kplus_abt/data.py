"""Authenticated fixed manifests and immutable shared feedback packets."""
import json
from pathlib import Path

from experiments.functional_step_geometry.data import validate_continuation
from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from .contract import DATASETS, EVENTS, PANEL_SIZE, SCHEMA


def read(path):
    return json.loads(Path(path).read_text())


def seal_shared_packets(feature_paths, canonical_paths, *, dataset, event, output):
    """Bind actual shared bytes under the ABT schema, without a tensor load."""
    if (dataset not in DATASETS or event not in EVENTS or len(feature_paths) != 128
            or len(canonical_paths) != 128 or len(set(map(str,feature_paths))) != 128
            or len(set(map(str,canonical_paths))) != 128):
        raise ValueError('Require exactly 128 distinct paired shared packets')
    rows = []
    for feature, canonical in zip(feature_paths, canonical_paths):
        record = read(canonical)
        rows.append(dict(record_id=record['record_id'],
            feature=dict(path=str(Path(feature).resolve()),**identity(feature)),
            canonical=dict(path=str(Path(canonical).resolve()),**identity(canonical))))
    if len({row['record_id'] for row in rows}) != 128:
        raise ValueError('Repeated shared record identity')
    inventory = dict(schema=SCHEMA,dataset=dataset,event=event,producer=f'KPLUS{event-1}',
                     records=rows,shared_bytes_required=True)
    if Path(output).exists():
        raise ValueError('A shared packet inventory is immutable')
    write_json(output,inventory)
    return inventory


def verify_shared_packets(inventory):
    if (inventory.get('schema') != SCHEMA or inventory.get('dataset') not in DATASETS
            or inventory.get('event') not in EVENTS
            or inventory.get('producer') != f"KPLUS{inventory['event']-1}"
            or len(inventory.get('records',[])) != 128
            or len({r['record_id'] for r in inventory['records']}) != 128):
        raise ValueError('Shared ABT packet contract changed')
    for row in inventory['records']:
        for name in ('feature','canonical'):
            expected=row[name]
            if identity(expected['path']) != {k:expected[k] for k in ('bytes','sha256')}:
                raise ValueError('Shared feedback packet changed: '+expected['path'])
    return dict(status='passed',packets=128,byte_identical=True)


def feedback_requests(plan, dataset, event):
    if dataset not in DATASETS or event not in EVENTS:
        raise ValueError('Unknown ABT feedback window')
    manifest = read(plan['datasets'][dataset]['training_manifest'])
    if manifest['dataset'] != dataset or manifest['start_ordinal'] != 512 or manifest['stop_ordinal'] != 768:
        raise ValueError('Training manifest differs from PR32 continuation')
    rows = [r for r in manifest['train_requests'] if r['event_id'] == f'event_{event:03d}']
    if len(rows) != 128 or len({r['request_id'] for r in rows}) != 128:
        raise ValueError('Require 128 distinct prescribed feedback requests')
    if any(r.get('dataset') != dataset or r.get('max_new_tokens') != 2048
           or not r.get('collect_feedback') for r in rows):
        raise ValueError('Feedback request contract changed')
    return rows


def validate_inputs(training, panel, canonicals, *, dataset):
    if dataset not in DATASETS or training['dataset'] != dataset or panel['dataset'] != dataset:
        raise ValueError('Cross-dataset inputs')
    if training['start_ordinal'] != 512 or training['stop_ordinal'] != 768 or len(training['train_requests']) != 256:
        raise ValueError('Require exactly the two original continuation windows')
    requests = panel['requests']
    ids = [r['request_id'] for r in requests]
    if len(ids) != PANEL_SIZE or len(set(ids)) != PANEL_SIZE or set(canonicals) != set(ids):
        raise ValueError('Require original 256-request panel and every canonical')
    if [r['panel_ordinal'] for r in requests] != list(range(PANEL_SIZE)):
        raise ValueError('Panel order changed')
    if panel.get('previously_used_development_panel') is not True or panel.get('input_only_selection') is not True:
        raise ValueError('Historical development-panel provenance is required')
    train_ids = []
    for event in EVENTS:
        rows = [r for r in training['train_requests'] if r['event_id'] == f'event_{event:03d}']
        if len(rows) != 128 or len({r['request_id'] for r in rows}) != 128:
            raise ValueError('Incomplete continuation window')
        train_ids.extend(r['request_id'] for r in rows)
    if len(set(train_ids)) != 256:
        raise ValueError('Repeated feedback request across events')
    return dict(status='passed', training_requests=256, panel_requests=256,
                canonical_generations=0, model_work=0)
