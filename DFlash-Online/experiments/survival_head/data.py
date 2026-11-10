"""Input-only panel selection and byte-authenticated shared response packets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from .contract import DATASET, EVENT, PANEL_SIZE, SOURCE_PANEL_SIZE, SCHEMA


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _request_ids(rows):
    ids = [row['request_id'] for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('Require distinct nonempty UTF-8 request IDs')
    return ids


def select_panel(source_panel, feedback_requests, *, output=None):
    """Seal the first128 by SHA256(survival-v1| + request_id), before outputs.

    The 256-row development panel is already historically used. Selection is
    input-only; it does not turn this subset into fresh confirmation evidence.
    """
    requests = source_panel['requests']
    source_ids, feedback_ids = _request_ids(requests), _request_ids(feedback_requests)
    if (source_panel.get('dataset') != DATASET or len(requests) != SOURCE_PANEL_SIZE
            or source_panel.get('previously_used_development_panel') is not True
            or source_panel.get('input_only_selection') is not True
            or [r['panel_ordinal'] for r in requests] != list(range(SOURCE_PANEL_SIZE))):
        raise ValueError('Require the unchanged existing 256-request GSM8K development panel')
    if len(feedback_ids) != 128:
        raise ValueError('Require all128 original event-five feedback identities')
    if set(source_ids) & set(feedback_ids):
        raise ValueError('Development and feedback requests must be disjoint')
    if any('response_ids' in r or 'finish_reason' in r for r in requests):
        raise ValueError('Panel selection consumes input requests only')
    rank = lambda request: (hashlib.sha256(('survival-v1|' + request['request_id']).encode('utf-8')).digest(), request['request_id'])
    selected = sorted(requests, key=rank)[:PANEL_SIZE]
    rows = []
    for ordinal, request in enumerate(selected):
        row = deepcopy(request)
        row.update(source_panel_ordinal=request['panel_ordinal'], panel_ordinal=ordinal,
                   selection_sha256=rank(request)[0].hex())
        rows.append(row)
    sealed = dict(schema=SCHEMA, dataset=DATASET, requests=rows,
        previously_used_development_panel=True, input_only_selection=True,
        selection_rule="sort SHA256(UTF8('survival-v1|' + request_id)); first128; request_id tie-break",
        source_panel_digest=digest(source_panel), source_request_count=SOURCE_PANEL_SIZE,
        feedback_request_ids=feedback_ids, feedback_disjoint=True,
        canonical_generations=0, model_work=0)
    sealed['binding'] = digest(sealed)
    if output is not None:
        if Path(output).exists():
            raise FileExistsError('Panel selection is immutable')
        write_json(output, sealed)
    return sealed


def verify_panel(panel, source_panel, feedback_requests):
    if panel != select_panel(source_panel, feedback_requests):
        raise ValueError('Sealed input-only panel or source identity changed')
    return dict(status='passed', requests=PANEL_SIZE, feedback_disjoint=True, model_work=0)


def feedback_requests(plan, dataset=DATASET, event=EVENT):
    if dataset != DATASET or event != EVENT:
        raise ValueError('Only original GSM8K event-five feedback is allowed')
    source = plan.get('training_manifest') or plan['datasets'][DATASET]['training_manifest']
    training = read(source)
    rows = [r for r in training['train_requests'] if r['event_id'] == 'event_005']
    if (training['dataset'] != DATASET or training['start_ordinal'] != 512
            or training['stop_ordinal'] != 768 or len(rows) != 128):
        raise ValueError('Require original PR32 continuation manifest and event five')
    _request_ids(rows)
    if any(r.get('dataset') != DATASET or r.get('max_new_tokens') != 2048
           or r.get('collect_feedback') is not True for r in rows):
        raise ValueError('Feedback request contract changed')
    return rows


def seal_shared_packets(feature_paths, canonical_paths, *, dataset=DATASET, event=EVENT,
                        output, expected_request_ids=None, provenance=None):
    if (dataset != DATASET or event != EVENT or len(feature_paths) != 128
            or len(canonical_paths) != 128 or len(set(map(str, feature_paths))) != 128
            or len(set(map(str, canonical_paths))) != 128):
        raise ValueError('Require exactly128 distinct original paired shared packets')
    rows = []
    for feature, canonical in zip(feature_paths, canonical_paths):
        record = read(canonical)
        rows.append(dict(record_id=record['record_id'],
            feature=dict(path=str(Path(feature).resolve()), **identity(feature)),
            canonical=dict(path=str(Path(canonical).resolve()), **identity(canonical))))
    ids = [r['record_id'] for r in rows]
    if len(set(ids)) != 128 or (expected_request_ids is not None and ids != list(expected_request_ids)):
        raise ValueError('Shared packet identities/order differ from original feedback')
    inventory = dict(schema=SCHEMA, dataset=dataset, event=event, producer='KPLUS4',
        records=rows, shared_bytes_required=True, provenance=provenance,
        expected_request_ids=list(expected_request_ids) if expected_request_ids is not None else ids)
    if Path(output).exists():
        raise FileExistsError('A shared packet inventory is immutable')
    write_json(output, inventory)
    return inventory


def verify_shared_packets(inventory):
    rows = inventory.get('records', [])
    if (inventory.get('schema') != SCHEMA or inventory.get('dataset') != DATASET
            or inventory.get('event') != EVENT or inventory.get('producer') != 'KPLUS4'
            or len(rows) != 128 or len({r['record_id'] for r in rows}) != 128
            or [r['record_id'] for r in rows] != inventory.get('expected_request_ids')):
        raise ValueError('Shared survival packet contract changed')
    for row in rows:
        for name in ('feature', 'canonical'):
            expected = row[name]
            if identity(expected['path']) != {k: expected[k] for k in ('bytes', 'sha256')}:
                raise ValueError('Shared feedback packet changed: ' + expected['path'])
        if read(row['canonical']['path'])['record_id'] != row['record_id']:
            raise ValueError('Canonical record identity changed')
    return dict(status='passed', packets=128, byte_identical=True, model_work=0)
