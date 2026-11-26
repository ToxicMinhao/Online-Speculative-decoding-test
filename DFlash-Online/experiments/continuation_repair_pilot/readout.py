"""Matched TAIL readouts at immutable PAIR1 origins; no regression is fitted."""

import argparse
import os
from pathlib import Path
import time

from experiments.ok_coding.storage import identity
from experiments.repair_conditional_kl_pilot.resources import read
from experiments.continuation_value_diagnostic.metrics import paired_block, legacy_pairs
from experiments.continuation_value_diagnostic.native import ReadOnlyEndpoint, compare_repeated
from online_feedback.storage import write_json
from .analysis import mechanism_summary


def authenticated_retained(plan):
    """Consume sealed historical readouts, not newly generated replacement data."""
    root = Path(plan['retained_diagnostic_root'])
    expected = plan['retained_diagnostic_files']

    def bound(path):
        path = Path(path)
        if str(path) not in expected or identity(path) != expected[str(path)]:
            raise ValueError('Unauthenticated retained diagnostic file: ' + str(path))
        return read(path)

    contract = bound(root / 'CONTRACT.json')
    if identity(root / 'CONTRACT.json') != plan['retained_diagnostic_contract_identity']:
        raise ValueError('The sealed PR64 diagnostic schedule changed')
    if bound(root / 'STATUS.json')['status'] != 'completed':
        raise ValueError('Retained diagnostic must be complete')
    if (len(contract['features']) != 128
            or sum(len(row['anchors']) for row in contract['features']) != 8192):
        raise ValueError('Require all128 sealed PR64 response geometries')
    parent, base = {}, {}
    for binding in contract['features']:
        directory = root / 'primary' / f'gpu{binding["physical_shard"]}'
        for endpoint, selected in (('parent', parent), ('child', base)):
            receipt = bound(directory / endpoint / f'{binding["ordinal"]:03d}.json')
            if (receipt['status'] != 'passed' or receipt['record_id'] != binding['record_id']
                    or receipt['anchors'] != binding['anchors']
                    or len(receipt['blocks']) != len(binding['anchors'])):
                raise ValueError('Retained readout identity or coverage differs')
            selected[binding['record_id']] = receipt
    legacy_parent, legacy_base = bound(root / 'legacy/parent.json'), bound(root / 'legacy/child.json')
    return contract, parent, base, legacy_parent, legacy_base


def paired_rows(parent, endpoint, bindings, *, cohort):
    rows = []
    for binding in bindings:
        before, after = parent[binding['record_id']], endpoint[binding['record_id']]
        if (before['record_id'] != after['record_id'] or before['anchors'] != after['anchors']
                or len(before['blocks']) != len(after['blocks'])
                or before['anchors'] != binding['anchors']):
            raise ValueError('Endpoint changed fixed-parent block bindings')
        for anchor, origin, current in zip(binding['anchors'], before['blocks'], after['blocks'], strict=True):
            rows.append(paired_block(origin, current, record_id=binding['record_id'], anchor=anchor, cohort=cohort))
    return rows


def run_readout(plan, endpoint_status=None):
    from .launch import boundary
    from .serving import endpoint_receipts, require_qualification

    root = Path(plan['root']) / 'mechanism'
    root.mkdir(parents=True, exist_ok=False)
    status_path = root / 'STATUS.json'
    started = time.perf_counter()
    status = dict(status='running', planned_response_forwards=137,
                  completed_primary_responses=0, completed_retention_responses=0,
                  qualification_response_forwards=1, automatic_retry=False,
                  backwards=0, adam_steps=0, teacher_forwards=0, generations=0)
    write_json(status_path, status)
    adapter = None
    try:
        boundary(plan)
        require_qualification(plan)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != plan['training_gpu']:
            raise ValueError('Readout must use the physical GPU sealed for this phase')
        receipts = endpoint_receipts(plan)
        tail_status = receipts['TAIL']
        if endpoint_status is not None and endpoint_status != tail_status:
            raise ValueError('Caller TAIL endpoint differs from the completed receipt')
        contract, parent, base, legacy_parent, legacy_base = authenticated_retained(plan)
        if receipts['BASE']['checkpoint_identity'] != contract['endpoints']['child']['identity']:
            raise ValueError('Retained mechanism BASE and serving BASE differ')
        adapter = ReadOnlyEndpoint.load(plan['original_config'], tail_status['checkpoint'],
            tail_status['checkpoint_identity'], endpoint='TAIL', allowed_endpoints=('TAIL',))
        write_json(root / 'LOADING.json', adapter.loading)
        current = {}
        for ordinal, binding in enumerate(contract['features']):
            boundary(plan)
            actual = adapter.read_response(binding, root / 'primary' / f'{ordinal:03d}.json')
            current[binding['record_id']] = actual
            status['completed_primary_responses'] += 1
            write_json(status_path, status)
            if ordinal == 0:
                repeated = adapter.read_response(binding, root / 'qualification_repeat.json')
                comparison = compare_repeated(actual, repeated)
                write_json(root / 'ADAPTER_QUALIFICATION.json', dict(comparison,
                    exact_export_load=adapter.loading['mapping'],
                    scope='Same native adapter with explicit TAIL identity; repeated first64-anchor geometry'))
                if comparison['status'] != 'passed':
                    raise ValueError('TAIL readout is not deterministic at fixed geometry')
        bindings = contract['legacy_binding']['sets']['parent_training']
        if len(bindings) != 8 or any(len(b['anchors']) != 8 for b in bindings):
            raise ValueError('Retain the separate eight-response/eight-anchor parent-training witness')
        retention = {}
        for ordinal, binding in enumerate(bindings):
            boundary(plan)
            retention[binding['record_id']] = adapter.read_response(binding, root / 'parent_training' / f'{ordinal:03d}.json')
            status['completed_retention_responses'] += 1
            write_json(status_path, status)
        base_rows = paired_rows(parent, base, contract['features'], cohort='primary')
        tail_rows = paired_rows(parent, current, contract['features'], cohort='primary')
        retained = legacy_pairs(legacy_parent, legacy_base, 'parent_training')
        # Reuse the original parent's compact block, including its frozen K.
        origin = {(r['record_id'], r['anchor']): r['parent'] for r in retained}
        retention_rows = []
        for binding in bindings:
            now = retention[binding['record_id']]
            for anchor, current_block in zip(binding['anchors'], now['blocks'], strict=True):
                retention_rows.append(paired_block(origin[binding['record_id'], anchor], current_block,
                    record_id=binding['record_id'], anchor=anchor, cohort='parent_training'))
        write_json(root / 'BLOCKS_PRIVATE.json', dict(BASE=base_rows, TAIL=tail_rows,
                   retention_BASE=retained, retention_TAIL=retention_rows))
        result = dict(primary=mechanism_summary(base_rows, tail_rows, cohort='primary'),
                      parent_training=mechanism_summary(retained, retention_rows, cohort='parent_training'))
        write_json(root / 'SUMMARY.json', result)
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        try:
            if adapter is not None:
                status['work'] = dict(adapter.work)
                closed = adapter.close()
                status['endpoint_state'] = closed
                write_json(root / 'ENDPOINT_STATE.json', closed)
        except BaseException as error:
            status.update(status='failed', closure_error=repr(error))
            raise
        finally:
            status['wall_s'] = time.perf_counter() - started
            status['unattempted_response_forwards'] = 137 - status.get('work', {}).get('response_reads_attempted', 0)
            write_json(status_path, status)
    if status['work']['response_reads_completed'] != 137:
        raise ValueError('Mechanism readout work did not complete its137 forwards')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    run_readout(read(parser.parse_args().plan))
