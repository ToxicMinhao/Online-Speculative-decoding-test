"""Matched TAIL readouts at immutable PAIR1 origins; no regression is fitted."""

import argparse
from collections import Counter
import math
import shutil
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


def resume_completed_readouts(root, contract, tail_status, *, preserve=True):
    """Authenticate and preserve the one interrupted attempt, without model work."""
    previous = read(root / 'STATUS.json')
    closed = read(root / 'ENDPOINT_STATE.json')
    loading = read(root / 'LOADING.json')
    if (previous['status'] != 'failed' or not previous.get('error', '').startswith('FileNotFoundError(')
            or previous['completed_primary_responses'] != 73
            or previous['completed_retention_responses'] != 0
            or closed['status'] != 'passed' or not closed['parameters_unchanged']
            or not closed['modes_unchanged'] or closed['optimizer_created']):
        raise ValueError('Resume requires the identified storage-writer failure after73 primary readouts')
    if (loading['mapping']['export_identity'] != tail_status['checkpoint_identity']
            or not loading['mapping']['exact_copy'] or loading['mapping']['tensors'] != 58
            or closed['loading'] != loading or previous['work'] != closed['work']):
        raise ValueError('Saved readouts do not bind the exact unchanged TAIL export')
    # OSError.__repr__ omits filename: authenticate the retained full traceback
    # instead of demanding that absent information from the compact error field.
    failure_log = root.parent / 'readout.log'
    original_repair = read(root.parent / 'resume_001' / 'REPAIR.json')
    failure_identity = original_repair['preserved_failures']['readout.log']
    if identity(failure_log) != failure_identity:
        raise ValueError('The preserved original readout traceback changed')
    traceback_text = failure_log.read_text()
    original_run = Path(original_repair['original_plan']).parent.parent
    if ('FileNotFoundError: [Errno 2]' not in traceback_text
            or str(original_run / 'STORAGE.json.tmp') not in traceback_text):
        raise ValueError('The authenticated traceback does not identify the storage-writer failure')
    expected_paths = {root / 'primary' / f'{i:03d}.json' for i in range(73)}
    if set((root / 'primary').glob('*.json')) != expected_paths or list((root / 'parent_training').glob('*.json')):
        raise ValueError('Resume must not replace an attempted or unexpected response')
    current, work, hashes = {}, Counter(), {}
    for ordinal, binding in enumerate(contract['features'][:73]):
        path = root / 'primary' / f'{ordinal:03d}.json'
        record = read(path)
        validate_saved_readout(record, binding)
        current[binding['record_id']] = record
        work.update(record['work'])
        hashes[str(path.relative_to(root))] = identity(path)
    repeated_path = root / 'qualification_repeat.json'
    repeated = read(repeated_path)
    validate_saved_readout(repeated, contract['features'][0])
    comparison = compare_repeated(current[contract['features'][0]['record_id']], repeated)
    if comparison['status'] != 'passed' or read(root / 'ADAPTER_QUALIFICATION.json')['status'] != 'passed':
        raise ValueError('The completed adapter repeat qualification is not reusable')
    work.update(repeated['work'])
    hashes[str(repeated_path.relative_to(root))] = identity(repeated_path)
    if dict(work) != previous['work'] or work['response_reads_completed'] != 74:
        raise ValueError('Saved successful records do not reconcile all previous physical work')
    attempt = root / 'attempts' / '001'
    if attempt.exists():
        raise FileExistsError('The original readout attempt is already preserved')
    if preserve:
        attempt.mkdir(parents=True, exist_ok=False)
    for name in ('STATUS.json', 'ENDPOINT_STATE.json', 'LOADING.json', 'ADAPTER_QUALIFICATION.json'):
        if preserve:
            shutil.copy2(root / name, attempt / name)
        hashes[name] = identity(root / name)
    if preserve:
        shutil.copy2(failure_log, attempt / 'FAILURE.log')
    receipt = dict(status='passed', prior_completed_response_forwards=74,
        remaining_response_forwards=63, prior_work=dict(work), retained_file_identities=hashes,
        TAIL_checkpoint_identity=tail_status['checkpoint_identity'],
        original_failure_log_identity=failure_identity,
        failure_diagnosis='Authenticated full traceback, since compact OSError repr omits filename',
        original_failure_preserved=str(attempt / 'STATUS.json'),
        repeat_check=comparison, no_completed_forward_repeated=True,
        current_file_binding_scope='Fresh hashes of retained receipts at continuation; no historical per-response hash claim',
        preservation_written=preserve)
    if preserve:
        write_json(root / 'RESUME_BINDING.json', receipt)
    return current, previous, receipt


def validate_saved_readout(record, binding):
    """Accept only complete native receipts for their original fixed geometry."""
    n = len(binding['anchors'])
    expected = dict(response_reads_attempted=1, response_reads_completed=1,
        draft_forwards_attempted=1, draft_forwards_completed=1,
        head_calls_attempted=math.ceil(n / 8), head_calls_completed=math.ceil(n / 8),
        blocks_completed=n, backwards=0, optimizer_steps=0, teacher_forwards=0,
        feedback_generations=0, serving_generations=0)
    if (record['status'] != 'passed' or record['endpoint'] != 'TAIL'
            or record['record_id'] != binding['record_id'] or record['anchors'] != binding['anchors']
            or record['feature_identity'] != binding['feature_identity']
            or len(record['blocks']) != n or record['work'] != expected
            or record['rng_before'] != record['rng_after']
            or not all(record[key] for key in ('parameters_unchanged', 'gradients_unchanged',
                                               'runtime_restored', 'rng_preserved'))):
        raise ValueError('Saved native readout is incomplete or has changed bindings')


def run_readout(plan, endpoint_status=None, *, resume_unexecuted=False):
    from .launch import boundary
    from .serving import endpoint_receipts, require_qualification

    root = Path(plan['root']) / 'mechanism'
    if not resume_unexecuted:
        root.mkdir(parents=True, exist_ok=False)
    status_path = root / 'STATUS.json'
    started = time.perf_counter()
    status = dict(status='running', planned_response_forwards=137,
                  completed_primary_responses=0, completed_retention_responses=0,
                  qualification_response_forwards=1, automatic_retry=False,
                  backwards=0, adam_steps=0, teacher_forwards=0, generations=0)
    adapter = None
    prior_work = {}
    prior_wall_s = 0.
    attempt_root = None
    current = {}
    attempt_started = False
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
        if resume_unexecuted:
            current, previous, receipt = resume_completed_readouts(root, contract, tail_status)
            prior_work = receipt['prior_work']
            prior_wall_s = previous['wall_s']
            status.update(completed_primary_responses=73, resumed_unexecuted=True,
                prior_attempt_response_forwards=74, current_attempt_planned_response_forwards=63,
                reused_qualification_response_forwards=1, qualification_response_forwards=0)
            attempt_root = root / 'attempts' / '002'
            attempt_root.mkdir(exist_ok=False)
        status.update(execution_commit=plan.get('execution_commit'),
                      training_execution_commit=plan.get('training_execution_commit', plan.get('execution_commit')))
        attempt_started = True
        write_json(status_path, status)
        adapter = ReadOnlyEndpoint.load(plan['original_config'], tail_status['checkpoint'],
            tail_status['checkpoint_identity'], endpoint='TAIL', allowed_endpoints=('TAIL',))
        write_json(root / 'LOADING.json', adapter.loading)
        if attempt_root is not None:
            write_json(attempt_root / 'LOADING.json', adapter.loading)
        for ordinal, binding in enumerate(contract['features']):
            if binding['record_id'] in current:
                continue
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
                status['attempt_work'] = dict(adapter.work)
                cumulative = Counter(prior_work)
                cumulative.update(adapter.work)
                # Keep explicit zero-work categories in the complete receipt.
                status['work'] = dict(cumulative)
                closed = adapter.close()
                if attempt_root is not None:
                    write_json(attempt_root / 'ENDPOINT_STATE.json', closed)
                    closed = dict(closed, work=status['work'], attempt_work=status['attempt_work'],
                        prior_attempt_work=prior_work, cumulative_attempts=2,
                        state_evidence='Exact export load and unchanged endpoint separately checked in both attempts')
                status['endpoint_state'] = closed
                write_json(root / 'ENDPOINT_STATE.json', closed)
        except BaseException as error:
            status.update(status='failed', closure_error=repr(error))
            raise
        finally:
            status['attempt_wall_s'] = time.perf_counter() - started
            status['wall_s'] = prior_wall_s + status['attempt_wall_s']
            status['unattempted_response_forwards'] = 137 - status.get('work', prior_work).get('response_reads_attempted', 0)
            if attempt_started or not resume_unexecuted:
                write_json(status_path, status)
                if attempt_root is not None:
                    write_json(attempt_root / 'STATUS.json', status)
    if status['work']['response_reads_completed'] != 137:
        raise ValueError('Mechanism readout work did not complete its137 forwards')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--resume-unexecuted', action='store_true')
    args = parser.parse_args()
    run_readout(read(args.plan), resume_unexecuted=args.resume_unexecuted)
