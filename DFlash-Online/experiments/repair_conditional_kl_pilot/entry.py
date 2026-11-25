"""Bounded native qualification with durable results before validation/cleanup."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from experiments.opencode_native_comparison.contract import digest
from experiments.opencode_native_comparison.evaluation import measure, label_for, SESSION
from experiments.opencode_native_comparison.serving_gate import _compare
from experiments.opencode_native_comparison.validity import validate_measurement
from .resources import read, boundary, Ledger, occupied_card
from .campaign import create_backend, check_publication, run_arm


def qualify(plan, arm):
    from .qualification import run_training_qualification
    from experiments.kplus_full_gsm8k.storage import retire_redundant_features
    from experiments.full_k_gsm8k.campaign import online_record_check

    root = Path(plan['root'])
    base = root / 'qualification_inputs' / arm
    base.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(base / 'WORK.json')
    inputs = read(plan['manifest'])
    saved = read(root / 'sealed/QUALIFICATION_REFERENCES.json')
    canonicals = read(root / 'sealed/CANONICALS.json')
    status = dict(
        status='running', arm=arm, phase='collection', execution_commit=plan['execution_commit']
    )
    backend = None
    started = time.perf_counter()
    try:
        write_json(base / 'ADMISSION.json', boundary(plan))
        backend = create_backend(plan, base / 'native', 'A' if arm == 'BASE' else 'B')
        status['backend_loading_s'] = backend.loading_s
        records = []
        paths = []
        for request in inputs['qualification_requests']:
            path = base / 'records' / (request['request_id'] + '.json')
            record = ledger.call(
                {'feedback_generation': 1},
                'disposable_feedback',
                backend.generate,
                request['prompt_ids'],
                max_new_tokens=2048,
                request_id=request['request_id'],
                session_id='repair_pair_qualification_' + arm,
                request_index=request['request_index'],
                purpose='serving',
                save_to=path,
            )
            write_json(path, record)
            online_record_check(record, request, plan['draft'], capture=True)
            _compare(record, saved[request['request_id']], compare_counts=True)
            feature = base / 'loop/features' / (record['record_id'] + '.pt')
            receipt = ledger.call(
                {'teacher_preparation': 1},
                'disposable_teacher',
                backend.prepare_feedback,
                record,
                feature,
            )
            retire_redundant_features(record, receipt, root, base)
            records.append(record)
            paths.append(feature)
        status['phase'] = 'training'
        write_json(base / 'STATUS.json', status)
        trained = run_training_qualification(plan, arm, feature_paths=paths, ledger=ledger)
        status['training'] = trained
        export = trained['export_path']
        status['phase'] = 'publication'
        write_json(base / 'STATUS.json', status)
        request = inputs['evaluation_requests'][0]
        first = None
        for repeat in range(2):
            if repeat:
                check_publication(backend, plan['draft'], base / 'FROZEN_RELOAD.json')
            check_publication(backend, export, base / f'EXPORT_RELOAD_{repeat}.json')
            slot = dict(
                activation=repeat,
                repeat=repeat,
                group=0,
                method=arm,
                phase='qualification',
                request_ordinal=request['request_index'],
            )
            label = label_for(slot)
            path = base / 'records' / (label + '.json')
            record = ledger.call(
                {'evaluation_generation': 1},
                label,
                backend.generate,
                request['prompt_ids'],
                max_new_tokens=2048,
                request_id=label,
                session_id=SESSION,
                request_index=request['request_index'],
                purpose='evaluation',
                save_to=path,
            )
            write_json(path, record)
            row, errors = measure(record, request, export, canonicals[request['request_id']], slot)
            if first is not None:
                _compare(record, first, compare_counts=True)
            write_json(
                base / f'EVALUATION_{repeat}.json', dict(row=row, errors=errors, valid=not errors)
            )
            if errors:
                raise ValueError('Packaged native evaluation failed: ' + repr(errors))
            first = record
        status.update(
            status='passed',
            phase='completed',
            canonical_and_repeat_checks=True,
            serving_card=plan['devices']['A' if arm == 'BASE' else 'B'],
            training_card=plan['devices']['B' if arm == 'BASE' else 'A'],
        )
    except BaseException as error:
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(base / 'STATUS.json', status)
        if backend is not None:
            try:
                if not backend.failed:
                    write_json(base / 'PHYSICAL_WORK.json', backend._call('work_receipt'))
            finally:
                backend.close()
        status['wall_s'] = time.perf_counter() - started
        write_json(base / 'STATUS.json', status)
    return status


def main(plan_path, phase, arm):
    from .resources import verify_inputs

    plan = read(plan_path)
    root = Path(plan['root'])
    verify_inputs(plan)
    if plan['binding'] != digest({k: v for k, v in plan.items() if k != 'binding'}):
        raise ValueError('Sealed plan identity changed')
    result = dict(status='running', phase=phase, arm=arm, execution_commit=plan['execution_commit'])
    path = root / 'phases' / (phase + '_' + arm + '.json')
    if path.exists():
        raise FileExistsError('No automatic repeat of an attempted phase')
    write_json(path, result)
    try:
        with ExitStack() as stack:
            for device in plan['devices'].values():
                stack.enter_context(occupied_card(root, phase + '_' + arm + '_' + device, device))
            result['result'] = qualify(plan, arm) if phase == 'qualify' else run_arm(plan, arm)
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(path, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--phase', choices=['qualify', 'train'], required=True)
    parser.add_argument('--arm', choices=['BASE', 'PAIR'], required=True)
    args = parser.parse_args()
    main(args.plan, args.phase, args.arm)
