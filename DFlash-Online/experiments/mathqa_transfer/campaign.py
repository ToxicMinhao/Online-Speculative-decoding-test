"""Run only the three controls and the sealed 9,216 serving measurements."""
import argparse
import json
from pathlib import Path
import time

from online_feedback.storage import write_json
from online_feedback.sglang_backend import SGLangDFlashBackend
from .contract import BUDGET, validate_plan
from .evaluation import audit_stage, evaluate
from .preflight import check_space, timing_barrier, verify_exports


def actual_work(campaign):
    fields = ('attempted_generations', 'returned_generations', 'completed_valid_generations',
              'invalid_generations', 'pending_canonical_generations',
              'unknown_generation_attempts', 'unattempted_generations')
    totals = dict.fromkeys(fields, 0)
    for stage, planned in (('controls', 3), ('final', 9216)):
        status = campaign / 'evaluation' / stage / 'STATUS.json'
        if status.exists():
            values = json.loads(status.read_text())
            for key in fields:
                totals[key] += values[key]
        else:
            totals['unattempted_generations'] += planned
    return dict(**totals, planned_generations=9219, optimizer_steps=0,
        training_presentations=0, feedback_collections=0, teacher_preparations=0,
        extra_warmup_generations=0)


def run(root, *, backend_factory=SGLangDFlashBackend):
    root = Path(root)
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    validate_plan(plan)
    admission = json.loads((root / 'PREFLIGHT.json').read_text())
    if admission['status'] != 'passed':
        raise ValueError('Complete CPU identity/resource admission must precede native work')
    campaign = root / 'campaign_001'
    campaign.mkdir(exist_ok=False)
    started = time.perf_counter()
    state = dict(status='running', execution_source=plan['execution_source'], budget=BUDGET,
                 cleanup_errors=[], loading_s=0., close_s=0., controls=None, evaluation=None)
    backend = None
    write_json(campaign / 'STATUS.json', state)

    def boundary():
        return dict(storage=check_space(root),
                    quiescence=timing_barrier(backend.worker.pid))

    try:
        check_space(root, initial=True)
        timing_barrier()
        backend = backend_factory(plan['target'], plan['draft'], worker_python=plan['worker_python'],
            output_root=campaign / 'native', serving_gpu='0', context_limit=4096,
            block_size=16, reasoning='off')
        state['loading_s'] = backend.loading_s
        control = dict(canonical=plan['control_canonical'], counts=plan['control_counts'])
        state['controls'] = evaluate(backend, [plan['control_request']], plan['checkpoints'],
            campaign / 'evaluation/controls', control=control, boundary_check=boundary)
        checked = audit_stage(campaign / 'evaluation/controls', [plan['control_request']],
            plan['checkpoints'], evidence_root=root, control=control)
        write_json(campaign / 'CONTROL_AUDIT.json', checked)
        if checked['status'] != 'passed':
            raise RuntimeError('Historical controls did not qualify: ' + repr(checked['errors']))
        state['phase'] = 'timed_evaluation'
        write_json(campaign / 'STATUS.json', state)
        state['evaluation'] = evaluate(backend, plan['evaluation_requests'], plan['checkpoints'],
            campaign / 'evaluation/final', boundary_check=boundary)
        checked = audit_stage(campaign / 'evaluation/final', plan['evaluation_requests'],
            plan['checkpoints'], evidence_root=root)
        write_json(campaign / 'FINAL_AUDIT.json', checked)
        if checked['status'] != 'passed':
            raise RuntimeError('Final raw/native audit failed: ' + repr(checked['errors']))
        state['unchanged_exports'] = verify_exports(plan,
            json.loads((root / 'sealed/EXPORT_AUDIT.json').read_text()))
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        if backend is not None:
            closing = time.perf_counter()
            try:
                backend.close()
            except BaseException as error:
                state['cleanup_errors'].append(repr(error))
                state['status'] = 'failed'
            state['close_s'] = time.perf_counter() - closing
        for field, stage in (('controls', 'controls'), ('evaluation', 'final')):
            saved = campaign / 'evaluation' / stage / 'STATUS.json'
            if saved.exists():
                state[field] = json.loads(saved.read_text())
        state['actual_work'] = actual_work(campaign)
        state['campaign_wall_s'] = time.perf_counter() - started
        write_json(campaign / 'STATUS.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps(dict(status=result['status'], actual_work=result['actual_work'])))
    raise SystemExit(0 if result['status'] == 'completed' else 1)
