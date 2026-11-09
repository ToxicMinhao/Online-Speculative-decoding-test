"""Complete the stopped qualification with separately discarded startup.

Copy the latest failed ledger intact, reuse successful GSM checks, then run only
the five unexecuted Adam applications and eight serving checks. This module does
not launch the pilot, even if all correctness and mechanism-activity gates pass.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

import torch

from experiments.ok_coding.storage import identity
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from online_feedback.storage import write_json
from .qualify import (CASES, REFERENCE_STARTUP, check_serving, cleanup,
    recover_warmed_ordinary_reference, restore_origin, run_case, runtime_state, warmup)
from .resume_qualification import (POLICY, correction_activity, formula_only,
                                  retained_reference, reuse_gsm_cases)
from .training import FunctionalGeometryTrainer
from .work import LIMITS, QualificationWork


PRIOR_COUNTS = {kind + '_' + status: count
    for kind, count in [('adam', 3), ('ordinary_forward', 8), ('ordinary_backward', 8),
                        ('additional_forward', 34), ('additional_backward', 20)]
    for status in ('attempted', 'completed')}
CUMULATIVE_LIMITS = dict(adam=8, ordinary_forward=18, ordinary_backward=18,
                          additional_forward=67, additional_backward=36,
                          serving_generation=8)


def remaining_work_plan():
    """Maximum calls, including candidate scoring when a norm guard permits it."""
    return dict(adam=5, ordinary_forward=10, ordinary_backward=10,
        additional_forward=33, additional_backward=16, serving_generation=8,
        additional_parts={
            'gsm_discarded_startup': dict(forward=2, backward=2),
            'gsm_reference_recapture': dict(forward=2, backward=2),
            'math_startup': dict(forward=2, backward=2),
            'gsm_scalar_filter': dict(forward=6, backward=2),
            'math_zero_filter': dict(forward=4, backward=2),
            'math_fsg_filter': dict(forward=6, backward=2),
            'math_scalar_filter': dict(forward=6, backward=2),
            'math_same_layout_oracle': dict(forward=5, backward=2)},
        repeated_completed_adam_steps=0, repeated_gsm_oracle_work=0,
        repeated_resource_fixture_work=0, main_pilot_work=0)


def admit_resume(plan, root):
    """Validate the new bounded continuation before copying its prior ledger."""
    repair = plan.get('startup_repair', {})
    if (repair.get('cumulative_limits') != CUMULATIVE_LIMITS
            or repair.get('prior_counts') != PRIOR_COUNTS
            or repair.get('remaining_work') != remaining_work_plan()
            or repair.get('reference_startup') != REFERENCE_STARTUP
            or plan['revised_qualification']['policy'] != POLICY):
        raise ValueError('Require the sealed startup repair and explicit cumulative work contract')
    prior = Path(repair['previous_root'])
    previous = json.loads((prior / 'qualification/QUALIFICATION.json').read_text())
    if (previous['status'] != 'failed' or previous['work'] != PRIOR_COUNTS
            or previous['main_pilot_started'] is not False
            or '/SCALAR/NATIVE_GRADIENTS.json' not in previous['error']):
        raise ValueError('Require the latest retained pre-Adam SCALAR qualification failure')
    previous_plan = json.loads((prior / 'sealed/PLAN.json').read_text())
    for key in ('binding', 'geometry', 'serving', 'numerical_recipe'):
        if plan[key] != previous_plan[key]:
            raise ValueError('The scientific qualification contract changed: ' + key)
    old_work = prior / 'QUALIFICATION_WORK.jsonl'
    if identity(old_work) != repair.get('previous_ledger_identity'):
        raise ValueError('Sealed latest qualification ledger identity differs')
    if QualificationWork(old_work).summary() != PRIOR_COUNTS:
        raise ValueError('Latest failed qualification accounting differs')
    source = old_work.read_bytes()
    destination = Path(root) / 'QUALIFICATION_WORK.jsonl'
    with destination.open('xb') as stream:
        stream.write(source)
        stream.flush()
        os.fsync(stream.fileno())
    if destination.read_bytes() != source:
        raise ValueError('Qualification work prefix was not preserved exactly')
    receipt = dict(status='passed', prior_counts=PRIOR_COUNTS,
        previous_root=str(prior), prior_work_identity=identity(old_work),
        copied_work_identity=identity(destination), original_limits=LIMITS,
        continuation_limits=CUMULATIVE_LIMITS, remaining=remaining_work_plan(),
        separately_budgeted_zero_adam_diagnostics_excluded=True,
        failed_scalar_work_preserved=True, completed_adam_steps_repeated=0)
    write_json(Path(root) / 'RESUME_ADMISSION.json', receipt)
    return prior, QualificationWork(destination, limits=CUMULATIVE_LIMITS)


def reuse_gsm_oracles(plan, output):
    """Use passed formula/resource evidence, with historical costs labeled."""
    previous = Path(plan['startup_repair']['previous_root'])
    expected = previous / 'qualification/gsm8k/revised_oracles'
    source = Path(plan['startup_repair']['retained_oracles'])
    if source != expected:
        raise ValueError('GSM oracle evidence must come from the latest failed qualification')
    receipt = json.loads((source / 'RESULT.json').read_text())
    state = json.loads((source / 'EXACT_STATE_RESTORATION.json').read_text())
    formula, resource = receipt['formula'], receipt['resource']
    tolerances = dict(atol=POLICY['atol'], rtol=POLICY['rtol'],
                      maximum_relative_l2=POLICY['maximum_relative_l2'])
    if (receipt['status'] != 'passed' or not state['exact_equal']
            or formula['status'] != 'passed' or resource['status'] != 'passed'
            or formula['numerical_protocol'] != 'independent_same_native_layout'
            or formula['tolerances'] != tolerances
            or formula['historical_cross_layout_gate_passed'] is not False
            or formula['native_vjp']['status'] != 'passed'
            or formula['chunks'] != [8, 1]
            or formula['fixture']['fixture_anchors'] != 9):
        raise ValueError('Retained GSM formula/resource checks are not qualified')
    reused = dict(status='passed', reused=True, source=str(source),
        evidence={name: identity(source / name)
                  for name in ('RESULT.json', 'EXACT_STATE_RESTORATION.json')},
        formula=formula, resource=resource,
        cost_and_work_fields='historical qualification_003 measurements; zero new oracle/resource work',
        new_work=dict(additional_forward=0, additional_backward=0, adam=0),
        synthetic_resource_is_representative_correction_evidence=False)
    write_json(Path(output) / 'REUSED_GSM_ORACLES.json', reused)
    return reused


def run(root):
    root = Path(root)
    plan = json.loads((root / 'sealed/PLAN.json').read_text())
    preparation = json.loads((root / 'qualification_feedback/PREPARATION.json').read_text())
    if preparation['status'] != 'completed' or set(plan['datasets']) != {'gsm8k', 'mathqa'}:
        raise ValueError('Both original prepared qualification datasets are required')
    prior, work = admit_resume(plan, root)
    output = root / 'qualification'
    output.mkdir(exist_ok=False)
    summary = dict(status='running', execution_commit=plan['execution_source']['commit'],
        policy=POLICY, limits=CUMULATIVE_LIMITS, original_limits=LIMITS,
        prior_work=PRIOR_COUNTS, datasets={}, main_pilot_started=False, automatic_retry=False,
        separately_budgeted_diagnostics=plan['revised_qualification']['separate_diagnostics'])
    trainer = None
    failure = None
    started = time.perf_counter()
    try:
        pin_retained_graph_runtime(output)
        for dataset in ('gsm8k', 'mathqa'):
            item = plan['datasets'][dataset]
            directory = output / dataset
            directory.mkdir()
            torch.cuda.reset_peak_memory_stats()
            rows = preparation['datasets'][dataset]
            if len(rows) != 2:
                raise ValueError('Each qualification window requires the original two responses')
            for row in rows:
                if identity(row['feature_path']) != row['feature_identity']:
                    raise ValueError('Retained qualification feature identity changed')
            features = [row['feature_path'] for row in rows]
            trainer = FunctionalGeometryTrainer(item['configs']['KPLUS'], directory / 'trainer',
                arm='KPLUS', dataset=dataset, binding=plan['binding'],
                original_config_text=Path(plan['original_config']).read_text(),
                origin_state=item['origin_state'], origin_export=item['origin_export'])
            baseline = runtime_state(trainer.bundle.model)
            cases = []
            reference = None
            summary['datasets'][dataset] = dict(cases=cases)
            remaining = CASES
            if dataset == 'gsm8k':
                # This validates existing evidence without repeating its forwards.
                summary['datasets'][dataset]['oracles'] = reuse_gsm_oracles(plan, directory)
                reference, ordinary = recover_warmed_ordinary_reference(
                    trainer, features, retained_reference(plan), work, baseline, directory,
                    startup_contract=plan['startup_repair']['reference_startup'])
                old_cases = Path(plan['revised_qualification']['previous_root'])
                zero, fsg = reuse_gsm_cases(plan, old_cases, ordinary)
                fsg['completion_requires_replacement_oracle'] = False
                fsg['replacement_oracle_evidence'] = str(prior / 'qualification/gsm8k/revised_oracles')
                cases.extend((ordinary, zero, fsg))
                remaining = CASES[-1:]
            else:
                work.context = dict(dataset=dataset, case='discarded_startup')
                warmup(trainer, features, work, baseline, directory)
            for label, strength in remaining:
                capture, case = run_case(trainer, features, label, strength, work,
                    baseline, directory / label, reference, oracle_runner=formula_only)
                if reference is None:
                    reference = capture
                cases.append(case)
                summary['work'] = work.summary()
                write_json(output / 'QUALIFICATION.json', summary)
            restore_origin(trainer, baseline, directory)
            trainer.close()
            trainer = None
            del reference, baseline, capture
            gc.collect()
            torch.cuda.empty_cache()
            summary['datasets'][dataset]['serving'] = check_serving(
                plan, dataset, cases, rows[0], work, directory / 'serving')
        activity = correction_activity(summary['datasets'])
        summary.update(status='passed', correction_activity=activity,
            numerical_qualification='passed revised contract; original cross-layout failures retained',
            pilot_mechanism_active=activity['nontrivial_accepted_fsg_cases'] > 0,
            next_scope=('pilot_requires_storage_and_complete_integration'
                if activity['nontrivial_accepted_fsg_cases'] else
                'stop_before_pilot_all_representative_corrections_fallback'))
    except BaseException as error:
        failure = error
        summary.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            cleanup(trainer.close, failure, summary, 'trainer close')
        summary.update(wall_s=time.perf_counter() - started, work=work.summary())
        cleanup(lambda: write_json(output / 'QUALIFICATION.json', summary),
                failure, summary, 'qualification receipt')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    result = run(parser.parse_args().root)
    print(json.dumps({key: result[key] for key in ('status', 'wall_s', 'work')}))
