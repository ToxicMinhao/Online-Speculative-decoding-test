"""Finish the retained qualification under the reviewed same-layout contract.

Completed Adam applications are reused, not repeated. The old cross-layout
failure stays a sensitivity result; only the new independent formula check can
satisfy the revised numerical requirement.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback

import torch

from experiments.ok_coding.storage import identity
from experiments.prefix_guard.qualify import pin_retained_graph_runtime, persist_comparison
from online_feedback.functional_geometry import FunctionalGeometry
from online_feedback.storage import write_json
from .diagnose_bf16_consequences import load_window
from .qualify import (CASES, check_serving, cleanup, physical_work, recover_ordinary_reference,
    restore_origin, run_case, runtime_state, scientific_state, warmup)
from .training import FunctionalGeometryTrainer
from .work import LIMITS, QualificationWork

POLICY = {
    'name': 'independent_same_layout_bf16_v1',
    'production_head_chunk_blocks': 8,
    'native_fixture_anchor_count': 9,
    'atol': 1e-5, 'rtol': .016, 'maximum_relative_l2': .016,
    'historical_cross_layout': 'failed_original_criterion_retained_as_sensitivity',
    'norm_guard': 'unchanged', 'risk_guard': 'unchanged',
}
PRIOR_COUNTS = {kind+'_'+status: count
    for kind, count in [('adam', 3), ('ordinary_forward', 6), ('ordinary_backward', 6),
                        ('additional_forward', 18), ('additional_backward', 12)]
    for status in ('attempted', 'completed')}


def admit_resume(plan, root):
    """Authenticate the stopped case and preserve every earlier work entry."""
    prior = Path(plan['revised_qualification']['previous_root'])
    original = json.loads((prior/'qualification/QUALIFICATION.json').read_text())
    if (original['status'] != 'failed' or original['work'] != PRIOR_COUNTS
            or 'Native chunked/unchunked oracle failed' not in original['error']):
        raise ValueError('Require the preserved three-step native-oracle failure')
    if plan['revised_qualification']['policy'] != POLICY:
        raise ValueError('The reviewed numerical policy differs')
    old_work = prior/'QUALIFICATION_WORK.jsonl'
    if QualificationWork(old_work).summary() != PRIOR_COUNTS:
        raise ValueError('Historical qualification accounting differs')
    target = Path(root)/'QUALIFICATION_WORK.jsonl'
    with target.open('xb') as stream:
        stream.write(old_work.read_bytes())
        stream.flush()
        import os
        os.fsync(stream.fileno())
    receipt = dict(status='passed', prior_counts=PRIOR_COUNTS,
        prior_work_identity=identity(old_work), copied_work_identity=identity(target),
        remaining_adam_steps=5, remaining_auxiliary_backwards=20,
        old_failure_reclassified_not_passed=True)
    write_json(Path(root)/'RESUME_ADMISSION.json', receipt)
    return prior, QualificationWork(target)


def retained_reference(plan):
    files = plan['repair']['retained_reference']
    return dict(payload=Path(files['disposable_state']['path']),
        case=Path(files['failed_result']['path']).parent,
        rows=json.loads(Path(files['presentations']['path']).read_text()),
        observations=Path(files['response_observations']['path']))


def reuse_gsm_cases(plan, prior, ordinary):
    """Complete the interrupted FSG case from its authenticated native fallback.

    Its Adam step and filtering already ran. The later sensitivity diagnostic
    reproduced its receipt exactly and independently observed the same fallback.
    We reuse the ordinary export because the represented endpoint is identical.
    """
    zero = json.loads((prior/'qualification/gsm8k/FSG_ZERO/RESULT.json').read_text())
    neutrality = json.loads((prior/'qualification/gsm8k/FSG_ZERO/EXACT_NEUTRALITY.json').read_text())
    step = json.loads(Path(plan['revised_qualification']['retained_fsg_step']).read_text())
    consequence = Path(plan['revised_qualification']['consequence_root'])
    reproduction = json.loads((consequence/'NATIVE_REPRODUCTION.json').read_text())
    result = json.loads((consequence/'RESULT.json').read_text())
    values = json.loads((consequence/'CONSEQUENCES.json').read_text())
    if (zero['status'] != 'passed' or neutrality.get('exact_equal') is not True
            or step['decision']['reason'] != 'rounded_norm_exceeds_native'
            or step['committed']['identical_native'] is not True
            or reproduction['status'] != 'passed' or reproduction['differences']
            or result['status'] != 'completed' or not result['restoration']['exact']):
        raise ValueError('Retained GSM neutrality/native fallback is not established')
    # The new diagnostic must have used the exact same historical scientific row.
    if (reproduction['expected'] != reproduction['observed']
            or any(step[key] != value for key, value in reproduction['expected'].items())
            or not values['committed_endpoint_comparison']['left_identical_native']
            or not values['decisions']['committed_endpoints_identical']):
        raise ValueError('The retained production receipt was not exactly reproduced')
    zero = dict(zero, optimizer_steps=0, reused_optimizer_steps=1,
        presentations=0, reused_presentations=2,
        historical_case=str(prior/'qualification/gsm8k/FSG_ZERO'),
        cost_and_work_fields='historical qualification_002 measurements, not fresh work')
    fsg = dict(status='passed', case='FSG', export=ordinary['export'],
        optimizer_steps=0, reused_optimizer_steps=1, reused_presentations=2,
        endpoint_source='saved exact ordinary endpoint; established FSG native fallback',
        historical_case=str(prior/'qualification/gsm8k/FSG'),
        historical_failure='cross-layout criterion remains failed sensitivity evidence',
        historical_decision=step['decision'], reused_step=step,
        completion_requires_replacement_oracle=True)
    return zero, fsg


def restored_gsm_oracles(trainer, features, retained, plan, work, root):
    """Run missing checks from saved endpoints, without another optimizer step."""
    from .qualification_checks import run_same_layout_check, run_resource_checks
    before = scientific_state(trainer)
    hook = FunctionalGeometry(trainer, work_callback=work.callback)
    step = json.loads(Path(plan['revised_qualification']['retained_fsg_step']).read_text())
    named = dict(trainer.bundle.draft_model.named_parameters())
    names = step['active_parameter_names']
    hook.active_parameters = tuple(named[name] for name in names)
    hook.theta0 = tuple(trainer.origin['draft_state_dict'][name].to(named[name].device).clone()
                        for name in names)
    hook.theta_native = tuple(named[name].detach().clone() for name in names)
    hook.responses, metadata = load_window(trainer, features,
        plan['repair']['retained_reference']['presentations']['path'])
    hook.vocab_size = trainer.bundle.model.lm_head.weight.shape[0]
    failure = None
    receipt = dict(status='running', reused_step=True, additional_adam_steps=0, fixture=metadata)
    try:
        formula, reference = run_same_layout_check(hook, root/'same_layout')
        resource = run_resource_checks(hook, root/'resource', reference,
                                       sequence_length=trainer.max_length)
        del reference
        persist_comparison(root/'EXACT_STATE_RESTORATION.json', scientific_state(trainer), before)
        receipt.update(status='passed', formula=formula, resource=resource)
        return receipt
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        receipt['physical_work'] = hook.report()
        cleanup(hook.close, failure, receipt, 'retained oracle hook close')
        cleanup(lambda: write_json(root/'RESULT.json', receipt), failure, receipt, 'retained oracle receipt')


def formula_only(hook, output):
    from .qualification_checks import run_same_layout_check
    receipt, reference = run_same_layout_check(hook, output)
    del reference
    return receipt


def correction_activity(datasets):
    rows = []
    for dataset, data in datasets.items():
        for case in data['cases']:
            if case['case'] != 'FSG':
                continue
            steps = ([case['reused_step']] if 'reused_step' in case
                     else case['hook']['steps'])
            for row in steps:
                nontrivial = row['decision']['accepted'] and not row['committed']['identical_native']
                rows.append(dict(dataset=dataset, case='FSG', decision=row['decision'],
                    nontrivial_accepted=bool(nontrivial), raw_norm=row['raw_fsg']['norm'],
                    native_norm=row['raw_fsg']['native_norm']))
    return dict(cases=rows, nontrivial_accepted_fsg_cases=sum(r['nontrivial_accepted'] for r in rows),
                all_fsg_fallback=bool(rows) and all(not r['nontrivial_accepted'] for r in rows))


def run(root):
    root = Path(root)
    output = root/'qualification'; output.mkdir(exist_ok=False)
    plan = json.loads((root/'sealed/PLAN.json').read_text())
    preparation = json.loads((root/'qualification_feedback/PREPARATION.json').read_text())
    prior, work = admit_resume(plan, root)
    summary = dict(status='running', execution_commit=plan['execution_source']['commit'],
        policy=POLICY, limits=LIMITS, datasets={}, main_pilot_started=False,
        separate_diagnostics_not_hidden_in_original_budget=plan['revised_qualification']['separate_diagnostics'])
    trainer = None; failure = None; started = time.perf_counter()
    try:
        pin_retained_graph_runtime(output)
        for dataset, item in plan['datasets'].items():
            directory = output/dataset; directory.mkdir()
            torch.cuda.reset_peak_memory_stats()
            features = [row['feature_path'] for row in preparation['datasets'][dataset]]
            for row in preparation['datasets'][dataset]:
                if identity(row['feature_path']) != row['feature_identity']:
                    raise ValueError('Prepared feedback identity changed')
            trainer = FunctionalGeometryTrainer(item['configs']['KPLUS'], directory/'trainer',
                arm='KPLUS', dataset=dataset, binding=plan['binding'],
                original_config_text=Path(plan['original_config']).read_text(),
                origin_state=item['origin_state'], origin_export=item['origin_export'])
            baseline = runtime_state(trainer.bundle.model)
            cases = []; reference = None
            summary['datasets'][dataset] = dict(cases=cases)
            remaining = CASES
            if dataset == 'gsm8k':
                retained = retained_reference(plan)
                # This two-backward native replay both warms the retained graph
                # and recaptures raw ordinary gradients; it performs no Adam step.
                reference, ordinary = recover_ordinary_reference(trainer, features,
                    retained, work, baseline, directory/'KPLUS')
                zero, fsg = reuse_gsm_cases(plan, prior, ordinary)
                cases.extend((ordinary, zero, fsg))
                summary['datasets'][dataset]['oracles'] = restored_gsm_oracles(
                    trainer, features, retained, plan, work, directory/'revised_oracles')
                fsg['completion_requires_replacement_oracle'] = False
                remaining = CASES[-1:]
            else:
                work.context = dict(dataset=dataset, case='startup')
                warmup(trainer, features, work, baseline, directory)
            for label, strength in remaining:
                capture, case = run_case(trainer, features, label, strength, work,
                    baseline, directory/label, reference, oracle_runner=formula_only)
                if reference is None:
                    reference = capture
                cases.append(case)
                summary['work'] = work.summary()
                write_json(output/'QUALIFICATION.json', summary)
            restore_origin(trainer, baseline, directory)
            trainer.close(); trainer = None
            del reference, baseline
            if 'capture' in locals(): del capture
            gc.collect(); torch.cuda.empty_cache()
            summary['datasets'][dataset]['serving'] = check_serving(plan, dataset,
                cases, preparation['datasets'][dataset][0], work, directory/'serving')
        activity = correction_activity(summary['datasets'])
        summary.update(status='passed', correction_activity=activity,
            numerical_qualification='passed_revised_contract; original cross-layout test remains failed',
            pilot_mechanism_active=activity['nontrivial_accepted_fsg_cases'] > 0,
            next_scope=('pilot_requires_storage_and_complete_integration' if activity['nontrivial_accepted_fsg_cases']
                        else 'stop_before_pilot_all_representative_corrections_fallback'))
    except BaseException as error:
        failure = error
        summary.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            cleanup(trainer.close, failure, summary, 'trainer close')
        summary.update(wall_s=time.perf_counter()-started, work=work.summary())
        cleanup(lambda: write_json(output/'QUALIFICATION.json', summary), failure, summary, 'qualification receipt')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({key:result[key] for key in ('status','wall_s','work')}))
