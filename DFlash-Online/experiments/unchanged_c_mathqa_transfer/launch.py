"""One admitted unchanged-C MathQA campaign; completed tasks are never repeated."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.bounded_prefix_allocation.launch import run_children
from experiments.persistent_prefix_withholding.launch import child_environment, check_compilers
from .runtime import initialize_work, read, clock_status, boundary, require_time_for

MODULE = 'experiments.unchanged_c_mathqa_transfer'


def task_job(plan, task, slot):
    return (task, slot, MODULE+'.task_runner', ['--task', task, '--gpu-uuid', plan['devices'][slot]])


def qualification_job(kind, slot):
    return (kind+'_'+slot, slot, MODULE+'.qualification_entry', ['--kind', kind, '--slot', slot])


def production_groups(plan):
    """Only C trains; its own event-6 feedback depends on its event-5 export."""
    return [[task_job(plan, task, 'A')] for task in ('C5', 'C_collect6', 'C6', 'final')]


def remaining_cost(groups, estimates):
    return (sum(max(estimates[job[0]] for job in group) for group in groups),
            sum(estimates[job[0]] for group in groups for job in group))


def serving_card_comparison(training_cross, left, right):
    """Compare identical disposable C weights; different weights have no count gate."""
    if left['status'] != 'passed' or right['status'] != 'passed':
        raise ValueError('Both same-card serving gates must pass')
    if training_cross['status'] != 'passed':
        return dict(status='not_comparable',reason='Disposable training endpoints differ across cards',
                    extra_generations=0)
    a,b=left['result']['qualification'],right['result']['qualification']
    if a['valid_generations'] != 4 or b['valid_generations'] != 4:
        raise ValueError('Incomplete same-card serving qualification')
    matched=a['signatures']==b['signatures']
    return dict(status='passed' if matched else 'failed',exact_native_counts_and_outputs=matched,
        same_card_checks_passed=True,extra_generations=0)


def require_cross_card_identity(cross_card):
    """Only numerical cross-card parity differences permit the qualified A fallback."""
    from .qualification import CASES, SCIENTIFIC_FIELDS
    permitted = {f"{case}: cross-card {field}" for case in CASES for field in SCIENTIFIC_FIELDS}
    permitted.add("serving: cross-card native counts")
    errors = cross_card.get("errors", [])
    if (cross_card.get("same_card_checks_passed") is not True
            or cross_card.get("status") not in ("passed", "failed")
            or (cross_card["status"] == "passed" and errors)
            or (cross_card["status"] == "failed" and not errors)
            or any(error not in permitted for error in errors)):
        raise ValueError("Cross-card provenance or same-card qualification failed; no fallback")
    return cross_card["status"] == "passed"


def qualify_resources(plan, cross_card):
    """Price remaining work from qualification and the exact canonical lengths."""
    import psutil
    require_cross_card_identity(cross_card)
    root=Path(plan['root'])
    training=[read(root/'qualification'/('training_'+s)/'STATUS.json') for s in ('A','B')]
    serving=[read(root/'qualification'/('serving_'+s)/'STATUS.json') for s in ('A','B')]
    if any(row['status']!='passed' for row in training+serving):
        raise ValueError('All same-card checks must pass before production')
    estimates=dict(plan['estimated_seconds'])
    pairs=[case['pair_wall_excluding_observers_s'] for row in training for case in row['result']['cases']]
    pair=max(pairs)
    for task in ('C5','C6'):
        measured=max(row['result']['loading_s']+row['result']['startup_s'] for row in training)
        measured+=128*pair+plan['checkpoint_boundary_allowance_s']
        estimates[task]=max(estimates[task],measured)
    serving_rows=[row for s in serving for row in s['result']['qualification']['rows']]
    per_token=max(r['request_latency_s']/max(1,r['output_tokens']) for r in serving_rows)
    references=read(plan['references'])
    estimates['final']=max(estimates['final'],9*sum(len(r['response_ids']) for r in references.values())*per_token+
        72*max(r['request_latency_s'] for r in serving_rows)+plan['final_activation_allowance_s'])
    common=read(plan['common_e5']['result'])
    estimates['C_collect6']=max(estimates['C_collect6'],
        common['loading_s']+common['generation_s']+common['preparation_s'])
    train_ram=max(row['sampled_process_tree_peak_rss_bytes'] for row in training)+12*2**30
    serve_ram=max(row['sampled_process_tree_peak_rss_bytes'] for row in serving)+4*2**30
    available=psutil.virtual_memory().available
    reserve=16*2**30
    if max(train_ram,serve_ram)+reserve>available:
        raise RuntimeError('Measured single-worker host memory no longer fits')
    groups=production_groups(plan)
    plan['estimated_seconds']=estimates
    elapsed,cards=remaining_cost(groups,estimates)
    return dict(status='passed',groups=groups,estimates=estimates,
        time=require_time_for(plan,elapsed,cards),storage=boundary(plan,compiler=True),
        measured_pair_compute_s=pair,measured_s_per_output_token=per_token,
        training_host_reservation_bytes=train_ram,serving_host_reservation_bytes=serve_ram,
        free_host_bytes=available,host_reserve_bytes=reserve,cross_card=cross_card,
        sequential_card_fallback=cross_card['status']!='passed')


def run(plan_path):
    plan_path=Path(plan_path);plan=read(plan_path);root=Path(plan['root'])
    if (root/'LAUNCH.json').exists():
        raise FileExistsError('Prior launch and original clock must be preserved')
    (root/'logs').mkdir(exist_ok=True)
    receipt=dict(status='cpu_preflight',execution_commit=plan['execution_commit'],
        started_epoch=time.time(),completed=[],automatic_retry=False)
    write_json(root/'LAUNCH.json',receipt)
    try:
        check_compilers()
        plan['compiler_baseline_allocated_bytes']=int(subprocess.check_output(
            ['du','-s','-B1',plan['compiler_root']],text=True).split()[0])
        write_json(plan_path,plan)
        with (root/'logs/CPU_PREFLIGHT.log').open('x') as log:
            subprocess.run([plan['training_python'],'-u','-m',MODULE+'.preflight','--plan',str(plan_path)],
                cwd=plan['source_root'],env=child_environment(plan),stdout=log,stderr=subprocess.STDOUT,check=True)
        preflight=read(root/'CPU_PREFLIGHT.json')
        if preflight['status']!='passed':raise RuntimeError('Assembled CPU preflight failed')
        receipt['completed'].append('cpu_preflight')
        initialize_work(root)
        initial=sum(plan['estimated_seconds'].values())
        receipt['time_admission']=require_time_for(plan,initial)
        receipt['status']='qualification_and_references';write_json(root/'LAUNCH.json',receipt)
        # A fresh origin reference worker is independent of disposable training.
        first=[qualification_job('training','A'),task_job(plan,'references','B')]
        if preflight.get('parallel_qualification_admitted',False):
            run_children(plan_path,plan,first)
        else:
            for job in first:run_children(plan_path,plan,[job])
        receipt['completed']+=['training_A','references'];write_json(root/'LAUNCH.json',receipt)
        # Do not infer that two trainer processes fit from a training+serving run.
        run_children(plan_path,plan,[qualification_job('training','B')])
        receipt['completed'].append('training_B')
        from .qualification import compare_training_cards
        left=read(root/'qualification/training_A/STATUS.json')['result']
        right=read(root/'qualification/training_B/STATUS.json')['result']
        cross=compare_training_cards(left,right)
        write_json(root/'CROSS_CARD_TRAINING.json',cross)
        require_cross_card_identity(cross)
        # Both per-card serving checks are required; these are not timed scores.
        for slot in ('A','B'):
            run_children(plan_path,plan,[qualification_job('serving',slot)])
            receipt['completed'].append('serving_'+slot)
        serving_cross=serving_card_comparison(cross,
            read(root/'qualification/serving_A/STATUS.json'),read(root/'qualification/serving_B/STATUS.json'))
        write_json(root/'CROSS_CARD_SERVING.json',serving_cross)
        if serving_cross['status']=='failed':
            cross=dict(cross,status='failed',serving_only_failure=True,
                errors=cross['errors']+['serving: cross-card native counts'])
        admission=qualify_resources(plan,cross)
        write_json(root/'RESOURCE_ADMISSION.json',admission);write_json(plan_path,plan)
        receipt.update(status='production',expected_remaining=admission['time'])
        write_json(root/'LAUNCH.json',receipt)
        groups=admission['groups']
        for index,jobs in enumerate(groups):
            boundary(plan,compiler=True)
            require_time_for(plan,*remaining_cost(groups[index:],plan['estimated_seconds']))
            receipt['active_tasks']=[j[0] for j in jobs];write_json(root/'LAUNCH.json',receipt)
            run_children(plan_path,plan,jobs)
            for task,_,_,_ in jobs:
                if read(root/'tasks'/task/'STATUS.json')['status']!='completed':
                    raise RuntimeError('Task ended without its completed receipt: '+task)
                receipt['completed'].append(task)
            write_json(root/'LAUNCH.json',receipt)
        work=read(root/'WORK.json')
        if work['pending'] or work['attempted']!=work['completed']:
            raise RuntimeError('Completed campaign still contains unresolved work')
        expected=dict(adam=268,response_forward=544,response_backward=544,training_initialization=4,
            feedback_generation=128,teacher_preparation=128,canonical_generation=258,
            qualification_generation=8,warmup_generation=72,scored_generation=2304)
        if work['completed']!=expected:raise ValueError('Actual final work differs from the fixed schedule')
        receipt.update(status='completed',work=work,postflight=boundary(plan,compiler=True))
    except BaseException as error:
        receipt.update(status='stopped',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        receipt['wall_s']=time.time()-receipt['started_epoch']
        if (root/'CLOCK.json').exists():receipt['clock']=clock_status(root)
        write_json(root/'LAUNCH.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    args=parser.parse_args();run(args.plan)


if __name__=='__main__':main()
