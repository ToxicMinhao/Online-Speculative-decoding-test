"""Run the fixed gates and three CODE trajectories, without automatic retries."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.launch import child_environment, check_compilers
from experiments.bounded_prefix_allocation.launch import stop_owned
from .contract import qualification_root, ARMS, PRODUCTION_WORK, RESOURCE_LIMITS, OUTPUT_POLICY
from .runtime import read, initialize, clock_status, boundary


def enforce_clock(plan):
    clock=clock_status(plan['root'])
    if (clock['elapsed_s']>=RESOURCE_LIMITS['elapsed_s']
            or clock['occupied_s']>=RESOURCE_LIMITS['occupied_card_s']):
        raise TimeoutError('Original CODE elapsed/occupied-card cap reached')
    return clock


def children(plan_path,plan,phase,slots,*,arm=None):
    root=Path(plan['root'])
    running=[]
    with ExitStack() as files:
        try:
            for slot in slots:
                name=phase+'_'+(arm or slot)
                command=[plan['training_python'],'-u','-m','experiments.opencode_native_comparison.entry',
                         '--plan',str(plan_path),'--phase',phase,'--slot',slot]
                if arm:
                    command.extend(['--arm',arm])
                log=files.enter_context((root/'logs'/(name+'.log')).open('x'))
                environment=child_environment(plan,slot)
                environment['PATH']=str(Path(plan['zstd']).parent)+os.pathsep+environment['PATH']
                process=subprocess.Popen(command,cwd=plan['source_root'],env=environment,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                receipt=dict(phase=phase,slot=slot,arm=arm,pid=process.pid,command=command,
                             started_epoch=time.time())
                running.append((name,process,receipt))
                write_json(root/'processes'/(name+'.json'),receipt)
            while any(process.poll() is None for _,process,_ in running):
                enforce_clock(plan)
                if any(process.poll() not in (None,0) for _,process,_ in running):
                    raise RuntimeError('CODE phase failed; preserve work without automatic retry')
                time.sleep(1)
            if any(process.returncode!=0 for _,process,_ in running):
                raise RuntimeError('CODE phase did not complete')
        finally:
            failure=sys.exc_info()[1]
            cleanup_errors=[]
            for name,process,receipt in running:
                try:
                    stop_owned(process)
                    receipt.update(exit_code=process.poll(),finished_epoch=time.time())
                    write_json(root/'processes'/(name+'.json'),receipt)
                except BaseException as error:
                    cleanup_errors.append(name+': '+repr(error))
            if cleanup_errors:
                if failure is not None:
                    failure.add_note('Owned process cleanup: '+repr(cleanup_errors))
                else:
                    raise RuntimeError('Owned process cleanup failed: '+repr(cleanup_errors))
    results={slot:read(root/'phases'/(phase+'_'+(arm or slot))/'STATUS.json') for slot in slots}
    if any(value['status']!='completed' for value in results.values()):
        raise ValueError('Native phase lacks a completed receipt')
    return results


def complete_qualification(plan):
    from .qualification import CASES, qualification_slots
    from .policy_resume import assert_historical_work_only, expected_qualification_totals
    root=Path(plan['root'])
    work=read(root/'WORK.json')
    exact=dict(qualification_slots()['totals'],generation=64,teacher_preparation=16)
    if plan['exact_qualification_work']!=exact:
        raise ValueError('The fixed64-generation/17-Adam/40-backward qualification changed')
    assert_historical_work_only(plan, work)
    totals = expected_qualification_totals(plan, exact)
    for field, counts in totals.items():
        for kind, value in counts.items():
            if work[field]['qualification.' + kind] != value:
                raise ValueError('Qualification physical work differs: ' + field + '.' + kind)
    for arm in ARMS:
        detail=read(qualification_root(plan)/arm/'DETAILS.json')
        if (detail['status']!='passed' or [row['case'] for row in detail['cases']]!=list(CASES[arm])
                or any(row['checks']!='passed' or row['status']!='completed' for row in detail['cases'])
                or detail['core_adam']!=len(CASES[arm])
                or detail['core_response_forward_backward_each']!=2*len(CASES[arm])
                or detail['startup_response_forward_backward_each']!=2
                or detail['origin_restored'] is not True or detail['exact_own_resume'] is not True
                or detail['exact_native_compatibility'] is not True):
            raise ValueError('Training qualification has not passed: '+arm)
    serving=read(qualification_root(plan)/'INITIAL_SERVING.json')
    if (serving['status']!='passed_initial_serving' or serving['executions']!=60
            or serving['cross_card_parity'] is not True or serving['capture_on_off_parity'] is not True):
        raise ValueError('Initial serving qualification is incomplete')
    publication=read(root/'phases/qualification_publication_A/STATUS.json')
    result=publication['result']
    if (publication['status']!='completed' or result['status']!='passed' or result['executions']!=4
            or result['exact_output_and_reload_counts'] is not True
            or result['no_additional_optimizer_steps'] is not True):
        raise ValueError('Disposable export publication/reload is unqualified')
    result=dict(status='passed',execution_commit=plan['execution_commit'],work=exact,
                physical_work=totals, historical_unknown_work=work['pending'],
                output_policy=OUTPUT_POLICY,
                native_output_checks=64,qualification_states_are_disposable=True,
                performance_improvement_required=False)
    write_json(qualification_root(plan)/'QUALIFICATION.json',result)
    return result


def run(plan_path, startup_retry_receipt=None, policy_resume_receipt=None):
    plan_path=Path(plan_path)
    plan=read(plan_path)
    root=Path(plan['root'])
    retry = None
    if startup_retry_receipt is not None and policy_resume_receipt is not None:
        raise ValueError('Choose one explicit recovery contract')
    if startup_retry_receipt is not None:
        from .startup_retry import validate_ready
        retry = validate_ready(plan, startup_retry_receipt)
    if policy_resume_receipt is not None:
        from .policy_resume import validate_ready
        retry = validate_ready(plan, policy_resume_receipt)
    if (root/'LAUNCH.json').exists() or ((root/'CLOCK.json').exists() and retry is None):
        raise FileExistsError('Do not restart an existing scientific attempt automatically')
    for name in ('logs','processes'):
        (root/name).mkdir(exist_ok=True)
    report=dict(status='preflight',execution_commit=plan['execution_commit'],
                started_epoch=time.time(),completed=[],automatic_retry=False)
    write_json(root/'LAUNCH.json',report)
    try:
        from .preflight import verify
        check_compilers()
        report['preflight']=verify(plan)
        if retry is None:
            initialize(root)
        else:
            report['authorized_policy_resume' if policy_resume_receipt else 'authorized_startup_retry'] = retry
            enforce_clock(plan)
        report['status']='qualification'; write_json(root/'LAUNCH.json',report)
        report['initial_serving']=children(plan_path,plan,'qualification_serving',('A','B'))
        from .serving_gate import merge_qualification, merge_panel
        inputs=read(plan['manifest'])
        merged=merge_qualification(plan,inputs,{slot:qualification_root(plan)/slot for slot in ('A','B')},
            qualification_root(plan)/'INITIAL_SERVING.json')
        write_json(qualification_root(plan)/'WARMUP_CANONICALS.json',merged['warmup_canonicals'])
        for arm in ARMS:
            children(plan_path,plan,'qualification_training',('B',),arm=arm)
        children(plan_path,plan,'qualification_publication',('A',))
        report['qualification']=complete_qualification(plan)
        from .qualification_cleanup import retire
        report['disposable_retirement']=retire(plan)
        report['completed'].append('qualification')
        report['status']='full_panel_precheck'; write_json(root/'LAUNCH.json',report)
        children(plan_path,plan,'precheck',('A','B'))
        precheck=merge_panel(inputs,[root/'precheck'/slot for slot in ('A','B')],
                            root/'precheck/CANONICALS.json')
        write_json(root/'precheck/AR_F_PRECHECK.json',precheck)
        report['completed'].append('full_panel_precheck')
        from .forecast import remaining
        report['full_campaign_admission']=remaining(plan)
        for index,arm in enumerate(ARMS):
            report['status']='training_'+arm; write_json(root/'LAUNCH.json',report)
            children(plan_path,plan,'train',('B',),arm=arm)
            report['completed'].append(arm)
            report['remaining_forecast']=remaining(plan,completed_arms=ARMS[:index+1])
        report['status']='final_evaluation'; write_json(root/'LAUNCH.json',report)
        write_json(root/'STORAGE_PHASE.json',dict(phase='evaluation',training_complete=True))
        children(plan_path,plan,'final',('A',))
        report['completed'].append('final_evaluation')
        work=read(root/'WORK.json')
        from .policy_resume import assert_historical_work_only
        assert_historical_work_only(plan, work)
        for kind,value in PRODUCTION_WORK.items():
            if any(work[field]['production.'+kind]!=value for field in ('attempted','completed')):
                raise ValueError('Final production work differs: '+kind)
        from .protected import check
        report['protected_dependencies']=check(root,full_hash=True,purpose='postflight')
        from .report import build_report
        report['audit_report']=build_report(plan,root/'final')
        if report['audit_report']['status']!='passed':
            raise ValueError('Final independent evidence audit did not pass; retain the partial report')
        report['postflight']=boundary(plan)
        report['status']='completed'
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report['wall_s']=time.time()-report['started_epoch']
        write_json(root/'LAUNCH.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--startup-retry-receipt')
    parser.add_argument('--policy-resume-receipt')
    args=parser.parse_args()
    run(args.plan,args.startup_retry_receipt,args.policy_resume_receipt)
