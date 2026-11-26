#!/usr/bin/env python3
"""CPU-only accounting of retained continuation-repair work, costs and limits.

No production reducers or model libraries are imported. Native-plumbing shared
Authorship is disclosed; input hashes and private locators are kept separately.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


def identity(path):
    digest=hashlib.sha256()
    size=0
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024):
            digest.update(chunk)
            size+=len(chunk)
    return dict(bytes=size,sha256=digest.hexdigest())


class CostAudit:
    def __init__(self,root):
        self.root=Path(root)
        self.checks=0
        self.inputs={}

    def check(self,condition,message):
        self.checks+=1
        if not condition:
            raise ValueError(message)

    def read(self,relative):
        path=self.root/relative
        self.inputs[str(path)]=identity(path)
        return json.loads(path.read_text())

    def clock_summary(self):
        clock=self.read('CLOCK.json')
        original=self.read('resume_001/prior/CLOCK.json')
        self.check(clock['first_gpu_action_epoch']==original['first_gpu_action_epoch'], 'Original clock epoch changed')
        rows=[]
        consumed=0.
        last_end=clock['first_gpu_action_epoch']
        for index,p in enumerate(clock['processes']):
            self.check(p['end_epoch'] is not None and p['end_epoch']>=p['start_epoch'], 'Unresolved occupied process')
            self.check(p['start_epoch']>=last_end, 'Serial GPU phases overlap')
            duration=p['end_epoch']-p['start_epoch']
            consumed+=duration
            last_end=p['end_epoch']
            rows.append(dict(process_ordinal=index,phase=p['phase'],
                physical_gpu=p.get('physical_gpu'), exit_code=p['exit_code'],
                occupied_card_s=duration, relative_start_s=p['start_epoch']-clock['first_gpu_action_epoch'],
                relative_end_s=p['end_epoch']-clock['first_gpu_action_epoch'],
                continuation=p.get('continuation',False), execution_commit=p.get('execution_commit')))
        self.check(math.isclose(consumed,clock['occupied_card_s'],rel_tol=0,abs_tol=1e-6), 'Card seconds do not sum')
        elapsed=clock['elapsed_s']
        self.check(elapsed>=last_end-clock['first_gpu_action_epoch']-1e-6,'Final elapsed clock excludes actual work')
        return dict(original_epoch=clock['first_gpu_action_epoch'], occupied_card_s=consumed,
                    elapsed_s=elapsed, non_process_elapsed_s=elapsed-consumed, phases=rows,
                    definition='Child process lifetime is charged conservatively as occupied-card time; nested timers are not added again')

    def startup(self,receipt):
        self.check(receipt['status']=='completed' and receipt['optimizer_steps']==0, 'Startup changed optimizer accounting')
        work=receipt['work']
        expected=dict(response_forwards=2,response_backwards=2,optimizer_steps=0)
        self.check(work['attempts']==expected and work['completed']==expected, 'Startup forwards/backwards changed')
        self.check(all(v==0 for v in work['unknown'].values()), 'Startup has unknown work')
        return dict(wall_s=receipt['wall_s'], logical_work=work, physical_work=receipt['physical_work']['counts'])

    def execute(self,cleanup,base_status=None,postflight=None):
        original=self.read('sealed/PLAN.json')
        final=self.read('STATUS.json')
        self.check(final['status']=='completed','Campaign is not complete')
        clock=self.clock_summary()
        self.check([(row['phase'],row['exit_code']) for row in clock['phases']]==[
            ('qualification',0),('training',0),('readout',1),('readout',1),('readout',0),('evaluation',0)],
            'Completed and failed process chain differs from the declared continuation')
        execution_view=self.read('resume_002/EXECUTION_PLAN.json')
        repair=self.read('resume_002/REPAIR.json')
        authorization=self.read('sealed/ELAPSED_EXTENSION_AUTHORIZATION.json')
        self.check(authorization['original_clock_epoch']==clock['original_epoch'], 'Authorization clock mismatch')
        self.check(authorization['original_plan_binding']==original['binding'], 'Authorization plan mismatch')
        self.check(original['limits']['elapsed_s']==14400 and execution_view['limits']['elapsed_s']==43200,
                   'Elapsed-only extension is not recorded correctly')
        for key,value in original['limits'].items():
            if key!='elapsed_s':self.check(execution_view['limits'][key]==value,'Other resource limit changed: '+key)
        self.check(execution_view['limits']['occupied_card_s']==14400, 'GPU-card cap changed')
        self.check(clock['occupied_card_s']<=14400 and clock['elapsed_s']<=43200,'Campaign exceeded approved time')
        ledger=self.read('ADAM_LEDGER.json')['calls']
        groups=Counter(row['phase'] for row in ledger)
        self.check(groups==Counter(qualification=3,training=128),'Unexpected optimizer application totals')
        self.check(all(row['status']=='completed' for row in ledger),'Unresolved optimizer application')
        qualification=self.read('qualification/RESULT.json')
        self.check(qualification['status']=='passed' and len(qualification['cases'])==3,'Qualification incomplete')
        qualification_rows=[]
        native_work=Counter()
        startup_work=Counter()
        for row in qualification['cases']:
            relative=Path('qualification')/f'fixture_{row["fixture"]}_{row["arm"]}'/'RESULT.json'
            case=self.read(relative)
            self.check(case['status']=='completed' and case['restoration_exact'], 'Disposable case state failed')
            self.check(case['counters']==[129,258], 'Qualification committed more than one update')
            work=case['actual_work']
            self.check(work['response_forwards']==2 and work['backwards_completed']==2
                       and work['backwards_attempted']==2,'Qualification logical work differs')
            self.check(case['schedule_advance_work']==dict(response_forwards=0,backwards=0,optimizer_steps=0),
                       'Unplanned qualification fixture advance')
            native_work.update(work)
            startup=self.startup(case['startup'])
            startup_work.update(startup['physical_work'])
            qualification_rows.append(dict(arm=row['arm'],fixture=row['fixture'],wall_s=case['wall_s'],
                loading_s=case['loading_s'],startup=startup,actual_work=work,
                objective_costs_nested=case['objective_costs']))
        training=self.read('arms/TAIL/STATUS.json')
        event=training['training']
        self.check(training['status']=='completed' and [training['global_step'],training['presentations'],training['completed_event']]
                   ==[256,512,2], 'Final warm state counters differ')
        self.check(event['response_presentations']==256 and event['optimizer_steps']==128,'Event work differs')
        native_work.update(training['actual_work'])
        training_startup=self.startup(training['startup'])
        startup_work.update(training_startup['physical_work'])
        records=event['records']
        self.check(len(records)==256 and sum(bool(r['optimizer_stepped']) for r in records)==128,'Presentation/step rows differ')
        timed=sum(r['training_s'] for r in records)
        self.check(math.isclose(timed,event['costs']['training_s'],rel_tol=0,abs_tol=1e-8),'Per-response timing sum differs')
        activity=Counter()
        for row in records:
            metrics=row['response_observation']['continuation_repair']
            self.check(all(math.isfinite(float(v)) for v in metrics.values()),'Nonfinite activity receipt')
            for key in ('continuation_changed_high_positions','continuation_active_response','continuation_high_count',
                        'continuation_low_count','continuation_moved_weight'):
                activity[key]+=metrics[key]
        self.check(activity['continuation_changed_high_positions']==training['activity']['continuation_changed_high_positions'],
                   'Changed HIGH occurrence count duplicated or omitted')
        self.check(activity['continuation_active_response']==training['activity']['active_response_count'],
                   'Active response count differs')
        training_cost=dict(training_s=timed, per_response_mean_s=timed/256, per_optimizer_window_mean_s=timed/128,
            costs=event['costs'],loading_s=training['loading_s'],startup=training_startup,
            process_receipt_wall_s=training['wall_s'],actual_work=training['actual_work'],
            objective_costs_nested=event['objective_work'],
            peak_allocated_bytes=training['peak_allocated_bytes'],peak_reserved_bytes=training['peak_reserved_bytes'],
            timer_scope='Training/persistence/export are nested in event/process wall; objective timers are nested in training, not additive')
        readout=self.read('mechanism/STATUS.json')
        first=self.read('mechanism/attempts/001/STATUS.json')
        second=self.read('mechanism/attempts/002/STATUS.json')
        self.check(first['status']=='failed' and first['work']['response_reads_completed']==74,'Original readout failure lost')
        self.check(readout['status']=='completed' and readout['work']['response_reads_completed']==137,
                   'Cumulative readout work incomplete')
        self.check(second['attempt_work']['response_reads_completed']==63,'Readout continuation repeated completed work')
        self.check(readout['work']==dict(Counter(first['work'])+Counter(second['attempt_work'])) or
                   all(readout['work'].get(k,0)==Counter(first['work'])[k]+Counter(second['attempt_work'])[k]
                       for k in set(readout['work'])|set(first['work'])|set(second['attempt_work'])),
                   'Readout attempts do not reconcile')
        self.check(readout['work']['head_calls_completed']==1040 and readout['work']['blocks_completed']==8320,
                   'Readout chunk/block budget differs')
        self.check(all(readout['work'][key]==0 for key in ('backwards','optimizer_steps','teacher_forwards',
                                                        'feedback_generations','serving_generations')),
                   'Readout performed forbidden work')
        states=[self.read('mechanism/attempts/001/ENDPOINT_STATE.json'),
                self.read('mechanism/attempts/002/ENDPOINT_STATE.json')]
        readout_cost=dict(cumulative_model_attempt_wall_s=readout['wall_s'],model_forward_attempts=[
            dict(attempt=1,status='failed_after74_successful_forwards',wall_s=first['wall_s'],work=first['work']),
            dict(attempt=2,status='completed_remaining63',wall_s=second['attempt_wall_s'],work=second['attempt_work'])],
            loading=[dict(wall_s=state['loading']['wall_s'],
                          device_initialization_s_nested=state['loading']['device_initialization_s_nested'],
                          exact_export_copy=state['loading']['mapping']['exact_copy'],
                          startup_forwards=state['loading']['startup_forwards'],
                          startup_backwards=state['loading']['startup_backwards']) for state in states],work=readout['work'],
            peaks=[dict(allocated_bytes=state['process_peak_allocated_bytes'],reserved_bytes=state['process_peak_reserved_bytes']) for state in states])
        readout_cost['process_attempts']=[dict(attempt=index+1,
            completed_response_forwards=completed,exit_code=phase['exit_code'],
            occupied_card_s=phase['occupied_card_s'],
            execution_commit=phase['execution_commit'] or training['execution_commit'])
            for index,(phase,completed) in enumerate(zip(clock['phases'][2:5],(74,0,63)))]
        readout_cost['cost_scope']='Two endpoint-loading/model attempts plus the separately charged zero-model checker rejection; process lifetimes include startup/cleanup and are not added to their nested model timers'
        self.check(readout_cost['cumulative_model_attempt_wall_s']==sum(row['wall_s'] for row in readout_cost['model_forward_attempts']),
                   'Readout attempt wall times changed')
        evaluation=self.read('evaluation/STATUS.json')
        summary=self.read('evaluation/SUMMARY.json')
        self.check(evaluation['status']=='completed' and evaluation['completed_valid']==792,'Serving incomplete')
        self.check(all(evaluation[k]==0 for k in ('invalid','unknown','unattempted')),'Serving has invalid/missing work')
        serving_cost=dict(recorded=summary['serving_costs'],status_wall_s=evaluation.get('wall_s'),
                          scored_generations=768,warmup_generations=24,
                          note='Serving, cache maintenance, activation, loading and warmups remain distinct; no historical serving latency claim')
        storage_records=[]
        for path in sorted(self.root.glob('STORAGE*.json')):
            value=self.read(path.name)
            if 'all_locations_charged_bytes' in value:
                storage_records.append(value)
        self.check(bool(storage_records),'No storage sampling receipts')
        minimum=min(row['free_bytes'] for row in storage_records)
        maximum=max(row['all_locations_charged_bytes'] for row in storage_records)
        self.check(minimum>=400000000000 and maximum<=24000000000,'Recorded storage cap or reserve failed')
        source_roots={str(Path(execution_view['source_manifest']).parent),
                      *execution_view.get('prior_charged_roots',[])}
        self.check(len(source_roots)==4,'Four actual source trees were not charged')
        storage=dict(sampled_minimum_free_bytes=minimum,sampled_maximum_all_locations_charged_bytes=maximum,
            allocated_limit_bytes=24000000000,reserve_bytes=400000000000,
            conservative_admission={key:original['resource_admission'][key] for key in
                ('free_bytes','remote_remaining_peak_bytes','local_bytes','all_locations_peak_bytes','checks','admitted')},
            evidence_scope='Boundary samples plus conservative admission; samples alone are not an exact instantaneous peak measurement',
            source_copies_charged=len(source_roots),
            runtime_local_allowance_bytes=execution_view['local_charged_bytes'],
            local_allowance_scope='Runtime boundary totals use the declared local allowance. Actual retrieval/archive/publication allocation is checked separately.')
        if postflight is not None:
            postflight_record=json.loads(Path(postflight).read_text())
            storage['postflight_free_bytes']=postflight_record['storage']['free_bytes']
            storage['postflight_remote_charged_bytes']=postflight_record['storage']['charged_bytes']
            self.check(postflight_record['status']=='passed' and postflight_record['four_sources_unchanged'],
                       'Postflight identities did not pass')
            self.check(set(postflight_record['source_roots'])==source_roots,
                       'Postflight does not authenticate the four charged sources')
            storage['postflight_identity']=identity(postflight)
            self.inputs[str(postflight)]=identity(postflight)
        delivery_path=self.root.parent/'RETRIEVAL.json'
        if delivery_path.exists():
            delivery=json.loads(delivery_path.read_text())
            self.inputs[str(delivery_path)]=identity(delivery_path)
            self.check(delivery['status']=='passed','Delivery authentication did not pass')
            self.check(delivery['observed_all_locations_with_archive_bytes']+
                       delivery['publication_allowance_remaining_bytes']<=24000000000,
                       'Delivery plus publication exceeded all-location allocation')
            storage['delivery']={key:delivery[key] for key in (
                'local_study_allocated_with_archive_bytes',
                'local_study_allocated_after_transport_removal_bytes',
                'additional_shared_git_allocated_bytes','observed_all_locations_with_archive_bytes',
                'all_locations_after_transport_removal_bytes','publication_allowance_remaining_bytes',
                'final_publication_recheck_required','fenrir_free_bytes','accounting_scope')}
        retired=json.loads(Path(cleanup).read_text())
        self.inputs[str(cleanup)]=identity(cleanup)
        self.check(retired['status']=='completed' and retired['freed_allocated_bytes']==19243552768,
                   'Storage retirement accounting changed')
        cleanup_public=dict(files_retired=len(retired['removed']),freed_allocated_bytes=retired['freed_allocated_bytes'],
            before_free_bytes=retired['before_free_bytes'],after_free_bytes=retired['after_free_bytes'],
            retained_selected_endpoints=len(retired['retained_endpoints']),
            no_results_or_failure_records_deleted=retired['no_results_or_failure_records_deleted'],
            lost_capability='Direct CODE event57 optimizer/RNG rollback; final event58 and compact scientific evidence remain')
        native_work.update(startup_work)
        self.check(native_work['response_forwards']==270 and native_work['backwards_completed']==270,
                   'Main+qualification+startup response work differs')
        historical=None
        if base_status is not None:
            old=json.loads(Path(base_status).read_text())
            self.inputs[str(base_status)]=identity(base_status)
            old_time=old['training']['costs']['training_s']
            historical=dict(BASE_training_s=old_time,TAIL_training_s=timed,ratio=timed/old_time,
                percent_change=100*(timed/old_time-1),
                limitation='Historical BASE event timer, not fresh matched causal overhead or training-inclusive speedup')
        for phase in clock['phases']:
            gpu=phase['physical_gpu']
            phase['physical_gpu_role']=('serving_A' if gpu==original['serving_gpu'] else
                                        'training_B' if gpu==original['training_gpu'] else 'unrecorded')
            phase.pop('physical_gpu')
            if phase['execution_commit'] is None:
                phase['execution_commit']=training['execution_commit']
        return dict(clock=clock,training=training_cost,qualification=qualification_rows,
            work=dict(qualification_adam_steps=3,committed_adam_steps=128,total_adam_steps=131,
                      process_attempts=6,completed_processes=4,failed_processes=2,
                      failed_readout_first_attempt_completed_forwards=74,
                      failed_pre_model_resume_forwards=0,
                      committed_presentations=256,qualification_presentations=6,discarded_startup_presentations=8,
                      native_training_and_startup_physical_work=dict(native_work),readout=readout['work'],
                      scored_generations=768,warmup_generations=24,new_feedback=0,new_teacher_preparation=0),
            activity=dict(activity),readout=readout_cost,serving=serving_cost,storage=storage,cleanup=cleanup_public,
            historical_BASE_context=historical,
            memory_limitations='Qualification and serving CUDA peaks were not recorded; available TAIL/readout peaks are not a measured campaign-wide maximum',
            execution=dict(original_training_commit=training['execution_commit'],
                           remaining_orchestration_commit=execution_view['execution_commit'],
                           original_clock_preserved=True,elapsed_limit_extended_only=True,
                           original_elapsed_limit_s=14400,approved_elapsed_limit_s=43200,occupied_card_limit_s=14400,
                           original_failure_and_pre_model_rejection_preserved=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',required=True)
    parser.add_argument('--cleanup',required=True)
    parser.add_argument('--base-status')
    parser.add_argument('--postflight')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    audit=CostAudit(args.raw_root)
    report=dict(status='running',independence='Separate standard-library receipt reduction; shared native/launcher authorship disclosed')
    started=time.perf_counter()
    try:
        report.update(audit.execute(args.cleanup,args.base_status,args.postflight))
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report.update(checks=audit.checks,audit_wall_s=time.perf_counter()-started)
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        (output/'COST_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (output/'COST_INPUTS_PRIVATE.json').write_text(json.dumps(audit.inputs,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checks=audit.checks)))


if __name__=='__main__':main()
