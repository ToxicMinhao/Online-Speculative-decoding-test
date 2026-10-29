"""One explicit operational resume of idle, completed event5; no scientific retry.

The original PLAN, scientific modules and seal remain unchanged. This entrypoint
admits only a recorded SIGTERM at retention_005, then runs events6..8 once using
the original collection/training/evaluation functions. All new bulky artifacts
remain local; no ACK is fabricated and no tensor is deleted.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import platform
import re
import shutil
import signal
import time
import traceback

from online_feedback.storage import write_json
from . import campaign as C
from .storage import StorageLifecycle, GIB, WEIGHTS, FEATURE_ROW_BYTES, required_bytes, identity

SCIENTIFIC_COMMIT = '04fac901f7af5478b62ff730833c026637ddd2cd'
ADDITIONS = {'manual_recovery.py', 'test_manual_recovery.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def verify_source(root):
    """Verify the old seal, allowing exactly the two separately sealed additions."""
    source = read(root/'sealed/RECOVERY_SOURCE.json')
    require(source.get('scientific_execution_commit') == SCIENTIFIC_COMMIT,
            'Recovery must preserve the original scientific execution revision')
    require(re.fullmatch('[0-9a-f]{40}', source.get('recovery_execution_commit', '')),
            'Require the complete separately published recovery revision')
    folder = Path(__file__).resolve().parent
    expected_additions = {str(folder/name) for name in ADDITIONS}
    require(set(source.get('files', {})) == expected_additions,
            'Recovery source seal must cover exactly both additive files')
    seal = read(root/'sealed/SEAL.json')
    for path, expected in {**seal['files'], **source['files']}.items():
        require(identity(path) == expected, 'Sealed file changed: '+path)
    for directory, names in seal['exact_file_sets'].items():
        expected = set(names) | (ADDITIONS if Path(directory).resolve() == folder else set())
        require({p.name for p in Path(directory).iterdir() if p.is_file()} == expected,
                'Unexpected change in sealed directory: '+directory)
    originals = read(root/'sealed/REMOTE_INPUT_IDENTITIES.json')
    for path, expected in originals.items():
        require(identity(path) == expected, 'Original input changed: '+path)
    plan = read(root/'sealed/PLAN.json')
    require(plan['execution_source'].get('commit') == SCIENTIFIC_COMMIT
        and plan['execution_source'].get('dirty') is False,
            'Original PLAN source binding changed')
    C.validate_plan(plan)
    return plan, source, dict(status='passed', old_files=len(seal['files']),
        original_inputs=len(originals), additive_files=2, original_seal=identity(root/'sealed/SEAL.json'),
        original_plan=identity(root/'sealed/PLAN.json'))


def verify_boundary(root, plan, *, decode=True):
    """Accept only the closed, exact event5 boundary, never partial science."""
    output = root/'campaign_001'
    status, costs = read(output/'STATUS.json'), read(output/'COSTS.json')
    stop = read(root/'MANUAL_STOP.json')
    require(all(stop.get(k) == v for k,v in dict(status='stopped',phase='retention_005',
        signal=15,scientific_controller_stopped=True,transfer_processes_stopped=True).items()),
        'Require an explicit stopped-controller and stopped-transfer receipt')
    require(status.get('status') == 'failed' and status.get('phase') == 'retention_005'
        and status.get('error') == "InterruptedError('Supervisor signal 15')"
        and not status.get('cleanup_error'), 'Only explicit SIGTERM while idle at retention005 is recoverable')
    required = dict(feedback_attempted=128,feedback_returned=128,optimizer_steps=256,presentations=512,
        evaluation_returned=0,active_feedback=None,arm_counters={'O':640,'K':640},events_completed=[5],
        extra_optimizer_steps=0,extra_generations=0,event=5)
    require(all(status.get(k) == v for k,v in required.items()), 'Completed-event5 work counters differ')
    require(not any(output.glob('event_00[678]'))
        and not any(output.glob('resume_00[678]_*')), 'Later scientific work already exists')
    require(not (output/'retention/event_005_ACK.json').exists()
        and not (output/'retention/event_005_RETENTION.json').exists(), 'Event5 offload already acknowledged')
    pool = read(output/'event_005/POOL.json')
    C.verify_pool(pool)
    require(pool['record_ids'] == [r['request_id'] for r in plan['windows'][0]['requests']],
            'Completed feedback differs from the sealed event5 requests')
    require(pool['checkpoint'] == plan['initial']['O']['export'], 'Event5 feedback came from a different O parent')
    request = read(output/'retention/event_005_REQUEST.json')
    require(request['event'] == 5 and request['campaign_root'] == str(output)
        and request['phase'] == 'idle_after_training_and_evaluation', 'Idle retention request binding differs')
    for relative, expected in request['files'].items():
        path = output/relative
        require(not path.is_symlink() and path.resolve().is_relative_to(output.resolve())
                and identity(path) == expected, 'Event5 retention byte changed: '+relative)
    parents, receipts, decoded, reference_rng, reference_draws = {}, {}, {}, None, None
    from .resume import validate_event_receipt, load_snapshot, differences
    from .preflight import history_draws
    for arm, objective in C.OBJECTIVES.items():
        folder = output/arm/'events/event_005'
        receipt = read(folder/'RESULT.json')
        parent = dict(state=str(output/arm/'training_state.pt'),export=str(folder/'export'),
                      receipt=str(folder/'RESULT.json'))
        require(receipt['full_state_path'] == parent['state'] and receipt['export_path'] == parent['export'],
                'Event5 result is not bound to this own restart/export')
        require(not (output/arm/'training_state.pending.pt').exists()
            and {p.name for p in (output/arm/'events').iterdir()} == {'event_005'},
            'Uncommitted or later own training state exists')
        audit = output/f'resume_005_{arm}'
        closed = read(audit/'CLOSED.json')
        require(closed.get('closed') is True and closed.get('failed') is False,
                'Both event5 trainers must be cleanly closed')
        validate_event_receipt(receipt,event_number=5,objective=objective,
            trainer_status=closed,reference_schedule=receipts.get('O') if arm == 'K' else None)
        for name in ('EVENT_CHECK.json','SAVED_ENDPOINT.json','LIVE_BOUNDARY.json'):
            require(read(audit/name).get('status') == 'passed', 'Missing successful event5 witness: '+name)
        boundary = read(audit/'LIVE_BOUNDARY.json')
        require(boundary.get('exact_boundary') is True and boundary.get('mismatches') == []
            and boundary.get('global_step') == 640 and boundary.get('presentations') == 1280,
            'Event5 saved/live boundary was not exact')
        identities = read(audit/'STATE_IDENTITIES.json')
        require(set(identities) == {parent['state'],*[str(p) for p in Path(parent['export']).iterdir() if p.is_file()]},
                'Event5 endpoint identity coverage differs')
        for path, expected in identities.items():
            require(identity(path) == expected, 'Event5 own endpoint changed: '+path)
        if decode:
            state, check = load_snapshot(parent['state'],parent['export'],objective=objective,
                config_text=Path(plan['config']).read_text(),completed_event=5)
            draws = history_draws(state['events'])
            if arm == 'O':
                reference_rng, reference_draws = state['rng'], draws
            else:
                require(not differences(reference_rng,state['rng'])
                    and not differences(reference_draws,draws), 'Event5 O/K natural RNG/history differ')
            decoded[arm] = check
            del state
            gc.collect()
        parents[arm], receipts[arm] = parent, receipt
    require(all(key in costs for key in ('event_005_feedback','event_005_O','event_005_K','campaign_wall_s')),
            'Original completed-event5 costs are missing')
    return status, costs, parents, pool, dict(status='passed',decoded=decoded,
        original_status=identity(output/'STATUS.json'),original_costs=identity(output/'COSTS.json'),
        event5_retention_request=identity(output/'retention/event_005_REQUEST.json'),
        model_passes=0,optimizer_steps=0,generations=0)


class LocalRetention(StorageLifecycle):
    """Keep every tensor locally, including immutable complete6/8 endpoints."""
    def __init__(self, plan, output):
        self.output = Path(output).resolve()
        self.root = self.output/'retention'
        require(self.root.is_dir(), 'Original retention evidence must remain')
        self.settings = plan['storage']
        require(self.settings['reserve_bytes'] == 8*GIB, 'Preserve the8GiB reserve')
        self.archived = {}

    def retire_obsolete_exports(self, parents):
        return None

    def after_event(self, event, parents, pool, *, evaluated):
        require(evaluated == (event in (6,8)), 'Local retention follows all required evaluation')
        started = time.perf_counter()
        files = {Path(path) for path in pool['features']}
        for path in (self.output/f'event_{event:03d}/responses').glob('*.json'):
            files.add(Path(read(path)['metadata']['capture']['feature_path']))
        features = self.output/f'event_{event:03d}/features'
        files.update(features.glob('*.pt'))
        for path in features.glob('*.terminal.json'):
            files.add(Path(read(path)['tensor_path']))
        for parent in parents.values():
            files.update(p for p in Path(parent['export']).iterdir() if p.is_file())
            if event in (6,8):
                files.add(Path(parent['state']))
        for path in files:
            require(not path.is_symlink() and path.resolve().is_relative_to(self.output),
                    'Local retention path escaped the campaign')
        hashes = {str(p.relative_to(self.output)):identity(p) for p in sorted(files)}
        self.archived.update(hashes)
        free = shutil.disk_usage(self.output).free
        require(free >= 8*GIB, 'Local retention reserve violated')
        receipt = dict(status='completed',event=event,mode='retained_locally',
            destination=str(self.output),verified_files=hashes,removed_local_copies=[],
            free_bytes=free,reserve_bytes=8*GIB,wall_s=time.perf_counter()-started,
            timing_scope='Idle after event; no transfer, deletion or extra scientific work')
        write_json(self.root/f'event_{event:03d}_LOCAL_RETENTION.json',receipt)
        return receipt

    def finalize(self, parents):
        for event in (6,8):
            for arm in C.OBJECTIVES:
                for name in ('training_state.pt','export/model.safetensors'):
                    relative=f'{arm}/events/event_{event:03d}/{name}'
                    require(relative in self.archived and identity(self.output/relative) == self.archived[relative],
                            'Required locally retained endpoint changed: '+relative)
        # Recheck every retained feature/export as well; no copies are retired.
        for relative, expected in self.archived.items():
            require(identity(self.output/relative) == expected, 'Local retained artifact changed: '+relative)
        write_json(self.root/'FINAL.json',dict(status='passed',mode='retained_locally',
            required_events=[6,8],destination=str(self.output),verified_files=self.archived,
            latest_states={arm:parent['state'] for arm,parent in parents.items()}))


def runtime_admission(root, plan):
    from .preflight import hardware_admission
    from online_feedback.qualify_sglang import check_recipe
    from experiments.osd_components.prepare import verify_files, verify_native_runtime
    import torch
    import transformers
    require(not torch.cuda.is_initialized(), 'Recovery preflight must not create CUDA context')
    hardware = hardware_admission()
    meminfo=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available=int(meminfo['MemAvailable'].split()[0])*1024
    free=shutil.disk_usage(root).free
    payloads=[sum(len(r['prompt_ids'])+r['max_new_tokens'] for r in w['requests'])*FEATURE_ROW_BYTES
              for w in plan['windows'][1:]]
    training=required_bytes(5*WEIGHTS)
    peaks=[2*payloads[0]+WEIGHTS+training,
           2*sum(payloads[:2])+6*WEIGHTS+training,
           2*sum(payloads)+11*WEIGHTS+training]
    capacity=dict(free_bytes=free,host_available_bytes=available,host_required_bytes=80*GIB,
        next_collection_required_bytes=required_bytes(payloads[0]),
        remaining_all_cap_phase_peaks_bytes=dict(zip(('6','7','8'),peaks)),
        remaining_all_cap_required_bytes=max(peaks),all_future_maximum_payloads_fit=free>=max(peaks),
        basis='Incremental from occupied completed-event5 boundary; all new feature copies and exports retained, complete6/8states retained; phase-specific admission repeats using actual occupied bytes.',
        reserve_bytes=8*GIB,no_guarantee_all_future_maximum_payloads_fit=True)
    require(available>=80*GIB and free>=capacity['next_collection_required_bytes'],
            'Recovery host/next-collection storage admission failed')
    pins = read(root/'sealed/MODEL_PIN.json')
    for name in ('target','draft'):
        verify_files(plan[name],pins[name]['files'],exact_files=True)
    runtime = verify_native_runtime(plan['worker_python'],read(root/'sealed/RUNTIME_PIN.json'))
    require(dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__)
        == read(root/'sealed/TRAINING_PIN.json')['environment'], 'Qualified training environment changed')
    return dict(status='passed',hardware=hardware,capacity=capacity,runtime=runtime,recipe=check_recipe(plan['config']))


def run(root):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .evaluate import evaluate
    root = Path(root).resolve()
    control = root/'recovery_001'
    control.mkdir(exist_ok=False)
    output=root/'campaign_001'
    backend=None
    started=time.perf_counter()
    progress=costs=None
    receipt=dict(status='validating',manual_operational_resume=True,automatic_retry=False,
        scientific_execution_commit=SCIENTIFIC_COMMIT,start_event=6,end_event=8,
        native_output=str(output/'native_recovery_001'))
    try:
        # Preserve exact failure/cost bytes before any shared status is changed.
        for name in ('STATUS.json','COSTS.json'):
            shutil.copyfile(output/name,control/('ORIGINAL_'+name))
        plan,source,source_check=verify_source(root)
        receipt.update(recovery_execution_commit=source['recovery_execution_commit'],source_check=source_check)
        prior,prior_costs,parents,pool,boundary=verify_boundary(root,plan)
        receipt['boundary']=boundary
        receipt['runtime_admission']=runtime_admission(root,plan)
        storage=LocalRetention(plan,output)
        storage.after_event(5,parents,pool,evaluated=False)
        costs=copy.deepcopy(prior_costs)
        costs['execution_segments']=[dict(execution_commit=SCIENTIFIC_COMMIT,phase='original_through_idle_event5',
            wall_s=prior_costs['campaign_wall_s'],backend_loading_s=prior_costs.get('backend_loading_s',0))]
        costs['operational_recovery']=dict(scientific_execution_commit=SCIENTIFIC_COMMIT,
            recovery_execution_commit=source['recovery_execution_commit'],receipt=str(control/'RECOVERY.json'))
        progress=copy.deepcopy(prior)
        for key in ('error','traceback','cleanup_error'):
            progress.pop(key,None)
        progress.update(status='running',phase='loading_native_recovery',manual_operational_resume=True)
        def save():
            write_json(output/'STATUS.json',progress)
        save()
        backend=SGLangDFlashBackend(plan['target'],plan['draft'],worker_python=plan['worker_python'],
            output_root=output/'native_recovery_001',serving_gpu='0',context_limit=4096,block_size=16)
        receipt['status']='running'
        write_json(control/'RECOVERY.json',receipt)
        for window in plan['windows'][1:]:
            number=window['event_index']
            progress.update(event=number,phase='collecting_feedback');save()
            pool,pool_cost=C.collect_pool(backend,window,parents['O']['export'],
                output/window['event_id'],progress,save,storage)
            costs[f'event_{number:03d}_feedback']=pool_cost
            ordinary=None
            for arm in C.OBJECTIVES:
                progress['phase']='training_'+arm;save()
                C.verify_pool(pool);storage.admit_training(arm,number,pool,parents)
                result,cost=C.train_event(plan,arm,number,parents[arm],pool,output,ordinary=ordinary)
                if arm=='O':ordinary=result
                C.verify_pool(pool)
                parents[arm]=dict(state=result['full_state_path'],export=result['export_path'],
                    receipt=str(output/arm/'events'/f'event_{number:03d}'/'RESULT.json'))
                progress['optimizer_steps']+=128;progress['presentations']+=256
                progress['arm_counters'][arm]=result['end_step'];save()
                costs[f'event_{number:03d}_{arm}']=cost;write_json(output/'COSTS.json',costs)
            progress['events_completed'].append(number);save()
            if number in (6,8):
                for panel in plan['panels']:
                    if panel['name']=='fresh' and number!=8:continue
                    progress['phase']=f"evaluation_{number:03d}_{panel['name']}";save()
                    status=evaluate(backend,panel['requests'],{arm:parents[arm]['export'] for arm in C.OBJECTIVES},
                        output/f'event_{number:03d}'/'evaluation'/panel['name'],event_number=number,
                        panel_name=panel['name'],canonical_mode=panel['canonical_mode'],
                        qualification_requests=8 if number==6 and panel['name']=='development' else 0)
                    progress['evaluation_returned']+=status['returned_generations'];save()
            progress['phase']=f'retention_{number:03d}';save()
            storage.after_event(number,parents,pool,evaluated=number in (6,8))
            write_json(output/'CURRENT_PARENTS.json',parents)
        storage.finalize(parents);C.reconcile_work(output,progress)
        require(tuple(progress[k] for k in ('optimizer_steps','presentations','feedback_returned',
            'evaluation_returned','evaluation_valid'))==(1024,2048,512,1920,1920)
            and progress['arm_counters']=={'O':1024,'K':1024},'Completed whole campaign budget differs')
        progress.update(status='completed',phase='finished');receipt['status']='completed'
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        if progress is not None:
            progress.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            try:backend.close()
            except BaseException as error:
                receipt.update(status='failed',cleanup_error=repr(error))
                if progress is not None:progress.update(status='failed',cleanup_error=repr(error))
        if receipt['status']=='completed':
            try:
                _,_,postflight=verify_source(root)
                write_json(control/'SOURCE_POSTFLIGHT.json',postflight)
            except BaseException as error:
                receipt.update(status='failed',source_postflight_error=repr(error))
                progress.update(status='failed',error=repr(error))
                write_json(control/'SOURCE_POSTFLIGHT.json',dict(status='failed',error=repr(error)))
        receipt['recovery_wall_s']=time.perf_counter()-started
        if progress is not None:
            C.reconcile_work(output,progress)
            costs['execution_segments'].append(dict(execution_commit=receipt['recovery_execution_commit'],
                phase='manual_event5_boundary_resume6_through8',wall_s=receipt['recovery_wall_s'],
                backend_loading_s=backend.loading_s if backend is not None else 0))
            costs['campaign_wall_s']=sum(x['wall_s'] for x in costs['execution_segments'])
            costs['backend_loading_s']=sum(x['backend_loading_s'] for x in costs['execution_segments'])
            costs['wall_time_scope']='Sum of both executed controller segments, including idle transfer wait; excludes stopped interval.'
            write_json(output/'COSTS.json',costs);write_json(output/'STATUS.json',progress)
        write_json(control/'RECOVERY.json',receipt)
    require(receipt['status']=='completed','Recovery cleanup or source postflight failed')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    args=parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('Manual recovery supervisor signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    run(args.root)
