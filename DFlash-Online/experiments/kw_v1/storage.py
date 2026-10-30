"""Full-capacity retention budget and phase admission, with no offload assumption."""
import json
from pathlib import Path
import shutil
import time

from online_feedback.storage import write_json
from experiments.ok_events_5_8.storage import GIB, WEIGHTS, FEATURE_ROW_BYTES, identity, required_bytes

ARMS=('O','K','KW')


def full_campaign_budget(plan):
    """Simulate incremental bytes beyond already-retained O8/K8 and source data.

    All capture/feature tensors persist. Event10/12 states and exports persist.
    Event9/11 complete rolling snapshots persist through the next complete event.
    Include a full pending snapshot, export and export workspace before replacement.
    """
    occupied=0; phases=[]; states={}; exports={}
    def phase(name, temporary=0):
        payload=occupied+temporary
        phases.append(dict(phase=name,payload_bytes=payload,required_bytes=required_bytes(payload)))
    for window in plan['windows']:
        event=window['event_index']
        payload=sum(len(r['prompt_ids'])+r['max_new_tokens'] for r in window['requests'])*FEATURE_ROW_BYTES
        occupied+=payload;phase(f'event_{event:03d}_capture')
        occupied+=payload;phase(f'event_{event:03d}_prepared')
        for arm in ARMS:
            phase(f'event_{event:03d}_{arm}_pending',5*WEIGHTS)
            # Immutable event link aliases latest; it costs one complete3W state.
            occupied+=4*WEIGHTS
            states[(arm,event)]=3*WEIGHTS; exports[(arm,event)]=WEIGHTS
            phase(f'event_{event:03d}_{arm}_committed')
        phase(f'event_{event:03d}_closed_before_retention')
        if event in (10,12):
            for arm in ARMS:
                occupied-=states.pop((arm,event-1))+exports.pop((arm,event-1))
            phase(f'event_{event:03d}_retained')
    peak=max(phases,key=lambda r:r['required_bytes'])
    return dict(mode='retain_local',assumed_transfer_bytes=0,feature_cap_rows=sum(
        len(r['prompt_ids'])+r['max_new_tokens'] for w in plan['windows'] for r in w['requests']),
        complete_capture_and_prepared_copies=True,phases=phases,peak_phase=peak['phase'],
        peak_required_bytes=peak['required_bytes'],final_incremental_payload_bytes=occupied,
        reserve_bytes=8*GIB,slack_fraction=.10,compact_and_workspace_bytes=4*GIB,
        initial_original_states_scope='Already occupied; never retired by this campaign.')


class StorageLifecycle:
    def __init__(self,plan,output,*,resume=False):
        self.plan=plan;self.output=Path(output).resolve();self.root=self.output/'retention'
        self.root.mkdir(exist_ok=resume)
        settings=plan['storage']
        if settings.get('mode')!='retain_local' or settings.get('offload_assumed') is not False or settings['reserve_bytes']!=8*GIB:
            raise ValueError('Require full local retention with8GiB reserve; no assumed offload throughput')

    def admit(self,phase,payload,basis):
        free=shutil.disk_usage(self.output).free
        row=dict(phase=phase,payload_bytes=payload,required_bytes=required_bytes(payload),
            free_bytes=free,reserve_bytes=8*GIB,basis=basis,admitted=free>=required_bytes(payload))
        write_json(self.root/(phase+'_STORAGE.json'),row)
        if not row['admitted']: raise RuntimeError('Insufficient peak storage: '+phase)
        return row

    def admit_collection(self,window):
        payload=sum(len(r['prompt_ids'])+r['max_new_tokens'] for r in window['requests'])*FEATURE_ROW_BYTES
        return self.admit(f"event_{window['event_index']:03d}_collection",payload,
            'All128 maximum-length captures; every earlier pool remains retained')

    def admit_preparation(self,window,records):
        payload=sum(len(r['prompt_ids'])+len(r['response_ids']) for r in records)*FEATURE_ROW_BYTES
        return self.admit(f"event_{window['event_index']:03d}_preparation",payload,
            'Prepared copies while captured originals remain retained')

    def admit_training(self,arm,event,pool,parents):
        return self.admit(f'event_{event:03d}_{arm}_training',5*WEIGHTS,
            '3W pending complete state+W export+W export workspace; previous boundary remains retained')

    def after_event(self,event,parents,pool,*,evaluated):
        if evaluated!=(event in (10,12)): raise ValueError('Retention must follow required evaluation')
        started=time.perf_counter();before=shutil.disk_usage(self.output).free
        # Read the exact decoded-state/export audits produced after every save.
        for arm in ARMS:
            audit=self.output/f'resume_{event:03d}_{arm}'
            check=json.loads((audit/'SAVED_ENDPOINT.json').read_text())
            boundary=json.loads((audit/'LIVE_BOUNDARY.json').read_text())
            if (check['status']!='passed' or not check['export_exact'] or not boundary['exact_boundary']
                    or check['completed_event']!=event or check['global_step']!=event*128):
                raise ValueError('Successor full-state/export verification missing; no retirement')
            for path,expected in json.loads((audit/'STATE_IDENTITIES.json').read_text()).items():
                if identity(path)!=expected: raise ValueError('Saved successor identity changed: '+path)
        removed=[];retire=[]
        if event in (10,12):
            previous=event-1
            for arm in ARMS:
                old=self.output/arm/'events'/f'event_{previous:03d}'
                old_ids=json.loads((self.output/f'resume_{previous:03d}_{arm}'/'STATE_IDENTITIES.json').read_text())
                candidates=[old/'training_state.pt',*(old/'export').glob('*')]
                for path in candidates:
                    if not path.is_file(): continue
                    if path.is_symlink() or not path.resolve().is_relative_to(self.output) or identity(path)!=old_ids.get(str(path)):
                        raise ValueError('Rolling boundary identity differs; no retirement: '+str(path))
                    retire.append(dict(path=str(path),identity=old_ids[str(path)],
                        reason=f'Complete own successor event{event} verified; rolling boundary superseded',
                        retained_successor=parents[arm]['state'],
                        lost_capability=f'Exact event{previous} resume and serving; compact receipt remains'))
            write_json(self.root/f'event_{event:03d}_RETIREMENT_INTENT.json',dict(retire=retire,
                before_free_bytes=before,successors_verified=True))
            # All candidates are positively identified before the first unlink.
            for row in retire:
                Path(row['path']).unlink();removed.append(row)
                write_json(self.root/f'event_{event:03d}_RETIREMENT_PROGRESS.json',dict(removed=removed,
                    before_free_bytes=before,free_bytes=shutil.disk_usage(self.output).free))
        receipt=dict(status='completed',event=event,removed=removed,
            retained_latest=parents,all_new_feedback_features_retained=True,offload_bytes=0,
            before_free_bytes=before,free_bytes=shutil.disk_usage(self.output).free,
            wall_s=time.perf_counter()-started,timing_scope='Idle after training/evaluation; no timed transfer')
        write_json(self.root/f'event_{event:03d}_RETENTION.json',receipt)
        return receipt

    def finalize(self,parents):
        files={}
        for event in (10,12):
            for arm in ARMS:
                audit=self.output/f'resume_{event:03d}_{arm}'/'STATE_IDENTITIES.json'
                bindings=json.loads(audit.read_text())
                for path,expected in bindings.items():
                    if identity(path)!=expected:raise ValueError('Required retained endpoint changed: '+path)
                    files[path]=expected
        for event in (9,10,11,12):
            from .campaign import verify_pool
            pool=json.loads((self.output/f'event_{event:03d}'/'POOL.json').read_text())
            verify_pool(pool)
            if not pool.get('retained_source_identities'):
                raise ValueError('Native captures/terminal retention inventory missing')
            for path,expected in pool['retained_source_identities'].items():
                if identity(path)!=expected:raise ValueError('Retained feedback source changed: '+path)
        write_json(self.root/'FINAL.json',dict(status='passed',required_events=[10,12],
            original8_retained=True,retained_files=files,all_new_feedback_features_retained=True,
            latest_states={a:p['state'] for a,p in parents.items()},offload_assumed=False))
