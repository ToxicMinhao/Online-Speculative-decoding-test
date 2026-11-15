"""One authoritative native event and sixteen cached-gradient scalar calibrations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import traceback

from experiments.threatened_repair_replay.runner import export_weights, host_admission, termination_cleanup
from experiments.threatened_repair_replay.state import capture_state, restore_state, verify_state, tensor_change, compare_final_native
from experiments.threatened_repair_replay.training import make_trainer
from experiments.threatened_repair_replay.work import Ledger, TrainingInstrumentation, WorkLimit
from online_feedback.storage import write_json
from .calibration import append_receipt, cached_trial, calibrate, ordinary_pair, norm_agreement
from .qualification import qualify

WINDOWS = tuple(range(8,129,8))


def read(path):
    return json.loads(Path(path).read_text())


def write_table(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_geometry(pre, endpoint, native):
    """Detached FP64 displacement geometry, without a model forward."""
    import math
    import torch

    layers = {}
    total = dict(candidate_squared=0.,native_squared=0.,dot=0.,changed_entries=0,elements=0)
    if pre.keys() != endpoint.keys() or pre.keys() != native.keys():
        raise ValueError('Geometry tensor membership differs')
    for name,original in pre.items():
        parts = name.split('.')
        layer = '.'.join(parts[:2]) if len(parts)>1 and parts[0]=='layers' else 'other'
        row = layers.setdefault(layer,dict(candidate_squared=0.,native_squared=0.,dot=0.,changed_entries=0,elements=0))
        a,b,c = (value.detach().flatten() for value in (original,endpoint[name],native[name]))
        for offset in range(0,a.numel(),1 << 20):
            p = a[offset:offset+(1 << 20)].to(device='cpu',dtype=torch.float64)
            candidate = b[offset:offset+(1 << 20)].to(device='cpu',dtype=torch.float64)
            ordinary = c[offset:offset+(1 << 20)].to(device='cpu',dtype=torch.float64)
            if not bool(torch.isfinite(p).all() and torch.isfinite(candidate).all() and torch.isfinite(ordinary).all()):
                raise ValueError('Nonfinite update geometry: '+name)
            delta,delta_native = candidate-p,ordinary-p
            values = dict(candidate_squared=float(delta.square().sum()),native_squared=float(delta_native.square().sum()),
                dot=float((delta*delta_native).sum()),changed_entries=int((candidate!=p).sum()),elements=p.numel())
            for key,value in values.items():
                row[key] += value
                total[key] += value
    def finish(values):
        first,second = math.sqrt(values['candidate_squared']),math.sqrt(values['native_squared'])
        return dict(norm=first,comparison_norm=second,
            cosine=None if not first or not second else values['dot']/(first*second),
            changed_entries=values['changed_entries'],elements=values['elements'])
    return dict(global_geometry=finish(total),layers={name:finish(values) for name,values in layers.items()})


def price_remaining(plan, qualification, deadline, *, windows_done=0, native_done=0,
                    first_window_s=None, first_window_trials=None):
    """Price full remaining scope; a failure never narrows the requested panel."""
    native_case = qualification['cases'][0]
    pair_s = native_case['update']['wall_s']
    cached_s = max(row['update']['wall_s'] for row in qualification['cases'][1:])
    native_s = (128-native_done)*pair_s
    scalar_s = (16-windows_done)*10*cached_s
    export_s = float(plan.get('validation_export_estimate_s',8.))
    validation_s = max(0,32-2*windows_done)*export_s
    selected_export_s = max(0,32-2*windows_done)*export_s
    overhead_s = float(plan.get('remaining_overhead_estimate_s',900.))
    if first_window_s is not None:
        # Observed enclosing window includes restores, identity checks, norm scans
        # and exports. Extrapolate a full ten trials, even if its bracket was early.
        completed_trials = int(first_window_trials)
        per_window = first_window_s + (10-completed_trials)*cached_s
        measured_windows_s = (16-windows_done)*per_window
        calibrated_s = max(scalar_s+validation_s+selected_export_s,measured_windows_s)
    else:
        calibrated_s = scalar_s+validation_s+selected_export_s
    probe_s = float(plan['remaining_probe_estimate_s'])
    estimate = native_s+calibrated_s+probe_s+overhead_s
    remaining = deadline-time.monotonic()
    return dict(status='passed' if estimate<=remaining else 'blocked',estimated_remaining_s=estimate,
        remaining_cap_s=remaining,native_s=native_s,scalar_calibration_s=scalar_s,
        native_validation_io_s=validation_s,selected_control_io_s=selected_export_s,
        calibrated_window_allowance_s=calibrated_s,native_probe_s=probe_s,other_overhead_s=overhead_s,
        qualification_native_pair_s=pair_s,qualification_max_cached_trial_s=cached_s,
        windows_done=windows_done,native_steps_done=native_done,
        basis='Full remaining panel and ten-trial ceiling; no outcome-dependent scope change')


class WindowController:
    """Only N continues; scalar alternatives never change its optimizer/RNG."""
    def __init__(self,plan,output,ledger,qualification,deadline,*,authenticate=None,
                 witness_check=None,exporter=export_weights,storage_check=None):
        from .identity_gate import authenticate_endpoint, compare_boundary_witness
        from .storage import check_write
        self.plan,self.output,self.ledger = plan,Path(output),ledger
        self.qualification,self.deadline = qualification,deadline
        self.authenticate = authenticate or authenticate_endpoint
        self.witness_check = witness_check or compare_boundary_witness
        self.export,self.storage_check = exporter,storage_check or check_write
        self.windows,self.exports,self.trials,self.geometry,self.parity = [],[],[],[],[]

    def publish(self):
        for name,rows in (('WINDOWS',self.windows),('PROBE_EXPORTS',self.exports),
                          ('CALIBRATION_TRIALS',self.trials),('UPDATE_GEOMETRY',self.geometry),
                          ('STATE_PARITY',self.parity)):
            write_json(self.output/(name+'.json'),rows)
        trial_rows = []
        for row in self.trials:
            trial_rows.append(dict(window=row['window'],trial_index=row['trial_index'],alpha=row['alpha'],
                status=row['status'],actual_lr=json.dumps(row.get('requested_lrs')),
                actual_norm=row.get('actual_norm'),changed_entries=row.get('displacement',{}).get('changed_entries'),
                clip_norm=row.get('clip_norm'),clip_factor=row.get('clip_factor'),
                moment_parity=row.get('moment_clip_option_parity',{}).get('status'),
                selected_as=','.join(row.get('selected_as',[])),wall_s=row.get('wall_s'),error=row.get('error')))
        write_table(self.output/'CALIBRATION_TRIALS.csv',trial_rows)
        controls = [{key:row.get(key) for key in ('window','state','alpha','target_norm','actual_norm',
                    'relative_mismatch','trial_index','checkpoint','reused_from')} | dict(
                    export_sha256=row['weights_identity']['sha256']) for row in self.exports]
        write_table(self.output/'SELECTED_CONTROLS.csv',controls)

    def _selected_export(self,point,window,state,target):
        self.ledger.check()
        size = sum(value.numel()*value.element_size() for value in point.weights.values())+(1 << 20)
        self.storage_check(self.plan,size,'selected_control_'+str(window)+'_'+state)
        destination = self.output/'exports'/f'window_{window:03d}_{state}'
        began = time.perf_counter()
        receipt = self.export(point.weights,self.plan['origin_export'],destination)
        # Check the stored BF16 payload against the selected in-memory endpoint;
        # file identity alone cannot establish that the writer used these weights.
        from safetensors.torch import load_file
        from experiments.ok_continuation.resume import differences
        restored = load_file(str(destination/'model.safetensors'),device='cpu')
        mismatch = differences(dict(restored),dict(point.weights),'selected_export')
        if mismatch:
            write_json(destination/'FAILED.json',dict(status='failed',mismatches=mismatch[:64]))
            raise ValueError('Selected scalar export differs from its attained endpoint')
        del restored
        row = dict(window=window,state=state,reused_from=None,alpha=point.alpha,
            target_norm=target,actual_norm=point.norm,relative_mismatch=abs(point.norm-target)/target,
            trial_index=point.trial_index,seal='weights_only_norm_selected_before_scoring',**receipt)
        row['export_and_authentication_s'] = time.perf_counter()-began
        self.exports.append(row)
        return row

    def __call__(self,trainer,window,pair,pass_index,ordinal):
        from experiments.functional_step_geometry.qualify import cleanup

        self.ledger.check()
        directory = self.output/f'window_{window:03d}'
        directory.mkdir()
        record = dict(window=window,status='running',request_ids=[row['record_id'] for row in pair])
        self.windows.append(record)
        before = capture_state(trainer)
        native,failure = None,None
        began = time.perf_counter()
        history = self.plan['historical_windows'][str(window)]
        try:
            record['P_identity'] = self.authenticate(self.plan,window,'P',before['scientific']['draft_state_dict'],directory)
            if window in (8,128):
                record['boundary_witness'] = self.witness_check(before,window,self.plan,directory)
            result = ordinary_pair(trainer,pair,pass_index,ordinal,category='native',label='N')
            native = capture_state(trainer)
            record['N_identity'] = self.authenticate(self.plan,window,'N',native['scientific']['draft_state_dict'],directory)
            displacement = tensor_change(before['scientific']['draft_state_dict'],native['scientific']['draft_state_dict'])
            record['native_update'] = result['update']
            record['native_displacement'] = displacement
            if not norm_agreement(displacement['difference_norm'],history['N_norm']):
                raise ValueError('Reconstructed N displacement differs from authenticated historical measurement')
            native_gradients = result['preclip_gradients']['native']
            def trial(alpha,index):
                self.ledger.check()
                return cached_trial(trainer,before,native,native_gradients,result['update'],alpha,
                    window=window,trial_index=index,output=directory/'trials')
            def observed(row):
                receipt = dict(row,window=window)
                append_receipt(self.output/'CALIBRATION_ATTEMPTS.jsonl',receipt)
                if row['status'] != 'started':
                    self.trials.append(receipt)
                    self.publish()
            lower,upper,attempts = calibrate(history['E_norm'],displacement['difference_norm'],
                native['scientific']['draft_state_dict'],trial,on_attempt=observed)
            selected = {lower.trial_index:['S_low']}
            selected.setdefault(upper.trial_index,[]).append('S_high')
            for row in self.trials:
                if row['window'] == window:
                    row['selected_as'] = selected.get(row['trial_index'],[])
                    append_receipt(self.output/'CALIBRATION_ATTEMPTS.jsonl',dict(
                        phase='selection_sealed',window=window,trial_index=row['trial_index'],selected_as=row['selected_as']))
            low_export = self._selected_export(lower,window,'S_low',history['E_norm'])
            if upper is lower:
                high_export = dict(low_export,state='S_high',reused_from='S_low',
                    alias_reason='exact attained norm match; one physical endpoint')
                self.exports.append(high_export)
            else:
                high_export = self._selected_export(upper,window,'S_high',history['E_norm'])
            for state,point in (('S_low',lower),('S_high',upper)):
                geometry = update_geometry(before['scientific']['draft_state_dict'],point.weights,
                                           native['scientific']['draft_state_dict'])
                row = dict(window=window,state=state,against='N',**geometry)
                if window in (8,128):
                    from safetensors.torch import load_file
                    e_weights = load_file(str(Path(history['E']['checkpoint'])/'model.safetensors'),device='cpu')
                    row['against_E'] = update_geometry(before['scientific']['draft_state_dict'],point.weights,e_weights)
                    del e_weights
                else:
                    row['against_E'] = dict(status='unavailable',reason='historical E weights retired; no regeneration')
                self.geometry.append(row)
            record.update(status='completed',trials=len(attempts),target_norm=history['E_norm'],
                lower_norm=lower.norm,upper_norm=upper.norm,exact_alias=upper is lower,
                lower_alpha=lower.alpha,upper_alpha=upper.alpha)
            del lower,upper
            result['preclip_gradients'].clear()
            if window == 8:
                record['remaining_cost_admission'] = price_remaining(self.plan,self.qualification,self.deadline,
                    windows_done=1,native_done=8,first_window_s=time.perf_counter()-began,
                    first_window_trials=len(attempts))
                write_json(self.output/'AFTER_FIRST_WINDOW_ADMISSION.json',record['remaining_cost_admission'])
                if record['remaining_cost_admission']['status'] != 'passed':
                    raise WorkLimit('Full remaining comparison does not fit after first calibration window')
            return result
        except BaseException as error:
            failure = error
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
            write_json(directory/'FAILED.json',record)
            raise
        finally:
            endpoint = native if native is not None else before
            cleanup(lambda: restore_state(trainer,endpoint),failure,record,'restore authoritative native endpoint')
            cleanup(lambda: self.parity.append(dict(window=window,
                **verify_state(trainer,endpoint,directory/'RESTORED_NATIVE.json'))),failure,record,'verify native endpoint')
            record['wall_s'] = time.perf_counter()-began
            cleanup(lambda: write_json(directory/'RESULT.json',record),failure,record,'save calibration window')
            cleanup(self.publish,failure,record,'save completed and partial calibration evidence')


def run(plan,output,deadline):
    import torch
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from .storage import check_write

    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    if tuple(plan['constants']['windows']) != WINDOWS:
        raise ValueError('The sixteen native calibration windows changed')
    native_plan = read(plan['native_plan_path'])
    if any(native_plan[name] != plan[name] for name in ('origin_state','origin_export','origin_config','schedule')):
        raise ValueError('Native trajectory inputs differ from sealed plan')
    receipt = dict(status='running',execution_commit=plan['execution_commit'],
        source_fitting_forwards=0,complete_response_generations=0,teacher_feature_preparation=0)
    ledger = Ledger(output/'PHYSICAL_WORK.jsonl',limits=dict(
        optimizer_step=294,response_forward=260,response_backward=260),deadline=deadline)
    trainer = instrumentation = controller = None
    began = time.perf_counter()
    try:
        weight_bytes = (Path(plan['origin_export'])/'model.safetensors').stat().st_size
        receipt['host_admission'] = host_admission(2*Path(plan['origin_state']).stat().st_size+5*weight_bytes)
        pin_retained_graph_runtime(output)
        started = time.perf_counter()
        trainer = make_trainer(native_plan,output/'trainer',ledger=ledger)
        receipt['loading_s'] = time.perf_counter()-started
        instrumentation = TrainingInstrumentation(trainer,ledger)
        with ledger.scope(category='startup'):
            receipt['startup'] = discarded_startup(trainer,plan['feature_paths'],output/'startup')
        receipt['qualification'] = qualify(trainer,plan['feature_paths'],output/'qualification')
        receipt['duration_admission'] = price_remaining(plan,receipt['qualification'],deadline)
        write_json(output/'DURATION_ADMISSION.json',receipt['duration_admission'])
        write_json(output/'QUALIFICATION_PASSED.json',receipt['qualification'])
        if receipt['duration_admission']['status'] != 'passed':
            raise WorkLimit('The full comparison cannot fit after bounded qualification')
        # Declare unavoidable native checkpoint/export publication and atomic
        # overlap before ordinary train_event is allowed to create these files.
        check_write(plan,2*Path(plan['origin_state']).stat().st_size+weight_bytes+(1 << 20),
                    'framework_native_publication_allowance')
        controller = WindowController(plan,output,ledger,receipt['qualification'],deadline)
        trainer.on_window = controller
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with ledger.scope(category='native'):
            result = trainer.train_event('event_007',[Path(path) for path in plan['feature_paths']],passes=2)
        receipt['native_event'] = result
        receipt['trajectory_with_calibration_s'] = time.perf_counter()-started
        receipt['final_native_parity'] = compare_final_native(trainer,plan['historical_native_state'],output/'FINAL_NATIVE_PARITY.json')
        if (len(controller.windows),len(controller.exports)) != (16,32):
            raise ValueError('Every prescribed window and both logical controls are required')
        if any(row['status'] != 'completed' for row in controller.windows):
            raise ValueError('An incomplete numerical bracket cannot enter scoring')
        actual_trials = sum(row['trials'] for row in controller.windows)
        if ledger.summary()['completed'].get('optimizer_step') != 128+6+actual_trials:
            raise ValueError('Native, qualification and scalar Adam applications do not reconcile')
        receipt.update(status='completed',global_step=trainer.global_step,presentations=trainer.presentations,
            scalar_calibration_applications=actual_trials,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        seal = dict(status='sealed',execution_commit=plan['execution_commit'],windows=list(WINDOWS),
                    exports=controller.exports,historical_E_reused=True,scoring_started=False)
        write_json(plan['control_seal'],seal)
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        write_json(output/'FAILED.json',receipt)
        raise
    finally:
        if controller is not None:
            controller.publish()
            receipt.update(probe_exports=controller.exports,windows=controller.windows,
                           calibration_trials=controller.trials,geometry=controller.geometry)
        receipt.update(wall_s=time.perf_counter()-began,work=ledger.summary())
        if 'qualification' not in receipt and (output/'qualification/RESULT.json').exists():
            receipt['qualification'] = read(output/'qualification/RESULT.json')
        write_json(output/'RESULT.json',receipt)
        if instrumentation is not None:
            instrumentation.close()
        if trainer is not None:
            trainer.close()
        ledger.close()
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--deadline',type=float,required=True)
    args = parser.parse_args()
    with termination_cleanup():
        result = run(read(args.plan),args.output,args.deadline)
    print(json.dumps(dict(status=result['status'],controls=len(result['probe_exports']))))
