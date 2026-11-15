"""Six applications: one ordinary pair and five cached-gradient native controls."""
from pathlib import Path
import time
import traceback

from experiments.ok_continuation.resume import differences
from experiments.recent_repair_retention.training import schedule_adapter
from experiments.threatened_repair_replay.state import capture_state, restore_state, verify_state, tensor_change
from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.storage import write_json
from .calibration import ordinary_pair, cached_trial

CASES = (('N', None), ('alpha_1', 1.), ('alpha_1_repeat', 1.),
         ('alpha_0', 0.), ('alpha_half', .5), ('alpha_half_repeat', .5))


def qualify(trainer, feature_paths, output):
    """Use the fixed first pair, never select by support or observed benefit."""
    import torch
    from experiments.functional_step_geometry.qualify import cleanup

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    origin = capture_state(trainer)
    started, failure = time.perf_counter(), None
    report = dict(status='running', cases=[], optimizer_steps=0,
        native_response_presentations=0, response_forwards=0, response_backwards=0,
        complete_response_generations=0, cached_gradient_response_work=0,
        startup='separate; at most two backwards and zero Adam applications')
    native = gradients = native_update = half_weights = None
    on_cuda = next(trainer.bundle.draft_model.parameters()).device.type == 'cuda'
    try:
        if trainer.on_window is not None or getattr(trainer.bundle.model, 'native_prefix_roles', None) is not None:
            raise ValueError('Ordinary qualification must have no intervention controller')
        features = trainer._prepare_event('event_007', feature_paths, 2)
        trainer._active_event = 'qualification_prepared'
        trainer._presentation_schedule = PresentationSchedule(schedule_adapter(trainer.schedule, features),
            [row['record_id'] for row in features], 2, features=features)
        trainer.bundle.model.begin_event()
        order = torch.randperm(len(features)).tolist()
        pair = [features[index] for index in order[:2]]
        pre = capture_state(trainer)
        report['fixed_pair'] = [row['record_id'] for row in pair]
        for field in ('draft_state_dict','backend'):
            if differences(pre['scientific'][field],origin['scientific'][field]):
                raise ValueError('Qualification preparation changed weights or native optimizer')
            origin['scientific'][field] = pre['scientific'][field]
        for index, (name,alpha) in enumerate(CASES):
            directory = output/name
            directory.mkdir()
            case = dict(case=name,alpha=alpha,status='running')
            report['cases'].append(case)
            write_json(output/'RESULT.json',report)
            restore_state(trainer,pre)
            case['prestate'] = verify_state(trainer,pre,directory/'PRE_STATE.json')
            if on_cuda:
                torch.cuda.reset_peak_memory_stats()
            mismatches = []
            if name == 'N':
                result = ordinary_pair(trainer,pair,0,0,category='qualification',label='N')
                native = capture_state(trainer)
                gradients = result['preclip_gradients']['native']
                native_update = result['update']
                case.update(update=native_update,
                    displacement=tensor_change(pre['scientific']['draft_state_dict'],native['scientific']['draft_state_dict']))
                report['optimizer_steps'] += 1
                report['native_response_presentations'] += len(result['rows'])
                report['response_forwards'] += len(result['rows'])
                report['response_backwards'] += len(result['rows'])
                case['native_gradient_present'] = [key for key,value in gradients.items() if value is not None]
                case['native_gradient_none'] = [key for key,value in gradients.items() if value is None]
            else:
                weights,update = cached_trial(trainer,pre,native,gradients,native_update,alpha,
                    window=1,trial_index=index,output=directory,category='qualification')
                report['optimizer_steps'] += update['optimizer_applications']
                case.update(update=update,displacement=update['displacement'])
                if alpha == 1.:
                    mismatches += differences(weights,native['scientific']['draft_state_dict'],'alpha1_native_weights')
                elif alpha == 0.:
                    mismatches += differences(weights,pre['scientific']['draft_state_dict'],'alpha0_pre_weights')
                    if update['displacement']['difference_norm'] != 0.:
                        mismatches.append('alpha0_nonzero_displacement')
                elif name == 'alpha_half':
                    half_weights = weights
                else:
                    mismatches += differences(weights,half_weights,'half_repeat_weights')
                if name != 'alpha_half':
                    del weights
            case.update(status='failed' if mismatches else 'passed',mismatches=mismatches[:64])
            if on_cuda:
                case.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(directory/'RESULT.json',case)
            write_json(output/'RESULT.json',report)
            if mismatches:
                raise ValueError('Native cached-gradient qualification differs: '+str(mismatches[:8]))
        if (report['optimizer_steps'],report['native_response_presentations']) != (6,2):
            raise ValueError('Qualification must use six Adam applications and only two full presentations')
        report['status'] = 'passed'
    except BaseException as error:
        failure = error
        if report['cases'] and report['cases'][-1]['status'] == 'running':
            case = report['cases'][-1]
            case.update(status='failed',error=repr(error))
            trial_path = output/case['case']/f'TRIAL_{len(report["cases"])-1:02d}.json'
            if trial_path.exists():
                import json
                case['update'] = json.loads(trial_path.read_text())
                report['optimizer_steps'] += case['update']['optimizer_applications']
            write_json(output/case['case']/'RESULT.json',case)
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        write_json(output/'FAILED.json',report)
        raise
    finally:
        cleanup(lambda: restore_state(trainer,origin),failure,report,'restore qualification origin')
        cleanup(lambda: verify_state(trainer,origin,output/'RESTORED_ORIGIN.json'),failure,report,'verify qualification origin')
        report['wall_s'] = time.perf_counter()-started
        if trainer.ledger is not None:
            report['physical_work_at_exit'] = trainer.ledger.summary()
        cleanup(lambda: write_json(output/'RESULT.json',report),failure,report,'save qualification evidence')
    return report
