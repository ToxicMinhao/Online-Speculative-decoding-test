"""Exactly one disposable native Adam update and export/reload per arm."""
from pathlib import Path
import resource
import time
import traceback

from online_feedback.storage import save_torch, write_json


def run_training_qualification(plan, arm, *, feature_paths, ledger=None):
    """Use two declared real packets; production origins/RNG live elsewhere.

    The caller subsequently serves the returned export through the packaged
    evaluator and verifies the frozen-F canonical on each physical card.
    A training receipt alone deliberately does not claim serving qualification.
    """
    import torch
    from specforge.export.to_hf import export_to_hf
    from experiments.full_k_gsm8k import training as native
    from experiments.ok_coding.training import _export_equal
    from experiments.ok_coding.storage import identity
    from experiments.ok_continuation.resume import differences
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from online_feedback.rng import preserve_rng
    from online_feedback.training import SpecForgeEventTrainer
    from .training import PilotTrainer, ARMS, discarded_startup

    if arm not in ARMS or len(feature_paths) != 2 or len(set(map(str, feature_paths))) != 2:
        raise ValueError('One disposable update requires exactly two distinct real feedback packets')
    root = Path(plan['root']) / 'qualification' / arm
    root.mkdir(parents=True, exist_ok=False)
    report_path = root / 'TRAINING_QUALIFICATION.json'
    report = dict(status='running', arm=arm, binding=plan['binding'],
        execution_commit=plan['execution_commit'], started_epoch=time.time(), counts={},
        serving_reload_status='pending_packaged_evaluator_and_canonical_check',
        feature_identities=[identity(path) for path in feature_paths])
    trainer, failed = None, False

    def counted(kind, label, function, *args, **kwargs):
        def execute():
            report['counts'][kind + '_attempted'] = report['counts'].get(kind + '_attempted', 0) + 1
            write_json(report_path, report)
            result = function(*args, **kwargs)
            report['counts'][kind + '_completed'] = report['counts'].get(kind + '_completed', 0) + 1
            write_json(report_path, report)
            return result
        return ledger.call({kind: 1}, label, execute) if ledger is not None else execute()

    def wrap(owner, name, kind):
        original = getattr(owner, name)
        setattr(owner, name, lambda *args, **kwargs: counted(kind, arm + ':' + name, original, *args, **kwargs))

    write_json(report_path, report)
    try:
        report['runtime_compatibility'] = pin_retained_graph_runtime(root)
        report['runtime_compatibility'].update(applied_before='pilot trainer construction and compile startup',
            scope='this disposable ' + arm + ' qualification process')
        write_json(root / 'RUNTIME_COMPATIBILITY.json', report['runtime_compatibility'])
        trainer = counted('training_initialization', arm, PilotTrainer,
            plan['configs'][arm], root / 'trainer', arm=arm, binding=plan['binding'],
            original_config_text=Path(plan['original_config']).read_text(),
            maximum_events=plan.get('maximum_events', 2), qualification_only=True,
            expected_cuda_rng_states=plan.get('expected_cuda_rng_states', 1))
        report['initialization'] = trainer.save_initial(plan.get('initial_witness'))
        report['startup'] = discarded_startup(trainer, feature_paths, root / 'startup')
        report['startup_counts'] = dict(response_forward_completed=2,
                                       response_backward_completed=2, adam_completed=0)
        features = SpecForgeEventTrainer._prepare_event(trainer, 'disposable', feature_paths, 1)
        trainer._active_event = 'disposable'
        (trainer.events_root / trainer._active_event).mkdir()
        wrap(trainer.strategy, 'forward_loss', 'response_forward')
        wrap(trainer.backend, 'backward', 'response_backward')
        wrap(trainer.backend.optimizer.optimizer, 'step', 'adam')
        ordinary_step = trainer._native_step
        def inspected_step():
            model = trainer.bundle.model
            draft_gradients = [parameter.grad for parameter in model.draft_model.parameters()
                               if parameter.requires_grad]
            if not draft_gradients or not all(value is not None and bool(torch.isfinite(value).all())
                                              for value in draft_gradients):
                raise ValueError('Missing or nonfinite native draft gradients before clipping')
            if any(parameter.requires_grad or parameter.grad is not None
                   for module in (model.lm_head, model.embed_tokens) for parameter in module.parameters()):
                raise ValueError('Frozen head/embedding received training gradients')
            report['gradient_contract'] = dict(draft_gradient_tensors=len(draft_gradients),
                frozen_head_embedding=True, extra_backward_calls=0)
            return ordinary_step()
        trainer._native_step = inspected_step
        trainer.bundle.model.begin_event()
        try:
            rows = trainer.run_pair(features)
        finally:
            report['objective_work'] = trainer.bundle.model.end_event()
        if trainer.global_step != 1 or trainer.presentations != 2 or trainer.core.accumulation_remainder:
            raise ValueError('Disposable qualification exceeded its one-step budget')
        observations = [row['response_observation'] for row in rows]
        active = sum(row['sum_metrics'].get('repair_pair_active_rows', 0) for row in observations)
        auxiliary = sum(row['ratio_metrics'].get('repair_pair_auxiliary', [0, 1])[0]
                        for row in observations)
        if any(('repair_pair_auxiliary' in row['ratio_metrics']) != (arm == 'PAIR')
               for row in observations):
            raise ValueError('Auxiliary flag did not reach the real training path')
        report.update(records=rows, windows=trainer.window_records,
            auxiliary_active_rows=active, auxiliary_numerator=auxiliary,
            intervention_activity='active' if active > 0 and auxiliary > 0 else 'inactive',
            training_s=sum(row['training_s'] for row in rows))
        state_path, export = root / 'disposable_state.pt', root / 'export'
        with preserve_rng():
            state = native.live_state(trainer)
            save_torch(state_path, state)
            config = Path(trainer.cfg.model.draft_model_config)
            config = config / 'config.json' if config.is_dir() else config
            export_to_hf(str(state_path), str(config), str(export),
                embedding_source=trainer.cfg.model.target_model_path,
                embedding_key=trainer.cfg.model.embedding_key)
            saved = torch.load(state_path, map_location='cpu', weights_only=False, mmap=True)
            _export_equal(saved, export)
            # Load the actual exported tensors, then reload moments/counters/RNG.
            # No second optimizer step or extra response forward is performed.
            from safetensors.torch import load_file
            tensors = load_file(str(export / 'model.safetensors'))
            names = saved['draft_state_dict'].keys()
            trainer.bundle.draft_model.load_state_dict({key: tensors[key] for key in names}, strict=True)
            trainer._load_state(saved)
            trainer.check_installed_arm()
            if differences(native.live_state(trainer), saved):
                raise ValueError('Disposable serialized native state did not reload exactly')
        report.update(status='passed', export_path=str(export), state_path=str(state_path),
            state_identity=identity(state_path), export_identity=identity(export / 'model.safetensors'),
            exact_export_and_state_reload=True, global_step=trainer.global_step,
            presentations=trainer.presentations, automatic_retry=False)
        expected = dict(training_initialization_completed=1, response_forward_completed=2,
                        response_backward_completed=2, adam_completed=1)
        if any(report['counts'].get(key) != value for key, value in expected.items()):
            raise ValueError('Qualification native work counts differ from the bounded plan')
    except BaseException as error:
        failed = True
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        if trainer is not None:
            trainer.failed = True
        write_json(report_path, report)
        raise
    finally:
        if trainer is not None:
            report['physical_work'] = dict(trainer.work)
        report.update(wall_s=time.time() - report['started_epoch'],
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        write_json(report_path, report)
        if trainer is not None:
            try:
                trainer.close()
            except BaseException as error:
                report['cleanup_error'] = repr(error)
                write_json(report_path, report)
                if not failed:
                    raise
    return report
