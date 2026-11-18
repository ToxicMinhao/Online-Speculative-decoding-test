"""Six released-start disposable updates; no production event or RNG fork."""
from pathlib import Path
import resource
import time
import traceback

from experiments.unchanged_c_mathqa_transfer.qualification import _assert_equal, _native_forward_identity
from online_feedback.storage import write_json
from experiments.unchanged_c_mathqa_transfer.state import validate_allocation_observation

CASES = ('bare_native', 'native', 'cap_zero', 'active', 'continued_next_pair', 'reloaded_next_pair')
SCIENTIFIC_FIELDS = ('forward', 'preclip', 'clip', 'endpoint', 'allocation', 'block_inputs')


def run_training_qualification(plan, slot, *, feature_paths=None, ledger):
    import torch
    from experiments.functional_step_geometry.qualify import runtime_state
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.persistent_prefix_withholding.qualification import scientific_identity, tensor_identity, tree_identity
    from online_feedback.rng import capture_rng, preserve_rng
    from online_feedback.training import SpecForgeEventTrainer
    from specforge.export.to_hf import export_to_hf
    from .training import FullCTrainer

    features_paths = plan['qualification_feature_paths'] if feature_paths is None else feature_paths
    if len(features_paths) != 4 or len(set(map(str, features_paths))) != 4:
        raise ValueError('Qualification requires the four prespecified distinct feature packets')
    base = Path(plan['root'])/'qualification'/f'training_{slot}'
    base.mkdir(parents=True, exist_ok=True)
    destination = base/'DETAILS.json'
    if destination.exists():
        raise FileExistsError('Never overwrite qualification evidence')
    report = dict(status='running', dataset='gsm8k', slot=slot,
        physical_gpu_uuid=plan['devices'][slot], binding=plan['binding'],
        execution_commit=plan['execution_commit'], cases=[], planned_adam=6,
        planned_core_forward_backward_each=12, startup_forward_backward_each=2,
        started_epoch=time.time(), operation_counts={}, phase='loading')
    trainer, failure = None, None
    write_json(destination, report)

    def counted(kind, label, function, *args, **kwargs):
        counts = report['operation_counts'].setdefault(report['phase'], {})
        def execute():
            counts[kind+'_attempted'] = counts.get(kind+'_attempted',0)+1
            write_json(base/'WORK.json', report['operation_counts'])
            value = function(*args, **kwargs)
            counts[kind+'_completed'] = counts.get(kind+'_completed',0)+1
            write_json(base/'WORK.json', report['operation_counts'])
            return value
        return ledger.call({kind:1}, label, execute)

    def wrap(owner, name, kind):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            return counted(kind, name, original, *args, **kwargs)
        setattr(owner, name, wrapped)

    def source_identity():
        source = torch.load(trainer.output_root/'initial_state.pt', map_location='cpu', weights_only=False, mmap=True)
        return dict(weights=tree_identity(source['draft_state_dict']),
            optimizer=tree_identity(source['backend']), rng=tree_identity(source['rng']))

    try:
        report['runtime_compatibility'] = pin_retained_graph_runtime(base)
        loading_began = time.perf_counter()
        trainer = counted('training_initialization', 'load', FullCTrainer,
            plan['config_path'], base/'trainer', binding=plan['binding'],
            original_config_text=Path(plan['original_config']).read_text(), qualification_only=True)
        report['loading_s'] = time.perf_counter()-loading_began
        checking_began = time.perf_counter()
        report['initialization'] = trainer.save_initial(plan['initial_witness'])
        report['initialization_check_s'] = time.perf_counter()-checking_began
        wrap(trainer.strategy, 'forward_loss', 'response_forward')
        wrap(trainer.backend, 'backward', 'response_backward')
        wrap(trainer.backend.optimizer.optimizer, 'step', 'adam')
        report['origin_before'] = source_identity()
        original = scientific_identity(trainer)
        report['phase'] = 'startup'
        report['startup'] = discarded_startup(trainer, features_paths, base/'startup')
        report['startup_s'] = report['startup']['wall_s']
        _assert_equal(scientific_identity(trainer), original, 'Startup exact origin restoration')
        runtime = runtime_state(trainer.bundle.model)
        report['stochastic_consumers'] = dict(
            response_shuffle='torch.randperm on CPU', anchors='qualified native training-device random sampler',
            positive_dropout_modules=[dict(name=name, p=module.p) for name,module in trainer.bundle.model.named_modules()
                                      if isinstance(module, torch.nn.Dropout) and module.p > 0],
            attention_dropout=[dict(name=name, value=float(module.attention_dropout))
                               for name,module in trainer.bundle.model.named_modules()
                               if hasattr(module, 'attention_dropout')],
            constructor_seed17_retained=True, explicit_continuation_seed=None)
        features = SpecForgeEventTrainer._prepare_event(trainer, 'qualification_features', features_paths, 2)

        def reset(cap, label, *, draw_order=True):
            trainer.reset_qualification(relative_cap=cap, runtime=runtime)
            _assert_equal(scientific_identity(trainer), original, 'Released scientific state before disposable update')
            trainer._active_event = label
            (trainer.events_root/label).mkdir(exist_ok=False)
            trainer.bundle.model.begin_event()
            if not draw_order:
                return None
            before = tree_identity(capture_rng())
            order = torch.randperm(len(features)).tolist()
            report.setdefault('sampling', {})[label] = dict(reseeded=False, rng_before=before,
                rng_after_permutation=tree_identity(capture_rng()), order=order,
                record_order=[features[index]['record_id'] for index in order])
            return order

        def run_pair(label, pair, ordinal):
            row = dict(case=label, status='attempted', pair=[x['record_id'] for x in pair], checks='pending', block_inputs=[])
            report['phase'] = label
            report['cases'].append(row)
            write_json(destination, report)
            observer_s, block_observer_s = 0., 0.
            def observe(current):
                nonlocal observer_s
                began = time.perf_counter()
                row['preclip'] = {name:tensor_identity(parameter.grad) for name,parameter in current.bundle.draft_model.named_parameters() if parameter.requires_grad}
                write_json(destination, report)
                observer_s += time.perf_counter()-began
            in_forward = False
            def block(proposals, labels, weights, selection):
                nonlocal block_observer_s
                if in_forward:
                    began = time.perf_counter()
                    row['block_inputs'].append({key:tensor_identity(value) for key,value in (
                        ('proposals',proposals),('labels',labels),('weights',weights),('selection',selection))})
                    block_observer_s += time.perf_counter()-began
            forward = trainer.strategy.forward_loss
            def original_forward(*args, **kwargs):
                nonlocal in_forward
                in_forward = True
                try:
                    return forward(*args, **kwargs)
                finally:
                    in_forward = False
            model = trainer.bundle.model
            previous = getattr(model, 'block_observer', None)
            if previous is not None:
                raise ValueError('Qualification block observer is already occupied')
            trainer.strategy.forward_loss, trainer.step_observer, model.block_observer = original_forward, observe, block
            began = time.perf_counter()
            try:
                rows = trainer.run_pair(pair, 0, ordinal)
            finally:
                trainer.strategy.forward_loss, trainer.step_observer, model.block_observer = forward, None, previous
            wall_s = time.perf_counter()-began
            row.update(status='completed', wall_s=wall_s,
                preclip_observer_s=observer_s, block_observer_s=block_observer_s,
                pair_wall_excluding_observers_s=wall_s-observer_s-block_observer_s,
                forward=_native_forward_identity(rows), clip=trainer.window_records[-1]['gradient_norm'],
                allocation=[r['response_observation']['prefix_allocation'] for r in rows], endpoint=scientific_identity(trainer))
            write_json(destination, report)
            return row

        baseline = None
        for label,cap in (('bare_native',None),('native',None),('cap_zero',0.),('active',.25)):
            order = reset(cap,label)
            wrapper = trainer._native_forward
            if label == 'bare_native':
                trainer._native_forward = trainer._ordinary_forward
            try:
                row = run_pair(label,[features[i] for i in order[:2]],0)
            finally:
                trainer._native_forward = wrapper
            if baseline is None:
                baseline = row
            elif label in ('native','cap_zero'):
                for key in ('forward','preclip','clip','endpoint','block_inputs'):
                    _assert_equal(row[key],baseline[key],label+' native equivalence '+key)
                _assert_equal(report['sampling'][label]['order'], report['sampling']['bare_native']['order'], 'Natural permutation equality')
                row['native_equivalence'] = 'exact'
            else:
                _assert_equal(row['block_inputs'],baseline['block_inputs'],'Active native block inputs')
                for actual,expected in zip(row['forward'],baseline['forward']):
                    unchanged = lambda x: {k:v for k,v in x.items() if k != 'main_loss_numerator'}
                    _assert_equal(unchanged(actual['sums']),unchanged(expected['sums']),'Native support/denominator/first-error')
                for summary in row['allocation']:
                    validate_allocation_observation({'prefix_allocation':summary})
                movement = sum(x['changed_positions'] for x in row['allocation'])
                changed = sum(value != baseline['endpoint']['weights'][name] for name,value in row['endpoint']['weights'].items())
                row.update(changed_coefficient_occurrences=movement, changed_parameter_tensors=changed)
                if movement <= 0 or changed <= 0:
                    raise ValueError('Fixed qualification batch produced no actual intervention; no search allowed')
            row['checks'] = 'passed'
            write_json(destination, report)

        report['phase'] = 'save_reload'
        snapshot, export = base/'active_state.pt', base/'active_export'
        report['snapshot'] = trainer.save_qualification(snapshot)
        saved_runtime = runtime_state(trainer.bundle.model)
        config = Path(trainer.cfg.model.draft_model_config)
        if config.is_dir():
            config /= 'config.json'
        with preserve_rng():
            export_to_hf(str(snapshot),str(config),str(export),embedding_source=trainer.cfg.model.target_model_path,
                         embedding_key=trainer.cfg.model.embedding_key)
        report.update(export=str(export),snapshot_path=str(snapshot),draft_config=str(config),
            embedding_source=trainer.cfg.model.target_model_path,embedding_key=trainer.cfg.model.embedding_key)
        pair = [features[i] for i in order[2:4]]
        continued = run_pair('continued_next_pair',pair,2)
        report['restore'] = trainer.restore_qualification(snapshot,runtime=saved_runtime)
        before_rng = tree_identity(capture_rng())
        _assert_equal(tree_identity(capture_rng()),before_rng,'Reload leaves restored RNG intact')
        replay = run_pair('reloaded_next_pair',pair,2)
        for key in SCIENTIFIC_FIELDS:
            _assert_equal(replay[key],continued[key],'Own-state next update '+key)
        continued['checks'] = replay['checks'] = 'passed'
        reset(.25,'restored_origin',draw_order=False)
        _assert_equal(scientific_identity(trainer),original,'Final original restoration')
        report['origin_after'] = source_identity()
        _assert_equal(report['origin_after'],report['origin_before'],'Origin immutability')
        report.update(status='running',phase='cpu_export_pending',optimizer_applications=6,response_presentations=12,
            exact_native_wrapper=True,exact_native_cap0=True,actual_intervention=True,
            exact_own_resume=True,origin_restored=True)
    except BaseException as error:
        failure = error
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        if trainer is not None:
            trainer.failed = True
        raise
    finally:
        if trainer is not None:
            trainer.step_observer = None
            report['physical_work'] = dict(trainer.work)
        report.update(peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0,
            peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()) if torch.cuda.is_initialized() else 0,
            host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            wall_s=time.time()-report['started_epoch'])
        write_json(destination,report)
        if trainer is not None:
            try:
                trainer.close()
            except BaseException as error:
                report['cleanup_error'] = repr(error)
                if failure is None:
                    report.update(status='failed',error=repr(error))
                write_json(destination,report)
                if failure is None:
                    raise
                failure.add_note('Cleanup also failed: '+repr(error))
    # Release the training runtime before the CPU-only author exporter loads a
    # complete recovery; no model forward or optimizer initialization is added.
    trainer = None
    import gc
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    from experiments.c_paired_confirmation.export import qualify_cpu_export
    reconstruction = base/'cpu_reconstructed_export'
    try:
        report['cpu_export_qualification'] = qualify_cpu_export(
            report['snapshot_path'], report['draft_config'], report['export'], reconstruction,
            embedding_source=report['embedding_source'], embedding_key=report['embedding_key'],
            receipt_path=base/'CPU_EXPORT_QUALIFICATION.json')
        disposable_files = [report['snapshot_path'], report['initialization']['state_path']]
        for folder in (Path(report['export']), reconstruction):
            disposable_files.extend(str(path) for path in sorted(folder.rglob('*')) if path.is_file())
        report.update(status='passed',phase='complete',cpu_reconstruction_verified=True,
            cpu_export=report['cpu_export_qualification'],disposable_payload_files=disposable_files,
            disposable_payload_paths=[report['initialization']['state_path'],report['snapshot_path'],report['export'],str(reconstruction)])
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        write_json(destination,report)
        raise
    report['wall_s'] = time.time()-report['started_epoch']
    write_json(destination,report)
    return report
