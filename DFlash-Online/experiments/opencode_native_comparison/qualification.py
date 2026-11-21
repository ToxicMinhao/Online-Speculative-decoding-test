"""Prespecified disposable native controls; never production initialization."""
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback

from .contract import qualification_root
from online_feedback.storage import save_torch, write_json

CASES = {
    'O_CODE58': ('original', 'adapted', 'continued', 'reloaded', 'synthetic4096'),
    'AREPO_CODE58': ('original', 'adapted', 'continued', 'reloaded', 'synthetic4096'),
    'C_CODE58': ('bare_native', 'native', 'cap_zero', 'active', 'continued', 'reloaded', 'synthetic4096'),
}


def qualification_slots():
    rows = [dict(arm=arm, case=case, adam=1, response_forward=2,
                 response_backward=2, teacher_preparation=0, generation=0)
            for arm, cases in CASES.items() for case in cases]
    startup = [dict(arm=arm, case='qualified_disposable_startup', adam=0,
                    response_forward=2, response_backward=2,
                    teacher_preparation=0, generation=0) for arm in CASES]
    return dict(cases=rows, startup=startup, training_initialization=3,
        totals=dict(adam=17, response_forward=40, response_backward=40,
                    teacher_preparation=0, generation=0, training_initialization=3),
        full_context_fixture='two synthetic4096 packets; no teacher/model generation',
        pair_selection='sealed longest-input pair followed by sealed continuation pair')


def create_synthetic_packets(folder, *, hidden_width, target_width):
    """A disclosed uniform-teacher resource fixture, never serving feedback.

    Zero teacher features give a uniform distribution from the frozen bias-free
    head. Seeded auxiliary features exercise the whole4096-row storage/attention
    path. Labels are token zero, a deterministic maximizer of that distribution.
    """
    import torch
    from online_feedback.rng import preserve_rng
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f'synthetic4096_{index}.pt' for index in range(2)]
    manifest_path = folder / 'IDENTITIES.json'
    def identity(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
    if any(path.exists() for path in paths):
        if not manifest_path.exists() or not all(path.exists() for path in paths):
            raise ValueError('Incomplete prior synthetic fixture; preserve it without automatic regeneration')
        expected = json.loads(manifest_path.read_text())
        actual = dict(hidden_width=hidden_width, target_width=target_width,
                      files={path.name: identity(path) for path in paths})
        _equal(actual, expected, 'Retained synthetic resource fixture')
        return paths
    for index, path in enumerate(paths):
        with preserve_rng():
            generator = torch.Generator(device='cpu').manual_seed(20261120 + index)
            hidden = (torch.randn((1, 4096, hidden_width), generator=generator) * .01).to(torch.bfloat16)
            ids = torch.zeros((1, 4096), dtype=torch.long)
            mask = torch.zeros((1, 4096), dtype=torch.float32)
            mask[:, 2048:] = 1
            packet = dict(record_id=f'qualification_synthetic4096_{index}', prompt_length=2048,
                input_ids=ids, loss_mask=mask, hidden_states=hidden,
                target_last_hidden_states=torch.zeros((1, 4096, target_width), dtype=torch.bfloat16),
                label_binding=dict(schema='CODE_synthetic_resource_fixture_v1',
                    native_feedback=False, uniform_teacher=True, seed=20261120 + index))
            save_torch(path, packet)
    write_json(manifest_path, dict(hidden_width=hidden_width, target_width=target_width,
                                  files={path.name: identity(path) for path in paths}))
    return paths


def synthetic_metadata(trainer, paths):
    import torch
    from online_feedback.training import validate_feature
    result = []
    for index, path in enumerate(paths):
        raw = torch.load(path, map_location='cpu', weights_only=True)
        expected = dict(schema='CODE_synthetic_resource_fixture_v1', native_feedback=False,
                        uniform_teacher=True, seed=20261120 + index)
        if (raw['label_binding'] != expected or raw['input_ids'].shape != (1, 4096)
                or raw['prompt_length'] != 2048 or bool(raw['input_ids'].any())
                or raw['hidden_states'].dtype != torch.bfloat16
                or raw['target_last_hidden_states'].shape != (1, 4096, trainer.bundle.target_hidden_size)
                or raw['target_last_hidden_states'].dtype != torch.bfloat16
                or bool(raw['target_last_hidden_states'].any())):
            raise ValueError('The declared synthetic4096 resource fixture changed')
        metadata = validate_feature(raw, max_length=4096, hidden_width=trainer.hidden_width,
                                    vocab_size=trainer.vocab_size)
        result.append(dict(metadata, path=str(path)))
    return result


def _equal(actual, expected, label):
    from experiments.ok_continuation.resume import differences
    mismatch = differences(actual, expected)
    if mismatch:
        raise ValueError(label + ': ' + repr(mismatch[:12]))


def response_costs(rows, *, preclip_observer_s, block_observer_s):
    """Reuse native timers; do not charge endpoint hashing as training time."""
    response_s = [row['training_s'] for row in rows]
    native_total = sum(response_s)
    return dict(response_training_s=response_s, native_response_training_s=native_total,
        preclip_observer_s=preclip_observer_s, block_observer_s=block_observer_s,
        observer_adjusted_training_s=native_total-preclip_observer_s-block_observer_s,
        timing_scope='Native response train_step timers; subtract measured qualification observers. '
            'Excludes endpoint hashing, snapshot serialization and whole-case bookkeeping.')


def run_training_qualification(plan, arm, *, feature_paths, ledger):
    """Five/five/seven one-step trials, with three constructors in total.

    Each call owns a fresh process/runtime. The first two sealed feature paths
    are the first update; the next two are its continuation. Inputs are never
    selected or replaced in response to loss, length or performance outcomes.
    """
    import torch
    from experiments.functional_step_geometry.native_pilot import discarded_startup
    from experiments.functional_step_geometry.qualify import runtime_state
    from experiments.persistent_prefix_withholding.qualification import scientific_identity, tensor_identity
    from experiments.prefix_guard.qualify import pin_retained_graph_runtime
    from experiments.unchanged_c_mathqa_transfer.qualification import _native_forward_identity
    from experiments.unchanged_c_mathqa_transfer.state import validate_allocation_observation
    from experiments.ok_coding.training import _export_equal
    from online_feedback.adaflash_repo import LiteralSourceAdaFlashOPD, ReleasedAdaFlashOPD
    from online_feedback.rng import preserve_rng
    from online_feedback.training import SpecForgeEventTrainer
    from specforge.export.to_hf import export_to_hf
    from .training import CodeTrainer

    if arm not in CASES or len(feature_paths) != 4 or len(set(map(str, feature_paths))) != 4:
        raise ValueError('Require a declared arm and four sealed distinct qualification packets')
    root = qualification_root(plan) / arm
    root.mkdir(parents=True, exist_ok=False)
    destination = root / 'DETAILS.json'
    report = dict(status='running', arm=arm, execution_commit=plan['execution_commit'],
        binding=plan['binding'], physical_gpu_uuid=plan['devices']['B'], cases=[],
        feature_paths=list(map(str, feature_paths)), operation_counts={},
        exact_case_order=list(CASES[arm]), started_epoch=time.time(), phase='loading')
    write_json(destination, report)
    trainer = None
    failure = None

    def counted(kind, label, function, *args, **kwargs):
        counts = report['operation_counts'].setdefault(report['phase'], {})
        def execute():
            counts[kind + '_attempted'] = counts.get(kind + '_attempted', 0) + 1
            write_json(root / 'WORK.json', report['operation_counts'])
            value = function(*args, **kwargs)
            counts[kind + '_completed'] = counts.get(kind + '_completed', 0) + 1
            write_json(root / 'WORK.json', report['operation_counts'])
            return value
        return ledger.call({kind: 1}, label, execute)

    def wrap(owner, name, kind):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            return counted(kind, name, original, *args, **kwargs)
        setattr(owner, name, wrapped)

    try:
        report['runtime_compatibility'] = pin_retained_graph_runtime(root)
        trainer = counted('training_initialization', arm, CodeTrainer,
            plan['configs'][arm], root / 'trainer', arm=arm, binding=plan['binding'],
            original_config_text=Path(plan['original_config']).read_text(), qualification_only=True,
            expected_cuda_rng_states=plan.get('expected_cuda_rng_states', 1))
        report['initialization'] = trainer.save_initial(plan.get('initial_witness'))
        wrap(trainer.strategy, 'forward_loss', 'response_forward')
        wrap(trainer.backend, 'backward', 'response_backward')
        wrap(trainer.backend.optimizer.optimizer, 'step', 'adam')
        original = scientific_identity(trainer)
        report['origin_before'] = original
        report['phase'] = 'startup'
        report['startup'] = discarded_startup(trainer, feature_paths, root / 'startup')
        _equal(scientific_identity(trainer), original, 'Startup exact restoration')
        runtime = runtime_state(trainer.bundle.model)
        origin_path = root / 'disposable_origin.pt'
        trainer.save_qualification(origin_path)
        features = SpecForgeEventTrainer._prepare_event(trainer, 'qualification_features', feature_paths, 2)
        report['features'] = features

        def reset(label, cap=.25):
            if arm == 'AREPO_CODE58':
                trainer.bundle.model.point_loss_extension = ReleasedAdaFlashOPD()
            trainer.restore_qualification(origin_path, runtime=runtime)
            if arm == 'C_CODE58':
                trainer.qualification_allocation(cap)
            _equal(scientific_identity(trainer), original, 'Released scientific-state restoration')
            trainer._active_event = label
            (trainer.events_root / label).mkdir()
            trainer.bundle.model.begin_event()

        def pair(label, inputs):
            report['phase'] = label
            row = dict(case=label, status='attempted', checks='pending',
                       record_ids=[value['record_id'] for value in inputs], block_inputs=[])
            before_work = dict(trainer.work)
            report['cases'].append(row)
            write_json(destination, report)
            observer_s = 0.
            block_observer_s = 0.
            in_forward = False
            def preclip(current):
                nonlocal observer_s
                started = time.perf_counter()
                row['preclip'] = {name: tensor_identity(parameter.grad)
                    for name, parameter in current.bundle.draft_model.named_parameters() if parameter.requires_grad}
                write_json(destination, report)
                observer_s += time.perf_counter() - started
            def blocks(proposals, labels, weights, selection):
                nonlocal block_observer_s
                if in_forward:
                    started = time.perf_counter()
                    row['block_inputs'].append({key: tensor_identity(value) for key, value in
                        (('proposals', proposals), ('labels', labels), ('weights', weights), ('selection', selection))})
                    block_observer_s += time.perf_counter() - started
            forward = trainer.strategy.forward_loss
            def original_forward(*args, **kwargs):
                nonlocal in_forward
                in_forward = True
                try:
                    return forward(*args, **kwargs)
                finally:
                    in_forward = False
            trainer.strategy.forward_loss = original_forward
            trainer.step_observer = preclip
            trainer.bundle.model.block_observer = blocks
            began = time.perf_counter()
            try:
                rows = trainer.run_pair(inputs, 0, trainer.presentations)
            finally:
                trainer.strategy.forward_loss = forward
                trainer.step_observer = None
                trainer.bundle.model.block_observer = None
            row.update(status='completed', wall_s=time.perf_counter() - began,
                **response_costs(rows, preclip_observer_s=observer_s, block_observer_s=block_observer_s),
                forward=_native_forward_identity(rows),
                clip=trainer.window_records[-1]['gradient_norm'], endpoint=scientific_identity(trainer),
                allocation=[value['response_observation'].get('prefix_allocation') for value in rows],
                full_sequence_lengths=[value['length'] for value in inputs],
                anchor_counts=[len(value['sampled_anchors'][0]['positions'][0]) for value in rows])
            row['physical_work'] = {key: value - before_work.get(key, 0) for key, value in trainer.work.items()}
            write_json(destination, report)
            return row

        comparison_keys = ('forward', 'preclip', 'clip', 'endpoint', 'block_inputs')
        baseline = None
        initial_cases = CASES[arm][:-3]
        for label in initial_cases:
            cap = None if label in ('bare_native', 'native') else (0. if label == 'cap_zero' else .25)
            reset(label, cap)
            wrapped = trainer._native_forward
            if label in ('original', 'bare_native'):
                trainer._native_forward = trainer._ordinary_forward
                if arm == 'AREPO_CODE58':
                    trainer.bundle.model.point_loss_extension = LiteralSourceAdaFlashOPD()
            try:
                row = pair(label, features[:2])
            finally:
                trainer._native_forward = wrapped
                if arm == 'AREPO_CODE58':
                    trainer.bundle.model.point_loss_extension = ReleasedAdaFlashOPD()
            if baseline is None:
                baseline = row
            elif label != 'active':
                for key in comparison_keys:
                    _equal(row[key], baseline[key], label + ' exact native ' + key)
                row['native_equivalence'] = 'exact'
            else:
                _equal(row['block_inputs'], baseline['block_inputs'], 'C unchanged native block conditioning')
                for summary in row['allocation']:
                    validate_allocation_observation({'prefix_allocation': summary})
                for actual, expected in zip(row['forward'], baseline['forward']):
                    unchanged = lambda values: {key: value for key, value in values.items() if key != 'main_loss_numerator'}
                    _equal(unchanged(actual['sums']), unchanged(expected['sums']), 'C unchanged support/denominator')
                row['changed_coefficient_occurrences'] = sum(value['changed_positions'] for value in row['allocation'])
                row['nonzero_intervention_observed'] = row['changed_coefficient_occurrences'] > 0
            row['checks'] = 'passed'
            write_json(destination, report)

        report['phase'] = 'serialization'
        active_path = root / 'disposable_active.pt'
        trainer.save_qualification(active_path)
        saved_runtime = runtime_state(trainer.bundle.model)
        if arm == 'C_CODE58':
            export = root / 'active_export'
            model_config = Path(trainer.cfg.model.draft_model_config)
            if model_config.is_dir():
                model_config /= 'config.json'
            with preserve_rng():
                export_to_hf(str(active_path), str(model_config), str(export),
                    embedding_source=trainer.cfg.model.target_model_path,
                    embedding_key=trainer.cfg.model.embedding_key)
                _export_equal(torch.load(active_path, map_location='cpu', weights_only=False, mmap=True), export)
            report['active_C_export'] = str(export)
        continued = pair('continued', features[2:])
        report['restore'] = trainer.restore_qualification(active_path, runtime=saved_runtime)
        reloaded = pair('reloaded', features[2:])
        for key in (*comparison_keys, 'allocation'):
            _equal(reloaded[key], continued[key], 'Exact own-state next update ' + key)
        continued['checks'] = reloaded['checks'] = 'passed'

        reset('synthetic4096')
        if getattr(trainer.bundle.model.lm_head, 'bias', None) is not None:
            raise ValueError('Uniform synthetic teacher requires the pinned bias-free vocabulary head')
        synthetic_paths = create_synthetic_packets(qualification_root(plan) / 'synthetic4096',
            hidden_width=trainer.hidden_width, target_width=trainer.bundle.target_hidden_size)
        synthetic = pair('synthetic4096', synthetic_metadata(trainer, synthetic_paths))
        synthetic.update(checks='passed', resource_fixture_only=True,
                         native_canonical_claim=False, targets='uniform distribution from zero final hidden')
        report['synthetic_paths'] = list(map(str, synthetic_paths))
        reset('restored_origin')
        report['origin_after'] = scientific_identity(trainer)
        _equal(report['origin_after'], original, 'Final disposable origin restoration')
        report.update(status='passed', phase='complete', exact_native_compatibility=True,
            exact_own_resume=True, origin_restored=True,
            core_adam=len(CASES[arm]), core_response_forward_backward_each=2 * len(CASES[arm]),
            startup_response_forward_backward_each=2,
            disposable_states=[str(origin_path), str(active_path)])
    except BaseException as error:
        failure = error
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        write_json(destination, report)  # Preserve primary failure before cleanup.
        raise
    finally:
        if trainer is not None:
            report['physical_work'] = dict(trainer.work)
        report.update(wall_s=time.time() - report['started_epoch'],
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        write_json(destination, report)
        if trainer is not None:
            try:
                trainer.close()
            except BaseException as error:
                report['cleanup_error'] = repr(error)
                write_json(destination, report)
                if failure is None:
                    raise
                failure.add_note('Cleanup also failed: ' + repr(error))
    return report
