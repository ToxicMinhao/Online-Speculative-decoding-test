"""One own-feedback event per arm, using the unchanged two-card native lifecycle."""

from pathlib import Path
import shutil
import time

from online_feedback.controller import OnlineController
from online_feedback.sglang_backend import SGLangDFlashBackend
from online_feedback.storage import write_json
from experiments.full_k_gsm8k.campaign import online_record_check
from experiments.kplus_full_gsm8k.storage import retire_redundant_features
from experiments.fresh_confirmation.protocol import publication_errors, weight_check_errors
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from experiments.ok_coding.storage import identity
from experiments.opencode_native_comparison.training import (
    recovery_size_plan,
    validate_recovery_file,
)
from .resources import read, boundary, Ledger
from .training import PilotTrainer, discarded_startup
from .witness import freeze_witness, evaluate_witness


def create_backend(plan, path, card='A'):
    return SGLangDFlashBackend(
        plan['target'],
        plan['draft'],
        worker_python=plan['worker_python'],
        output_root=path,
        serving_gpu=plan['devices'][card],
        context_limit=4096,
        block_size=16,
        worker_module='experiments.persistent_prefix_withholding.worker',
    )


def check_publication(backend, checkpoint, destination):
    publication = backend.publish(checkpoint)
    weights = backend.check_weights(checkpoint)
    errors = publication_errors(publication, str(checkpoint)) + weight_check_errors(
        weights, str(checkpoint)
    )
    write_json(destination, dict(publication=publication, weights=weights, errors=errors))
    if errors:
        raise ValueError('Native endpoint activation failed: ' + repr(errors))
    return publication


def qualification_reuse(plan):
    """Consume authenticated historical qualification without claiming new work."""
    receipt = read(plan['qualification_reuse'])
    if (
        receipt.get('status') != 'reused_passed'
        or receipt.get('qualified_execution_commit') != '79b8c076610e9b4d310169b3891cb76ac6d0f9bb'
        or receipt.get('optimizer_steps_executed') != 0
    ):
        raise ValueError(
            'Production requires an explicit authenticated qualification-reuse receipt'
        )
    return receipt


def production_paths(plan, arm):
    """Resolve an admitted real feature location, never an invented offload marker."""
    root = Path(plan['root'])
    if plan.get('execution_mode') != 'one_event_production':
        # Retain the historical wrapper's gate for its original plan shape.
        if (
            arm == 'PAIR'
            and read(root / 'BASE_FEATURE_OFFLOAD.json')['status'] != 'verified_retired'
        ):
            raise ValueError('BASE feature offload must complete before PAIR allocation')
        if read(root / 'QUALIFICATION.json')['status'] != 'passed':
            raise ValueError('Complete bounded native qualification is required')
        return root, root / 'arms' / arm / 'event_001', root / 'arms' / arm / 'native'
    if arm not in ('BASE', 'PAIR') or plan.get('maximum_events') != 1:
        raise ValueError('Production requires exactly one BASE/PAIR event')
    if plan.get('feature_strategy') not in ('local_single', 'local_split'):
        raise ValueError('No verified offload implementation is admitted by this coordinator')
    qualification_reuse(plan)
    feature_root = Path(plan['feature_roots'][arm])
    if not feature_root.is_absolute():
        raise ValueError('The sealed feature root must be absolute')
    event_root = feature_root / 'event_001'
    return feature_root, event_root, event_root / 'native'


def retain_witness_features(paths, witness_root):
    """Keep eight exact BASE packets on the main filesystem for both endpoints."""
    if len(paths) < 8:
        raise ValueError('The fixed witness requires the first eight BASE responses')
    destination = Path(witness_root) / 'features'
    destination.mkdir(parents=True, exist_ok=False)
    retained, rows = [], []
    for ordinal, source in enumerate(paths[:8]):
        source = Path(source)
        target = destination / source.name
        expected = identity(source)
        shutil.copy2(source, target)
        if identity(target) != expected:
            raise ValueError('A retained witness packet differs from its own BASE feature')
        sidecar = source.with_suffix('.json')
        if sidecar.exists():
            shutil.copy2(sidecar, target.with_suffix('.json'))
            if identity(sidecar) != identity(target.with_suffix('.json')):
                raise ValueError('A retained witness sidecar differs')
        retained.append(target)
        rows.append(
            dict(ordinal=ordinal, source=str(source), retained=str(target), identity=expected)
        )
        write_json(Path(witness_root) / 'FEATURE_RETENTION.json', dict(status='copying', rows=rows))
    write_json(Path(witness_root) / 'FEATURE_RETENTION.json', dict(status='verified', rows=rows))
    return retained


def run_arm(plan, arm):
    if not plan.get('production_admitted', False):
        raise RuntimeError('Production resource admission is blocked; qualification-only plan')
    root = Path(plan['root'])
    allocation_root, event_root, native_root = production_paths(plan, arm)
    base = root / 'arms' / arm
    base.mkdir(parents=True, exist_ok=False)
    inputs = read(plan['manifest'])
    original = Path(plan['original_config']).read_text()
    status = dict(
        status='running', arm=arm, phase='initializing', execution_commit=plan['execution_commit']
    )
    write_json(base / 'STATUS.json', status)
    backend = trainer = None
    ledger = Ledger(base / 'WORK.json')
    started = time.perf_counter()
    witness = root / 'witness'
    witness.mkdir(exist_ok=True)
    event_root.mkdir(parents=True, exist_ok=False)
    try:
        write_json(base / 'ADMISSION.json', boundary(plan))
        runtime = pin_retained_graph_runtime(base)
        runtime.update(
            applied_before='production trainer construction and compile startup',
            scope='this one-event ' + arm + ' production process',
        )
        write_json(base / 'RUNTIME_COMPATIBILITY.json', runtime)
        backend = create_backend(plan, native_root, card='A')
        loading_started = time.perf_counter()
        trainer = PilotTrainer(
            plan['configs'][arm],
            base / 'trainer',
            arm=arm,
            binding=plan['binding'],
            original_config_text=original,
            maximum_events=1,
        )
        status['trainer_loading_s'] = time.perf_counter() - loading_started
        status['serving_card'] = plan['devices']['A']
        status['training_card'] = plan['devices']['B']
        status['initialization'] = trainer.save_initial(
            root / 'arms/BASE/trainer/INITIAL_WITNESS.pt' if arm == 'PAIR' else None
        )
        controller = OnlineController(backend, trainer, event_root / 'loop')
        generate, prepare = backend.generate, backend.prepare_feedback

        def generate_once(*args, **kwargs):
            boundary(plan)
            return ledger.call(
                {'feedback_generation': 1}, 'own_feedback', generate, *args, **kwargs
            )

        def prepare_once(record, path):
            boundary(plan)
            receipt = ledger.call(
                {'teacher_preparation': 1}, 'native_teacher', prepare, record, path
            )
            retire_redundant_features(record, receipt, allocation_root, event_root)
            return receipt

        backend.generate, backend.prepare_feedback = generate_once, prepare_once
        status.update(phase='collecting', backend_loading_s=backend.loading_s)
        write_json(base / 'STATUS.json', status)
        records = []
        for request in inputs['train_requests']:
            record = controller.serve(
                request['prompt_ids'],
                request_id=request['request_id'],
                session_id='repair_pair_train_' + arm,
                request_index=request['request_index'],
                max_new_tokens=2048,
            )
            online_record_check(record, request, plan['draft'], capture=True)
            if len(record['response_ids']) < 2:
                raise ValueError(
                    'Retained short own response has no training anchors; no replacement'
                )
            records.append(record)
            status['feedback_completed'] = len(records)
            write_json(base / 'STATUS.json', status)
        native_train = trainer.train_event

        def train_ready(event_id, paths, passes=2):
            status['phase'] = 'startup'
            write_json(base / 'STATUS.json', status)
            status['startup'] = discarded_startup(trainer, paths, base / 'startup')
            status['startup_work'] = dict(response_forward=2, response_backward=2, adam=0)
            if arm == 'BASE':
                witness_paths = (
                    retain_witness_features(paths, witness)
                    if plan.get('feature_strategy') == 'local_split'
                    else paths
                )
                status['witness_seal'] = freeze_witness(
                    trainer, witness_paths, witness / 'bindings.pt'
                )
                status['origin_witness'] = evaluate_witness(
                    trainer, witness / 'bindings.pt', witness / 'origin.json', endpoint='origin'
                )
            for owner, name, kind in (
                (trainer.strategy, 'forward_loss', 'response_forward'),
                (trainer.backend, 'backward', 'response_backward'),
                (trainer.backend.optimizer.optimizer, 'step', 'adam'),
            ):
                ledger.wrap(owner, name, kind)
            status['phase'] = 'training'
            write_json(base / 'STATUS.json', status)
            native_checkpoint = trainer._checkpoint_state

            def bounded_checkpoint(event):
                state = native_checkpoint(event)
                write_json(base / 'RECOVERY_SIZE_PLAN.json', recovery_size_plan(state))
                return state

            trainer._checkpoint_state = bounded_checkpoint
            try:
                receipt = native_train(event_id, paths, passes=passes)
            finally:
                trainer._checkpoint_state = native_checkpoint
            status['recovery_size'] = validate_recovery_file(trainer.latest)
            status['endpoint_witness'] = evaluate_witness(
                trainer,
                witness / 'bindings.pt',
                witness / (arm + '1.json'),
                origin_path=witness / 'origin.json',
                endpoint=arm + '1',
            )
            return receipt

        trainer.train_event = train_ready
        receipt = controller.update('event_001')
        checkpoint = receipt['training']['checkpoint_path']
        status['boundary'] = trainer.verify_boundary(checkpoint, base / 'FINAL_STATE_CHECK.json')
        status['publication'] = check_publication(
            backend, checkpoint, base / 'PUBLICATION_CHECK.json'
        )
        status.update(
            status='completed',
            phase='completed',
            checkpoint=checkpoint,
            recovery=str(trainer.latest),
            training=receipt['training']['costs'],
            checkpoint_identity=identity(Path(checkpoint) / 'model.safetensors'),
            recovery_identity=identity(trainer.latest),
            teacher_preparation_s=sum(r['teacher_preparation_s'] for r in receipt['preparation']),
            feature_persistence_s=sum(r['feature_persistence_s'] for r in receipt['preparation']),
        )
        features = [
            dict(path=r['feature_path'], identity=identity(r['feature_path']))
            for r in receipt['preparation']
        ]
        write_json(base / 'FEATURE_IDENTITIES.json', features)
        actual = read(base / 'WORK.json')
        expected = dict(
            feedback_generation=128,
            teacher_preparation=128,
            response_forward=256,
            response_backward=256,
            adam=128,
        )
        if actual['pending'] or actual['completed'] != expected or actual['attempted'] != expected:
            raise ValueError('Production one-event accounting differs from sealed budget')
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(base / 'STATUS.json', status)
        errors = []
        if backend is not None and not backend.failed:
            try:
                write_json(base / 'PHYSICAL_WORK.json', backend._call('work_receipt'))
            except BaseException as error:
                errors.append(repr(error))
        for owner in (trainer, backend):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as error:
                    errors.append(repr(error))
        status.update(wall_s=time.perf_counter() - started, cleanup_errors=errors)
        if errors:
            status['status'] = 'failed'
        write_json(base / 'STATUS.json', status)
        if errors:
            raise RuntimeError('Owned runtime cleanup failed: ' + repr(errors))
    return status
