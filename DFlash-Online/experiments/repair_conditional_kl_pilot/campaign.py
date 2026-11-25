"""One own-feedback event per arm, using the unchanged two-card native lifecycle."""

from contextlib import ExitStack
from pathlib import Path
import time

from online_feedback.controller import OnlineController
from online_feedback.sglang_backend import SGLangDFlashBackend
from online_feedback.storage import write_json
from experiments.full_k_gsm8k.campaign import online_record_check
from experiments.kplus_full_gsm8k.storage import retire_redundant_features
from experiments.fresh_confirmation.protocol import publication_errors, weight_check_errors
from experiments.prefix_guard.qualify import pin_retained_graph_runtime
from experiments.ok_coding.storage import identity
from .resources import read, boundary, Ledger, occupied_card
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


def run_arm(plan, arm):
    if not plan.get('production_admitted', False):
        raise RuntimeError('Production resource admission is blocked; qualification-only plan')
    root = Path(plan['root'])
    base = root / 'arms' / arm
    base.mkdir(parents=True, exist_ok=False)
    if arm == 'PAIR' and read(root / 'BASE_FEATURE_OFFLOAD.json')['status'] != 'verified_retired':
        raise ValueError('BASE feature offload must complete before PAIR allocation')
    if read(root / 'QUALIFICATION.json')['status'] != 'passed':
        raise ValueError('Complete bounded native qualification is required')
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
    event_root = base / 'event_001'
    event_root.mkdir()
    try:
        write_json(base / 'ADMISSION.json', boundary(plan))
        write_json(base / 'RUNTIME_COMPATIBILITY.json', pin_retained_graph_runtime(base))
        backend = create_backend(plan, base / 'native')
        trainer = PilotTrainer(
            plan['configs'][arm],
            base / 'trainer',
            arm=arm,
            binding=plan['binding'],
            original_config_text=original,
            maximum_events=1,
        )
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
            retire_redundant_features(record, receipt, root, event_root)
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
                status['witness_seal'] = freeze_witness(trainer, paths, witness / 'bindings.pt')
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
            receipt = native_train(event_id, paths, passes=passes)
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
