"""Exact decoded endpoint comparison with explicitly separated metadata."""
import json
from pathlib import Path
import shutil
import time

from experiments.ok_continuation.resume import differences, validate_snapshot
from online_feedback.storage import write_json

SCIENTIFIC_STATE_KEYS = (
    'event_schema', 'draft_state_dict', 'backend', 'rng', 'global_step', 'presentations',
    'strategy', 'world_size', 'config_text', 'algorithm_contract', 'reduction',
    'accumulation_steps', 'horizon',
)
EVENT_KEYS = (
    'event_id', 'features', 'passes', 'response_presentations', 'start_step', 'end_step',
    'optimizer_steps', 'reduction', 'objective', 'gradient_denominator',
    'selection_totals', 'selection_totals_source',
)


def scientific_view(state):
    result = {key: state[key] for key in SCIENTIFIC_STATE_KEYS}
    # Timing and instrumentation counters can change without changing the
    # continued model. Preserve all ordinary metrics, LR, identities and draws.
    result['events'] = [dict(
        **{key: event[key] for key in EVENT_KEYS if key in event},
        records=[{key: value for key, value in row.items() if key != 'training_s'}
                 for row in event['records']]) for event in state['events']]
    return result


def metadata_view(state):
    return dict(top_level={key: value for key, value in state.items()
                         if key not in SCIENTIFIC_STATE_KEYS and key != 'events'},
        events=[dict(fields={key: value for key, value in event.items()
                             if key not in EVENT_KEYS and key != 'records'},
                     training_s=[row.get('training_s') for row in event['records']])
                for event in state['events']])


def compare_endpoints(actual, original, *, event_number, config_text, receipt_path=None,
                      expected_cuda_rng_states=1):
    started = time.perf_counter()
    for state in (actual, original):
        validate_snapshot(state, objective='osd_first_error', config_text=config_text,
            completed_event=event_number, expected_cuda_rng_states=expected_cuda_rng_states)
    mismatch = differences(scientific_view(actual), scientific_view(original), 'scientific')
    metadata = differences(metadata_view(actual), metadata_view(original), 'metadata')
    receipt = dict(status='passed' if not mismatch else 'failed',
        exact_decoded_scientific_state=not mismatch, event_number=event_number,
        scientific_mismatch_count=len(mismatch), scientific_mismatches=mismatch,
        metadata_difference_count=len(metadata), metadata_difference_paths=metadata,
        metadata_scope='run ID; event timing, objective instrumentation and optional diagnostic receipts',
        tensor_comparison='native dtype/shape and bitwise bytes, including Adam moments and RNG',
        ordinary_history='complete features, order, anchors, LR, metrics and counters compared',
        serialized_file_hash_equality_required=False, extra_model_passes=0,
        extra_optimizer_steps=0, wall_s=time.perf_counter() - started)
    if receipt_path is not None:
        write_json(receipt_path, receipt)
    if mismatch:
        raise ValueError('Decoded scientific endpoint differs: ' + str(mismatch[:8]))
    return receipt


def verify_live_endpoint(trainer, state, receipt_path):
    """Verify empty live accumulation after the diagnostic controller closes."""
    from online_feedback.rng import capture_rng
    actual = dict(weights=dict(trainer.bundle.draft_model.state_dict()),
        optimizer=trainer.backend.optimizer.state_dict(), rng=capture_rng(), events=trainer.events,
        global_step=trainer.global_step, presentations=trainer.presentations,
        contract=trainer.contract, config=trainer.config_text, horizon=trainer.horizon,
        accumulation=trainer.accumulation, reduction=trainer.reduction)
    expected = dict(weights=state['draft_state_dict'], optimizer=state['backend']['optimizer'],
        rng=state['rng'], events=state['events'], global_step=state['global_step'],
        presentations=state['presentations'], contract=state['algorithm_contract'], config=state['config_text'],
        horizon=state['horizon'], accumulation=state['accumulation_steps'], reduction=state['reduction'])
    mismatch = differences(actual, expected)
    empty = (trainer.core.accumulation_remainder == 0 and trainer.core._micro == trainer.presentations
        and not trainer.core._ratio_totals and not trainer.core._sum_totals
        and trainer._presentation_schedule is None
        and not getattr(trainer, '_krc_late_event', None)
        and not getattr(trainer, '_krc_continuation_active', False)
        and trainer.bundle.model.reference is None and not trainer.bundle.model.ranking_enabled
        and not hasattr(trainer.bundle.model, 'prefix_guard')
        and all(parameter.grad is None for parameter in trainer.bundle.draft_model.parameters())
        and trainer.bundle.model.training and not trainer.failed and not trainer.closed)
    if not empty:
        mismatch.append('live_accumulation_mode_gradient_or_controller_boundary')
    receipt = dict(status='passed' if not mismatch else 'failed', exact_boundary=not mismatch,
        mismatches=mismatch, no_rng_restore=True, extra_model_passes=0, extra_optimizer_steps=0)
    write_json(receipt_path, receipt)
    if mismatch:
        raise ValueError('Live endpoint boundary changed: ' + str(mismatch[:8]))
    return receipt


def retire_verified_replay(event_root, receipt, item, actual_state, comparison, sealed_files):
    """Retire only regenerated event-2 tensor copies after exact qualification.

    Original KRC2 and all compact records remain. The separately preserved
    metadata plus original scientific state reconstruct the decoded replay
    state; the retired serialized tensor containers are not called archived.
    """
    from experiments.krc_events_3_4.campaign import artifact_identity
    event_root = Path(event_root).resolve()
    if (item['event_number'] != 2 or comparison.get('exact_decoded_scientific_state') is not True
            or comparison.get('status') != 'passed'):
        raise ValueError('Only an exactly qualified regenerated event-2 endpoint may be retired')
    started = time.perf_counter()
    metadata = metadata_view(actual_state)
    metadata_path = event_root / 'DECODED_REPLAY_METADATA.json'
    write_json(metadata_path, metadata)
    if differences(json.loads(metadata_path.read_text()), metadata):
        raise ValueError('Retained decoded metadata did not round-trip exactly')
    # These identities are checked against the seal again immediately before
    # deletion, after endpoint comparison. Never rely on a remembered location.
    original_paths = [Path(item['endpoint_state'])] + [path for path in
        sorted(Path(item['endpoint_checkpoint']).iterdir()) if path.is_file()]
    originals = [artifact_identity(path) for path in original_paths]
    for row in originals:
        expected = sealed_files[row['path']]
        if {key: row[key] for key in ('bytes', 'sha256')} != expected:
            raise ValueError('Original endpoint changed before regenerated-copy retirement')
    state_path, export_root = Path(receipt['full_state_path']), Path(receipt['checkpoint_path'])
    removable = [state_path] + [path for path in sorted(export_root.iterdir())
        if path.is_file() and path.suffix in {'.safetensors', '.bin', '.pt'}]
    for path in removable:
        if (path.is_symlink() or not path.resolve().is_relative_to(event_root / 'replay')
                or path.resolve() in {value.resolve() for value in original_paths}):
            raise ValueError('Retirement path escapes newly generated replay tensors')
    identities = [artifact_identity(path) for path in removable]
    result = dict(status='verified_before_retirement', event_number=2,
        exact_decoded_scientific_state=True, retained_originals=originals,
        regenerated_artifacts_to_retire=identities,
        retained_metadata=artifact_identity(metadata_path),
        retained_evidence=[str(event_root / name) for name in ('ENDPOINT_EQUALITY.json',
            'SAVED_STATE_CHECK.json', 'ORIGINAL_ENDPOINT_CHECK.json', 'LIVE_ENDPOINT.json',
            'TRAINING_CHECK.json', 'CORRECTION_EVIDENCE.json', 'REPLAY_STATE_IDENTITIES.json')],
        reason='Verified regenerated scientific duplicate; original complete KRC2 remains unchanged',
        recovery='Original endpoint scientific fields plus DECODED_REPLAY_METADATA reproduce decoded replay state',
        lost_capability='Direct load/hash reproduction of the retired serialized replay container and export tensors',
        free_before_bytes=shutil.disk_usage(event_root).free, removed_paths=[])
    for path in result['retained_evidence']:
        if not Path(path).is_file():
            raise ValueError('Compact replay validity evidence is unavailable: ' + path)
    manifest = event_root / 'REPLAY_RETENTION.json'
    write_json(manifest, result)
    if json.loads(manifest.read_text()) != result:
        raise ValueError('Retirement manifest did not round-trip')
    for path in removable:
        path.unlink()
        result['removed_paths'].append(str(path))
        write_json(manifest, result)
    result.update(status='completed', free_after_bytes=shutil.disk_usage(event_root).free,
                  wall_s=time.perf_counter() - started)
    write_json(manifest, result)
    return result
