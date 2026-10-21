"""CPU tensor receipts through real P8 validators; native rows are synthetic.

This does not launch the native runtime, export a serving checkpoint, or qualify
GPU execution. The temporary toy checkpoint contains draft weights only.
"""
import copy
import hashlib
import json

import pytest

torch = pytest.importorskip('torch')
from . import experiment, loss, training
from .protocol import SERVING_CONTRACT
from .test_loss import diagnostic_pool, inputs
from .test_training import new


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_actual_training_and_posttraining_loss_receipts_pass_controller_validators(dtype, tmp_path):
    torch.set_num_threads(1)
    pool = diagnostic_pool(dtype)
    trainer, _ = new(pool, dtype=dtype)
    account = trainer.train_eight(pool['support'])
    assert trainer.optimizer_steps == 8 and trainer.response_presentations == 16

    # Hash real tiny trained-weight bytes; the wrapper is synthetic export metadata.
    checkpoint_dir = tmp_path/'checkpoints'/'P8'
    checkpoint_dir.mkdir(parents=True)
    weights = checkpoint_dir/'toy_draft.pt'
    torch.save(trainer.draft.state_dict(), weights)
    payload = weights.read_bytes()
    files = {weights.name: dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())}
    identity = training.digest(files)
    checkpoint = dict(checkpoint_id='P8', pool_id=pool['pool_id'], domain=pool['domain'],
        group_ids=pool['group_ids'], optimizer_steps=8, learning_rate=3e-4,
        checkpoint_identity_sha256=identity, files=files, path=str(checkpoint_dir))
    trainer.bind_p8_checkpoint(identity)

    receipt = dict(status='passed', pool_id=pool['pool_id'], checkpoint=checkpoint,
        accounting=account, losses={}, loss_diagnostics={})
    sampler = trainer.generator.get_state().clone()
    for population in ('source', 'heldout'):
        records, references = inputs(pool, population)
        diagnostic = loss.evaluate_responses(trainer.draft, trainer.embed_tokens, trainer.lm_head,
            trainer.teacher_features, records, references, original_pool=pool, population=population,
            checkpoint_id='P8', checkpoint_identity_sha256=identity, mask_token_id=19,
            attention_backend='sdpa', projection_chunk=7)
        receipt['losses'][population] = diagnostic['response_losses']
        receipt['loss_diagnostics'][population] = diagnostic
        assert diagnostic['attempted_response_calls'] == diagnostic['returned_response_rows'] == 8
        assert diagnostic['unknown_response_calls'] == diagnostic['unattempted_response_calls'] == 0
        assert not diagnostic['optimizer_created'] and diagnostic['optimizer_steps'] == 0
    assert trainer.optimizer_steps == 8 and torch.equal(sampler, trainer.generator.get_state())

    # Match the worker RPC transport, including tuple-to-array conversion in Adam betas.
    receipt = json.loads(json.dumps(receipt, allow_nan=False))
    assert experiment.validate_training(receipt, pool) == dict(status='passed', errors=[])

    # Native output rows below are explicitly constructed fixtures, not generations.
    # They exercise final pair admission with the actual post-training KL receipts.
    for population in ('source', 'heldout'):
        for reference, diagnostic in zip(pool['reference_pairs'][population], receipt['losses'][population]):
            pair = copy.deepcopy(reference)
            native = dict(copy.deepcopy(reference['F']['native']), checkpoint_id='P8',
                checkpoint_identity_sha256=identity, pool_id=pool['pool_id'],
                pool_request_id=reference['pool_request_id'], feedback_capture_enabled=False,
                serving_contract=copy.deepcopy(SERVING_CONTRACT))
            pair['checkpoint_bindings']['P8'] = identity
            pair['P8'] = dict(native=native, kl=diagnostic)
            assert experiment.validate_pair(pair) == dict(status='passed', errors=[])

    # The controller must reject a damaged post-training diagnostic receipt.
    damaged = copy.deepcopy(receipt)
    damaged['losses']['heldout'][0]['normalization_denominator'] += 1
    with pytest.raises(ValueError):
        experiment.validate_training(damaged, pool)
