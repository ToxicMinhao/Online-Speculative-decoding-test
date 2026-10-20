"""Small deterministic numerical fixtures rerun before native qualification."""
from __future__ import annotations

from .core import agreement_mask, apply_adamw_step, NUMERICS
from .controls import norm_match


def run_deterministic_checks():
    import torch
    tiny = torch.tensor([1e-30, -1e-30, 0.], dtype=torch.float32)
    mask = agreement_mask(tiny, tiny, tiny.abs())
    if mask.tolist() != [False, True, True]:
        raise AssertionError('Tiny positive product must not underflow into admission')
    parameters = {'x': torch.nn.Parameter(torch.tensor([1., -1.], dtype=torch.bfloat16))}
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=3e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    a = {'x': torch.tensor([1., -1.])}; b = {'x': -a['x']}
    # This synthetic objective has an explicitly zero full gradient. Production
    # obtains its full gradient by a separate backward, never an A+B fallback.
    full = {'x': torch.zeros_like(parameters['x'])}
    before = parameters['x'].detach().clone()
    stats = apply_adamw_step(parameters, optimizer, a, b, full_loss_gradient=full, mode='G')
    if not (torch.equal(before, parameters['x']) and stats['optimizer_steps_after'] == 1 and
            stats['optimizer_step_calls'] == 1 and
            stats['proposal_gradient_source'] == 'direct_full_loss_gradient'):
        raise AssertionError('Zero applied step must still advance Adam exactly once')
    if norm_match(1.4142135623730951, 1.)['status'] != 'failed' or norm_match(0., 1e-30)['status'] != 'failed':
        raise AssertionError('Rounded N control must not admit oversized or erased steps')
    return dict(status='passed', device='cpu', numerics=dict(NUMERICS),
                checks=['product_underflow_rejected', 'non_strict_zero_sign', 'zero_write_keeps_adam_step', 'N_oversized_or_erased_norm_rejected'],
                scope='Deterministic guards; full numerical tests and native first-stream qualification remain separate.')
