"""Opt-in precision for the frozen head's auxiliary input VJP.

The forward is the existing BF16 Linear call. Only the first-order backward
product changes: round the incoming cotangent to BF16 as native autograd does,
compute S_B @ W_B in IEEE FP32, then return a BF16 hidden-state adjoint.
Neither ordinary training nor serving uses this helper unless explicitly called.
"""
from contextlib import contextmanager

import torch
from torch.autograd.function import once_differentiable


@contextmanager
def ieee_fp32_matmul(device_type):
    """Disable autocast/TF32 for one product, restoring the caller's settings.

    The pinned Torch 2.13 runtime provides the per-backend precision API. Its
    setting is process-global, so this context must not overlap other GPU work.
    It ends before autograd continues through the draft model.
    """
    previous = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        if torch.backends.cuda.matmul.fp32_precision != 'ieee':
            raise RuntimeError('IEEE FP32 auxiliary head multiplication was not enabled')
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous


@torch.no_grad()
def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Nonfinite '+label)


@torch.no_grad()
def fp32_head_vjp(cotangent, weight, *, return_precast=False):
    """Return the BF16 hidden adjoint; optionally expose its FP32 pre-cast value.

    ``weight`` is the frozen BF16 [vocabulary, hidden] matrix. There is one
    matrix product and one transient FP32 weight conversion per invocation.
    No FP32 weight copy or master parameter survives this call.
    """
    if weight.dtype != torch.bfloat16 or weight.requires_grad:
        raise ValueError('The auxiliary head requires a frozen BF16 weight')
    incoming = cotangent.to(torch.bfloat16)
    _finite(incoming, 'BF16 auxiliary head cotangent')
    with ieee_fp32_matmul(incoming.device.type):
        precast = torch.matmul(incoming.float(), weight.float())
    _finite(precast, 'FP32 auxiliary head adjoint')
    adjoint = precast.to(torch.bfloat16)
    _finite(adjoint, 'BF16 auxiliary head adjoint')
    return (adjoint, precast) if return_precast else adjoint


class _AuxiliaryHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, head):
        ctx.save_for_backward(head.weight)
        return head(hidden)

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        weight, = ctx.saved_tensors
        return fp32_head_vjp(cotangent, weight), None


def auxiliary_head(head, hidden):
    """Use native frozen-head logits with the isolated FP32 auxiliary VJP."""
    if not isinstance(head, torch.nn.Linear) or head.bias is not None:
        raise ValueError('The qualified vocabulary head is a bias-free Linear')
    if (hidden.dtype != torch.bfloat16 or head.weight.dtype != torch.bfloat16
            or head.weight.requires_grad):
        raise ValueError('The auxiliary head requires BF16 hidden states and a frozen BF16 weight')
    logits = _AuxiliaryHead.apply(hidden, head)
    _finite(logits, 'native auxiliary head logits')
    return logits
