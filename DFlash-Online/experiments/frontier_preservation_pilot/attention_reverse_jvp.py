"""Prototype: transpose native first backward only in its incoming cotangent.

This is deliberately not a general second derivative of FlexAttention. The
Q/K/V primal state is frozen inside the native-VJP node. Its only differentiable
input is the incoming output cotangent; its transpose uses attention_tangent.
Use it for d/du <direction, J(theta)^T u> with theta fixed. Parameter Hessians
and third derivatives are outside this interface and must not be requested.

Both the attention primal and first VJP execute the original native operator.
Ordinary training and serving never install this scoped prototype. Native
identity and whole-model duality require separate real-runtime qualification.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import importlib

import torch
from torch.autograd.function import once_differentiable
from torch.nn.attention.flex_attention import AuxRequest

from .attention_jvp import attention_tangent


@dataclass
class _NativeState:
    inputs: tuple
    output: torch.Tensor
    lse: torch.Tensor
    options: dict
    accounting: dict
    native_graph_used: bool = False


def _native_vjp(incoming, state):
    """Consume one native graph; the later transpose uses only its primal values.

    No second native backward is needed. AOT buffer donation excludes forward
    inputs and public outputs, which protect our Q/K/V, output and returned LSE.
    Independent VJP checks must each construct a fresh outer forward.
    """
    if state.native_graph_used:
        raise RuntimeError('Native attention graph already consumed; run a fresh forward')
    state.native_graph_used = True
    gradients = torch.autograd.grad(
        state.output,
        state.inputs,
        grad_outputs=incoming.detach(),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    state.accounting['native_first_vjps'] = state.accounting.get('native_first_vjps', 0) + 1
    return gradients


class _CotangentVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, incoming, state):
        # State is a Python object, so no parameter/activation second-derivative
        # edges are introduced. Only incoming remains a differentiable input.
        ctx.state = state
        ctx.set_materialize_grads(False)
        return _native_vjp(incoming, state)

    @staticmethod
    @once_differentiable
    def backward(ctx, query_adjoint, key_adjoint, value_adjoint):
        state = ctx.state
        tangent = attention_tangent(
            *state.inputs,
            query_adjoint,
            key_adjoint,
            value_adjoint,
            output=state.output.detach(),
            lse=state.lse.detach(),
            block_mask=state.options.get('block_mask'),
            scale=state.options.get('scale'),
            enable_gqa=state.options.get('enable_gqa', False),
            accounting=state.accounting,
        )
        state.accounting['cotangent_transposes'] = (
            state.accounting.get('cotangent_transposes', 0) + 1
        )
        return tangent, None


class _NativeAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, native_attention, options, accounting):
        # Build one native inner graph on leaves with identical values. The
        # outer custom node owns the connection to the original model graph.
        with torch.enable_grad():
            inputs = tuple(tensor.detach().requires_grad_() for tensor in (query, key, value))
            output, auxiliary = native_attention(
                *inputs, **dict(options, return_aux=AuxRequest(lse=True))
            )
        accounting['native_primal_forwards'] = accounting.get('native_primal_forwards', 0) + 1
        ctx.state = _NativeState(inputs, output, auxiliary.lse, options, accounting)
        ctx.set_materialize_grads(False)
        return output.detach()

    @staticmethod
    def backward(ctx, incoming):
        if incoming is None:
            return (None,) * 6
        if torch.is_grad_enabled():
            gradients = _CotangentVJP.apply(incoming, ctx.state)
        else:
            gradients = _native_vjp(incoming, ctx.state)
        return (*gradients, None, None, None)


def attention_reverse_jvp_wrapper(native_attention, accounting):
    """Install the missing cotangent transpose on vanilla attention only."""

    def wrapped(query, key, value, **options):
        if any(
            torch.autograd.forward_ad.unpack_dual(value).tangent is not None
            for value in (query, key, value)
        ):
            raise ValueError(
                'The cotangent-transpose prototype cannot be combined with forward AD'
            )
        if not torch.is_grad_enabled() or not any(
            value.requires_grad for value in (query, key, value)
        ):
            accounting['passthrough_calls'] = accounting.get('passthrough_calls', 0) + 1
            return native_attention(query, key, value, **options)
        if options.get('score_mod') is not None:
            raise ValueError('Only vanilla attention scores are supported')
        if options.get('return_lse') or options.get('return_aux') is not None:
            raise ValueError('The cotangent-transpose prototype owns native LSE collection')
        if (options.get('kernel_options') or {}).get('PRESCALE_QK', False):
            raise ValueError('Prescaled QK derivative is unsupported')
        allowed = {
            'score_mod',
            'block_mask',
            'scale',
            'enable_gqa',
            'kernel_options',
            'return_lse',
            'return_aux',
        }
        if set(options) - allowed:
            raise ValueError('Unsupported native attention arguments')
        return _NativeAttention.apply(query, key, value, native_attention, options, accounting)

    return wrapped


@contextmanager
def native_attention_reverse_jvp():
    """Scope the prototype to DFlash's attention call site, then restore it."""
    module = importlib.import_module('specforge.modeling.draft.dflash')
    original = module.compile_friendly_flex_attention
    accounting = dict(
        passthrough_calls=0,
        native_primal_forwards=0,
        native_first_vjps=0,
        cotangent_transposes=0,
        query_tiles=0,
        key_tiles=0,
        max_score_elements=0,
    )
    module.compile_friendly_flex_attention = attention_reverse_jvp_wrapper(original, accounting)
    try:
        yield accounting
    finally:
        module.compile_friendly_flex_attention = original
