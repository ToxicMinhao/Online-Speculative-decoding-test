"""Training-only forward AD at the existing vanilla FlexAttention boundary.

The primal is always the original compiled native operator. Its returned LSE
and rounded output define the tangent's local state. For a masked probability
matrix P and D = scale * (dQ K^T + Q dK^T), the tangent is

    cast_native(P) dV + (P * D) V - native_output * sum(P * D).

Products and reductions are FP32 for native BF16 inputs; the final tangent is
cast to the primal dtype. FP64 fixtures retain FP64 throughout. The P cast in
the value term follows the native backward's value-gradient branch. The score
term stays FP32: transposing a cast in the backward does not justify inserting
a new cast of P * D. Native output centering follows the backward's saved-output
delta. This is a mixed-precision analytic derivative, not a derivative of
discrete BF16 rounding. Native duality qualification is required separately.
"""

from contextlib import contextmanager
import importlib
import math

import torch
from torch.nn.attention.flex_attention import AuxRequest

from online_feedback.fsg_head_precision import ieee_fp32_matmul


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sparse_rows(block_mask):
    """Copy compact sparsity metadata once, never a dense attention matrix."""
    if block_mask is None:
        return None
    groups = []
    for prefix in ('', 'full_'):
        counts = getattr(block_mask, prefix + 'kv_num_blocks')
        indices = getattr(block_mask, prefix + 'kv_indices')
        if counts is not None:
            groups.append(
                (
                    counts.reshape(-1, counts.shape[-1]).detach().cpu().tolist(),
                    indices.reshape(-1, *indices.shape[-2:]).detach().cpu().tolist(),
                )
            )
    return block_mask.BLOCK_SIZE, groups


def _key_ranges(metadata, query_start, query_stop, key_length, key_tile):
    if metadata is None:
        return [
            (start, min(start + key_tile, key_length))
            for start in range(0, key_length, key_tile)
        ]
    (query_block, key_block), groups = metadata
    columns = set()
    first_row = query_start // query_block
    last_row = (query_stop - 1) // query_block
    for counts, indices in groups:
        for batch_counts, batch_indices in zip(counts, indices):
            for row in range(first_row, last_row + 1):
                columns.update(batch_indices[row][: batch_counts[row]])
    ranges = []
    for column in sorted(columns):
        begin, end = column * key_block, min((column + 1) * key_block, key_length)
        for start in range(begin, end, key_tile):
            ranges.append((start, min(start + key_tile, end)))
    return ranges


def _expanded_key_tile(value, start, stop, batch, groups, dtype):
    tile = value[:, :, start:stop].to(dtype)
    return tile.repeat_interleave(groups, dim=1).expand(batch, -1, -1, -1)


def attention_tangent(
    query,
    key,
    value,
    query_tangent,
    key_tangent,
    value_tangent,
    *,
    output,
    lse,
    block_mask=None,
    scale=None,
    enable_gqa=False,
    query_tile=128,
    key_tile=128,
    accounting=None,
):
    """Pin derivative-only IEEE arithmetic and restore the caller's precision.

    The existing precision context disables autocast and TF32 only while computing
    this tangent. It is process-global and must not overlap another GPU operation;
    the native primal has already finished and ordinary calls never enter it.
    """
    previous_precision = torch.backends.cuda.matmul.fp32_precision
    if accounting is not None:
        accounting['last_precision_before'] = previous_precision
    try:
        with ieee_fp32_matmul(query.device.type):
            if accounting is not None:
                accounting['derivative_fp32_precision'] = (
                    torch.backends.cuda.matmul.fp32_precision
                )
                accounting['derivative_autocast'] = torch.is_autocast_enabled(query.device.type)
            return _attention_tangent(
                query,
                key,
                value,
                query_tangent,
                key_tangent,
                value_tangent,
                output=output,
                lse=lse,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=enable_gqa,
                query_tile=query_tile,
                key_tile=key_tile,
                accounting=accounting,
            )
    finally:
        if accounting is not None:
            restored = torch.backends.cuda.matmul.fp32_precision
            accounting['last_precision_after'] = restored
            accounting['precision_restored'] = (
                accounting.get('precision_restored', True) and restored == previous_precision
            )


@torch.no_grad()
def _attention_tangent(
    query,
    key,
    value,
    query_tangent,
    key_tangent,
    value_tangent,
    *,
    output,
    lse,
    block_mask=None,
    scale=None,
    enable_gqa=False,
    query_tile=128,
    key_tile=128,
    accounting=None,
):
    """Compute one direction with bounded tiles and the exact native mask.

    Sparse metadata selects a union of permitted key blocks across query rows,
    batches and heads. The native mask_mod is still evaluated on every selected
    tile, including full blocks, so the union never changes attention support.
    """
    _require(query.ndim == key.ndim == value.ndim == 4, 'Expected BHSD attention tensors')
    batch, heads, query_length, width = query.shape
    key_batch, key_heads, key_length, key_width = key.shape
    _require(query.dtype == key.dtype == value.dtype, 'Attention dtypes differ')
    _require(
        query.dtype in (torch.bfloat16, torch.float32, torch.float64),
        'Unsupported attention derivative dtype',
    )
    _require(
        key.shape[:3] == value.shape[:3] and key_width == width, 'Attention QKV shapes differ'
    )
    _require(
        key_batch in (1, batch) and heads % key_heads == 0,
        'Unsupported attention batch or GQA shape',
    )
    _require(enable_gqa or heads == key_heads, 'GQA requires enable_gqa')
    _require(query_tile > 0 and key_tile > 0, 'Attention tile sizes must be positive')
    _require(
        output.shape == (batch, heads, query_length, value.shape[-1]),
        'Native attention output shape differs',
    )
    _require(lse.shape == (batch, heads, query_length), 'Native LSE shape differs')
    _require((torch.isfinite(lse) | torch.isneginf(lse)).all(), 'Invalid native LSE')
    for primal, tangent in ((query, query_tangent), (key, key_tangent), (value, value_tangent)):
        _require(
            tangent is None or tangent.shape == primal.shape, 'Attention tangent shape differs'
        )
    scale = width**-0.5 if scale is None else float(scale)
    _require(math.isfinite(scale), 'Nonfinite attention scale')
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    groups = heads // key_heads
    metadata = _sparse_rows(block_mask)
    result = torch.zeros_like(output)
    batch_indices = torch.arange(batch, device=query.device).view(-1, 1, 1, 1)
    head_indices = torch.arange(heads, device=query.device).view(1, -1, 1, 1)
    for start in range(0, query_length, query_tile):
        stop = min(start + query_tile, query_length)
        q = query[:, :, start:stop].to(dtype)
        dq = None if query_tangent is None else query_tangent[:, :, start:stop].to(dtype)
        tile_lse = lse[:, :, start:stop].to(dtype)
        finite_row = torch.isfinite(tile_lse)
        safe_lse = torch.where(finite_row, tile_lse, 0.0)
        accumulated = torch.zeros_like(output[:, :, start:stop], dtype=dtype)
        row_direction = torch.zeros_like(tile_lse)
        query_indices = torch.arange(start, stop, device=query.device).view(1, 1, -1, 1)
        if accounting is not None:
            accounting['query_tiles'] = accounting.get('query_tiles', 0) + 1
        for key_start, key_stop in _key_ranges(metadata, start, stop, key_length, key_tile):
            k = _expanded_key_tile(key, key_start, key_stop, batch, groups, dtype)
            v = _expanded_key_tile(value, key_start, key_stop, batch, groups, dtype)
            scores = (q @ k.transpose(-2, -1)) * scale
            mask = finite_row.unsqueeze(-1)
            if block_mask is not None:
                key_indices = torch.arange(key_start, key_stop, device=query.device).view(
                    1, 1, 1, -1
                )
                mask = mask & block_mask.mask_mod(
                    batch_indices, head_indices, query_indices, key_indices
                )
            probabilities = torch.exp(scores - safe_lse.unsqueeze(-1)).masked_fill(~mask, 0.0)
            score_direction = torch.zeros_like(scores)
            if dq is not None:
                score_direction.add_(dq @ k.transpose(-2, -1))
            if key_tangent is not None:
                dk = _expanded_key_tile(key_tangent, key_start, key_stop, batch, groups, dtype)
                score_direction.add_(q @ dk.transpose(-2, -1))
            weighted_direction = probabilities * (score_direction * scale)
            accumulated.add_(weighted_direction @ v)
            row_direction.add_(weighted_direction.sum(-1))
            if value_tangent is not None:
                dv = _expanded_key_tile(
                    value_tangent, key_start, key_stop, batch, groups, dtype
                )
                accumulated.add_(probabilities.to(query.dtype).to(dtype) @ dv)
            if accounting is not None:
                accounting['key_tiles'] = accounting.get('key_tiles', 0) + 1
                accounting['max_score_elements'] = max(
                    accounting.get('max_score_elements', 0), scores.numel()
                )
        accumulated.sub_(output[:, :, start:stop].to(dtype) * row_direction.unsqueeze(-1))
        result[:, :, start:stop] = accumulated.to(output.dtype)
    _require(torch.isfinite(result).all(), 'Nonfinite attention tangent')
    return result


def attention_jvp_wrapper(native_attention, accounting):
    """Adapt only dual calls; ordinary calls use the original callable verbatim."""

    def wrapped(query, key, value, **kwargs):
        unpacked = [
            torch.autograd.forward_ad.unpack_dual(tensor) for tensor in (query, key, value)
        ]
        if all(pair.tangent is None for pair in unpacked):
            accounting['ordinary_calls'] = accounting.get('ordinary_calls', 0) + 1
            return native_attention(query, key, value, **kwargs)
        _require(kwargs.get('score_mod') is None, 'Only vanilla attention scores are supported')
        _require(
            not kwargs.get('return_lse') and kwargs.get('return_aux') is None,
            'The derivative wrapper owns native LSE collection',
        )
        options = kwargs.get('kernel_options') or {}
        _require(
            not options.get('PRESCALE_QK', False), 'Prescaled QK derivative is unsupported'
        )
        _require(
            set(kwargs)
            <= {
                'score_mod',
                'block_mask',
                'scale',
                'enable_gqa',
                'kernel_options',
                'return_lse',
                'return_aux',
            },
            'Unsupported native attention arguments',
        )
        primals = [pair.primal for pair in unpacked]
        tangents = [pair.tangent for pair in unpacked]
        native_kwargs = dict(kwargs, return_aux=AuxRequest(lse=True))
        output, auxiliary = native_attention(*primals, **native_kwargs)
        accounting['native_aux_forwards'] = accounting.get('native_aux_forwards', 0) + 1
        tangent = attention_tangent(
            *primals,
            *tangents,
            output=output,
            lse=auxiliary.lse,
            block_mask=kwargs.get('block_mask'),
            scale=kwargs.get('scale'),
            enable_gqa=kwargs.get('enable_gqa', False),
            accounting=accounting,
        )
        accounting['dual_calls'] = accounting.get('dual_calls', 0) + 1
        return torch.autograd.forward_ad.make_dual(output, tangent)

    return wrapped


@contextmanager
def native_attention_jvp():
    """Scope the training-only adapter to the author's DFlash call site."""
    module = importlib.import_module('specforge.modeling.draft.dflash')
    original = module.compile_friendly_flex_attention
    accounting = dict(
        ordinary_calls=0,
        dual_calls=0,
        native_aux_forwards=0,
        query_tiles=0,
        key_tiles=0,
        max_score_elements=0,
    )
    module.compile_friendly_flex_attention = attention_jvp_wrapper(original, accounting)
    try:
        yield accounting
    finally:
        module.compile_friendly_flex_attention = original
