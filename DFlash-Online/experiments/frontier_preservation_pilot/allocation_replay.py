"""Replay fixed preservation weights through the unchanged native K+PAIR loss.

The binding uses tensor slice offsets, not a mutable chunk counter. Native head
checkpoint recomputation therefore reads the same assignment in any order.
"""

from contextlib import contextmanager

import torch

from online_feedback.prefix_allocation import PrefixAllocation, allocation_summaries
from .math import current_roles, require


class CachedPrefixAllocation(PrefixAllocation):
    """One response's detached allocation; no preview loss enters training."""

    def __init__(self, snapshot, coefficients):
        super().__init__()
        self.snapshot = snapshot
        self.coefficients = coefficients.detach()
        self.contract = dict(name='frontier_preservation_v1_cached_native_replay',
                             numerical_contract='native_token_contraction_v2')
        self.labels = None
        roles = snapshot['roles']
        require(coefficients.dtype == torch.float32
                and coefficients.shape == roles['beta'].shape,
                'Cached coefficients must use the original FP32 response shape')
        require(torch.equal(coefficients[~roles['prefix']], roles['beta'][~roles['prefix']]),
                'Cached allocation changed non-prefix positions')

    def bind_labels(self, labels, weights):
        roles = self.snapshot['roles']
        require(torch.equal(labels, roles['labels']) and torch.equal(weights, roles['beta']),
                'Native replay changed labels, validity or positional decay')
        require(labels.ndim == 3 and labels.shape[0] == 1 and labels.is_contiguous(),
                'Require native single-response contiguous block labels')
        self.labels = labels

    def chunk_slice(self, labels):
        require(self.labels is not None, 'Bind native full-response labels before projection')
        require(labels.untyped_storage().data_ptr() == self.labels.untyped_storage().data_ptr(),
                'Head chunk no longer refers to the bound native label tensor')
        offset = labels.storage_offset() - self.labels.storage_offset()
        width = self.labels.shape[-1]
        require(offset >= 0 and offset % width == 0, 'Invalid native block slice offset')
        start = offset // width
        stop = start + labels.shape[1]
        require(stop <= self.labels.shape[1]
                and torch.equal(labels, self.labels[:, start:stop]), 'Invalid head chunk binding')
        return slice(start, stop)

    def chunk_terms(self, log_probability, token_loss, proposals, labels, weights,
                    selection, eos_token_ids=()):
        part = self.chunk_slice(labels)
        snapshot = self.snapshot
        roles = snapshot['roles']
        current = current_roles(proposals, labels, weights, eos_token_ids)
        require(torch.equal(proposals, snapshot['proposals'][:, part])
                and torch.equal(current['prefix'], roles['prefix'][:, part])
                and torch.equal(current['repair'], roles['repair'][:, part]),
                'Current native teaching no longer matches the frozen pre-step roles')
        base = weights.detach().float()
        assigned = self.coefficients[:, part]
        summaries = allocation_summaries(base, assigned, current['valid'], current['prefix'])
        return ((assigned - base) * token_loss).sum(), *summaries


@contextmanager
def replay_allocation(model, snapshot, coefficients):
    """Install only the coefficient lookup and restore it even after failure.

    Caller must keep this context alive through the response backward: native
    checkpoint recomputation calls the extension again. The trainer separately
    validates its explicit new fork contract; this is not the historical C rule.
    """
    extension = CachedPrefixAllocation(snapshot, coefficients)
    previous = model.prefix_allocation
    original_labels = model._block_labels_and_weights

    def bound_labels(*args, **kwargs):
        labels, weights, indices = original_labels(*args, **kwargs)
        extension.bind_labels(labels, weights)
        return labels, weights, indices

    model.prefix_allocation = extension
    model._block_labels_and_weights = bound_labels
    try:
        yield extension
    finally:
        model.prefix_allocation = previous
        model._block_labels_and_weights = original_labels
