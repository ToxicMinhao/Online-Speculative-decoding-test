"""Cached coefficients survive partial chunks and backward replay order."""

from types import SimpleNamespace
import unittest

import torch

from .allocation_replay import CachedPrefixAllocation, replay_allocation
from .math import allocate_prefix, current_roles


def fixture():
    labels = torch.arange(16).expand(1, 11, 16).clone()
    proposals = labels.clone()
    proposals[:, :, 4] = 99
    beta = torch.exp(-torch.arange(16).float() / 7).expand_as(labels).clone()
    beta[:, :, 0] = 0
    roles = current_roles(proposals, labels, beta)
    coefficients, _ = allocate_prefix(labels.float(), beta, roles['prefix'])
    return dict(roles=roles, proposals=proposals), coefficients


class AllocationReplayTests(unittest.TestCase):
    def test_chunks_use_bound_offsets_and_are_independent_of_recomputation_order(self):
        snapshot, coefficients = fixture()
        roles = snapshot['roles']
        extension = CachedPrefixAllocation(snapshot, coefficients)
        extension.bind_labels(roles['labels'], roles['beta'])
        losses = torch.randn_like(coefficients, requires_grad=True)
        results = []
        for part in (slice(8, 11), slice(0, 8), slice(8, 11)):
            results.append(extension.chunk_terms(
                None, losses[:, part], snapshot['proposals'][:, part],
                roles['labels'][:, part], roles['beta'][:, part],
                (roles['prefix'] | roles['repair'])[:, part]))
        self.assertTrue(torch.equal(results[0][0], results[2][0]))
        total = results[0][0] + results[1][0]
        gradient, = torch.autograd.grad(total, losses)
        self.assertTrue(torch.equal(gradient, coefficients - roles['beta']))
        self.assertTrue(torch.equal(gradient[~roles['prefix']], torch.zeros_like(gradient[~roles['prefix']])))

    def test_coefficients_are_detached_and_beta_has_no_auxiliary_gradient(self):
        snapshot, _ = fixture()
        roles = snapshot['roles']
        coefficients = roles['beta'].clone().requires_grad_(True)
        extension = CachedPrefixAllocation(snapshot, coefficients)
        extension.bind_labels(roles['labels'], roles['beta'])
        losses = torch.randn_like(coefficients, requires_grad=True)
        terms = extension.chunk_terms(None, losses, snapshot['proposals'], roles['labels'],
                                      roles['beta'], roles['prefix'] | roles['repair'])
        native = losses.sum()
        reduced, _ = extension.reduce_response(native, terms)
        self.assertIs(reduced, native)
        reduced.backward()
        self.assertIsNone(coefficients.grad)

    def test_changed_prediction_or_foreign_label_storage_is_rejected(self):
        snapshot, coefficients = fixture()
        roles = snapshot['roles']
        extension = CachedPrefixAllocation(snapshot, coefficients)
        extension.bind_labels(roles['labels'], roles['beta'])
        with self.assertRaisesRegex(ValueError, 'bound native label'):
            extension.chunk_slice(roles['labels'].clone())
        proposals = snapshot['proposals'].clone()
        proposals[0, 0, 1] = 99
        with self.assertRaisesRegex(ValueError, 'frozen pre-step roles'):
            extension.chunk_terms(None, torch.zeros_like(coefficients), proposals,
                                  roles['labels'], roles['beta'], roles['prefix'] | roles['repair'])

    def test_scope_restores_original_labels_and_extension_after_failure(self):
        snapshot, coefficients = fixture()
        roles = snapshot['roles']
        original = lambda: (roles['labels'], roles['beta'], None)
        previous = object()
        model = SimpleNamespace(prefix_allocation=previous, _block_labels_and_weights=original)
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with replay_allocation(model, snapshot, coefficients) as extension:
                model._block_labels_and_weights()
                self.assertIs(model.prefix_allocation, extension)
                raise RuntimeError('fixture')
        self.assertIs(model.prefix_allocation, previous)
        self.assertIs(model._block_labels_and_weights, original)


if __name__ == '__main__':
    unittest.main()
