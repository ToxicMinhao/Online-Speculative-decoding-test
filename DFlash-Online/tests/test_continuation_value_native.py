"""CPU qualification for the optimizer-free native endpoint adapter."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file

from experiments.continuation_value_diagnostic.native import (
    ReadOnlyEndpoint, compare_legacy, compare_repeated, load_export,
    qualify_endpoint, rng_identity,
)
from experiments.frontier_preservation_pilot.allocation_witness import block_readouts
from experiments.ok_coding.storage import identity


class DraftFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.draft_model = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.draft_model.weight.copy_(torch.eye(32))
        self.lm_head = nn.Identity()
        self.block_size, self.objective_chunk_blocks = 16, 8
        self.loss_dtype, self.eos_token_ids = torch.bfloat16, (29,)
        self.costs, self.shapes = {'existing': 7}, []
        self.register_buffer('scratch', torch.tensor([3]))
        self._cached_seq_len = 12
        self.fail = self.mutate = False
        self.lm_head.register_forward_pre_hook(lambda module, args: self.shapes.append(tuple(args[0].shape)))

    def _sample_anchor_positions(self, *args, **kwargs):
        raise AssertionError('Diagnostic must not sample anchors')

    def _forward_draft_blocks(self, input_ids, hidden_states, loss_mask):
        assert not torch.is_grad_enabled() and not self.training
        torch.rand(1)
        self.scratch.add_(1)
        self._cached_seq_len = 99
        self.costs['temporary'] = 1
        if self.fail:
            raise RuntimeError('fixture forward failed')
        if self.mutate:
            self.draft_model.weight.add_(1)
        anchors, keep = self._sample_anchor_positions()
        indices = anchors.unsqueeze(-1) + torch.arange(16)
        hidden = hidden_states[:, indices[0]].reshape(1, -1, 32)
        return anchors, keep, self.draft_model(hidden)

    def _block_labels_and_weights(self, input_ids, loss_mask, anchors, keep):
        indices = anchors.unsqueeze(-1) + torch.arange(16)
        return input_ids[:, indices[0]], loss_mask[:, indices[0]], indices


def fixture(root, *, blocks=9):
    labels = torch.arange(16).repeat(blocks).reshape(1, -1)
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[:, ::16] = 0
    proposals = labels.clone()
    proposals[:, 4::16] = proposals[:, 7::16] = 31
    hidden = torch.full((1, blocks * 16, 32), -2., dtype=torch.bfloat16)
    hidden.scatter_(-1, proposals.unsqueeze(-1), 2.)
    packet = Path(root) / 'features.pt'
    torch.save(dict(record_id='response', input_ids=labels, loss_mask=weights, hidden_states=hidden), packet)
    binding = dict(record_id='response', anchors=list(range(0, blocks * 16, 16)),
                   feature_path=str(packet), feature_identity=identity(packet))
    model = DraftFixture()
    adapter = ReadOnlyEndpoint(model, endpoint='parent', maximum_length=2272)
    return adapter, binding, hidden, labels, weights


class NativeReadOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_exact_chunk_geometry_state_and_no_backward(self):
        with TemporaryDirectory() as root:
            adapter, binding, _, _, _ = fixture(root)
            model = adapter.model
            model.lm_head.eval()
            before_rng = rng_identity()
            with patch('torch.autograd.backward', side_effect=AssertionError('unexpected backward')):
                result = adapter.read_response(binding, Path(root) / 'response.json')
            self.assertEqual(model.shapes, [(1, 8, 16, 32), (1, 1, 16, 32)])
            self.assertEqual(result['work']['head_calls_completed'], 2)
            self.assertEqual(result['work']['draft_forwards_completed'], 1)
            self.assertEqual(result['work']['blocks_completed'], 9)
            self.assertEqual(result['work']['backwards'], 0)
            self.assertEqual(result['work']['optimizer_steps'], 0)
            self.assertEqual(model.scratch.item(), 3)
            self.assertEqual(model._cached_seq_len, 12)
            self.assertEqual(model.costs, {'existing': 7})
            self.assertTrue(model.training)
            self.assertFalse(model.lm_head.training)
            self.assertEqual(rng_identity(), before_rng)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_old_geometry_exact_proposals_roles_margins_and_all_valid_slots(self):
        with TemporaryDirectory() as root:
            adapter, binding, hidden, labels, weights = fixture(root, blocks=8)
            result = adapter.read_response(binding, Path(root) / 'response.json')
            old = dict(record_id='response', anchors=binding['anchors'], blocks=block_readouts(
                hidden.reshape(1, 8, 16, 32), labels.reshape(1, 8, 16), weights.reshape(1, 8, 16)))
            self.assertEqual(compare_legacy(result, old)['status'], 'passed')
            self.assertEqual(len(result['blocks'][0]['positions']), 15)
            self.assertEqual(len(old['blocks'][0]['positions']), 6)
            changed = deepcopy(old)
            changed['blocks'][0]['positions'][0]['margin'] += 0.001
            self.assertEqual(compare_legacy(result, changed)['status'], 'failed')

    def test_forward_failure_saved_before_validation_and_state_restored(self):
        with TemporaryDirectory() as root:
            adapter, binding, *_ = fixture(root)
            model = adapter.model
            model.fail = True
            before_rng = rng_identity()
            output = Path(root) / 'failed.json'
            with self.assertRaisesRegex(RuntimeError, 'fixture forward failed'):
                adapter.read_response(binding, output)
            saved = json.loads(output.read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['work']['draft_forwards_attempted'], 1)
            self.assertEqual(saved['work']['draft_forwards_completed'], 0)
            self.assertEqual(saved['work']['head_calls_attempted'], 0)
            self.assertEqual(rng_identity(), before_rng)
            self.assertEqual(model.scratch.item(), 3)
            self.assertEqual(model._cached_seq_len, 12)

    def test_nonfinite_head_saved_as_completed_but_invalid_call(self):
        with TemporaryDirectory() as root:
            adapter, binding, *_ = fixture(root)
            adapter.model.lm_head.register_forward_hook(lambda module, args, output: output * float('nan'))
            output = Path(root) / 'failed.json'
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                adapter.read_response(binding, output)
            saved = json.loads(output.read_text())
            self.assertEqual(saved['work']['draft_forwards_completed'], 1)
            self.assertEqual(saved['work']['head_calls_attempted'], 1)
            self.assertEqual(saved['work']['head_calls_completed'], 1)
            self.assertEqual(saved['work']['blocks_completed'], 0)
            self.assertEqual(saved['work']['optimizer_steps'], 0)

    def test_parameter_mutation_is_a_failure_not_silent_restoration(self):
        with TemporaryDirectory() as root:
            adapter, binding, *_ = fixture(root)
            adapter.model.mutate = True
            with self.assertRaisesRegex(ValueError, 'changed parameters'):
                adapter.read_response(binding, Path(root) / 'failed.json')

    def test_empty_anchor_response_retained_without_forward(self):
        with TemporaryDirectory() as root:
            adapter, binding, *_ = fixture(root)
            binding['anchors'] = []
            result = adapter.read_response(binding, Path(root) / 'empty.json')
            self.assertEqual(result['blocks'], [])
            self.assertEqual(result['work']['draft_forwards_attempted'], 0)
            self.assertEqual(result['work']['response_reads_completed'], 1)
            self.assertEqual(adapter.close()['status'], 'passed')

    def test_qualification_is_four_reads_and_preserves_parity_failure(self):
        with TemporaryDirectory() as root:
            adapter, binding, hidden, labels, weights = fixture(root, blocks=8)
            old = dict(record_id='response', anchors=binding['anchors'], blocks=block_readouts(
                hidden.reshape(1, 8, 16, 32), labels.reshape(1, 8, 16), weights.reshape(1, 8, 16)))
            result = qualify_endpoint(adapter, [binding, binding], [old, old], binding, Path(root) / 'qual')
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['response_reads_completed'], 4)
            self.assertEqual(adapter.work['draft_forwards_completed'], 4)
            first = json.loads((Path(root) / 'qual/primary_0.json').read_text())
            second = json.loads((Path(root) / 'qual/primary_1.json').read_text())
            self.assertEqual(compare_repeated(first, second)['status'], 'passed')
            self.assertTrue(adapter.close()['parameters_unchanged'])
        with TemporaryDirectory() as root:
            adapter, binding, hidden, labels, weights = fixture(root, blocks=8)
            old = dict(record_id='response', anchors=binding['anchors'], blocks=block_readouts(
                hidden.reshape(1, 8, 16, 32), labels.reshape(1, 8, 16), weights.reshape(1, 8, 16)))
            old['blocks'][0]['current']['L'] += 1
            with self.assertRaisesRegex(ValueError, 'parity failed'):
                qualify_endpoint(adapter, [binding, binding], [old, old], binding, Path(root) / 'qual')
            saved = json.loads((Path(root) / 'qual/QUALIFICATION.json').read_text())
            self.assertEqual(saved['response_reads_completed'], 1)
            self.assertEqual(saved['comparisons'][0]['status'], 'failed')
            self.assertTrue((Path(root) / 'qual/legacy_0.json').is_file())
            self.assertFalse((Path(root) / 'qual/legacy_1.json').exists())

    def test_export_mapping_requires_exact_native_tensor_names_shapes_dtype(self):
        with TemporaryDirectory() as root:
            path = Path(root) / 'model.safetensors'
            draft = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            value = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
            save_file({'weight': value}, path)
            receipt = load_export(draft, path, identity(path))
            self.assertTrue(torch.equal(draft.weight, value))
            self.assertTrue(receipt['exact_copy'])
            with self.assertRaisesRegex(ValueError, 'authenticated'):
                load_export(draft, path, {'sha256': 'wrong', 'bytes': 0})
            for values, message in (({'wrong': value}, 'names'),
                                    ({'weight': value[:, :2].contiguous()}, 'shape'),
                                    ({'weight': value.float()}, 'dtype')):
                save_file(values, path)
                with self.assertRaisesRegex(ValueError, message):
                    load_export(draft, path, identity(path))

    def test_loader_parses_real_recipe_without_constructing_optimizer_or_trainer(self):
        with TemporaryDirectory() as root:
            model = DraftFixture()
            model.embed_tokens = nn.Embedding(32, 32, dtype=torch.bfloat16).requires_grad_(False)
            path = Path(root) / 'model.safetensors'
            save_file(model.draft_model.state_dict(), path)
            config = Path(__file__).parents[1] / 'configs/runs/dflash_online_osd_rerun_20261023/sealed/config.yaml'
            def bundle(cfg, **kwargs):
                self.assertEqual(cfg.model.torch_dtype, 'bfloat16')
                self.assertEqual(cfg.data.max_length, 2272)
                torch.rand(1)
                return SimpleNamespace(model=model)
            with patch('specforge.application.resolve_run', return_value=SimpleNamespace(algorithm=object())), \
                    patch('specforge.training.assembly.build_model_bundle', side_effect=bundle), \
                    patch('online_feedback.objective.TargetObjectiveModel', return_value=model) as objective, \
                    patch('online_feedback.training.ranking_eos_token_ids', return_value=(29,)), \
                    patch('specforge.training.assembly._optimizer_factory', side_effect=AssertionError('optimizer')), \
                    patch('online_feedback.optimizer.DirectBF16AdamW', side_effect=AssertionError('optimizer')), \
                    patch('torch.autograd.backward', side_effect=AssertionError('backward')):
                adapter = ReadOnlyEndpoint.load(config, path, identity(path), endpoint='child', device='cpu')
            self.assertEqual(adapter.maximum_length, 2272)
            self.assertEqual(objective.call_args.kwargs['loss_precision'], 'bfloat16')
            self.assertEqual(adapter.loading['rng_before'], adapter.loading['rng_after'])
            self.assertEqual(adapter.loading['startup_forwards'], 0)
            self.assertFalse(adapter.loading['optimizer_created'])
            self.assertEqual(adapter.close()['status'], 'passed')

    def test_retained_readout_is_not_overwritten(self):
        with TemporaryDirectory() as root:
            adapter, binding, *_ = fixture(root)
            path = Path(root) / 'response.json'
            adapter.read_response(binding, path)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                adapter.read_response(binding, path)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
