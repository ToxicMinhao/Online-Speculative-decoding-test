"""CPU contracts for the pinned Transformers 5.12 full-attention cache adapter."""
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import DynamicCache, LlamaConfig, Qwen3Config

from dflash.model import _crop_to, _make_cache


@unittest.skipIf(hasattr(DynamicCache, "activate_past_recording"),
                 "Legacy compatibility applies to the pinned pre-recording API")
class LegacyCacheTests(unittest.TestCase):
    def make_cache(self):
        config = Qwen3Config(num_hidden_layers=2, hidden_size=8,
                             num_attention_heads=2, num_key_value_heads=2)
        return _make_cache(config)

    def prefill(self, cache, length=5):
        originals = []
        for i in range(2):
            key = torch.arange(length * 8, dtype=torch.float32).reshape(1, 2, length, 4) + i * 100
            value = key + 1000
            cache.update(key, value, i)
            originals.append((key, value))
        return originals

    def assert_contents(self, cache, expected):
        for layer, (key, value) in zip(cache.layers, expected):
            self.assertTrue(torch.equal(layer.keys, key))
            self.assertTrue(torch.equal(layer.values, value))
            self.assertEqual(layer.get_seq_length(), key.shape[-2])

    def test_real_cache_prefill_and_same_length_crop_is_noop(self):
        cache = self.make_cache()
        self.assertIsInstance(cache, DynamicCache)
        self.assertTrue(cache._dflash_legacy_absolute_crop)
        originals = self.prefill(cache)
        pointers = [(x.keys.data_ptr(), x.values.data_ptr()) for x in cache.layers]
        _crop_to(cache, 5)
        self.assert_contents(cache, originals)
        self.assertEqual(pointers, [(x.keys.data_ptr(), x.values.data_ptr()) for x in cache.layers])

    def test_partial_rewind_then_append_preserves_exact_prefix(self):
        cache = self.make_cache()
        originals = self.prefill(cache)
        _crop_to(cache, 3)
        prefixes = [(key[..., :3, :], value[..., :3, :]) for key, value in originals]
        self.assert_contents(cache, prefixes)
        expected = []
        for i, (key, value) in enumerate(prefixes):
            next_key = torch.full((1, 2, 2, 4), 2000.0 + i)
            next_value = next_key + 1000
            cache.update(next_key, next_value, i)
            expected.append((torch.cat((key, next_key), dim=-2),
                             torch.cat((value, next_value), dim=-2)))
        self.assert_contents(cache, expected)
        _crop_to(cache, 5)
        self.assert_contents(cache, expected)

    def test_zero_crop_empties_then_allows_fresh_append(self):
        cache = self.make_cache()
        _crop_to(cache, 0)
        self.prefill(cache)
        _crop_to(cache, 0)
        self.assertTrue(all(layer.get_seq_length() == 0 for layer in cache.layers))
        originals = self.prefill(cache, length=2)
        self.assert_contents(cache, originals)

    def test_invalid_forward_negative_and_noninteger_crops_do_not_mutate(self):
        cache = self.make_cache()
        originals = self.prefill(cache)
        for length in (6, -1, 2.5, True):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    _crop_to(cache, length)
                self.assert_contents(cache, originals)

    def test_unsupported_configs_are_rejected(self):
        def qwen(**changes):
            config = Qwen3Config(num_hidden_layers=2)
            for name, value in changes.items():
                setattr(config, name, value)
            return config

        configs = [
            LlamaConfig(num_hidden_layers=2),
            qwen(use_sliding_window=True),
            qwen(sliding_window=8),
            qwen(attention_chunk_size=8),
            qwen(layer_types=["full_attention", "sliding_attention"], sliding_window=8),
            qwen(layer_types=["full_attention", "linear_attention"]),
            qwen(layer_types=["full_attention"]),
            qwen(num_kv_shared_layers=1),
        ]
        for config in configs:
            with self.subTest(config=config.to_dict()):
                with self.assertRaisesRegex(ValueError, "full-attention Qwen3"):
                    _make_cache(config)


class RecordingCacheTests(unittest.TestCase):
    def test_modern_recording_activation_and_relative_crop_remain_unchanged(self):
        cache = Mock(spec=["activate_past_recording", "get_seq_length", "crop"])
        cache.get_seq_length.return_value = 5
        config = object()  # Modern upstream supports configurations beyond Qwen3.
        with patch("dflash.model.DynamicCache", return_value=cache) as factory:
            self.assertIs(_make_cache(config), cache)
        factory.assert_called_once_with(config=config)
        cache.activate_past_recording.assert_called_once_with()
        _crop_to(cache, 3)
        cache.crop.assert_called_with(-2)
        _crop_to(cache, 5)
        cache.crop.assert_called_with(0)


if __name__ == "__main__":
    unittest.main()
