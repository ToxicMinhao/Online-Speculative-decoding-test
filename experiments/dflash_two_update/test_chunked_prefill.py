"""Recorder contract regressions, without models, CUDA, or dense large tensors.

Set TWO_UPDATE_WORKSPACE to the qualified dependency checkout. These tests run
the actual compatibility-loaded recorder with lightweight immutable CPU tensor
stand-ins. They check chronology, row identity, shape/dtype and byte accounting;
native numerical/output preservation still requires the separate GPU check.
"""
import os
import sys
from types import ModuleType
import unittest
from unittest import mock

import native as N


class Features:
    """Represent a constant-valued row by one integer, retaining logical width."""
    def __init__(self, rows, width=20480, dtype='bfloat16', ndim=2):
        self.rows = tuple(rows)
        self.width, self.dtype, self.ndim = width, dtype, ndim
        self.shape = (len(self.rows), width) if ndim == 2 else (len(self.rows),)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return Features(self.rows[index], self.width, self.dtype, self.ndim)

    def numel(self):
        return len(self.rows) * self.width

    def element_size(self):
        return 2 if self.dtype == 'bfloat16' else 4

    def detach(self):
        return self

    def to(self, *, device, copy, non_blocking):
        if (device, copy, non_blocking) != ('cpu', True, False):
            raise AssertionError('Recorder copy semantics changed')
        return Features(self.rows, self.width, self.dtype, self.ndim)

    def contiguous(self):
        return self


def tensor_api():
    module = ModuleType('torch')
    for dtype in ('bfloat16', 'float32', 'int64', 'int32'):
        setattr(module, dtype, dtype)
    module.equal = lambda left, right: (
        left.shape == right.shape and left.dtype == right.dtype and
        left.rows == right.rows)
    return module


@unittest.skipUnless(os.environ.get('TWO_UPDATE_WORKSPACE'),
                     'Installed-source recorder checks need TWO_UPDATE_WORKSPACE')
class ChunkedPrefillTests(unittest.TestCase):
    def setUp(self):
        self.saved = {name: sys.modules.get(name) for name in ('records', 'cache')}
        self.records, self.cache = N.capture_modules(os.environ['TWO_UPDATE_WORKSPACE'])
        self.torch_patch = mock.patch.dict(sys.modules, {'torch': tensor_api()})
        self.torch_patch.start()

    def tearDown(self):
        self.torch_patch.stop()
        for name, original in self.saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def capture(self, size, max_bytes=2 * 2**30):
        return self.cache.ResponseCapture(
            {'role': N.ROLE, 'draft_version': 'F'}, list(range(size)),
            max_bytes=max_bytes)

    def append(self, capture, start, end):
        capture.append_context(Features(range(start, end)), list(range(start, end)))

    def draft(self, capture):
        capture.draft(noise_ids=[101] + [128002] * 9,
                      positions=list(range(capture.context_rows, capture.context_rows + 10)),
                      candidates=list(range(101, 111)), mask_token_id=128002)

    def verified_branch(self, capture):
        self.draft(capture)
        anchor = capture.context_rows
        capture.before_verify(Features(range(10), width=4096),
                              Features(range(anchor, anchor + 10)))
        capture.native_labels(list(range(101, 111)), [102, 999] + [200] * 8,
                              accepted=1, correction=999)
        capture.after_verify(output_before=[101], output_after=[101, 102, 999],
                             finished=False, commit_rows=2)

    def test_single_and_chunked_prefill_keep_identical_rows_and_accounting(self):
        # 8431 reproduces the failed turn. Longer cases cover the 32K extension.
        for size, chunks in ((8192, [8192]), (8431, [8192, 239]),
                             (16384, [8192, 8192]), (8431, [4096, 4096, 239]),
                             (25930, [8192, 8192, 8192, 1354]),
                             (31744, [8192, 8192, 8192, 7168])):
            with self.subTest(size=size, chunks=chunks):
                full, split = self.capture(size), self.capture(size)
                self.append(full, 0, size)
                cursor = 0
                for count in chunks:
                    self.append(split, cursor, cursor + count)
                    cursor += count
                full_rows = tuple(row for group in full.groups for row in group.rows)
                split_rows = tuple(row for group in split.groups for row in group.rows)
                self.assertEqual(split_rows, full_rows)
                self.assertEqual(split.context_rows, size)
                self.assertEqual(split.bytes, full.bytes)
                self.assertEqual(split.bytes, size * 20480 * 2)
                self.assertEqual(split.peak_bytes, full.peak_bytes)
                self.assertEqual(split.costs['captured_device_to_host_bytes'], split.bytes)
                self.assertEqual(split.costs['extra_teacher_passes'], 0)
                self.draft(split)
                self.assertEqual(split.active['anchor'], size)

    def test_gap_overlap_duplicate_or_future_rows_are_rejected(self):
        for start, end in ((0, 8192), (8191, 8430), (8193, 8431), (8431, 8440)):
            with self.subTest(start=start, end=end):
                capture = self.capture(8431)
                self.append(capture, 0, 8192)
                with self.assertRaises(ValueError):
                    self.append(capture, start, end)
                self.assertEqual(capture.context_rows, 8192)

    def test_initial_overrun_and_unverified_append_after_prefill_are_rejected(self):
        for size, first_end, second_end in ((8431, 0, 8432),
                                           (8431, 8192, 8432),
                                           (8192, 8192, 8193)):
            with self.subTest(size=size, first_end=first_end, second_end=second_end):
                capture = self.capture(size)
                if first_end:
                    self.append(capture, 0, first_end)
                with self.assertRaises(ValueError):
                    self.append(capture, first_end, second_end)
                self.assertEqual(capture.context_rows, first_end)

    def test_draft_is_rejected_until_every_input_feature_has_arrived(self):
        for rows in (0, 8192):
            with self.subTest(rows=rows):
                capture = self.capture(8431)
                if rows:
                    self.append(capture, 0, rows)
                with self.assertRaises(ValueError):
                    self.draft(capture)
                self.assertIsNone(capture.active)

    def test_chunk_shape_dtype_and_position_count_remain_strict(self):
        for raw, positions in ((Features(range(2), width=4096), [0, 1]),
                               (Features(range(2), ndim=1), [0, 1]),
                               (Features(range(2), dtype='float32'), [0, 1]),
                               (Features(range(2)), [0]),
                               (Features(range(2)), [0, 2])):
            with self.subTest(shape=raw.shape, dtype=raw.dtype, positions=positions):
                capture = self.capture(8431)
                with self.assertRaises(ValueError):
                    capture.append_context(raw, positions)
                self.assertEqual(capture.context_rows, 0)
                self.assertEqual(capture.bytes, 0)

    def test_capture_budget_accumulates_across_chunks_without_truncation(self):
        capture = self.capture(8431, max_bytes=8192 * 20480 * 2)
        self.append(capture, 0, 8192)
        with self.assertRaisesRegex(ValueError, 'memory budget exceeded'):
            self.append(capture, 8192, 8431)
        self.assertEqual(capture.context_rows, 8192)
        self.assertEqual(len(capture.groups), 1)
        self.assertEqual(capture.costs['captured_device_to_host_bytes'], capture.bytes)

    def test_verified_commit_still_requires_exact_branch_rows(self):
        capture = self.capture(8431)
        self.append(capture, 0, 8192)
        self.append(capture, 8192, 8431)
        self.verified_branch(capture)
        self.append(capture, 8431, 8433)
        self.assertIsNone(capture.last_branch)
        self.assertEqual(capture.context_rows, 8433)
        self.assertEqual(capture.groups[-1].rows, (8431, 8432))
        # The correction has not yet been processed and cannot be appended.
        with self.assertRaises(ValueError):
            self.append(capture, 8433, 8434)
        self.draft(capture)
        self.assertEqual(capture.active['anchor'], 8433)

    def test_verified_commit_rejects_rejected_or_wrong_feature_rows(self):
        for rows in ((8192,), (8192, 8193, 8194), (8192, -1)):
            with self.subTest(rows=rows):
                capture = self.capture(8192)
                self.append(capture, 0, 8192)
                self.verified_branch(capture)
                with self.assertRaises(ValueError):
                    capture.append_context(Features(rows), list(range(8192, 8192 + len(rows))))
                self.assertEqual(capture.context_rows, 8192)
                self.assertIsNotNone(capture.last_branch)


if __name__ == '__main__':
    unittest.main()
