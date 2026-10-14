"""Cap-256 admission, EOS and final partial-block regressions; no GPU imports."""
import ast
import copy
import hashlib
import inspect
import os
from pathlib import Path
import sys
import textwrap
from types import ModuleType
import unittest
from unittest import mock

import experiment as E
import native as N
import training as T
from test_native import native_record


def terminal_record(output_length):
    """All-nine accepted proposals; the final native block is capped or EOS-clipped."""
    doc = native_record()
    doc['output_ids'], doc['blocks'] = [1000], []
    while len(doc['output_ids']) < output_length:
        offset = len(doc['output_ids'])
        anchor = len(doc['input_ids']) + offset - 1
        seed = doc['output_ids'][-1]
        candidates = list(range(seed, seed + 10))
        predictions = list(range(seed + 1, seed + 11))
        emitted = predictions[:min(10, output_length - offset)]
        finished = offset + len(emitted) == output_length
        end = len(doc['input_ids']) + (output_length if finished else N.OUTPUT_CAP)
        mask = [False] + [anchor + slot < end for slot in range(1, 10)]
        doc['blocks'].append({'block_index': len(doc['blocks']), 'anchor': anchor,
            'context_length': anchor, 'positions': list(range(anchor, anchor + 10)),
            'teacher_row_for_slot': [None] + list(range(9)),
            'teacher_prediction_positions': list(range(anchor + 1, anchor + 11)),
            'noise_input_ids': [seed] + [128002] * 9,
            'verification_input_token_ids': candidates, 'native_top1': predictions,
            'accepted_before_stop': 9, 'bonus_or_correction': predictions[-1],
            'outcome': {'output_offset': offset, 'committed_token_ids': emitted, 'finished': finished},
            'eligible_mask': mask, 'eligible_positions': sum(mask),
            'attention': 'all_original_context_and_all_same_block_noise_slots'})
        doc['output_ids'].extend(emitted)
    doc['output_ids_sha256'] = N.object_hash(doc['output_ids'])
    doc['block_count'] = doc['eligible_block_count'] = len(doc['blocks'])
    doc['eligible_position_count'] = sum(b['eligible_positions'] for b in doc['blocks'])
    doc['context_feature_rows'] = len(doc['input_ids']) + output_length - 1
    N.add_response_ownership(doc)
    return doc


class OutputCapTests(unittest.TestCase):
    def test_controller_native_training_share_fixed_256_cap(self):
        self.assertEqual((E.OUTPUT_CAP, N.OUTPUT_CAP, T.OUTPUT_CAP), (256, 256, 256))
        self.assertEqual((N.INPUT_LIMIT, N.CONTEXT_LIMIT, N.PREFILL_CHUNK_SIZE, N.BLOCK_SIZE),
                         (31744, 32768, 8192, 10))
        doc = terminal_record(256)
        self.assertEqual(T.audit_response(doc)['eligible_positions'], 230)
        self.assertEqual(doc['blocks'][-1]['owned_eligible_slots'], [1, 2, 3, 4, 5])
        old = copy.deepcopy(doc); old['output_cap'] = 512
        with self.assertRaisesRegex(ValueError, 'mixed-cap feedback'):
            T.audit_response(old)

    def test_controller_refuses_response_above_declared_cap(self):
        result = {'output_ids': [1] * 257, 'text': 'too long'}
        with self.assertRaisesRegex(ValueError, 'Output cap'):
            E.valid_result(result, [7], 'F', False)


@unittest.skipUnless(os.environ.get('TWO_UPDATE_WORKSPACE'),
                     'Installed-source cap checks need TWO_UPDATE_WORKSPACE')
class InstalledOutputCapTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(os.environ['TWO_UPDATE_WORKSPACE'])
        self.saved = {name: sys.modules.get(name) for name in
                      ('records', 'cache', 'tv_d202_serving', 'trajectory_trace')}

    def tearDown(self):
        for name, original in self.saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def test_actual_collector_cap_and_eos_masks_match_training(self):
        records, cache = N.capture_modules(self.root)
        self.assertEqual((records.OUTPUT_CAP, cache.OUTPUT_CAP), (256, 256))
        for length, final_slots in ((256, [1, 2, 3, 4, 5]), (4, [1, 2, 3])):
            doc = terminal_record(length)
            records.validate(doc)
            T.audit_response(doc)
            last = doc['blocks'][-1]
            self.assertEqual(last['owned_eligible_slots'], final_slots)
            self.assertEqual(last['eligible_mask'], records.eligibility(
                last['anchor'], len(doc['input_ids']), 256, last['outcome']))
            malformed = copy.deepcopy(doc)
            malformed['blocks'][-1]['eligible_mask'][-1] = True
            with self.assertRaisesRegex(ValueError, 'Eligibility changed'):
                records.validate(malformed)
        old = terminal_record(256); old['output_cap'] = 512
        with self.assertRaises(ValueError):
            records.validate(old)

    def test_wrapper_256_branch_and_real_trace_constructor_are_source_bound(self):
        tv_path = self.root / 'research/target_variant_specialization_001/native_tools/tv_d202_serving.py'
        trace_path = self.root / 'research/multiturn_adaptation_d202/runtime_001/trajectory_trace.py'
        original_hashes = (N.sha(tv_path), N.sha(trace_path))
        trace = ModuleType('trajectory_trace'); trace.require = N.require
        tree = ast.parse(trace_path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RequestTrace')
        constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        exec(compile(ast.Module(body=[constructor], type_ignores=[]), str(trace_path), 'exec'), trace.__dict__)
        trace.RequestTrace = type('RequestTrace', (), {'__init__': trace.__init__})
        sys.modules['trajectory_trace'] = trace
        sys.modules.pop('tv_d202_serving', None)

        def exact_edit(function, before, after, namespace):
            source = textwrap.dedent(inspect.getsource(function))
            self.assertEqual(source.count(before), 1)
            updated = source.replace(before, after)
            local = {}; exec(compile(updated, function.__code__.co_filename, 'exec'), namespace, local)
            return local[function.__name__], {'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
                                             'effective_sha256': hashlib.sha256(updated.encode()).hexdigest()}

        def metadata_only_load(path, name, replacements):
            # Execute the actual transformed cap declarations/branch, omitting
            # unrelated heavyweight serving imports from this CPU contract test.
            source = Path(path).read_text()
            for before, after in replacements:
                self.assertEqual(source.count(before), 1)
                source = source.replace(before, after)
            tree = ast.parse(source)
            nodes = [node for node in tree.body if
                     (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                       t.id in ('CAP', 'CAP_EDITS') for t in node.targets)) or
                     (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
                       ast.unparse(node.value).startswith('require(CAP')) or
                     (isinstance(node, ast.If) and ast.unparse(node.test).startswith('CAP'))]
            module = ModuleType(name)
            module.os, module.require, module.exact_guard_edit = os, N.require, exact_edit
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
            module.two_update_compatibility = {'source_sha256': N.sha(path),
                'effective_sha256': hashlib.sha256(source.encode()).hexdigest(), 'edits': replacements}
            sys.modules[name] = module
            return module

        with mock.patch.dict(os.environ, {'TV_OUTPUT_CAP': '256'}), \
                mock.patch.object(N, '_load', side_effect=metadata_only_load):
            module = N.serving_module(self.root)
            self.assertIs(N.serving_module(self.root), module)
        self.assertEqual(module.CAP, 256)
        self.assertEqual(len(module.CAP_EDITS), 1)
        context = {'request_index': 1, 'conversation_id': 'fixture', 'turn_index': 1,
                   'kind': 'qualification', 'draft_version': 0}
        self.assertEqual(trace.RequestTrace(context, [7], 256).max_new_tokens, 256)
        with self.assertRaisesRegex(ValueError, 'Unqualified output cap'):
            trace.RequestTrace(context, [7], 257)
        self.assertEqual((N.sha(tv_path), N.sha(trace_path)), original_hashes)


if __name__ == '__main__':
    unittest.main()
