"""Bounded controller regression tests; no GPUs, model weights or source data."""
import copy
import unittest

from . import experiment as e


def checkpoint(label):
    return dict(checkpoint_id=label, checkpoint_identity_sha256=e.digest(label),
                path='/fixture/' + label)


def reference(name='wildchat_00'):
    canonical = []
    for turn, ids, output in [(1, [100, 11, 101], [21, 22, 23, 128009]),
                              (2, [100, 11, 101, 21, 22, 23, 128009, 12, 101],
                               [31, 32, 33, 128009])]:
        canonical.append(dict(turn=turn, input_ids=ids, output_ids=output,
            input_ids_sha256=e.digest(ids), checkpoint_id='F' if turn == 1 else 'D1',
            checkpoint_identity_sha256=e.digest('F' if turn == 1 else 'D1'),
            source_user_text='  code\n    whitespace\n' + str(turn)))
    scores = []
    for turn, labels in [(1, ['F']), (2, ['F', 'D1'])]:
        for label in labels:
            for repeat in range(3):
                scores.append(dict(turn=turn, checkpoint_id=label, repeat=repeat,
                    same_generation_valid=True, input_ids=canonical[turn-1]['input_ids'][:],
                    output_ids=canonical[turn-1]['output_ids'][:],
                    verification_calls=3 if label == 'D1' else 2))
    return dict(episode_id=name, source='personamem' if name.startswith('personamem') else 'wildchat',
        synthetic=name.startswith('personamem'), canonical=canonical,
        original_scores=scores, original_update=dict(accounting=dict(
            optimizer_steps_this_event=2, anchor_exposure_sha256=e.digest('same anchors'))),
        original_result_binding=dict(path='/fixture/RESULT.json', sha256=e.digest('result')))


class FakeBackend:
    """A backend exposing only fixed-input replay and independent branch training."""
    def __init__(self, source, *, change_update=None, change_output=None, change_support=None):
        self.reference = copy.deepcopy(source)
        self.change_update = change_update or (lambda label, update: None)
        self.change_output = change_output or (lambda turn, repeat, label, output: None)
        self.change_support = change_support or (lambda support: None)
        self.checkpoints = {'F': checkpoint('F')}
        self.calls = []
        self.updates = []
        self.events = []
        self.supports = []
        self.closed = False

    def start_episode(self, name):
        self.events.append(('start', name))
        return dict(status='passed', startup_s=0.25)

    def prepare_support(self, item):
        self.events.append(('support', item['turn']))
        support = dict(record_id=self.reference['episode_id'] + ':t01', turn=1,
            input_ids=list(item['input_ids']), output_ids=list(item['output_ids']))
        self.change_support(support)
        self.supports.append(copy.deepcopy(support))
        return support

    def train_branch(self, label, learning_rate, support, original_update):
        self.events.append(('train', label))
        start, count = len(support['input_ids']), len(support['output_ids'])
        update = dict(checkpoint=checkpoint(label), checkpoint_id=label,
            fresh_initialization_verified=True, original_exposure_match=True,
            initialization=dict(released_checkpoint='F', optimizer_steps=0, seed=17),
            accounting=dict(recipe=dict(learning_rate=learning_rate, seed=17, passes=2,
                responses_per_step=1, loss='forward_KL'),
                optimizer_steps_total=2, optimizer_steps_this_event=2,
                response_presentations_total=2, response_presentations_this_event=2,
                support_turns=[1],
                anchor_exposure_sha256=original_update['accounting']['anchor_exposure_sha256'],
                response_audits=[dict(loss_mask_sha256=e.digest([0]*start + [1]*count + [0]*9))]),
            training_s=0.2, export_s=0.1)
        self.change_update(label, update)
        self.checkpoints[label] = update['checkpoint']
        self.updates.append(copy.deepcopy(update))
        return update

    def serve(self, ids, label, *, capture, turn_index, repeat):
        self.calls.append((turn_index, repeat, label, list(ids), capture))
        self.events.append(('serve', turn_index, repeat, label))
        item = self.reference['canonical'][turn_index-1]
        output = dict(input_ids=list(ids), input_ids_sha256=e.digest(ids),
            output_ids=item['output_ids'][:], text='saved-output fixture',
            checkpoint_identity_sha256=self.checkpoints[label]['checkpoint_identity_sha256'],
            verification_calls=3 if label == 'D1' else 2,
            accepted_draft_tokens=1, verification_committed_tokens=3,
            request_s=1.0 + repeat * .01, activation_s=.2, native_activation_s=.1,
            cache_maintenance_s=.01, feedback=None, capture_costs=None, capture=False)
        self.change_output(turn_index, repeat, label, output)
        return output

    def finish_episode(self):
        self.closed = True
        self.events.append(('finish',))
        return dict(status='passed')


class FirstUpdateControllerTests(unittest.TestCase):
    def run_one(self, **backend_options):
        ref = reference()
        backend = FakeBackend(ref, **backend_options)
        events, snapshots = [], []
        result = e.run_stream(ref, backend,
            emit=lambda kind, row: events.append((kind, copy.deepcopy(row))),
            snapshot=lambda row: snapshots.append(copy.deepcopy(row)))
        return result, backend, events, snapshots

    def test_complete_bound_no_new_training_answer_or_history(self):
        ref = reference()
        before = copy.deepcopy(ref)
        backend = FakeBackend(ref)
        result = e.run_stream(ref, backend)
        self.assertEqual(result['status'], 'complete', result.get('error'))
        self.assertEqual(ref, before)
        self.assertEqual(result['canonical'], before['canonical'])
        self.assertEqual(len(backend.supports), 1)
        self.assertEqual(backend.supports[0]['input_ids'], before['canonical'][0]['input_ids'])
        self.assertEqual(backend.supports[0]['output_ids'], before['canonical'][0]['output_ids'])
        self.assertEqual(len(result['updates']), 2)
        self.assertEqual(len(result['scores']), 18)
        self.assertEqual(len(backend.calls), 18)
        self.assertTrue(backend.closed)
        self.assertTrue(all(row['training_data'] is False for row in result['scores']))
        self.assertEqual({row['probe'] for row in result['scores'] if row['turn'] == 1}, {'in_sample_fit'})
        self.assertEqual({row['probe'] for row in result['scores'] if row['turn'] == 2}, {'next_request_transfer'})

    def test_two_independent_fresh_two_step_receipts(self):
        result, backend, _, _ = self.run_one()
        self.assertEqual(result['status'], 'complete', result.get('error'))
        self.assertEqual([u['checkpoint_id'] for u in result['updates']], ['D1', 'L1'])
        self.assertEqual([u['accounting']['recipe']['learning_rate'] for u in result['updates']], [3e-4, 1e-4])
        for update in result['updates']:
            self.assertEqual(update['initialization'], dict(released_checkpoint='F', optimizer_steps=0, seed=17))
            self.assertEqual(update['accounting']['optimizer_steps_total'], 2)
            self.assertEqual(update['accounting']['support_turns'], [1])
        self.assertEqual(result['exposure_checks']['optimizer_steps'], 4)
        last_training = max(i for i, event in enumerate(backend.events) if event[0] == 'train')
        first_replay = min(i for i, event in enumerate(backend.events) if event[0] == 'serve')
        self.assertLess(last_training, first_replay)

    def test_exact_inputs_capture_disabled_and_rotating_orders(self):
        result, backend, events, _ = self.run_one()
        for turn, repeat, label, ids, capture in backend.calls:
            self.assertEqual(ids, backend.reference['canonical'][turn-1]['input_ids'])
            self.assertFalse(capture)
            self.assertIn(repeat, range(3))
        for turn in (1, 2):
            orders = [[c[2] for c in backend.calls if c[0] == turn and c[1] == repeat] for repeat in range(3)]
            self.assertEqual(orders, [e.benchmark_order(turn, repeat) for repeat in range(3)])
            for position in range(3):
                self.assertEqual({order[position] for order in orders}, {'F', 'D1', 'L1'})
        self.assertEqual(backend.calls[0][:3], (2, 0, 'D1'))
        gate = next(i for i, item in enumerate(events) if item[0] == 'reproduction_gate')
        first_low = next(i for i, item in enumerate(events) if item[0] == 'score' and item[1]['checkpoint_id'] == 'L1')
        self.assertLess(gate, first_low)
        self.assertEqual(result['reproduction_check']['status'], 'passed')

    def test_original_high_rate_count_mismatch_stops_before_low_rate_replay(self):
        def change(turn, repeat, label, output):
            if turn == 2 and label == 'D1':
                output['verification_calls'] += 1
        result, backend, events, snapshots = self.run_one(change_output=change)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('Original request-2 verification counts not reproduced', result['error'])
        self.assertEqual(len(backend.calls), 1)
        self.assertFalse(any(call[2] == 'L1' for call in backend.calls))
        self.assertEqual(len(result['scores']), 1)
        self.assertFalse(result['scores'][0]['original_verification_count_match'])
        self.assertFalse(any(kind == 'reproduction_gate' for kind, _ in events))
        self.assertTrue(backend.closed)
        self.assertEqual(snapshots[-1]['status'], 'failed')

    def test_output_input_and_checkpoint_mismatches_retained_then_halt(self):
        changes = {
            'output': lambda row: row['output_ids'].__setitem__(0, 999),
            'input': lambda row: row['input_ids'].__setitem__(0, 999),
            'identity': lambda row: row.update(checkpoint_identity_sha256=e.digest('other')),
        }
        for name, mutate in changes.items():
            with self.subTest(name=name):
                result, backend, _, _ = self.run_one(change_output=lambda t, r, l, row: mutate(row))
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['scores'][0]['same_generation_valid'])
                self.assertEqual(len(backend.calls), 1)
                self.assertTrue(backend.closed)

    def test_wrong_mask_exposure_or_carried_optimizer_fail_before_benchmarks(self):
        mutations = {
            'mask': lambda u: u['accounting']['response_audits'][0].update(loss_mask_sha256=e.digest([1])),
            'exposure_receipt': lambda u: u.update(original_exposure_match=False),
            'fresh_state_receipt': lambda u: u.update(fresh_initialization_verified=False),
            'carried_steps': lambda u: u['accounting'].update(optimizer_steps_total=4),
            'anchors': lambda u: u['accounting'].update(anchor_exposure_sha256=e.digest('different')),
            'rate': lambda u: u['accounting']['recipe'].update(learning_rate=.001),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                result, backend, _, _ = self.run_one(change_update=lambda label, update: mutate(update) if label == 'L1' else None)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(backend.calls, [])
                self.assertTrue(backend.closed)

    def test_modified_saved_support_fails_before_any_training(self):
        result, backend, _, _ = self.run_one(change_support=lambda support: support['output_ids'].__setitem__(0, 9))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('Saved feedback changed', result['error'])
        self.assertEqual(backend.updates, [])
        self.assertEqual(backend.calls, [])

    def test_capture_in_measurement_is_rejected(self):
        result, backend, _, _ = self.run_one(change_output=lambda t, r, l, row: row.update(feedback={'invalid': True}))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('disable feedback capture', result['error'])
        self.assertEqual(len(backend.calls), 1)

    def test_six_stream_total_scope_exact(self):
        updates = steps = timed = 0
        for name in e.EXPECTED_STREAMS:
            ref = reference(name)
            result = e.run_stream(ref, FakeBackend(ref))
            self.assertEqual(result['status'], 'complete', result.get('error'))
            updates += len(result['updates'])
            steps += sum(u['accounting']['optimizer_steps_this_event'] for u in result['updates'])
            timed += len(result['scores'])
        self.assertEqual((updates, steps, timed), (12, 24, 108))

    def test_invalid_original_records_rejected_before_start(self):
        ref = reference()
        ref['original_scores'][-1]['verification_calls'] = 999
        backend = FakeBackend(ref)
        with self.assertRaisesRegex(ValueError, 'Original verification counts unstable'):
            e.run_stream(ref, backend)
        self.assertEqual(backend.events, [])


if __name__ == '__main__':
    unittest.main()
