"""Local synthetic-scalar checks only; no model/experiment imports or operations."""

import csv
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import export_results as audit


def measurements():
    rows = []
    for index in range(128):
        for arm in audit.ARMS:
            for repeat in range(3):
                elapsed = (1 + index / 1000 + repeat / 100) * (
                    0.9 if arm == 'FRONTIER_P' else 1
                )
                calls = 4 if arm == 'FRONTIER_P' else 5
                rows.append(
                    dict(
                        phase='scored',
                        request_id='private-request-' + str(index),
                        source_id='private-source-' + str(index),
                        method=arm,
                        repeat=repeat,
                        valid_measurement=True,
                        errors=[],
                        checkpoint_id='/private/' + arm,
                        physical_gpu_uuid='GPU-fixture',
                        output_tokens=11,
                        verification_calls=calls,
                        accepted_draft_tokens=10 - calls,
                        native_accepted_draft_tokens=10 - calls,
                        committed_progress=10,
                        finish_reason='eos',
                        request_latency_s=elapsed,
                        native_generate_s=elapsed * 0.9,
                        cache_maintenance_s=0.01,
                    )
                )
    return rows


class ExportTests(unittest.TestCase):
    def test_raw_row_termination_and_checkpoint_are_bound(self):
        checkpoint = '/private/endpoint'
        request = dict(prompt_ids=[4, 5], request_index=0)
        canonical = dict(
            prompt_ids=[4, 5],
            response_ids=[1, 9],
            finish_reason='eos',
            stop_token_ids=[9],
            max_new_tokens=2048,
        )
        block = dict(
            draft_tokens=[6],
            target_argmax=[9, 0],
            verify_size=2,
            committed_progress=1,
            committed_tokens=[9],
            output_start=0,
            anchor_token=1,
            positions=[2, 3],
            accepted_draft_tokens=0,
            native_accepted_draft_tokens=0,
            raw_accepted_draft_tokens_before_stop=0,
            stop_clipped=False,
            finished=True,
        )
        capture = dict(
            prompt_ids=[4, 5],
            response_ids=[1, 9],
            blocks=[block],
            collect_features=False,
            feature_bytes=0,
            native_capture_validated=True,
            cache_maintenance_s=0.1,
        )
        metadata = dict(
            trace=[block],
            capture=capture,
            feedback_capture_in_timed_request=False,
            block_size=16,
            context_limit=4096,
            runtime='pinned_SGLang_FA3',
            cache_policy='idle_full_flush_before_every_request_and_publication',
            verification_calls=1,
            accepted_draft_tokens=0,
            committed_progress=1,
            request_latency_s=1.0,
            native_generate_s=0.8,
        )
        record = dict(
            canonical,
            checkpoint_id=checkpoint,
            purpose='evaluation',
            session_id='opencode_final58',
            request_index=0,
            record_id='a000_scored_0000_CONTROL',
            metadata=metadata,
        )
        row = dict(
            activation=0,
            phase='scored',
            request_ordinal=0,
            method='CONTROL',
            checkpoint_id=checkpoint,
            finish_reason='eos',
            output_tokens=2,
            verification_calls=1,
            accepted_draft_tokens=0,
            native_accepted_draft_tokens=0,
            committed_progress=1,
            request_latency_s=1.0,
            native_generate_s=0.8,
            cache_maintenance_s=0.1,
        )
        self.assertEqual(
            audit.audit_raw(record, row, request, canonical, checkpoint), 1
        )
        for key, value, message in (
            ('finish_reason', 'length', 'termination'),
            ('checkpoint_id', '/substituted', 'endpoint'),
        ):
            changed = dict(row, **{key: value})
            with self.assertRaisesRegex(ValueError, message):
                audit.audit_raw(record, changed, request, canonical, checkpoint)

    def test_known_ratio_and_new_bootstrap_seed_with_scalar_request_export(self):
        summary = audit.serving(measurements())
        primary = summary['contrasts']['FRONTIER_P_vs_CONTROL'][
            'request_latency_s_reduction_pct'
        ]
        self.assertAlmostEqual(primary['value'], 10.0)
        for bound in primary['ci95']:
            self.assertAlmostEqual(bound, 10.0)
        self.assertEqual(summary['bootstrap']['seed'], 20261125)
        self.assertEqual(summary['bootstrap']['draws'], 10000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'PER_REQUEST.csv'
            requests = [
                dict(request_id='private-request-' + str(i), request_index=i)
                for i in range(128)
            ]
            audit.export_requests(summary, requests, path)
            text = path.read_text()
            self.assertNotIn('private-request', text)
            self.assertNotIn('/private/', text)
            rows = list(csv.DictReader(text.splitlines()))
            self.assertEqual(len(rows), 128)
            self.assertEqual(
                [int(row['request_ordinal']) for row in rows], list(range(128))
            )

    def test_truncated_evidence_retains_rows_but_withholds_gains_and_private_fields(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'raw'
            (root / 'evaluation').mkdir(parents=True)
            row = dict(
                phase='scored',
                request_ordinal=3,
                method='CONTROL',
                repeat=0,
                activation=0,
                group=0,
                valid_measurement=False,
                errors=None,
                request_id='secret-private-id',
                raw_path='/private/raw/path',
                prompt_ids=[999999],
                response_ids=[888888],
            )
            (root / 'evaluation/MEASUREMENTS.jsonl').write_text(
                json.dumps(row) + '\n{"incomplete":'
            )
            output = Path(directory) / 'export'
            result = audit.export(root, output)
            self.assertEqual(result['status'], 'incomplete_or_invalid')
            summary = json.loads((output / 'SERVING_SUMMARY.json').read_text())
            self.assertTrue(summary['gain_estimates_withheld'])
            self.assertEqual(summary['observed_repeat_rows'], 1)
            self.assertEqual(
                len(
                    list(
                        csv.DictReader(
                            (output / 'PER_REQUEST.csv').read_text().splitlines()
                        )
                    )
                ),
                0,
            )
            for name in (
                'ALL_REPEATS.csv',
                'SERVING_SUMMARY.json',
                'COSTS_SUMMARY.json',
                'WITNESS_SUMMARY.json',
            ):
                text = (output / name).read_text()
                for private in (
                    'secret-private-id',
                    '/private/raw/path',
                    '999999',
                    '888888',
                ):
                    self.assertNotIn(private, text)
            private = json.loads((output / 'PRIVATE_AUDIT.json').read_text())
            self.assertEqual(private['measurement_decode_error']['line'], 2)

    def test_witness_derived_roles_useful_repair_and_followers(self):
        labels = list(range(16))
        valid = [False] + [True] * 3 + [False] * 12
        origin = dict(
            labels=labels,
            valid=valid,
            prefix=[False, True] + [False] * 14,
            repair=[False, False, True] + [False] * 13,
            L=1,
            e=2,
            H=2,
            K=3,
            m=3,
        )
        q = list(labels)
        q[2] = 9
        positions = []
        for j in range(1, 4):
            correct = j != 2
            margin = 1.0 if correct else -1.0
            positions.append(
                dict(
                    position=j,
                    canonical=j,
                    rival=9,
                    origin_rival=9,
                    prediction=q[j],
                    correct=correct,
                    margin=margin,
                    origin_rival_margin=margin,
                    margin_tie=False,
                    wrong_rival_tie=False,
                    rival_changed=False,
                    origin_rival_addressed=correct,
                )
            )
        margins = np.array([p['margin'] for p in positions], dtype=np.float32)
        assessment = float(
            (
                np.array([3, 2, 1], dtype=np.float32)
                * np.logaddexp(np.float32(0), -margins)
                / np.float32(math.log(2))
            ).sum()
        )
        parent = dict(
            origin=origin,
            current=deepcopy(origin),
            proposals=q,
            positions=positions,
            metrics=dict(
                raw_repair=0,
                useful_repair=0,
                damaged_prefix_positions=0,
                damaged_prefix_blocks=0,
                surviving_followers=1,
                reachable_followers=0,
                matching_prefix_length=1,
                fixed_K_shortfall=2,
                assessment=assessment,
            ),
        )
        audit.witness_metrics(parent, parent)
        changed = deepcopy(parent)
        changed['metrics']['useful_repair'] = 1
        with self.assertRaisesRegex(ValueError, 'derived metric'):
            audit.witness_metrics(changed, parent)


if __name__ == '__main__':
    unittest.main()
