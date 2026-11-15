"""Fixed five-state grid, bounded work and invalid-row retention controls."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

from experiments.native_prefix_loss import campaign, probe_client
from experiments.native_prefix_loss.prepare import CONSTANTS
from test_threat_native_probe import record_fixture, keys_fixture, observed


class PrefixProbeTests(unittest.TestCase):
    def test_grid_is_five_exact_states_at_all_windows(self):
        rows=[dict(window=w,state=s,checkpoint=f'{w}/{s}')
              for w in probe_client.WINDOWS for s in probe_client.STATES]
        self.assertEqual(len(probe_client.validate_exports(rows)),80)
        for changed in (rows[:-1], rows+[rows[0]], [dict(r,state='U') if r['state']=='E' else r for r in rows]):
            with self.assertRaises(ValueError):probe_client.validate_exports(changed)

    def test_invalid_output_never_becomes_valid_prefix(self):
        record=record_fixture();key=keys_fixture()[0]
        raw=observed(record,0)
        row=probe_client.score_observation(record,key,raw,window=8,state='E',checkpoint='E')
        self.assertTrue(row['valid']);self.assertEqual(row['accepted_prefix'],0)
        raw['response_ids']=[999]
        with self.assertRaises(ValueError):
            probe_client.score_observation(record,key,raw,window=8,state='E',checkpoint='E')

    def test_failed_probe_is_durable_before_raise(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);record=record_fixture();keys=keys_fixture()[:1]
            exports={(8,s):dict(window=8,state=s,checkpoint=s) for s in probe_client.STATES}
            backend=SimpleNamespace(one_block=lambda *a: (_ for _ in ()).throw(ValueError('mismatch')))
            budget=SimpleNamespace(check_time=lambda:None)
            receipt=dict(reused_probe_rows=0,probe_attempts=0,physical_probe_blocks=0,invalid_probe_rows=0)
            outcomes=[]
            with self.assertRaisesRegex(ValueError,'mismatch'):
                probe_client.evaluate_exports(backend,exports,keys,{'r':record},
                    {probe_client.probe_key(keys[0]):{}},root,budget,receipt,outcomes)
            saved=json.loads((root/'PROBE_OUTCOMES.jsonl').read_text())
            self.assertFalse(saved['valid']);self.assertEqual(receipt['invalid_probe_rows'],1)
            self.assertEqual(receipt['probe_attempts'],1)

    def test_work_budget_separates_saved_reconstruction_from_scored_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'parent.json';path.write_text(json.dumps({'metadata':{'trace':[1,2,3]}}))
            plan=dict(probe_requests=[{'request_id':'r'}],parent_records={'r':str(path)})
            limits=campaign.native_limits(plan)
            self.assertEqual(limits['draft_block'],20480+3+3+32+256)
            self.assertEqual(limits['model_load'],84)
            self.assertEqual(CONSTANTS['maximum_adam'],182)
            self.assertEqual(CONSTANTS['new_complete_response_generations'],0)

    def test_fixed_precision_and_no_prior_failed_work_import(self):
        accounting=campaign.checked_accounting({})
        self.assertEqual(accounting['prior_adam_steps'],0)
        self.assertEqual(accounting['current_attempt_adam_limit'],182)
        self.assertEqual(CONSTANTS['head_chunk_blocks'],8)
        self.assertEqual(CONSTANTS['source_scoring_response_forwards_max'],160)


if __name__=='__main__':unittest.main()
