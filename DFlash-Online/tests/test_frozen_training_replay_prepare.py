"""Join regressions use public scalar fixtures; no private tokens or model work."""
import copy
import csv
from pathlib import Path
import unittest
from experiments.frozen_training_repair_replay.prepare import join_occurrences

PUBLIC = Path(__file__).resolve().parents[1]/'reports/dflash_recent_repair_retention_event7_20261112'

class HistoricalJoinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (PUBLIC/'assigned_slots.csv').open() as f:
            cls.public=list(csv.DictReader(f))
        with (PUBLIC/'training_response_order.csv').open() as f:
            order=list(csv.DictReader(f))
        with (PUBLIC/'training_anchor_bindings.csv').open() as f:
            anchors=list(csv.DictReader(f))
        records=[]
        for p in order:
            rows=[r for r in anchors if r['presentation_id']==p['presentation_id']]
            records.append(dict(record_id=p['request_id'],pass_index=int(p['pass_index']),sampled_anchors=[dict(
                positions=[[int(r['sequence_offset']) for r in rows]],keep_mask=[[r['kept']=='True' for r in rows]])]))
        cls.schedule=dict(records=records)
        cls.assignments=[];cls.parents=[];labels={}
        for public in cls.public:
            if public['arm']!='RECENT': continue
            row={k:int(public[k]) for k in ('presentation_id','pass_index','anchor_ordinal','canonical_start','proposal_slot')}
            row.update(request_id=public['request_id'],margin=float(public['margin']),floor=float(public['floor']),weight=float(public['weight']))
            row.update(canonical_token=7,expected_canonical_token=7,label_match=True,valid=True,native_correct=True,correct_prefix=int(public['a6']),sequence_start=records[row['presentation_id']]['sampled_anchors'][0]['positions'][0][row['anchor_ordinal']])
            cls.assignments.append(row);cls.parents.append(dict(row))
            labels[(row['request_id'],row['canonical_start'],row['proposal_slot'])]=dict(row)
        cls.candidates=list(labels.values())

    def run_join(self,assignments=None,parents=None,public=None,schedule=None):
        return join_occurrences(self.assignments if assignments is None else assignments,
            self.parents if parents is None else parents,self.candidates,
            self.public if public is None else public,self.schedule if schedule is None else schedule)

    def test_full_published_population_preserves_singleton(self):
        rows=self.run_join()
        singleton=[r for r in rows if (r['request_id'],r['canonical_start'],r['proposal_slot'])==('train_00786',91,13)]
        self.assertEqual(len(rows),1629);self.assertEqual(len(singleton),1)
        self.assertEqual(singleton[0]['pass_index'],1)

    def test_wrong_label_rejected(self):
        rows=copy.deepcopy(self.parents);rows[0]['canonical_token']=8
        with self.assertRaisesRegex(ValueError,'canonical'):self.run_join(parents=rows)

    def test_wrong_floor_rejected(self):
        rows=copy.deepcopy(self.assignments);rows[0]['floor']+=.01
        with self.assertRaisesRegex(ValueError,'floor'):self.run_join(assignments=rows)

    def test_missing_occurrence_rejected(self):
        with self.assertRaisesRegex(ValueError,'assignments'):self.run_join(assignments=self.assignments[1:])

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.run_join(assignments=self.assignments+[self.assignments[0]])

    def test_keep_mask_mismatch_rejected(self):
        schedule=copy.deepcopy(self.schedule);r=self.assignments[0]
        schedule['records'][r['presentation_id']]['sampled_anchors'][0]['keep_mask'][0][r['anchor_ordinal']]=False
        with self.assertRaisesRegex(ValueError,'anchor'):self.run_join(schedule=schedule)

if __name__=='__main__':unittest.main()
