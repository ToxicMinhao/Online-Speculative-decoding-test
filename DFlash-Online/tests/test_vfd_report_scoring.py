"""Report recomputation rejects saved scores or selected identities that changed."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from online_feedback.storage import write_json
from online_feedback.vfd_scoring import score_response_pair, select_responses
from experiments.vfd_second_pass.report import score_evidence,audit_qualification
from test_vfd_scoring import snapshot, block


class ScoreEvidenceTests(unittest.TestCase):
    def test_shared_event1_snapshot_recomputation_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            order = [f"r{i}" for i in range(128)]
            before = {
                key: snapshot([block([0, 1, 8, 3, 4])], response_id=key)
                for key in order
            }
            after = deepcopy(before)
            for value in after.values():
                value["blocks"][0]["margins"][2] = -1.0
            write_json(root / "SHARED/SCORES_BEFORE.json", before)
            write_json(root / "SHARED/SCORES_AFTER.json", after)
            rows = [score_response_pair(before[k], after[k]) for k in before]
            selection = select_responses(rows, order, event_id=1, policy="PROGRESS64")
            write_json(root / "PROGRESS64/SCORING.json", rows)
            self.assertEqual(
                score_evidence(root, "PROGRESS64", 1, selection, order), rows
            )
            changed = deepcopy(rows)
            changed[0]["progress_score"] = 0.999
            write_json(root / "PROGRESS64/SCORING.json", changed)
            with self.assertRaisesRegex(ValueError, "scores differ"):
                score_evidence(root, "PROGRESS64", 1, selection, order)
            write_json(root / "PROGRESS64/SCORING.json", rows)
            selection["selected_ids"].reverse()
            with self.assertRaisesRegex(ValueError, "fixed hash"):
                score_evidence(root, "PROGRESS64", 1, selection, order)

    def test_qualification_rechecks_separate_work_and_before_after_examples(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            before={f'r{i}':snapshot([block([0,1,8,3,4])],response_id=f'r{i}') for i in range(2)}
            after=deepcopy(before)
            for value in after.values():value['blocks'][0]['margins'][2]=-1.
            expected=dict(training_presentations=4,backwards=4,optimizer_steps=2,scoring_forwards=4)
            actions=[dict(kind=key,ordinal=i,phase=phase) for key,count in expected.items()
                     for i in range(1,count+1) for phase in ('started','completed')]
            (root/'WORK.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in actions))
            result=dict(status='passed',fork_equality=True,restoration_after_discard={'status':'passed'},
                work=dict(attempts=expected,completed=expected,unknown={key:0 for key in expected}),
                extra_generations=0,before=before,after=after,
                examples=[score_response_pair(before[key],after[key]) for key in before])
            write_json(root/'RESULT.json',result)
            audit=audit_qualification(root,dict(qualification_work=expected))
            self.assertEqual(audit['status'],'passed')
            result['examples'][0]['progress_score']=1.
            write_json(root/'RESULT.json',result)
            with self.assertRaisesRegex(ValueError,'Qualification examples'):
                audit_qualification(root,dict(qualification_work=expected))

    def test_missing_selector_score_file_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                score_evidence(Path(temp), "STATIC64", 2, {}, list(range(128)))


if __name__ == "__main__":
    unittest.main()
