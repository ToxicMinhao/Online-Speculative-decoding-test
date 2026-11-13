from copy import deepcopy
import unittest

from experiments.recent_repair_retention.panel import order_hash, select_panel, verify_panel, PATH_PREFIX
from experiments.survival_confirmation.data import digest, REVISION, STOP_IDS


def row(index, **changes):
    value=dict(request_id=f"r{index}",source_id=f"gsm8k:main:test:{index}",
        dataset_id=f"gsm8k:main:test:{index}",source_row=index,source_revision=REVISION,
        prompt_ids=[index+10],prompt_ids_sha256=digest([index+10]),
        normalized_question_sha256=f"q{index}",rendered_prompt_sha256=f"p{index}",
        stop_token_ids=list(STOP_IDS),max_new_tokens=2048,collect_feedback=False)
    value.update(changes)
    return value


class PanelTests(unittest.TestCase):
    def test_exposure_tier_then_hash_and_shortfall(self):
        rows=[row(i) for i in range(6)]
        exposure={rows[1]["source_id"]:["old"],rows[3]["source_id"]:["old"]}
        panel=select_panel(rows,[],exposure,count=8)
        expected=sorted(rows,key=lambda r:(int(r["source_id"] in exposure),order_hash(r["source_id"])))
        self.assertEqual([r["source_id"] for r in panel["requests"]],[r["source_id"] for r in expected])
        self.assertEqual(panel["shortfall"],2)
        verify_panel(panel)

    def test_excludes_normalized_source_and_rendered_token_identities(self):
        for field in ("source_id","normalized_question_sha256","prompt_ids_sha256","rendered_prompt_sha256"):
            with self.subTest(field=field):
                exclusion=row(99,reason="PR43_outcome_input")
                exclusion[field]=row(0)[field]
                panel=select_panel([row(0),row(1)],[exclusion],{})
                self.assertEqual([r["source_row"] for r in panel["requests"]],[1])

    def test_dedup_before_ranking_and_no_output_selection(self):
        rows=[row(0),row(1,normalized_question_sha256="q0"),row(2)]
        a=select_panel(rows,[],{})
        rows[0]["fake_bad_latency"]=999999
        b=select_panel(rows,[],{})
        self.assertEqual([r["source_id"] for r in a["requests"]],[r["source_id"] for r in b["requests"]])
        self.assertEqual(a["exclusion_status_counts"]["duplicate"],1)

    def test_independent_fixed_start_hash_order(self):
        panel=select_panel([row(i) for i in range(100)],[],{})
        expected=sorted(panel["requests"],key=lambda r:order_hash(r["source_id"],PATH_PREFIX))[:64]
        self.assertEqual(panel["path_request_ids"],[r["request_id"] for r in expected])
        verify_panel(panel)

    def test_changed_seal_or_token_fails(self):
        panel=select_panel([row(0)],[],{})
        panel["requests"][0]["prompt_ids"]=[999]
        with self.assertRaisesRegex(ValueError,"seal changed"):verify_panel(panel)


if __name__=="__main__":unittest.main()
