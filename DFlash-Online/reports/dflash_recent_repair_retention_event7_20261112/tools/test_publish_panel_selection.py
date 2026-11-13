import copy
import json
from pathlib import Path
import tempfile
import unittest

import publish_panel_selection as P


def fixture():
    audit = [dict(source_id=f"gsm8k:main:test:{i}", source_row=i,
        status="eligible" if i < 3 else "excluded", prior_endpoint_exposure=bool(i),
        exclusion_matches=[] if i < 3 else [dict(reason="PR43_outcome_input", request_id="old-3",
            shared_keys=[["source_id", "gsm8k:main:test:3"], ["prompt_ids_sha256", "a"*64],
                         ["private_text", "PRIVATE_SOURCE_TEXT"]],
            private_path="/private/server/raw.json")], duplicate_of=[]) for i in range(4)]
    eligible = sorted([row["source_id"] for row in audit[:3]],
        key=lambda sid: (sid != "gsm8k:main:test:0", P.stable_hash(P.SELECTION_PREFIX, sid), sid))
    selected = [dict(source_id=sid,request_id="chosen-"+str(i),panel_ordinal=i,
        selection_sha256=P.stable_hash(P.SELECTION_PREFIX,sid),prior_endpoint_exposure=bool(i),
        exposure_tier=int(bool(i)),prompt_ids=[987654321],stop_token_ids=[998877665],
        prior_endpoint_evidence=[dict(evidence="/private/server/inventory.json")],question="PRIVATE_SOURCE_TEXT")
        for i,sid in enumerate(eligible[:2])]
    path_sources = sorted(selected,key=lambda row:(P.stable_hash(P.PATH_PREFIX,row["source_id"]),row["source_id"]))
    return dict(source_revision=P.REVISION,selection_prefix=P.SELECTION_PREFIX,path_prefix=P.PATH_PREFIX,
        binding="b"*64,input_only_selection=True,outcomes_used=False,generated_lengths_used=False,
        reference_availability_used=False,exclusion_audit=audit,requests=selected,
        candidate_count=4,count=2,requested_count=2,shortfall=0,eligibility_count=3,
        eligible_order=eligible,exclusion_status_counts={"eligible":3,"excluded":1},
        eligible_tier_counts={"0":1,"1":2},selected_tier_counts={"0":1,"1":1},
        path_request_ids=[row["request_id"] for row in path_sources])


class PanelPublicationTests(unittest.TestCase):
    def test_full_population_and_public_privacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"PANEL.json"
            source.write_text(json.dumps(fixture()))
            result=P.publish(source,root/"out")
            self.assertEqual(result["published_status_counts"],dict(selected=2,eligible_unselected=1,excluded=1))
            self.assertEqual(result["fixed_start_request_count"],2)
            self.assertEqual(result["excluded_candidate_counts_by_reason"],{"PR43_outcome_input":1})
            contents="\n".join(path.read_text() for path in (root/"out").iterdir())
            for private in ("PRIVATE_SOURCE_TEXT","/private/server","987654321","998877665","prompt_ids\"", "stop_token_ids"):
                self.assertNotIn(private,contents)

    def test_changed_order_counts_or_outcome_selection_fail(self):
        for mutate in (lambda x:x["eligible_order"].reverse(),
                       lambda x:x.update(candidate_count=5),
                       lambda x:x.update(outcomes_used=True),
                       lambda x:x.update(selected_tier_counts={"1":2}),
                       lambda x:x["requests"][0].update(selection_sha256="c"*64)):
            with self.subTest(mutate=mutate):
                panel=fixture();mutate(panel)
                with self.assertRaises(ValueError):P.reduce_panel(panel)

    def test_identifiers_cannot_smuggle_paths_or_prose(self):
        for key,value in (("request_id","/private/server/secret"),("reason","PRIVATE SOURCE TEXT")):
            panel=fixture();panel["exclusion_audit"][-1]["exclusion_matches"][0][key]=value
            with self.assertRaisesRegex(ValueError,"Non-identifier"):P.reduce_panel(panel)

    def test_shortfall_is_disclosed_without_relaxing_exclusions(self):
        panel=fixture();panel["requested_count"]=5
        third=copy.deepcopy(panel["requests"][1]);third.update(source_id=panel["eligible_order"][2],request_id="chosen-2",panel_ordinal=2)
        third["selection_sha256"]=P.stable_hash(P.SELECTION_PREFIX,third["source_id"])
        panel["requests"].append(third)
        panel.update(count=3,shortfall=2,selected_tier_counts={"0":1,"1":2})
        panel["path_request_ids"]=[r["request_id"] for r in sorted(panel["requests"],key=lambda r:P.stable_hash(P.PATH_PREFIX,r["source_id"]))]
        rows,summary=P.reduce_panel(panel)
        self.assertEqual(summary["shortfall"],2)
        self.assertEqual(sum(row["status"]=="excluded" for row in rows),1)


if __name__ == "__main__":unittest.main()
