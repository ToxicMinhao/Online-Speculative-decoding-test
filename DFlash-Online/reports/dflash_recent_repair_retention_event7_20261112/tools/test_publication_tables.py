"""Synthetic privacy, schedule, identity and qualification metadata checks."""
import csv
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import publication_tables as P


def fixture(root):
    original = "/private/server/run"
    requests = [dict(request_id="train_00001", dataset_id="gsm8k_train", dataset_index=1,
                     prompt_ids=[901101,901102], response_ids=[998877], question="PRIVATE_SOURCE_TEXT"),
                dict(request_id="train_00002", dataset_id="gsm8k_train", dataset_index=2,
                     prompt_ids=[901103], response_ids=[998878])]
    records = [dict(presentation_id=i, pass_index=i//2, record_id=requests[i%2]["request_id"],
        sampled_anchors=[dict(positions=[[len(requests[i%2]["prompt_ids"]),4]],keep_mask=[[True,False]])])
        for i in range(4)]
    schedule = dict(records=records,passes=2,response_presentations=4)
    schedule_path=root/"SCHEDULE.json"
    P.save(schedule_path,schedule)
    pinned=dict(bytes=10,sha256="a"*64)
    plan=dict(root=original,execution_commit="b"*40,origin_state="/private/server/parent/training_state.pt",
        schedule="/private/server/old/SCHEDULE.json",qualification_root=original+"/qualification",
        input_identities={"/private/server/old/SCHEDULE.json":P.identity(schedule_path),
            "/private/server/parent/training_state.pt":pinned,
            "/private/server/old/train_00001/features.pt":pinned},
        checkpoints={"K6":dict(weights_identity=pinned,config_identity=pinned,export="PRIVATE_PATH")},
        original_plan=dict(requests=requests))
    (root/"sealed").mkdir()
    P.save(root/"sealed/PLAN.json",plan)
    return plan,schedule_path,schedule,requests


class PublicationTablesTests(unittest.TestCase):
    def test_offsets_are_not_tokens_and_every_presentation_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,_,schedule,requests=fixture(root)
            order,anchors=P.schedule_tables(schedule,requests)
            self.assertEqual(len(order),4)
            self.assertEqual(len(anchors),8)
            self.assertEqual([r["global_adam_step_for_window"] for r in order],[769,769,770,770])
            self.assertEqual(anchors[0]["sequence_offset"],2)
            self.assertEqual(anchors[0]["canonical_response_offset"],0)
            self.assertEqual(sum(row["kept"] for row in anchors),4)
            self.assertEqual(order[0]["sampled_anchor_payload_sha256"],P.canonical_digest(schedule["records"][0]["sampled_anchors"]))

    def test_duplicate_or_reordered_response_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _,_,schedule,requests=fixture(Path(directory))
            schedule["records"][1]["record_id"]="train_00001"
            with self.assertRaisesRegex(ValueError,"exactly once"):
                P.schedule_tables(schedule,requests)
            schedule["records"][0]["presentation_id"]=1
            with self.assertRaisesRegex(ValueError,"identity/order"):
                P.schedule_tables(schedule,requests)

    def test_public_files_never_include_private_arrays_text_or_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan,schedule,_,_=fixture(root)
            qualification=root/"qualification/DISABLED"
            qualification.mkdir(parents=True)
            same=dict(exact_equal=True,compared_tensors=5,compared_coordinates=100,
                tensor_layouts=[dict(path="PRIVATE_PARAMETER_PATH")],tensor_differences=[],metadata_differences=[])
            P.save(qualification/"RESULT.json",dict(status="passed",execution_commit=plan["execution_commit"],
                exact_native=same,exact_selections=same,peak_allocated_bytes=123,peak_reserved_bytes=456,
                observations=[dict(canonical_token=998877,prompt_ids=[901101])],
                work=dict(completed=dict(response_forwards=2,response_backwards=2,optimizer_steps=1))))
            output=root/"public"
            result=P.publish(root,output,schedule_path=schedule)
            contents="\n".join(path.read_text() for path in output.iterdir())
            for private in ("PRIVATE_SOURCE_TEXT","PRIVATE_PARAMETER_PATH","PRIVATE_PATH","/private/server",
                            "901101","998877",'"prompt_ids"','"response_ids"','"canonical_token"'):
                self.assertNotIn(private,contents)
            self.assertEqual(result["execution_commit"],"b"*40)
            self.assertEqual(result["model_operations"],0)
            with (output/"qualification_checks.csv").open() as stream:
                values={row["case"]:row for row in csv.DictReader(stream)}
            self.assertEqual(values["DISABLED"]["exact_native_exact_equal"],"True")
            self.assertEqual(values["DISABLED"]["completed_optimizer_steps"],"1")
            self.assertEqual(values["NATIVE"]["status"],"unavailable")

    def test_schedule_hash_mismatch_stops_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,schedule,content,_=fixture(root)
            content["records"][0]["sampled_anchors"][0]["positions"][0][0]+=1
            P.save(schedule,content)
            with self.assertRaisesRegex(ValueError,"sealed file identity"):
                P.publish(root,root/"public",schedule_path=schedule)
            self.assertFalse((root/"public").exists())

    def test_exact_archive_mapping_has_precedence_over_live_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/"qualification/NATIVE/RESULT.json"
            path.parent.mkdir(parents=True)
            P.save(path,{})
            self.assertEqual(P.resolve("/original/run/qualification/NATIVE/RESULT.json",root,"/original/run"),path)


if __name__=="__main__":
    unittest.main()
