"""Retained ordinary-OSD schedules are checked, never silently approximated."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from online_feedback.presentation_schedule import PresentationSchedule


def receipt(record_ids=("a", "b"), passes=2):
    records = [
        {"pass": pass_index, "record_id": record_id, "sampled_anchors": [
            {"positions": [[4, 6]], "keep_mask": [[True, True]]},
        ]}
        for pass_index in range(passes) for record_id in record_ids
    ]
    return {"status": "completed", "passes": passes,
            "response_presentations": len(records), "records": records}


class PresentationScheduleTests(unittest.TestCase):
    def test_complete_128_response_two_pass_schedule_and_file_loading(self):
        identities = [f"response_{index}" for index in range(128)]
        original = receipt(identities)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "RESULT.json"
            path.write_text(json.dumps(original))
            schedule = PresentationSchedule(path, identities, 2)
        for row in original["records"]:
            schedule.begin_presentation(row["pass"], row["record_id"])
            schedule.check_anchors(row["sampled_anchors"][0])
            schedule.finish_presentation()
        self.assertEqual(schedule.position, 256)

    def test_partial_event_duplicate_response_and_pass_boundary_fail(self):
        for field, value in (("status", "failed"), ("passes", 1),
                             ("response_presentations", 3)):
            bad = receipt()
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                PresentationSchedule(bad, ["a", "b"], 2)
        for row_index, field, value in ((1, "record_id", "a"), (2, "pass", 0),
                                       (0, "sampled_anchors", [])):
            bad = receipt()
            bad["records"][row_index][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                PresentationSchedule(bad, ["a", "b"], 2)

    def test_order_disagreement_stops_at_presentation(self):
        schedule = PresentationSchedule(receipt(), ["a", "b"], 2)
        with self.assertRaisesRegex(ValueError, "Response ordering differs"):
            schedule.begin_presentation(0, "b")
        self.assertEqual(schedule.position, 0)

    def test_anchor_position_and_keep_disagreement_stop_without_advancing(self):
        for key, value in (("positions", [[4, 7]]), ("keep_mask", [[True, False]])):
            schedule = PresentationSchedule(receipt(), ["a", "b"], 2)
            schedule.begin_presentation(0, "a")
            draw = deepcopy(schedule.rows[0]["sampled_anchors"][0])
            draw[key] = value
            with self.assertRaisesRegex(ValueError, "Sampled anchors differ"):
                schedule.check_anchors(draw)
            self.assertEqual(schedule.checked_draws, 0)

    def test_missing_or_extra_draw_fails(self):
        schedule = PresentationSchedule(receipt(), ["a", "b"], 2)
        schedule.begin_presentation(0, "a")
        with self.assertRaisesRegex(ValueError, "not checked"):
            schedule.finish_presentation()
        draw = schedule.rows[0]["sampled_anchors"][0]
        schedule.check_anchors(draw)
        with self.assertRaisesRegex(ValueError, "Sampled anchors differ"):
            schedule.check_anchors(draw)

    def test_caller_cannot_mutate_loaded_schedule(self):
        original = receipt()
        schedule = PresentationSchedule(original, ["a", "b"], 2)
        original["records"][0]["sampled_anchors"][0]["positions"] = [[9]]
        self.assertEqual(schedule.rows[0]["sampled_anchors"][0]["positions"], [[4, 6]])


if __name__ == "__main__":
    unittest.main()
