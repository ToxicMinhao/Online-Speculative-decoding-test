"""Check new arms against an ordinary OSD event without changing its RNG draws."""
from copy import deepcopy
import json
from pathlib import Path


class PresentationSchedule:
    """A retained RESULT.json is an assertion, not a replacement sampler.

    The trainer still shuffles and samples normally. Each identity and anchor
    draw must match before that response reaches the draft forward.
    """

    def __init__(self, result, record_ids, passes, *, features=None):
        if isinstance(result, (str, Path)):
            result = json.loads(Path(result).read_text())
        if features is not None and result.get("features") != features:
            raise ValueError("Retained response feature paths or metadata differ")
        self.rows = deepcopy(result["records"])
        expected_count = len(record_ids) * passes
        if (result.get("status") != "completed"
                or result.get("passes") != passes
                or result.get("response_presentations") != expected_count
                or len(self.rows) != expected_count):
            raise ValueError("Retained schedule must be one complete matching event")
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("Retained schedule requires distinct response identities")
        for pass_index in range(passes):
            rows = self.rows[pass_index * len(record_ids):(pass_index + 1) * len(record_ids)]
            if (any(row["pass"] != pass_index for row in rows)
                    or sorted(row["record_id"] for row in rows) != sorted(record_ids)):
                raise ValueError("Retained schedule pass/response identities differ")
            if any(len(row["sampled_anchors"]) != 1 for row in rows):
                raise ValueError("Expected one ordinary anchor draw per response presentation")
        self.position = 0
        self.checked_draws = 0

    def begin_presentation(self, pass_index, record_id):
        row = self.rows[self.position]
        if (row["pass"], row["record_id"]) != (pass_index, record_id):
            raise ValueError(f"Response ordering differs from retained schedule at presentation {self.position}")
        self.checked_draws = 0

    def check_anchors(self, observation):
        expected = self.rows[self.position]["sampled_anchors"]
        if self.checked_draws >= len(expected) or observation != expected[self.checked_draws]:
            raise ValueError(f"Sampled anchors differ from retained schedule at presentation {self.position}")
        self.checked_draws += 1

    def finish_presentation(self):
        if self.checked_draws != len(self.rows[self.position]["sampled_anchors"]):
            raise ValueError("Retained anchor draw was not checked")
        self.position += 1
