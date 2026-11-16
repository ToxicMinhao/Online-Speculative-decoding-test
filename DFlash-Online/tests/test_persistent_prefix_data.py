"""Input-only panel controls; no model or private response data are needed."""
from copy import deepcopy
import hashlib

import numpy as np
import unittest

from experiments.persistent_prefix_withholding.data import (
    DATASET, block_permutations, choose_warmups, identity_index,
    normalize, overlap_reasons, select_panel, selection_key,
)


def row(index, **updates):
    value = dict(source_id=f"gsm8k:main:test:{index}", original_index=index,
                 normalized_question_sha256=f"question{index}",
                 rendered_prompt_sha256=f"render{index}", prompt_ids_sha256=f"tokens{index}",
                 prompt_ids=[index + 1], stop_token_ids=[151645, 151643],
                 request_id=f"old{index}")
    value.update(updates)
    return value


def panel_fixture():
    return [row(index) for index in range(140)]


class PersistentPrefixDataTests(unittest.TestCase):
    def test_selection_key_is_the_exact_declared_string(self):
        source = "gsm8k:main:test:17"
        expected = hashlib.sha256(f"kplus-persistent-prefix-v1|eval|{DATASET}|{source}".encode()).hexdigest()
        assert selection_key(source) == expected


    def test_all_four_identity_channels_exclude_adaptation(self):
        candidates = panel_fixture()
        adaptation = [row(1000 + index, **{key: candidates[index][key]}) for index, key in enumerate(
            ("source_id", "normalized_question_sha256", "rendered_prompt_sha256", "prompt_ids_sha256"))]
        selected, audit, counts = select_panel(candidates, adaptation, [], {})
        assert len(selected) == 128
        assert {record["original_index"] for record in audit if record["status"] == "excluded"} == set(range(4))
        assert counts["eligible_tier_counts"] == {"A": 136, "B": 0}


    def test_a_preferred_then_b_only_for_shortfall(self):
        candidates = panel_fixture()
        direct = {candidate["source_id"]: ["PR44"] for candidate in candidates[:20]}
        selected, _, counts = select_panel(candidates, [], [], direct)
        assert counts["selected_tier_counts"] == {"A": 120, "B": 8}
        expected_b = sorted(candidates[:20], key=lambda r: (selection_key(r["source_id"]), r["source_id"]))[:8]
        assert {r["source_id"] for r in selected if r["exposure_tier"] == "B"} == {r["source_id"] for r in expected_b}


    def test_unknown_direct_classification_goes_to_b(self):
        candidates = panel_fixture()
        selected, _, counts = select_panel(candidates, [], [], {}, [r["source_id"] for r in candidates])
        assert counts["eligible_tier_counts"] == {"A": 0, "B": 140}
        assert all(r["direct_exposure_uncertain"] for r in selected)


    def test_final_panel_resorted_across_tiers_without_new_selection(self):
        candidates = panel_fixture()
        direct = {r["source_id"]: ["PR44"] for r in candidates[:20]}
        selected, _, _ = select_panel(candidates, [], [], direct)
        assert [r["selection_key"] for r in selected] == sorted(r["selection_key"] for r in selected)
        assert [r["panel_ordinal"] for r in selected] == list(range(128))
        assert [r["block_ordinal"] for r in selected] == [i // 32 for i in range(128)]


    def test_warmups_are_excluded_by_identity_not_request_name(self):
        candidates = panel_fixture()
        warmups = [dict(candidates[0], request_id="renamed_short"), dict(candidates[1], request_id="renamed_long")]
        selected, audit, _ = select_panel(candidates, [], warmups, {})
        assert not {r["source_id"] for r in selected} & {r["source_id"] for r in warmups}
        assert all(any(reason.startswith("warmup_overlap:") for reason in r["exclusion_reasons"]) for r in audit[:2])


    def test_duplicate_source_id_is_unresolved_input_error(self):
        candidates = panel_fixture()
        candidates[1]["source_id"] = candidates[0]["source_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            select_panel(candidates, [], [], {})


    def _check_duplicates(self, key):
        candidates = panel_fixture()
        candidates[5][key] = candidates[0][key]
        _, audit, _ = select_panel(candidates[::-1], [], [], {})
        assert audit[0]["status"] == "eligible"
        assert f"duplicate_prior_source_row:{key}" in audit[5]["exclusion_reasons"]


    def test_admissibility_and_shortfall_are_hard_gates(self):
        candidates = panel_fixture()
        for r in candidates[:13]:
            r["prompt_ids"] = list(range(2049))
        with self.assertRaisesRegex(ValueError, "127 / 128"):
            select_panel(candidates, [], [], {})


    def test_no_mutation_or_selection_by_outcome_fields(self):
        candidates = panel_fixture()
        before = deepcopy(candidates)
        selected1, _, _ = select_panel(candidates, [], [], {})
        assert candidates == before
        for i, r in enumerate(candidates):
            r.update(accuracy=i, output_tokens=9999-i, reference_available=i % 2 == 0,
                     latency=100-i, answer="DO NOT USE")
        selected2, _, _ = select_panel(candidates, [], [], {})
        assert [r["source_id"] for r in selected1] == [r["source_id"] for r in selected2]


    def test_warmup_rule_uses_only_input_length_and_source_id(self):
        candidates = [row(8, prompt_ids=[1] * 8), row(1, prompt_ids=[1]), row(2, prompt_ids=[1] * 8)]
        warmups = choose_warmups(candidates)
        assert [r["source_id"] for r in warmups] == ["gsm8k:main:test:1", "gsm8k:main:test:8"]
        assert [r["panel_ordinal"] for r in warmups] == [0, 1]


    def test_permutations_use_repeat_then_block_draw_order_and_global_ordinals(self):
        actual = block_permutations()
        rng = np.random.Generator(np.random.PCG64(1708))
        assert len(actual) == 3
        for repeat in range(3):
            for block in range(4):
                expected = (rng.permutation(32) + block * 32).tolist()
                assert actual[repeat][block] == expected
                assert sorted(expected) == list(range(block * 32, (block + 1) * 32))
        assert block_permutations() == actual


    def test_normalization_preserves_case_but_handles_unicode_and_whitespace(self):
        assert normalize("Ａ  B\nC") == "A B C"
        assert normalize("A") != normalize("a")


    def test_identity_index_requires_complete_identities(self):
        incomplete = row(1)
        del incomplete["prompt_ids_sha256"]
        with self.assertRaises(KeyError):
            identity_index([incomplete])


    def test_overlap_channels_are_reported_separately(self):
        original = row(1)
        assert overlap_reasons(original, identity_index([original])) == [
            "source_id", "normalized_question_sha256", "rendered_prompt_sha256", "prompt_ids_sha256"]


    def test_duplicate_normalized_text(self):
        self._check_duplicates("normalized_question_sha256")

    def test_duplicate_rendered_prompt(self):
        self._check_duplicates("rendered_prompt_sha256")

    def test_duplicate_token_array(self):
        self._check_duplicates("prompt_ids_sha256")


if __name__ == "__main__":
    unittest.main()
