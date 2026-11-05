import copy
import unittest

from experiments.block_depth.full_plan import (make_plan, validate_plan, capacity_plan,
    check_capacity, work_budget, OUTPUT_ROOT, QUALIFICATION_COMMIT)
from experiments.block_depth.preflight import (GIB, SOURCE_ROOT, EXPORT, TARGET,
                                              WORKER_PYTHON, SERVING)


def fixtures():
    base = "/data/sources/minhao/runs/dflash_block_depth_20261105"
    parts, index = {}, 0
    for role, count in (("READOUT", 384), ("GATE", 128), ("FINAL", 384)):
        rows = []
        for ordinal in range(count):
            rows.append(dict(request_id=f"{role}-{ordinal}", role=role, panel_ordinal=ordinal,
                source_id=f"source-{index}", question=f"Question {index}", prompt_ids=[index + 100, 9],
                max_new_tokens=2048, stop_token_ids=[151645, 151643]))
            index += 1
        parts[role] = dict(count=count, requests=rows)
    manifests = dict(schema="block_depth_manifests_v1", study_id="vfd_depth35_gsm8k_v1", manifests=parts)
    old = dict(code_commit=QUALIFICATION_COMMIT, checkpoint_path=str(EXPORT),target_path=str(TARGET),
        worker_python=WORKER_PYTHON, output_root=base + "/qualification_002", source_root=str(SOURCE_ROOT),
        manifest_file=base + "/prepared_001/MANIFESTS.json", model_pin=base + "/prepared_001/MODEL_PIN.json",
        runtime_pin=base + "/prepared_001/RUNTIME_PIN.json", serving=SERVING,
        stop_after="first qualification and pricing report; no fitting", qualification_modes=["FORCED5"],
        prior_attempt="preserved startup failure")
    paths = dict(output_root=str(OUTPUT_ROOT), canonical_file=base + "/full_preparation_001/CANONICALS.json",
        canonical_audit_file=base + "/full_preparation_001/CANONICAL_REUSE_AUDIT.json",
        pricing_file=base + "/qualification_002/PRICING.json", qualification_status_file=base + "/qualification_002/STATUS.json",
        training_config=base + "/full_preparation_001/CONFIG.yaml")
    identities = {value: dict(bytes=1, sha256="0" * 64) for value in list(paths.values()) +
                  [old[key] for key in ("manifest_file", "model_pin", "runtime_pin")]}
    sources = {"experiments/block_depth/" + name + ".py": dict(bytes=1,sha256="0" * 64)
               for name in ("gate", "endpoint", "full_plan", "readout_training", "native_routes", "worker", "reconstruct", "full_campaign", "full_report")}
    audit = dict(status="passed", requested=384, reused=306, missing_count=78, binding_digest="target-runtime")
    pricing = dict(status="complete", decision="continue_to_readout_fit", continuation_cost_ms=.5176)
    return make_plan(old, manifests, audit, pricing, code_commit="f" * 40, source_files=sources,
                     input_identities=identities, paths=paths)


class FullPlanTests(unittest.TestCase):
    def test_counts_reuse_initial16_and_keep_auxiliary_work_separate(self):
        plan = fixtures()
        self.assertEqual(validate_plan(plan)["status"], "passed")
        budget = plan["budget"]
        self.assertEqual(budget["new_complete_generations_max"], 2878)
        self.assertEqual(budget["readout_presentations"], 768)
        self.assertEqual(budget["readout_optimizer_steps"], 384)
        self.assertEqual(budget["backbone_optimizer_steps"], 0)
        self.assertEqual(budget["extra_qualification_generations"], 0)
        self.assertEqual(budget["gate_proposal_replays_max"], 16384)
        self.assertEqual(len(plan["final_schedule"]), 2304)
        self.assertEqual(plan["config_file"], plan["training_config"])
        self.assertEqual(plan["stop_after"], "full_FINAL_report")
        self.assertNotIn("qualification_modes", plan)
        self.assertEqual(plan["qualification_provenance"]["prior_attempt"], "preserved startup failure")

    def test_capacity_accounts_for_group_duplicate_padding_and_reserve(self):
        budget = capacity_plan()
        raw = 129 * 2272 * 6 * 4096 * 2
        self.assertGreaterEqual(budget["teacher_feature_pool_peak_bytes"], raw * 1.1)
        self.assertEqual(budget["peak_required_bytes"], budget["teacher_feature_pool_peak_bytes"] + 19 * GIB)
        self.assertEqual(budget["new_backbone_checkpoint_bytes"], 0)
        self.assertFalse(budget["offload_assumed"])

    def test_capacity_exact_boundary_passes_one_byte_less_fails(self):
        capacity = capacity_plan()
        self.assertEqual(check_capacity(free_bytes=capacity["peak_required_bytes"], host_available_bytes=32 * GIB)["status"], "passed")
        with self.assertRaisesRegex(RuntimeError, "capacity unavailable"):
            check_capacity(free_bytes=capacity["peak_required_bytes"] - 1, host_available_bytes=32 * GIB)
        with self.assertRaises(RuntimeError):
            check_capacity(free_bytes=capacity["peak_required_bytes"], host_available_bytes=32 * GIB - 1)
        self.assertEqual(check_capacity(free_bytes=8 * GIB, host_available_bytes=32 * GIB, initial=False)["status"], "passed")

    def test_no_retuned_budget_unsealed_input_or_alternate_destination(self):
        original = fixtures()
        for mutate in (lambda p: p["budget"].update(readout_optimizer_steps=385),
                       lambda p: p["input_identities"].pop(p["canonical_file"]),
                       lambda p: p.update(output_root=str(OUTPUT_ROOT) + "_retry"),
                       lambda p: p["numerical"].update(ridge_penalty=.01),
                       lambda p: p["analysis"].update(bootstrap_draws=10),
                       lambda p: p.update(code_commit="working-tree")):
            plan = copy.deepcopy(original)
            mutate(plan)
            with self.assertRaises(ValueError):
                validate_plan(plan)

    def test_reference_work_follows_availability_not_generation_results(self):
        self.assertEqual(work_budget(384)["untimed_reference_generations_max"], 0)
        self.assertEqual(work_budget(0)["untimed_reference_generations_max"], 384)
        with self.assertRaises(ValueError):
            work_budget(385)


if __name__ == "__main__":
    unittest.main()
