"""Joint paired-request successor-shuffle endpoint analysis; no arm aliases."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from experiments.full_k_gsm8k.statistics import (
    comparison,
    method_totals as base_method_totals,
)
from .evaluation import METHODS, audit_stage
from .contract import ANALYSIS

ARMS = METHODS
DATASETS = ("gsm8k",)
PANEL_SIZE = 256
COUNT_FIELDS = (
    "verification_calls",
    "accepted_draft_tokens",
    "native_accepted_draft_tokens",
    "committed_progress",
    "output_tokens",
)
TIMINGS = ("request_latency_s", "native_generate_s", "cache_maintenance_s")
BOOTSTRAP_SEED = 20261111
RESAMPLES = 10000


def file_identity(path):
    payload = Path(path).read_bytes()
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


class Evidence:
    def __init__(self, root, plan):
        self.root, self.plan = Path(root), plan
        self.files, self.missing, self.errors = {}, {}, []

    def remember(self, path, observed):
        name = str(path)
        if name in self.files and self.files[name] != observed:
            self.errors.append(dict(path=name, error="Input changed between reads"))
        self.files.setdefault(name, observed)

    def local(self, path):
        path = Path(path)
        if path.is_absolute():
            try:
                return self.root / path.relative_to(self.plan["root"])
            except ValueError:
                return path
        return self.root / path

    def read(self, path, *, lines=False):
        path = self.local(path)
        if not path.is_file():
            self.missing[str(path)] = (
                "No retained receipt; not imputed as zero or success."
            )
            return None
        raw = path.read_bytes()
        self.remember(
            path, dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        )
        try:
            return (
                [json.loads(row) for row in raw.splitlines() if row.strip()]
                if lines
                else json.loads(raw)
            )
        except (ValueError, TypeError) as error:
            self.errors.append(dict(path=str(path), error=repr(error)))
            return None

    def remember_tree(self, path):
        """Hash compact records only; never load weights or tensor diagnostics."""
        path = self.local(path)
        for file in sorted(path.rglob("*")) if path.exists() else []:
            if file.is_file() and file.suffix in (".json", ".jsonl", ".csv"):
                self.remember(file, file_identity(file))

    def stability(self):
        changed = []
        for name, expected in self.files.items():
            path = Path(name)
            if not path.is_file() or file_identity(path) != expected:
                changed.append(name)
        return dict(
            status="passed" if not changed else "changed_during_analysis",
            changed=changed,
        )


def dump_json(path, value):
    def safe(item):
        if isinstance(item, float) and not math.isfinite(item):
            return {"nonfinite_saved_value": repr(item)}
        if isinstance(item, dict):
            return {key: safe(v) for key, v in item.items()}
        if isinstance(item, list):
            return [safe(v) for v in item]
        return item

    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


def dump_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        json.dumps(v, sort_keys=True)
                        if isinstance(v, (dict, list))
                        else v
                    )
                    for k, v in row.items()
                }
            )


def reduce_stage(rows, *, dataset, event, panel_size=PANEL_SIZE, allow_seed_eos=False):
    """Average repeats within request; keep deterministic native counts once."""
    if dataset not in DATASETS or event != 5:
        raise ValueError("Unknown dataset or evaluation event")
    repeats = 2
    grouped = defaultdict(list)
    identities = {}
    for row in rows:
        if (
            row["phase"] != "final"
            or row["event"] != event
            or row["dataset"] != dataset
            or row["seed"] != 17
            or row["method"] not in METHODS
            or row["validation_status"] != "valid"
            or row["valid_measurement"] is not True
            or row["errors"]
        ):
            raise ValueError(
                "Only valid same-dataset/stage executions enter a comparison"
            )
        ordinal, request_id = row["panel_ordinal"], row["request_id"]
        if ordinal in identities and identities[ordinal] != request_id:
            raise ValueError("A panel ordinal has multiple request identities")
        identities[ordinal] = request_id
        grouped[ordinal, row["method"]].append(row)
    if (
        len(identities) != panel_size
        or sorted(identities) != list(range(panel_size))
        or len(set(identities.values())) != len(identities)
    ):
        raise ValueError("Missing or duplicated paired request identities")
    paired = []
    for ordinal, request_id in sorted(identities.items()):
        value = dict(
            seed=17,
            dataset=dataset,
            event=event,
            panel_ordinal=ordinal,
            request_id=request_id,
        )
        for method in METHODS:
            records = sorted(grouped[ordinal, method], key=lambda row: row["repeat"])
            if [row["repeat"] for row in records] != list(range(repeats)):
                raise ValueError(
                    "Comparison requires every distinct planned repeat for every arm"
                )
            first = records[0]
            for field in COUNT_FIELDS:
                if (
                    type(first[field]) is not int
                    or first[field] < 0
                    or any(row[field] != first[field] for row in records)
                ):
                    raise ValueError("Invalid or inconsistent repeated counts")
            seed_eos = (
                allow_seed_eos
                and first["verification_calls"] == 0
                and first["output_tokens"] == 1
                and first["committed_progress"] == 0
                and first.get("finish_reason") == "eos"
                and first.get("first_round_present") is False
            )
            if first["verification_calls"] <= 0 and not seed_eos:
                raise ValueError("Verification denominator must be positive")
            if (
                first["accepted_draft_tokens"] > first["committed_progress"]
                or first["native_accepted_draft_tokens"]
                != first["committed_progress"] - first["verification_calls"]
                or first["committed_progress"] != first["output_tokens"] - 1
            ):
                raise ValueError(
                    "Native acceptance or committed-progress accounting differs"
                )
            if (
                any(
                    row["checkpoint_id"] != first["checkpoint_id"]
                    or row["finish_reason"] != first["finish_reason"]
                    or row["cap_hit"] != first["cap_hit"]
                    for row in records
                )
                or type(first["cap_hit"]) is not bool
                or first["cap_hit"] != (first["finish_reason"] == "length")
                or first["finish_reason"] not in ("eos", "length")
                or (first["cap_hit"] and first["output_tokens"] != 2048)
            ):
                raise ValueError("Checkpoint or termination changes within a method")
            times = {
                field: float(np.mean([row[field] for row in records]))
                for field in TIMINGS
            }
            if any(
                not np.isfinite(row[field])
                or row[field] < 0
                or (field != "cache_maintenance_s" and row[field] == 0)
                for row in records
                for field in TIMINGS
            ):
                raise ValueError("Invalid latency measurement")
            value[method] = {field: first[field] for field in COUNT_FIELDS}
            value[method].update(
                **times,
                cap_hit=first["cap_hit"],
                finish_reason=first["finish_reason"],
                checkpoint_id=first["checkpoint_id"],
                count_trajectory_repeat=0,
                timing_repeats=repeats,
            )
            first_fields = (
                "first_round_present",
                "first_round_start_digest",
                "first_round_accepted_prefix_tokens",
                "first_round_raw_prefix_tokens",
                "first_round_output_start",
            )
            if any(
                row.get(field) != first.get(field)
                for row in records
                for field in first_fields
            ):
                raise ValueError("First native round disagrees across timing repeats")
            if type(first.get("first_round_present")) is not bool:
                raise ValueError("Missing shared first-round evidence")
            for field in (
                "first_round_accepted_prefix_tokens",
                "first_round_raw_prefix_tokens",
            ):
                if type(first.get(field)) is not int or not 0 <= first[field] <= 15:
                    raise ValueError("Invalid first-round acceptance")
            if (
                first["first_round_accepted_prefix_tokens"]
                > first["first_round_raw_prefix_tokens"]
            ):
                raise ValueError("Stop-clipped acceptance exceeds raw prefix")
            if first["first_round_present"]:
                if (
                    first["first_round_output_start"] != 0
                    or not isinstance(first["first_round_start_digest"], str)
                    or len(first["first_round_start_digest"]) != 64
                ):
                    raise ValueError("Invalid shared first-round prefix identity")
            elif any(first[field] for field in first_fields[1:]):
                raise ValueError("Absent first round has invented acceptance")
            value[method].update({field: first[field] for field in first_fields})
        if (
            len(
                {
                    (value[m]["output_tokens"], value[m]["finish_reason"])
                    for m in METHODS
                }
            )
            != 1
        ):
            raise ValueError("Paired output length or termination differs")
        if (
            len(
                {
                    (
                        value[m]["first_round_present"],
                        value[m]["first_round_start_digest"],
                    )
                    for m in METHODS
                }
            )
            != 1
        ):
            raise ValueError("First rounds do not share the same starting prefix")
        paired.append(value)
    return paired


def method_totals(values):
    """Keep the native counter distinct from stop-clipped matching acceptance."""
    total = base_method_totals(values)
    total["native_accepted_draft_tokens"] = int(
        values["native_accepted_draft_tokens"].sum()
    )
    total["native_accepted_tokens_per_call"] = (
        total["native_accepted_draft_tokens"] / total["verification_calls"]
    )
    total["matched_accepted_tokens_per_call"] = total["accepted_tokens_per_call"]
    return total


def summarize(rows, *, dataset, event, panel_size=PANEL_SIZE, allow_seed_eos=False):
    paired = reduce_stage(
        rows,
        dataset=dataset,
        event=event,
        panel_size=panel_size,
        allow_seed_eos=allow_seed_eos,
    )
    repeats = 2
    arrays = {
        method: {
            field: np.array([row[method][field] for row in paired], dtype=np.float64)
            for field in (*COUNT_FIELDS, *TIMINGS)
        }
        for method in METHODS
    }
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(paired), size=(RESAMPLES, len(paired))
    )
    effects = {
        a + "_vs_" + b: comparison(arrays[a], arrays[b], indices)
        for a, b in (("TRUE", "SHUFFLED"), ("TRUE", "K"), ("SHUFFLED", "K"))
    }
    for name, effect in effects.items():
        candidate, reference = name.split("_vs_")
        delta = (
            arrays[reference]["request_latency_s"]
            - arrays[candidate]["request_latency_s"]
        )
        effect.update(
            request_latency_wins=int((delta > 0).sum()),
            request_latency_regressions=int((delta < 0).sum()),
            request_latency_ties=int((delta == 0).sum()),
        )
        prefix = np.array(
            [
                row[candidate]["first_round_accepted_prefix_tokens"]
                - row[reference]["first_round_accepted_prefix_tokens"]
                for row in paired
            ]
        )
        effect["first_native_round"] = dict(
            requests=len(prefix),
            present_requests=sum(
                row[candidate]["first_round_present"] for row in paired
            ),
            seed_eos_requests=sum(
                not row[candidate]["first_round_present"] for row in paired
            ),
            wins=int((prefix > 0).sum()),
            losses=int((prefix < 0).sum()),
            ties=int((prefix == 0).sum()),
            gross_prefix_tokens_gained=int(np.maximum(prefix, 0).sum()),
            gross_prefix_tokens_lost=int(np.maximum(-prefix, 0).sum()),
            net_prefix_tokens_gained=int(prefix.sum()),
            scope="First native round at authenticated shared starting prefix; stop-clipped acceptance",
        )
    totals = {method: method_totals(values) for method, values in arrays.items()}
    for method in METHODS:
        totals[method]["cap_hit_requests"] = sum(
            row[method]["cap_hit"] for row in paired
        )
    repeat_totals = {
        str(repeat): {
            method: {
                field: sum(
                    row[field]
                    for row in rows
                    if row["repeat"] == repeat and row["method"] == method
                )
                for field in (*COUNT_FIELDS, *TIMINGS)
            }
            for method in METHODS
        }
        for repeat in range(repeats)
    }
    return dict(
        dataset=dataset,
        event=event,
        request_count=len(paired),
        bootstrap=dict(
            resamples=RESAMPLES,
            seed=BOOTSTRAP_SEED,
            generator="PCG64",
            percentiles=[2.5, 97.5],
            method="linear",
            unit="One paired request carrying every arm; repeats form one mean time.",
        ),
        method_totals=totals,
        comparisons=effects,
        repeat_totals=repeat_totals,
        primary=["TRUE_vs_SHUFFLED"],
        controls=["TRUE_vs_K", "SHUFFLED_vs_K"],
        intervals="Unadjusted paired-request intervals; no familywise confirmation claim.",
        timing_scope="Fresh contemporaneous checkpoint-ready serving; not training-inclusive speedup.",
        comparison_scope="Shared saved KPLUS4 feedback; one continuing seed and one permutation; held out from training and SURVIVAL panels.",
    )


def contrast_rows(endpoint_result):
    if endpoint_result.get("status") != "passed":
        return []
    rows = []
    for name, effect in endpoint_result["summary"]["comparisons"].items():
        calls, timing = effect["call_reduction"], effect["request_latency_s_reduction"]
        prefix = effect["first_native_round"]
        rows.append(
            dict(
                contrast=name,
                candidate_calls=int(calls["candidate_sum"]),
                reference_calls=int(calls["reference_sum"]),
                calls_saved=effect["gross_calls_saved"],
                calls_added=effect["gross_calls_added"],
                call_reduction_pct=calls["point_pct"],
                call_ci95_pct=calls["ci95_pct"],
                request_latency_reduction_pct=timing["point_pct"],
                request_latency_ci95_pct=timing["ci95_pct"],
                candidate_latency_total_s=timing["candidate_sum"],
                reference_latency_total_s=timing["reference_sum"],
                call_wins=effect["request_wins"],
                call_losses=effect["request_regressions"],
                call_ties=effect["request_ties"],
                prefix_wins=prefix["wins"],
                prefix_losses=prefix["losses"],
                prefix_ties=prefix["ties"],
                prefix_tokens_gained=prefix["gross_prefix_tokens_gained"],
                prefix_tokens_lost=prefix["gross_prefix_tokens_lost"],
            )
        )
    return rows


def validate_panel_size(n):
    if type(n) is not int or not 64 <= n <= 256 or n % 8:
        raise ValueError("Require sealed N in complete eight-request blocks,64..256")
    return n


def endpoint_from_audit(audit, panel_size):
    validate_panel_size(panel_size)
    try:
        if (
            audit.get("status") != "passed"
            or audit.get("errors")
            or len(audit.get("rows", [])) != 6 * panel_size
        ):
            raise ValueError("Require passed native raw audit and all6N executions")
        paired = reduce_stage(
            audit["rows"],
            dataset="gsm8k",
            event=5,
            panel_size=panel_size,
            allow_seed_eos=True,
        )
        if any(sum(r[m]["verification_calls"] for r in paired) <= 0 for m in METHODS):
            raise ValueError("No positive pooled call denominator")
        summary = summarize(
            audit["rows"],
            dataset="gsm8k",
            event=5,
            panel_size=panel_size,
            allow_seed_eos=True,
        )
        return dict(status="passed", inference_permitted=True, summary=summary), paired
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return (
            dict(
                status="invalid_or_incomplete",
                inference_permitted=False,
                errors=[repr(error)],
            ),
            [],
        )


def decision(summary):
    if summary is None:
        return dict(
            outcome="invalid_or_incomplete", automatic_followup_authorized=False
        )
    primary = summary["comparisons"]["TRUE_vs_SHUFFLED"]["call_reduction"]["ci95_pct"]
    versus_k = summary["comparisons"]["TRUE_vs_K"]["call_reduction"]["ci95_pct"]
    outcome = (
        "informative_placement_supported"
        if primary[0] > 0
        else (
            "prioritization_unfavorable_for_this_fork"
            if primary[1] < 0
            else "successor_placement_unresolved"
        )
    )
    latency = {}
    for name, effect in summary["comparisons"].items():
        a, b = name.split("_vs_")
        ci = effect["request_latency_s_reduction"]["ci95_pct"]
        sweeps = [
            100
            * (
                1
                - summary["repeat_totals"][str(i)][a]["request_latency_s"]
                / summary["repeat_totals"][str(i)][b]["request_latency_s"]
            )
            for i in range(2)
        ]
        latency[name] = dict(
            ci95_pct=ci,
            sweep_reductions_pct=sweeps,
            benefit_supported=ci[0] > 0 and all(v > 0 for v in sweeps),
            slowdown_supported=ci[1] < 0 and all(v < 0 for v in sweeps),
        )
    return dict(
        outcome=outcome,
        incremental_verifier_benefit_vs_K=versus_k[0] > 0,
        informative_placement_and_incremental_verifier_benefit=primary[0] > 0
        and versus_k[0] > 0,
        latency=latency,
        automatic_followup_authorized=False,
        scope="One continuing training seed and one fixed permutation; unresolved intervals do not establish equivalence.",
    )


def fallacy_checks():
    items = [
        ("Simpson paradox", "Report every arm, contrast and timing sweep."),
        (
            "Ecological fallacy",
            "Joint request bootstrap; repeat timings are averaged inside requests.",
        ),
        (
            "Berkson selection bias",
            "Sealed official-row hash order from remaining eligible rows.",
        ),
        (
            "Collider bias",
            "No output, length, historical outcome or reference availability selected inputs.",
        ),
        (
            "Base rate neglect",
            "Keep all N, cap hits, ties, gross call saves/additions and parent-bit denominators.",
        ),
        (
            "Regression to mean",
            "New training-fork/SURVIVAL-held-out panel; no promising-request selection.",
        ),
        (
            "Survivorship bias",
            "Any missing/invalid cell prevents endpoint inference; no complete-case filtering.",
        ),
        (
            "Look-elsewhere effect",
            "One primary call contrast; all intervals unadjusted, not familywise guarantees.",
        ),
        (
            "Forking paths",
            "Fixed panel/permutation/event/bootstrap10000 PCG64 seed20261111; no automatic extension.",
        ),
        (
            "Correlation versus causation",
            "TRUE/SHUFFLED isolates the assignment intervention for this fork; local repair does not establish full-request mediation.",
        ),
        (
            "Reverse causality",
            "Training precedes fresh timing; no serving result changes training or panel.",
        ),
    ]
    return dict(
        coverage="11/11 checked", checks=[dict(name=k, disposition=v) for k, v in items]
    )


def resource_rows(receipts):
    rows = []

    def walk(value, path):
        if not isinstance(value, dict):
            return
        for name, item in value.items():
            if isinstance(item, dict):
                walk(item, path + "/" + name)
            elif (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and (
                    name.endswith("_s")
                    or any(
                        k in name
                        for k in (
                            "steps",
                            "calls",
                            "generations",
                            "presentations",
                            "bytes",
                            "memory",
                            "requests",
                            "blocks",
                        )
                    )
                )
            ):
                rows.append(
                    dict(
                        phase=path,
                        metric=name,
                        value=item,
                        note="Recorded component/enclosing timer; do not add nested clocks.",
                    )
                )

    for phase, receipt in receipts.items():
        walk(receipt, phase)
    return rows


def audit_manipulation(receipt, training_result, true_weights_identity):
    from .campaign import validate_observation

    records = training_result["records"]
    if (
        len(records) != 256
        or receipt.get("presentations") != 256
        or receipt.get("status") != "passed"
    ):
        raise ValueError("Require all256 shuffled presentation observations")
    rows = []
    for item in records:
        observation = item["response_observation"]
        validate_observation(observation)
        if observation["successor_shuffle"]["response_id"] != item["record_id"]:
            raise ValueError("Wrong response manipulation identity")
        rows.append(observation["successor_shuffle"])
    if receipt.get("rows") != rows:
        raise ValueError("Manipulation summary rows differ from training observations")
    for name in (
        "eligible_errors",
        "ineligible_errors",
        "mixed_strata",
        "flipped_bits",
        "uniform_expected_flips",
        "reassigned_weight",
        "projection_calls",
        "projection_s",
        "assignment_s",
    ):
        if receipt[name] != sum(row[name] for row in rows):
            raise ValueError("Manipulation total mismatch: " + name)
    if receipt["labels_changed"] != (receipt["flipped_bits"] > 0) or receipt[
        "final_export_changed"
    ] != (training_result["weights_identity"] != true_weights_identity):
        raise ValueError("Actuation label or export change disagrees with source")
    return dict(status="passed", presentations=256, all_observations_validated=True)


def render(result, comparisons):
    lines = [
        "# K+ successor assignment control",
        "",
        "## Material Passport",
        "",
        "ARS experiment-agent / validate; ANALYZED. Native-record reconstruction; one continuing seed17 and one fixed permutation.",
        "",
        f"Status: **{result['status']}**. Decision: **{result['decision']['outcome']}**.",
        "",
        f"Execution commit: `{result['execution_source']['commit']}`.",
        "",
        f"Sealed N={result['panel_size']}; {6*result['panel_size']} planned timed completions. All three real step640 exports are timed afresh.",
        "",
        "The panel excludes the full development set, training feedback and the768 SURVIVAL confirmation questions. Historical project exposure is documented; global unusedness is not claimed.",
        "",
    ]
    if result["status"] != "complete":
        lines += [
            "No complete valid endpoint inference. Preserve all partial rows and failures.",
            "",
            repr(result["completion_errors"]),
        ]
    else:
        lines += [
            "| Contrast | Calls candidate/reference | Gross saved/added | Call reduction % [95% CI] | Complete-time reduction % [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in comparisons:
            lines.append(
                f"| {r['contrast']} | {r['candidate_calls']}/{r['reference_calls']} | {r['calls_saved']}/{r['calls_added']} | {r['call_reduction_pct']:.6f} {r['call_ci95_pct']} | {r['request_latency_reduction_pct']:.6f} {r['request_latency_ci95_pct']} |"
            )
        lines += [
            "",
            "| Arm | Calls once | Summed mean request seconds | Cap-hit requests |",
            "|---|---:|---:|---:|",
        ]
        for arm, total in result["endpoint"]["summary"]["method_totals"].items():
            lines.append(
                f"| {arm} | {total['verification_calls']} | {total['request_latency_s']:.6f} | {total['cap_hit_requests']} |"
            )
        lines += [
            "",
            "| Contrast | Sweep1 complete-time reduction % | Sweep2 reduction % |",
            "|---|---:|---:|",
        ]
        for name, value in result["decision"]["latency"].items():
            lines.append(
                f"| {name} | {value['sweep_reductions_pct'][0]:.6f} | {value['sweep_reductions_pct'][1]:.6f} |"
            )
    lines += [
        "",
        "Training, extra student projections, reference preparation, parent diagnostics and serving costs are retained separately in RESOURCES.json/RESOURCE_TIMERS.csv. Missing values are not imputed. Nested timers are not added. No training-inclusive speedup is inferred.",
        "",
        "Matched-start repair counts and all parent-bit denominators are in MATCHED_START.json and MATCHED_START_COUNTS.csv. They address only the common first conditioning prefix; all later rounds contribute to endpoint calls.",
        "",
        "The primary endpoint is TRUE versus SHUFFLED call reduction. An interval wholly above zero supports informative placement for this fork/permutation; a supported TRUE versus K call gain additionally establishes incremental verifier benefit. A resolved unfavorable TRUE/SHUFFLED contrast counts against this prioritization rule. An interval spanning zero leaves the successor-bit question open and does not establish equivalence.",
        "",
        "Jointly resample whole requests across all arms:10,000 PCG64 draws, seed20261111, linear2.5/97.5 percentiles. Calls use one repeat after agreement; latency averages repeats within requests. Intervals are unadjusted. Latency benefits require their own favorable interval and both favorable sweep directions.",
        "",
        "The historical58-event gain was only0.28% fewer calls; this one-event screen may lack precision at that scale. Manipulation strength and local repair do not override serving evidence. All11 fallacy checks are retained. Stop after event five and this sealed panel; no automatic follow-up.",
    ]
    manipulation = result["resources"].get("manipulation") or {}
    lines += ["", "| Manipulation quantity | Recorded value |", "|---|---:|"]
    for key in (
        "presentations",
        "eligible_errors",
        "ineligible_errors",
        "mixed_strata",
        "flipped_bits",
        "uniform_expected_flips",
        "reassigned_weight",
        "projection_calls",
        "projection_s",
        "assignment_s",
        "labels_changed",
        "final_export_changed",
    ):
        lines.append(f"| {key} | {manipulation.get(key,'unavailable')} |")
    lines += [
        "",
        "The uniform-permutation benchmark is the sum of2k(n−k)/n over eligible strata; the executed permutation is one fixed deterministic hash assignment. Counts across diverged arms need not match.",
    ]
    return "\n".join(lines) + "\n"


def build(root, output):
    from . import matched_start

    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("Preserve prior report")
    plan = json.loads((root / "sealed/PLAN.json").read_text())
    n = validate_panel_size(plan["panel_size"])
    if plan["analysis"] != ANALYSIS:
        raise ValueError("Changed analysis contract")
    evidence = Evidence(root, plan)
    evidence.read("sealed/PLAN.json")
    panel = evidence.read(plan.get("panel_manifest", "sealed/PANEL.json"))
    canonicals = evidence.read(plan.get("references", "references/CANONICAL.json"))
    matched_panel = evidence.read(
        plan.get(
            "matched_panel",
            plan.get("matched_start_manifest", "sealed/MATCHED_START_PANEL.json"),
        )
    )
    parents = evidence.read(
        plan.get("matched_start_parents", "matched_start/PARENT.json")
    )
    checkpoint_receipt = evidence.read("CHECKPOINTS.json")
    receipts = {
        name: evidence.read(path)
        for name, path in dict(
            run="STATUS.json",
            postflight="POSTFLIGHT.json",
            references="references/STATUS.json",
            worker="evaluation/WORKER.json",
            timing="evaluation/STATUS.json",
            matched_start="matched_start/STATUS.json",
            manipulation=plan.get("manipulation_result", "manipulation/RESULT.json"),
        ).items()
    }
    receipts["campaign_resources"] = evidence.read("RESOURCES.json")
    for arm, item in (checkpoint_receipt or {}).items():
        if item.get("result_path"):
            receipts["training_" + arm] = evidence.read(item["result_path"])
    for arm, path in plan.get("training_results", {}).items():
        receipts["training_" + arm] = evidence.read(path)
    rows = evidence.read("evaluation/MEASUREMENTS.jsonl", lines=True) or []
    errors = []
    audit = dict(status="invalid_or_incomplete", errors=["Missing inputs"], rows=[])
    matched = None
    if panel is not None and canonicals is not None:
        try:
            checkpoints = {m: checkpoint_receipt[m]["checkpoint_path"] for m in METHODS}
            audit = audit_stage(
                root / "evaluation",
                panel["requests"],
                checkpoints,
                canonicals=canonicals,
                sealed_n=n,
                schedule=plan["evaluation_schedule"],
                physical_gpu=plan["serving_gpu_identity"],
                evidence_root=root,
            )
            if parents is not None and matched_panel is not None:
                children = {}
                for matched_index, req in enumerate(matched_panel["requests"]):
                    parent_raw = evidence.read(
                        f"matched_start/parent_{matched_index:03d}.json"
                    )
                    parent_trace = evidence.read(
                        f"matched_start/parent_{matched_index:03d}_trace.json"
                    )
                    if (
                        parent_raw["metadata"]["capture"] != parent_trace
                        or len(parent_raw["metadata"]["trace"]) > 1
                        or parent_raw["response_ids"]
                        != canonicals[req["request_id"]]["response_ids"][
                            : len(parent_raw["response_ids"])
                        ]
                    ):
                        raise ValueError(
                            "Parent raw sidecar, block budget or canonical prefix changed"
                        )
                    rebuilt_parent = matched_start.observation(
                        parent_raw,
                        req,
                        canonicals[req["request_id"]],
                        checkpoint=plan["origin_export"],
                    )
                    if rebuilt_parent != parents[req["request_id"]]:
                        raise ValueError(
                            "Parent normalized observation disagrees with raw trace"
                        )
                    children[req["request_id"]] = {}
                    for m in METHODS:
                        observed = []
                        for repeat in (0, 1):
                            raw = evidence.read(
                                f"evaluation/{req['panel_ordinal']:04d}_r{repeat}_{m}.json"
                            )
                            observed.append(
                                matched_start.observation(
                                    raw,
                                    req,
                                    canonicals[req["request_id"]],
                                    checkpoint=checkpoints[m],
                                )
                            )
                        if observed[0] != observed[1]:
                            raise ValueError(
                                "Child first-round proposals differ across repeats"
                            )
                        children[req["request_id"]][m] = observed[0]
                matched = matched_start.analyze(
                    parents, children, matched_panel["requests"], canonicals
                )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            errors.append(repr(error))
    endpoint, paired = endpoint_from_audit(audit, n)
    if endpoint["status"] != "passed":
        errors += endpoint["errors"]
    for name in ("run", "references", "worker", "matched_start"):
        if (receipts[name] or {}).get("status") != "completed":
            errors.append(name + "_not_completed")
    if (receipts["postflight"] or {}).get("status") != "passed":
        errors.append("postflight_not_passed")
    parent_work = receipts["matched_start"] or {}
    if (
        any(
            parent_work.get(k) != v
            for k, v in dict(
                planned_requests=64,
                attempted_requests=64,
                returned_requests=64,
                maximum_draft_blocks=64,
                complete_response_generations=0,
                unattempted_requests=0,
                unknown_requests=0,
            ).items()
        )
        or type(parent_work.get("draft_blocks")) is not int
        or not 0 <= parent_work["draft_blocks"] <= 64
    ):
        errors.append("parent_one_block_accounting")
    if matched is None or matched.get("status") != "passed":
        errors.append("matched_start_not_valid")
    if (receipts["manipulation"] or {}).get("status") != "passed":
        errors.append("manipulation_not_validated")
    try:
        manipulation_audit = audit_manipulation(
            receipts["manipulation"],
            receipts["training_SHUFFLED"],
            checkpoint_receipt["TRUE"]["weights_identity"],
        )
    except (KeyError, TypeError, ValueError) as error:
        manipulation_audit = dict(status="failed", error=repr(error))
        errors.append("raw_manipulation_reconstruction_failed")
    evidence.remember_tree("evaluation")
    evidence.remember_tree("matched_start")
    stability = evidence.stability()
    if stability["status"] != "passed" or evidence.errors:
        errors.append("changed_or_malformed_evidence")
    complete = not errors
    if not complete:
        endpoint = dict(
            status="invalid_or_incomplete", inference_permitted=False, errors=errors
        )
        paired = []
    result = dict(
        status="complete" if complete else "partial_or_incomplete",
        panel_size=n,
        execution_source=plan.get("execution_source"),
        analysis=ANALYSIS,
        endpoint=endpoint,
        completion_errors=errors,
        decision=decision(endpoint["summary"] if complete else None),
        matched_start=matched,
        resources=receipts,
        manipulation_audit=manipulation_audit,
        input_stability=stability,
        model_work=0,
        automatic_followup_authorized=False,
    )
    output.mkdir(parents=True)
    comparisons = contrast_rows(endpoint)
    pair_table = []
    for pair in paired:
        row = {k: v for k, v in pair.items() if k not in METHODS}
        for method in METHODS:
            row.update({method + "_" + k: v for k, v in pair[method].items()})
        pair_table.append(row)
    sweeps = [
        dict(repeat=int(repeat), method=method, **values)
        for repeat, arms in (
            endpoint.get("summary", {}).get("repeat_totals", {})
        ).items()
        for method, values in arms.items()
    ]
    for name, value in [
        ("SUMMARY.json", result),
        ("PAIRED_REQUESTS.json", paired),
        ("INTEGRITY_AUDIT.json", audit),
        ("SEALED_PANEL.json", panel),
        ("MATCHED_START.json", matched),
        ("CHECKPOINT_IDENTITIES.json", checkpoint_receipt),
        ("RESOURCES.json", receipts),
        ("FALLACY_CHECKS.json", fallacy_checks()),
        (
            "INPUT_MANIFEST.json",
            dict(
                files=evidence.files,
                missing=evidence.missing,
                parse_errors=evidence.errors,
                stability=stability,
            ),
        ),
    ]:
        dump_json(output / name, value)
    for name, value in [
        ("COMPARISONS.csv", comparisons),
        ("TIMING_REPEATS.csv", rows),
        ("PAIRED_REQUESTS.csv", pair_table),
        ("SWEEPS.csv", sweeps),
        ("RESOURCE_TIMERS.csv", resource_rows(receipts)),
        ("MATCHED_START_COUNTS.csv", (matched or {}).get("groups", [])),
    ]:
        dump_csv(output / name, value)
    (output / "REPORT.md").write_text(render(result, comparisons))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(args.root, args.output)
    raise SystemExit(0 if result["status"] == "complete" else 1)
