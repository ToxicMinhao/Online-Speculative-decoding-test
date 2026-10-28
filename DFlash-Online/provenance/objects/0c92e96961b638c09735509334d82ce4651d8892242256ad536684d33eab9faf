"""Read-only final-checkpoint comparison on the original source anchor layouts.

Run as experiments.projected_target.fit_check inside A800 DFlash-Online.
The existing author forward supplies conditioning; only anchor sampling is
replaced with its retained results. No optimizer or generation is constructed.
"""
import argparse
import copy
import csv
import json
import resource
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from specforge.application import resolve_run
from specforge.config import load_config
from specforge.training.assembly import build_model_bundle

from online_feedback.objective import _ReferenceBlocks, _model_options
from online_feedback.projected_target import projected_targets
from online_feedback.training import validate_feature, validate_target_binding
from .fit_metrics import summarize_positions, block_prefix_metrics


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class Tables:
    """Stream compact evidence to disk as each complete layout finishes."""
    def __init__(self, root):
        self.root, self.files, self.writers = root, {}, {}

    def append(self, name, rows):
        if not rows:
            return
        if name not in self.files:
            self.files[name] = (self.root / f"{name}.csv").open("w", newline="")
            self.writers[name] = csv.DictWriter(self.files[name], fieldnames=list(rows[0]))
            self.writers[name].writeheader()
        self.writers[name].writerows(rows)
        self.files[name].flush()

    def close(self):
        for handle in self.files.values():
            handle.close()


def source_plan(campaign):
    """Bind both arms' retained layouts before any model work."""
    groups = []
    for group in ("gsm8k", "codealpaca"):
        receipts = {arm: read_json(campaign / group / arm / "TRAINING.json") for arm in ("C", "P")}
        c, p = receipts["C"], receipts["P"]
        assert len(c["records"]) == len(p["records"]) == 16
        assert c["features"] == p["features"]
        for cr, pr in zip(c["records"], p["records"]):
            for key in ("record_id", "pass", "sampled_anchors"):
                assert cr[key] == pr[key], (group, key)
            assert len(pr["sampled_anchors"]) == 1
            assert len(pr["sampled_anchors"][0]["positions"][0]) == 64
            assert cr["metrics"]["supervised_position_exposure"] == pr["metrics"]["supervised_position_exposure"]
        paths = {arm: str(campaign / group / arm / "trainer/events/update/export/model.safetensors") for arm in ("C", "P")}
        features = {f["record_id"]: f["path"] for f in p["features"]}
        for path in [*paths.values(), *features.values()]:
            assert Path(path).is_file(), path
        groups.append(dict(group=group, checkpoints=paths, features=features, layouts=p["records"]))
    return groups


def matched_labels(ids, mask, anchors, keep, block_size, gamma):
    offsets = torch.arange(block_size, device=ids.device).view(1, 1, -1)
    indices = anchors.unsqueeze(-1) + offsets
    safe = indices.clamp(max=ids.shape[1] - 1)
    labels = ids.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe)
    response = mask.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, safe)
    weights = keep.unsqueeze(-1).float() * (indices < ids.shape[1]).float()
    weights = weights * (offsets > 0).float() * response
    weights *= torch.exp(-(offsets - 1).clamp(min=0).float() / gamma)
    return labels, weights


def exported_model(reference, path):
    draft = copy.deepcopy(reference.draft_model)
    saved = load_file(path, device="cuda")
    current = draft.state_dict()
    assert saved.keys() == current.keys()
    assert all(saved[k].shape == current[k].shape and saved[k].dtype == current[k].dtype for k in saved)
    draft.load_state_dict(saved, strict=True)
    assert all(torch.equal(saved[k], current[k]) for k in saved)
    return _ReferenceBlocks(draft_model=draft, **_model_options(reference)).eval().requires_grad_(False)


def layout_tables(identity, metrics, labels, weights, anchors, raw, record):
    """Keep original denominators and conditional partitions separately."""
    labels, weights, anchors = labels.cpu()[0], weights.cpu()[0], anchors.cpu()[0]
    eligible = weights > 0
    total_weight = weights.sum().item()
    positions, prefixes, summaries = [], [], []
    for arm, values in metrics.items():
        values = {key: value[0] for key, value in values.items()}
        key = {**identity, "arm": arm}
        for block, slot in eligible.nonzero().tolist():
            positions.append({**key, "block": block, "anchor": anchors[block].item(), "slot": slot,
                "absolute_position": anchors[block].item() + slot, "label": labels[block, slot].item(),
                "weight": weights[block, slot].item(), **{k: v[block, slot].item() for k, v in values.items()}})
        for block in range(len(anchors)):
            cap_tail = record["finish_reason"] == "length" and anchors[block].item() + labels.shape[1] >= raw["input_ids"].shape[1]
            prefix = block_prefix_metrics(values["predicted_id"][block, 1:], labels[block, 1:],
                eligible[block, 1:], record["stop_token_ids"], record["finish_reason"], truncated_by_cap=cap_tail)
            prefixes.append({**key, "block": block, "anchor": anchors[block].item(), **prefix})
        for partition in ("all", "reference_correct", "reference_wrong"):
            selected = eligible.clone()
            if partition != "all":
                selected &= values["reference_correct"] if partition == "reference_correct" else ~values["reference_correct"]
            subset_weight = weights[selected].sum().item()
            row = {**key, "partition": partition, "exposure": selected.sum().item(),
                "weight": subset_weight, "original_denominator": total_weight,
                "correct": values["correct"][selected].sum().item(), "repairs": values["repaired"][selected].sum().item(),
                "damage": values["damaged"][selected].sum().item()}
            for field in ("soft_ce", "target_entropy", "kl_to_r", "target_token_logit_margin_vs_best_other", "projected_probability_margin", "projected_log_margin"):
                numerator = (values[field][selected] * weights[selected]).sum().item()
                row[field + "_sum"] = numerator
                row[field + "_mean"] = numerator / subset_weight if subset_weight else None
                row[field + "_contribution"] = numerator / total_weight
            summaries.append(row)
    return positions, prefixes, summaries


@torch.no_grad()
def run(args):
    started = time.perf_counter()
    output, campaign = Path(args.output), Path(args.campaign)
    output.mkdir(parents=True, exist_ok=False)
    groups = source_plan(campaign)
    free = shutil.disk_usage(output).free
    assert free >= 8 * 2**30 + 256 * 2**20, "Need 8 GiB reserve plus 256 MiB diagnostic budget"
    cfg = load_config(args.config)
    write_json(output / "PLAN.json", dict(groups=groups, config=args.config, free_bytes=free,
        reserve_bytes=8*2**30, write_budget_bytes=256*2**20, layouts=32, forwards=96, logical_blocks=6144,
        optimizer_steps=0, new_answers=0, precision="original BF16 model/head; FP32 distribution arithmetic",
        reduction="equal mean of each response-layout weighted mean; partitions retain original denominator",
        negative_kl="retain numerical negative roundoff; no clamp", seed=17,
        source_bindings=str(campaign.parent / "BINDINGS.json"),
        initial_objective_check_atol=2e-5, initial_objective_check_rtol=2e-5))
    shutil.copyfile(args.config, output / "execution_config.yaml")
    for source in (Path(__file__), Path(__file__).with_name("fit_metrics.py")):
        shutil.copyfile(source, output / source.name)
    torch.cuda.set_device(0)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    bundle = build_model_bundle(cfg, algorithm=resolve_run(cfg).algorithm)
    reference = _ReferenceBlocks(draft_model=bundle.draft_model, **_model_options(bundle.model)).eval().requires_grad_(False)
    assert reference.block_size == 16 and reference.objective_chunk_blocks == 8
    assert reference.loss_decay_gamma == 7 and reference.attention_backend == "eager"
    torch.cuda.synchronize()
    loading_s = time.perf_counter() - started
    tables = Tables(output)
    completed = forwards = 0
    group_loading_s = []
    analysis_started = time.perf_counter()
    for group in groups:
        load_started = time.perf_counter()
        models = {"F": reference, **{a: exported_model(reference, p) for a, p in group["checkpoints"].items()}}
        torch.cuda.synchronize()
        group_loading_s.append(time.perf_counter() - load_started)
        previous_window_exposure = 0
        for layout in group["layouts"]:
            layout_started = time.perf_counter()
            rid = layout["record_id"]
            raw = torch.load(group["features"][rid], map_location="cpu", weights_only=False)
            validate_feature(raw, max_length=cfg.data.max_length)
            validate_target_binding(raw)
            record = read_json(campaign / group["group"] / "F/records" / f"{rid}.json")
            assert raw["input_ids"][0].tolist() == record["prompt_ids"] + record["response_ids"]
            ids, mask = raw["input_ids"].cuda(), raw["loss_mask"].cuda()
            hidden = raw["hidden_states"].cuda()
            saved = layout["sampled_anchors"][0]
            anchors = torch.tensor(saved["positions"], device="cuda", dtype=torch.int64)
            keep = torch.tensor(saved["keep_mask"], device="cuda", dtype=torch.bool)
            labels, weights = matched_labels(ids, mask, anchors, keep, 16, 7.0)
            exposure = (weights > 0).sum().item()
            expected_window_exposure = layout["metrics"]["supervised_position_exposure"]
            assert exposure + previous_window_exposure == expected_window_exposure
            previous_window_exposure = 0 if layout["optimizer_stepped"] else expected_window_exposure
            draft_hidden = {}
            rng_before = torch.cuda.get_rng_state().clone()
            for arm, model in models.items():
                model.fixed_anchors = (anchors, keep)
                aa, kk, hh = model._forward_draft_blocks(ids, hidden, mask)
                assert torch.equal(aa, anchors) and torch.equal(kk, keep)
                draft_hidden[arm] = hh.reshape(1, 64, 16, -1)
                forwards += 1
            assert torch.equal(rng_before, torch.cuda.get_rng_state()), "Fixed-layout forward consumed RNG"
            chunks = {a: [] for a in models}
            initial_loss_sum = torch.zeros((), device="cuda")
            initial_denominator = torch.zeros((), device="cuda")
            for start in range(0, 64, 8):
                sl = slice(start, start + 8)
                f_logits = reference.lm_head(draft_hidden["F"][:, sl])
                targets = projected_targets(f_logits, labels[:, sl])
                for arm in models:
                    logits = f_logits if arm == "F" else reference.lm_head(draft_hidden[arm][:, sl])
                    values = summarize_positions(f_logits, logits, targets, labels[:, sl], weights[:, sl])
                    assert all(torch.isfinite(v).all() for k, v in values.items() if k != "projected_log_margin")
                    chunks[arm].append(values)
                    if completed == 0 and arm == "F":
                        initial_loss_sum += (values["soft_ce"].cuda() * weights[:, sl]).sum()
                        initial_denominator += weights[:, sl].sum()
                del f_logits, logits, targets
            if completed == 0:
                observed = (initial_loss_sum / initial_denominator).item()
                original = layout["metrics"]["objective_loss"]
                tolerance = 2e-5 + 2e-5 * abs(original)
                qualification = dict(observed=observed, original=original, absolute_error=abs(observed-original),
                    tolerance=tolerance, passed=abs(observed-original) <= tolerance,
                    description="Released F reproduces the first P objective before its first optimizer step")
                write_json(output / "QUALIFICATION.json", qualification)
                assert qualification["passed"], qualification
            metrics = {a: {k: torch.cat([c[k] for c in cc], dim=1) for k in cc[0]} for a, cc in chunks.items()}
            identity = dict(group=group["group"], record_id=rid, pass_index=layout["pass"])
            positions, prefixes, summaries = layout_tables(identity, metrics, labels, weights, anchors, raw, record)
            tables.append("positions", positions)
            tables.append("prefixes", prefixes)
            tables.append("layouts", summaries)
            torch.cuda.synchronize()
            completed += 1
            elapsed = time.perf_counter() - analysis_started
            print(json.dumps(dict(**identity, completed_layouts=completed, forwards=forwards,
                layout_s=time.perf_counter()-layout_started, elapsed_s=elapsed,
                estimated_remaining_s=elapsed/completed*(32-completed))), flush=True)
        del models, draft_hidden
    tables.close()
    assert completed == 32 and forwards == 96
    write_json(output / "COMPLETION.json", dict(layouts=completed, forwards=forwards, logical_blocks=forwards*64,
        optimizer_steps=0, generated_answers=0, initial_loading_s=loading_s, group_loading_s=group_loading_s,
        wall_s=time.perf_counter()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), free_bytes_after=shutil.disk_usage(output).free,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        torch_version=torch.__version__, device=torch.cuda.get_device_name()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
