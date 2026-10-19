"""Synthetic inventory fixtures only; never inspect a recorded campaign."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.dflash_correction_bank.artifacts import binding
from experiments.dflash_osd_attribution.prepare import (
    Resolver, eos_inventory, inventory_campaign, inventory_checkpoint,
)
from experiments.dflash_osd_attribution.protocol import digest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return binding(path)


def checkpoint(path, name):
    save(path / "config.json", {"checkpoint": name})
    (path / "model.safetensors").write_bytes(("synthetic-" + name).encode())
    files = {
        p.name: {key: value for key, value in binding(p).items() if key != "path"}
        for p in sorted(path.iterdir())
    }
    return dict(path=str(path), checkpoint_id=name, files=files,
                checkpoint_identity_sha256=digest(files))


def synthetic_campaign(tmp_path):
    root = tmp_path / "campaign_root"
    stream, turn = "wildchat_00", 20
    stream_dir = root / "campaign" / stream
    native_dir = stream_dir / "native" / stream
    f = checkpoint(tmp_path / "released", "F")
    o = checkpoint(tmp_path / "exact_serving_D19", "D19")
    target_dir = tmp_path / "target"
    generation = save(target_dir / "generation_config.json", {"eos_token_id": [99, 100]})
    files = {"generation_config.json": {key: value for key, value in generation.items() if key != "path"}}
    config = dict(episode_id=stream, bank_qualification=False,
                  target=dict(model_path=str(target_dir), checkpoint_files=files),
                  tokenizer_path=str(target_dir))
    save(native_dir / "NATIVE_CONFIG.json", config)
    projection_path = native_dir / "training_worker" / "bank" / "fixed_projections.npz"
    projection_path.parent.mkdir(parents=True)
    projection_path.write_bytes(b"synthetic fixed projection artifact")
    original_projection_binding = binding(projection_path)
    inputs, outputs = [1, 2, 3], list(range(10, 22))
    proposals = outputs[1:10]
    trace = dict(input_token_ids=inputs, final_output_token_ids=outputs,
                 events=[dict(kind="verification", output_offset=1,
                              draft_proposed_token_ids=proposals,
                              draft_proposed_positions=list(range(len(inputs)+1, len(inputs)+10)),
                              verification_input_token_ids=[outputs[0]]+proposals,
                              context=dict(committed_output_prefix_length=1))])
    trajectory = save(native_dir / "trajectory.json", trace)
    states = {
        arm: dict(turn=turn, after_turn=0 if arm == "F" else turn-1,
                  state_sha256=cp["checkpoint_identity_sha256"], checkpoint=cp)
        for arm, cp in (("F", f), ("OSD", o))
    }
    scores = [dict(arm=arm, turn=turn, repeat=repeat, trajectory=trajectory,
                   same_generation_valid=True, capture=False, training_data=False,
                   input_ids=inputs, output_ids=outputs,
                   state_sha256=states[arm]["state_sha256"])
              for arm in ("F", "OSD") for repeat in range(3)]
    result = dict(episode_id=stream, status="complete", qualification=False,
                  planned_requests=turn, scores=scores,
                  startup=dict(bank_preparation=dict(projections=original_projection_binding)),
                  pre_request_states=[dict(turn=turn, states=states)],
                  canonical=[dict(arm="BANK", turn=turn, input_ids=inputs, output_ids=outputs)])
    result_path = stream_dir / "RESULT.json"
    save(result_path, result)
    return dict(root=root, stream=stream, result=result, result_path=result_path,
                original_projection_binding=original_projection_binding,
                projection_path=projection_path, frozen=f, osd=o, config=config,
                trajectory=trajectory)


def test_resolver_relocation_preserves_original_hash(tmp_path):
    original = tmp_path / "old" / "nested" / "value.json"
    reference = save(original, {"value": 3})
    moved = tmp_path / "new" / "nested" / "value.json"
    moved.parent.mkdir(parents=True)
    original.rename(moved)
    resolver = Resolver([(tmp_path / "old", tmp_path / "new")])
    assert resolver.checked(reference) == moved
    assert resolver.load(reference) == {"value": 3}
    receipt = resolver.present(reference)
    assert receipt["original"] == reference
    assert receipt["resolved"]["path"] == str(moved)
    moved.write_text('{"value": 4}')
    assert resolver.present(reference)["status"] == "unavailable"


def test_resolver_does_not_match_sibling_path_prefix(tmp_path):
    resolver = Resolver([(tmp_path / "old", tmp_path / "new")])
    sibling = tmp_path / "older" / "file"
    assert resolver.path(sibling) == sibling


def test_checkpoint_inventory_uses_every_exact_original_file(tmp_path):
    cp = checkpoint(tmp_path / "weights", "D19")
    original = copy.deepcopy(cp)
    receipt = inventory_checkpoint(cp, Resolver())
    assert receipt["status"] == "available"
    assert len(receipt["files"]) == 2
    assert receipt["resolved_checkpoint"]["checkpoint_identity_sha256"] == cp["checkpoint_identity_sha256"]
    assert cp == original
    (Path(cp["path"]) / "model.safetensors").unlink()
    checkpoint(tmp_path / "later_weights", "D20")
    missing = inventory_checkpoint(cp, Resolver())
    assert missing["status"] == "unavailable"
    assert missing["resolved_checkpoint"] is None
    assert missing["original"] == original


def test_checkpoint_inventory_rejects_identity_mismatch_and_path_escape(tmp_path):
    cp = checkpoint(tmp_path / "weights", "D19")
    cp["checkpoint_identity_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        inventory_checkpoint(cp, Resolver())
    cp["files"] = {"../model.safetensors": {"sha256": "irrelevant", "bytes": 1}}
    cp["checkpoint_identity_sha256"] = digest(cp["files"])
    with pytest.raises(ValueError, match="Unsafe"):
        inventory_checkpoint(cp, Resolver())


@pytest.mark.parametrize("ids,expected", [(99, [99]), ([100, 99, 100], [99, 100])])
def test_eos_comes_from_pinned_configuration(tmp_path, ids, expected):
    ref = save(tmp_path / "generation_config.json", {"eos_token_id": ids})
    config = dict(target=dict(model_path=str(tmp_path), checkpoint_files={
        "generation_config.json": {k: v for k, v in ref.items() if k != "path"}}))
    actual, retained = eos_inventory(config, Resolver())
    assert actual == expected and retained == ref
    save(tmp_path / "generation_config.json", {"eos_token_id": 7})
    with pytest.raises(ValueError, match="checksum"):
        eos_inventory(config, Resolver())


def test_eos_config_fallback_is_explicit_not_a_default_token(tmp_path):
    ref = save(tmp_path / "config.json", {"eos_token_id": 77})
    config = dict(target=dict(model_path=str(tmp_path), checkpoint_files={
        "config.json": {k: v for k, v in ref.items() if k != "path"}}))
    assert eos_inventory(config, Resolver())[0] == [77]
    ref = save(tmp_path / "config.json", {"not_eos": 77})
    config["target"]["checkpoint_files"]["config.json"] = {k: v for k, v in ref.items() if k != "path"}
    with pytest.raises(ValueError, match="no usable EOS"):
        eos_inventory(config, Resolver())


def test_inventory_uses_final_saved_pre_learning_checkpoint_and_original_projection(tmp_path):
    data = synthetic_campaign(tmp_path)
    plan = inventory_campaign(data["root"])
    assert plan["available_cases"] == 1
    assert plan["planned_draft_block_evaluations"] == 2
    assert len(plan["trace_inventory"]) == 6
    assert plan["training_authorized"] is False
    anchor = plan["anchors"][0]
    assert anchor["turn"] == 20
    assert anchor["checkpoints"]["OSD"]["path"] == data["osd"]["path"]
    assert anchor["checkpoints"]["OSD"]["checkpoint_identity_sha256"] == data["osd"]["checkpoint_identity_sha256"]
    assert anchor["projections"] == data["original_projection_binding"]
    assert anchor["canonical_prefix_ids"] == [1, 2, 3, 10]
    assert anchor["eos_ids"] == [99, 100]


def test_missing_serving_checkpoint_does_not_use_a_later_available_checkpoint(tmp_path):
    data = synthetic_campaign(tmp_path)
    (Path(data["osd"]["path"]) / "model.safetensors").unlink()
    later = checkpoint(tmp_path / "later_D20", "D20")
    before = {p: p.read_bytes() for p in Path(later["path"]).iterdir()}
    plan = inventory_campaign(data["root"])
    assert plan["available_cases"] == 0 and plan["anchors"] == []
    case = plan["cases"][0]
    assert "OSD: exact serving checkpoint unavailable" in case["issues"]
    assert case["checkpoint_inventory"]["OSD"]["resolved_checkpoint"] is None
    assert case["checkpoint_inventory"]["OSD"]["original"] == data["osd"]
    assert all(p.read_bytes() == content for p, content in before.items())


def test_changed_same_path_projection_is_rejected_by_historical_binding(tmp_path):
    data = synthetic_campaign(tmp_path)
    original = data["projection_path"].read_bytes()
    # Equal length ensures a checksum test, not merely a size guard.
    data["projection_path"].write_bytes(b"X" + original[1:])
    plan = inventory_campaign(data["root"])
    assert plan["available_cases"] == 0 and not plan["anchors"]
    assert any("checksum" in issue for issue in plan["cases"][0]["issues"])


def test_missing_historical_projection_binding_is_not_recreated_from_current_file(tmp_path):
    data = synthetic_campaign(tmp_path)
    del data["result"]["startup"]["bank_preparation"]["projections"]
    save(data["result_path"], data["result"])
    plan = inventory_campaign(data["root"])
    assert plan["available_cases"] == 0 and not plan["anchors"]
    assert any("projections" in issue for issue in plan["cases"][0]["issues"])


@pytest.mark.parametrize("change", ["qualification", "after_turn", "output_mismatch"])
def test_wrong_campaign_or_checkpoint_chronology_does_not_launch(tmp_path, change):
    data = synthetic_campaign(tmp_path)
    if change == "qualification":
        data["result"]["qualification"] = True
    elif change == "after_turn":
        data["result"]["pre_request_states"][0]["states"]["OSD"]["after_turn"] = 20
    else:
        data["result"]["scores"][0]["output_ids"] = [999]
    save(data["result_path"], data["result"])
    plan = inventory_campaign(data["root"])
    assert plan["available_cases"] == 0 and not plan["anchors"]
    assert plan["cases"][0]["issues"]
