"""Causal routing and separate cumulative protection on finite-vocabulary fixtures."""
import copy

import numpy as np
import pytest

from .core import Factors
from .qualification import (ToyObjective, fusion_fixture, install, run_fixtures,
                            toy_block, toy_evaluator, toy_state)
from .state import BankState, action_key, actions


def test_deterministic_qualification_paths_and_fusion():
    receipt = run_fixtures()
    assert receipt["passed"] and all(receipt["checks"].values())
    assert fusion_fixture()["passed"]


def test_action_set_uses_only_allocated_slots_and_has_seventeen_actions():
    assert len(actions([False] * 4)) == 1
    choices = actions([True] * 4)
    assert len(choices) == 17
    assert len({action_key(a) for a in choices}) == 17
    assert all(np.count_nonzero(a) <= 2 for a in choices)
    partial = actions([False, True, False, True])
    assert len(partial) == 5
    assert all(a[0] == a[2] == 0 for a in partial)
    assert any(np.array_equal(a, [0, .5, 0, .5]) for a in partial)
    assert any(np.array_equal(a, [0, 1, 0, 1]) for a in partial)


def test_no_memory_is_frozen_even_if_an_unmeasured_module_is_allocated():
    state = toy_state()
    install(state, 0, [[-1., 0.], [2., 0.]])
    before = state.identity()
    read = state.read(np.array([1., 0.]), 100)
    assert not read.alpha.any() and read.effective.rank == 0
    assert state.identity() == before
    with pytest.raises(ValueError, match="normalized"):
        state.read(np.array([2., 0.]), 100)


def test_response_kernel_uses_one_response_unit_and_zero_gain_prior():
    state = toy_state()
    install(state, 0, [[-1., 0.], [2., 0.]])
    state.add_transfer([toy_block("r1:b0"), toy_block("r1:b1")], [1, 0, 0, 0], toy_evaluator)
    state.refresh(toy_evaluator)
    read = state.read(np.array([1., 0.]), 100)
    assert read.diagnostics["memory_responses"] == 1
    assert read.forecast == pytest.approx(.5)  # 1 gain / (one prior + one response)
    assert np.array_equal(read.alpha, [1, 0, 0, 0])
    assert len(state.metric_memory()) == 2
    assert {r.record_id for r in state.metric_memory()} == {"r1:b0", "r1:b1"}


def test_read_uses_exact_boundary_progress_not_one_plus_prefix():
    state = toy_state()
    install(state, 0, [[-1., 0.], [2., 0.]])
    state.add_transfer([toy_block()], [1, 0, 0, 0], toy_evaluator)

    def boundary_evaluator(block, factors):
        result = toy_evaluator(block, factors)
        # Both paths terminate at the same one-token cap despite differing prefix.
        result["progress"] = 1.
        return result

    state.refresh(boundary_evaluator)
    read = state.read(np.array([1., 0.]), 100)
    assert not read.alpha.any()  # prefix gain alone cannot invent boundary progress
    singleton = read.diagnostics["actions"][1]
    assert singleton["predicted_gain"] == 0
    assert singleton["predicted_prefix_gain"] == pytest.approx(.5)


def test_cache_refresh_only_evaluates_new_or_changed_actions():
    state = toy_state()
    install(state, 0, [[-1., 0.], [2., 0.]])
    state.add_transfer([toy_block()], [1, 0, 0, 0], toy_evaluator)
    assert state.refresh(toy_evaluator)["head_evaluations"] == 2
    assert state.refresh(toy_evaluator)["head_evaluations"] == 0
    install(state, 1, [[.1, 0.], [0., 0.]])
    # frozen/old singleton remain valid: new singleton + mean + sum need values.
    assert state.refresh(toy_evaluator, changed_owner=1)["head_evaluations"] == 3
    state.core.versions[0] += 1
    with pytest.raises(ValueError, match="Stale"):
        state.read(np.array([1., 0.]), 100)
    # Changing owner zero invalidates its singleton, pair mean and pair sum.
    assert state.refresh(toy_evaluator, changed_owner=0)["head_evaluations"] == 3
    state.read(np.array([1., 0.]), 100)


def test_protection_preserves_earlier_correct_positions_in_improved_prefix():
    state = toy_state()
    install(state, 0, [[-1., 0.], [2., 0.]])
    block = toy_block()
    block.update(h=np.asarray([[0., 1.], [1., 0.]], dtype=np.float32),
                 teacher=np.asarray([[0., 1.], [0., 1.]]), omega=np.asarray([.5, .5]),
                 token_ids=np.asarray([1, 1]))
    state.add_transfer([block], [1, 0, 0, 0], toy_evaluator)
    assert state.protection[0]["reference_prefix"] == 2
    candidate = list(state.core.modules)
    # Keep the newly repaired second position, but break the earlier correct one.
    candidate[0] = Factors(np.eye(2), np.asarray([[-1., 2.], [2., -1.]]))
    result = toy_evaluator(block, candidate[0])
    assert result["predictions"][1] == 1 and result["predictions"][0] == 0
    valid, detail = state.protect(candidate, toy_evaluator)
    assert not valid and "prefix_protection" in detail["reasons"]
    assert "native_precision" in detail["reasons"]


def test_nonpositive_reference_margin_uses_native_check_without_invented_floor():
    state = toy_state()
    install(state, 0, [[-1., 0.], [2., 0.]])
    state.add_transfer([toy_block()], [1, 0, 0, 0], toy_evaluator)
    # Explicit fixture for a native/math rounding disagreement branch.
    state.protection[0]["reference_margins"][:] = -.01
    state.protection[0]["kl_protected"] = False

    def preserved_native(block, factors):
        result = toy_evaluator(block, factors)
        result.update(margins=np.asarray([-.2]), prefix=1)
        return result

    assert state.protect(state.core.modules, preserved_native)[0]

    def broken_native(block, factors):
        result = preserved_native(block, factors)
        result["prefix"] = 0
        return result

    valid, detail = state.protect(state.core.modules, broken_native)
    assert not valid and detail["reasons"] == ["native_precision"]


def test_cumulative_kl_budget_does_not_refresh_after_permitted_erosion():
    state = toy_state()
    install(state, 0, [[-1., 0.], [2., 0.]])
    state.add_transfer([toy_block()], [1, 0, 0, 0], toy_evaluator)
    record = state.protection[0]
    record["prefix_protected"] = False
    reference, gain = record["reference_loss"], record["kl_gain"]

    def loss_at_fraction(fraction):
        def evaluator(block, factors):
            result = toy_evaluator(block, factors)
            result["loss"] = reference + fraction * gain
            return result
        return evaluator

    assert state.protect(state.core.modules, loss_at_fraction(.2))[0]
    assert state.protect(state.core.modules, loss_at_fraction(.24))[0]
    valid, detail = state.protect(state.core.modules, loss_at_fraction(.26))
    assert not valid and detail["reasons"] == ["kl_protection"]
    assert record["reference_loss"] == reference and record["kl_gain"] == gain


def test_unused_historical_coefficients_and_duplicate_insertions_are_rejected():
    state = toy_state()
    with pytest.raises(ValueError, match="Unused"):
        state.add_transfer([toy_block()], [1, 0, 0, 0], toy_evaluator)
    with pytest.raises(ValueError, match="Duplicate"):
        state.add_transfer([toy_block(), toy_block()], [0, 0, 0, 0], toy_evaluator)
    assert not state.memory
    state.add_transfer([toy_block()], [0, 0, 0, 0], toy_evaluator)
    with pytest.raises(ValueError, match="Duplicate"):
        state.add_transfer([toy_block()], [0, 0, 0, 0], toy_evaluator)


def test_write_preserves_protection_references_and_refreshes_cache():
    state = toy_state()
    install(state, 0, [[-1., 0.], [2., 0.]])
    state.add_transfer([toy_block()], [1, 0, 0, 0], toy_evaluator)
    state.refresh(toy_evaluator)
    reference = copy.deepcopy(state.protection)
    result = state.write(ToyObjective(toy_block("r2:b0", target=(1., 0.), token=0,
                                              response_id="r2")), toy_evaluator)
    assert result["committed"] and result["owner"] == 1
    assert len(state.protection) == len(reference)
    for before, after in zip(reference, state.protection):
        for key in before:
            np.testing.assert_equal(before[key], after[key])
    assert result["cache_refresh"]["head_evaluations"] == 3
    assert result["trials"] and "attempts" in result["trials"][0]
    assert "trials" not in state.write_log[-1]
    assert state.write_log[-1]["owner"] == result["owner"]
    assert state.write_log[-1]["versions"] == result["versions"]
    assert state.protect(state.core.modules, toy_evaluator)[0]
    state.read(np.array([1., 0.]), 100)


def test_snapshot_replay_is_independent_and_read_does_not_mutate_state():
    state = toy_state()
    install(state, 0, [[-1., 0.], [2., 0.]])
    state.add_transfer([toy_block()], [1, 0, 0, 0], toy_evaluator)
    state.refresh(toy_evaluator)
    restored = BankState.from_snapshot(state.snapshot())
    before = restored.identity()
    restored.read(np.array([1., 0.]), 100)
    assert restored.identity() == before == state.identity()
    restored.memory[0]["descriptor"][0] = 0
    assert state.memory[0]["descriptor"][0] == 1


def test_missing_cost_profile_cannot_silently_route_with_memory():
    state = BankState(2)
    state.add_transfer([toy_block()], [0, 0, 0, 0], toy_evaluator)
    state.refresh(toy_evaluator)
    with pytest.raises(ValueError, match="qualified cycle-cost"):
        state.read(np.array([1., 0.]), 100)
