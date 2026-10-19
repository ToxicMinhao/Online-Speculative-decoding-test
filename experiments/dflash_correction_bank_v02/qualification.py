"""Deterministic numerical fixtures supplementary to the three-request GPU gate.

Passing these fixtures does not qualify model integration or demonstrate useful
adaptation. The caller must also execute the native zero/mask/feedback/precision
and output-preservation checks on the declared first three requests.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.special import logsumexp, softmax

from .core import BankConfig, Factors, GradientFactors, fuse
from .state import BankState, action_key, actions


def toy_block(identifier="r1:b0", *, h=(1., 0.), target=(0., 1.), token=1,
              response_id="r1", descriptor=(1., 0.)):
    """A finite vocabulary test record with one candidate, not a model trace."""
    h = np.asarray(h, dtype=np.float32)[None]
    target = np.asarray(target, dtype=np.float64)[None]
    return dict(block_id=identifier, response_id=response_id, h=h, omega=np.ones(1),
                coefficients=np.zeros(4), descriptor=np.asarray(descriptor),
                E=np.eye(h.shape[1]), teacher=target,
                token_ids=np.asarray([token], dtype=np.int64))


def toy_evaluator(block, factors):
    h = np.asarray(block["h"], dtype=np.float64)
    E = np.asarray(block["E"], dtype=np.float64)
    probabilities = np.asarray(block["teacher"], dtype=np.float64)
    weights = np.asarray(block["omega"], dtype=np.float64)
    weights /= weights.sum()
    z0 = h @ E.T
    z = (h + factors.apply(h)) @ E.T
    loss = float(weights @ (logsumexp(z, axis=1) - logsumexp(z0, axis=1)
                             - np.sum(probabilities * (z - z0), axis=1)))
    ids = np.asarray(block["token_ids"])
    winners = np.argmax(z.astype(np.float32), axis=1)
    margins = []
    for row, token in zip(z, ids):
        competitors = row.copy()
        competitors[token] = -np.inf
        margins.append(row[token] - np.max(competitors))
    prefix = 0
    for predicted, canonical in zip(winners, ids):
        if predicted != canonical:
            break
        prefix += 1
    return dict(loss=loss, margins=np.asarray(margins), prefix=prefix,
                progress=1 + prefix, predictions=winners)


def toy_state(d=2):
    return BankState(d, config=BankConfig(rank=1), cost_profile={
        "bins": {"4096": {"cycle_s": {"0": 1., "1": 1., "2": 1.}}}})


def install(state, owner, dense):
    """Explicit fixture setup; production modules allocate only via accepted writes."""
    u, s, vt = np.linalg.svd(np.asarray(dense, dtype=np.float64), full_matrices=False)
    k = state.core.config.rank
    state.core.modules[owner] = Factors(u[:, :k] * np.sqrt(s[:k]),
                                        np.sqrt(s[:k, None]) * vt[:k])
    state.core.allocated[owner] = True
    state.core.versions[owner] += 1


class ToyObjective:
    def __init__(self, block):
        self.block = block
        self.features = np.asarray(block["h"], dtype=np.float64)
        self.weights = np.asarray(block["omega"], dtype=np.float64)
        self.weights = self.weights / self.weights.sum()

    def loss(self, factors):
        return toy_evaluator(self.block, factors)["loss"]

    def loss_gradient(self, factors):
        E = self.block["E"]
        z = (self.features + factors.apply(self.features)) @ E.T
        residual = (softmax(z, axis=1) - self.block["teacher"]) @ E
        return self.loss(factors), GradientFactors((self.weights[:, None] * residual).T,
                                                   self.features)


def fusion_fixture():
    rng = np.random.default_rng(17)
    d, k = 12, 3
    modules = [Factors(rng.normal(size=(d, k)), rng.normal(size=(k, d))) for _ in range(4)]
    h = rng.normal(size=(9, d)).astype(np.float32)
    errors = []
    for alpha in ([0., 0., 0., 0.], [1., 0., 0., 0.],
                  [.5, .5, 0., 0.], [1., 1., 0., 0.]):
        compiled = fuse(modules, alpha)
        explicit = sum(float(a) * module.apply(h) for a, module in zip(alpha, modules))
        error = float(np.max(np.abs(compiled.apply(h) - explicit)))
        assert np.allclose(compiled.apply(h), explicit, atol=2e-5, rtol=2e-6)
        assert compiled.rank == k * np.count_nonzero(alpha)
        errors.append(error)
    return {"passed": True, "scope": "FP32 thin-factor fixture; native GPU hook checked separately",
            "max_abs_errors": errors, "atol": 2e-5, "rtol": 2e-6,
            "pair_rank_not_recompressed": True}


def run_fixtures():
    """Exercise protection paths that three natural requests need not produce."""
    checks = {}
    # An action genuinely used before writing repairs the next record's token.
    state = toy_state()
    install(state, 0, [[-1., 0.], [2., 0.]])
    block = toy_block()
    state.add_transfer([block], [1, 0, 0, 0], toy_evaluator)
    assert len(state.protection) == 1
    record = copy.deepcopy(state.protection[0])
    assert record["kl_protected"] and record["prefix_protected"]
    assert state.protect(state.core.modules, toy_evaluator)[0]
    degraded = list(state.core.modules)
    degraded[0] = Factors.zero(2)
    valid, detail = state.protect(degraded, toy_evaluator)
    assert not valid and "kl_protection" in detail["reasons"]
    assert "prefix_protection" in detail["reasons"]
    assert "native_precision" in detail["reasons"]
    checks["rollback_path_rejects_lost_kl_and_native_prefix"] = True

    # Bounds remain anchored at the original gain after repeated successful checks.
    original_identity = state.identity()
    for _ in range(3):
        assert state.protect(state.core.modules, toy_evaluator)[0]
    assert state.identity() == original_identity
    np.testing.assert_array_equal(state.protection[0]["reference_margins"], record["reference_margins"])
    assert state.protection[0]["reference_loss"] == record["reference_loss"]
    checks["cumulative_references_and_budgets_do_not_reset"] = True

    # A fresh unused module has historical coefficient zero; it cannot change
    # that record's served composition, even if its singleton is very large.
    separate = list(state.core.modules)
    separate[1] = Factors(np.asarray([[100.], [-100.]]), np.asarray([[1., 0.]]))
    assert state.protect(separate, toy_evaluator)[0]
    assert all(a[1] == 0 for a in actions(state.core.allocated))
    checks["unused_slot_has_no_historical_coefficient_or_read_action"] = True

    # A conflicting new source can use an independent unused slot, preserving
    # the old transfer record and its original budget.
    state.refresh(toy_evaluator)
    result = state.write(ToyObjective(toy_block("r2:b0", target=(1., 0.), token=0,
                                              response_id="r2")), toy_evaluator)
    assert result["committed"] and result["owner"] == 1
    assert state.core.allocated[:2] == [True, True]
    assert state.protect(state.core.modules, toy_evaluator)[0]
    assert len(state.protection) == 1
    assert state.protection[0]["reference_loss"] == record["reference_loss"]
    checks["new_slot_write_preserves_historical_composition"] = True

    # KL improvement without prefix improvement creates only a KL certificate.
    kl_state = toy_state()
    install(kl_state, 0, [[1., 0.], [0., 0.]])
    kl_state.add_transfer([toy_block(h=(2., 0.), target=(1., 0.), token=0)],
                          [1, 0, 0, 0], toy_evaluator)
    assert kl_state.protection[0]["kl_protected"]
    assert not kl_state.protection[0]["prefix_protected"]
    checks["kl_and_prefix_protection_are_separate"] = True

    # A sharp correct argmax can improve the prefix while worsening forward KL.
    prefix_state = toy_state(3)
    h = np.array([2., 1., 0.])
    desired = np.array([0., 10., -10.])
    W = np.outer(desired-h, h) / (h @ h)
    install(prefix_state, 0, W)
    prefix_state.add_transfer([toy_block(h=h, target=(.3, .4, .3), token=1)],
                              [1, 0, 0, 0], toy_evaluator)
    assert prefix_state.protection[0]["prefix_protected"]
    assert not prefix_state.protection[0]["kl_protected"]
    checks["prefix_gain_without_kl_gain_is_retained"] = True

    # Post-write counterfactual cache refresh does not mint transfer certificates.
    source_state = toy_state()
    source_state.add_transfer([toy_block()], [0, 0, 0, 0], toy_evaluator)
    assert not source_state.protection
    source_state.refresh(toy_evaluator)
    install(source_state, 0, [[-1., 0.], [2., 0.]])
    source_state.refresh(toy_evaluator, changed_owner=0)
    assert not source_state.protection
    current = source_state.read(np.array([1., 0.]), 100)
    assert current.alpha[0] == 1 and current.forecast > 0
    checks["source_fit_cache_is_not_hard_transfer_protection"] = True
    checks["turn_two_can_use_fitted_module_without_forced_inactivity"] = True

    source_state.core.versions[0] += 1
    try:
        source_state.read(np.array([1., 0.]), 100)
    except ValueError as error:
        assert "Stale" in str(error)
    else:
        raise AssertionError("stale action cache was accepted")
    source_state.refresh(toy_evaluator, changed_owner=0)
    source_state.read(np.array([1., 0.]), 100)
    checks["stale_current_version_routing_evidence_is_rejected"] = True
    checks["factor_fusion_matches_explicit_sum"] = fusion_fixture()["passed"]
    return {"scope": "deterministic finite-vocabulary CPU fixtures; not native GPU qualification",
            "passed": all(checks.values()), "checks": checks}
