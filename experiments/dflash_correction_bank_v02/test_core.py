"""Synthetic numerical contracts, not evidence of native LLM efficacy."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import linalg

from . import core
from .core import (BankConfig, CorrectionBank, Factors, GradientFactors,
                   MetricMemory, NormalizedMetric, build_metric, difference_norm,
                   fuse, spectral_proposal)


class Quadratic:
    def __init__(self, target, curvature=1., factored=False):
        self.target = np.asarray(target, dtype=np.float64)
        self.features = np.eye(len(target))
        self.weights = np.ones(len(target)) / len(target)
        self.curvature = curvature
        self.factored = factored

    def loss(self, factors):
        residual = factors.dense() - self.target
        return .5 * self.curvature * (np.sum(residual ** 2) - np.sum(self.target ** 2))

    def loss_gradient(self, factors):
        G = self.curvature * (factors.dense() - self.target)
        return self.loss(factors), (GradientFactors(G, np.eye(len(G))) if self.factored else G)


def factors(dense, rank=None):
    u, s, vt = linalg.svd(dense, full_matrices=False)
    k = len(s) if rank is None else rank
    return Factors(u[:, :k] * np.sqrt(s[:k]), np.sqrt(s[:k, None]) * vt[:k])


def test_fusion_preserves_two_module_rank_and_float32_masters():
    d, k = 80, 32
    first = Factors(np.eye(d)[:, :k], np.eye(d)[:k])
    second = Factors(np.eye(d)[:, k:2*k], np.eye(d)[k:2*k])
    merged = fuse([first, second, Factors.zero(d)], [.5, 1., 0.])
    assert merged.rank == 64
    assert merged.B.dtype == merged.A.dtype == np.float32
    assert np.linalg.matrix_rank(merged.dense()) == 64
    np.testing.assert_array_equal(merged.dense(), .5 * first.dense() + second.dense())
    h = np.random.default_rng(17).normal(size=(7, d)).astype(np.float32)
    np.testing.assert_allclose(merged.apply(h), .5 * first.apply(h) + second.apply(h), atol=1e-7)
    assert fuse([first, second], [0, 0]).rank == 0
    with pytest.raises(ValueError):
        first.B[0, 0] = 7


def test_metric_response_balance_historical_coefficients_and_duplicate_certificates():
    cfg = BankConfig(rank=1)
    h = np.eye(2)
    weights = [.5, .5]
    # Two blocks of response A together contribute C_A, not twice its weight.
    records = [
        MetricMemory("A", [[2., 0]], [.25], [1, 0, 0, 0], "A:0"),
        MetricMemory("A", [[0., 2]], [.75], [1, 0, 0, 0], "A:1"),
        MetricMemory("B", [[1., 0]], [1.], [2, 0, 0, 0], "B:0"),
    ]
    S = .5 * np.eye(2)
    M = (np.diag([1., 3.]) + 4 * np.diag([1., 0.])) / 5.
    C = S + M
    expected = C / (np.trace(C) / 2) + cfg.epsilon * np.eye(2)
    metric = build_metric(h, weights, records, 0, cfg)
    np.testing.assert_allclose(metric.dense(), expected, atol=2e-14)
    np.testing.assert_allclose(build_metric(h, weights, records + records, 0, cfg).dense(), expected)
    # Unused owner has zero historical coefficients, hence no memory term.
    np.testing.assert_allclose(build_metric(h, weights, records, 1, cfg).dense(), 1.001 * np.eye(2))
    # Scaling all source/within-response weights leaves the normalized metric.
    np.testing.assert_allclose(build_metric(h, [5, 5], records, 0, cfg).dense(), expected)
    with pytest.raises(ValueError, match="conflicting"):
        build_metric(h, weights, records + [replace(records[0], features=[[9, 0]])], 0, cfg)


@pytest.mark.parametrize("d,n", [(7, 3), (3, 7)])
def test_matrix_free_powers_agree_with_dense_eigendecomposition(d, n):
    rng = np.random.default_rng(20)
    metric = NormalizedMetric(rng.normal(size=(d, n)))
    values, U = np.linalg.eigh(metric.dense())
    X = rng.normal(size=(4, d))
    for power in [.5, -.5, 1.]:
        expected = X @ (U * values ** power) @ U.T
        np.testing.assert_allclose(metric.right_power(X, power), expected, atol=2e-10)
    assert np.trace(metric.dense()) == pytest.approx(d * 1.001)


def test_proposal_matches_full_svd_and_checks_rounded_factors():
    rng = np.random.default_rng(11)
    d, k = 6, 2
    current = Factors(rng.normal(size=(d, k)), rng.normal(size=(k, d)))
    G = GradientFactors(rng.normal(size=(d, 3)), rng.normal(size=(3, d)))
    metric = NormalizedMetric(rng.normal(size=(d, 3)))
    cfg = BankConfig(rank=k)
    eta = .07
    proposal, receipt = spectral_proposal(current, G, metric, eta, cfg)
    values, U = np.linalg.eigh(metric.dense())
    sqrtQ = (U * np.sqrt(values)) @ U.T
    invsqrtQ = (U / np.sqrt(values)) @ U.T
    Z = current.dense() @ sqrtQ - eta * G.dense() @ invsqrtQ
    zu, zs, zv = np.linalg.svd(Z)
    expected = (zu[:, :k] * zs[:k]) @ zv[:k] @ invsqrtQ
    np.testing.assert_allclose(proposal.dense(), expected, rtol=3e-6, atol=3e-6)
    assert proposal.rank == k
    delta = proposal.dense() - current.dense()
    assert receipt["gradient_inner_step"] == pytest.approx(np.sum(G.dense() * delta), abs=1e-9)
    assert receipt["metric_step_squared"] == pytest.approx(np.sum((delta @ sqrtQ) ** 2), rel=2e-9)
    assert receipt["tail_squared"] == pytest.approx(np.sum(zs[k:] ** 2), rel=1e-9)
    assert receipt["svd_valid"] and receipt["quadratic_valid"]


def test_zero_factors_escape_and_only_one_unused_owner_is_tried():
    bank = CorrectionBank(4, BankConfig(rank=1))
    objective = Quadratic(np.diag([1., 0, 0, 0]), factored=True)
    result = bank.write(objective)
    assert result["committed"] and result["owner"] == 0
    assert len(result["trials"]) == 1
    assert result["accepted_steps"] == 2
    assert bank.allocated == [True, False, False, False]
    assert bank.versions == [1, 0, 0, 0]
    assert objective.loss(bank.modules[0]) < -.49
    attempts = result["trials"][0]["attempts"]
    assert attempts[0]["eta"] == 1.
    assert attempts[1]["eta"] == 2 * attempts[0]["eta"]
    assert bank.last_eta[0] == attempts[-1]["eta"]


def test_backtracking_recomputes_proposal_and_keeps_metric_fixed(monkeypatch):
    bank = CorrectionBank(3, BankConfig(rank=1))
    bank.modules[0] = factors(np.diag([1., 0., 0.]), 1)
    bank.allocated[0] = True
    bank.versions[0] = 1
    bank.last_eta[0] = 1.
    objective = Quadratic(np.diag([0., 1., 0.]), curvature=10.)
    original = core.spectral_proposal
    calls = []

    def spy(current, gradient, metric, eta, cfg):
        candidate, receipt = original(current, gradient, metric, eta, cfg)
        calls.append((current.dense(), id(metric), eta, candidate.rank))
        return candidate, receipt

    monkeypatch.setattr(core, "spectral_proposal", spy)
    result = bank.write(objective)
    occupied = result["trials"][0]
    assert occupied["owner"] == 0
    assert occupied["attempts"][0]["eta"] == 2.
    attempts = occupied["attempts"]
    assert any("source_armijo_failure" in row["reasons"] for row in attempts)
    first_step = [row for row in attempts if row["step"] == 1]
    assert len(first_step) >= 2
    for a, b in zip(first_step, first_step[1:]):
        assert b["eta"] == a["eta"] / 2
    occupied_calls = calls[:len(attempts)]
    assert len({row[1] for row in occupied_calls}) == 1
    for current, _, _, rank in occupied_calls[:len(first_step)]:
        np.testing.assert_array_equal(current, np.diag([1., 0., 0.]))
        assert rank <= 1
    # Interpolation between old and rotated rank-one operators would be rank two.
    assert np.linalg.matrix_rank(.5 * np.diag([1., 1., 0.])) == 2


def test_reject_all_preserves_bank_and_does_not_allocate_unused_slot():
    bank = CorrectionBank(3, BankConfig(rank=1))
    objective = Quadratic(np.diag([1., 0, 0]))
    observed = []

    def reject(modules):
        assert bank.allocated == [False] * 4
        assert bank.versions == [0] * 4
        observed.append(modules[0].dense())
        return False, {"reasons": ["native_prefix_protection_failure"]}

    result = bank.write(objective, protection_callback=reject)
    assert not result["committed"] and result["owner"] is None
    assert bank.allocated == [False] * 4
    assert bank.last_eta == [None] * 4
    assert len(result["trials"][0]["attempts"]) == 8
    assert len(observed) == 8
    assert all("native_prefix_protection_failure" in row["reasons"]
               for row in result["trials"][0]["attempts"])


def test_second_step_budget_failure_keeps_first_accepted_step():
    bank = CorrectionBank(3, BankConfig(rank=1))
    objective = Quadratic(np.diag([1., 0, 0]))
    checks = []

    def allow_first_only(modules):
        checks.append(modules[0].dense())
        return (len(checks) == 1), {"reasons": ["kl_protection_failure"]}

    result = bank.write(objective, protection_callback=allow_first_only)
    assert result["committed"] and result["accepted_steps"] == 1
    assert len(result["trials"][0]["attempts"]) == 9
    np.testing.assert_array_equal(bank.modules[0].dense(), checks[0])
    assert bank.last_eta[0] == 1.


def test_independent_trials_commit_only_winner_and_rank_by_final_loss(monkeypatch):
    bank = CorrectionBank(3, BankConfig(rank=1))
    bank.modules[:2] = [factors(np.diag([.1, 0, 0]), 1), factors(np.diag([.2, 0, 0]), 1)]
    bank.allocated[:2] = [True, True]
    bank.versions[:2] = [3, 8]
    originals = [x.dense() for x in bank.modules]

    def fake_trial(owner, objective, metric, callback):
        for module, original in zip(bank.modules, originals):
            np.testing.assert_array_equal(module.dense(), original)
        # Owner one has the largest improvement; owner zero has the best final loss.
        initial = [.25, 10., 8.][owner]
        final = [.2, .3, .4][owner]
        return {"owner": owner, "was_occupied": bank.allocated[owner],
                "initial_loss": initial, "final_loss": final, "source_fit_gain": initial-final,
                "accepted_steps": 1, "last_accepted_eta": .125, "changed": True,
                "attempts": [], "metric": metric.metadata(),
                "costs": {k: 0. for k in ("trial_s", "source_loss_s", "source_gradient_s",
                                          "spectral_s", "retention_s")},
                "factors": factors(np.diag([.7 + owner, 0, 0]), 1)}

    monkeypatch.setattr(bank, "_trial", fake_trial)
    result = bank.write(Quadratic(np.eye(3)))
    assert result["owner"] == 0
    assert bank.versions == [4, 8, 0, 0]
    assert bank.allocated == [True, True, False, False]
    assert bank.last_eta == [.125, None, None, None]
    np.testing.assert_array_equal(bank.modules[1].dense(), originals[1])
    assert [trial["owner"] for trial in result["trials"]] == [0, 1, 2]


def test_tie_prefers_occupied_then_lower_index_and_zero_write_no_allocation():
    bank = CorrectionBank(3, BankConfig(rank=1))
    # Every slot is zero, but slot one represents existing allocated experience.
    bank.allocated[1] = True
    bank.versions[1] = 1
    result = bank.write(Quadratic(np.zeros((3, 3))))
    assert not result["committed"]
    assert result["selected_owner"] == 1
    assert bank.allocated == [False, True, False, False]
    assert all(row["accepted_steps"] == 0 for row in result["trials"])
    bank.allocated[0] = True
    bank.versions[0] = 1
    assert bank.write(Quadratic(np.zeros((3, 3))))["selected_owner"] == 0


def test_bad_svd_or_quadratic_certificate_rejects_even_if_source_improves(monkeypatch):
    original = core.spectral_proposal

    def corrupt(current, gradient, metric, eta, cfg):
        candidate, receipt = original(current, gradient, metric, eta, cfg)
        receipt["svd_valid"] = False
        receipt["quadratic_valid"] = False
        return candidate, receipt

    monkeypatch.setattr(core, "spectral_proposal", corrupt)
    bank = CorrectionBank(3, BankConfig(rank=1))
    result = bank.write(Quadratic(np.diag([1., 0, 0])))
    assert not result["committed"]
    assert all("svd_numerical_failure" in attempt["reasons"] and
               "quadratic_proposal_failure" in attempt["reasons"]
               for attempt in result["trials"][0]["attempts"])


def test_roundtrip_restores_versions_eta_and_immutable_factors():
    bank = CorrectionBank(4, BankConfig(rank=2))
    bank.write(Quadratic(np.diag([1., .5, 0, 0])))
    snapshot = bank.state_dict()
    restored = CorrectionBank.from_state_dict(snapshot)
    assert restored.allocated == bank.allocated
    assert restored.versions == bank.versions
    assert restored.last_eta == bank.last_eta
    np.testing.assert_array_equal(restored.modules[0].B, bank.modules[0].B)
    snapshot["modules"][0]["B"][:] = 100
    assert np.max(np.abs(restored.modules[0].B)) < 2
    bad = bank.state_dict()
    bad["versions"][1] = 1
    with pytest.raises(ValueError, match="unused"):
        CorrectionBank.from_state_dict(bad)


def test_zero_feature_feedback_is_explicit_skip():
    objective = Quadratic(np.eye(3))
    objective.features[:] = 0
    bank = CorrectionBank(3)
    result = bank.write(objective)
    assert result["skip_reason"] == "degenerate_zero_feature_feedback"
    assert not result["committed"] and result["trials"] == []


def test_difference_norm_does_not_turn_equal_nonzero_factors_into_write():
    rng = np.random.default_rng(7)
    W = Factors(rng.normal(size=(100, 32)), rng.normal(size=(32, 100)))
    assert difference_norm(W, W) < 1e-11


def test_objective_mutation_is_not_silently_accepted():
    objective = Quadratic(np.eye(3))
    original = objective.loss_gradient
    objective.loss_gradient = lambda w: (original(w)[0] + 1., original(w)[1])
    with pytest.raises(ValueError, match="objective changed"):
        CorrectionBank(3).write(objective)
