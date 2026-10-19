"""Small finite-softmax checks of the mathematical contracts, not LLM results."""
from dataclasses import replace
import copy

import numpy as np
import pytest
from scipy.special import logsumexp, softmax

from .core import (BankConfig, CorrectionBank, FeedbackSummary, ProtectionRecord,
                   ReadResult, inner, covariance_square)
from .projections import build_projections, gaussian_orthonormal


class FiniteObjective:
    def __init__(self, P, phi, z0, teacher, weights):
        self.P, self.phi, self.z0, self.teacher, self.weights = P, phi, z0, teacher, weights
        self.M = np.einsum("a,vo,av,ai->oi", weights, P, teacher, phi)
        self.base = float(weights @ logsumexp(z0, axis=1))
        q = softmax(z0, axis=1)
        self.T = np.stack([P.T @ (qa[:, None] * P) - np.outer(qa @ P, qa @ P) for qa in q])
        self.S = (phi.T * weights) @ phi

    def loss_gradient(self, C):
        logits = self.z0 + (self.P @ C @ self.phi.T).T
        q = softmax(logits, axis=1)
        value = float(self.weights @ logsumexp(logits, axis=1) - self.base - inner(C, self.M))
        gradient = np.einsum("a,vo,av,ai->oi", self.weights, self.P, q, self.phi) - self.M
        return value, gradient

    def direct_kl(self, C):
        logits = self.z0 + (self.P @ C @ self.phi.T).T
        return float(np.sum(self.weights[:, None] * self.teacher *
                            (np.log(self.teacher) - logits + logsumexp(logits, axis=1)[:, None])))

    def summary(self, read, response_id):
        G = self.loss_gradient(np.zeros((self.P.shape[1], self.phi.shape[1])))[1]
        return FeedbackSummary(read.descriptor, G, self.S, self.weights, self.phi,
                               self.T, read.forecast, G - read.forecast, response_id)


def fixture(seed=2, rank=3):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(9, rank))
    phi = rng.normal(size=(7, rank))
    phi /= np.maximum(1, np.linalg.norm(phi, axis=1, keepdims=True))
    z0 = rng.normal(size=(7, 9))
    teacher = softmax(rng.normal(size=(7, 9)), axis=1)
    weights = rng.uniform(size=7)
    weights /= weights.sum()
    kappa = 2 * np.linalg.norm(P - P.mean(axis=0), axis=1).max()
    return FiniteObjective(P, phi, z0, teacher, weights), BankConfig(float(kappa), rank=rank)


def test_exact_teacher_moment_gradient_and_global_bound():
    objective, cfg = fixture()
    rng = np.random.default_rng(4)
    C = rng.normal(size=(3, 3)) * 0.04
    D = rng.normal(size=(3, 3)) * 0.3
    loss, grad = objective.loss_gradient(C)
    assert loss == pytest.approx(objective.direct_kl(C) - objective.direct_kl(np.zeros_like(C)), abs=1e-14)
    numeric = np.zeros_like(C)
    for index in np.ndindex(C.shape):
        delta = np.zeros_like(C)
        delta[index] = 1e-6
        numeric[index] = (objective.direct_kl(C + delta) - objective.direct_kl(C - delta)) / 2e-6
    np.testing.assert_allclose(grad, numeric, atol=2e-10)
    change = objective.direct_kl(C + D) - objective.direct_kl(C)
    assert change <= inner(grad, D) + cfg.gamma * covariance_square(D, objective.S) + 1e-12


def test_projection_and_exact_fusion(tmp_path):
    rng = np.random.default_rng(17)
    E = rng.normal(size=(13, 7))
    projections = build_projections(E, 8, rank=3, chunk_rows=4)
    _, _, vt = np.linalg.svd(E, full_matrices=False)
    np.testing.assert_allclose(projections.U @ projections.U.T, vt[:3].T @ vt[:3], atol=2e-14)
    np.testing.assert_allclose(projections.V_in.T @ projections.V_in, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(projections.V_in, gaussian_orthonormal(7, 3), atol=0)
    h = rng.normal(size=(5, 7))
    phi = projections.phi(h)
    assert np.linalg.norm(phi, axis=1).max() <= 1 + 1e-14
    matrices = rng.normal(size=(4, 3, 3))
    a = np.array([0., .7, .8, 0.])  # Deliberately sums to >1: additive, not convex fusion.
    compiled = np.einsum("k,koi->oi", a, matrices)
    explicit = sum(a[j] * (phi @ matrices[j].T @ projections.U.T @ E.T) for j in range(4))
    np.testing.assert_allclose((h + phi @ compiled.T @ projections.U.T) @ E.T,
                               h @ E.T + explicit, atol=1e-14)
    path = tmp_path / "fixed.npz"
    projections.save(path)
    assert projections.metadata() == projections.load(path).metadata()


def test_first_two_frozen_snapshot_and_final_no_write():
    objective, cfg = fixture()
    bank = CorrectionBank(cfg)
    descriptor = np.array([1., 0., 0.])
    first = bank.read(descriptor)
    assert not first.alpha.any()
    entry = bank.observe_and_write(objective.summary(first, "response1"), objective, first)
    assert entry["owner"] == 0
    assert len(entry["candidates"]) == 4 and len(entry["steps"]) == 2
    assert bank.allocated[0]
    assert entry["singleton_fit_gain"] > 0
    second = bank.read(descriptor)
    assert not second.alpha.any()
    assert second.diagnostics["epsilon"] == pytest.approx(1e-3 * np.linalg.norm(bank.feedback[0].G))
    restored = CorrectionBank.from_snapshot(bank.snapshot())
    np.testing.assert_array_equal(restored.matrices, bank.matrices)
    np.testing.assert_array_equal(restored.read(descriptor).forecast, second.forecast)
    old = bank.matrices.copy()
    last = bank.observe_and_write(objective.summary(second, "response2"), objective, second, final=True)
    assert last["steps"] == [] and last["owner"] is None
    np.testing.assert_array_equal(bank.matrices, old)
    with pytest.raises(ValueError, match="already observed"):
        bank.observe_and_write(objective.summary(second, "response2"), objective, second, final=True)
    with pytest.raises(ValueError):
        bank.feedback[0].G[0, 0] = 1


def test_served_protection_precedes_owner_trials_and_is_cumulative(monkeypatch):
    objective, cfg = fixture()
    bank = CorrectionBank(cfg)
    x = np.array([1., 0., 0.])
    first = bank.read(x)
    bank.observe_and_write(objective.summary(first, "first"), objective, first)
    # Explicitly serve the learned singleton to test pre-write insertion timing.
    served = ReadResult(np.array([1., 0., 0., 0.]), bank.matrices[0],
                        bank.read(x).forecast, x, {})
    pre_records = copy.deepcopy(bank.snapshot()["protection"])
    original_trial = bank._trial
    count = []
    def checked_trial(*args):
        assert bank.protection[-1].kind == "served_transfer"
        assert bank.protection[-1].response_id == "second"
        count.append(args[0])
        return original_trial(*args)
    monkeypatch.setattr(bank, "_trial", checked_trial)
    entry = bank.observe_and_write(objective.summary(served, "second"), objective, served)
    assert len(count) == 5
    assert entry["records_before_owner_trials"] == len(pre_records) + 1
    for old, current in zip(pre_records, bank.snapshot()["protection"]):
        for key in old:
            if isinstance(old[key], np.ndarray):
                np.testing.assert_array_equal(old[key], current[key])
            else:
                assert old[key] == current[key]
    # Check actual historical losses, independently of the stored quadratic gate.
    for record in bank.protection:
        composed = np.einsum("k,koi->oi", record.coefficients, bank.matrices)
        gain_now = objective.direct_kl(np.zeros_like(composed)) - objective.direct_kl(composed)
        assert gain_now >= (1 - cfg.rho) * record.gain - 1e-9


def test_conflicting_updates_keep_actual_historical_gain():
    first_objective, cfg = fixture(2)
    bank = CorrectionBank(cfg)
    x = np.array([1., 0., 0.])
    read = bank.read(x)
    bank.observe_and_write(first_objective.summary(read, "first"), first_objective, read)
    record = bank.protection[0]
    frozen_reference = record.reference.copy()
    gain = record.gain
    for i in range(2):
        objective, _ = fixture(10 + i)
        # Use the same frozen projection P so kappa remains a valid bound.
        objective = FiniteObjective(first_objective.P, objective.phi, objective.z0,
                                    objective.teacher, objective.weights)
        read = bank.read(x)
        bank.observe_and_write(objective.summary(read, f"later{i}"), objective, read)
        actual = np.einsum("k,koi->oi", record.coefficients, bank.matrices)
        actual_gain = first_objective.direct_kl(np.zeros_like(actual)) - first_objective.direct_kl(actual)
        assert actual_gain >= 0.75 * gain - 1e-9
        np.testing.assert_array_equal(record.reference, frozen_reference)
        assert record.gain == gain
        assert bank.check_constraints()["feasible"]


def test_directional_error_can_enable_pair_when_singletons_do_not():
    bank = CorrectionBank(BankConfig(kappa=1., rank=2))
    bank.matrices[0, 0, 0] = 1
    bank.matrices[1, 1, 1] = 1
    bank.allocated[:2] = True
    desired_gram = np.array([[1., -.8], [-.8, 1.]])
    errors = np.linalg.cholesky(desired_gram).T * np.sqrt(2)
    G = np.diag([-.45, -.45])
    for i in range(2):
        E = np.diag(errors[i])
        bank.feedback.append(FeedbackSummary(np.array([1., 0.]), G, np.eye(2) / 2,
            np.array([.5, .5]), np.eye(2), np.zeros((2, 2, 2)), G - E, E, str(i)))
    read = bank.read(np.array([1., 0.]))
    assert read.alpha[0] > 0 and read.alpha[1] > 0
    assert read.diagnostics["objective"] < -1e-3
    singles = [c for c in read.diagnostics["candidates"] if len(c["support"]) == 1]
    assert all(c["objective"] > -1e-8 for c in singles)
    assert np.linalg.norm(read.effective) <= 1 + 1e-7


def test_rank64_write_matrix_free_smoke():
    cfg = BankConfig(kappa=2., rank=64, solver_maxiter=100)
    bank = CorrectionBank(cfg)
    S = np.eye(64) / 64
    gradient = np.zeros((64, 64))
    gradient[0, 0] = -.01
    trial = bank._trial(0, 0., gradient, S)
    assert trial["accepted"] and trial["nonzero"]
    assert trial["majorizer_change"] < 0
    assert not trial["dense_hessian_constructed"]
    assert np.linalg.norm(trial["proposed"]) <= cfg.radius + 1e-8


def test_zero_supervision_preserves_state_and_logs_two_scheduled_noops():
    cfg = BankConfig(kappa=2., rank=3)
    bank = CorrectionBank(cfg)
    read = bank.read(np.array([1., 0., 0.]))
    zero = np.zeros((3, 3))
    summary = FeedbackSummary(read.descriptor, zero, zero, np.empty(0), np.empty((0, 3)),
                              np.empty((0, 3, 3)), read.forecast, zero - read.forecast,
                              "zero_exposure", {"zero_supervision": True})
    class EmptyObjective:
        def loss_gradient(self, C):
            return 0., np.zeros_like(C)
    entry = bank.observe_and_write(summary, EmptyObjective(), read)
    assert len(entry["steps"]) == 2
    assert all(step["accepted"] and not step["nonzero"] for step in entry["steps"])
    assert not bank.matrices.any() and not bank.allocated.any()
    assert not bank.protection
    assert len(entry["candidates"]) == 4


def test_failed_solver_trial_cannot_change_feasible_state(monkeypatch):
    import sys
    core = sys.modules[CorrectionBank.__module__]
    objective, cfg = fixture()
    bank = CorrectionBank(cfg)
    gradient = objective.loss_gradient(np.zeros((3, 3)))[1]
    class InvalidResult:
        x = np.ones(9) * 1e6
        success = True
        status = 1
        message = "Fake solver success with an infeasible proposal"
        niter = 1
        cg_niter = 1
        optimality = 0.
        constr_violation = 0.
    monkeypatch.setattr(core, "minimize", lambda *a, **kw: InvalidResult())
    trial = bank._trial(0, 0., gradient, objective.S)
    assert not trial["accepted"] and not trial["nonzero"]
    np.testing.assert_array_equal(trial["proposed"], np.zeros((3, 3)))
    assert not trial["feasibility"]["feasible"]
    assert not bank.matrices.any()


def test_nonfinite_solver_failures_are_logged_and_json_serializable(monkeypatch):
    import json
    import sys
    from .native_hooks import _jsonable
    core = sys.modules[CorrectionBank.__module__]
    objective, cfg = fixture()
    bank = CorrectionBank(cfg)
    descriptor = np.array([1., 0., 0.])
    first = bank.read(descriptor)
    bank.observe_and_write(objective.summary(first, "first"), objective, first)
    class InvalidResult:
        success = False
        status = 0
        message = "nonfinite simulated solver failure"
        nit = niter = 1
        optimality = float('nan')
        constr_violation = float('inf')
        def __init__(self, x): self.x = np.full_like(x, np.nan)
    monkeypatch.setattr(core, "minimize", lambda _f, x, **kwargs: InvalidResult(x))
    read = bank.read(descriptor)
    assert not read.alpha.any()
    assert sum(not c['valid'] for c in read.diagnostics['candidates']) == 10
    json.dumps(_jsonable(read.diagnostics), allow_nan=False)
    trial = bank._trial(0, 0., objective.G if hasattr(objective, 'G') else
                        objective.loss_gradient(np.zeros((3, 3)))[1], objective.S)
    assert not trial['accepted']
    json.dumps(_jsonable(trial), allow_nan=False)
    assert trial['raw_majorizer_change']['finite'] is False


def test_read_rejects_coefficients_outside_exact_box(monkeypatch):
    import sys
    core = sys.modules[CorrectionBank.__module__]
    objective, cfg = fixture()
    bank = CorrectionBank(cfg)
    x = np.array([1., 0., 0.])
    first = bank.read(x)
    bank.observe_and_write(objective.summary(first, 'first'), objective, first)
    class Result:
        success = True
        message = 'slightly negative coefficient'
        nit = 1
        def __init__(self, start): self.x = np.full_like(start, -1e-12)
    monkeypatch.setattr(core, 'minimize', lambda _f, start, **kwargs: Result(start))
    read = bank.read(x)
    assert not read.alpha.any()
    assert all(not c['valid'] for c in read.diagnostics['candidates'][1:])


def test_full_rank64_bank_write_read_and_snapshot_are_bounded(tmp_path, record_property):
    """All four default-size owner trials, with no dense parameter Hessian.

    This is a geometry/serialization check with an explicitly synthetic convex
    quadratic, not a model, efficacy result, or target-distribution experiment.
    """
    import json
    import resource
    import time
    from .artifacts import save_snapshot, load_snapshot
    cfg = BankConfig(kappa=2.)  # Unchanged default rank64, four owners and solver tolerances.
    bank = CorrectionBank(cfg)
    rng = np.random.default_rng(17)
    target = rng.normal(size=(64, 64))
    target *= .3 / np.linalg.norm(target)
    curvature = 1 / 64**2
    class Quadratic:
        def loss_gradient(self, C):
            return (curvature * (.5 * inner(C, C) - inner(C, target)),
                    curvature * (C - target))
    objective = Quadratic()
    descriptor = np.zeros(64)
    descriptor[0] = 1
    first = bank.read(descriptor)
    G = objective.loss_gradient(np.zeros((64, 64)))[1]
    summary = FeedbackSummary(descriptor, G, np.eye(64) / 64,
        np.full(64, 1 / 64), np.eye(64), np.broadcast_to(np.eye(64) / 64, (64, 64, 64)),
        first.forecast, G - first.forecast, 'synthetic_rank64')
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    write = bank.observe_and_write(summary, objective, first)
    second = bank.read(descriptor)
    reference = save_snapshot(bank.snapshot(), tmp_path / 'rank64_snapshot.json')
    restored = CorrectionBank.from_snapshot(load_snapshot(reference))
    elapsed = time.perf_counter() - started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    total_bytes = sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())
    assert len(write['candidates']) == 4 and len(write['steps']) == 2
    assert all(c['accepted'] and not c['dense_hessian_constructed'] for c in write['candidates'])
    assert not second.alpha.any()  # The specified second-turn read still uses frozen predictions.
    assert bank.check_constraints()['feasible']
    np.testing.assert_array_equal(restored.matrices, bank.matrices)
    np.testing.assert_array_equal(restored.feedback[0].T, bank.feedback[0].T)
    # The only potentially large retained statistic is the explicit anchor Fisher
    # array (64*64*64 values = 2MiB); no 4096-square Hessian belongs in the snapshot.
    assert total_bytes < 16 * 2**20
    measurement = dict(elapsed_s=elapsed, process_peak_rss_kib=peak_rss,
                       incremental_peak_rss_kib=max(0, peak_rss-before_rss),
                       serialized_bytes=total_bytes,
                       candidate_iterations=[c['iterations'] for c in write['candidates']],
                       second_step_iterations=write['steps'][1]['iterations'])
    record_property('rank64_geometry_measurement', json.dumps(measurement))
    print('rank64_geometry_measurement=' + json.dumps(measurement, sort_keys=True))
