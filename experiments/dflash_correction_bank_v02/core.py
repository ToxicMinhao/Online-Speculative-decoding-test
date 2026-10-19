"""Rank-constrained v0.2 writes; no factor SGD and no serving-time dense matrix.

Master factors are FP32. Covariance geometry, deterministic reduced QR/``gesvd``
and all acceptance arithmetic are FP64. This module does not interpret memory
as a validation set: the caller supplies immutable historical coefficients and
the actual KL/native-prefix protection callback. Only the winning owner commits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy import linalg


def _array(value, dtype=np.float64):
    result = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite numerical input")
    return result


@dataclass(frozen=True)
class Factors:
    """Immutable FP32 factors of W = B A; a zero module has zero columns."""

    B: np.ndarray
    A: np.ndarray

    def __post_init__(self):
        B = np.array(self.B, dtype=np.float32, order="C", copy=True)
        A = np.array(self.A, dtype=np.float32, order="C", copy=True)
        if (B.ndim != 2 or A.ndim != 2 or B.shape[1] != A.shape[0]
                or B.shape[0] != A.shape[1] or not np.isfinite(B).all()
                or not np.isfinite(A).all()):
            raise ValueError("finite square-operator factors B[d,k], A[k,d] required")
        B.setflags(write=False)
        A.setflags(write=False)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "A", A)

    @classmethod
    def zero(cls, d: int) -> "Factors":
        return cls(np.empty((d, 0), np.float32), np.empty((0, d), np.float32))

    @property
    def d(self):
        return self.B.shape[0]

    @property
    def rank(self):
        """Stored factor width, an upper bound on the matrix's numerical rank."""
        return self.A.shape[0]

    def dense(self, dtype=np.float64):
        return self.B.astype(dtype) @ self.A.astype(dtype)

    def apply(self, features):
        """Return residuals Wh (rows), in the input's floating-point dtype."""
        h = np.asarray(features)
        dtype = h.dtype if np.issubdtype(h.dtype, np.floating) else np.float32
        return (h.astype(dtype) @ self.A.astype(dtype).T) @ self.B.astype(dtype).T

    def state_dict(self):
        return {"B": self.B.copy(), "A": self.A.copy()}


def fuse(modules: Sequence[Factors], coefficients) -> Factors:
    """Concatenate active factors; never recompress a two-module action to k."""
    alpha = _array(coefficients)
    if not modules or alpha.shape != (len(modules),):
        raise ValueError("one finite coefficient per module required")
    if len({m.d for m in modules}) != 1:
        raise ValueError("module dimensions differ")
    active = [(float(a), m) for a, m in zip(alpha, modules) if a != 0 and m.rank]
    if not active:
        return Factors.zero(modules[0].d)
    return Factors(np.concatenate([a * m.B for a, m in active], axis=1),
                   np.concatenate([m.A for _, m in active], axis=0))


@dataclass(frozen=True)
class GradientFactors:
    """An exact factorization G = left @ right, usually with one column/anchor."""

    left: np.ndarray
    right: np.ndarray

    def __post_init__(self):
        left, right = _array(self.left), _array(self.right)
        if (left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]
                or left.shape[0] != right.shape[1]):
            raise ValueError("gradient factors must represent a square matrix")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    def dense(self):
        return self.left @ self.right

    def inner(self, operator: Factors):
        if not operator.rank:
            return 0.0
        return float(np.sum((self.left.T @ operator.B.astype(np.float64))
                            * (self.right @ operator.A.astype(np.float64).T)))


@dataclass(frozen=True)
class MetricMemory:
    """One distinct retained block; response_id groups blocks before averaging.

    Repeated certificates for the same block must not increase its weight. The
    optional record_id (otherwise a content hash) identifies that block. All
    blocks from a response must have its same immutable historical coefficients.
    """

    response_id: str
    features: np.ndarray
    weights: np.ndarray
    coefficients: np.ndarray
    record_id: str | None = None


@dataclass(frozen=True)
class BankConfig:
    slots: int = 4
    rank: int = 32
    epsilon: float = 1e-3
    armijo: float = 1e-4
    max_steps: int = 2
    max_attempts: int = 8
    check_atol: float = 1e-10
    check_rtol: float = 1e-7
    svd_atol: float = 1e-10
    svd_rtol: float = 1e-8
    tie_atol: float = 1e-10
    tie_rtol: float = 1e-7
    nonzero_atol: float = 1e-12
    svd_driver: str = "gesvd"
    step_initialization: str = "one_if_none_else_twice_last_accepted_each_step"

    def __post_init__(self):
        if self.slots < 1 or self.rank < 1 or not 0 < self.epsilon:
            raise ValueError("positive dimensions and epsilon required")
        if not 0 < self.armijo < 1 or self.max_steps != 2 or self.max_attempts != 8:
            raise ValueError("v0.2 requires two steps/eight attempts and Armijo in (0,1)")
        if self.svd_driver != "gesvd":
            raise ValueError("pinned deterministic reduced SVD is scipy LAPACK gesvd")
        if self.step_initialization != "one_if_none_else_twice_last_accepted_each_step":
            raise ValueError("unknown step-size continuation rule")
        if any(not np.isfinite(x) or x < 0 for x in (
                self.check_atol, self.check_rtol, self.svd_atol, self.svd_rtol,
                self.tie_atol, self.tie_rtol, self.nonzero_atol)):
            raise ValueError("finite nonnegative tolerances required")

    def tolerance(self, *values):
        return self.check_atol + self.check_rtol * max(1.0, *(abs(float(x)) for x in values))


def _weighted_features(features, weights, d=None):
    h, w = _array(features), _array(weights)
    if h.ndim != 2 or w.shape != (h.shape[0],) or h.shape[0] == 0:
        raise ValueError("nonempty feature rows and one weight per row required")
    if d is not None and h.shape[1] != d:
        raise ValueError("feature dimension mismatch")
    if np.any(w < 0) or not w.sum() > 0:
        raise ValueError("nonnegative weights with positive sum required")
    w = w / w.sum()
    return h, w


class NormalizedMetric:
    """Q = epsilon I + C/(tr(C)/d), represented by its non-isotropic eigenspace.

    The eigendecomposition is exact in the span of weighted feature columns.
    When their count exceeds d the dense d-dimensional route is used. No
    covariance eigenvalue truncation or output-side Fisher is introduced.
    """

    def __init__(self, columns, epsilon=1e-3):
        columns = _array(columns)
        if columns.ndim != 2:
            raise ValueError("covariance columns must be a matrix")
        self.d = columns.shape[0]
        self.epsilon = float(epsilon)
        self.trace = float(np.sum(columns * columns))
        if self.trace == 0:
            raise ValueError("degenerate_zero_feature_feedback")
        self.scale = self.trace / self.d
        columns = columns / math.sqrt(self.scale)
        if columns.shape[1] < self.d:
            q, r = linalg.qr(columns, mode="economic", pivoting=False,
                             check_finite=False)
            values, vectors = linalg.eigh(r @ r.T, check_finite=False, driver="evd")
            self.U = q @ vectors
        else:
            values, self.U = linalg.eigh(columns @ columns.T, check_finite=False,
                                        driver="evd")
        negative_tol = 1e-10 + 1e-10 * max(1.0, float(np.max(np.abs(values))))
        if float(values.min(initial=0)) < -negative_tol:
            raise ValueError("covariance eigendecomposition is not positive semidefinite")
        self.values = np.maximum(values, 0.0)

    def right_power(self, rows, power):
        """rows @ Q**power without materializing Q's isotropic complement."""
        rows = _array(rows)
        base = self.epsilon ** power
        shift = (self.epsilon + self.values) ** power - base
        return base * rows + ((rows @ self.U) * shift) @ self.U.T

    def dense(self):
        return self.epsilon * np.eye(self.d) + (self.U * self.values) @ self.U.T

    def metadata(self):
        return {"trace_C": self.trace, "normalization_scale": self.scale,
                "epsilon": self.epsilon, "eigenspace_columns": self.U.shape[1],
                "Q_trace": float(self.values.sum() + self.epsilon * self.d)}


def build_metric(features, weights, memory: Iterable[MetricMemory], owner: int,
                 config: BankConfig) -> NormalizedMetric:
    """Source covariance plus coefficient-squared, response-balanced memory."""
    h, w = _weighted_features(features, weights)
    d = h.shape[1]
    columns = [h.T * np.sqrt(w)]
    responses: dict[str, dict[str, Any]] = {}
    seen = {}
    for record in memory:
        mh, mw = _weighted_features(record.features, record.weights, d)
        alpha = _array(record.coefficients)
        if alpha.shape != (config.slots,):
            raise ValueError("memory coefficients do not match bank slots")
        # Keep original within-response relative weights before normalizing all
        # that response's distinct blocks together below.
        raw_weights = _array(record.weights)
        digest = hashlib.sha256(mh.tobytes() + raw_weights.tobytes() + alpha.tobytes()).hexdigest()
        key = (str(record.response_id), record.record_id or digest)
        if key in seen:
            if seen[key] != digest:
                raise ValueError("same memory record identity has conflicting contents")
            continue
        seen[key] = digest
        response = responses.setdefault(str(record.response_id), {
            "alpha": alpha.copy(), "features": [], "weights": []})
        if not np.array_equal(response["alpha"], alpha):
            raise ValueError("one response has inconsistent immutable action coefficients")
        response["features"].append(mh)
        response["weights"].append(raw_weights)
    total = sum(float(row["alpha"][owner]) ** 2 for row in responses.values())
    if total:
        for row in responses.values():
            coefficient = float(row["alpha"][owner]) ** 2
            if not coefficient:
                continue
            mh = np.concatenate(row["features"], axis=0)
            mw = np.concatenate(row["weights"])
            mw = mw / mw.sum() * (coefficient / total)
            columns.append(mh.T * np.sqrt(mw))
    return NormalizedMetric(np.concatenate(columns, axis=1), config.epsilon)


def _as_gradient(gradient, d):
    if isinstance(gradient, GradientFactors):
        if gradient.left.shape[0] != d:
            raise ValueError("gradient dimension mismatch")
        return gradient
    G = _array(gradient)
    if G.shape != (d, d):
        raise ValueError("dense gradient dimension mismatch")
    return GradientFactors(G, np.eye(d))


def _factor_norm_squared(left, right):
    if not left.shape[1]:
        return 0.0
    # A Gram trace subtracts nearly equal large terms for W_new - W_old.
    # Reducing first avoids spurious nonzero writes from that cancellation.
    _, reduced = linalg.qr(left, mode="economic", pivoting=False, check_finite=False)
    product = reduced @ right
    return float(np.sum(product * product))


def difference_factors(new: Factors, old: Factors):
    return (np.concatenate([new.B.astype(np.float64), -old.B.astype(np.float64)], axis=1),
            np.concatenate([new.A.astype(np.float64), old.A.astype(np.float64)], axis=0))


def difference_norm(new: Factors, old: Factors):
    left, right = difference_factors(new, old)
    return math.sqrt(_factor_norm_squared(left, right))


def spectral_proposal(current: Factors, gradient, metric: NormalizedMetric,
                      eta: float, config: BankConfig):
    """Recompute equation (5), including a fresh reduced SVD, for this eta.

    The two thin QR factorizations span the *complete* left/right operand
    spaces. Truncating their small product therefore implements the full
    spectral truncation (up to declared numerical arithmetic), without a
    randomized approximation or a d-by-d serving operator.
    """
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("finite positive eta required")
    gradient = _as_gradient(gradient, current.d)
    start = time.perf_counter()
    left = np.concatenate([current.B.astype(np.float64), -eta * gradient.left], axis=1)
    right = np.concatenate([metric.right_power(current.A, .5),
                            metric.right_power(gradient.right, -.5)], axis=0)
    if left.shape[1] == 0:
        return Factors.zero(current.d), {"svd_valid": True, "spectral_s": 0.,
                                       "singular_values": [], "tail_squared": 0., "rank": 0,
                                       "quadratic_valid": True, "step_norm": 0.,
                                       "gradient_inner_step": 0., "metric_step_squared": 0.}
    ql, rl = linalg.qr(left, mode="economic", pivoting=False, check_finite=False)
    qr, rr = linalg.qr(right.T, mode="economic", pivoting=False, check_finite=False)
    reduced = rl @ rr.T
    u, values, vt = linalg.svd(reduced, full_matrices=False, check_finite=False,
                              lapack_driver=config.svd_driver)
    reconstruction = (u * values) @ vt
    residual = float(linalg.norm(reduced - reconstruction, ord="fro"))
    operand_norm = float(linalg.norm(reduced, ord="fro"))
    svd_tol = config.svd_atol + config.svd_rtol * max(1., operand_norm)
    k = min(config.rank, len(values))
    roots = np.sqrt(values[:k])
    B = (ql @ u[:, :k]) * roots
    A = metric.right_power(roots[:, None] * (vt[:k] @ qr.T), -.5)
    candidate = Factors(B, A)
    delta_left, delta_right = difference_factors(candidate, current)
    metric_step_squared = _factor_norm_squared(delta_left, metric.right_power(delta_right, .5))
    inner = gradient.inner(candidate) - gradient.inner(current)
    bound = -metric_step_squared / (2 * eta)
    proposal_tol = config.tolerance(inner, bound)
    # The check uses the actual FP32 master factors, not unrounded SVD output.
    proposal_ok = bool(inner <= bound + proposal_tol)
    return candidate, {
        "spectral_s": time.perf_counter() - start,
        "svd_method": "FP64 scipy economic QR + reduced LAPACK gesvd",
        "svd_residual": residual, "svd_tolerance": svd_tol,
        "svd_valid": bool(residual <= svd_tol),
        "operand_frobenius": operand_norm,
        "singular_values": values.tolist(),
        "tail_squared": float(np.sum(values[k:] ** 2)),
        "rank": candidate.rank,
        "nonzero_singular_values": int(np.count_nonzero(values[:k] > 0)),
        "gradient_inner_step": inner,
        "metric_step_squared": metric_step_squared,
        "quadratic_bound": bound,
        "quadratic_value": inner - bound,
        "quadratic_tolerance": proposal_tol,
        "quadratic_valid": proposal_ok,
        "step_norm": difference_norm(candidate, current),
    }


def _protection_result(result):
    if isinstance(result, (bool, np.bool_)):
        return bool(result), {} if result else {"reasons": ["protection_rejected"]}
    if isinstance(result, tuple) and len(result) == 2:
        valid, detail = result
        if not isinstance(detail, dict):
            detail = {"reasons": detail}
        return bool(valid), detail
    if isinstance(result, dict) and ("valid" in result or "passed" in result):
        return bool(result.get("valid", result.get("passed"))), result
    raise TypeError("protection callback must return bool, (bool, details), or valid/passed dict")


class CorrectionBank:
    """Numerical bank state. Routing, records, and causal event order are external."""

    def __init__(self, d: int, config: BankConfig | None = None):
        self.d = int(d)
        if self.d <= 0:
            raise ValueError("positive hidden dimension required")
        self.config = config or BankConfig()
        self.modules = [Factors.zero(self.d) for _ in range(self.config.slots)]
        self.allocated = [False] * self.config.slots
        self.versions = [0] * self.config.slots
        self.last_eta = [None] * self.config.slots

    def state_dict(self):
        return {"schema": "dflash_correction_bank_v02.core.v1", "d": self.d,
                "config": asdict(self.config),
                "modules": [module.state_dict() for module in self.modules],
                "allocated": list(self.allocated), "versions": list(self.versions),
                "last_eta": list(self.last_eta)}

    @classmethod
    def from_state_dict(cls, state):
        if state.get("schema") != "dflash_correction_bank_v02.core.v1":
            raise ValueError("unsupported numerical bank schema")
        bank = cls(state["d"], BankConfig(**state["config"]))
        if any(len(state[key]) != bank.config.slots for key in
               ("modules", "allocated", "versions", "last_eta")):
            raise ValueError("bank state slot count mismatch")
        bank.modules = [Factors(**entry) for entry in state["modules"]]
        bank.allocated = [bool(x) for x in state["allocated"]]
        bank.versions = [int(x) for x in state["versions"]]
        bank.last_eta = [None if x is None else float(x) for x in state["last_eta"]]
        for module, allocated, version, eta in zip(bank.modules, bank.allocated,
                                                    bank.versions, bank.last_eta):
            if module.d != bank.d or module.rank > bank.config.rank or version < 0:
                raise ValueError("invalid stored module dimensions/rank/version")
            if not allocated and (module.rank != 0 or version != 0 or eta is not None):
                raise ValueError("unused slots must remain zero and unversioned")
            if eta is not None and (not np.isfinite(eta) or eta <= 0):
                raise ValueError("invalid last accepted step size")
        return bank

    @staticmethod
    def _loss(objective, factors):
        value = (objective.loss(factors) if hasattr(objective, "loss")
                 else objective.loss_gradient(factors)[0])
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite source loss")
        return value

    def _trial(self, owner, objective, metric, protection_callback):
        cfg = self.config
        current = self.modules[owner]
        initial = current
        occupied = self.allocated[owner]
        last_eta = self.last_eta[owner] if occupied else None
        attempts = []
        accepted_steps = 0
        gradient_s = 0.
        retention_s = 0.
        source_start = time.perf_counter()
        initial_loss = self._loss(objective, current)
        source_loss_s = time.perf_counter() - source_start
        current_loss = initial_loss
        trial_start = time.perf_counter()
        for step in range(cfg.max_steps):
            gradient_start = time.perf_counter()
            measured_loss, gradient = objective.loss_gradient(current)
            gradient_s += time.perf_counter() - gradient_start
            measured_loss = float(measured_loss)
            if not np.isfinite(measured_loss):
                raise FloatingPointError("nonfinite source loss during gradient evaluation")
            if abs(measured_loss - current_loss) > cfg.tolerance(measured_loss, current_loss):
                raise ValueError("objective changed across owner-trial evaluations")
            gradient = _as_gradient(gradient, self.d)
            eta = 1. if last_eta is None else 2. * last_eta
            accepted_this_step = False
            for attempt in range(cfg.max_attempts):
                candidate, diagnostics = spectral_proposal(current, gradient, metric, eta, cfg)
                record = {"step": step + 1, "attempt": attempt + 1, "eta": eta,
                          "source_loss_before": current_loss, **diagnostics}
                reasons = []
                if not diagnostics["svd_valid"]:
                    reasons.append("svd_numerical_failure")
                if not diagnostics["quadratic_valid"]:
                    reasons.append("quadratic_proposal_failure")
                if diagnostics["step_norm"] <= cfg.nonzero_atol:
                    record.update(accepted=False, noop=True, reasons=["zero_proposal"])
                    attempts.append(record)
                    break
                value_start = time.perf_counter()
                candidate_loss = self._loss(objective, candidate)
                source_loss_s += time.perf_counter() - value_start
                rhs = current_loss + cfg.armijo * diagnostics["gradient_inner_step"]
                armijo_tol = cfg.tolerance(candidate_loss, rhs)
                if candidate_loss > rhs + armijo_tol:
                    reasons.append("source_armijo_failure")
                record.update(source_loss_after=candidate_loss, armijo_rhs=rhs,
                              armijo_tolerance=armijo_tol)
                if not reasons and protection_callback is not None:
                    tentative = list(self.modules)
                    tentative[owner] = candidate
                    retention_start = time.perf_counter()
                    protected, protection_detail = _protection_result(protection_callback(tentative))
                    elapsed = time.perf_counter() - retention_start
                    retention_s += elapsed
                    record.update(protection=protection_detail, retention_s=elapsed)
                    if not protected:
                        detailed = protection_detail.get("reasons", ["protection_rejected"])
                        reasons.extend([detailed] if isinstance(detailed, str) else list(detailed))
                record.update(accepted=not reasons, noop=False, reasons=reasons)
                attempts.append(record)
                if not reasons:
                    current, current_loss, last_eta = candidate, candidate_loss, eta
                    accepted_steps += 1
                    accepted_this_step = True
                    break
                eta /= 2.
            if not accepted_this_step:
                break
        return {
            "owner": owner, "was_occupied": occupied, "initial_loss": initial_loss,
            "final_loss": current_loss, "source_fit_gain": initial_loss - current_loss,
            "accepted_steps": accepted_steps, "last_accepted_eta": last_eta,
            "changed": difference_norm(current, initial) > cfg.nonzero_atol,
            "attempts": attempts, "metric": metric.metadata(),
            "costs": {"trial_s": time.perf_counter() - trial_start,
                      "source_loss_s": source_loss_s, "source_gradient_s": gradient_s,
                      "spectral_s": sum(r.get("spectral_s", 0.) for r in attempts),
                      "retention_s": retention_s},
            "factors": current,
        }

    def write(self, objective, memory: Iterable[MetricMemory] = (),
              protection_callback: Callable | None = None):
        """Trial independent owners; commit only the lowest final singleton loss.

        Caller must insert pre-update transfer certificates BEFORE this method.
        This method never inserts/evicts a memory or protection record.
        """
        start = time.perf_counter()
        features = getattr(objective, "features", getattr(objective, "h", None))
        weights = getattr(objective, "weights", getattr(objective, "omega", None))
        h, w = _weighted_features(features, weights, self.d)
        if not float(np.sum(w[:, None] * h * h)) > 0:
            return {"committed": False, "owner": None, "accepted_steps": 0,
                    "skip_reason": "degenerate_zero_feature_feedback", "trials": [],
                    "costs": {"write_s": time.perf_counter() - start}}
        memory = list(memory)
        owners = [i for i, occupied in enumerate(self.allocated) if occupied]
        unused = next((i for i, occupied in enumerate(self.allocated) if not occupied), None)
        if unused is not None:
            owners.append(unused)
        trials = []
        metric_s = 0.
        for owner in owners:
            metric_start = time.perf_counter()
            metric = build_metric(h, w, memory, owner, self.config)
            metric_s += time.perf_counter() - metric_start
            trials.append(self._trial(owner, objective, metric, protection_callback))
        winner = trials[0]
        for trial in trials[1:]:
            tolerance = self.config.tie_atol + self.config.tie_rtol * max(
                1., abs(trial["final_loss"]), abs(winner["final_loss"]))
            if trial["final_loss"] < winner["final_loss"] - tolerance:
                winner = trial
            elif abs(trial["final_loss"] - winner["final_loss"]) <= tolerance:
                priority = (not trial["was_occupied"], trial["owner"])
                old_priority = (not winner["was_occupied"], winner["owner"])
                if priority < old_priority:
                    winner = trial
        committed = bool(winner["changed"] and winner["accepted_steps"])
        owner = winner["owner"]
        if committed:
            self.modules[owner] = winner["factors"]
            self.allocated[owner] = True
            self.versions[owner] += 1
            self.last_eta[owner] = winner["last_accepted_eta"]
        for trial in trials:
            del trial["factors"]
        return {
            "committed": committed, "owner": owner if committed else None,
            "selected_owner": owner, "accepted_steps": winner["accepted_steps"] if committed else 0,
            "source_fit_gain": winner["source_fit_gain"], "trials": trials,
            "allocated": list(self.allocated), "versions": list(self.versions),
            "costs": {"write_s": time.perf_counter() - start, "metric_s": metric_s,
                      **{name: sum(t["costs"][name] for t in trials) for name in
                         ("trial_s", "source_loss_s", "source_gradient_s", "spectral_s", "retention_s")}},
        }
