"""Correction-bank v0.1, equations (6)--(16), in fixed float64 coordinates.

The write solver uses matrix-free Hessian products. It never materializes a
(rank**2, rank**2) Hessian. SciPy trust-constr is only a candidate generator:
independent feasibility and majorizer checks decide whether a step is committed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
import copy
import time
from typing import Any, Protocol

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize
from scipy.sparse.linalg import LinearOperator


class ResponseObjective(Protocol):
    def loss_gradient(self, correction: np.ndarray) -> tuple[float, np.ndarray]: ...


def _immutable(array: Any) -> np.ndarray:
    value = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(value)):
        raise ValueError("Nonfinite feedback/state")
    value.flags.writeable = False
    return value


def inner(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.einsum("ij,ij->", a, b))


def covariance_square(matrix: np.ndarray, covariance: np.ndarray) -> float:
    return inner(matrix @ covariance, matrix)


def _diagnostic_safe(value):
    """Failure diagnostics must survive strict JSON without accepting bad state.

    Finite arrays retain their binary format. A nonfinite scalar is represented
    explicitly, rather than silently changed to zero or lost during logging.
    """
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return {"nonfinite": repr(float(value)), "finite": False}
    if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
        return {"nonfinite_array": True, "shape": list(value.shape),
                "values": _diagnostic_safe(value.tolist())}
    if isinstance(value, dict):
        return {key: _diagnostic_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_diagnostic_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class BankConfig:
    kappa: float
    rank: int = 64
    slots: int = 4
    seed: int = 17
    ell: float = 1.0
    beta: float = 0.5
    rho: float = 0.25
    zeta: float = 0.05
    error_floor_multiplier: float = 1e-3
    damping_multiplier: float = 1e-2
    kernel_zero_threshold: float = 1e-15
    feasibility_atol: float = 1e-9
    feasibility_rtol: float = 1e-7
    majorizer_atol: float = 1e-10
    majorizer_rtol: float = 1e-8
    read_tie_atol: float = 1e-10
    read_tie_rtol: float = 1e-8
    owner_tie_atol: float = 1e-10
    solver_gtol: float = 1e-9
    solver_xtol: float = 1e-10
    solver_barrier_tol: float = 1e-10
    solver_maxiter: int = 200
    read_maxiter: int = 200

    def __post_init__(self):
        if not np.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.rank < 1 or self.slots != 4:
            raise ValueError("v0.1 requires four slots and positive rank")
        if self.ell <= 0 or not 0 <= self.rho < 1:
            raise ValueError("Invalid kernel/protection configuration")

    @property
    def radius(self) -> float:
        return 1.0 / self.kappa

    @property
    def gamma(self) -> float:
        return self.kappa**2 / 8

    @property
    def damping(self) -> float:
        return self.damping_multiplier * self.kappa**2 / (4 * self.rank)

    @property
    def c_plus(self) -> float:
        R = self.kappa * self.radius
        return float((np.expm1(R) - R) / R**2) if R else 0.5


@dataclass(frozen=True)
class FeedbackSummary:
    descriptor: np.ndarray
    G: np.ndarray
    S: np.ndarray
    omega: np.ndarray
    phi: np.ndarray
    T: np.ndarray
    forecast: np.ndarray
    error: np.ndarray
    response_id: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("descriptor", "G", "S", "omega", "phi", "T", "forecast", "error"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))
        ro, ri = self.G.shape
        n = len(self.omega)
        if self.S.shape != (ri, ri) or self.phi.shape != (n, ri) or self.T.shape != (n, ro, ro):
            raise ValueError("Inconsistent exact Fisher factor dimensions")
        if n == 0 and (np.any(self.G) or np.any(self.S)):
            raise ValueError("A zero-supervision sample must have zero gradient and covariance")
        if n and (np.min(self.omega) < 0 or not np.isclose(self.omega.sum(), 1, atol=1e-10)):
            raise ValueError("Response weights must be nonnegative and normalized")
        if n and np.max(np.linalg.norm(self.phi, axis=1)) > 1 + 1e-9:
            raise ValueError("Input map must satisfy the unit bound")
        if not np.allclose(self.S, (self.phi.T * self.omega) @ self.phi, atol=1e-10, rtol=1e-8):
            raise ValueError("S does not equal the sampled response covariance")
        if not np.allclose(self.error, self.G - self.forecast, atol=1e-12, rtol=1e-10):
            raise ValueError("Forecast error must use the saved pre-response forecast")
        if not np.isclose(np.linalg.norm(self.descriptor), 1, atol=1e-8):
            raise ValueError("Read descriptor must be unit normalized")

    def projected_fisher(self, matrices: np.ndarray) -> np.ndarray:
        # Shape: matrices [K,ro,ri], transformed [anchors,K,ro].
        transformed = np.einsum("koi,ai->ako", matrices, self.phi, optimize=True)
        return np.einsum("a,ako,aop,alp->kl", self.omega, transformed,
                         self.T, transformed, optimize=True)


@dataclass(frozen=True)
class ProtectionRecord:
    coefficients: np.ndarray
    reference: np.ndarray
    gradient: np.ndarray
    covariance: np.ndarray
    gain: float
    response_id: str
    kind: str

    def __post_init__(self):
        for name in ("coefficients", "reference", "gradient", "covariance"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        if not np.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("Only strictly positive gains are protected")
        if self.kind not in ("served_transfer", "singleton_fit"):
            raise ValueError("Unknown protection provenance")


@dataclass(frozen=True)
class ReadResult:
    alpha: np.ndarray
    effective: np.ndarray
    forecast: np.ndarray
    descriptor: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        for name in ("alpha", "effective", "forecast", "descriptor"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


class CorrectionBank:
    def __init__(self, config: BankConfig):
        self.config = config
        self.matrices = np.zeros((config.slots, config.rank, config.rank), dtype=np.float64)
        self.allocated = np.zeros(config.slots, dtype=bool)
        self.feedback: list[FeedbackSummary] = []
        self.protection: list[ProtectionRecord] = []
        self.write_log: list[dict] = []

    def snapshot(self) -> dict:
        """Serializable CPU arrays; use NPZ/pickle/torch.save, not lossy JSON floats."""
        return {"schema": "dflash-correction-bank-v0.1", "config": asdict(self.config),
                "matrices": self.matrices.copy(), "allocated": self.allocated.copy(),
                "feedback": [{name: copy.deepcopy(getattr(item, name))
                              for name in item.__dataclass_fields__} for item in self.feedback],
                "protection": [{name: copy.deepcopy(getattr(item, name))
                                for name in item.__dataclass_fields__} for item in self.protection],
                "write_log": copy.deepcopy(self.write_log)}

    @classmethod
    def from_snapshot(cls, state: dict) -> "CorrectionBank":
        if state.get("schema") != "dflash-correction-bank-v0.1":
            raise ValueError("Unknown bank snapshot schema")
        bank = cls(BankConfig(**state["config"]))
        bank.matrices[:] = state["matrices"]
        bank.allocated[:] = state["allocated"]
        bank.feedback = [FeedbackSummary(**item) for item in state["feedback"]]
        bank.protection = [ProtectionRecord(**item) for item in state["protection"]]
        bank.write_log = copy.deepcopy(state["write_log"])
        check = bank.check_constraints()
        if not check["feasible"]:
            raise ValueError("Snapshot violates immutable protection/radius constraints")
        return bank

    def _read_statistics(self, descriptor: np.ndarray) -> tuple:
        cfg = self.config
        distances = np.array([np.sum((descriptor - s.descriptor)**2) for s in self.feedback])
        kernels = np.exp(-distances / (2 * cfg.ell**2))
        Z = float(kernels.sum())
        if Z <= cfg.kernel_zero_threshold:
            return Z, kernels, None, None, None, None
        weights = kernels / Z
        forecast = sum((k * s.G for k, s in zip(kernels, self.feedback)),
                       start=np.zeros_like(self.matrices[0])) / (1 + Z)
        floor = cfg.error_floor_multiplier * max(float(np.linalg.norm(s.G)) for s in self.feedback)
        errors = np.array([[inner(s.error, C) for C in self.matrices] for s in self.feedback])
        gram = np.einsum("koi,loi->kl", self.matrices, self.matrices)
        error_gram = errors.T @ (weights[:, None] * errors) + floor**2 * gram
        fisher = sum((w * s.projected_fisher(self.matrices) for w, s in zip(weights, self.feedback)),
                     start=np.zeros((cfg.slots, cfg.slots)))
        return Z, weights, forecast, floor, (error_gram + error_gram.T) / 2, (fisher + fisher.T) / 2

    def read(self, descriptor: np.ndarray) -> ReadResult:
        started = time.perf_counter()
        descriptor = np.array(descriptor, dtype=np.float64, copy=True)
        if not np.isclose(np.linalg.norm(descriptor), 1, atol=1e-8):
            raise ValueError("Read requires the pinned unit-normalized descriptor")
        cfg = self.config
        Z, weights, forecast, floor, error_gram, fisher = self._read_statistics(descriptor)
        if forecast is None:
            return ReadResult(np.zeros(cfg.slots), np.zeros_like(self.matrices[0]),
                              np.zeros_like(self.matrices[0]), descriptor,
                              {"reason": "no_prior_feedback_or_kernel_support", "history_count": len(self.feedback),
                               "kernel_sum": Z, "objective": 0.0, "epsilon": None,
                               "solver_s": time.perf_counter() - started, "candidates": []})
        linear = np.array([inner(forecast, C) for C in self.matrices])
        norm_gram = np.einsum("koi,loi->kl", self.matrices, self.matrices)
        candidates = [{"alpha": np.zeros(cfg.slots), "objective": 0.0, "norm": 0.0,
                       "support": [], "status": "frozen", "valid": True}]
        for support in [(j,) for j in range(cfg.slots)] + list(combinations(range(cfg.slots), 2)):
            idx = np.array(support)
            b, Q, H, N = (linear[idx], error_gram[np.ix_(idx, idx)],
                           fisher[np.ix_(idx, idx)], norm_gram[np.ix_(idx, idx)])
            def objective(a):
                return float(b @ a + cfg.beta * np.sqrt(max(0, a @ Q @ a)) + cfg.c_plus * (a @ H @ a))
            def jac(a):
                norm = np.sqrt(max(0, a @ Q @ a))
                return b + (cfg.beta * (Q @ a) / norm if norm > 1e-30 else 0) + 2 * cfg.c_plus * H @ a
            def constraint(a):
                return 1 - float(a @ N @ a) / cfg.radius**2
            def constraint_jac(a):
                return -2 * (N @ a) / cfg.radius**2
            # Positive interior start avoids an arbitrary subgradient at the norm's apex.
            start = np.full(len(idx), 0.1)
            try:
                result = minimize(objective, start, jac=jac, method="SLSQP",
                                  bounds=[(0., 1.)] * len(idx),
                                  constraints={"type": "ineq", "fun": constraint, "jac": constraint_jac},
                                  options={"maxiter": cfg.read_maxiter, "ftol": 1e-12})
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
                candidates.append({"alpha": np.zeros(cfg.slots), "objective": float("inf"),
                                   "norm": float("inf"), "support": list(support),
                                   "status": repr(exc), "success": False, "valid": False})
                continue
            alpha = np.zeros(cfg.slots)
            alpha[idx] = result.x
            norm = float(np.sqrt(max(0, alpha @ norm_gram @ alpha)))
            valid = (np.all(np.isfinite(alpha)) and np.min(alpha) >= 0
                     and np.max(alpha) <= 1
                     and norm <= cfg.radius + cfg.feasibility_atol + cfg.feasibility_rtol * cfg.radius)
            # Do not silently clip a failed solution into a different one.
            score = objective(result.x) if valid else float("inf")
            candidates.append({"alpha": alpha, "objective": score, "norm": norm,
                               "support": list(support), "status": str(result.message),
                               "success": bool(result.success), "iterations": int(result.nit), "valid": bool(valid)})
        best_score = min(c["objective"] for c in candidates)
        tie = cfg.read_tie_atol + cfg.read_tie_rtol * abs(best_score)
        tied = [c for c in candidates if c["valid"] and c["objective"] <= best_score + tie]
        chosen = min(tied, key=lambda c: (bool(np.any(c["alpha"])), c["norm"],
                                         int(np.count_nonzero(c["alpha"])), c["support"]))
        alpha = chosen["alpha"]
        effective = np.einsum("k,koi->oi", alpha, self.matrices)
        diagnostics = {"reason": "selected" if np.any(alpha) else "frozen_objective_preferred",
                       "history_count": len(self.feedback), "kernel_sum": Z, "kernel_weights": weights,
                       "epsilon": floor, "objective": chosen["objective"],
                       "predicted_linear_change": float(linear @ alpha),
                       "prediction_error_penalty": float(cfg.beta * np.sqrt(max(0, alpha @ error_gram @ alpha))),
                       "curvature_penalty": float(cfg.c_plus * alpha @ fisher @ alpha),
                       "error_gram": error_gram, "projected_fisher": fisher,
                       "candidates": candidates, "solver_s": time.perf_counter() - started}
        return ReadResult(alpha, effective, forecast, descriptor, _diagnostic_safe(diagnostics))

    def check_constraints(self, matrices: np.ndarray | None = None) -> dict:
        matrices = self.matrices if matrices is None else matrices
        cfg = self.config
        norms = np.linalg.norm(matrices, axis=(1, 2))
        radius_tol = cfg.feasibility_atol + cfg.feasibility_rtol * cfg.radius
        valid = bool(np.all(np.isfinite(matrices)) and np.all(norms <= cfg.radius + radius_tol))
        details = []
        for record in self.protection:
            composition = np.einsum("k,koi->oi", record.coefficients, matrices)
            D = composition - record.reference
            Q = inner(record.gradient, D) + cfg.gamma * covariance_square(D, record.covariance)
            budget = cfg.rho * record.gain
            tol = cfg.feasibility_atol + cfg.feasibility_rtol * budget
            norm = float(np.linalg.norm(composition))
            okay = bool(Q <= budget + tol and norm <= cfg.radius + radius_tol)
            valid &= okay
            details.append({"response_id": record.response_id, "kind": record.kind,
                            "bound": Q, "budget": budget, "slack": budget - Q,
                            "tolerance": tol, "composition_norm": norm, "feasible": okay})
        return {"feasible": valid, "module_norms": norms, "radius": cfg.radius, "records": details}

    def _trial(self, owner: int, loss: float, gradient: np.ndarray, S: np.ndarray) -> dict:
        cfg, old = self.config, self.matrices[owner].copy()
        if not np.any(gradient):
            # The PSD curvature plus positive damping is minimized exactly at
            # zero. In particular, an author sample with no eligible supervised
            # positions must not create a barrier-induced artificial update.
            feasibility = self.check_constraints()
            return {"owner": owner, "proposed": old, "base_loss": float(loss),
                    "majorizer_change": 0., "raw_majorizer_change": 0.,
                    "majorizer_tolerance": cfg.majorizer_atol, "score": float(loss),
                    "accepted": feasibility["feasible"], "nonzero": False,
                    "solver_success": True, "solver_status": 0,
                    "solver_message": "Exact zero-gradient quadratic minimizer; no solver needed",
                    "iterations": 0, "cg_iterations": 0, "optimality": 0.,
                    "constraint_violation": 0., "feasibility": feasibility,
                    "solver_s": 0., "constraint_count": 0, "dense_hessian_constructed": False}
        rank, radius, gamma = cfg.rank, cfg.radius, cfg.gamma
        size = rank * rank
        # y = D/r: positive scalar rescaling improves conditioning without changing
        # the objective, constraint set, majorizer, or chosen mathematical policy.
        right = radius**2 * (2 * gamma * S + cfg.damping * np.eye(rank))
        scale = max(radius * float(np.linalg.norm(gradient)), float(np.linalg.norm(right)), 1e-12)
        def fun(y):
            D = radius * y.reshape(rank, rank)
            return (inner(gradient, D) + gamma * covariance_square(D, S)
                    + cfg.damping / 2 * inner(D, D)) / scale
        def jac(y):
            Y = y.reshape(rank, rank)
            return ((radius * gradient + Y @ right) / scale).ravel()
        H = LinearOperator((size, size), matvec=lambda v: (v.reshape(rank, rank) @ right / scale).ravel(), dtype=np.float64)
        # Constraint specifications are quadratic or radius squared, each <= 0.
        specs = [("radius", old, 1.0, None, None, 1.0)]
        for record in self.protection:
            a = float(record.coefficients[owner])
            if a == 0:
                continue  # An independent constraint remains unchanged, checked again below.
            composition = np.einsum("k,koi->oi", record.coefficients, self.matrices)
            D0 = composition - record.reference
            cscale = max(record.gain, radius * np.linalg.norm(record.gradient),
                         gamma * radius**2 * np.linalg.norm(record.covariance), 1e-12)
            specs.append(("protection", D0, a, record, None, cscale))
            specs.append(("radius", composition, a, None, None, 1.0))
        def constraints(y):
            step = radius * y.reshape(rank, rank)
            result = []
            for kind, base, a, record, _, cscale in specs:
                point = base + a * step
                if kind == "radius":
                    result.append(inner(point, point) / radius**2 - 1)
                else:
                    result.append((inner(record.gradient, point) + gamma * covariance_square(point, record.covariance)
                                   - cfg.rho * record.gain) / cscale)
            return np.array(result)
        def constraint_jac(y):
            step = radius * y.reshape(rank, rank)
            rows = []
            for kind, base, a, record, _, cscale in specs:
                point = base + a * step
                value = 2 * a / radius * point if kind == "radius" else a * radius * (record.gradient + 2 * gamma * point @ record.covariance) / cscale
                rows.append(value.ravel())
            return np.array(rows)
        def constraint_hess(_y, multipliers):
            right_hess = np.zeros((rank, rank))
            for multiplier, (kind, _, a, record, _, cscale) in zip(multipliers, specs):
                right_hess += (2 * multiplier * a**2 * np.eye(rank) if kind == "radius" else
                               2 * multiplier * gamma * a**2 * radius**2 * record.covariance / cscale)
            return LinearOperator((size, size), matvec=lambda v: (v.reshape(rank, rank) @ right_hess).ravel(), dtype=np.float64)
        constraint = NonlinearConstraint(constraints, -np.inf, 0,
                                         jac=constraint_jac, hess=constraint_hess)
        started = time.perf_counter()
        try:
            result = minimize(fun, np.zeros(size), jac=jac, hess=lambda _: H,
                              method="trust-constr", constraints=[constraint],
                              options={"maxiter": cfg.solver_maxiter, "gtol": cfg.solver_gtol,
                                       "xtol": cfg.solver_xtol, "barrier_tol": cfg.solver_barrier_tol,
                                       "verbose": 0})
            proposed = old + radius * result.x.reshape(rank, rank)
            candidate = self.matrices.copy()
            candidate[owner] = proposed
            feasibility = self.check_constraints(candidate)
            change = float(fun(result.x) * scale)
            tol = cfg.majorizer_atol + cfg.majorizer_rtol * max(abs(loss), abs(change), scale)
            accepted = bool(feasibility["feasible"] and np.isfinite(change) and change <= tol)
            return _diagnostic_safe({"owner": owner, "proposed": proposed if accepted else old,
                    "base_loss": float(loss), "majorizer_change": change if accepted else 0.,
                    "raw_majorizer_change": change, "majorizer_tolerance": tol,
                    "score": float(loss + change) if accepted else float(loss),
                    "accepted": accepted, "nonzero": bool(accepted and np.linalg.norm(proposed - old) > 1e-14),
                    "solver_success": bool(result.success), "solver_status": int(result.status),
                    "solver_message": str(result.message), "iterations": int(result.niter),
                    "cg_iterations": int(getattr(result, "cg_niter", 0)),
                    "optimality": float(result.optimality), "constraint_violation": float(result.constr_violation),
                    "feasibility": feasibility, "solver_s": time.perf_counter() - started,
                    "constraint_count": len(specs), "dense_hessian_constructed": False})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
            return {"owner": owner, "proposed": old, "base_loss": float(loss),
                    "majorizer_change": 0., "raw_majorizer_change": None, "score": float(loss),
                    "accepted": False, "nonzero": False, "solver_success": False,
                    "solver_message": repr(exc), "solver_s": time.perf_counter() - started,
                    "feasibility": self.check_constraints(), "dense_hessian_constructed": False}

    def observe_and_write(self, summary: FeedbackSummary, objective: ResponseObjective,
                          served_read: ReadResult, final: bool = False) -> dict:
        """Append immutable feedback; protect served gain BEFORE all owner trials."""
        if any(s.response_id == summary.response_id for s in self.feedback):
            raise ValueError("Response already observed; benchmark replay cannot add feedback")
        if not np.array_equal(summary.forecast, served_read.forecast) or not np.array_equal(summary.descriptor, served_read.descriptor):
            raise ValueError("Feedback must match the actual saved causal read forecast")
        actual = np.einsum("k,koi->oi", served_read.alpha, self.matrices)
        if not np.allclose(actual, served_read.effective, atol=1e-12, rtol=1e-10):
            raise ValueError("Bank changed between serving and observing its response")
        started = time.perf_counter()
        objective_calls = []
        def evaluate(C, purpose):
            tick = time.perf_counter()
            loss, gradient = objective.loss_gradient(C)
            gradient = np.array(gradient, dtype=np.float64, copy=True)
            if gradient.shape != C.shape or not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
                raise ValueError("Invalid current-response objective result")
            objective_calls.append({"purpose": purpose, "seconds": time.perf_counter() - tick,
                                    "loss": float(loss), "gradient_norm": float(np.linalg.norm(gradient))})
            return float(loss), gradient
        zero_loss, zero_gradient = evaluate(np.zeros_like(self.matrices[0]), "frozen_reference")
        if not np.allclose(zero_gradient, summary.G, atol=1e-8, rtol=1e-6):
            raise ValueError("Stored frozen gradient differs from current fixed objective")
        served_loss, served_gradient = evaluate(served_read.effective, "served_fusion_record")
        protection_before = len(self.protection)
        if zero_loss - served_loss > 0:
            self.protection.append(ProtectionRecord(served_read.alpha, served_read.effective,
                                                    served_gradient, summary.S, zero_loss - served_loss,
                                                    summary.response_id, "served_transfer"))
        protection_before_candidates = len(self.protection)
        self.feedback.append(summary)
        entry = {"response_id": summary.response_id, "final": bool(final), "owner": None,
                 "records_before": protection_before, "records_before_owner_trials": protection_before_candidates,
                 "served_gain": zero_loss - served_loss, "served_loss": served_loss,
                 "frozen_loss": zero_loss, "candidates": [], "steps": [],
                 "objective_calls": objective_calls}
        if not final:
            for owner in range(self.config.slots):
                loss, gradient = evaluate(self.matrices[owner], f"candidate_{owner}")
                entry["candidates"].append(self._trial(owner, loss, gradient, summary.S))
            unused = [trial for trial in entry["candidates"] if not self.allocated[trial["owner"]]]
            activation_penalty = self.config.zeta * max([max(0., -trial["majorizer_change"]) for trial in unused], default=0.)
            for trial in entry["candidates"]:
                trial["activation_penalty"] = activation_penalty if not self.allocated[trial["owner"]] else 0.
                trial["rank_score"] = trial["score"] + trial["activation_penalty"]
            minimum = min(trial["rank_score"] for trial in entry["candidates"])
            tied = [trial for trial in entry["candidates"] if trial["rank_score"] <= minimum + self.config.owner_tie_atol]
            winner = min(tied, key=lambda trial: (not self.allocated[trial["owner"]], trial["owner"]))
            owner = winner["owner"]
            entry["owner"] = owner
            entry["steps"].append(copy.deepcopy(winner))
            self.matrices[owner] = winner["proposed"]
            if winner["nonzero"]:
                self.allocated[owner] = True
            loss, gradient = evaluate(self.matrices[owner], "selected_second_step")
            second = self._trial(owner, loss, gradient, summary.S)
            entry["steps"].append(second)
            self.matrices[owner] = second["proposed"]
            if second["nonzero"]:
                self.allocated[owner] = True
            fit_loss, fit_gradient = evaluate(self.matrices[owner], "postwrite_singleton_record")
            entry["singleton_fit_gain"] = zero_loss - fit_loss
            if zero_loss - fit_loss > 0:
                a = np.zeros(self.config.slots)
                a[owner] = 1
                self.protection.append(ProtectionRecord(a, self.matrices[owner], fit_gradient, summary.S,
                                                        zero_loss - fit_loss, summary.response_id, "singleton_fit"))
        entry["constraints_after"] = self.check_constraints()
        if not entry["constraints_after"]["feasible"]:
            raise RuntimeError("Accepted state violated immutable retention; qualification must stop")
        entry["records_after"] = len(self.protection)
        entry["candidate_gradient_s"] = sum(c["seconds"] for c in objective_calls if c["purpose"].startswith("candidate_"))
        entry["all_objective_s"] = sum(c["seconds"] for c in objective_calls)
        entry["solver_s"] = sum(c["solver_s"] for c in entry["candidates"]) + (entry["steps"][1]["solver_s"] if len(entry["steps"]) == 2 else 0)
        entry["adaptation_s"] = time.perf_counter() - started
        self.write_log.append(copy.deepcopy(entry))
        return entry
