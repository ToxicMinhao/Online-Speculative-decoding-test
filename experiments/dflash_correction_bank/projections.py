"""Fixed coordinates for correction-bank v0.1; no feedback-dependent fitting."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _canonical_sign(matrix: np.ndarray) -> np.ndarray:
    result = np.array(matrix, dtype=np.float64, copy=True)
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1
    return result


def gaussian_orthonormal(rows: int, rank: int, seed: int = 17) -> np.ndarray:
    if rows < rank:
        raise ValueError("Projection rank exceeds input dimension")
    matrix = np.random.Generator(np.random.PCG64(seed)).standard_normal((rows, rank))
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return _canonical_sign(q)


def _as_numpy(value: Any) -> np.ndarray:
    # Handles torch CPU/GPU blocks without converting the full vocabulary at once.
    if hasattr(value, "detach"):
        return value.detach().to(device="cpu", dtype=__import__("torch").float64).numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class FixedProjections:
    U: np.ndarray
    V_in: np.ndarray
    V_descriptor: np.ndarray
    P: np.ndarray
    kappa: float
    seed: int = 17
    sign_convention: str = "largest-absolute-entry-positive; first-index-ties"
    descriptor_construction: str = "PCG64(17) Gaussian reduced-QR; unit-normalized output"

    def __post_init__(self):
        for name in ("U", "V_in", "V_descriptor", "P"):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.all(np.isfinite(array)):
                raise ValueError("Projection artifact contains nonfinite values")
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        if not np.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("Projection artifact requires finite positive kappa")

    def phi(self, features: np.ndarray) -> np.ndarray:
        raw = np.asarray(features) @ self.V_in
        return raw / np.maximum(1.0, np.linalg.norm(raw, axis=-1, keepdims=True))

    def descriptor(self, feature: np.ndarray) -> np.ndarray:
        raw = np.asarray(feature) @ self.V_descriptor
        norm = float(np.linalg.norm(raw))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("A finite nonzero prefill descriptor is required")
        return raw / norm

    def metadata(self) -> dict:
        return {"kappa": self.kappa, "radius": 1 / self.kappa, "seed": self.seed,
                "sign_convention": self.sign_convention,
                "descriptor_construction": self.descriptor_construction,
                "hashes": {name: array_sha256(getattr(self, name))
                           for name in ("U", "V_in", "V_descriptor", "P")}}

    def save(self, path: str | Path) -> None:
        np.savez(path, U=self.U, V_in=self.V_in, V_descriptor=self.V_descriptor,
                 P=self.P, kappa=np.array(self.kappa), seed=np.array(self.seed))

    @classmethod
    def load(cls, path: str | Path) -> "FixedProjections":
        with np.load(path, allow_pickle=False) as data:
            return cls(*(data[name].copy() for name in ("U", "V_in", "V_descriptor", "P")),
                       float(data["kappa"]), int(data["seed"]))


def build_projections(E: Any, descriptor_dimension: int, rank: int = 64,
                      seed: int = 17, chunk_rows: int = 4096,
                      gram: np.ndarray | None = None) -> FixedProjections:
    """Build top right singular vectors from E.T@E and pin their numerical result.

    A caller may supply a separately recorded Gram matrix computation (e.g. GPU
    float64); otherwise this uses float64 CPU row-chunk accumulation. No randomized
    SVD or learned basis is substituted. E itself is not modified.
    """
    vocab, hidden = E.shape
    if hidden < rank or descriptor_dimension < rank:
        raise ValueError("Rank exceeds a frozen feature dimension")
    if gram is None:
        gram = np.zeros((hidden, hidden), dtype=np.float64)
        for start in range(0, vocab, chunk_rows):
            block = _as_numpy(E[start:start + chunk_rows])
            gram += block.T @ block
    else:
        gram = np.asarray(gram, dtype=np.float64)
        if gram.shape != (hidden, hidden) or not np.all(np.isfinite(gram)):
            raise ValueError("Invalid supplied vocabulary Gram matrix")
    values, vectors = eigh((gram + gram.T) / 2,
                          subset_by_index=(hidden - rank, hidden - 1), driver="evr")
    U = _canonical_sign(vectors[:, ::-1])
    P = np.empty((vocab, rank), dtype=np.float64)
    for start in range(0, vocab, chunk_rows):
        P[start:start + chunk_rows] = _as_numpy(E[start:start + chunk_rows]) @ U
    center = P.mean(axis=0)
    kappa = 2 * float(np.linalg.norm(P - center, axis=1).max())
    if not np.isfinite(kappa) or kappa <= 0 or values[-1] <= 0:
        raise ValueError("The specification requires nondegenerate E and kappa > 0")
    return FixedProjections(U, gaussian_orthonormal(hidden, rank, seed),
                            gaussian_orthonormal(descriptor_dimension, rank, seed),
                            P, kappa, seed)
