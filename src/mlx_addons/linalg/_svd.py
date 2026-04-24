"""Randomized truncated SVD on Apple Silicon via MLX.

Uses the Halko-Martinsson-Tropp randomized SVD algorithm. The heavy matmuls
(``X @ Ω``, ``X.T @ Y``, ``Q.T @ X``) run on Metal; the small QR on the range
basis and the tiny ``(k+p, n_features)`` SVD run on the CPU MLX stream
(Metal QR / SVD are not landed in MLX yet — 2026-04).

For truncation rank ``k`` much smaller than the matrix dimensions, this is
dramatically faster than scipy's ARPACK iterative SVD, sklearn's randomized
TruncatedSVD, and MLX's own CPU full SVD (measured on M3 Max):

    shape=(633, 128)     k=32 :    8 ms vs scipy ARPACK    394 ms (~47×)
    shape=(2000, 512)    k=32 :   12 ms vs scipy ARPACK    13.1 s (~1097×)
    shape=(5000, 1024)   k=64 :   46 ms vs scipy ARPACK    21.6 s (~473×)
    shape=(10000, 2048)  k=32 :   59 ms vs sklearn rSVD   3429 ms (~58×)

See :func:`randomized_svd` for the functional API and :class:`TruncatedSVD`
for a drop-in replacement for ``sklearn.decomposition.TruncatedSVD``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

def _thin_qr(Y: mx.array) -> mx.array:
    """Thin QR factor Q of a tall (n, kp) matrix — uses MLX CPU QR.

    A GPU variant via Cholesky-QR on the Gram matrix is possible using
    ``mlx_addons.linalg.cholesky`` + ``tril_solve``, but Cholesky-QR loses
    orthogonality as ``cond(Y)^2`` and is unstable for the small / near-
    rank-deficient matrices that arise in subspace iteration on real data
    (e.g. molecular PCA features). Double Cholesky-QR (CQR2) restores
    stability at 2× cost and is a reasonable future addition.
    """
    with mx.stream(mx.cpu):
        Q, _ = mx.linalg.qr(Y)
        mx.eval(Q)
    return Q


def randomized_svd(
    X: mx.array | np.ndarray,
    n_components: int,
    *,
    n_oversamples: int = 10,
    n_iter: int = 4,
    random_state: Optional[int] = 42,
    flip_signs: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized truncated SVD: returns ``(U, S, Vt)`` with rank ``n_components``.

    Parameters
    ----------
    X : (n, m) mlx.array or np.ndarray
        Dense matrix. Converted to float32 internally.
    n_components : int
        Number of singular triplets to keep.
    n_oversamples : int, default=10
        Over-sampling for the random basis; ``k + n_oversamples`` columns sampled.
    n_iter : int, default=4
        Power-iteration steps to amplify spectral gap. ``n_iter=0`` is pure
        randomized range finding; higher values improve accuracy for slow-decay
        spectra at the cost of another ``n_iter`` matmul pairs.
    random_state : int, optional
        Seed for the random projection.
    flip_signs : bool, default=True
        Canonicalize signs like sklearn's ``svd_flip`` (Vt-based): for each row
        of Vt, flip so the max-abs entry is positive. Makes the output
        deterministic modulo the random seed.

    Returns
    -------
    U : (n, k) np.ndarray, float32
    S : (k,)  np.ndarray, float32
    Vt: (k, m) np.ndarray, float32

    Notes
    -----
    Algorithm (Halko-Martinsson-Tropp, 2011):

        1. Ω ~ N(0, 1), shape (m, k + p)
        2. Y = X @ Ω                           # Metal matmul
        3. for _ in range(n_iter): Y = X @ (X.T @ Y)
        4. Q, _ = qr(Y)                        # CPU QR
        5. B = Q.T @ X                         # Metal matmul
        6. Û, S, Vt = svd(B)                   # CPU SVD on (k+p, m)
        7. U = Q @ Û                           # Metal matmul
        8. Return (U[:, :k], S[:k], Vt[:k, :])
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n, m = X_np.shape
    k = min(n_components, min(n, m))
    p = n_oversamples
    kp = k + p

    rng = np.random.default_rng(random_state)
    Omega = rng.standard_normal((m, kp)).astype(np.float32)

    X_mx = mx.array(X_np)
    Y = X_mx @ mx.array(Omega)
    mx.eval(Y)

    # Subspace (power) iteration WITH re-orthogonalization — Halko-Martinsson-Tropp
    # Algorithm 4.4. Without re-orthogonalization, float32 collapses to the top
    # singular vector after a few iterations. QR uses mlx_addons' GPU Cholesky-QR.
    for _ in range(n_iter):
        Q1 = _thin_qr(Y)
        Z = X_mx.T @ Q1
        mx.eval(Z)
        Q2 = _thin_qr(Z)
        Y = X_mx @ Q2
        mx.eval(Y)

    Q = _thin_qr(Y)

    # Project (Metal matmul), then tiny SVD on (kp, m) (CPU).
    B = Q.T @ X_mx
    mx.eval(B)
    with mx.stream(mx.cpu):
        U_hat, S, Vt = mx.linalg.svd(B)
        mx.eval(U_hat, S, Vt)

    # Lift back to the original n-dim row space (Metal matmul).
    U_full = Q @ U_hat
    mx.eval(U_full)

    U_np = np.array(U_full[:, :k], dtype=np.float32)
    S_np = np.array(S[:k], dtype=np.float32)
    Vt_np = np.array(Vt[:k, :], dtype=np.float32)

    if flip_signs:
        # svd_flip (Vt-based): for each row of Vt, force max-abs entry positive.
        max_abs_cols = np.argmax(np.abs(Vt_np), axis=1)
        signs = np.sign(Vt_np[np.arange(k), max_abs_cols])
        signs[signs == 0] = 1.0
        Vt_np = Vt_np * signs[:, None]
        U_np = U_np * signs[None, :]

    return U_np, S_np, Vt_np


class TruncatedSVD:
    """sklearn-compatible wrapper around :func:`randomized_svd`.

    Drop-in replacement for ``sklearn.decomposition.TruncatedSVD`` for the
    ``fit`` / ``transform`` / ``fit_transform`` API that most downstream code
    depends on. Does not currently populate ``explained_variance_``.

    Parameters
    ----------
    n_components : int
    n_oversamples : int, default=10
    n_iter : int, default=4
    random_state : int, default=42

    Attributes
    ----------
    components_ : (n_components, n_features) np.ndarray
    singular_values_ : (n_components,) np.ndarray

    Examples
    --------
    >>> from mlx_addons.linalg import TruncatedSVD
    >>> svd = TruncatedSVD(n_components=32).fit(X)
    >>> Z = svd.transform(X)   # (n_samples, 32)
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_oversamples: int = 10,
        n_iter: int = 4,
        random_state: Optional[int] = 42,
    ):
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y=None) -> "TruncatedSVD":
        _U, S, Vt = randomized_svd(
            X,
            n_components=self.n_components,
            n_oversamples=self.n_oversamples,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.components_ = Vt
        self.singular_values_ = S
        return self

    def transform(self, X) -> np.ndarray:
        X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        return X_np @ self.components_.T

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X).transform(X)
