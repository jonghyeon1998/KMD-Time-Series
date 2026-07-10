"""
nested_kriging.py
=================
Two-layer nested kriging for time-series forecasting with a Gibbs kernel.

Layer 1 — LengthscaleField (fit_gibbs_lengthscale_field)
---------------------------------------------------------
  Stage 1: Partition the N pooled windows into M_1 blocks of size ≈ block_size_l1.
           Within each block fit a scalar RBF lengthscale ℓ̂_m by maximising the
           local GP marginal log-likelihood.  When ℓ is constant inside a block,
           the Gibbs kernel reduces to an ordinary RBF kernel, so the local MLE is
           a standard 1-D optimisation.  The inverse curvature at the optimum gives
           the noise variance r_m (uncertainty in ℓ̂_m for that block).
  Stage 2: Fit a small heteroscedastic GP over the M_1 (centroid, log ℓ̂_m) pairs,
           with per-anchor noise r_m, using an RBF meta-kernel with fixed bandwidth.
  Stage 3: Return ℓ̃(x) = exp(ĝ(x) + v(x)/2)  (log-normal mean correction).

Layer 2 — NestedKrigingForecaster
----------------------------------
  Partition N windows into M_2 ≤ max_blocks blocks.  Within each block fit a local
  kriging model using k_Gibbs(·,·;ℓ̃).  Aggregate via the Rullière et al. (2018)
  cross-covariance-weighted formula — no N×N matrix is ever formed.

Dependencies: numpy only (no scipy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-10
_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _safe_cholesky(mat: np.ndarray) -> np.ndarray:
    """Cholesky with adaptive jitter; never raises LinAlgError."""
    jitter = 0.0
    n = mat.shape[0]
    for exp in range(15):
        try:
            return np.linalg.cholesky(mat + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10.0, _EPS * (10.0 ** exp))
    # Last resort
    return np.linalg.cholesky(mat + 1e-3 * np.eye(n))


# ---------------------------------------------------------------------------
# Low-level kernel helpers
# ---------------------------------------------------------------------------

def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return (n, m) matrix of squared Euclidean distances."""
    A_sq = np.sum(A * A, axis=1, keepdims=True)
    B_sq = np.sum(B * B, axis=1, keepdims=True).T
    sq = A_sq + B_sq - 2.0 * (A @ B.T)
    return np.maximum(sq, 0.0)


def _rbf_kernel(A: np.ndarray, B: np.ndarray, *, length_scale: float, amplitude: float = 1.0) -> np.ndarray:
    """Standard RBF / squared-exponential kernel matrix."""
    sq = _pairwise_sq_dists(A, B)
    return amplitude * np.exp(-0.5 * sq / (length_scale ** 2 + _EPS))


def gibbs_kernel_matrix(
    X: np.ndarray,
    ell_X: np.ndarray,
    Y: np.ndarray | None = None,
    ell_Y: np.ndarray | None = None,
    *,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Gibbs (nonstationary) kernel matrix.

    k(x_i, x_j) = amplitude
                  · sqrt(2 ℓ_i ℓ_j / (ℓ_i² + ℓ_j²))
                  · exp(−‖x_i − x_j‖² / (ℓ_i² + ℓ_j²))

    Parameters
    ----------
    X      : (n, d) context windows.
    ell_X  : (n,)  lengthscale ℓ̃(x_i) for each row of X  (must be > 0).
    Y      : (m, d) second set of windows; if None, compute X vs X.
    ell_Y  : (m,)  required when Y is not None.
    amplitude : scalar σ_f².

    Returns
    -------
    K : (n, m) kernel matrix.
    """
    A = np.asarray(X, dtype=float)
    lA = np.asarray(ell_X, dtype=float).ravel()
    if A.shape[0] != lA.shape[0]:
        raise ValueError("X and ell_X must have the same number of rows.")

    if Y is None:
        B, lB = A, lA
    else:
        B = np.asarray(Y, dtype=float)
        if ell_Y is None:
            raise ValueError("ell_Y must be provided when Y is provided.")
        lB = np.asarray(ell_Y, dtype=float).ravel()
        if B.shape[0] != lB.shape[0]:
            raise ValueError("Y and ell_Y must have the same number of rows.")

    sq_dists = _pairwise_sq_dists(A, B)                   # (n, m)
    lA2 = (lA ** 2)[:, None]                               # (n, 1)
    lB2 = (lB ** 2)[None, :]                               # (1, m)
    l_sum2 = lA2 + lB2                                     # (n, m)

    prefactor = np.sqrt(2.0 * lA[:, None] * lB[None, :] / (l_sum2 + _EPS))
    exponent = np.exp(-sq_dists / (l_sum2 + _EPS))
    return float(amplitude) * prefactor * exponent


# ---------------------------------------------------------------------------
# Block partitioning
# ---------------------------------------------------------------------------

def partition_into_blocks(
    n: int,
    *,
    block_size: int,
    max_blocks: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Randomly shuffle indices and split into blocks of size ≤ block_size.

    Parameters
    ----------
    n          : total number of windows.
    block_size : target block size; the last block may be smaller.
    max_blocks : if set, block_size is increased so that at most this many
                 blocks are created.
    rng        : numpy random generator; defaults to a fresh default_rng().

    Returns
    -------
    List of index arrays, one per block.
    """
    if n <= 0:
        return []
    if rng is None:
        rng = np.random.default_rng()

    if max_blocks is not None and max_blocks > 0:
        block_size = max(block_size, int(np.ceil(n / max_blocks)))

    indices = rng.permutation(n)
    return [indices[i: i + block_size] for i in range(0, n, block_size)]


# ---------------------------------------------------------------------------
# Layer 1 — local lengthscale MLE
# ---------------------------------------------------------------------------

def _neg_log_marginal(
    log_ell: float,
    X_block: np.ndarray,
    y_block: np.ndarray,
    *,
    sigma_n: float,
    amplitude: float,
) -> float:
    """Negative GP marginal log-likelihood for a block, treating ℓ as constant.

    When ℓ is constant inside a block, the Gibbs kernel is identical to an RBF
    kernel with that lengthscale (the normalising sqrt factor equals 1).

    log p = −½ log|K + σ_n² I| − ½ tr(Yᵀ(K+σ_n²I)⁻¹Y)/H  (+const)

    y_block may be (n,) or (n, H); H > 1 contributions are averaged.
    """
    ell = float(np.exp(log_ell))
    K = _rbf_kernel(X_block, X_block, length_scale=ell, amplitude=amplitude)
    n = K.shape[0]
    reg = K + (sigma_n ** 2) * np.eye(n)

    try:
        L = _safe_cholesky(reg)
    except np.linalg.LinAlgError:
        return 1e12

    # log|reg| = 2 · sum(log diag(L))
    log_det = 2.0 * np.sum(np.log(np.maximum(np.diag(L), _EPS)))

    y = np.asarray(y_block, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    # solve L·L^T·alpha = y  via two triangular solves
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    data_fit = float(np.sum(y * alpha)) / y.shape[1]

    return 0.5 * (log_det + data_fit + n * _LOG_2PI)


def _golden_section_min(
    f,
    lo: float,
    hi: float,
    *,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """Minimise a scalar function on [lo, hi] via golden-section search."""
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    resphi = 2.0 - phi
    x1 = lo + resphi * (hi - lo)
    x2 = hi - resphi * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(max_iter):
        if abs(hi - lo) < tol:
            break
        if f1 < f2:
            hi = x2
            x2, f2 = x1, f1
            x1 = lo + resphi * (hi - lo)
            f1 = f(x1)
        else:
            lo = x1
            x1, f1 = x2, f2
            x2 = hi - resphi * (hi - lo)
            f2 = f(x2)
    return 0.5 * (lo + hi)


def fit_block_lengthscale(
    X_block: np.ndarray,
    y_block: np.ndarray,
    *,
    sigma_n: float = 0.1,
    amplitude: float = 1.0,
    log_ell_bounds: tuple[float, float] = (-3.0, 5.0),
    curvature_h: float = 0.1,
) -> tuple[float, float]:
    """Find the MLE lengthscale for one block and estimate its uncertainty.

    Parameters
    ----------
    X_block        : (n, L) context windows in this block.
    y_block        : (n,) or (n, H) targets in this block.
    sigma_n        : observation noise std.
    amplitude      : kernel amplitude σ_f².
    log_ell_bounds : search interval [lo, hi] in log-space.
    curvature_h    : finite-difference step for curvature estimate (in log-space).

    Returns
    -------
    log_ell_hat : scalar — MLE log-lengthscale.
    r_m         : scalar — inverse Fisher information (uncertainty in log_ell_hat).
    """
    obj = lambda le: _neg_log_marginal(
        le, X_block, y_block, sigma_n=sigma_n, amplitude=amplitude
    )
    lo, hi = float(log_ell_bounds[0]), float(log_ell_bounds[1])
    log_ell_hat = _golden_section_min(obj, lo, hi)

    # Curvature via central finite difference of the negative log-likelihood
    h = float(curvature_h)
    f_plus = obj(log_ell_hat + h)
    f_mid = obj(log_ell_hat)
    f_minus = obj(log_ell_hat - h)
    curvature = (f_plus - 2.0 * f_mid + f_minus) / (h ** 2)
    # r_m = 1 / curvature  (variance of log_ell_hat under local quadratic approx)
    r_m = 1.0 / max(abs(curvature), _EPS)
    return float(log_ell_hat), float(r_m)


# ---------------------------------------------------------------------------
# Layer 1 — meta-GP (Stage 2 heteroscedastic GP over anchor points)
# ---------------------------------------------------------------------------

@dataclass
class LengthscaleField:
    """Trained Gibbs kernel lengthscale field.

    Attributes
    ----------
    centroids      : (M, L) block centroids in window space (x̄_m).
    log_ell_hats   : (M,)   MLE log-lengthscale per block (ĝ_m = log ℓ̂_m).
    noise_vars     : (M,)   Fisher-info uncertainty r_m per block.
    meta_amplitude : scalar amplitude of the RBF meta-kernel k_g.
    meta_bandwidth : scalar bandwidth of k_g  (fixed, typically large).
    _alpha         : (M,)   GP dual weights — populated by fit_gibbs_lengthscale_field.
    _chol          : (M, M) Cholesky of (K_g + diag(r)) — cached at fit time.
    """
    centroids: np.ndarray
    log_ell_hats: np.ndarray
    noise_vars: np.ndarray
    meta_amplitude: float
    meta_bandwidth: float
    _alpha: np.ndarray = field(repr=False)
    _chol: np.ndarray = field(repr=False)

    # ------------------------------------------------------------------
    def predict_log_ell(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean ĝ(x) and variance v(x) of log ℓ at each row of X.

        Returns
        -------
        mean : (n,) posterior mean.
        var  : (n,) posterior variance  (always ≥ 0).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        k_star = _rbf_kernel(
            X, self.centroids,
            length_scale=self.meta_bandwidth,
            amplitude=self.meta_amplitude,
        )                                                   # (n, M)
        mean = k_star @ self._alpha                         # (n,)

        # v(x) = k_g(x,x) − k_star · (K_g + diag(r))⁻¹ · k_star^T
        k_self = float(self.meta_amplitude)                 # k_g(x,x) = amplitude (RBF at dist=0)
        v_tmp = np.linalg.solve(self._chol, k_star.T)       # (M, n)
        var = k_self - np.sum(v_tmp ** 2, axis=0)           # (n,)
        var = np.maximum(var, 0.0)
        return mean, var

    def predict_ell(self, X: np.ndarray) -> np.ndarray:
        """Predict ℓ̃(x) = exp(ĝ(x) + v(x)/2) for each row of X.

        The v(x)/2 correction is the log-normal mean adjustment: if
        log ℓ ~ N(ĝ, v) then E[ℓ] = exp(ĝ + v/2).
        """
        mean, var = self.predict_log_ell(X)
        return np.exp(mean + 0.5 * var)

    @property
    def n_anchors(self) -> int:
        return int(self.centroids.shape[0])


def fit_gibbs_lengthscale_field(
    contexts: np.ndarray,
    targets: np.ndarray,
    *,
    block_size_l1: int = 40,
    sigma_n: float = 0.1,
    amplitude: float = 1.0,
    log_ell_bounds: tuple[float, float] = (-3.0, 5.0),
    curvature_h: float = 0.1,
    meta_amplitude: float = 1.0,
    meta_bandwidth: float | None = None,
    rng: np.random.Generator | None = None,
) -> LengthscaleField:
    """Layer 1: learn the Gibbs kernel lengthscale field from pooled windows.

    Parameters
    ----------
    contexts       : (N, L) pooled context windows (all series, all time steps).
    targets        : (N,) or (N, H) corresponding targets (must already be normalised).
    block_size_l1  : target block size for Stage 1.  Smaller → more anchors for the
                     meta-GP but noisier individual estimates.  Suggested: 30–80.
    sigma_n        : observation noise std assumed in the local MLE.
    amplitude      : kernel amplitude σ_f² assumed in the local MLE.
    log_ell_bounds : search interval [lo, hi] in log-space for the 1-D MLE.
    curvature_h    : finite-difference step size for curvature estimate.
    meta_amplitude : amplitude of the RBF meta-kernel k_g.
    meta_bandwidth : bandwidth of k_g.  If None, set to the median pairwise
                     distance between block centroids (median heuristic).
    rng            : numpy random generator for block partitioning.

    Returns
    -------
    LengthscaleField with .predict_ell(X) method.
    """
    X = np.asarray(contexts, dtype=float)
    y = np.asarray(targets, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    N = X.shape[0]

    blocks = partition_into_blocks(N, block_size=block_size_l1, rng=rng)
    M = len(blocks)

    centroids = np.zeros((M, X.shape[1]), dtype=float)
    log_ell_hats = np.zeros(M, dtype=float)
    noise_vars = np.zeros(M, dtype=float)

    for m, idx in enumerate(blocks):
        X_m = X[idx]
        y_m = y[idx]
        centroids[m] = X_m.mean(axis=0)
        log_ell_hats[m], noise_vars[m] = fit_block_lengthscale(
            X_m, y_m,
            sigma_n=sigma_n,
            amplitude=amplitude,
            log_ell_bounds=log_ell_bounds,
            curvature_h=curvature_h,
        )

    # Stage 2 — heteroscedastic meta-GP
    if meta_bandwidth is None:
        # Median pairwise distance heuristic (subsample if M is large)
        if M > 1:
            sq = _pairwise_sq_dists(centroids, centroids)
            off_diag = sq[np.triu_indices(M, k=1)]
            meta_bandwidth = float(np.sqrt(np.median(off_diag) + _EPS))
        else:
            meta_bandwidth = 1.0

    K_g = _rbf_kernel(centroids, centroids, length_scale=meta_bandwidth, amplitude=meta_amplitude)
    C = K_g + np.diag(noise_vars)                          # (M, M) noisy meta-gram
    chol_C = _safe_cholesky(C)

    alpha = np.linalg.solve(chol_C.T, np.linalg.solve(chol_C, log_ell_hats))

    return LengthscaleField(
        centroids=centroids,
        log_ell_hats=log_ell_hats,
        noise_vars=noise_vars,
        meta_amplitude=float(meta_amplitude),
        meta_bandwidth=float(meta_bandwidth),
        _alpha=alpha,
        _chol=chol_C,
    )


# ---------------------------------------------------------------------------
# Layer 2 — nested kriging regression with Gibbs kernel
# ---------------------------------------------------------------------------

@dataclass
class _FittedBlock:
    """Per-block state stored after fitting Layer 2."""
    indices: np.ndarray      # (n_m,) training indices for this block
    X: np.ndarray            # (n_m, L) context windows
    ell: np.ndarray          # (n_m,) ℓ̃(x_i) for each window
    chol: np.ndarray         # (n_m, n_m) Cholesky of (K_mm + σ_n² I)
    dual: np.ndarray         # (n_m, H) dual coefficients α_m = (K+σ²I)⁻¹ Y_m


class NestedKrigingForecaster:
    """Layer 2: nested kriging regression using the Gibbs kernel.

    This implements the Rullière et al. (2018) aggregation:

        ŷ(x*) = Σ_m w_m(x*) ŷ_m(x*)

    where the weights w = C⁻¹c are derived from the cross-block covariances
    computed analytically from the shared Gibbs kernel.

    Parameters
    ----------
    ell_field   : fitted LengthscaleField from fit_gibbs_lengthscale_field.
    block_size  : target block size; actual blocks may be larger if max_blocks
                  is binding.
    max_blocks  : cap on M (number of Layer-2 blocks).  Keeps the M×M
                  aggregation system small.  Default 30.
    sigma_n     : observation noise std for the regression layer.
    amplitude   : Gibbs kernel amplitude σ_f².
    rng         : random generator for block partitioning.
    """

    def __init__(
        self,
        ell_field: LengthscaleField,
        *,
        block_size: int = 100,
        max_blocks: int = 30,
        sigma_n: float = 0.1,
        amplitude: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.ell_field = ell_field
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.sigma_n = float(sigma_n)
        self.amplitude = float(amplitude)
        self.rng = rng or np.random.default_rng()
        self._fitted_blocks: list[_FittedBlock] | None = None
        self._horizon_length: int = 0

    # ------------------------------------------------------------------
    def fit(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
    ) -> "NestedKrigingForecaster":
        """Fit local kriging models on each block using the Gibbs kernel.

        Parameters
        ----------
        contexts : (N, L) pooled context windows (normalised).
        targets  : (N,) or (N, H) corresponding targets (normalised).
        """
        X = np.asarray(contexts, dtype=float)
        Y = np.asarray(targets, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        N = X.shape[0]
        self._horizon_length = Y.shape[1]

        # Precompute ℓ̃ at all training windows (batch call to ell_field)
        ell_all = self.ell_field.predict_ell(X)             # (N,)

        # Partition into Layer-2 blocks
        blocks = partition_into_blocks(
            N,
            block_size=self.block_size,
            max_blocks=self.max_blocks,
            rng=self.rng,
        )

        self._fitted_blocks = []
        for idx in blocks:
            X_m = X[idx]
            Y_m = Y[idx]
            ell_m = ell_all[idx]

            K_mm = gibbs_kernel_matrix(X_m, ell_m, amplitude=self.amplitude)
            reg = K_mm + (self.sigma_n ** 2) * np.eye(len(idx))
            chol_m = _safe_cholesky(reg)

            dual_m = np.linalg.solve(chol_m.T, np.linalg.solve(chol_m, Y_m))

            self._fitted_blocks.append(_FittedBlock(
                indices=idx,
                X=X_m,
                ell=ell_m,
                chol=chol_m,
                dual=dual_m,
            ))

        return self

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self._fitted_blocks is None:
            raise RuntimeError("Call fit() before predict().")

    # ------------------------------------------------------------------
    def _predict_single(
        self,
        x_star: np.ndarray,
        ell_star: float,
    ) -> tuple[np.ndarray, float]:
        """Predict for one query window x_star with its lengthscale ell_star.

        Returns
        -------
        y_hat : (H,) aggregated forecast.
        sigma2 : scalar posterior variance.
        """
        self._check_fitted()
        blocks = self._fitted_blocks  # type: ignore[assignment]
        M = len(blocks)
        x_star_2d = x_star[None, :]                         # (1, L)
        ell_star_arr = np.array([ell_star])

        # Per-block quantities
        y_locals = np.zeros((M, self._horizon_length), dtype=float)
        c_vec = np.zeros(M, dtype=float)
        v_list: list[np.ndarray] = []

        for m, blk in enumerate(blocks):
            # k(x*, X_m) : (1, n_m) → (n_m,)
            k_star_m = gibbs_kernel_matrix(
                x_star_2d, ell_star_arr,
                blk.X, blk.ell,
                amplitude=self.amplitude,
            ).ravel()

            y_locals[m] = k_star_m @ blk.dual             # (H,)

            # v_m = (K_mm + σ²I)⁻¹ k(X_m, x*)
            v_m = np.linalg.solve(blk.chol.T, np.linalg.solve(blk.chol, k_star_m))
            v_list.append(v_m)

            c_vec[m] = float(k_star_m @ v_m)              # scalar

        # C_{m,m'} = v_m^T K_{mm'} v_{m'}  — compute on the fly (M×M matrix)
        # For M ≤ max_blocks this is fast: O(M² × n²) per prediction.
        C_mat = np.zeros((M, M), dtype=float)
        for m, blk_m in enumerate(blocks):
            for mp, blk_mp in enumerate(blocks):
                if mp < m:
                    C_mat[m, mp] = C_mat[mp, m]
                    continue
                K_cross = gibbs_kernel_matrix(
                    blk_m.X, blk_m.ell,
                    blk_mp.X, blk_mp.ell,
                    amplitude=self.amplitude,
                )                                           # (n_m, n_mp)
                C_mat[m, mp] = float(v_list[m] @ K_cross @ v_list[mp])

        # Aggregation weights: w = C⁻¹ c
        chol_C_mat = _safe_cholesky(C_mat)
        w = np.linalg.solve(chol_C_mat.T, np.linalg.solve(chol_C_mat, c_vec))

        y_hat = w @ y_locals                                # (H,)

        # Posterior variance: k(x*,x*) − c^T w
        k_self = float(self.amplitude)                      # Gibbs k(x*,x*;ℓ*) = amplitude
        sigma2 = max(k_self - float(c_vec @ w), 0.0)

        return y_hat, sigma2

    # ------------------------------------------------------------------
    def predict(self, query_contexts: np.ndarray) -> np.ndarray:
        """Predict for one or more query windows.

        Parameters
        ----------
        query_contexts : (L,) single context or (n_q, L) batch.

        Returns
        -------
        predictions : (H,) if single query, else (n_q, H).
        """
        self._check_fitted()
        Q = np.atleast_2d(np.asarray(query_contexts, dtype=float))
        ell_Q = self.ell_field.predict_ell(Q)               # (n_q,)

        results = []
        for i in range(Q.shape[0]):
            y_hat, _ = self._predict_single(Q[i], float(ell_Q[i]))
            results.append(y_hat)

        out = np.stack(results, axis=0)
        if query_contexts.ndim == 1:
            return out[0]
        return out

    def predict_with_uncertainty(
        self, query_contexts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with posterior variance for one or more query windows.

        Returns
        -------
        predictions : (H,) or (n_q, H) posterior mean.
        variances   : scalar or (n_q,) posterior variance of f(x*).
        """
        self._check_fitted()
        Q = np.atleast_2d(np.asarray(query_contexts, dtype=float))
        ell_Q = self.ell_field.predict_ell(Q)

        means, vars_ = [], []
        for i in range(Q.shape[0]):
            y_hat, sigma2 = self._predict_single(Q[i], float(ell_Q[i]))
            means.append(y_hat)
            vars_.append(sigma2)

        mean_out = np.stack(means, axis=0)
        var_out = np.array(vars_, dtype=float)

        if query_contexts.ndim == 1:
            return mean_out[0], float(var_out[0])
        return mean_out, var_out

    # ------------------------------------------------------------------
    def save_state(
        self,
        path: "str | pathlib.Path",
        *,
        dataset_name: str = "",
    ) -> None:
        """Pre-compute and save block-centroid state for cross-dataset kriging.

        For each Layer-2 block m the following are stored:

        * ``centroids[m]``       — x̄_m = mean of block-m windows  (L,)
        * ``centroid_preds[m]``  — block-m local GP prediction at x̄_m  (H,)
        * ``centroid_vars[m]``   — block-m local GP variance at x̄_m  scalar

        These are the ingredients needed by
        :class:`CrossDatasetNestedKrigingForecaster` to incorporate this
        model as a data source without re-running the full fit.

        Parameters
        ----------
        path         : output path; a ``.npz`` suffix is added if absent.
        dataset_name : tag stored verbatim for bookkeeping.
        """
        import pathlib as _pathlib
        self._check_fitted()
        blocks = self._fitted_blocks  # type: ignore[assignment]
        M      = len(blocks)
        if M == 0:
            raise RuntimeError("No fitted blocks; cannot save state.")

        L = blocks[0].X.shape[1]
        H = self._horizon_length
        centroids      = np.zeros((M, L), dtype=float)
        centroid_preds = np.zeros((M, H), dtype=float)
        centroid_vars  = np.zeros(M, dtype=float)

        for m, blk in enumerate(blocks):
            x_bar   = blk.X.mean(axis=0)                       # (L,)
            centroids[m] = x_bar
            ell_bar = float(self.ell_field.predict_ell(x_bar[None, :])[0])

            # k(x̄_m, X_m) — cross-covariance row  (n_m,)
            k_c = gibbs_kernel_matrix(
                x_bar[None, :], np.array([ell_bar]),
                blk.X, blk.ell,
                amplitude=self.amplitude,
            ).ravel()

            # Local prediction: ŷ_m(x̄_m) = k_c @ α_m
            centroid_preds[m] = k_c @ blk.dual

            # Local variance: σ²_m = amplitude − k_c^T (K+σ²I)⁻¹ k_c
            alpha_c = np.linalg.solve(blk.chol.T, np.linalg.solve(blk.chol, k_c))
            centroid_vars[m] = max(float(self.amplitude) - float(k_c @ alpha_c), 0.0)

        np.savez_compressed(
            _pathlib.Path(path),
            centroids=centroids,
            centroid_preds=centroid_preds,
            centroid_vars=centroid_vars,
            context_length=np.array(L, dtype=int),
            horizon_length=np.array(H, dtype=int),
            dataset_name=np.array(dataset_name),
        )


# ---------------------------------------------------------------------------
# Convenience: pool windows from a collection of series
# ---------------------------------------------------------------------------

def pool_windows_from_series(
    series_list: Sequence[np.ndarray],
    *,
    context_length: int,
    horizon_length: int,
    stride: int = 1,
    max_windows_per_series: int | None = None,
    normalization: str = "robust_series",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build a pooled window dataset from multiple univariate series.

    Each series is instance-normalised (robust median/IQR) before windowing so
    that all windows live in a common normalised ℝ^L space.

    Parameters
    ----------
    series_list     : list of 1-D numpy arrays (training values, no NaN).
    context_length  : L — number of past observations per window.
    horizon_length  : H — number of future observations per window.
    stride          : step between consecutive window positions.
    max_windows_per_series : cap on windows extracted per series.
    normalization   : 'robust_series' (median/IQR per series) or 'none'.
    rng             : used to subsample windows when max_windows_per_series
                      is binding (uniform subsample, not stride truncation).

    Returns
    -------
    contexts : (N, L) pooled context windows.
    targets  : (N, H) pooled target windows.
    meta     : list of dicts with 'series_idx', 'window_end', 'loc', 'scale'
               for each row — useful for denormalising predictions.
    """
    if rng is None:
        rng = np.random.default_rng()

    all_contexts: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    meta: list[dict] = []

    for s_idx, raw in enumerate(series_list):
        arr = np.asarray(raw, dtype=float)
        T = arr.shape[0]
        if T < context_length + horizon_length:
            continue                                        # series too short

        # Instance normalisation
        if normalization == "robust_series":
            loc = float(np.median(arr))
            q75, q25 = float(np.percentile(arr, 75)), float(np.percentile(arr, 25))
            scale = max((q75 - q25) / 1.349, float(np.std(arr)), 1e-6)
        else:
            loc, scale = 0.0, 1.0

        normed = (arr - loc) / scale

        # Rolling window end positions
        ends = list(range(context_length, T - horizon_length + 1, stride))
        if not ends:
            continue

        if max_windows_per_series is not None and len(ends) > max_windows_per_series:
            chosen = rng.choice(len(ends), size=max_windows_per_series, replace=False)
            ends = [ends[i] for i in sorted(chosen)]

        for end in ends:
            ctx = normed[end - context_length: end]
            tgt = normed[end: end + horizon_length]
            all_contexts.append(ctx)
            all_targets.append(tgt)
            meta.append({
                "series_idx": s_idx,
                "window_end": end,
                "loc": loc,
                "scale": scale,
            })

    if not all_contexts:
        return np.zeros((0, context_length)), np.zeros((0, horizon_length)), []

    return (
        np.vstack(all_contexts),
        np.vstack(all_targets),
        meta,
    )


# ---------------------------------------------------------------------------
# Missing-data helpers
# ---------------------------------------------------------------------------

def _linear_impute(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation for NaN gaps; boundary NaNs are filled by nearest
    observed value (equivalent to constant extrapolation)."""
    out = arr.copy()
    n = len(out)
    finite_idx = np.where(np.isfinite(out))[0]
    if len(finite_idx) == 0:
        return out                                          # all NaN
    if len(finite_idx) == n:
        return out                                          # no NaN
    # np.interp clamps to boundary values outside the observed range
    out = np.interp(np.arange(n), finite_idx, out[finite_idx])
    return out


def pool_windows_from_series_with_missing(
    series_list: Sequence[np.ndarray],
    *,
    context_length: int,
    horizon_length: int,
    stride: int = 1,
    max_windows_per_series: int | None = None,
    normalization: str = "robust_series",
    min_observed_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Pool windows from series that may contain NaN gaps.

    NaN entries are filled by linear interpolation before windowing so that
    the context vectors are always fully numeric.  Normalization uses only the
    observed (non-NaN) values.  Windows whose target is entirely missing or
    whose context has fewer than `min_observed_fraction` observed entries are
    skipped.

    Intended for gap-missing datasets (electricity, kdd_cup, restaurant, …)
    where NaN marks sensor outages or calendar gaps rather than structural zeros.
    For intermittent demand (car_parts) use `pool_intermittent_windows` instead.
    """
    if rng is None:
        rng = np.random.default_rng()

    all_contexts: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    meta: list[dict] = []

    for s_idx, raw in enumerate(series_list):
        arr = np.asarray(raw, dtype=float)
        T = arr.shape[0]
        if T < context_length + horizon_length:
            continue

        arr_filled = _linear_impute(arr)
        raw_obs_mask = np.isfinite(arr)

        obs = arr[raw_obs_mask]
        if len(obs) < 2:
            continue

        if normalization == "robust_series":
            loc = float(np.median(obs))
            q75, q25 = float(np.percentile(obs, 75)), float(np.percentile(obs, 25))
            scale = max((q75 - q25) / 1.349, float(np.std(obs)), 1e-6)
        else:
            loc, scale = 0.0, 1.0

        normed = (arr_filled - loc) / scale

        ends = list(range(context_length, T - horizon_length + 1, stride))
        valid_ends = [
            e for e in ends
            if raw_obs_mask[e - context_length: e].mean() >= min_observed_fraction
            and raw_obs_mask[e: e + horizon_length].any()
        ]
        if not valid_ends:
            continue

        if max_windows_per_series is not None and len(valid_ends) > max_windows_per_series:
            chosen = rng.choice(len(valid_ends), size=max_windows_per_series, replace=False)
            valid_ends = [valid_ends[i] for i in sorted(chosen)]

        for end in valid_ends:
            all_contexts.append(normed[end - context_length: end])
            all_targets.append(normed[end: end + horizon_length])
            meta.append({"series_idx": s_idx, "window_end": end, "loc": loc, "scale": scale})

    if not all_contexts:
        return np.zeros((0, context_length)), np.zeros((0, horizon_length)), []

    return np.vstack(all_contexts), np.vstack(all_targets), meta


# ---------------------------------------------------------------------------
# Intermittent demand: pool_intermittent_windows
# ---------------------------------------------------------------------------

def pool_intermittent_windows(
    series_list: Sequence[np.ndarray],
    *,
    context_length: int,
    horizon_length: int,
    stride: int = 1,
    max_windows_per_series: int | None = None,
    min_observed_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Build indicator and magnitude window pools for intermittent demand series.

    Intended for datasets like car_parts where y ≥ 0 with many structural zeros
    AND some NaN gaps (sensor outages / stockout periods with no record).

    Design
    ------
    The context is always linearly imputed so all windows are fully numeric.
    Normalisation uses the std of the strictly-positive observed values (the IQR
    of a series with many zeros is 0).  The location is fixed at 0 because zeros
    are structural and meaningful.

    Indicator pool  (ALL valid windows)
        z_h = 1[y_h > 0]  for each observed target step.
        NaN target steps are treated as z_h = 0 (conservatively).

    Magnitude pool  (windows where ALL H target steps are observed AND > 0)
        Targets: log(y_h / scale) for each step.

    Returns
    -------
    ind_ctx    : (N_ind, L) context windows for the indicator model.
    ind_labels : (N_ind, H) binary labels z ∈ {0, 1}.
    mag_ctx    : (N_mag, L) context windows for the magnitude model.
    mag_log_y  : (N_mag, H) log-demand targets (log-normalised).
    ind_meta   : list[dict] — series_idx, window_end, loc=0, scale per indicator row.
    mag_meta   : list[dict] — same fields for magnitude rows.
    """
    if rng is None:
        rng = np.random.default_rng()

    ind_ctx_l, ind_lab_l, ind_meta_l = [], [], []
    mag_ctx_l, mag_tgt_l, mag_meta_l = [], [], []

    for s_idx, raw in enumerate(series_list):
        arr = np.asarray(raw, dtype=float)
        T = arr.shape[0]
        if T < context_length + horizon_length:
            continue

        arr_filled = _linear_impute(arr)
        raw_obs = arr                                       # original, with NaN

        # Scale by std of strictly-positive observed values; loc = 0
        pos_vals = arr[np.isfinite(arr) & (arr > 0)]
        if len(pos_vals) == 0:
            continue                                        # series is all-zero: skip
        scale = max(float(np.std(pos_vals)), 1e-6)

        normed = arr_filled / scale

        ends = list(range(context_length, T - horizon_length + 1, stride))
        valid_ends = []
        for end in ends:
            ctx_obs = np.isfinite(raw_obs[end - context_length: end])
            tgt_obs = np.isfinite(raw_obs[end: end + horizon_length])
            if ctx_obs.mean() < min_observed_fraction:
                continue
            if not tgt_obs.any():
                continue
            valid_ends.append(end)

        if not valid_ends:
            continue

        if max_windows_per_series is not None and len(valid_ends) > max_windows_per_series:
            chosen = rng.choice(len(valid_ends), size=max_windows_per_series, replace=False)
            valid_ends = [valid_ends[i] for i in sorted(chosen)]

        for end in valid_ends:
            ctx = normed[end - context_length: end]
            tgt_raw = raw_obs[end: end + horizon_length]
            tgt_obs = np.isfinite(tgt_raw)

            # Indicator labels: z_h = 1[y_h > 0] where observed; 0 elsewhere
            z_h = np.where(tgt_obs, (tgt_raw > 0).astype(float), 0.0)
            ind_ctx_l.append(ctx)
            ind_lab_l.append(z_h)
            ind_meta_l.append({"series_idx": s_idx, "window_end": end, "loc": 0.0, "scale": scale})

            # Magnitude pool: all H steps observed AND positive
            if tgt_obs.all() and (tgt_raw > 0).all():
                mag_ctx_l.append(ctx)
                mag_tgt_l.append(np.log(tgt_raw / scale))
                mag_meta_l.append({"series_idx": s_idx, "window_end": end, "loc": 0.0, "scale": scale})

    def _stack(lst_ctx, lst_tgt, clen, hlen):
        if lst_ctx:
            return np.vstack(lst_ctx), np.vstack(lst_tgt)
        return np.zeros((0, clen)), np.zeros((0, hlen))

    ind_ctx_a, ind_lab_a = _stack(ind_ctx_l, ind_lab_l, context_length, horizon_length)
    mag_ctx_a, mag_tgt_a = _stack(mag_ctx_l, mag_tgt_l, context_length, horizon_length)

    return ind_ctx_a, ind_lab_a, mag_ctx_a, mag_tgt_a, ind_meta_l, mag_meta_l


# ---------------------------------------------------------------------------
# Bernoulli GP classification with Laplace approximation
# ---------------------------------------------------------------------------

def _sigmoid(f: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid σ(f) = 1 / (1 + e^{-f})."""
    return np.where(
        f >= 0,
        1.0 / (1.0 + np.exp(-f)),
        np.exp(f) / (1.0 + np.exp(f)),
    )


def _laplace_mode(
    K: np.ndarray,
    z: np.ndarray,
    *,
    n_iter: int = 20,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Newton-Raphson for the Bernoulli GP Laplace posterior mode.

    Logistic (sigmoid) link: P(z_i = 1 | f_i) = σ(f_i).

    The log-posterior is ψ(f) = Σ_i [z_i log σ(f_i) + (1-z_i) log(1-σ(f_i))]
    − ½ fᵀ K⁻¹ f  +  const.

    Gradient:  ∇ψ = (z − π) − K⁻¹ f
    Hessian:   ∇²ψ = −W − K⁻¹   where  W = diag(π_i(1−π_i))

    At convergence ∇ψ = 0, so K⁻¹ f_MAP = z − π_MAP.
    This identity is used throughout to avoid explicitly inverting K.

    The Newton update is derived via the Woodbury identity:
        f_new = K · a
        where  b = W·f + (z − π),
               a = b − W^{1/2} · B⁻¹ · W^{1/2} · K · b,
               B = I + W^{1/2} K W^{1/2}  (solved via Cholesky).

    Returns
    -------
    f_map  : (n,) posterior mode.
    pi_map : (n,) σ(f_map) — predicted probabilities at training points.
    W_diag : (n,) π_i(1−π_i) — diagonal of W evaluated at f_map.
    """
    n = K.shape[0]
    f = np.zeros(n)
    z = np.asarray(z, dtype=float)

    for _ in range(n_iter):
        pi = _sigmoid(f)
        W = np.maximum(pi * (1.0 - pi), 1e-10)
        sqrt_W = np.sqrt(W)

        B = np.eye(n) + (sqrt_W[:, None] * K) * sqrt_W[None, :]
        L_B = _safe_cholesky(B)

        b = W * f + (z - pi)
        tmp = sqrt_W * (K @ b)
        a = b - sqrt_W * np.linalg.solve(L_B.T, np.linalg.solve(L_B, tmp))
        f_new = K @ a

        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new

    pi = _sigmoid(f)
    W_diag = np.maximum(pi * (1.0 - pi), 1e-10)
    return f, pi, W_diag


def _bernoulli_neg_log_marginal(
    log_ell: float,
    X_block: np.ndarray,
    z_block: np.ndarray,
    *,
    amplitude: float = 1.0,
    n_iter: int = 20,
) -> float:
    """Laplace-approximated negative log marginal likelihood for binary labels.

    log p(z | X, ℓ)  ≈  log p(z | f_MAP)
                        − ½ f_MAP^T (z − π_MAP)     [using K⁻¹f_MAP = z−π_MAP]
                        − ½ log|B|

    where B = I + W^{1/2} K W^{1/2} and W = diag(π(1−π)) at f_MAP.
    """
    ell = float(np.exp(log_ell))
    K = _rbf_kernel(X_block, X_block, length_scale=ell, amplitude=amplitude)
    n = K.shape[0]

    z = np.asarray(z_block, dtype=float)
    f_map, pi_map, W_diag = _laplace_mode(K, z, n_iter=n_iter)

    # Bernoulli log-likelihood at f_MAP
    pi_c = np.clip(pi_map, 1e-10, 1.0 - 1e-10)
    log_lik = float(np.sum(z * np.log(pi_c) + (1.0 - z) * np.log(1.0 - pi_c)))

    # ½ f_MAP^T K^{-1} f_MAP = ½ f_MAP^T (z − π_MAP)  (KKT identity)
    quad = float(np.dot(f_map, z - pi_map))

    # ½ log|B|
    sqrt_W = np.sqrt(W_diag)
    B = np.eye(n) + (sqrt_W[:, None] * K) * sqrt_W[None, :]
    L_B = _safe_cholesky(B)
    log_det_B = 2.0 * float(np.sum(np.log(np.maximum(np.diag(L_B), _EPS))))

    return -(log_lik - 0.5 * quad - 0.5 * log_det_B)


def fit_block_lengthscale_bernoulli(
    X_block: np.ndarray,
    z_block: np.ndarray,
    *,
    amplitude: float = 1.0,
    log_ell_bounds: tuple[float, float] = (-3.0, 5.0),
    curvature_h: float = 0.1,
    n_iter_laplace: int = 20,
) -> tuple[float, float]:
    """MLE lengthscale for one block with binary labels via Laplace approximation.

    Returns (log_ell_hat, r_m) with the same semantics as fit_block_lengthscale.
    Blocks where all labels are identical (degenerate) return (0.0, 10.0).
    """
    z = np.asarray(z_block, dtype=float)
    if z.std() < 1e-8:
        return 0.0, 10.0                                    # degenerate block

    obj = lambda le: _bernoulli_neg_log_marginal(
        le, X_block, z, amplitude=amplitude, n_iter=n_iter_laplace
    )
    lo, hi = float(log_ell_bounds[0]), float(log_ell_bounds[1])
    log_ell_hat = _golden_section_min(obj, lo, hi)

    h = float(curvature_h)
    f_mid = obj(log_ell_hat)
    f_plus = obj(log_ell_hat + h)
    f_minus = obj(log_ell_hat - h)
    curvature = (f_plus - 2.0 * f_mid + f_minus) / (h ** 2)
    r_m = 1.0 / max(abs(curvature), _EPS)
    return float(log_ell_hat), float(r_m)


def fit_bernoulli_lengthscale_field(
    contexts: np.ndarray,
    labels: np.ndarray,
    *,
    block_size_l1: int = 40,
    amplitude: float = 1.0,
    log_ell_bounds: tuple[float, float] = (-3.0, 5.0),
    curvature_h: float = 0.1,
    n_iter_laplace: int = 20,
    meta_amplitude: float = 1.0,
    meta_bandwidth: float | None = None,
    rng: np.random.Generator | None = None,
) -> LengthscaleField:
    """Layer 1 for the indicator model: Bernoulli MLE per block → meta-GP.

    Identical pipeline to fit_gibbs_lengthscale_field except Stage 1 uses
    Laplace-approximated Bernoulli log-marginal likelihood rather than the
    exact Gaussian one.

    If labels has shape (N, H) with H > 1, labels are averaged across the
    horizon before fitting (each block gives a single scalar ℓ̂_m^z).
    """
    X = np.asarray(contexts, dtype=float)
    Z = np.asarray(labels, dtype=float)
    if Z.ndim == 2:
        Z = Z.mean(axis=1)                                  # (N,) average over H
    N = X.shape[0]

    blocks = partition_into_blocks(N, block_size=block_size_l1, rng=rng)
    M = len(blocks)

    centroids = np.zeros((M, X.shape[1]), dtype=float)
    log_ell_hats = np.zeros(M, dtype=float)
    noise_vars = np.zeros(M, dtype=float)

    for m, idx in enumerate(blocks):
        X_m = X[idx]
        z_m = Z[idx]
        centroids[m] = X_m.mean(axis=0)
        log_ell_hats[m], noise_vars[m] = fit_block_lengthscale_bernoulli(
            X_m, z_m,
            amplitude=amplitude,
            log_ell_bounds=log_ell_bounds,
            curvature_h=curvature_h,
            n_iter_laplace=n_iter_laplace,
        )

    if meta_bandwidth is None:
        if M > 1:
            sq = _pairwise_sq_dists(centroids, centroids)
            off_diag = sq[np.triu_indices(M, k=1)]
            meta_bandwidth = float(np.sqrt(np.median(off_diag) + _EPS))
        else:
            meta_bandwidth = 1.0

    K_g = _rbf_kernel(centroids, centroids, length_scale=meta_bandwidth, amplitude=meta_amplitude)
    C = K_g + np.diag(noise_vars)
    chol_C = _safe_cholesky(C)
    alpha = np.linalg.solve(chol_C.T, np.linalg.solve(chol_C, log_ell_hats))

    return LengthscaleField(
        centroids=centroids,
        log_ell_hats=log_ell_hats,
        noise_vars=noise_vars,
        meta_amplitude=float(meta_amplitude),
        meta_bandwidth=float(meta_bandwidth),
        _alpha=alpha,
        _chol=chol_C,
    )


# ---------------------------------------------------------------------------
# Bernoulli nested kriging classifier
# ---------------------------------------------------------------------------

@dataclass
class _BernoulliBlock:
    """Per-block state for the Bernoulli nested kriging classifier."""
    X: np.ndarray        # (n_m, L) context windows
    ell: np.ndarray      # (n_m,) ℓ̃(x_i)
    alpha_f: np.ndarray  # (n_m, H) K_mm^{-1} f_MAP per horizon step = z_m − π_MAP_m
    chol_K: np.ndarray   # (n_m, n_m) Cholesky of K_mm (Gibbs, with jitter)


class BernoulliKrigingClassifier:
    """Nested kriging binary classifier using a Bernoulli likelihood + Laplace.

    For each horizon step h, the latent GP f_h: ℝ^L → ℝ is fitted by finding
    the Laplace posterior mode in each Layer-2 block.  At prediction the local
    latent values are aggregated via the Rullière et al. cross-covariance
    formula in latent space, then squeezed through the sigmoid to give
    P(z_h = 1 | x*).

    Since the Rullière aggregation is linear and we apply σ(·) afterwards,
    this is not exact GP classification but a pragmatic approximation that
    inherits the computational benefits of nested kriging.

    Parameters
    ----------
    ell_field       : LengthscaleField from fit_bernoulli_lengthscale_field.
    block_size      : Layer-2 block target size.
    max_blocks      : Cap on M (default 30).
    amplitude       : Gibbs kernel amplitude.
    n_iter_laplace  : Newton iterations per block per horizon step.
    """

    def __init__(
        self,
        ell_field: LengthscaleField,
        *,
        block_size: int = 100,
        max_blocks: int = 30,
        amplitude: float = 1.0,
        n_iter_laplace: int = 20,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.ell_field = ell_field
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.amplitude = float(amplitude)
        self.n_iter_laplace = int(n_iter_laplace)
        self.rng = rng or np.random.default_rng()
        self._fitted_blocks: list[_BernoulliBlock] | None = None
        self._horizon_length: int = 0

    def fit(
        self,
        contexts: np.ndarray,
        labels: np.ndarray,
    ) -> "BernoulliKrigingClassifier":
        """Fit local Bernoulli GP classifiers per block.

        Parameters
        ----------
        contexts : (N, L) pooled context windows (normalised / imputed).
        labels   : (N,) or (N, H) binary targets z = 1[y > 0].
        """
        X = np.asarray(contexts, dtype=float)
        Z = np.asarray(labels, dtype=float)
        if Z.ndim == 1:
            Z = Z[:, None]
        N, H = X.shape[0], Z.shape[1]
        self._horizon_length = H

        ell_all = self.ell_field.predict_ell(X)

        blocks = partition_into_blocks(
            N, block_size=self.block_size, max_blocks=self.max_blocks, rng=self.rng
        )

        self._fitted_blocks = []
        for idx in blocks:
            X_m, Z_m, ell_m = X[idx], Z[idx], ell_all[idx]

            K_gibbs = gibbs_kernel_matrix(X_m, ell_m, amplitude=self.amplitude)
            chol_K = _safe_cholesky(K_gibbs)

            # Laplace mode per horizon step; dual α_h = K^{-1} f_MAP = z_h − π_MAP_h
            alpha_f = np.zeros((len(idx), H), dtype=float)
            for h in range(H):
                z_h = Z_m[:, h]
                if z_h.std() < 1e-8:
                    pi_h = _sigmoid(np.zeros(len(idx)))
                    alpha_f[:, h] = z_h - pi_h
                else:
                    _, pi_h, _ = _laplace_mode(K_gibbs, z_h, n_iter=self.n_iter_laplace)
                    alpha_f[:, h] = z_h - pi_h

            self._fitted_blocks.append(_BernoulliBlock(
                X=X_m, ell=ell_m, alpha_f=alpha_f, chol_K=chol_K,
            ))

        return self

    def _predict_single(self, x_star: np.ndarray, ell_star: float) -> np.ndarray:
        """P(z_h = 1 | x*) for each h via Rullière aggregation in latent space."""
        blocks = self._fitted_blocks
        M = len(blocks)
        x2 = x_star[None, :]
        ell_arr = np.array([ell_star])

        f_locals = np.zeros((M, self._horizon_length))
        c_vec = np.zeros(M)
        v_list: list[np.ndarray] = []

        for m, blk in enumerate(blocks):
            k_star = gibbs_kernel_matrix(
                x2, ell_arr, blk.X, blk.ell, amplitude=self.amplitude
            ).ravel()                                       # (n_m,)

            f_locals[m] = k_star @ blk.alpha_f             # (H,)
            v_m = np.linalg.solve(blk.chol_K.T, np.linalg.solve(blk.chol_K, k_star))
            v_list.append(v_m)
            c_vec[m] = float(k_star @ v_m)

        C_mat = np.zeros((M, M))
        for m, blk_m in enumerate(blocks):
            for mp, blk_mp in enumerate(blocks):
                if mp < m:
                    C_mat[m, mp] = C_mat[mp, m]
                    continue
                K_cross = gibbs_kernel_matrix(
                    blk_m.X, blk_m.ell, blk_mp.X, blk_mp.ell,
                    amplitude=self.amplitude,
                )
                C_mat[m, mp] = float(v_list[m] @ K_cross @ v_list[mp])

        chol_C = _safe_cholesky(C_mat)
        w = np.linalg.solve(chol_C.T, np.linalg.solve(chol_C, c_vec))

        f_star = w @ f_locals                               # (H,) aggregated latent
        return _sigmoid(f_star)                             # P(z=1|x*) per horizon step

    def predict_proba(self, query_contexts: np.ndarray) -> np.ndarray:
        """P(z_h = 1 | x*) for each query window and each horizon step.

        Returns (H,) for a single (L,) query or (n_q, H) for (n_q, L) batch.
        """
        if self._fitted_blocks is None:
            raise RuntimeError("Call fit() before predict_proba().")
        Q = np.atleast_2d(np.asarray(query_contexts, dtype=float))
        ell_Q = self.ell_field.predict_ell(Q)

        out = np.stack(
            [self._predict_single(Q[i], float(ell_Q[i])) for i in range(Q.shape[0])],
            axis=0,
        )
        return out[0] if query_contexts.ndim == 1 else out


# ---------------------------------------------------------------------------
# Two-model intermittent demand forecaster
# ---------------------------------------------------------------------------

class TwoModelIntermittentForecaster:
    """E[y | x*] for intermittent demand (e.g. car_parts).

    Implements the factorisation:
        E[y_h | x*] = P(z_h=1 | x*) · exp(μ_h(x*) + σ²(x*)/2)

    where z_h = 1[y_h > 0] and the two factors come from independent models:

    Model 1 — BernoulliKrigingClassifier
        Fitted on the indicator pool (all valid windows).
        Gives P(z_h=1 | x*) for each horizon step.

    Model 2 — NestedKrigingForecaster on log y
        Fitted on the magnitude pool (windows where all target steps y_h > 0).
        Posterior mean μ_h(x*) = E[log y_h | z_h=1, x*].
        Posterior variance σ²(x*) (scalar per query).

    The exp(μ + σ²/2) correction is the log-normal mean: if log y | x* ~ N(μ, σ²)
    then E[y | x*] = exp(μ + σ²/2).  Omitting σ²/2 would systematically
    underestimate demand.

    Parameters
    ----------
    indicator : fitted BernoulliKrigingClassifier (Model 1).
    magnitude : fitted NestedKrigingForecaster on log y (Model 2).
    """

    def __init__(
        self,
        *,
        indicator: BernoulliKrigingClassifier,
        magnitude: NestedKrigingForecaster,
    ) -> None:
        self.indicator = indicator
        self.magnitude = magnitude

    def predict(self, query_contexts: np.ndarray) -> np.ndarray:
        """E[y_h | x*] for each query window and horizon step.

        Returns (H,) for a single (L,) query or (n_q, H) for a batch.
        """
        Q = np.atleast_2d(np.asarray(query_contexts, dtype=float))

        prob_z1 = self.indicator.predict_proba(Q)           # (n_q, H)

        ell_Q = self.magnitude.ell_field.predict_ell(Q)     # (n_q,)
        log_means, log_vars = [], []
        for i in range(Q.shape[0]):
            mu_h, sigma2 = self.magnitude._predict_single(Q[i], float(ell_Q[i]))
            log_means.append(mu_h)
            log_vars.append(float(sigma2))
        log_mean = np.stack(log_means, axis=0)              # (n_q, H)
        log_var = np.array(log_vars, dtype=float)           # (n_q,)

        # E[y_h | x*] = P(z_h=1|x*) · exp(μ_h + σ²/2)
        E_y = prob_z1 * np.exp(log_mean + 0.5 * log_var[:, None])

        return E_y[0] if query_contexts.ndim == 1 else E_y


# ===========================================================================
# Multivariate extensions
# ===========================================================================
# Design: each output channel is handled by its own NestedKrigingForecaster.
# All channels share a single LengthscaleField trained on the pooled (cross-
# channel) windows.  This captures the observation that "smoothness structure"
# is often similar across channels of the same dataset while keeping per-channel
# local kriging tractable.
#
# Window pooling strategy
# -----------------------
# For a (T, p) multivariate series, we extract windows channel-by-channel:
#   context for channel i at time t : y^(i)_{t-L : t}  ∈ ℝ^L
#   target  for channel i at time t : y^(i)_{t : t+H}  ∈ ℝ^H
# All (channel, window) pairs are stacked into a single (N_total, L) context
# matrix for the shared ℓ-field estimation, then split back by channel for the
# per-channel nested kriging fits.
# ---------------------------------------------------------------------------


def pool_windows_from_multivariate_series(
    series_list,
    *,
    context_length: int,
    horizon_length: int,
    stride: int = 1,
    max_windows_per_series: int | None = None,
    normalization: str = "robust_series",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], int]:
    """Pool windows from multivariate (T, p) series, one channel at a time.

    Each element of *series_list* must be a 2-D array of shape ``(T, p)``.
    All channels are independently normalised.  Windows are pooled across all
    channels *and* series so that the shared LengthscaleField sees diverse
    context patterns.

    Parameters
    ----------
    series_list : list of (T, p) arrays.
    context_length, horizon_length, stride, max_windows_per_series :
        same meaning as in :func:`pool_windows_from_series`.
    normalization : 'robust_series' or 'none'.
    rng : random generator used for window sub-sampling.

    Returns
    -------
    all_contexts : (N, L) array — pooled across all channels and series.
    all_targets  : (N, H) array — corresponding single-channel targets.
    meta         : list of dicts — keys: series_idx, channel_idx, window_end,
                   loc, scale.
    p            : int — number of channels (inferred from first valid series).
    """
    if rng is None:
        rng = np.random.default_rng()

    all_ctx: list[np.ndarray] = []
    all_tgt: list[np.ndarray] = []
    meta: list[dict] = []
    p_ref: int = -1

    for s_idx, raw in enumerate(series_list):
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]                               # treat as (T, 1)
        T, p = arr.shape
        if p_ref < 0:
            p_ref = p
        if T < context_length + horizon_length:
            continue

        for ch in range(p):
            ch_arr = arr[:, ch]                             # (T,)

            if normalization == "robust_series":
                loc = float(np.median(ch_arr))
                q75, q25 = float(np.percentile(ch_arr, 75)), float(np.percentile(ch_arr, 25))
                scale = max((q75 - q25) / 1.349, float(np.std(ch_arr)), 1e-6)
            else:
                loc, scale = 0.0, 1.0

            normed = (ch_arr - loc) / scale

            ends = list(range(context_length, T - horizon_length + 1, stride))
            if not ends:
                continue

            if max_windows_per_series is not None and len(ends) > max_windows_per_series:
                chosen = rng.choice(len(ends), size=max_windows_per_series, replace=False)
                ends = [ends[i] for i in sorted(chosen)]

            for end in ends:
                all_ctx.append(normed[end - context_length: end])
                all_tgt.append(normed[end: end + horizon_length])
                meta.append({
                    "series_idx": s_idx,
                    "channel_idx": ch,
                    "window_end": end,
                    "loc": loc,
                    "scale": scale,
                })

    if not all_ctx:
        p_ref = max(p_ref, 1)
        return (
            np.zeros((0, context_length)),
            np.zeros((0, horizon_length)),
            [],
            p_ref,
        )

    return np.vstack(all_ctx), np.vstack(all_tgt), meta, max(p_ref, 1)


def pool_windows_from_multivariate_series_with_missing(
    series_list,
    *,
    context_length: int,
    horizon_length: int,
    stride: int = 1,
    max_windows_per_series: int | None = None,
    normalization: str = "robust_series",
    min_observed_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], int]:
    """Like :func:`pool_windows_from_multivariate_series` but NaN-safe.

    NaN values in each channel are linearly imputed before windowing.
    Normalisation uses only the observed (non-NaN) values per channel.
    Windows are skipped if fewer than *min_observed_fraction* of the context
    entries are observed **or** if the entire target horizon is missing.

    Parameters
    ----------
    series_list           : list of (T, p) arrays (may contain NaN).
    min_observed_fraction : minimum fraction of non-NaN entries in a context
                            window required to include that window.

    Returns
    -------
    Same four values as :func:`pool_windows_from_multivariate_series`.
    """
    if rng is None:
        rng = np.random.default_rng()

    all_ctx: list[np.ndarray] = []
    all_tgt: list[np.ndarray] = []
    meta: list[dict] = []
    p_ref: int = -1

    for s_idx, raw in enumerate(series_list):
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        T, p = arr.shape
        if p_ref < 0:
            p_ref = p
        if T < context_length + horizon_length:
            continue

        for ch in range(p):
            ch_arr = arr[:, ch]                             # (T,) with possible NaN
            ch_filled = _linear_impute(ch_arr)
            raw_obs = np.isfinite(ch_arr)

            obs_vals = ch_arr[raw_obs]
            if len(obs_vals) < 2:
                continue

            if normalization == "robust_series":
                loc = float(np.median(obs_vals))
                q75, q25 = float(np.percentile(obs_vals, 75)), float(np.percentile(obs_vals, 25))
                scale = max((q75 - q25) / 1.349, float(np.std(obs_vals)), 1e-6)
            else:
                loc, scale = 0.0, 1.0

            normed = (ch_filled - loc) / scale

            ends = list(range(context_length, T - horizon_length + 1, stride))
            valid_ends = [
                e for e in ends
                if raw_obs[e - context_length: e].mean() >= min_observed_fraction
                and raw_obs[e: e + horizon_length].any()
            ]
            if not valid_ends:
                continue

            if max_windows_per_series is not None and len(valid_ends) > max_windows_per_series:
                chosen = rng.choice(len(valid_ends), size=max_windows_per_series, replace=False)
                valid_ends = [valid_ends[i] for i in sorted(chosen)]

            for end in valid_ends:
                all_ctx.append(normed[end - context_length: end])
                all_tgt.append(normed[end: end + horizon_length])
                meta.append({
                    "series_idx": s_idx,
                    "channel_idx": ch,
                    "window_end": end,
                    "loc": loc,
                    "scale": scale,
                })

    if not all_ctx:
        p_ref = max(p_ref, 1)
        return (
            np.zeros((0, context_length)),
            np.zeros((0, horizon_length)),
            [],
            p_ref,
        )

    return np.vstack(all_ctx), np.vstack(all_tgt), meta, max(p_ref, 1)


# ---------------------------------------------------------------------------
# MultivariateNestedKrigingForecaster
# ---------------------------------------------------------------------------


class MultivariateNestedKrigingForecaster:
    """p-output nested kriging with a shared Gibbs lengthscale field.

    Architecture
    ~~~~~~~~~~~~
    *   One :class:`LengthscaleField` (``ell_field``) is shared across all p
        output channels.  It is estimated *once* on the pooled cross-channel
        window pool via :func:`fit_gibbs_lengthscale_field`.
    *   A separate :class:`NestedKrigingForecaster` is fitted for each channel
        using only the windows from that channel but querying the shared
        ``ell_field`` to obtain point-wise lengthscales.

    This factored design keeps the per-channel fit O(N_channel × block_size²)
    while leveraging all p channels to estimate a robust ℓ field.

    Parameters
    ----------
    ell_field  : fitted :class:`LengthscaleField` (shared across channels).
    n_channels : number of output channels p.
    block_size : target block size for each channel's NestedKrigingForecaster.
    max_blocks : maximum number of blocks per channel (None = unlimited).
    sigma_n    : noise std (same for all channels; tune per-channel if needed).
    amplitude  : Gibbs kernel amplitude σ_f².
    rng        : random generator; a fresh one is created if None.
    """

    def __init__(
        self,
        ell_field: LengthscaleField,
        *,
        n_channels: int,
        block_size: int = 64,
        max_blocks: int | None = 50,
        sigma_n: float = 0.1,
        amplitude: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.ell_field = ell_field
        self.n_channels = n_channels
        self._forecasters: list[NestedKrigingForecaster] = [
            NestedKrigingForecaster(
                ell_field,
                block_size=block_size,
                max_blocks=max_blocks,
                sigma_n=sigma_n,
                amplitude=amplitude,
                rng=np.random.default_rng(rng.integers(2**32)),
            )
            for _ in range(n_channels)
        ]

    def fit(
        self,
        all_contexts: np.ndarray,
        all_targets: np.ndarray,
        meta: list[dict],
    ) -> "MultivariateNestedKrigingForecaster":
        """Fit each channel's forecaster from the pooled window pool.

        Parameters
        ----------
        all_contexts : (N, L) pooled contexts from
                       :func:`pool_windows_from_multivariate_series`.
        all_targets  : (N, H) pooled targets.
        meta         : list of dicts with key ``'channel_idx'`` — used to
                       split rows by channel.

        Returns
        -------
        self
        """
        channel_idx = np.array([m["channel_idx"] for m in meta], dtype=int)
        for ch in range(self.n_channels):
            mask = channel_idx == ch
            if mask.sum() == 0:
                continue
            self._forecasters[ch].fit(all_contexts[mask], all_targets[mask])
        return self

    def predict(
        self,
        query_contexts: np.ndarray,
    ) -> np.ndarray:
        """Predict all p channels for each query window.

        Parameters
        ----------
        query_contexts : ``(n_q, p, L)`` or ``(p, L)`` (single query).
                         The slice ``query_contexts[q, i, :]`` is the
                         L-step context vector for query q, channel i.

        Returns
        -------
        predictions : ``(n_q, p, H)`` or ``(p, H)`` when input was ``(p, L)``.
        """
        Q = np.asarray(query_contexts, dtype=float)
        squeeze = Q.ndim == 2               # single (p, L) query
        if squeeze:
            Q = Q[None]                     # → (1, p, L)
        n_q, p, _ = Q.shape
        if p != self.n_channels:
            raise ValueError(
                f"query_contexts has {p} channels but forecaster has {self.n_channels}."
            )

        preds = np.empty((n_q, p, self._forecasters[0]._horizon_length), dtype=float)
        for ch in range(p):
            ctx_ch = Q[:, ch, :]            # (n_q, L)
            preds[:, ch, :] = self._forecasters[ch].predict(ctx_ch)

        return preds[0] if squeeze else preds

    def predict_with_uncertainty(
        self,
        query_contexts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance for all p channels.

        Returns
        -------
        means     : ``(n_q, p, H)``
        variances : ``(n_q, p)``  — scalar per (query, channel) from block agg.
        """
        Q = np.asarray(query_contexts, dtype=float)
        squeeze = Q.ndim == 2
        if squeeze:
            Q = Q[None]
        n_q, p, _ = Q.shape
        if p != self.n_channels:
            raise ValueError(
                f"query_contexts has {p} channels but forecaster has {self.n_channels}."
            )

        H = self._forecasters[0]._horizon_length
        means = np.empty((n_q, p, H), dtype=float)
        variances = np.empty((n_q, p), dtype=float)

        for ch in range(p):
            ctx_ch = Q[:, ch, :]
            mu, var = self._forecasters[ch].predict_with_uncertainty(ctx_ch)
            means[:, ch, :] = mu
            variances[:, ch] = var

        if squeeze:
            return means[0], variances[0]
        return means, variances

    # ------------------------------------------------------------------
    def save_state(
        self,
        path: "str | pathlib.Path",
        *,
        dataset_name: str = "",
    ) -> None:
        """Save cross-dataset state for all p channels into one ``.npz`` file.

        All channels' block centroids are concatenated; a ``channel_indices``
        array records which block belongs to which channel.

        Parameters
        ----------
        path         : output ``.npz`` file path.
        dataset_name : tag for bookkeeping.
        """
        import pathlib as _pathlib

        all_centroids: list[np.ndarray] = []
        all_preds:     list[np.ndarray] = []
        all_vars:      list[float]      = []
        all_chidx:     list[int]        = []
        L_ref = H_ref = None

        for ch, fc in enumerate(self._forecasters):
            if fc._fitted_blocks is None:
                continue
            fc._check_fitted()
            blocks = fc._fitted_blocks  # type: ignore[assignment]
            L = blocks[0].X.shape[1]
            H = fc._horizon_length
            if L_ref is None:
                L_ref, H_ref = L, H

            for blk in blocks:
                x_bar   = blk.X.mean(axis=0)
                ell_bar = float(fc.ell_field.predict_ell(x_bar[None, :])[0])
                k_c = gibbs_kernel_matrix(
                    x_bar[None, :], np.array([ell_bar]),
                    blk.X, blk.ell,
                    amplitude=fc.amplitude,
                ).ravel()
                alpha_c = np.linalg.solve(blk.chol.T, np.linalg.solve(blk.chol, k_c))

                all_centroids.append(x_bar)
                all_preds.append(k_c @ blk.dual)
                all_vars.append(max(float(fc.amplitude) - float(k_c @ alpha_c), 0.0))
                all_chidx.append(ch)

        if not all_centroids:
            raise RuntimeError("No fitted blocks found in any channel; cannot save state.")

        np.savez_compressed(
            _pathlib.Path(path),
            centroids=np.vstack(all_centroids),          # (M_total, L)
            centroid_preds=np.vstack(all_preds),         # (M_total, H)
            centroid_vars=np.array(all_vars, dtype=float),
            channel_indices=np.array(all_chidx, dtype=int),
            context_length=np.array(L_ref, dtype=int),
            horizon_length=np.array(H_ref, dtype=int),
            n_channels=np.array(self.n_channels, dtype=int),
            dataset_name=np.array(dataset_name),
        )


# ===========================================================================
# Cross-dataset nested kriging layer
# ===========================================================================

import pathlib as _pathlib


# ---------------------------------------------------------------------------
# ACF embedding
# ---------------------------------------------------------------------------

def acf_embedding(X: np.ndarray, d: int = 20) -> np.ndarray:
    """Map context windows to a d-dimensional sample-ACF feature vector.

    For each row x ∈ ℝ^L the sample autocorrelation at lag k is:

        ρ̂_k = Σ_{t=0}^{L-k-1} (x_t − x̄)(x_{t+k} − x̄)
               ─────────────────────────────────────────────
               Σ_{t=0}^{L-1}   (x_t − x̄)²

    The result lives in [−1, 1]^d and is well-defined for any context
    length L ≥ d+1.  Rows with near-zero variance return the zero vector.

    Parameters
    ----------
    X : (N, L) or (L,) array of context windows.
    d : number of ACF lags (embedding dimension).

    Returns
    -------
    phi : (N, d) or (d,) ACF embeddings.
    """
    squeeze = X.ndim == 1
    X = np.atleast_2d(np.asarray(X, dtype=float))
    N, L = X.shape
    d_eff = min(d, L - 1)           # can't exceed L-1 lags

    X_c = X - X.mean(axis=1, keepdims=True)              # (N, L) centred
    var = np.sum(X_c ** 2, axis=1)                        # (N,)

    phi = np.zeros((N, d_eff), dtype=float)
    for k in range(1, d_eff + 1):
        cov_k = np.sum(X_c[:, : L - k] * X_c[:, k:], axis=1)   # (N,)
        phi[:, k - 1] = np.where(var > 1e-12, cov_k / np.where(var > 1e-12, var, 1.0), 0.0)

    # Pad with zeros if d_eff < d (only when L is very short)
    if d_eff < d:
        phi = np.pad(phi, ((0, 0), (0, d - d_eff)))

    return phi[0] if squeeze else phi


# ---------------------------------------------------------------------------
# load_nk_state  (pair of NestedKrigingForecaster.save_state)
# ---------------------------------------------------------------------------

def load_nk_state(path: "str | _pathlib.Path") -> dict:
    """Load a saved nested-kriging model state.

    Accepts files written by either
    :meth:`NestedKrigingForecaster.save_state` or
    :meth:`MultivariateNestedKrigingForecaster.save_state`.

    Returns
    -------
    dict with keys:

    ``centroids``       : (M, L) block centroid context windows
    ``centroid_preds``  : (M, H) local predictions at each centroid
    ``centroid_vars``   : (M,) local GP variance at each centroid
    ``context_length``  : int
    ``horizon_length``  : int
    ``dataset_name``    : str
    ``channel_indices`` : (M,) int – only present in multivariate states
    ``n_channels``      : int     – only present in multivariate states
    """
    data = np.load(_pathlib.Path(path), allow_pickle=True)
    out: dict = {
        "centroids":      data["centroids"],
        "centroid_preds": data["centroid_preds"],
        "centroid_vars":  data["centroid_vars"],
        "context_length": int(data["context_length"]),
        "horizon_length": int(data["horizon_length"]),
        "dataset_name":   str(data["dataset_name"]),
    }
    if "channel_indices" in data:
        out["channel_indices"] = data["channel_indices"]
        out["n_channels"]      = int(data["n_channels"])
    return out


# ---------------------------------------------------------------------------
# CrossDatasetNestedKrigingForecaster
# ---------------------------------------------------------------------------

class CrossDatasetNestedKrigingForecaster:
    """Second-level Rullière aggregation across multiple per-dataset NK models.

    Each fitted :class:`NestedKrigingForecaster` (or
    :class:`MultivariateNestedKrigingForecaster`) contributes M_d block
    centroids to a global pool.  Because different datasets may have
    different context lengths L_d, all centroids are projected to a common
    d-dimensional ACF feature space via :func:`acf_embedding` before the
    cross-dataset kernel matrix is assembled.

    At prediction time the query context x* is ACF-embedded and the
    standard Rullière aggregation is applied in ℝ^d:

        ŷ(x*) = c(φ*)^T [C + D]⁻¹ μ̂

    where
      * C_{ij} = k_RBF(φ_i, φ_j)  — RBF kernel in ACF space
      * D_{ii} = σ̂²_i + σ_n       — local GP variance (noise floor)
      * μ̂_i   = ŷ_i(x̄_i)         — local block prediction at centroid i
      * φ*     = φ(x*)            — ACF embedding of the query

    Parameters
    ----------
    acf_dim      : dimension of the ACF embedding (number of lags).  Default 20.
    length_scale : RBF length-scale in ACF space.  ``None`` (default) triggers
                   the median-heuristic: ls = median pairwise distance.
    amplitude    : RBF amplitude σ_f² in ACF space.  Default 1.0.
    sigma_n      : noise floor added to ``centroid_vars`` for numerical
                   stability.  Default 1e-3.
    """

    def __init__(
        self,
        *,
        acf_dim: int = 20,
        length_scale: float | None = None,
        amplitude: float = 1.0,
        sigma_n: float = 1e-3,
    ) -> None:
        self.acf_dim        = int(acf_dim)
        self._length_scale  = length_scale
        self.amplitude      = float(amplitude)
        self.sigma_n        = float(sigma_n)

        # Set after fit()
        self._phi_centroids:  np.ndarray | None = None
        self._centroid_preds: np.ndarray | None = None
        self._max_H: int   = 0
        self._chol_CD: np.ndarray | None = None
        self.length_scale_: float = 1.0

    # ------------------------------------------------------------------
    def fit(
        self,
        states: list[dict],
    ) -> "CrossDatasetNestedKrigingForecaster":
        """Build the cross-dataset aggregation system from saved model states.

        Parameters
        ----------
        states : list of dicts returned by :func:`load_nk_state`.  Each must
                 contain ``centroids`` (M, L), ``centroid_preds`` (M, H), and
                 ``centroid_vars`` (M,).

        Returns
        -------
        self
        """
        if not states:
            raise ValueError("states must be a non-empty list.")

        d     = self.acf_dim
        max_H = max(s["horizon_length"] for s in states)

        phi_list  = []
        pred_list = []
        var_list  = []

        for s in states:
            centroids = np.asarray(s["centroids"],      dtype=float)   # (M, L)
            preds     = np.asarray(s["centroid_preds"], dtype=float)   # (M, H_d)
            cvars     = np.asarray(s["centroid_vars"],  dtype=float)   # (M,)

            phi = acf_embedding(centroids, d)                          # (M, d)

            # Pad predictions to max_H with constant extrapolation
            M, H_d = preds.shape
            padded = np.zeros((M, max_H), dtype=float)
            padded[:, :H_d] = preds
            if H_d < max_H:
                padded[:, H_d:] = preds[:, -1:]  # repeat last step

            phi_list.append(phi)
            pred_list.append(padded)
            var_list.append(cvars)

        self._phi_centroids  = np.vstack(phi_list)    # (M_total, d)
        self._centroid_preds = np.vstack(pred_list)   # (M_total, max_H)
        self._max_H          = max_H

        centroid_vars = np.concatenate(var_list)       # (M_total,)

        # Length-scale: median heuristic on pairwise ACF distances
        if self._length_scale is None:
            dists2 = _pairwise_sq_dists(self._phi_centroids, self._phi_centroids)
            positive = dists2[dists2 > 0]
            median_d2 = float(np.median(positive)) if positive.size > 0 else 1.0
            self.length_scale_ = float(np.sqrt(max(median_d2, _EPS)))
        else:
            self.length_scale_ = float(self._length_scale)

        # C = RBF(phi_centroids, phi_centroids) in ACF space
        C = _rbf_kernel(
            self._phi_centroids, self._phi_centroids,
            length_scale=self.length_scale_,
            amplitude=self.amplitude,
        )                                              # (M_total, M_total)

        D = centroid_vars + self.sigma_n               # noise floor
        self._chol_CD = _safe_cholesky(C + np.diag(D))

        return self

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self._chol_CD is None:
            raise RuntimeError("Call fit() before predict().")

    # ------------------------------------------------------------------
    def predict(
        self,
        query_context: np.ndarray,
        target_horizon: int | None = None,
    ) -> np.ndarray:
        """Predict for a single query context using cross-dataset aggregation.

        Parameters
        ----------
        query_context  : (L,) normalised context window from any dataset.
        target_horizon : output steps.  Defaults to the maximum horizon seen
                         during ``fit``.  Must not exceed that maximum.

        Returns
        -------
        prediction : (target_horizon,) normalised forecast.  De-normalise
                     using the target series' own location and scale.
        """
        self._check_fitted()
        if target_horizon is None:
            target_horizon = self._max_H
        if target_horizon > self._max_H:
            raise ValueError(
                f"target_horizon={target_horizon} > max_H={self._max_H} seen during fit."
            )

        phi_star = acf_embedding(
            np.asarray(query_context, dtype=float), self.acf_dim
        )                                              # (d,)

        c = _rbf_kernel(
            phi_star[None, :], self._phi_centroids,
            length_scale=self.length_scale_,
            amplitude=self.amplitude,
        ).ravel()                                      # (M_total,)

        # Rullière weights: w = (C + D)⁻¹ c
        w = np.linalg.solve(
            self._chol_CD.T,
            np.linalg.solve(self._chol_CD, c),
        )                                              # (M_total,)

        return w @ self._centroid_preds[:, :target_horizon]  # (target_horizon,)

    # ------------------------------------------------------------------
    def predict_batch(
        self,
        query_contexts: np.ndarray,
        target_horizon: int | None = None,
    ) -> np.ndarray:
        """Predict for multiple query contexts.

        Parameters
        ----------
        query_contexts : (N, L) context windows.
        target_horizon : steps to forecast.

        Returns
        -------
        predictions : (N, target_horizon).
        """
        self._check_fitted()
        Q = np.atleast_2d(np.asarray(query_contexts, dtype=float))
        return np.stack(
            [self.predict(Q[i], target_horizon) for i in range(Q.shape[0])],
            axis=0,
        )
