from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


EPS = 1e-8

KernelFamily = Literal[
    "rbf",
    "matern25",
    "polynomial",
    "laplace",
    "inverse_quadratic",
    "linear",
]
RepresentationMode = Literal["plain", "interleaved", "masked"]
NormalizationMode = Literal["none", "robust_series", "robust_context"]


@dataclass(frozen=True)
class WindowedForecastDataset:
    series_values: np.ndarray
    series_mask: np.ndarray
    series_deltas: np.ndarray
    contexts: np.ndarray
    targets: np.ndarray
    context_masks: np.ndarray
    target_masks: np.ndarray
    context_deltas: np.ndarray
    target_deltas: np.ndarray
    normalization: NormalizationMode
    window_locations: np.ndarray
    window_scales: np.ndarray

    @property
    def series_length(self) -> int:
        return int(self.series_values.shape[0])

    @property
    def n_windows(self) -> int:
        return int(self.contexts.shape[0])

    @property
    def context_length(self) -> int:
        return int(self.contexts.shape[1])

    @property
    def horizon_length(self) -> int:
        return int(self.targets.shape[1])


@dataclass(frozen=True)
class ObservedTimeSeries:
    values: np.ndarray
    observed_mask: np.ndarray
    deltas: np.ndarray

    @property
    def length(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class KernelConfig:
    family: KernelFamily = "rbf"
    amplitude: float = 1.0
    length_scale: float = 1.0
    degree: int = 2
    gamma: float | None = None
    coef0: float = 1.0
    mask_decay_gamma: float = 1.0


@dataclass(frozen=True)
class NormalizationStats:
    location: float = 0.0
    scale: float = 1.0


def _canonical_kernel_family(name: str) -> KernelFamily:
    normalized = name.lower().replace("-", "_").replace(".", "")
    aliases = {
        "rbf": "rbf",
        "gaussian": "rbf",
        "matern25": "matern25",
        "matern_25": "matern25",
        "matern_2_5": "matern25",
        "matern52": "matern25",
        "poly": "polynomial",
        "polynomial": "polynomial",
        "laplace": "laplace",
        "inverse_quadratic": "inverse_quadratic",
        "inversequadratic": "inverse_quadratic",
        "iq": "inverse_quadratic",
        "linear": "linear",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported kernel family '{name}'. "
            "Supported families are: rbf, matern25, polynomial, laplace, inverse_quadratic, linear."
        )
    return aliases[normalized]  # type: ignore[return-value]


def _ensure_2d(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D. Received shape {arr.shape}.")
    return arr


def _ensure_2d_bool(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=bool)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D. Received shape {arr.shape}.")
    return arr


def _validate_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive. Received {value}.")
    return value


def _validate_normalization_mode(mode: str) -> NormalizationMode:
    normalized = str(mode).lower()
    if normalized not in {"none", "robust_series", "robust_context"}:
        raise ValueError(
            f"Unsupported normalization mode '{mode}'. "
            "Supported modes are: none, robust_series, robust_context."
        )
    return normalized  # type: ignore[return-value]


def _validate_kernel_config(config: KernelConfig) -> KernelConfig:
    family = _canonical_kernel_family(config.family)
    if config.amplitude <= 0:
        raise ValueError(f"Kernel amplitude must be positive. Received {config.amplitude}.")
    if family in {"rbf", "matern25", "laplace", "inverse_quadratic"} and config.length_scale <= 0:
        raise ValueError(
            f"Kernel length_scale must be positive for family '{family}'. "
            f"Received {config.length_scale}."
        )
    if family == "polynomial" and config.degree < 0:
        raise ValueError(f"Polynomial degree must be non-negative. Received {config.degree}.")
    if not (0.0 < config.mask_decay_gamma <= 1.0):
        raise ValueError(
            "mask_decay_gamma must lie in the interval (0, 1]. "
            f"Received {config.mask_decay_gamma}."
        )
    return KernelConfig(
        family=family,
        amplitude=float(config.amplitude),
        length_scale=float(config.length_scale),
        degree=int(config.degree),
        gamma=None if config.gamma is None else float(config.gamma),
        coef0=float(config.coef0),
        mask_decay_gamma=float(config.mask_decay_gamma),
    )


def deltas_from_timestamps(
    timestamps: np.ndarray,
    *,
    initial_delta: float = 0.0,
) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=float).reshape(-1)
    if ts.ndim != 1:
        raise ValueError("timestamps must be 1D.")
    if ts.size == 0:
        return np.zeros(0, dtype=float)
    diffs = np.diff(ts)
    if diffs.size > 0 and np.any(diffs < 0):
        raise ValueError("timestamps must be non-decreasing.")
    return np.concatenate([[float(initial_delta)], diffs.astype(float)])


def build_regular_deltas(length: int, *, step: float = 1.0, initial_delta: float | None = None) -> np.ndarray:
    length = _validate_positive_int(length, "length")
    delta0 = float(step if initial_delta is None else initial_delta)
    deltas = np.full(length, float(step), dtype=float)
    deltas[0] = delta0
    return deltas


def build_gap_deltas_from_mask(
    observed_mask: np.ndarray,
    *,
    step: float = 1.0,
    initial_delta: float | None = None,
) -> np.ndarray:
    mask = np.asarray(observed_mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return np.zeros(0, dtype=float)
    if step <= 0:
        raise ValueError(f"step must be positive. Received {step}.")

    deltas = np.empty(mask.size, dtype=float)
    current_gap = float(step if initial_delta is None else initial_delta)
    for idx in range(mask.size):
        deltas[idx] = current_gap
        if mask[idx]:
            current_gap = float(step)
        else:
            current_gap += float(step)
    return deltas


def _prepare_series_and_mask(
    values: np.ndarray,
    *,
    observed_mask: np.ndarray | None = None,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if observed_mask is None:
        mask = np.isfinite(arr)
    else:
        mask = np.asarray(observed_mask, dtype=bool).reshape(-1)
        if mask.shape != arr.shape:
            raise ValueError(f"{name} and its observed_mask must have the same shape.")
        mask &= np.isfinite(arr)
    filled = np.where(mask, arr, 0.0)
    return filled, mask


def _prepare_deltas(
    deltas: np.ndarray | None,
    *,
    length: int,
    name: str,
) -> np.ndarray:
    if deltas is None:
        return build_regular_deltas(length)
    arr = np.asarray(deltas, dtype=float).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}. Received {arr.shape[0]}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative.")
    return arr


def prepare_observed_time_series(
    values: np.ndarray,
    *,
    observed_mask: np.ndarray | None = None,
    deltas: np.ndarray | None = None,
) -> ObservedTimeSeries:
    series, mask = _prepare_series_and_mask(
        values,
        observed_mask=observed_mask,
        name="values",
    )
    delta_arr = _prepare_deltas(deltas, length=series.shape[0], name="deltas")
    return ObservedTimeSeries(values=series, observed_mask=mask, deltas=delta_arr)


def robust_location_scale(
    values: np.ndarray,
    *,
    observed_mask: np.ndarray | None = None,
    min_scale: float = 1e-6,
) -> NormalizationStats:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if observed_mask is None:
        mask = np.isfinite(arr)
    else:
        mask = np.asarray(observed_mask, dtype=bool).reshape(-1)
        if mask.shape != arr.shape:
            raise ValueError("values and observed_mask must have the same shape.")
        mask &= np.isfinite(arr)

    observed = arr[mask]
    if observed.size == 0:
        return NormalizationStats(location=0.0, scale=1.0)

    loc = float(np.median(observed))
    q75, q25 = np.percentile(observed, [75.0, 25.0])
    iqr_scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    mad = float(np.median(np.abs(observed - loc)))
    mad_scale = 1.4826 * mad
    std_scale = float(np.std(observed))
    scale = max(min_scale, iqr_scale, mad_scale, std_scale)
    return NormalizationStats(location=loc, scale=scale)


def compute_normalization_stats(
    values: np.ndarray,
    *,
    observed_mask: np.ndarray | None = None,
    mode: NormalizationMode = "none",
    min_scale: float = 1e-6,
) -> NormalizationStats:
    validated_mode = _validate_normalization_mode(mode)
    if validated_mode == "none":
        return NormalizationStats(location=0.0, scale=1.0)
    return robust_location_scale(values, observed_mask=observed_mask, min_scale=min_scale)


def apply_normalization(
    values: np.ndarray,
    *,
    stats: NormalizationStats,
    observed_mask: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    transformed = (arr - float(stats.location)) / max(float(stats.scale), EPS)
    if observed_mask is None:
        return transformed
    mask = np.asarray(observed_mask, dtype=bool)
    if mask.shape != arr.shape:
        raise ValueError("values and observed_mask must have the same shape.")
    return np.where(mask, transformed, 0.0)


def invert_normalization(
    values: np.ndarray,
    *,
    stats: NormalizationStats,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr * float(stats.scale) + float(stats.location)


def rolling_window_end_indices(
    series_length: int,
    *,
    context_length: int,
    horizon_length: int,
    stride: int = 1,
    max_windows: int | None = None,
    include_last: bool = True,
) -> np.ndarray:
    series_length = int(series_length)
    context_length = _validate_positive_int(context_length, "context_length")
    horizon_length = _validate_positive_int(horizon_length, "horizon_length")
    stride = _validate_positive_int(stride, "stride")

    if context_length + horizon_length > series_length:
        raise ValueError(
            f"context_length + horizon_length = {context_length + horizon_length} exceeds "
            f"series length {series_length}."
        )

    last_end = series_length - horizon_length
    ends = np.arange(context_length, last_end + 1, stride, dtype=int)
    if include_last and ends[-1] != last_end:
        ends = np.append(ends, last_end)

    if max_windows is not None:
        max_windows = _validate_positive_int(max_windows, "max_windows")
        if ends.size > max_windows:
            keep = np.linspace(0, ends.size - 1, num=max_windows, dtype=int)
            ends = ends[np.unique(keep)]

    return ends


def make_windowed_forecasting_dataset(
    values: np.ndarray,
    *,
    context_length: int,
    horizon_length: int,
    observed_mask: np.ndarray | None = None,
    deltas: np.ndarray | None = None,
    stride: int = 1,
    max_windows: int | None = None,
    include_last: bool = True,
    drop_all_missing_targets: bool = True,
    normalization: NormalizationMode = "none",
) -> WindowedForecastDataset:
    context_length = _validate_positive_int(context_length, "context_length")
    horizon_length = _validate_positive_int(horizon_length, "horizon_length")
    normalization = _validate_normalization_mode(normalization)

    prepared = prepare_observed_time_series(
        values,
        observed_mask=observed_mask,
        deltas=deltas,
    )
    series = prepared.values
    mask = prepared.observed_mask
    delta_arr = prepared.deltas

    end_indices = rolling_window_end_indices(
        series.shape[0],
        context_length=context_length,
        horizon_length=horizon_length,
        stride=stride,
        max_windows=max_windows,
        include_last=include_last,
    )

    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    context_masks: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    context_deltas: list[np.ndarray] = []
    target_deltas: list[np.ndarray] = []
    window_locations: list[float] = []
    window_scales: list[float] = []

    series_stats = None
    if normalization == "robust_series":
        series_stats = compute_normalization_stats(
            series,
            observed_mask=mask,
            mode="robust_series",
        )

    for end_idx in end_indices:
        context_slice = slice(end_idx - context_length, end_idx)
        target_slice = slice(end_idx, end_idx + horizon_length)

        context_values = series[context_slice]
        target_values = series[target_slice]
        context_mask = mask[context_slice]
        target_mask = mask[target_slice]
        if drop_all_missing_targets and not target_mask.any():
            continue

        if normalization == "robust_context":
            stats = compute_normalization_stats(
                context_values,
                observed_mask=context_mask,
                mode="robust_context",
            )
        elif normalization == "robust_series":
            assert series_stats is not None
            stats = series_stats
        else:
            stats = NormalizationStats(location=0.0, scale=1.0)

        contexts.append(apply_normalization(context_values, stats=stats, observed_mask=context_mask))
        targets.append(apply_normalization(target_values, stats=stats, observed_mask=target_mask))
        context_masks.append(context_mask.copy())
        target_masks.append(target_mask.copy())
        context_deltas.append(delta_arr[context_slice].copy())
        target_deltas.append(delta_arr[target_slice].copy())
        window_locations.append(float(stats.location))
        window_scales.append(float(stats.scale))

    if not contexts:
        raise ValueError(
            "No valid windows could be formed. This can happen if the series is too short or "
            "if all candidate targets are missing and drop_all_missing_targets=True."
        )

    return WindowedForecastDataset(
        series_values=series.copy(),
        series_mask=mask.copy(),
        series_deltas=delta_arr.copy(),
        contexts=np.vstack(contexts),
        targets=np.vstack(targets),
        context_masks=np.vstack(context_masks),
        target_masks=np.vstack(target_masks),
        context_deltas=np.vstack(context_deltas),
        target_deltas=np.vstack(target_deltas),
        normalization=normalization,
        window_locations=np.asarray(window_locations, dtype=float),
        window_scales=np.asarray(window_scales, dtype=float),
    )


def build_interleaved_features(
    values: np.ndarray,
    deltas: np.ndarray,
    *,
    masks: np.ndarray | None = None,
) -> np.ndarray:
    X = _ensure_2d(values, "values")
    D = _ensure_2d(deltas, "deltas")
    if X.shape != D.shape:
        raise ValueError("values and deltas must have the same shape for interleaving.")

    if masks is None:
        M = np.isfinite(X)
    else:
        M = np.asarray(masks, dtype=bool)
        if M.shape != X.shape:
            raise ValueError("values and masks must have the same shape.")
        M &= np.isfinite(X)

    X_clean = np.where(M, X, 0.0)
    features = np.empty((X.shape[0], 2 * X.shape[1]), dtype=float)
    features[:, 0::2] = X_clean
    features[:, 1::2] = D
    return features


def build_observed_interleaved_features(
    values: np.ndarray,
    deltas: np.ndarray,
    *,
    masks: np.ndarray | None = None,
    max_observed_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    X = _ensure_2d(values, "values")
    D = _ensure_2d(deltas, "deltas")
    if X.shape != D.shape:
        raise ValueError("values and deltas must have the same shape for interleaving.")

    if masks is None:
        M = np.isfinite(X)
    else:
        M = np.asarray(masks, dtype=bool)
        if M.shape != X.shape:
            raise ValueError("values and masks must have the same shape.")
        M &= np.isfinite(X)

    observed_counts = M.sum(axis=1).astype(int)
    if max_observed_count is None:
        max_observed = int(observed_counts.max()) if observed_counts.size > 0 else 0
    else:
        max_observed = int(max_observed_count)
        if max_observed < 0:
            raise ValueError("max_observed_count must be non-negative.")
        if observed_counts.size > 0 and int(observed_counts.max()) > max_observed:
            raise ValueError(
                "max_observed_count is smaller than the number of observed entries in at least one sample."
            )

    features = np.zeros((X.shape[0], 2 * max_observed), dtype=float)
    feature_mask = np.zeros((X.shape[0], 2 * max_observed), dtype=bool)

    for row_idx in range(X.shape[0]):
        obs_idx = np.flatnonzero(M[row_idx])
        n_obs = int(obs_idx.size)
        if n_obs == 0:
            continue
        stop = 2 * n_obs
        features[row_idx, 0:stop:2] = X[row_idx, obs_idx]
        features[row_idx, 1:stop:2] = D[row_idx, obs_idx]
        feature_mask[row_idx, :stop] = True

    return features, feature_mask


def _resolve_gamma(config: KernelConfig, feature_dim: int) -> float:
    if config.gamma is not None:
        return float(config.gamma)
    return 1.0 / max(int(feature_dim), 1)


def _coordinate_decay_weights(length: int, *, decay_gamma: float) -> np.ndarray:
    length = _validate_positive_int(length, "length")
    if decay_gamma == 1.0:
        return np.ones(length, dtype=float)
    # Windows are stored oldest -> newest, so the most recent coordinate gets weight 1.0
    # and older coordinates are geometrically downweighted by powers of decay_gamma.
    powers = np.arange(length - 1, -1, -1, dtype=float)
    return decay_gamma ** powers


def _pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_sq = np.sum(X * X, axis=1, keepdims=True)
    Y_sq = np.sum(Y * Y, axis=1, keepdims=True).T
    sq_dists = X_sq + Y_sq - 2.0 * (X @ Y.T)
    return np.maximum(sq_dists, 0.0)


def _pairwise_l1_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)


def _pairwise_weighted_l1_dists(X: np.ndarray, Y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(X[:, None, :] - Y[None, :, :]) * weights[None, None, :], axis=2)


def build_feature_kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    config: KernelConfig = KernelConfig(),
) -> np.ndarray:
    config = _validate_kernel_config(config)
    A = _ensure_2d(X, "X")
    B = A if Y is None else _ensure_2d(Y, "Y")
    family = config.family
    coord_weights = _coordinate_decay_weights(A.shape[1], decay_gamma=config.mask_decay_gamma)
    sqrt_coord_weights = np.sqrt(coord_weights)

    if B.shape[1] != A.shape[1]:
        raise ValueError("X and Y must have the same feature dimension.")

    weighted_dot = (A * coord_weights) @ B.T

    if family == "linear":
        return config.amplitude * weighted_dot

    if family == "polynomial":
        gamma = _resolve_gamma(config, A.shape[1])
        return config.amplitude * (gamma * weighted_dot + config.coef0) ** config.degree

    if family == "rbf":
        sq_dists = _pairwise_sq_dists(A * sqrt_coord_weights, B * sqrt_coord_weights)
        return config.amplitude * np.exp(-0.5 * sq_dists / (config.length_scale ** 2))

    if family == "laplace":
        l1_dists = _pairwise_weighted_l1_dists(A, B, coord_weights)
        return config.amplitude * np.exp(-l1_dists / max(config.length_scale, EPS))

    if family == "inverse_quadratic":
        sq_dists = _pairwise_sq_dists(A * sqrt_coord_weights, B * sqrt_coord_weights)
        return config.amplitude / (1.0 + sq_dists / (config.length_scale ** 2))

    if family == "matern25":
        sq_dists = _pairwise_sq_dists(A * sqrt_coord_weights, B * sqrt_coord_weights)
        r = np.sqrt(sq_dists + EPS)
        scaled = np.sqrt(5.0) * r / max(config.length_scale, EPS)
        return config.amplitude * (1.0 + scaled + (scaled ** 2) / 3.0) * np.exp(-scaled)

    raise AssertionError(f"Unhandled kernel family '{family}'.")


def _masked_pair_kernel(
    x: np.ndarray,
    mask_x: np.ndarray,
    y: np.ndarray,
    mask_y: np.ndarray,
    *,
    config: KernelConfig,
) -> float:
    overlap = np.logical_and(mask_x, mask_y)
    overlap_count = int(overlap.sum())
    if overlap_count == 0:
        return 0.0

    coord_weights = _coordinate_decay_weights(x.shape[0], decay_gamma=config.mask_decay_gamma)
    active_weights = coord_weights[overlap]
    x_obs = x[overlap]
    y_obs = y[overlap]
    diff = x_obs - y_obs
    sqdist = float(np.dot(active_weights, diff * diff))
    l1dist = float(np.dot(active_weights, np.abs(diff)))
    dot = float(np.dot(active_weights, x_obs * y_obs))
    gamma = _resolve_gamma(config, overlap_count)

    if config.family == "linear":
        base = config.amplitude * dot
    elif config.family == "polynomial":
        base = config.amplitude * (gamma * dot + config.coef0) ** config.degree
    elif config.family == "rbf":
        base = config.amplitude * np.exp(-0.5 * sqdist / (config.length_scale ** 2))
    elif config.family == "laplace":
        base = config.amplitude * np.exp(-l1dist / max(config.length_scale, EPS))
    elif config.family == "inverse_quadratic":
        base = config.amplitude / (1.0 + sqdist / (config.length_scale ** 2))
    elif config.family == "matern25":
        r = float(np.sqrt(sqdist + EPS))
        scaled = np.sqrt(5.0) * r / max(config.length_scale, EPS)
        base = config.amplitude * (1.0 + scaled + (scaled ** 2) / 3.0) * np.exp(-scaled)
    else:
        raise AssertionError(f"Unhandled kernel family '{config.family}'.")

    return base


def build_masked_kernel_matrix(
    X: np.ndarray,
    masks_X: np.ndarray,
    Y: np.ndarray | None = None,
    masks_Y: np.ndarray | None = None,
    *,
    config: KernelConfig = KernelConfig(),
) -> np.ndarray:
    config = _validate_kernel_config(config)
    A = _ensure_2d(X, "X")
    M_A = np.asarray(masks_X, dtype=bool)
    if M_A.shape != A.shape:
        raise ValueError("X and masks_X must have the same shape.")

    if Y is None:
        B = A
        M_B = M_A
        symmetric = True
    else:
        B = _ensure_2d(Y, "Y")
        if masks_Y is None:
            raise ValueError("masks_Y must be provided when Y is provided.")
        M_B = np.asarray(masks_Y, dtype=bool)
        if M_B.shape != B.shape:
            raise ValueError("Y and masks_Y must have the same shape.")
        symmetric = False

    A_clean = np.where(M_A, A, 0.0)
    B_clean = np.where(M_B, B, 0.0)
    kernel = np.zeros((A.shape[0], B.shape[0]), dtype=float)

    if symmetric:
        for i in range(A.shape[0]):
            kernel[i, i] = _masked_pair_kernel(
                A_clean[i],
                M_A[i],
                B_clean[i],
                M_B[i],
                config=config,
            )
            for j in range(i + 1, B.shape[0]):
                value = _masked_pair_kernel(
                    A_clean[i],
                    M_A[i],
                    B_clean[j],
                    M_B[j],
                    config=config,
                )
                kernel[i, j] = value
                kernel[j, i] = value
        return kernel

    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            kernel[i, j] = _masked_pair_kernel(
                A_clean[i],
                M_A[i],
                B_clean[j],
                M_B[j],
                config=config,
            )

    return kernel


def build_window_kernel_matrix(
    contexts: np.ndarray,
    *,
    representation: RepresentationMode = "masked",
    config: KernelConfig = KernelConfig(),
    context_masks: np.ndarray | None = None,
    context_deltas: np.ndarray | None = None,
    other_contexts: np.ndarray | None = None,
    other_masks: np.ndarray | None = None,
    other_deltas: np.ndarray | None = None,
) -> np.ndarray:
    rep = representation.lower()
    X = _ensure_2d(contexts, "contexts")
    Y = None if other_contexts is None else _ensure_2d(other_contexts, "other_contexts")

    if rep == "plain":
        X_mask = np.isfinite(X) if context_masks is None else np.asarray(context_masks, dtype=bool)
        if X_mask.shape != X.shape:
            raise ValueError("contexts and context_masks must have the same shape.")
        X_clean = np.where(X_mask, X, 0.0)
        if Y is None:
            return build_feature_kernel_matrix(X_clean, config=config)
        Y_mask = np.isfinite(Y) if other_masks is None else np.asarray(other_masks, dtype=bool)
        if Y_mask.shape != Y.shape:
            raise ValueError("other_contexts and other_masks must have the same shape.")
        Y_clean = np.where(Y_mask, Y, 0.0)
        return build_feature_kernel_matrix(X_clean, Y_clean, config=config)

    if rep == "interleaved":
        if context_deltas is None:
            raise ValueError("context_deltas must be provided for representation='interleaved'.")
        X_mask = np.isfinite(X) if context_masks is None else np.asarray(context_masks, dtype=bool)
        if X_mask.shape != X.shape:
            raise ValueError("contexts and context_masks must have the same shape.")
        if Y is None:
            X_feat, X_feat_mask = build_observed_interleaved_features(
                X,
                context_deltas,
                masks=X_mask,
            )
            return build_masked_kernel_matrix(X_feat, X_feat_mask, config=config)
        if other_deltas is None:
            raise ValueError("other_deltas must be provided when using interleaved cross-kernels.")
        Y_mask = np.isfinite(Y) if other_masks is None else np.asarray(other_masks, dtype=bool)
        if Y_mask.shape != Y.shape:
            raise ValueError("other_contexts and other_masks must have the same shape.")
        max_obs_x = int(X_mask.sum(axis=1).max()) if X_mask.shape[0] > 0 else 0
        max_obs_y = int(Y_mask.sum(axis=1).max()) if Y_mask.shape[0] > 0 else 0
        max_obs = max(max_obs_x, max_obs_y)
        X_feat, X_feat_mask = build_observed_interleaved_features(
            X,
            context_deltas,
            masks=X_mask,
            max_observed_count=max_obs,
        )
        Y_feat, Y_feat_mask = build_observed_interleaved_features(
            Y,
            other_deltas,
            masks=Y_mask,
            max_observed_count=max_obs,
        )
        return build_masked_kernel_matrix(
            X_feat,
            X_feat_mask,
            Y_feat,
            Y_feat_mask,
            config=config,
        )

    if rep == "masked":
        if context_masks is None:
            raise ValueError("context_masks must be provided for representation='masked'.")
        if Y is None:
            return build_masked_kernel_matrix(X, context_masks, config=config)
        if other_masks is None:
            raise ValueError("other_masks must be provided for masked cross-kernels.")
        return build_masked_kernel_matrix(
            X,
            context_masks,
            Y,
            other_masks,
            config=config,
        )

    raise ValueError(
        f"Unsupported representation '{representation}'. "
        "Supported representations are: plain, interleaved, masked."
    )


def relative_l2_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if mask is None:
        obs = np.isfinite(truth)
    else:
        obs = np.asarray(mask, dtype=bool)
        if obs.shape != truth.shape:
            raise ValueError("mask must have the same shape as y_true.")
        obs &= np.isfinite(truth)

    if not obs.any():
        raise ValueError("relative_l2_error requires at least one observed target.")

    num = np.linalg.norm((pred - truth)[obs])
    denom = np.linalg.norm(truth[obs])
    return float(num / max(denom, EPS))


class WindowKernelRidgeForecaster:
    def __init__(
        self,
        *,
        kernel_config: KernelConfig = KernelConfig(),
        representation: RepresentationMode = "masked",
        regularization: float = 1e-3,
    ) -> None:
        self.kernel_config = _validate_kernel_config(kernel_config)
        self.representation = representation.lower()
        if self.representation not in {"plain", "interleaved", "masked"}:
            raise ValueError(
                f"Unsupported representation '{representation}'. "
                "Supported representations are: plain, interleaved, masked."
            )
        self.regularization = float(regularization)
        if self.regularization < 0:
            raise ValueError(f"regularization must be non-negative. Received {regularization}.")

        self._is_fitted = False

    def fit(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        *,
        context_masks: np.ndarray | None = None,
        target_masks: np.ndarray | None = None,
        context_deltas: np.ndarray | None = None,
    ) -> "WindowKernelRidgeForecaster":
        X = _ensure_2d(contexts, "contexts")
        Y = _ensure_2d(targets, "targets")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("contexts and targets must have the same number of rows.")

        if context_masks is None:
            X_mask = np.isfinite(X)
        else:
            X_mask = np.asarray(context_masks, dtype=bool)
            if X_mask.shape != X.shape:
                raise ValueError("contexts and context_masks must have the same shape.")
            X_mask &= np.isfinite(X)

        if target_masks is None:
            Y_mask = np.isfinite(Y)
        else:
            Y_mask = np.asarray(target_masks, dtype=bool)
            if Y_mask.shape != Y.shape:
                raise ValueError("targets and target_masks must have the same shape.")
            Y_mask &= np.isfinite(Y)

        X_clean = np.where(X_mask, X, 0.0)
        Y_clean = np.where(Y_mask, Y, 0.0)

        if context_deltas is None:
            X_deltas = np.tile(build_regular_deltas(X.shape[1]), (X.shape[0], 1))
        else:
            X_deltas = _ensure_2d(context_deltas, "context_deltas")
            if X_deltas.shape != X.shape:
                raise ValueError("contexts and context_deltas must have the same shape.")

        K = build_window_kernel_matrix(
            X_clean,
            representation=self.representation,  # type: ignore[arg-type]
            config=self.kernel_config,
            context_masks=X_mask,
            context_deltas=X_deltas,
        )

        self._train_contexts = X_clean
        self._train_context_masks = X_mask
        self._train_targets = Y_clean
        self._train_target_masks = Y_mask
        self._train_context_deltas = X_deltas
        self._kernel_train = K
        self._context_length = X.shape[1]
        self._horizon_length = Y.shape[1]

        regularized = K + self.regularization * np.eye(K.shape[0])
        fully_observed_steps = np.all(Y_mask, axis=0)

        self._dual_coef_full = None
        if fully_observed_steps.all():
            self._dual_coef_full = np.linalg.solve(regularized, Y_clean)
            self._step_supports = None
            self._step_dual_coefs = None
        else:
            self._step_supports = []
            self._step_dual_coefs = []
            for step in range(Y.shape[1]):
                observed = Y_mask[:, step]
                if not observed.any():
                    raise ValueError(
                        f"Target step {step} has no observed labels. "
                        "Cannot fit a KRR model for that horizon step."
                    )
                idx = np.flatnonzero(observed)
                subgram = regularized[np.ix_(idx, idx)]
                coef = np.linalg.solve(subgram, Y_clean[idx, step])
                self._step_supports.append(idx)
                self._step_dual_coefs.append(coef)

        self._is_fitted = True
        return self

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("The forecaster must be fitted before calling predict or forecast.")

    def _prepare_query(
        self,
        query_contexts: np.ndarray,
        *,
        query_masks: np.ndarray | None,
        query_deltas: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Q = _ensure_2d(query_contexts, "query_contexts")
        if Q.shape[1] != self._context_length:
            raise ValueError(
                f"Each query context must have length {self._context_length}. "
                f"Received {Q.shape[1]}."
            )

        if query_masks is None:
            Q_mask = np.isfinite(Q)
        else:
            Q_mask = _ensure_2d_bool(query_masks, "query_masks")
            if Q_mask.shape[0] == 1 and Q.shape[0] > 1:
                Q_mask = np.repeat(Q_mask, Q.shape[0], axis=0)
            if Q_mask.shape != Q.shape:
                raise ValueError("query_contexts and query_masks must have the same shape.")
            Q_mask &= np.isfinite(Q)

        Q_clean = np.where(Q_mask, Q, 0.0)

        if query_deltas is None:
            Q_deltas = np.tile(build_regular_deltas(self._context_length), (Q.shape[0], 1))
        else:
            Q_deltas = _ensure_2d(query_deltas, "query_deltas")
            if Q_deltas.shape[0] == 1 and Q.shape[0] > 1:
                Q_deltas = np.repeat(Q_deltas, Q.shape[0], axis=0)
            if Q_deltas.shape != Q.shape:
                raise ValueError("query_contexts and query_deltas must have the same shape.")

        return Q_clean, Q_mask, Q_deltas

    def predict(
        self,
        query_contexts: np.ndarray,
        *,
        query_masks: np.ndarray | None = None,
        query_deltas: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        Q_clean, Q_mask, Q_deltas = self._prepare_query(
            query_contexts,
            query_masks=query_masks,
            query_deltas=query_deltas,
        )

        K_query = build_window_kernel_matrix(
            Q_clean,
            representation=self.representation,  # type: ignore[arg-type]
            config=self.kernel_config,
            context_masks=Q_mask,
            context_deltas=Q_deltas,
            other_contexts=self._train_contexts,
            other_masks=self._train_context_masks,
            other_deltas=self._train_context_deltas,
        )

        if self._dual_coef_full is not None:
            return K_query @ self._dual_coef_full

        prediction = np.zeros((Q_clean.shape[0], self._horizon_length), dtype=float)
        assert self._step_supports is not None
        assert self._step_dual_coefs is not None
        for step, (idx, coef) in enumerate(zip(self._step_supports, self._step_dual_coefs)):
            prediction[:, step] = K_query[:, idx] @ coef
        return prediction

    def forecast(
        self,
        query_contexts: np.ndarray,
        *,
        strategy: Literal["direct", "chunked"] = "direct",
        total_horizon: int | None = None,
        chunk_length: int | None = None,
        query_masks: np.ndarray | None = None,
        query_deltas: np.ndarray | None = None,
        future_deltas: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        if strategy == "direct":
            prediction = self.predict(
                query_contexts,
                query_masks=query_masks,
                query_deltas=query_deltas,
            )
            if total_horizon is None:
                return prediction
            total_horizon = _validate_positive_int(total_horizon, "total_horizon")
            if total_horizon > self._horizon_length:
                raise ValueError(
                    f"Direct forecasting can return at most the fitted horizon {self._horizon_length}. "
                    f"Received total_horizon={total_horizon}."
                )
            return prediction[:, :total_horizon]

        if strategy != "chunked":
            raise ValueError("strategy must be either 'direct' or 'chunked'.")

        total_horizon = self._horizon_length if total_horizon is None else total_horizon
        total_horizon = _validate_positive_int(total_horizon, "total_horizon")
        chunk_length = self._horizon_length if chunk_length is None else chunk_length
        chunk_length = _validate_positive_int(chunk_length, "chunk_length")
        if chunk_length > self._horizon_length:
            raise ValueError(
                f"chunk_length={chunk_length} exceeds the fitted target horizon {self._horizon_length}."
            )

        Q_clean, Q_mask, Q_deltas = self._prepare_query(
            query_contexts,
            query_masks=query_masks,
            query_deltas=query_deltas,
        )

        if future_deltas is None:
            future_delta_arr = np.tile(build_regular_deltas(total_horizon), (Q_clean.shape[0], 1))
        else:
            future_delta_arr = _ensure_2d(future_deltas, "future_deltas")
            if future_delta_arr.shape[0] == 1 and Q_clean.shape[0] > 1:
                future_delta_arr = np.repeat(future_delta_arr, Q_clean.shape[0], axis=0)
            if future_delta_arr.shape != (Q_clean.shape[0], total_horizon):
                raise ValueError(
                    "future_deltas must have shape (n_queries, total_horizon) "
                    f"= {(Q_clean.shape[0], total_horizon)}."
                )
            if np.any(~np.isfinite(future_delta_arr)) or np.any(future_delta_arr < 0):
                raise ValueError("future_deltas must be finite and non-negative.")

        predictions: list[np.ndarray] = []
        produced = 0
        current_context = Q_clean.copy()
        current_mask = Q_mask.copy()
        current_deltas = Q_deltas.copy()

        while produced < total_horizon:
            step_horizon = min(chunk_length, total_horizon - produced)
            chunk_prediction = self.predict(
                current_context,
                query_masks=current_mask,
                query_deltas=current_deltas,
            )[:, :step_horizon]
            predictions.append(chunk_prediction)

            next_deltas = future_delta_arr[:, produced : produced + step_horizon]
            current_context = np.concatenate(
                [current_context[:, step_horizon:], chunk_prediction],
                axis=1,
            )
            current_mask = np.concatenate(
                [current_mask[:, step_horizon:], np.ones_like(chunk_prediction, dtype=bool)],
                axis=1,
            )
            current_deltas = np.concatenate(
                [current_deltas[:, step_horizon:], next_deltas],
                axis=1,
            )
            produced += step_horizon

        return np.concatenate(predictions, axis=1)
