from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from gift_eval.data import Dataset


EPS = 1e-8


@dataclass(frozen=True)
class PreparedKernelSeries:
    name: str
    raw_train: np.ndarray
    raw_future: np.ndarray
    context_raw: np.ndarray
    future_raw: np.ndarray
    context_mask: np.ndarray
    future_mask: np.ndarray
    context_scaled: np.ndarray
    future_scaled: np.ndarray
    context_filled: np.ndarray
    future_filled: np.ndarray
    loc: float
    scale: float
    transform: str

    @property
    def context_length(self) -> int:
        return int(self.context_raw.shape[0])

    @property
    def horizon_length(self) -> int:
        return int(self.future_raw.shape[0])


def _nth_item(data_it: Iterable[dict], index: int) -> dict:
    for i, item in enumerate(data_it):
        if i == index:
            return item
    raise IndexError(f"Series index {index} is out of range.")


def _as_1d_array(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1D after loading. Received shape {arr.shape}. "
            "Use Dataset(..., to_univariate=True) or select a single channel first."
        )
    return arr


def load_train_validation_series(
    dataset_name: str,
    *,
    term: str = "short",
    series_index: int = 0,
    to_univariate: bool = False,
    storage_env_var: str = "GIFT_EVAL",
) -> tuple[np.ndarray, np.ndarray]:
    dataset = Dataset(
        name=dataset_name,
        term=term,
        to_univariate=to_univariate,
        storage_env_var=storage_env_var,
    )
    train_entry = _nth_item(dataset.training_dataset, series_index)
    val_entry = _nth_item(dataset.validation_dataset, series_index)

    train = _as_1d_array(train_entry["target"], f"{dataset_name} training target")
    val_full = _as_1d_array(val_entry["target"], f"{dataset_name} validation target")
    future = val_full[len(train) :]
    return train, future


def load_training_series_collection(
    dataset_name: str,
    *,
    term: str = "short",
    to_univariate: bool = False,
    storage_env_var: str = "GIFT_EVAL",
    max_series: int | None = None,
) -> list[tuple[int, np.ndarray]]:
    dataset = Dataset(
        name=dataset_name,
        term=term,
        to_univariate=to_univariate,
        storage_env_var=storage_env_var,
    )

    series_collection: list[tuple[int, np.ndarray]] = []
    for series_index, entry in enumerate(dataset.training_dataset):
        if max_series is not None and series_index >= max_series:
            break
        series_collection.append(
            (
                series_index,
                _as_1d_array(entry["target"], f"{dataset_name} training target[{series_index}]"),
            )
        )

    return series_collection


def infer_common_context_length(
    train_series: Sequence[np.ndarray],
    requested_length: int | None = None,
    *,
    round_down_to_power_of_two: bool = False,
) -> int:
    max_common_length = min(int(np.asarray(series).shape[0]) for series in train_series)
    if requested_length is None:
        context_length = max_common_length
    else:
        if requested_length > max_common_length:
            raise ValueError(
                f"Requested context_length={requested_length} exceeds the common "
                f"available prefix length {max_common_length}."
            )
        context_length = int(requested_length)

    if round_down_to_power_of_two and context_length > 0:
        context_length = 2 ** int(np.floor(np.log2(context_length)))

    return context_length


def infer_common_rolling_context_length(
    train_series: Sequence[np.ndarray],
    *,
    horizon_length: int,
    requested_length: int | None = None,
    round_down_to_power_of_two: bool = False,
) -> int:
    max_common_length = min(int(np.asarray(series).shape[0]) - horizon_length for series in train_series)
    if max_common_length <= 0:
        raise ValueError(
            "At least one training series is too short to provide a context/future split "
            f"with horizon_length={horizon_length}."
        )

    if requested_length is None:
        context_length = max_common_length
    else:
        if requested_length > max_common_length:
            raise ValueError(
                f"Requested context_length={requested_length} exceeds the common "
                f"rolling-window limit {max_common_length} for horizon_length={horizon_length}."
            )
        context_length = int(requested_length)

    if round_down_to_power_of_two and context_length > 0:
        context_length = 2 ** int(np.floor(np.log2(context_length)))

    return context_length


def robust_location_scale(
    values: np.ndarray,
    *,
    min_scale: float = 1e-6,
) -> tuple[float, float]:
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        return 0.0, 1.0

    loc = float(np.median(observed))
    q25, q75 = np.quantile(observed, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    mad_scale = float(1.4826 * np.median(np.abs(observed - loc)))
    std_scale = float(np.std(observed))
    scale = max(iqr_scale, mad_scale, std_scale, min_scale)
    return loc, scale


def transform_with_stats(
    values: np.ndarray,
    *,
    loc: float,
    scale: float,
    transform: str = "asinh",
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    centered = (arr - loc) / max(scale, EPS)

    if transform == "identity":
        return centered
    if transform == "asinh":
        return np.arcsinh(centered)

    raise ValueError(f"Unsupported transform '{transform}'.")


def inverse_transform_with_stats(
    values: np.ndarray,
    *,
    loc: float,
    scale: float,
    transform: str = "asinh",
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if transform == "identity":
        return loc + scale * arr
    if transform == "asinh":
        return loc + scale * np.sinh(arr)
    raise ValueError(f"Unsupported transform '{transform}'.")


def impute_missing_1d(
    values: np.ndarray,
    *,
    strategy: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float).copy()
    observed_mask = np.isfinite(arr)

    if observed_mask.all():
        return arr, observed_mask
    if not observed_mask.any():
        return np.zeros_like(arr), observed_mask

    if strategy == "zero":
        arr[~observed_mask] = 0.0
        return arr, observed_mask

    if strategy == "ffill":
        last = arr[observed_mask][0]
        for i in range(arr.shape[0]):
            if observed_mask[i]:
                last = arr[i]
            else:
                arr[i] = last
        return arr, observed_mask

    if strategy != "linear":
        raise ValueError(f"Unsupported imputation strategy '{strategy}'.")

    x = np.arange(arr.shape[0], dtype=float)
    arr[~observed_mask] = np.interp(x[~observed_mask], x[observed_mask], arr[observed_mask])
    return arr, observed_mask


def prepare_kernel_series(
    train_values: np.ndarray,
    future_values: np.ndarray,
    *,
    name: str = "",
    context_length: int,
    horizon_length: int | None = None,
    transform: str = "asinh",
    imputation_strategy: str = "linear",
    min_scale: float = 1e-6,
) -> PreparedKernelSeries:
    train = _as_1d_array(train_values, f"{name or 'series'} training values")
    future = _as_1d_array(future_values, f"{name or 'series'} future values")

    if context_length > train.shape[0]:
        raise ValueError(
            f"context_length={context_length} exceeds training length {train.shape[0]} for {name}."
        )

    if horizon_length is None:
        horizon_length = future.shape[0]
    if horizon_length > future.shape[0]:
        raise ValueError(
            f"horizon_length={horizon_length} exceeds future length {future.shape[0]} for {name}."
        )

    context_raw = train[-context_length:].copy()
    future_raw = future[:horizon_length].copy()

    loc, scale = robust_location_scale(context_raw, min_scale=min_scale)
    context_scaled = transform_with_stats(context_raw, loc=loc, scale=scale, transform=transform)
    future_scaled = transform_with_stats(future_raw, loc=loc, scale=scale, transform=transform)

    context_filled, context_mask = impute_missing_1d(
        context_scaled, strategy=imputation_strategy
    )
    future_filled, future_mask = impute_missing_1d(
        future_scaled, strategy=imputation_strategy
    )

    return PreparedKernelSeries(
        name=name,
        raw_train=train,
        raw_future=future,
        context_raw=context_raw,
        future_raw=future_raw,
        context_mask=context_mask,
        future_mask=future_mask,
        context_scaled=context_scaled,
        future_scaled=future_scaled,
        context_filled=context_filled,
        future_filled=future_filled,
        loc=loc,
        scale=scale,
        transform=transform,
    )


def prepare_dataset_bank(
    dataset_names: Sequence[str],
    *,
    unseen_name: str | None = None,
    term: str = "short",
    series_index: int = 0,
    to_univariate: bool = False,
    context_length: int | None = None,
    horizon_length: int | None = None,
    transform: str = "asinh",
    imputation_strategy: str = "linear",
    round_down_to_power_of_two: bool = False,
    storage_env_var: str = "GIFT_EVAL",
) -> tuple[list[PreparedKernelSeries], PreparedKernelSeries | None]:
    train_pairs: list[tuple[str, np.ndarray, np.ndarray]] = []

    for name in dataset_names:
        train, future = load_train_validation_series(
            name,
            term=term,
            series_index=series_index,
            to_univariate=to_univariate,
            storage_env_var=storage_env_var,
        )
        train_pairs.append((name, train, future))

    unseen_pair: tuple[str, np.ndarray, np.ndarray] | None = None
    if unseen_name is not None:
        train, future = load_train_validation_series(
            unseen_name,
            term=term,
            series_index=series_index,
            to_univariate=to_univariate,
            storage_env_var=storage_env_var,
        )
        unseen_pair = (unseen_name, train, future)

    train_lengths = [train.shape[0] for _, train, _ in train_pairs]
    if unseen_pair is not None:
        train_lengths.append(unseen_pair[1].shape[0])

    context_length = infer_common_context_length(
        [np.empty(length, dtype=float) for length in train_lengths],
        requested_length=context_length,
        round_down_to_power_of_two=round_down_to_power_of_two,
    )

    prepared_train = [
        prepare_kernel_series(
            train,
            future,
            name=name,
            context_length=context_length,
            horizon_length=horizon_length,
            transform=transform,
            imputation_strategy=imputation_strategy,
        )
        for name, train, future in train_pairs
    ]

    prepared_unseen = None
    if unseen_pair is not None:
        prepared_unseen = prepare_kernel_series(
            unseen_pair[1],
            unseen_pair[2],
            name=unseen_pair[0],
            context_length=context_length,
            horizon_length=horizon_length,
            transform=transform,
            imputation_strategy=imputation_strategy,
        )

    return prepared_train, prepared_unseen


def stack_context_bank(
    prepared_series: Sequence[PreparedKernelSeries],
    *,
    include_masks: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    contexts = np.vstack([series.context_filled for series in prepared_series])
    if not include_masks:
        return contexts, None
    masks = np.vstack([series.context_mask.astype(float) for series in prepared_series])
    return contexts, masks


def stack_future_bank(
    prepared_series: Sequence[PreparedKernelSeries],
    *,
    include_masks: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    futures = np.vstack([series.future_filled for series in prepared_series])
    if not include_masks:
        return futures, None
    masks = np.vstack([series.future_mask.astype(float) for series in prepared_series])
    return futures, masks


def normalized_time_grid(length: int) -> np.ndarray:
    if length < 2:
        return np.zeros(length, dtype=float)
    return np.linspace(0.0, 1.0, length, dtype=float)


def _uniform_grid_shift_steps(
    time_grid: np.ndarray,
    tau_grid: np.ndarray,
    *,
    atol: float = 1e-9,
) -> list[int] | None:
    if time_grid.ndim != 1 or tau_grid.ndim != 1:
        return None
    if time_grid.size == 0 or tau_grid.size == 0:
        return []
    if time_grid.size == 1:
        return [0 for _ in tau_grid]

    step = float(time_grid[1] - time_grid[0])
    if not np.allclose(np.diff(time_grid), step, atol=atol, rtol=0.0):
        return None

    origin = float(time_grid[0])
    shift_steps: list[int] = []
    for tau in tau_grid:
        raw_shift = (float(tau) - origin) / step
        rounded_shift = int(np.round(raw_shift))
        if not np.isclose(raw_shift, rounded_shift, atol=atol, rtol=0.0):
            return None
        shift_steps.append(rounded_shift)

    return shift_steps


def _shift_with_zeros(values: np.ndarray, shift_steps: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    shifted = np.zeros_like(arr)
    if shift_steps == 0:
        shifted[:] = arr
        return shifted

    if shift_steps > 0:
        shifted[shift_steps:] = arr[:-shift_steps]
        return shifted

    shifted[:shift_steps] = arr[-shift_steps:]
    return shifted


def build_mama_kernel(
    prototype_bank: np.ndarray,
    *,
    prototype_masks: np.ndarray | None = None,
    time_grid: np.ndarray | None = None,
    n_tau: np.ndarray | None = None,
    alpha: float = 1.0,
    omega: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    prototypes = np.asarray(prototype_bank, dtype=float)
    if prototypes.ndim != 2:
        raise ValueError("prototype_bank must have shape (n_series, context_length).")

    _, context_length = prototypes.shape
    if time_grid is None:
        time_grid = normalized_time_grid(context_length)

    time_grid = np.asarray(time_grid, dtype=float)
    kernel = np.zeros((context_length, context_length), dtype=float)
    masks = None
    if prototype_masks is not None:
        masks = np.asarray(prototype_masks, dtype=float)
        if masks.shape != prototypes.shape:
            raise ValueError("prototype_masks must have the same shape as prototype_bank.")

    #shift_steps = _uniform_grid_shift_steps(time_grid, tau_grid)

    for prototype_index, prototype in enumerate(prototypes):
        observed_prototype = prototype
        prototype_mask = None
        if masks is not None:
            prototype_mask = masks[prototype_index]
            observed_prototype = prototype * prototype_mask

        for tau_index, tau in enumerate(tau_grid):
            if shift_steps is not None:
                shifted = _shift_with_zeros(observed_prototype, shift_steps[tau_index])
                shifted_mask = (
                    _shift_with_zeros(prototype_mask, shift_steps[tau_index])
                    if prototype_mask is not None
                    else None
                )
            else:
                shifted = np.interp(time_grid - tau, time_grid, observed_prototype, left=0.0, right=0.0)
                shifted_mask = None
                if prototype_mask is not None:
                    shifted_mask = np.interp(
                        time_grid - tau,
                        time_grid,
                        prototype_mask,
                        left=0.0,
                        right=0.0,
                    )

            if shifted_mask is not None:
                shifted = shifted * shifted_mask
            chi = np.exp(-alpha * omega**2 * np.square(time_grid - tau)) * shifted
            kernel += np.outer(chi, chi)

    if normalize and prototypes.shape[0] > 0 and n_tau.shape[0] > 0:
        kernel /= float(prototypes.shape[0] * sum(n_tau).shape[0])

    return kernel


def build_gamma_gram(
    contexts: np.ndarray,
    *,
    time_kernel: np.ndarray,
    lambda_k: float = 1e-3,
    masks: np.ndarray | None = None,
) -> np.ndarray:
    X = np.asarray(contexts, dtype=float)
    if masks is not None:
        mask_arr = np.asarray(masks, dtype=bool)
        if mask_arr.shape != X.shape:
            raise ValueError("contexts and masks must have the same shape.")
        X = np.where(mask_arr, X, 0.0)

    regularized = np.asarray(time_kernel, dtype=float) + lambda_k * np.eye(time_kernel.shape[0])
    solved = np.linalg.solve(regularized, X.T)
    return X @ solved


def build_gamma_query(
    query_context: np.ndarray,
    contexts: np.ndarray,
    *,
    time_kernel: np.ndarray,
    lambda_k: float = 1e-3,
    query_mask: np.ndarray | None = None,
    context_masks: np.ndarray | None = None,
) -> np.ndarray:
    q = np.asarray(query_context, dtype=float).reshape(1, -1)
    X = np.asarray(contexts, dtype=float)

    if query_mask is not None:
        query_mask_arr = np.asarray(query_mask, dtype=bool).reshape(1, -1)
        if query_mask_arr.shape != q.shape:
            raise ValueError("query_context and query_mask must have the same shape.")
        q = np.where(query_mask_arr, q, 0.0)
    if context_masks is not None:
        context_mask_arr = np.asarray(context_masks, dtype=bool)
        if context_mask_arr.shape != X.shape:
            raise ValueError("contexts and context_masks must have the same shape.")
        X = np.where(context_mask_arr, X, 0.0)

    regularized = np.asarray(time_kernel, dtype=float) + lambda_k * np.eye(time_kernel.shape[0])
    solved = np.linalg.solve(regularized, X.T)
    return (q @ solved).reshape(-1)


def predict_future_from_gamma(
    gamma_query: np.ndarray,
    gamma_gram: np.ndarray,
    future_targets: np.ndarray,
    *,
    lambda_gamma: float = 1e-3,
    target_masks: np.ndarray | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    g = np.asarray(gamma_query, dtype=float).reshape(-1)
    G = np.asarray(gamma_gram, dtype=float)
    Y = np.asarray(future_targets, dtype=float)

    squeeze_output = False
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
        squeeze_output = True

    if G.shape[0] != G.shape[1]:
        raise ValueError("gamma_gram must be square.")
    if g.shape[0] != G.shape[0] or Y.shape[0] != G.shape[0]:
        raise ValueError("gamma_query, gamma_gram, and future_targets must agree on support size.")

    if target_masks is None:
        prediction = g @ np.linalg.solve(G + lambda_gamma * np.eye(G.shape[0]), Y)
        return prediction.reshape(-1) if squeeze_output else prediction

    M = np.asarray(target_masks, dtype=bool)
    if M.shape != Y.shape:
        raise ValueError("future_targets and target_masks must have the same shape.")

    prediction = np.full(Y.shape[1], fill_value, dtype=float)
    cached_subgrams: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    for step in range(Y.shape[1]):
        observed = M[:, step]
        if not observed.any():
            continue

        key = observed.tobytes()
        if key not in cached_subgrams:
            idx = np.flatnonzero(observed)
            subgram = G[np.ix_(idx, idx)] + lambda_gamma * np.eye(idx.size)
            cached_subgrams[key] = (idx, subgram)

        idx, subgram = cached_subgrams[key]
        prediction[step] = float(g[idx] @ np.linalg.solve(subgram, Y[idx, step]))

    return prediction[0] if squeeze_output else prediction


def summarize_prepared_bank(
    prepared_series: Sequence[PreparedKernelSeries],
) -> list[dict[str, float | int | str]]:
    summary: list[dict[str, float | int | str]] = []
    for series in prepared_series:
        summary.append(
            {
                "name": series.name,
                "context_length": series.context_length,
                "horizon_length": series.horizon_length,
                "context_missing_pct": float(100.0 * (~series.context_mask).mean()),
                "future_missing_pct": float(100.0 * (~series.future_mask).mean()),
                "loc": float(series.loc),
                "scale": float(series.scale),
            }
        )
    return summary


def prepare_support_bank(
    dataset_names: Sequence[str],
    *,
    term: str = "short",
    to_univariate: bool = False,
    context_length: int | None = None,
    horizon_length: int,
    transform: str = "asinh",
    imputation_strategy: str = "linear",
    round_down_to_power_of_two: bool = False,
    max_series_per_dataset: int | None = None,
    storage_env_var: str = "GIFT_EVAL",
) -> list[PreparedKernelSeries]:
    all_series: list[tuple[str, int, np.ndarray]] = []

    for dataset_name in dataset_names:
        series_collection = load_training_series_collection(
            dataset_name,
            term=term,
            to_univariate=to_univariate,
            storage_env_var=storage_env_var,
            max_series=max_series_per_dataset,
        )
        for series_index, series in series_collection:
            all_series.append((dataset_name, series_index, series))

    if not all_series:
        return []

    context_length = infer_common_rolling_context_length(
        [series for _, _, series in all_series],
        horizon_length=horizon_length,
        requested_length=context_length,
        round_down_to_power_of_two=round_down_to_power_of_two,
    )

    prepared_support: list[PreparedKernelSeries] = []
    for dataset_name, series_index, series in all_series:
        train, val = load_train_validation_series(
            dataset_name,
            term=term,
            series_index=series_index,
            to_univariate= False,
            storage_env_var= "GIFT_EVAL",
        ) 
        prepared_support.append(
            prepare_kernel_series(
                train,
                val,
                name=f"{dataset_name}#series{series_index}",
                context_length=context_length,
                horizon_length=horizon_length,
                transform=transform,
                imputation_strategy=imputation_strategy,
            )
        )

    return prepared_support
