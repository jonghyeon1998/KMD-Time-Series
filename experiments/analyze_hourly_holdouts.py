from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.kernel_preprocessing import (
    PreparedKernelSeries,
    build_gamma_gram,
    build_gamma_query,
    build_mama_kernel,
    inverse_transform_with_stats,
    prepare_dataset_bank,
    prepare_support_window_bank,
    predict_future_from_gamma,
    stack_context_bank,
    stack_future_bank,
)


HOURLY_DATASETS = [
    "LOOP_SEATTLE/H",
    "solar/H",
    "electricity/H",
    "M_DENSE/H",
    "SZ_TAXI/H",
    "kdd_cup_2018_with_missing/H",
]

CONTEXT_LENGTH = 50
HORIZON_LENGTH = 40
ALPHA = 1.0
OMEGA = 6.0
LAMBDA_K = 1e-2
LAMBDA_GAMMA = 1e-1
WINDOW_STRIDE = HORIZON_LENGTH
MAX_WINDOWS_PER_SERIES = 20
MAX_SERIES_PER_DATASET = 20
TRANSFORM = "asinh"
IMPUTATION_STRATEGY = "linear"
SEASONAL_PERIOD = 24
PERCENTILE_VALUES = list(range(5, 101, 5))


@dataclass
class ForecastResult:
    unseen_name: str
    prepared_unseen: PreparedKernelSeries
    pred_scaled: np.ndarray
    pred_raw: np.ndarray
    pred_last_raw: np.ndarray
    pred_seasonal_raw: np.ndarray
    pred_last_scaled: np.ndarray
    pred_seasonal_scaled: np.ndarray
    best_gammawx_pruned_pred_scaled: np.ndarray
    best_gammawx_pruned_pred_raw: np.ndarray
    gammawx_percentile_results: list[dict[str, float | int]]
    best_gammawx_percentile: dict[str, float | int]
    metrics: dict[str, float | int | str]


def relative_l2_error(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps))


def relative_mae(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs(pred - true)) / (np.mean(np.abs(true)) + eps))


def build_last_value_baseline(prepared: PreparedKernelSeries) -> tuple[np.ndarray, np.ndarray]:
    observed = prepared.context_raw[np.isfinite(prepared.context_raw)]
    if observed.size == 0:
        raw = np.zeros(prepared.horizon_length, dtype=float)
    else:
        raw = np.full(prepared.horizon_length, float(observed[-1]), dtype=float)
    scaled = prepared.context_filled[prepared.context_mask][-1] if prepared.context_mask.any() else 0.0
    scaled = np.full(prepared.horizon_length, float(scaled), dtype=float)
    return raw, scaled


def build_seasonal_baseline(
    prepared: PreparedKernelSeries,
    period: int = SEASONAL_PERIOD,
) -> tuple[np.ndarray, np.ndarray]:
    raw_context = prepared.context_raw
    scaled_context = prepared.context_filled
    raw_pred = np.zeros(prepared.horizon_length, dtype=float)
    scaled_pred = np.zeros(prepared.horizon_length, dtype=float)

    for step in range(prepared.horizon_length):
        idx = raw_context.shape[0] - period + (step % period)
        idx = max(0, min(idx, raw_context.shape[0] - 1))
        raw_value = raw_context[idx]
        if not np.isfinite(raw_value):
            observed = raw_context[np.isfinite(raw_context)]
            raw_value = float(observed[-1]) if observed.size else 0.0
        raw_pred[step] = float(raw_value)
        scaled_pred[step] = float(scaled_context[idx])

    return raw_pred, scaled_pred


def evaluate_holdout(unseen_name: str) -> ForecastResult:
    seen_names = [name for name in HOURLY_DATASETS if name != unseen_name]
    support = prepare_support_window_bank(
        seen_names,
        context_length=CONTEXT_LENGTH,
        horizon_length=HORIZON_LENGTH,
        window_stride=WINDOW_STRIDE,
        max_windows_per_series=MAX_WINDOWS_PER_SERIES,
        max_series_per_dataset=MAX_SERIES_PER_DATASET,
        transform=TRANSFORM,
        imputation_strategy=IMPUTATION_STRATEGY,
    )

    X, X_mask = stack_context_bank(support, include_masks=True)
    Y, _ = stack_future_bank(support, include_masks=True)
    Y_scaled = np.vstack([np.where(series.future_mask, series.future_scaled, 0.0) for series in support])
    Y_mask = np.vstack([series.future_mask.astype(bool) for series in support])
    X_scaled = np.vstack([series.context_scaled for series in support])
    _, unseen = prepare_dataset_bank(
        seen_names,
        unseen_name=unseen_name,
        context_length=CONTEXT_LENGTH,
        horizon_length=HORIZON_LENGTH,
        transform=TRANSFORM,
        imputation_strategy=IMPUTATION_STRATEGY,
    )
    assert unseen is not None

    K = build_mama_kernel(X, prototype_masks=X_mask, alpha=ALPHA, omega=OMEGA)
    Gamma_xx = build_gamma_gram(
        X_scaled,
        time_kernel=K,
        masks=X_mask,
        lambda_k=LAMBDA_K,
    )
    Gamma_wx = build_gamma_query(
        unseen.context_scaled,
        X_scaled,
        time_kernel=K,
        query_mask=unseen.context_mask,
        context_masks=X_mask,
        lambda_k=LAMBDA_K,
    )

    pred_scaled = predict_future_from_gamma(
        Gamma_wx,
        Gamma_xx,
        Y_scaled,
        lambda_gamma=LAMBDA_GAMMA,
        target_masks=Y_mask,
    )
    pred_raw = inverse_transform_with_stats(
        pred_scaled,
        loc=unseen.loc,
        scale=unseen.scale,
        transform=unseen.transform,
    )

    pred_last_raw, pred_last_scaled = build_last_value_baseline(unseen)
    pred_seasonal_raw, pred_seasonal_scaled = build_seasonal_baseline(unseen)

    observed_future = unseen.future_mask.astype(bool)
    true_scaled_obs = unseen.future_scaled[observed_future]
    true_raw_obs = unseen.future_raw[observed_future]

    pred_scaled_obs = pred_scaled[observed_future]
    pred_raw_obs = pred_raw[observed_future]
    pred_last_raw_obs = pred_last_raw[observed_future]
    pred_last_scaled_obs = pred_last_scaled[observed_future]
    pred_seasonal_raw_obs = pred_seasonal_raw[observed_future]
    pred_seasonal_scaled_obs = pred_seasonal_scaled[observed_future]

    gamma_diag = np.diag(Gamma_xx)
    valid_support = np.isfinite(gamma_diag) & (gamma_diag > 1e-12) & np.isfinite(Gamma_wx)
    valid_support_idx = np.where(valid_support)[0]
    Gamma_xx_valid = Gamma_xx[np.ix_(valid_support_idx, valid_support_idx)]
    Gamma_wx_valid = Gamma_wx[valid_support_idx]
    Y_valid = Y_scaled[valid_support_idx]
    Y_mask_valid = Y_mask[valid_support_idx]
    sorted_gammawx_idx = np.argsort(-np.abs(Gamma_wx_valid))

    gammawx_percentile_results: list[dict[str, float | int]] = []
    best_gammawx_pred_scaled = pred_scaled.copy()
    best_gammawx_pred_raw = pred_raw.copy()
    best_gammawx_percentile: dict[str, float | int] | None = None

    for pct in PERCENTILE_VALUES:
        k_pct = max(1, int(np.ceil(len(valid_support_idx) * pct / 100.0)))
        idx_local = sorted_gammawx_idx[:k_pct]
        pred_scaled_pct = predict_future_from_gamma(
            Gamma_wx_valid[idx_local],
            Gamma_xx_valid[np.ix_(idx_local, idx_local)],
            Y_valid[idx_local],
            lambda_gamma=LAMBDA_GAMMA,
            target_masks=Y_mask_valid[idx_local],
        )
        pred_raw_pct = inverse_transform_with_stats(
            pred_scaled_pct,
            loc=unseen.loc,
            scale=unseen.scale,
            transform=unseen.transform,
        )
        result_pct = {
            "top_percent_kept": int(pct),
            "k": int(k_pct),
            "scaled_relative_l2_error": relative_l2_error(
                pred_scaled_pct[observed_future], true_scaled_obs
            ),
            "unscaled_relative_l2_error": relative_l2_error(
                pred_raw_pct[observed_future], true_raw_obs
            ),
        }
        gammawx_percentile_results.append(result_pct)
        if (
            best_gammawx_percentile is None
            or float(result_pct["unscaled_relative_l2_error"])
            < float(best_gammawx_percentile["unscaled_relative_l2_error"])
        ):
            best_gammawx_percentile = result_pct
            best_gammawx_pred_scaled = pred_scaled_pct
            best_gammawx_pred_raw = pred_raw_pct

    assert best_gammawx_percentile is not None

    metrics = {
        "unseen_name": unseen_name,
        "support_window_count": int(len(support)),
        "context_observed_points": int(unseen.context_mask.sum()),
        "future_observed_points": int(observed_future.sum()),
        "context_missing_pct": float(100.0 * (~unseen.context_mask).mean()),
        "future_missing_pct": float(100.0 * (~unseen.future_mask).mean()),
        "loc": float(unseen.loc),
        "scale": float(unseen.scale),
        "true_raw_mean_abs": float(np.nanmean(np.abs(unseen.future_raw))),
        "true_raw_std": float(np.nanstd(unseen.future_raw)),
        "kernel_scaled_relative_l2_error": relative_l2_error(pred_scaled_obs, true_scaled_obs),
        "kernel_unscaled_relative_l2_error": relative_l2_error(pred_raw_obs, true_raw_obs),
        "kernel_scaled_relative_mae": relative_mae(pred_scaled_obs, true_scaled_obs),
        "kernel_unscaled_relative_mae": relative_mae(pred_raw_obs, true_raw_obs),
        "last_value_scaled_relative_l2_error": relative_l2_error(pred_last_scaled_obs, true_scaled_obs),
        "last_value_unscaled_relative_l2_error": relative_l2_error(pred_last_raw_obs, true_raw_obs),
        "seasonal24_scaled_relative_l2_error": relative_l2_error(
            pred_seasonal_scaled_obs, true_scaled_obs
        ),
        "seasonal24_unscaled_relative_l2_error": relative_l2_error(
            pred_seasonal_raw_obs, true_raw_obs
        ),
        "best_gammawx_pruned_percent": int(best_gammawx_percentile["top_percent_kept"]),
        "best_gammawx_pruned_k": int(best_gammawx_percentile["k"]),
        "best_gammawx_pruned_scaled_relative_l2_error": float(
            best_gammawx_percentile["scaled_relative_l2_error"]
        ),
        "best_gammawx_pruned_unscaled_relative_l2_error": float(
            best_gammawx_percentile["unscaled_relative_l2_error"]
        ),
    }

    return ForecastResult(
        unseen_name=unseen_name,
        prepared_unseen=unseen,
        pred_scaled=pred_scaled,
        pred_raw=pred_raw,
        pred_last_raw=pred_last_raw,
        pred_seasonal_raw=pred_seasonal_raw,
        pred_last_scaled=pred_last_scaled,
        pred_seasonal_scaled=pred_seasonal_scaled,
        best_gammawx_pruned_pred_scaled=best_gammawx_pred_scaled,
        best_gammawx_pruned_pred_raw=best_gammawx_pred_raw,
        gammawx_percentile_results=gammawx_percentile_results,
        best_gammawx_percentile=best_gammawx_percentile,
        metrics=metrics,
    )


def save_metrics_csv(results: list[ForecastResult], output_path: Path) -> None:
    rows = [result.metrics for result in results]
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_prediction_grid(results: list[ForecastResult], output_path: Path) -> None:
    cols = 2
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows), sharex=False)
    axes = np.asarray(axes).reshape(-1)

    for ax, result in zip(axes, results):
        unseen = result.prepared_unseen
        steps = np.arange(unseen.horizon_length)
        missing = ~unseen.future_mask.astype(bool)

        ax.plot(steps, unseen.future_raw, marker="o", label="true", linewidth=1.8)
        ax.plot(steps, result.pred_raw, marker="x", label="kernel", linewidth=1.5)
        ax.plot(steps, result.pred_last_raw, linestyle="--", label="last value", linewidth=1.2)
        ax.plot(steps, result.pred_seasonal_raw, linestyle=":", label="seasonal(24)", linewidth=1.5)
        if missing.any():
            ax.scatter(
                steps[missing],
                np.zeros(int(missing.sum()), dtype=float),
                marker="v",
                color="tab:red",
                s=25,
                label="missing future",
            )
        ax.set_title(
            (
                f"{result.unseen_name}\n"
                f"kernel raw rel-L2={result.metrics['kernel_unscaled_relative_l2_error']:.3f}, "
                f"obs={result.metrics['future_observed_points']}"
            )
        )
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("Original units")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, frameon=False)
    for ax in axes[len(results) :]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_metric_bar_chart(results: list[ForecastResult], output_path: Path) -> None:
    sorted_results = sorted(
        results,
        key=lambda result: float(result.metrics["kernel_unscaled_relative_l2_error"]),
    )
    names = [result.unseen_name for result in sorted_results]
    kernel = [float(result.metrics["kernel_unscaled_relative_l2_error"]) for result in sorted_results]
    last = [float(result.metrics["last_value_unscaled_relative_l2_error"]) for result in sorted_results]
    seasonal = [
        float(result.metrics["seasonal24_unscaled_relative_l2_error"]) for result in sorted_results
    ]

    x = np.arange(len(sorted_results))
    width = 0.28
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, kernel, width=width, label="kernel")
    ax.bar(x, last, width=width, label="last value")
    ax.bar(x + width, seasonal, width=width, label="seasonal(24)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Unscaled relative L2 error")
    ax.set_title("Leave-one-dataset-out hourly holdout comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_variability_scatter(results: list[ForecastResult], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for result in results:
        mean_abs = float(result.metrics["true_raw_mean_abs"])
        std = float(result.metrics["true_raw_std"])
        cv = std / max(mean_abs, 1e-8)
        error = float(result.metrics["kernel_unscaled_relative_l2_error"])
        missing = float(result.metrics["future_missing_pct"])
        ax.scatter(cv, error, s=70 + 6 * missing, alpha=0.8)
        ax.annotate(
            result.unseen_name,
            (cv, error),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Validation variability ratio (std / mean abs)")
    ax.set_ylabel("Kernel unscaled relative L2 error")
    ax.set_title("More variable hourly targets are much harder for the current kernel model")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_gammawx_percentile_grid(results: list[ForecastResult], output_path: Path) -> None:
    cols = 2
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.1 * rows), sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(-1)

    for ax, result in zip(axes, results):
        percentiles = [int(item["top_percent_kept"]) for item in result.gammawx_percentile_results]
        scaled = [float(item["scaled_relative_l2_error"]) for item in result.gammawx_percentile_results]
        unscaled = [float(item["unscaled_relative_l2_error"]) for item in result.gammawx_percentile_results]
        best_pct = int(result.best_gammawx_percentile["top_percent_kept"])
        best_idx = percentiles.index(best_pct)

        ax.plot(percentiles, scaled, marker="o", linewidth=1.8, label="scaled rel-L2")
        ax.plot(percentiles, unscaled, marker="s", linewidth=1.8, label="unscaled rel-L2")
        ax.scatter(
            [best_pct],
            [unscaled[best_idx]],
            color="tab:red",
            s=45,
            zorder=5,
            label="best unscaled",
        )
        ax.axvline(100, color="0.75", linestyle="--", linewidth=1.0)
        ax.set_title(
            f"{result.unseen_name}\nbest={best_pct}% kept, raw rel-L2={unscaled[best_idx]:.3f}"
        )
        ax.set_xlabel("Top |Gamma(w, x)| percentile kept")
        ax.set_ylabel("Relative L2 error")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3, frameon=False)
    for ax in axes[len(results) :]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_best_pruned_prediction_grid(results: list[ForecastResult], output_path: Path) -> None:
    cols = 2
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows), sharex=False)
    axes = np.asarray(axes).reshape(-1)

    for ax, result in zip(axes, results):
        unseen = result.prepared_unseen
        steps = np.arange(unseen.horizon_length)
        missing = ~unseen.future_mask.astype(bool)

        ax.plot(steps, unseen.future_raw, marker="o", label="true", linewidth=1.8)
        ax.plot(
            steps,
            result.best_gammawx_pruned_pred_raw,
            marker="x",
            label="best |Gamma_wx| pruned",
            linewidth=1.5,
        )
        ax.plot(steps, result.pred_raw, linestyle="--", label="full kernel", linewidth=1.2, alpha=0.9)
        if missing.any():
            ax.scatter(
                steps[missing],
                np.zeros(int(missing.sum()), dtype=float),
                marker="v",
                color="tab:red",
                s=25,
                label="missing future",
            )
        ax.set_title(
            (
                f"{result.unseen_name}\n"
                f"best={int(result.best_gammawx_percentile['top_percent_kept'])}% "
                f"(k={int(result.best_gammawx_percentile['k'])}), "
                f"raw rel-L2={float(result.best_gammawx_percentile['unscaled_relative_l2_error']):.3f}"
            )
        )
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("Original units")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, frameon=False)
    for ax in axes[len(results) :]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    output_dir = Path("/Users/jonghyeonlee/KMD_TS/experiments/hourly_holdout_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [evaluate_holdout(unseen_name) for unseen_name in HOURLY_DATASETS]
    save_metrics_csv(results, output_dir / "hourly_holdout_metrics.csv")
    save_prediction_grid(results, output_dir / "hourly_holdout_predictions.png")
    save_metric_bar_chart(results, output_dir / "hourly_holdout_unscaled_l2_bar.png")
    save_variability_scatter(results, output_dir / "hourly_holdout_variability_vs_error.png")
    save_gammawx_percentile_grid(results, output_dir / "hourly_holdout_gammawx_percentile_l2.png")
    save_best_pruned_prediction_grid(
        results, output_dir / "hourly_holdout_best_gammawx_pruned_predictions.png"
    )

    rows = [result.metrics for result in results]
    best = min(rows, key=lambda row: float(row["kernel_unscaled_relative_l2_error"]))
    worst = max(rows, key=lambda row: float(row["kernel_unscaled_relative_l2_error"]))
    print("best_kernel_unscaled", best)
    print("worst_kernel_unscaled", worst)
    for row in sorted(rows, key=lambda item: float(item["kernel_unscaled_relative_l2_error"])):
        print(row)


if __name__ == "__main__":
    main()
