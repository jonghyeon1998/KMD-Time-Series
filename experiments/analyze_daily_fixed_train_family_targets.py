from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gift_eval.data import Dataset
from src.kernel_preprocessing import (
    PreparedKernelSeries,
    _as_1d_array,
    _tau_localization_weights,
    build_auto_tau_array,
    build_gamma_gram,
    build_gamma_query,
    inverse_transform_with_stats,
    predict_future_from_gamma,
    prepare_kernel_series,
)


TRAIN_FAMILIES = [
    "covid_deaths",
    "electricity/D",
    "hierarchical_sales/D",
    "LOOP_SEATTLE/D",
    "temperature_rain_with_missing",
]
TEMPERATURE_FAMILY = "temperature_rain_with_missing"
TARGET_DATASETS = ["saugeenday/D", "solar/D", "us_births/D"]

CONTEXT_LENGTH = 100
HORIZON_LENGTH = 30
ALPHA = 1.0
OMEGA = 0.75
LAMBDA_K = 1e-5
LAMBDA_GAMMA = 1e-2
MAX_AUTO_TAUS = 32
TRANSFORM = "asinh"
IMPUTATION_STRATEGY = "linear"
TEMPERATURE_SAMPLE_FRACTION = 0.1
RNG_SEED = 0

OUTPUT_DIR = Path("/Users/jonghyeonlee/KMD_TS/experiments/daily_fixed_train_family_targets")


@dataclass(frozen=True)
class SupportEntry:
    family: str
    series_index: int
    prepared: PreparedKernelSeries


class StepwisePredictor:
    def __init__(
        self,
        gamma_gram: np.ndarray,
        future_targets: np.ndarray,
        target_masks: np.ndarray,
        lambda_gamma: float,
    ) -> None:
        self.horizon = int(future_targets.shape[1])
        self.models: list[tuple[np.ndarray, np.ndarray]] = []

        G = np.asarray(gamma_gram, dtype=float)
        Y = np.asarray(future_targets, dtype=float)
        M = np.asarray(target_masks, dtype=bool)

        for step in range(self.horizon):
            observed = np.flatnonzero(M[:, step])
            if observed.size == 0:
                self.models.append((observed, np.zeros(0, dtype=float)))
                continue
            G_step = G[np.ix_(observed, observed)]
            y_step = Y[observed, step]
            coeffs = np.linalg.solve(
                G_step + lambda_gamma * np.eye(observed.size),
                y_step,
            )
            self.models.append((observed, coeffs))

    def predict(self, gamma_query: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        g = np.asarray(gamma_query, dtype=float).reshape(-1)
        pred = np.full(self.horizon, fill_value, dtype=float)
        for step, (observed, coeffs) in enumerate(self.models):
            if observed.size == 0:
                continue
            pred[step] = g[observed] @ coeffs
        return pred


def relative_l2_error(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    return float(np.linalg.norm(pred_arr - true_arr) / (np.linalg.norm(true_arr) + eps))


def dataset_count(dataset_name: str) -> int:
    dataset = Dataset(name=dataset_name, term="short", to_univariate=False, storage_env_var="GIFT_EVAL")
    return sum(1 for _ in dataset.training_dataset)


def load_selected_series(
    dataset_name: str,
    *,
    selected_indices: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    dataset = Dataset(name=dataset_name, term="short", to_univariate=False, storage_env_var="GIFT_EVAL")
    wanted = np.asarray(np.sort(selected_indices), dtype=int)
    if wanted.size == 0:
        return []

    result: list[tuple[int, np.ndarray, np.ndarray]] = []
    ptr = 0
    for series_index, (train_entry, val_entry) in enumerate(
        zip(dataset.training_dataset, dataset.validation_dataset)
    ):
        if ptr >= wanted.size:
            break
        target_idx = int(wanted[ptr])
        if series_index < target_idx:
            continue
        if series_index > target_idx:
            raise RuntimeError(
                f"Iterator skipped requested series {target_idx} in {dataset_name}."
            )
        train = _as_1d_array(train_entry["target"], f"{dataset_name} training target[{series_index}]")
        val_full = _as_1d_array(val_entry["target"], f"{dataset_name} validation target[{series_index}]")
        future = val_full[len(train) :]
        result.append((series_index, train, future))
        ptr += 1

    if ptr != wanted.size:
        raise IndexError(
            f"Requested {wanted.size} indices from {dataset_name}, but only found {ptr}."
        )
    return result


def build_baby_from_prepared(prepared: PreparedKernelSeries) -> np.ndarray:
    series_length = prepared.train_raw.shape[0]
    tau_values = build_auto_tau_array(
        series_length,
        context_length=CONTEXT_LENGTH,
        max_taus=MAX_AUTO_TAUS,
        include_current_window=True,
    )
    max_lookback = max(series_length - CONTEXT_LENGTH, 0)
    kernel = np.zeros((CONTEXT_LENGTH, CONTEXT_LENGTH), dtype=float)
    for tau in tau_values:
        start = series_length - CONTEXT_LENGTH - int(tau)
        end = series_length - int(tau)
        if start < 0 or end > series_length or (end - start) != CONTEXT_LENGTH:
            continue
        weights = _tau_localization_weights(
            context_length=CONTEXT_LENGTH,
            tau=int(tau),
            max_lookback=max_lookback,
            alpha=ALPHA,
            omega=OMEGA,
        )
        chi = prepared.train_filled[start:end] * weights
        kernel += np.outer(chi, chi)
    return kernel


def prepare_support_entries() -> tuple[list[SupportEntry], dict[str, np.ndarray]]:
    rng = np.random.default_rng(RNG_SEED)
    selected_indices_by_family: dict[str, np.ndarray] = {}
    support_entries: list[SupportEntry] = []

    for family in TRAIN_FAMILIES:
        count = dataset_count(family)
        if family == TEMPERATURE_FAMILY:
            sample_size = max(1, int(np.floor(count * TEMPERATURE_SAMPLE_FRACTION)))
            selected_indices = np.sort(rng.choice(count, size=sample_size, replace=False))
        else:
            selected_indices = np.arange(count, dtype=int)
        selected_indices_by_family[family] = selected_indices

        for series_index, train, future in load_selected_series(family, selected_indices=selected_indices):
            prepared = prepare_kernel_series(
                train,
                future,
                name=f"{family}#series{series_index}",
                context_length=CONTEXT_LENGTH,
                horizon_length=HORIZON_LENGTH,
                transform=TRANSFORM,
                imputation_strategy=IMPUTATION_STRATEGY,
            )
            support_entries.append(
                SupportEntry(
                    family=family,
                    series_index=series_index,
                    prepared=prepared,
                )
            )

    return support_entries, selected_indices_by_family


def stack_support(entries: list[SupportEntry]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_scaled = np.vstack([entry.prepared.context_scaled for entry in entries])
    X_mask = np.vstack([entry.prepared.context_mask.astype(bool) for entry in entries])
    Y_scaled = np.vstack(
        [
            np.where(entry.prepared.future_mask.astype(bool), entry.prepared.future_scaled, 0.0)
            for entry in entries
        ]
    )
    Y_mask = np.vstack([entry.prepared.future_mask.astype(bool) for entry in entries])
    return X_scaled, X_mask, Y_scaled, Y_mask


def load_target_entries(dataset_name: str) -> list[tuple[int, PreparedKernelSeries]]:
    count = dataset_count(dataset_name)
    selected = np.arange(count, dtype=int)
    prepared_entries = []
    for series_index, train, future in load_selected_series(dataset_name, selected_indices=selected):
        prepared = prepare_kernel_series(
            train,
            future,
            name=f"{dataset_name}#series{series_index}",
            context_length=CONTEXT_LENGTH,
            horizon_length=HORIZON_LENGTH,
            transform=TRANSFORM,
            imputation_strategy=IMPUTATION_STRATEGY,
        )
        prepared_entries.append((series_index, prepared))
    return prepared_entries


def save_single_prediction_plot(
    *,
    dataset_name: str,
    prepared: PreparedKernelSeries,
    full_pred: np.ndarray,
    top1_pred: np.ndarray,
    family_pred: np.ndarray,
) -> Path:
    x = np.arange(1, HORIZON_LENGTH + 1)
    path = OUTPUT_DIR / f"{dataset_name.replace('/', '__')}_prediction.png"
    plt.figure(figsize=(10, 4))
    plt.plot(x, prepared.future_raw, marker="o", label="True")
    plt.plot(x, full_pred, marker="x", label="Full")
    plt.plot(x, top1_pred, marker="^", label="Top-1 energy")
    plt.plot(x, family_pred, marker="s", label="Best family")
    plt.xlabel("Forecast step")
    plt.ylabel("Raw value")
    plt.title(f"{dataset_name} raw validation forecast")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_solar_grid_plot(
    *,
    selected_series: list[tuple[int, PreparedKernelSeries]],
    prediction_lookup: dict[int, dict[str, np.ndarray]],
) -> Path:
    path = OUTPUT_DIR / "solar__D_prediction_grid.png"
    fig, axes = plt.subplots(len(selected_series), 1, figsize=(10, 3 * len(selected_series)), sharex=True)
    x = np.arange(1, HORIZON_LENGTH + 1)
    if len(selected_series) == 1:
        axes = [axes]
    for ax, (series_index, prepared) in zip(axes, selected_series):
        preds = prediction_lookup[series_index]
        ax.plot(x, prepared.future_raw, marker="o", label="True")
        ax.plot(x, preds["full"], marker="x", label="Full")
        ax.plot(x, preds["top1"], marker="^", label="Top-1 energy")
        ax.plot(x, preds["family"], marker="s", label="Best family")
        ax.set_title(f"solar/D series {series_index}")
        ax.set_ylabel("Raw value")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Forecast step")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    support_entries, selected_indices_by_family = prepare_support_entries()
    X_scaled, X_mask, Y_scaled, Y_mask = stack_support(support_entries)

    family_to_support_indices: dict[str, np.ndarray] = {}
    for family in TRAIN_FAMILIES:
        family_to_support_indices[family] = np.array(
            [idx for idx, entry in enumerate(support_entries) if entry.family == family],
            dtype=int,
        )

    baby_kernels = [build_baby_from_prepared(entry.prepared) for entry in support_entries]
    baby_tensor = np.stack(baby_kernels, axis=0)
    K_full = baby_tensor.sum(axis=0)

    family_kernels = {
        family: baby_tensor[indices].sum(axis=0)
        for family, indices in family_to_support_indices.items()
    }

    gamma_xx_full = build_gamma_gram(
        X_scaled,
        time_kernel=K_full,
        lambda_k=LAMBDA_K,
        masks=X_mask,
    )
    full_predictor = StepwisePredictor(gamma_xx_full, Y_scaled, Y_mask, LAMBDA_GAMMA)

    family_predictors: dict[str, tuple[np.ndarray, StepwisePredictor, np.ndarray, np.ndarray]] = {}
    for family, indices in family_to_support_indices.items():
        X_family = X_scaled[indices]
        X_mask_family = X_mask[indices]
        Y_family = Y_scaled[indices]
        Y_mask_family = Y_mask[indices]
        gamma_xx_family = build_gamma_gram(
            X_family,
            time_kernel=family_kernels[family],
            lambda_k=LAMBDA_K,
            masks=X_mask_family,
        )
        family_predictors[family] = (
            gamma_xx_family,
            StepwisePredictor(gamma_xx_family, Y_family, Y_mask_family, LAMBDA_GAMMA),
            X_family,
            X_mask_family,
        )

    top1_cache: dict[int, tuple[np.ndarray, StepwisePredictor, np.ndarray, np.ndarray]] = {}
    full_regularized = K_full + LAMBDA_K * np.eye(CONTEXT_LENGTH)

    rows: list[dict[str, object]] = []
    solar_prediction_lookup: dict[int, dict[str, np.ndarray]] = {}
    solar_entries: list[tuple[int, PreparedKernelSeries]] = []
    saved_prediction_paths: dict[str, Path] = {}

    for target_name in TARGET_DATASETS:
        target_entries = load_target_entries(target_name)
        dataset_metrics = {"full": [], "top1": [], "family": []}

        for series_index, prepared in target_entries:
            query_context = np.where(
                prepared.context_mask.astype(bool),
                prepared.context_scaled,
                0.0,
            )
            qA = np.linalg.solve(full_regularized, query_context)
            series_energies = np.einsum("i,nij,j->n", qA, baby_tensor, qA, optimize=True)
            top1_idx = int(np.argmax(series_energies))

            family_energies = {
                family: float(series_energies[indices].sum())
                for family, indices in family_to_support_indices.items()
            }
            best_family = max(family_energies.items(), key=lambda item: item[1])[0]

            gamma_wx_full = build_gamma_query(
                prepared.context_scaled,
                X_scaled,
                time_kernel=K_full,
                lambda_k=LAMBDA_K,
                query_mask=prepared.context_mask,
                context_masks=X_mask,
            )
            pred_scaled_full = full_predictor.predict(gamma_wx_full)
            pred_raw_full = inverse_transform_with_stats(
                pred_scaled_full,
                loc=prepared.loc,
                scale=prepared.scale,
                transform=prepared.transform,
            )

            if top1_idx not in top1_cache:
                X_one = X_scaled[top1_idx : top1_idx + 1]
                X_mask_one = X_mask[top1_idx : top1_idx + 1]
                Y_one = Y_scaled[top1_idx : top1_idx + 1]
                Y_mask_one = Y_mask[top1_idx : top1_idx + 1]
                gamma_xx_one = build_gamma_gram(
                    X_one,
                    time_kernel=baby_kernels[top1_idx],
                    lambda_k=LAMBDA_K,
                    masks=X_mask_one,
                )
                top1_cache[top1_idx] = (
                    gamma_xx_one,
                    StepwisePredictor(gamma_xx_one, Y_one, Y_mask_one, LAMBDA_GAMMA),
                    X_one,
                    X_mask_one,
                )

            _, top1_predictor, X_one, X_mask_one = top1_cache[top1_idx]
            gamma_wx_top1 = build_gamma_query(
                prepared.context_scaled,
                X_one,
                time_kernel=baby_kernels[top1_idx],
                lambda_k=LAMBDA_K,
                query_mask=prepared.context_mask,
                context_masks=X_mask_one,
            )
            pred_scaled_top1 = top1_predictor.predict(gamma_wx_top1)
            pred_raw_top1 = inverse_transform_with_stats(
                pred_scaled_top1,
                loc=prepared.loc,
                scale=prepared.scale,
                transform=prepared.transform,
            )

            _, family_predictor, X_family, X_mask_family = family_predictors[best_family]
            gamma_wx_family = build_gamma_query(
                prepared.context_scaled,
                X_family,
                time_kernel=family_kernels[best_family],
                lambda_k=LAMBDA_K,
                query_mask=prepared.context_mask,
                context_masks=X_mask_family,
            )
            pred_scaled_family = family_predictor.predict(gamma_wx_family)
            pred_raw_family = inverse_transform_with_stats(
                pred_scaled_family,
                loc=prepared.loc,
                scale=prepared.scale,
                transform=prepared.transform,
            )

            observed = prepared.future_mask.astype(bool)
            raw_true_obs = prepared.future_raw[observed]
            full_l2 = relative_l2_error(pred_raw_full[observed], raw_true_obs)
            top1_l2 = relative_l2_error(pred_raw_top1[observed], raw_true_obs)
            family_l2 = relative_l2_error(pred_raw_family[observed], raw_true_obs)
            dataset_metrics["full"].append(full_l2)
            dataset_metrics["top1"].append(top1_l2)
            dataset_metrics["family"].append(family_l2)

            rows.append(
                {
                    "target_dataset": target_name,
                    "series_index": series_index,
                    "full_raw_l2": full_l2,
                    "top1_raw_l2": top1_l2,
                    "family_raw_l2": family_l2,
                    "top1_family": support_entries[top1_idx].family,
                    "best_family": best_family,
                    "best_family_energy": family_energies[best_family],
                }
            )

            if target_name == "solar/D":
                solar_entries.append((series_index, prepared))
                solar_prediction_lookup[series_index] = {
                    "full": pred_raw_full,
                    "top1": pred_raw_top1,
                    "family": pred_raw_family,
                }
            elif target_name not in saved_prediction_paths:
                saved_prediction_paths[target_name] = save_single_prediction_plot(
                    dataset_name=target_name,
                    prepared=prepared,
                    full_pred=pred_raw_full,
                    top1_pred=pred_raw_top1,
                    family_pred=pred_raw_family,
                )

        dataset_summary = {
            "target_dataset": target_name,
            "series_count": len(target_entries),
            "full_mean_raw_l2": float(np.mean(dataset_metrics["full"])),
            "top1_mean_raw_l2": float(np.mean(dataset_metrics["top1"])),
            "family_mean_raw_l2": float(np.mean(dataset_metrics["family"])),
        }
        rows.append({"target_dataset": target_name, "series_index": "MEAN", **dataset_summary})

    solar_plot_path = save_solar_grid_plot(
        selected_series=solar_entries[:5],
        prediction_lookup=solar_prediction_lookup,
    )
    saved_prediction_paths["solar/D"] = solar_plot_path

    aggregate_rows = [row for row in rows if row.get("series_index") == "MEAN"]
    with (OUTPUT_DIR / "daily_fixed_train_family_targets_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "target_dataset",
                "series_index",
                "full_raw_l2",
                "top1_raw_l2",
                "family_raw_l2",
                "top1_family",
                "best_family",
                "best_family_energy",
                "series_count",
                "full_mean_raw_l2",
                "top1_mean_raw_l2",
                "family_mean_raw_l2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(aggregate_rows))
    width = 0.25
    plt.bar(
        x - width,
        [row["full_mean_raw_l2"] for row in aggregate_rows],
        width,
        label="Full",
    )
    plt.bar(
        x,
        [row["top1_mean_raw_l2"] for row in aggregate_rows],
        width,
        label="Top-1 energy",
    )
    plt.bar(
        x + width,
        [row["family_mean_raw_l2"] for row in aggregate_rows],
        width,
        label="Best family",
    )
    plt.xticks(x, [row["target_dataset"] for row in aggregate_rows], rotation=20, ha="right")
    plt.ylabel("Mean raw relative L2")
    plt.title("Raw relative L2 by pruning approach")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    error_plot_path = OUTPUT_DIR / "daily_fixed_train_family_targets_l2_bar.png"
    plt.savefig(error_plot_path, dpi=200)
    plt.close()

    summary = {
        "config": {
            "train_families": TRAIN_FAMILIES,
            "temperature_sample_fraction": TEMPERATURE_SAMPLE_FRACTION,
            "temperature_sample_size": int(selected_indices_by_family[TEMPERATURE_FAMILY].size),
            "rng_seed": RNG_SEED,
            "context_length": CONTEXT_LENGTH,
            "horizon_length": HORIZON_LENGTH,
            "alpha": ALPHA,
            "omega": OMEGA,
            "lambda_k": LAMBDA_K,
            "lambda_gamma": LAMBDA_GAMMA,
            "max_auto_taus": MAX_AUTO_TAUS,
        },
        "train_series_counts": {
            family: int(indices.size) for family, indices in selected_indices_by_family.items()
        },
        "prediction_plots": {key: str(path) for key, path in saved_prediction_paths.items()},
        "error_plot": str(error_plot_path),
        "aggregate_metrics": aggregate_rows,
        "note": (
            "us_births/D contains only one daily series, so the output includes its single "
            "forecast plot. The 5-panel plot is shown for solar/D, which is the multi-series target."
        ),
    }
    (OUTPUT_DIR / "daily_fixed_train_family_targets_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
