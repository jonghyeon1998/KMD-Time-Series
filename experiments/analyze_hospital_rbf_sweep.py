from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gift_eval.data import Dataset
from src.window_krr import (
    KernelConfig,
    WindowKernelRidgeForecaster,
    build_regular_deltas,
    make_windowed_forecasting_dataset,
    relative_l2_error,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "experiments" / "hospital_rbf_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DATASET_NAME = "hospital"
TERM = "short"
TO_UNIVARIATE = False

CONTEXT_LENGTH = 30
MAX_SERIES = None
MAX_WINDOWS_PER_SERIES = 200
SHORT_HORIZON = None
LONG_HORIZON = None
TRAIN_STRIDE = None
VALIDATION_STRIDE = None

REPRESENTATION = "plain"
AMPLITUDE = 1.0
# This grid is intentionally modest so the sweep stays interactive on a laptop.
LENGTH_SCALES = [5.0, 20.0, 50.0, 100.0, 500.0, 1000.0, 2000.0]
REGULARIZATIONS = [1e-5, 1e-3, 1e-1, 1.0]


def nth_item(data_it, index: int):
    for i, item in enumerate(data_it):
        if i == index:
            return item
    raise IndexError(index)


def series_from_entry(entry) -> np.ndarray:
    return np.asarray(entry["target"], dtype=float).reshape(-1)


def get_train_entry(dataset: Dataset, series_index: int):
    return nth_item(dataset.training_dataset, series_index)


def get_validation_entry(dataset: Dataset, series_index: int):
    return nth_item(dataset.validation_dataset, series_index)


def get_test_pair(dataset: Dataset, series_index: int, window_index: int):
    flat_index = series_index * dataset.windows + window_index
    return nth_item(dataset.test_data.input, flat_index), nth_item(dataset.test_data.label, flat_index)


def validation_minus_train(dataset: Dataset, series_index: int) -> np.ndarray:
    train = series_from_entry(get_train_entry(dataset, series_index))
    val = series_from_entry(get_validation_entry(dataset, series_index))
    return val[len(train) :]


def build_deltas_from_values(values: np.ndarray) -> np.ndarray:
    return build_regular_deltas(len(values), step=1.0)


def build_window_bank_from_split(
    dataset: Dataset,
    split_name: str,
    *,
    context_length: int,
    horizon_length: int,
    stride: int | None,
    max_series: int | None,
    max_windows_per_series: int | None,
):
    entry_iter = dataset.training_dataset if split_name == "train" else dataset.validation_dataset
    stride = horizon_length if stride is None else int(stride)

    contexts = []
    targets = []
    context_masks = []
    target_masks = []
    context_deltas = []
    used_series = 0

    for series_index, entry in enumerate(entry_iter):
        if max_series is not None and used_series >= max_series:
            break
        values = series_from_entry(entry)
        observed_mask = np.isfinite(values)

        try:
            ds = make_windowed_forecasting_dataset(
                values,
                context_length=context_length,
                horizon_length=horizon_length,
                observed_mask=observed_mask,
                deltas=build_deltas_from_values(values),
                stride=stride,
                max_windows=max_windows_per_series,
            )
        except ValueError:
            continue

        contexts.append(ds.contexts)
        targets.append(ds.targets)
        context_masks.append(ds.context_masks)
        target_masks.append(ds.target_masks)
        context_deltas.append(ds.context_deltas)
        used_series += 1

    return {
        "contexts": np.vstack(contexts),
        "targets": np.vstack(targets),
        "context_masks": np.vstack(context_masks),
        "target_masks": np.vstack(target_masks),
        "context_deltas": np.vstack(context_deltas),
        "used_series": used_series,
    }


def evaluate_short_term(dataset: Dataset, model: WindowKernelRidgeForecaster, short_horizon: int) -> float:
    errors = []
    for series_index in range(len(dataset.hf_dataset)):
        train = series_from_entry(get_train_entry(dataset, series_index))
        val_extra = validation_minus_train(dataset, series_index)
        if CONTEXT_LENGTH > len(train) or short_horizon > len(val_extra):
            continue
        query = train[-CONTEXT_LENGTH:]
        pred = model.forecast(
            query,
            strategy="direct",
            total_horizon=short_horizon,
            query_masks=np.isfinite(query),
            query_deltas=build_deltas_from_values(query),
        ).reshape(-1)
        truth = val_extra[:short_horizon]
        errors.append(relative_l2_error(truth, pred, mask=np.isfinite(truth)))
    return float(np.mean(errors))


def evaluate_long_term(dataset: Dataset, model: WindowKernelRidgeForecaster, long_horizon: int) -> float:
    errors = []
    for series_index in range(len(dataset.hf_dataset)):
        ctx_entry, lbl_entry = get_test_pair(dataset, series_index, 0)
        ctx = series_from_entry(ctx_entry)
        lbl = series_from_entry(lbl_entry)
        if CONTEXT_LENGTH > len(ctx) or long_horizon > len(lbl):
            continue
        query = ctx[-CONTEXT_LENGTH:]
        pred = model.forecast(
            query,
            strategy="direct",
            total_horizon=long_horizon,
            query_masks=np.isfinite(query),
            query_deltas=build_deltas_from_values(query),
        ).reshape(-1)
        truth = lbl[:long_horizon]
        errors.append(relative_l2_error(truth, pred, mask=np.isfinite(truth)))
    return float(np.mean(errors))


def main() -> None:
    dataset = Dataset(
        name=DATASET_NAME,
        term=TERM,
        to_univariate=TO_UNIVARIATE,
        storage_env_var="GIFT_EVAL",
    )

    short_horizon = dataset.prediction_length if SHORT_HORIZON is None else int(SHORT_HORIZON)
    long_horizon = dataset.prediction_length if LONG_HORIZON is None else int(LONG_HORIZON)

    short_bank = build_window_bank_from_split(
        dataset,
        "train",
        context_length=CONTEXT_LENGTH,
        horizon_length=short_horizon,
        stride=TRAIN_STRIDE,
        max_series=MAX_SERIES,
        max_windows_per_series=MAX_WINDOWS_PER_SERIES,
    )
    long_bank = build_window_bank_from_split(
        dataset,
        "validation",
        context_length=CONTEXT_LENGTH,
        horizon_length=long_horizon,
        stride=VALIDATION_STRIDE,
        max_series=MAX_SERIES,
        max_windows_per_series=MAX_WINDOWS_PER_SERIES,
    )

    rows = []
    for reg in REGULARIZATIONS:
        for length_scale in LENGTH_SCALES:
            cfg = KernelConfig(
                family="rbf",
                amplitude=AMPLITUDE,
                length_scale=length_scale,
            )

            short_model = WindowKernelRidgeForecaster(
                kernel_config=cfg,
                representation=REPRESENTATION,
                regularization=reg,
            )
            short_model.fit(
                short_bank["contexts"],
                short_bank["targets"],
                context_masks=short_bank["context_masks"],
                target_masks=short_bank["target_masks"],
                context_deltas=short_bank["context_deltas"],
            )
            short_error = evaluate_short_term(dataset, short_model, short_horizon)

            long_model = WindowKernelRidgeForecaster(
                kernel_config=cfg,
                representation=REPRESENTATION,
                regularization=reg,
            )
            long_model.fit(
                long_bank["contexts"],
                long_bank["targets"],
                context_masks=long_bank["context_masks"],
                target_masks=long_bank["target_masks"],
                context_deltas=long_bank["context_deltas"],
            )
            long_error = evaluate_long_term(dataset, long_model, long_horizon)

            rows.append(
                {
                    "length_scale": length_scale,
                    "regularization": reg,
                    "short_mean_relative_l2": short_error,
                    "long_mean_relative_l2": long_error,
                }
            )
            print(
                f"length_scale={length_scale:<7g} reg={reg:<7g} "
                f"short={short_error:.6f} long={long_error:.6f}"
            )

    csv_path = OUTPUT_DIR / "hospital_rbf_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "length_scale",
                "regularization",
                "short_mean_relative_l2",
                "long_mean_relative_l2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # line plots over length scale for each regularization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for reg in REGULARIZATIONS:
        sub = [r for r in rows if r["regularization"] == reg]
        sub = sorted(sub, key=lambda r: r["length_scale"])
        x = [r["length_scale"] for r in sub]
        y_short = [r["short_mean_relative_l2"] for r in sub]
        y_long = [r["long_mean_relative_l2"] for r in sub]
        axes[0].plot(x, y_short, marker="o", label=f"lambda={reg:g}")
        axes[1].plot(x, y_long, marker="o", label=f"lambda={reg:g}")

    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_title("Hospital short-term mean relative L2")
    axes[1].set_title("Hospital long-term mean relative L2")
    for ax in axes:
        ax.set_xlabel("RBF length_scale")
        ax.set_ylabel("mean relative L2")
        ax.grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hospital_rbf_sweep_lines.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # heatmaps
    ls_vals = LENGTH_SCALES
    reg_vals = REGULARIZATIONS
    short_grid = np.array(
        [
            [next(r["short_mean_relative_l2"] for r in rows if r["length_scale"] == ls and r["regularization"] == reg) for ls in ls_vals]
            for reg in reg_vals
        ]
    )
    long_grid = np.array(
        [
            [next(r["long_mean_relative_l2"] for r in rows if r["length_scale"] == ls and r["regularization"] == reg) for ls in ls_vals]
            for reg in reg_vals
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].imshow(short_grid, aspect="auto", cmap="viridis")
    im1 = axes[1].imshow(long_grid, aspect="auto", cmap="viridis")
    axes[0].set_title("Short-term mean relative L2")
    axes[1].set_title("Long-term mean relative L2")
    for ax in axes:
        ax.set_xticks(range(len(ls_vals)))
        ax.set_xticklabels([f"{v:g}" for v in ls_vals], rotation=45, ha="right")
        ax.set_yticks(range(len(reg_vals)))
        ax.set_yticklabels([f"{v:g}" for v in reg_vals])
        ax.set_xlabel("length_scale")
        ax.set_ylabel("regularization")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hospital_rbf_sweep_heatmaps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
