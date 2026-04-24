from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.kernel_preprocessing import (
    build_baby_kernel,
    build_gamma_gram,
    build_gamma_query,
    build_mama_kernel,
    compute_energy,
    inverse_transform_with_stats,
    predict_future_from_gamma,
    prepare_dataset_bank,
    prepare_support_bank,
    stack_context_bank,
    stack_future_bank,
)


DAILY_DATASETS = [
    "covid_deaths",
    "electricity/D",
    "hierarchical_sales/D",
    "kdd_cup_2018_with_missing/D",
    "LOOP_SEATTLE/D",
    "M_DENSE/D",
    "restaurant",
    "saugeenday/D",
    "solar/D",
    "temperature_rain_with_missing",
    "us_births/D",
]

CONTEXT_LENGTH = 64
HORIZON_LENGTH = 30
ALPHA = 1.0
OMEGA = 0.75
LAMBDA_K = 1e-5
LAMBDA_GAMMA = 1e-2
MAX_SERIES_PER_DATASET = 20
MAX_AUTO_TAUS = 32
TRANSFORM = "asinh"
IMPUTATION_STRATEGY = "linear"

OUTPUT_DIR = Path("/Users/jonghyeonlee/KMD_TS/experiments/daily_topk_leaveoneout")
SUMMARY_CSV = OUTPUT_DIR / "daily_topk_leaveoneout_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "daily_topk_leaveoneout_summary.json"
OVERVIEW_BAR = OUTPUT_DIR / "daily_topk_leaveoneout_raw_l2_overview.png"


def relative_l2_error(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    return float(np.linalg.norm(pred_arr - true_arr) / (np.linalg.norm(true_arr) + eps))


def relative_mae(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    return float(np.mean(np.abs(pred_arr - true_arr)) / (np.mean(np.abs(true_arr)) + eps))


def slugify(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def make_k_grid(total_count: int) -> list[int]:
    candidates = [
        1,
        2,
        3,
        5,
        8,
        10,
        12,
        15,
        20,
        25,
        30,
        40,
        50,
        64,
        80,
        100,
        128,
        160,
        total_count,
    ]
    return sorted({k for k in candidates if 1 <= k <= total_count})


def subset_prediction(
    selected_indices: np.ndarray,
    *,
    baby_kernels: list[np.ndarray],
    X_scaled: np.ndarray,
    X_mask: np.ndarray,
    Y_scaled: np.ndarray,
    Y_mask: np.ndarray,
    unseen,
) -> tuple[np.ndarray, np.ndarray]:
    subset = np.asarray(selected_indices, dtype=int)
    K_subset = np.sum([baby_kernels[idx] for idx in subset], axis=0)
    X_subset = X_scaled[subset]
    X_mask_subset = X_mask[subset]
    Y_subset = Y_scaled[subset]
    Y_mask_subset = Y_mask[subset]

    Gamma_xx_subset = build_gamma_gram(
        X_subset,
        time_kernel=K_subset,
        lambda_k=LAMBDA_K,
        masks=X_mask_subset,
    )
    Gamma_wx_subset = build_gamma_query(
        unseen.context_scaled,
        X_subset,
        time_kernel=K_subset,
        lambda_k=LAMBDA_K,
        query_mask=unseen.context_mask,
        context_masks=X_mask_subset,
    )
    pred_scaled_subset = predict_future_from_gamma(
        Gamma_wx_subset,
        Gamma_xx_subset,
        Y_subset,
        lambda_gamma=LAMBDA_GAMMA,
        target_masks=Y_mask_subset,
    )
    pred_raw_subset = inverse_transform_with_stats(
        pred_scaled_subset,
        loc=unseen.loc,
        scale=unseen.scale,
        transform=unseen.transform,
    )
    return pred_scaled_subset, pred_raw_subset


def run_holdout(unseen_name: str) -> dict[str, object]:
    seen_names = [name for name in DAILY_DATASETS if name != unseen_name]
    support = prepare_support_bank(
        seen_names,
        context_length=CONTEXT_LENGTH,
        horizon_length=HORIZON_LENGTH,
        max_series_per_dataset=MAX_SERIES_PER_DATASET,
        transform=TRANSFORM,
        imputation_strategy=IMPUTATION_STRATEGY,
    )
    X, X_mask = stack_context_bank(support, include_masks=True)
    _, _ = stack_future_bank(support, include_masks=True)
    X_scaled = np.vstack([series.context_scaled for series in support])
    Y_scaled = np.vstack(
        [
            np.where(series.future_mask.astype(bool), series.future_scaled, 0.0)
            for series in support
        ]
    )
    Y_mask = np.vstack([series.future_mask.astype(bool) for series in support])

    _, unseen = prepare_dataset_bank(
        seen_names,
        unseen_name=unseen_name,
        context_length=CONTEXT_LENGTH,
        horizon_length=HORIZON_LENGTH,
        transform=TRANSFORM,
        imputation_strategy=IMPUTATION_STRATEGY,
    )

    K_full = build_mama_kernel(
        dataset_names=seen_names,
        term="short",
        context_length=CONTEXT_LENGTH,
        horizon_length=HORIZON_LENGTH,
        tau_arrays=None,
        to_univariate=False,
        max_series_per_dataset=MAX_SERIES_PER_DATASET,
        alpha=ALPHA,
        omega=OMEGA,
        normalize=False,
        storage_env_var="GIFT_EVAL",
        transform=TRANSFORM,
        imputation_strategy=IMPUTATION_STRATEGY,
        round_down_to_power_of_two=False,
        max_auto_taus=MAX_AUTO_TAUS,
        include_current_window=True,
    )

    Gamma_xx_full = build_gamma_gram(
        X_scaled,
        time_kernel=K_full,
        lambda_k=LAMBDA_K,
        masks=X_mask,
    )
    Gamma_wx_full = build_gamma_query(
        unseen.context_scaled,
        X_scaled,
        time_kernel=K_full,
        lambda_k=LAMBDA_K,
        query_mask=unseen.context_mask,
        context_masks=X_mask,
    )
    pred_scaled_full = predict_future_from_gamma(
        Gamma_wx_full,
        Gamma_xx_full,
        Y_scaled,
        lambda_gamma=LAMBDA_GAMMA,
        target_masks=Y_mask,
    )
    pred_raw_full = inverse_transform_with_stats(
        pred_scaled_full,
        loc=unseen.loc,
        scale=unseen.scale,
        transform=unseen.transform,
    )

    observed_future = unseen.future_mask.astype(bool)
    true_scaled_obs = unseen.future_scaled[observed_future]
    true_raw_obs = unseen.future_raw[observed_future]
    query_context = np.where(unseen.context_mask.astype(bool), unseen.context_scaled, 0.0)

    energies = np.zeros(len(support), dtype=float)
    baby_kernels: list[np.ndarray] = []
    dataset_labels: list[str] = []
    dataset_energy = defaultdict(float)
    dataset_members = defaultdict(list)

    for idx, series in enumerate(support):
        dataset_name, series_suffix = series.name.split("#series")
        series_index = int(series_suffix)
        dataset_labels.append(dataset_name)
        dataset_members[dataset_name].append(idx)
        baby_kernel = build_baby_kernel(
            dataset_name=dataset_name,
            term="short",
            series_index=series_index,
            context_length=CONTEXT_LENGTH,
            tau_array=None,
            to_univariate=False,
            alpha=ALPHA,
            omega=OMEGA,
            normalize=False,
            storage_env_var="GIFT_EVAL",
            transform=TRANSFORM,
            imputation_strategy=IMPUTATION_STRATEGY,
            max_auto_taus=MAX_AUTO_TAUS,
            include_current_window=True,
        )
        baby_kernels.append(baby_kernel)
        energy = compute_energy(query_context, baby_kernel, K_full, nugget=LAMBDA_K)
        energies[idx] = energy
        dataset_energy[dataset_name] += float(energy)

    full_metrics = {
        "scaled_relative_l2_error": relative_l2_error(
            pred_scaled_full[observed_future], true_scaled_obs
        ),
        "unscaled_relative_l2_error": relative_l2_error(
            pred_raw_full[observed_future], true_raw_obs
        ),
        "scaled_relative_mae": relative_mae(pred_scaled_full[observed_future], true_scaled_obs),
        "unscaled_relative_mae": relative_mae(pred_raw_full[observed_future], true_raw_obs),
    }

    order = np.argsort(-energies)
    k_grid = make_k_grid(len(support))
    rows: list[dict[str, float | int | str]] = []
    best_topk: dict[str, object] | None = None
    top1_prediction: dict[str, object] | None = None

    for k in k_grid:
        selected = order[:k]
        pred_scaled_k, pred_raw_k = subset_prediction(
            selected,
            baby_kernels=baby_kernels,
            X_scaled=X_scaled,
            X_mask=X_mask,
            Y_scaled=Y_scaled,
            Y_mask=Y_mask,
            unseen=unseen,
        )
        row = {
            "k": k,
            "scaled_relative_l2_error": relative_l2_error(
                pred_scaled_k[observed_future], true_scaled_obs
            ),
            "unscaled_relative_l2_error": relative_l2_error(
                pred_raw_k[observed_future], true_raw_obs
            ),
            "scaled_relative_mae": relative_mae(pred_scaled_k[observed_future], true_scaled_obs),
            "unscaled_relative_mae": relative_mae(pred_raw_k[observed_future], true_raw_obs),
            "energy_fraction": float(energies[selected].sum() / max(energies.sum(), 1e-12)),
            "top_dataset_count": int(
                sum(1 for idx in selected if dataset_labels[idx] == dataset_labels[selected[0]])
            ),
            "top_dataset_name": dataset_labels[selected[0]],
        }
        rows.append(row)
        if k == 1:
            top1_prediction = {
                "pred_scaled": pred_scaled_k,
                "pred_raw": pred_raw_k,
                "metrics": row,
            }
        if (
            best_topk is None
            or row["unscaled_relative_l2_error"] < best_topk["metrics"]["unscaled_relative_l2_error"]
        ):
            best_topk = {
                "k": k,
                "indices": selected.copy(),
                "pred_scaled": pred_scaled_k,
                "pred_raw": pred_raw_k,
                "metrics": row,
            }

    best_dataset = max(dataset_energy.items(), key=lambda item: item[1])[0]
    best_dataset_indices = np.array(dataset_members[best_dataset], dtype=int)
    pred_scaled_dataset, pred_raw_dataset = subset_prediction(
        best_dataset_indices,
        baby_kernels=baby_kernels,
        X_scaled=X_scaled,
        X_mask=X_mask,
        Y_scaled=Y_scaled,
        Y_mask=Y_mask,
        unseen=unseen,
    )
    dataset_metrics = {
        "dataset_name": best_dataset,
        "series_count": int(best_dataset_indices.size),
        "energy_fraction": float(
            dataset_energy[best_dataset] / max(sum(dataset_energy.values()), 1e-12)
        ),
        "scaled_relative_l2_error": relative_l2_error(
            pred_scaled_dataset[observed_future], true_scaled_obs
        ),
        "unscaled_relative_l2_error": relative_l2_error(
            pred_raw_dataset[observed_future], true_raw_obs
        ),
        "scaled_relative_mae": relative_mae(pred_scaled_dataset[observed_future], true_scaled_obs),
        "unscaled_relative_mae": relative_mae(pred_raw_dataset[observed_future], true_raw_obs),
    }

    slug = slugify(unseen_name)
    holdout_dir = OUTPUT_DIR / slug
    holdout_dir.mkdir(parents=True, exist_ok=True)

    with (holdout_dir / f"{slug}_topk_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "k",
                "scaled_relative_l2_error",
                "unscaled_relative_l2_error",
                "scaled_relative_mae",
                "unscaled_relative_mae",
                "energy_fraction",
                "top_dataset_count",
                "top_dataset_name",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(10, 5))
    plt.plot(
        [row["k"] for row in rows],
        [row["unscaled_relative_l2_error"] for row in rows],
        marker="o",
        label="Top-k raw relative L2",
    )
    plt.plot(
        [row["k"] for row in rows],
        [row["scaled_relative_l2_error"] for row in rows],
        marker="s",
        label="Top-k scaled relative L2",
    )
    plt.axhline(
        full_metrics["unscaled_relative_l2_error"],
        color="tab:blue",
        linestyle="--",
        alpha=0.7,
        label="Full raw relative L2",
    )
    plt.axhline(
        top1_prediction["metrics"]["unscaled_relative_l2_error"],
        color="tab:red",
        linestyle=":",
        alpha=0.7,
        label="Top-1 raw relative L2",
    )
    plt.axhline(
        dataset_metrics["unscaled_relative_l2_error"],
        color="tab:green",
        linestyle="-.",
        alpha=0.7,
        label=f"Best dataset raw relative L2 ({best_dataset})",
    )
    plt.xlabel("Top-k support series")
    plt.ylabel("Relative L2 error")
    plt.title(f"{unseen_name} pruning sweep by energy-ranked top-k supports")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(holdout_dir / f"{slug}_topk_vs_error.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(unseen.future_raw, label="True raw", marker="o")
    plt.plot(pred_raw_full, label="Full model", marker="x")
    plt.plot(top1_prediction["pred_raw"], label="Top-1", marker="^")
    plt.plot(best_topk["pred_raw"], label=f"Best top-k (k={best_topk['k']})", marker="s")
    plt.plot(pred_raw_dataset, label=f"Best dataset ({best_dataset})", marker="d")
    plt.title(f"{unseen_name} raw validation forecast")
    plt.xlabel("Forecast step")
    plt.ylabel("Raw value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(holdout_dir / f"{slug}_prediction_comparison.png", dpi=200)
    plt.close()

    holdout_summary = {
        "unseen_name": unseen_name,
        "support_count": len(support),
        "full_metrics": full_metrics,
        "top1_metrics": top1_prediction["metrics"],
        "best_topk": {
            "k": int(best_topk["k"]),
            "metrics": best_topk["metrics"],
        },
        "best_dataset_subset": dataset_metrics,
        "dataset_energy": dict(sorted(dataset_energy.items(), key=lambda item: item[1], reverse=True)),
    }
    (holdout_dir / f"{slug}_summary.json").write_text(json.dumps(holdout_summary, indent=2))
    return holdout_summary


def plot_overview(summaries: list[dict[str, object]]) -> None:
    names = [entry["unseen_name"] for entry in summaries]
    full_raw = [entry["full_metrics"]["unscaled_relative_l2_error"] for entry in summaries]
    top1_raw = [entry["top1_metrics"]["unscaled_relative_l2_error"] for entry in summaries]
    best_topk_raw = [entry["best_topk"]["metrics"]["unscaled_relative_l2_error"] for entry in summaries]
    best_dataset_raw = [
        entry["best_dataset_subset"]["unscaled_relative_l2_error"] for entry in summaries
    ]

    x = np.arange(len(names))
    width = 0.2
    plt.figure(figsize=(14, 6))
    plt.bar(x - 1.5 * width, full_raw, width, label="Full")
    plt.bar(x - 0.5 * width, top1_raw, width, label="Top-1")
    plt.bar(x + 0.5 * width, best_topk_raw, width, label="Best top-k")
    plt.bar(x + 1.5 * width, best_dataset_raw, width, label="Best dataset subset")
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("Raw relative L2 error")
    plt.title("Daily leave-one-out comparison of pruning strategies")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OVERVIEW_BAR, dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for unseen_name in DAILY_DATASETS:
        print(f"Running holdout for {unseen_name}...")
        summaries.append(run_holdout(unseen_name))

    plot_overview(summaries)

    with SUMMARY_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "unseen_name",
                "support_count",
                "full_raw_l2",
                "top1_raw_l2",
                "best_topk_k",
                "best_topk_raw_l2",
                "best_dataset_name",
                "best_dataset_raw_l2",
            ],
        )
        writer.writeheader()
        for entry in summaries:
            writer.writerow(
                {
                    "unseen_name": entry["unseen_name"],
                    "support_count": entry["support_count"],
                    "full_raw_l2": entry["full_metrics"]["unscaled_relative_l2_error"],
                    "top1_raw_l2": entry["top1_metrics"]["unscaled_relative_l2_error"],
                    "best_topk_k": entry["best_topk"]["k"],
                    "best_topk_raw_l2": entry["best_topk"]["metrics"]["unscaled_relative_l2_error"],
                    "best_dataset_name": entry["best_dataset_subset"]["dataset_name"],
                    "best_dataset_raw_l2": entry["best_dataset_subset"]["unscaled_relative_l2_error"],
                }
            )

    SUMMARY_JSON.write_text(json.dumps(summaries, indent=2))
    print(f"Saved leave-one-out outputs under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
