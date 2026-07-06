"""
Generate Jupyter notebooks for univariate, no-missing GIFT-Eval datasets using
the nested kriging forecaster (src/nested_kriging.py).

One notebook per dataset/frequency; output goes to
experiments/univariate_nested_kriging_playgrounds/.

Usage:
    python experiments/generate_univariate_nested_kriging_playgrounds.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from textwrap import dedent


ROOT = Path("/Users/jonghyeonlee/KMD_TS")
AUDIT_JSON = ROOT / "experiments" / "gift_eval_univariate_split_audit.json"
KRR_PRESETS_JSON = ROOT / "experiments" / "window_krr_default_presets.json"
OUTPUT_DIR = ROOT / "experiments" / "univariate_nested_kriging_playgrounds"
README_PATH = OUTPUT_DIR / "README.md"


def notebook_filename(dataset_name: str) -> str:
    safe = dataset_name.replace("/", "__").replace(" ", "_").replace("-", "_")
    return f"{safe}_nested_kriging.ipynb"


def setup_cell() -> str:
    return dedent(
        """
        import os
        import sys
        from pathlib import Path

        def _find_repo_root() -> Path:
            cwd = Path.cwd().resolve()
            for candidate in [cwd, *cwd.parents]:
                if (candidate / "src").exists() and (candidate / "gift_eval").exists():
                    return candidate
            raise RuntimeError(
                "Could not locate the KMD_TS repository root from the current working directory."
            )

        REPO_ROOT = _find_repo_root()
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        os.environ.setdefault("GIFT_EVAL", str(REPO_ROOT / "GiftEval_data"))
        print("Repo root:", REPO_ROOT)
        print("GIFT_EVAL:", os.environ["GIFT_EVAL"])
        """
    ).strip()


def import_cell() -> str:
    return dedent(
        """
        import numpy as np
        import matplotlib.pyplot as plt

        from gift_eval.data import Dataset
        from src.nested_kriging import (
            LengthscaleField,
            NestedKrigingForecaster,
            fit_gibbs_lengthscale_field,
            pool_windows_from_series,
        )
        """
    ).strip()


def _default_block_size_l1(n_train_series: int, train_min_len: int, context_length: int, prediction_length: int) -> int:
    """Heuristic: block should contain ≥ 2·context_length data for reliable MLE."""
    estimated_windows = max(1, (train_min_len - context_length - prediction_length) // max(prediction_length, 1))
    total_est = estimated_windows * n_train_series
    # Aim for ~30 anchors but at least 10 windows per block
    target_anchors = 30
    return max(10, min(80, total_est // target_anchors))


def parameter_cell(dataset_name: str, row: dict, krr_preset: dict) -> str:
    context_length = int(krr_preset.get("context_length", 2 * int(row["prediction_length"])))
    max_series = int(krr_preset.get("max_series", 200))
    max_windows_per_series = int(krr_preset.get("max_windows_per_series", 20))
    prediction_length = int(row["prediction_length"])
    n_train_series = int(row["n_train_series"])
    train_min_len = int(row.get("train_min_len", 0))
    block_size_l1 = _default_block_size_l1(n_train_series, train_min_len, context_length, prediction_length)

    return dedent(
        f"""
        DATASET_NAME = "{dataset_name}"
        TERM = "short"
        TO_UNIVARIATE = False

        # Series to visualise and evaluate individually.
        SERIES_INDEX = 0

        # Window bank controls (keep conservative for interactivity).
        MAX_SERIES = {max_series}
        MAX_WINDOWS_PER_SERIES = {max_windows_per_series}

        CONTEXT_LENGTH = {context_length}        # L: context window length
        HORIZON_LENGTH = None                    # H: defaults to dataset.prediction_length

        # Instance normalisation applied before pooling windows.
        # "robust_series" uses median / IQR per series.
        NORMALIZATION = "robust_series"

        # ----------------------------------------------------------------
        # Layer 1 — lengthscale field (fit_gibbs_lengthscale_field)
        # ----------------------------------------------------------------
        BLOCK_SIZE_L1 = {block_size_l1}           # windows per block for local ℓ̂_m MLE
        SIGMA_N_L1 = 0.1                          # observation noise std assumed in MLE
        AMPLITUDE_L1 = 1.0                        # kernel amplitude in MLE
        LOG_ELL_LO = -3.0                         # lower bound of golden-section search (log-space)
        LOG_ELL_HI = 5.0                          # upper bound
        META_AMPLITUDE = 1.0                      # amplitude of the RBF meta-kernel k_g
        META_BANDWIDTH = None                     # None → median-heuristic bandwidth

        # ----------------------------------------------------------------
        # Layer 2 — nested kriging regression (NestedKrigingForecaster)
        # ----------------------------------------------------------------
        BLOCK_SIZE_L2 = 100                       # target Layer-2 block size
        MAX_BLOCKS_L2 = 30                        # cap on M (keeps M×M aggregation cheap)
        SIGMA_N_L2 = 0.1                          # observation noise std for regression
        AMPLITUDE_L2 = 1.0                        # Gibbs kernel amplitude

        # Evaluation scope
        EVALUATE_ALL_SERIES = False               # if False, evaluate SERIES_INDEX only
        EVAL_SERIES_LIMIT = None                  # optional cap when EVALUATE_ALL_SERIES=True
        """
    ).strip()


def metadata_cell() -> str:
    return dedent(
        """
        dataset = Dataset(
            name=DATASET_NAME,
            term=TERM,
            to_univariate=TO_UNIVARIATE,
            storage_env_var="GIFT_EVAL",
        )

        print("dataset_name       :", DATASET_NAME)
        print("frequency          :", dataset.freq)
        print("prediction_length  :", dataset.prediction_length)
        print("test windows/series:", dataset.windows)
        print("n_raw_series       :", len(dataset.hf_dataset))
        print("target_dim         :", dataset.target_dim)
        """
    ).strip()


def helper_cell() -> str:
    return dedent(
        """
        def _nth_item(data_it, index: int):
            for i, item in enumerate(data_it):
                if i == index:
                    return item
            raise IndexError(f"Index {index} is out of range.")


        def _series_from_entry(entry) -> np.ndarray:
            return np.asarray(entry["target"], dtype=float).reshape(-1)


        def _get_train_entry(series_index: int):
            return _nth_item(dataset.training_dataset, series_index)


        def _get_validation_entry(series_index: int):
            return _nth_item(dataset.validation_dataset, series_index)


        def _get_test_pair(series_index: int, window_index: int):
            if window_index < 0 or window_index >= dataset.windows:
                raise IndexError(
                    f"window_index={window_index} is out of range for dataset.windows={dataset.windows}."
                )
            flat_index = series_index * dataset.windows + window_index
            test_context = _nth_item(dataset.test_data.input, flat_index)
            test_label = _nth_item(dataset.test_data.label, flat_index)
            return test_context, test_label


        def _get_test_tail_labels(series_index: int, *, limit: int | None = None) -> list[np.ndarray]:
            n_windows = dataset.windows if limit is None else min(int(limit), dataset.windows)
            return [
                _series_from_entry(_get_test_pair(series_index, w)[1])
                for w in range(n_windows)
            ]


        def _collect_train_series(*, max_series: int | None) -> list[np.ndarray]:
            series = []
            for i, entry in enumerate(dataset.training_dataset):
                if max_series is not None and i >= max_series:
                    break
                arr = _series_from_entry(entry)
                if arr.size > 0 and np.isfinite(arr).all():
                    series.append(arr)
            return series


        def _collect_val_series(*, max_series: int | None) -> list[np.ndarray]:
            series = []
            for i, entry in enumerate(dataset.validation_dataset):
                if max_series is not None and i >= max_series:
                    break
                arr = _series_from_entry(entry)
                if arr.size > 0 and np.isfinite(arr).all():
                    series.append(arr)
            return series


        def _denorm(pred_norm: np.ndarray, loc: float, scale: float) -> np.ndarray:
            return pred_norm * scale + loc


        def relative_l2_error(true: np.ndarray, pred: np.ndarray) -> float:
            t = np.asarray(true, dtype=float).ravel()
            p = np.asarray(pred, dtype=float).ravel()
            denom = float(np.sum(t ** 2))
            if denom < 1e-12:
                return float("nan")
            return float(np.sum((t - p) ** 2) / denom)


        def _plot_forecast(y_true, y_pred, *, title: str):
            t = np.arange(1, len(y_true) + 1)
            plt.figure(figsize=(10, 4))
            plt.plot(t, y_true, label="true", linewidth=2)
            plt.plot(t, y_pred, label="predicted", linewidth=2)
            plt.title(title)
            plt.xlabel("horizon step")
            plt.ylabel("value")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.show()


        def _evaluation_series_indices(n_available: int) -> list[int]:
            if not EVALUATE_ALL_SERIES:
                return [SERIES_INDEX]
            limit = n_available if EVAL_SERIES_LIMIT is None else min(int(EVAL_SERIES_LIMIT), n_available)
            return list(range(max(1, limit)))
        """
    ).strip()


def preview_cell() -> str:
    return dedent(
        """
        train_entry = _get_train_entry(SERIES_INDEX)
        val_entry = _get_validation_entry(SERIES_INDEX)
        train_series = _series_from_entry(train_entry)
        val_series = _series_from_entry(val_entry)
        val_extra = val_series[len(train_series):]

        print("Selected series index:", SERIES_INDEX)
        print("train length         :", len(train_series))
        print("validation length    :", len(val_series))
        print("validation\\\\train len :", len(val_extra))
        print("test windows         :", dataset.windows)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.arange(len(val_series)), val_series, linewidth=1.5, label="up to validation")
        ax.axvline(len(train_series), color="tab:red", linestyle="--", label="train boundary")
        ax.set_title(f"{DATASET_NAME} series {SERIES_INDEX}")
        ax.set_xlabel("time index")
        ax.set_ylabel("value")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
        """
    ).strip()


def fit_l1_cell() -> str:
    return dedent(
        """
        rng = np.random.default_rng(42)
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)

        # ── collect training series ──────────────────────────────────────
        print("Collecting training series …")
        train_series_list = _collect_train_series(max_series=MAX_SERIES)
        print(f"  {len(train_series_list)} usable series")

        # ── pool windows from all training series ─────────────────────────
        print("Pooling windows …")
        ctx_train, tgt_train, meta_train = pool_windows_from_series(
            train_series_list,
            context_length=CONTEXT_LENGTH,
            horizon_length=horizon_length,
            stride=horizon_length,
            max_windows_per_series=MAX_WINDOWS_PER_SERIES,
            normalization=NORMALIZATION,
            rng=rng,
        )
        print(f"  pooled {ctx_train.shape[0]} windows, context shape: {ctx_train.shape}")

        # ── Layer 1: fit lengthscale field ───────────────────────────────
        print("\\nFitting Layer 1 (lengthscale field) …")
        ell_field = fit_gibbs_lengthscale_field(
            ctx_train, tgt_train,
            block_size_l1=BLOCK_SIZE_L1,
            sigma_n=SIGMA_N_L1,
            amplitude=AMPLITUDE_L1,
            log_ell_bounds=(LOG_ELL_LO, LOG_ELL_HI),
            meta_amplitude=META_AMPLITUDE,
            meta_bandwidth=META_BANDWIDTH,
            rng=rng,
        )
        print(f"  anchors (M_1) : {ell_field.n_anchors}")
        print(f"  meta_bandwidth: {ell_field.meta_bandwidth:.4f}")
        sample_ell = ell_field.predict_ell(ctx_train[:5])
        print(f"  sample ℓ̃ values: {sample_ell}")
        """
    ).strip()


def fit_l2_cell() -> str:
    return dedent(
        """
        # ── Layer 2: fit nested kriging regression ───────────────────────
        print("Fitting Layer 2 (nested kriging regression) …")
        forecaster = NestedKrigingForecaster(
            ell_field,
            block_size=BLOCK_SIZE_L2,
            max_blocks=MAX_BLOCKS_L2,
            sigma_n=SIGMA_N_L2,
            amplitude=AMPLITUDE_L2,
            rng=rng,
        ).fit(ctx_train, tgt_train)

        n_blocks = len(forecaster._fitted_blocks)
        block_sizes = [len(b.indices) for b in forecaster._fitted_blocks]
        print(f"  Layer-2 blocks (M_2): {n_blocks}")
        print(f"  Block sizes: min={min(block_sizes)}, max={max(block_sizes)}, mean={np.mean(block_sizes):.1f}")
        """
    ).strip()


def short_term_cell() -> str:
    return dedent(
        """
        # ── Short-term forecast: train -> validation\\train ────────────────
        print("Short-term evaluation (train -> validation\\\\train) …")
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)
        n_series_total = len(dataset.hf_dataset)

        short_errors = []
        short_skipped = []
        plotted = False

        for eval_idx in _evaluation_series_indices(n_series_total):
            train_arr = _series_from_entry(_get_train_entry(eval_idx))
            val_arr = _series_from_entry(_get_validation_entry(eval_idx))
            val_extra = val_arr[len(train_arr):]

            if len(train_arr) < CONTEXT_LENGTH:
                short_skipped.append((eval_idx, "train shorter than context_length"))
                continue
            if len(val_extra) < horizon_length:
                short_skipped.append((eval_idx, "validation\\\\train shorter than horizon_length"))
                continue

            # Instance-normalise using training median/IQR
            if NORMALIZATION == "robust_series":
                loc = float(np.median(train_arr))
                q75, q25 = float(np.percentile(train_arr, 75)), float(np.percentile(train_arr, 25))
                scale = max((q75 - q25) / 1.349, float(np.std(train_arr)), 1e-6)
            else:
                loc, scale = 0.0, 1.0

            query_raw = train_arr[-CONTEXT_LENGTH:]
            query_norm = (query_raw - loc) / scale
            pred_norm = forecaster.predict(query_norm)  # (H,)
            pred = _denorm(pred_norm, loc, scale)

            true_vals = val_extra[:horizon_length]
            err = relative_l2_error(true_vals, pred)
            short_errors.append(err)

            if (not EVALUATE_ALL_SERIES and eval_idx == SERIES_INDEX) or (
                EVALUATE_ALL_SERIES and not plotted
            ):
                _plot_forecast(
                    true_vals, pred,
                    title=f"{DATASET_NAME} short-term (series {eval_idx}): train → validation\\\\train",
                )
                plotted = True

        if short_errors:
            arr = np.asarray(short_errors, dtype=float)
            print(f"Short-term evaluated : {len(short_errors)} series")
            print(f"Short-term skipped   : {len(short_skipped)} series")
            print(f"Mean  relative L2    : {float(np.mean(arr)):.6f}")
            print(f"Median relative L2   : {float(np.median(arr)):.6f}")
        else:
            print("No short-term forecasts could be evaluated.")
            for s in short_skipped:
                print(" ", s)
        """
    ).strip()


def long_term_cell() -> str:
    return dedent(
        """
        # ── Re-fit Layer 1+2 on the validation split ─────────────────────
        print("Collecting validation series …")
        val_series_list = _collect_val_series(max_series=MAX_SERIES)
        print(f"  {len(val_series_list)} usable series")

        print("Pooling validation windows …")
        ctx_val, tgt_val, meta_val = pool_windows_from_series(
            val_series_list,
            context_length=CONTEXT_LENGTH,
            horizon_length=dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH),
            stride=dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH),
            max_windows_per_series=MAX_WINDOWS_PER_SERIES,
            normalization=NORMALIZATION,
            rng=rng,
        )
        print(f"  pooled {ctx_val.shape[0]} windows")

        print("\\nFitting Layer 1 on validation split …")
        ell_field_val = fit_gibbs_lengthscale_field(
            ctx_val, tgt_val,
            block_size_l1=BLOCK_SIZE_L1,
            sigma_n=SIGMA_N_L1,
            amplitude=AMPLITUDE_L1,
            log_ell_bounds=(LOG_ELL_LO, LOG_ELL_HI),
            meta_amplitude=META_AMPLITUDE,
            meta_bandwidth=META_BANDWIDTH,
            rng=rng,
        )
        print(f"  anchors: {ell_field_val.n_anchors}")

        print("Fitting Layer 2 on validation split …")
        forecaster_val = NestedKrigingForecaster(
            ell_field_val,
            block_size=BLOCK_SIZE_L2,
            max_blocks=MAX_BLOCKS_L2,
            sigma_n=SIGMA_N_L2,
            amplitude=AMPLITUDE_L2,
            rng=rng,
        ).fit(ctx_val, tgt_val)
        print("Done.")
        """
    ).strip()


def test_eval_cell() -> str:
    return dedent(
        """
        # ── Long-term: direct test-window evaluation ──────────────────────
        print("Long-term test evaluation …")
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)
        n_series_total = len(dataset.hf_dataset)

        test_errors = []
        test_skipped = []
        plotted = False

        for eval_idx in _evaluation_series_indices(n_series_total):
            val_arr = _series_from_entry(_get_validation_entry(eval_idx))
            if len(val_arr) < CONTEXT_LENGTH:
                test_skipped.append((eval_idx, "validation shorter than context_length"))
                continue

            # Normalise using validation stats
            if NORMALIZATION == "robust_series":
                loc = float(np.median(val_arr))
                q75, q25 = float(np.percentile(val_arr, 75)), float(np.percentile(val_arr, 25))
                scale = max((q75 - q25) / 1.349, float(np.std(val_arr)), 1e-6)
            else:
                loc, scale = 0.0, 1.0

            # Use the first test window (window_index=0)
            try:
                test_ctx_entry, test_lbl_entry = _get_test_pair(eval_idx, 0)
            except IndexError as e:
                test_skipped.append((eval_idx, str(e)))
                continue

            test_ctx = _series_from_entry(test_ctx_entry)
            test_lbl = _series_from_entry(test_lbl_entry)
            if len(test_ctx) < CONTEXT_LENGTH or len(test_lbl) < horizon_length:
                test_skipped.append((eval_idx, "test context or label too short"))
                continue

            query_raw = test_ctx[-CONTEXT_LENGTH:]
            query_norm = (query_raw - loc) / scale
            pred_norm = forecaster_val.predict(query_norm)   # (H,)
            pred = _denorm(pred_norm, loc, scale)
            true_vals = test_lbl[:horizon_length]

            err = relative_l2_error(true_vals, pred)
            test_errors.append(err)

            if (not EVALUATE_ALL_SERIES and eval_idx == SERIES_INDEX) or (
                EVALUATE_ALL_SERIES and not plotted
            ):
                _plot_forecast(
                    true_vals, pred,
                    title=f"{DATASET_NAME} test window 0 (series {eval_idx})",
                )
                plotted = True

        if test_errors:
            arr = np.asarray(test_errors, dtype=float)
            print(f"Test evaluated       : {len(test_errors)} series")
            print(f"Test skipped         : {len(test_skipped)} series")
            print(f"Mean  relative L2    : {float(np.mean(arr)):.6f}")
            print(f"Median relative L2   : {float(np.median(arr)):.6f}")
        else:
            print("No test forecasts could be evaluated.")
            for s in test_skipped:
                print(" ", s)
        """
    ).strip()


def uncertainty_cell() -> str:
    return dedent(
        """
        # ── Posterior uncertainty for the selected series ─────────────────
        print("Computing posterior uncertainty for series", SERIES_INDEX, "…")
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)

        val_arr = _series_from_entry(_get_validation_entry(SERIES_INDEX))
        if NORMALIZATION == "robust_series":
            loc_u = float(np.median(val_arr))
            q75_u, q25_u = float(np.percentile(val_arr, 75)), float(np.percentile(val_arr, 25))
            scale_u = max((q75_u - q25_u) / 1.349, float(np.std(val_arr)), 1e-6)
        else:
            loc_u, scale_u = 0.0, 1.0

        test_ctx_entry, test_lbl_entry = _get_test_pair(SERIES_INDEX, 0)
        test_ctx = _series_from_entry(test_ctx_entry)
        test_lbl = _series_from_entry(test_lbl_entry)

        query_raw = test_ctx[-CONTEXT_LENGTH:]
        query_norm = (query_raw - loc_u) / scale_u

        pred_norm, sigma2 = forecaster_val.predict_with_uncertainty(query_norm)
        pred = _denorm(pred_norm, loc_u, scale_u)
        true_vals = test_lbl[:horizon_length]
        std_denorm = float(np.sqrt(max(sigma2, 0.0))) * scale_u

        print(f"  posterior std (denormalised): {std_denorm:.4f}")

        t = np.arange(1, horizon_length + 1)
        plt.figure(figsize=(10, 4))
        plt.plot(t, true_vals, label="true", linewidth=2)
        plt.plot(t, pred, label="predicted", linewidth=2)
        plt.fill_between(
            t,
            pred - 2 * std_denorm,
            pred + 2 * std_denorm,
            alpha=0.25,
            label="±2σ",
        )
        plt.title(f"{DATASET_NAME} series {SERIES_INDEX} — posterior uncertainty")
        plt.xlabel("horizon step")
        plt.ylabel("value")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.show()
        """
    ).strip()


def _md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src,
    }


def _code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def make_notebook(dataset_name: str, row: dict, krr_preset: dict) -> dict:
    cells = [
        _md_cell(
            dedent(
                f"""
                # Nested Kriging Playground: `{dataset_name}`

                Two-layer nested kriging with Gibbs kernel, whose lengthscale is
                itself learned via a nested kriging process (Layer 1).

                **Architecture at a glance**
                - **Layer 1** (`fit_gibbs_lengthscale_field`): partition pooled windows into
                  small blocks → local RBF-MLE per block → heteroscedastic meta-GP over block
                  centroids → ℓ̃(x) = exp(ĝ(x) + v(x)/2).
                - **Layer 2** (`NestedKrigingForecaster`): partition into ≤ `MAX_BLOCKS_L2` blocks
                  → local Gibbs kriging per block → Rullière et al. cross-covariance aggregation.

                Edit the parameter cell freely. Fit cells are separated so you can
                re-run only the parts you want.
                """
            ).strip()
        ),
        _code_cell(setup_cell()),
        _code_cell(import_cell()),
        _code_cell(parameter_cell(dataset_name, row, krr_preset)),
        _code_cell(metadata_cell()),
        _code_cell(helper_cell()),
        _code_cell(preview_cell()),
        _md_cell("## Layer 1 — Fit Lengthscale Field (training split)"),
        _code_cell(fit_l1_cell()),
        _md_cell("## Layer 2 — Fit Nested Kriging Regression (training split)"),
        _code_cell(fit_l2_cell()),
        _md_cell("## Short-Term Evaluation: train → validation \\ train"),
        _code_cell(short_term_cell()),
        _md_cell("## Re-Fit on Validation Split (for test evaluation)"),
        _code_cell(long_term_cell()),
        _md_cell("## Long-Term Evaluation: direct forecast on test window 0"),
        _code_cell(test_eval_cell()),
        _md_cell("## Posterior Uncertainty: selected series"),
        _code_cell(uncertainty_cell()),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "cells": cells,
    }


def make_readme(dataset_names: list[str]) -> str:
    lines = [
        "# Univariate Nested Kriging Playgrounds",
        "",
        "Generated by `experiments/generate_univariate_nested_kriging_playgrounds.py`.",
        "",
        "Each notebook uses `src/nested_kriging.py` (two-layer: Gibbs kernel + learned",
        "lengthscale field) for univariate, no-missing GIFT-Eval datasets.",
        "",
        "## Notebooks",
        "",
    ]
    for ds in dataset_names:
        lines.append(f"- [{ds}](./{notebook_filename(ds)})")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    all_rows: list[dict] = json.loads(AUDIT_JSON.read_text())
    krr_presets: dict = json.loads(KRR_PRESETS_JSON.read_text())

    # Filter: univariate (target_dim == 1) and no missing data
    rows = [
        r for r in all_rows
        if not r["any_missing_raw"] and int(r["target_dim"]) == 1
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    for row in rows:
        dataset_name = row["dataset_name"]
        krr_preset = krr_presets.get(dataset_name, {
            "context_length": 2 * int(row["prediction_length"]),
            "max_series": 200,
            "max_windows_per_series": 20,
        })
        nb = make_notebook(dataset_name, row, krr_preset)
        path = OUTPUT_DIR / notebook_filename(dataset_name)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        generated.append(dataset_name)
        print(f"  wrote {path.name}")

    README_PATH.write_text(make_readme(generated))
    print(f"\nGenerated {len(generated)} notebooks in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
