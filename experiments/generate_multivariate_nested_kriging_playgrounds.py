"""
Generate Jupyter notebooks for multivariate GIFT-Eval datasets using the
MultivariateNestedKrigingForecaster (src/nested_kriging.py).

One notebook per dataset/frequency; output goes to
experiments/multivariate_nested_kriging_playgrounds/.

Each notebook:
  - Pools windows channel-by-channel from the (T, p) multivariate training series.
  - Fits a shared Gibbs lengthscale field on the pooled cross-channel windows.
  - Fits one NestedKrigingForecaster per output channel (shared ell_field).
  - Evaluates short-term (train→val) and long-term (test window 0) per channel.

Usage:
    python experiments/generate_multivariate_nested_kriging_playgrounds.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from textwrap import dedent


ROOT = Path("/Users/jonghyeonlee/KMD_TS")
OUTPUT_DIR = ROOT / "experiments" / "multivariate_nested_kriging_playgrounds"
README_PATH = OUTPUT_DIR / "README.md"

# ---------------------------------------------------------------------------
# Dataset catalogue (derived from dataset_info.json + data.py PRED_LENGTH_MAP)
# ---------------------------------------------------------------------------
# Format: (dataset_name, target_dim, n_train_series, pred_len_short, has_missing)
# has_missing = True means use pool_windows_from_multivariate_series_with_missing

PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}

MULTIVARIATE_DATASETS = [
    # (name, target_dim, n_series, freq_key, has_missing)
    ("bitbrains_fast_storage/5T", 2,  1250, "T",  False),
    ("bitbrains_fast_storage/H",  2,  1250, "H",  False),
    ("bitbrains_rnd/5T",          2,   500, "T",  False),
    ("bitbrains_rnd/H",           2,   500, "H",  False),
    ("bizitobs_application",      2,     1, "S",  False),
    ("bizitobs_l2c/5T",           7,     1, "T",  False),
    ("bizitobs_l2c/H",            7,     1, "H",  False),
    ("bizitobs_service",          2,    21, "S",  False),
    ("ett1/15T",                  7,     1, "T",  False),
    ("ett1/D",                    7,     1, "D",  False),
    ("ett1/H",                    7,     1, "H",  False),
    ("ett1/W",                    7,     1, "W",  False),
    ("ett2/15T",                  7,     1, "T",  False),
    ("ett2/D",                    7,     1, "D",  False),
    ("ett2/H",                    7,     1, "H",  False),
    ("ett2/W",                    7,     1, "W",  False),
    ("jena_weather/10T",         21,     1, "T",  False),
    ("jena_weather/D",           21,     1, "D",  False),
    ("jena_weather/H",           21,     1, "H",  False),
]


def pred_len(freq_key: str) -> int:
    return PRED_LENGTH_MAP.get(freq_key, 48)


def context_length_heuristic(p: int, H: int) -> int:
    """Reasonable per-channel context length.

    Larger p → shorter context to keep pL manageable for the kernel.
    """
    if p <= 2:
        cl = max(2 * H, 48)
    elif p <= 7:
        cl = max(2 * H, 32)
    else:                          # p = 21 (jena_weather)
        cl = max(H, 24)
    return min(cl, 200)


def block_size_l1_heuristic(n_series: int, H: int, context_length: int) -> int:
    """Heuristic: aim for ~30 anchors with at least 10 windows per block."""
    # Conservative estimate of windows per series
    estimated_windows_per = max(1, 20)
    total_est = estimated_windows_per * n_series
    return max(10, min(80, total_est // 30))


def max_series_heuristic(n_series: int) -> int:
    """Cap number of series loaded to keep notebooks interactive."""
    return min(n_series, 200)


def max_windows_heuristic(n_series: int) -> int:
    return 20 if n_series > 50 else 50


# ---------------------------------------------------------------------------
# Notebook cell generators
# ---------------------------------------------------------------------------


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


def import_cell(has_missing: bool) -> str:
    pool_fn = (
        "pool_windows_from_multivariate_series_with_missing"
        if has_missing
        else "pool_windows_from_multivariate_series"
    )
    return dedent(
        f"""
        import numpy as np
        import matplotlib.pyplot as plt

        from gift_eval.data import Dataset
        from src.nested_kriging import (
            LengthscaleField,
            MultivariateNestedKrigingForecaster,
            fit_gibbs_lengthscale_field,
            {pool_fn} as pool_windows_mv,
        )
        """
    ).strip()


def parameter_cell(
    dataset_name: str,
    target_dim: int,
    n_series: int,
    pred_len_short: int,
    has_missing: bool,
) -> str:
    CL = context_length_heuristic(target_dim, pred_len_short)
    BSL1 = block_size_l1_heuristic(n_series, pred_len_short, CL)
    MAX_S = max_series_heuristic(n_series)
    MAX_W = max_windows_heuristic(n_series)
    missing_note = (
        "  # NaN-aware pooler (use pool_windows_from_multivariate_series for clean data)"
        if has_missing
        else "  # No-missing pooler (switch to with_missing variant if NaNs detected)"
    )

    return dedent(
        f"""
        DATASET_NAME = "{dataset_name}"
        TERM = "short"
        TO_UNIVARIATE = False       # must be False for multivariate datasets

        # ── Dataset properties (read-only — change TERM if you want medium/long) ──
        TARGET_DIM = {target_dim}               # p: number of output channels

        # ── Series to preview ──────────────────────────────────────────────────
        SERIES_INDEX = 0
        CHANNEL_INDEX = 0           # which channel to plot in preview / uncertainty cells

        # ── Window bank controls ───────────────────────────────────────────────
        MAX_SERIES = {MAX_S}
        MAX_WINDOWS_PER_SERIES = {MAX_W}

        CONTEXT_LENGTH = {CL}       # L: per-channel context window length
        HORIZON_LENGTH = None       # H: defaults to dataset.prediction_length

        NORMALIZATION = "robust_series"  # per-channel robust normalisation{missing_note}
        MIN_OBS_FRACTION = 0.5      # only used by the with_missing pooler

        # ── Layer 1 — lengthscale field (shared across all channels) ───────────
        BLOCK_SIZE_L1 = {BSL1}      # windows per block for local ℓ̂_m MLE
        SIGMA_N_L1 = 0.1
        AMPLITUDE_L1 = 1.0
        LOG_ELL_LO = -3.0
        LOG_ELL_HI = 5.0
        META_AMPLITUDE = 1.0
        META_BANDWIDTH = None       # None → median-heuristic bandwidth

        # ── Layer 2 — MultivariateNestedKrigingForecaster ─────────────────────
        BLOCK_SIZE_L2 = 64          # target block size per channel
        MAX_BLOCKS_L2 = 30          # cap on M per channel
        SIGMA_N_L2 = 0.1
        AMPLITUDE_L2 = 1.0

        # ── Evaluation ─────────────────────────────────────────────────────────
        EVALUATE_ALL_SERIES = False
        EVAL_SERIES_LIMIT = None
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
        assert dataset.target_dim == TARGET_DIM, (
            f"Expected target_dim={TARGET_DIM} but got {dataset.target_dim}"
        )
        """
    ).strip()


def helper_cell() -> str:
    return dedent(
        """
        # ── Data access helpers (multivariate-aware) ──────────────────────────

        def _nth_item(data_it, index: int):
            for i, item in enumerate(data_it):
                if i == index:
                    return item
            raise IndexError(f"Index {index} is out of range.")


        def _series_from_entry_mv(entry) -> np.ndarray:
            \"\"\"Return (T, p) array from a GluonTS DataEntry.

            gift_eval stores multivariate targets as (p, T) — transpose to (T, p).
            \"\"\"
            tgt = np.asarray(entry["target"], dtype=float)
            if tgt.ndim == 1:
                return tgt[:, None]   # (T, 1) for safety
            return tgt.T              # (p, T) → (T, p)


        def _get_train_entry(series_index: int):
            return _nth_item(dataset.training_dataset, series_index)


        def _get_validation_entry(series_index: int):
            return _nth_item(dataset.validation_dataset, series_index)


        def _get_test_pair(series_index: int, window_index: int):
            if window_index < 0 or window_index >= dataset.windows:
                raise IndexError(
                    f"window_index={window_index} out of range "
                    f"(dataset.windows={dataset.windows})"
                )
            flat_index = series_index * dataset.windows + window_index
            test_ctx  = _nth_item(dataset.test_data.input, flat_index)
            test_lbl  = _nth_item(dataset.test_data.label, flat_index)
            return test_ctx, test_lbl


        def _collect_train_series_mv(*, max_series: int | None) -> list[np.ndarray]:
            \"\"\"Return list of (T, p) arrays from the training split.\"\"\"
            out = []
            for i, entry in enumerate(dataset.training_dataset):
                if max_series is not None and i >= max_series:
                    break
                arr = _series_from_entry_mv(entry)
                if arr.size > 0:
                    out.append(arr)
            return out


        def _collect_val_series_mv(*, max_series: int | None) -> list[np.ndarray]:
            out = []
            for i, entry in enumerate(dataset.validation_dataset):
                if max_series is not None and i >= max_series:
                    break
                arr = _series_from_entry_mv(entry)
                if arr.size > 0:
                    out.append(arr)
            return out


        def _denorm(pred_norm: np.ndarray, loc: float, scale: float) -> np.ndarray:
            return pred_norm * scale + loc


        def relative_l2_error(true: np.ndarray, pred: np.ndarray) -> float:
            t = np.asarray(true, dtype=float).ravel()
            p = np.asarray(pred, dtype=float).ravel()
            denom = float(np.sum(t ** 2))
            if denom < 1e-12:
                return float("nan")
            return float(np.sum((t - p) ** 2) / denom)


        def _plot_channel_forecast(y_true, y_pred, *, channel: int, title: str):
            t = np.arange(1, len(y_true) + 1)
            plt.figure(figsize=(10, 4))
            plt.plot(t, y_true, label="true", linewidth=2)
            plt.plot(t, y_pred, label="predicted", linewidth=2)
            plt.title(f"{title}  [channel {channel}]")
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
        # ── Preview: selected series, selected channel ────────────────────────
        train_entry = _get_train_entry(SERIES_INDEX)
        val_entry   = _get_validation_entry(SERIES_INDEX)
        train_mv    = _series_from_entry_mv(train_entry)   # (T_train, p)
        val_mv      = _series_from_entry_mv(val_entry)     # (T_val, p)
        p_actual    = train_mv.shape[1]

        print(f"Series {SERIES_INDEX}: train shape = {train_mv.shape}, val shape = {val_mv.shape}")
        print(f"Target channels (p) = {p_actual}")

        val_extra = val_mv[train_mv.shape[0]:]            # (val_train len, p)

        ch = min(CHANNEL_INDEX, p_actual - 1)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.arange(val_mv.shape[0]), val_mv[:, ch],
                linewidth=1.2, label=f"channel {ch} (up to validation)")
        ax.axvline(train_mv.shape[0], color="tab:red", linestyle="--", label="train boundary")
        ax.set_title(f"{DATASET_NAME}  series {SERIES_INDEX}  channel {ch}")
        ax.set_xlabel("time index")
        ax.set_ylabel("value")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
        """
    ).strip()


def fit_l1_cell(has_missing: bool) -> str:
    extra_kwargs = (
        "\n    min_observed_fraction=MIN_OBS_FRACTION,"
        if has_missing
        else ""
    )
    return dedent(
        f"""
        rng = np.random.default_rng(42)
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)

        # ── Collect training series ─────────────────────────────────────────
        print("Collecting training series …")
        train_series_list = _collect_train_series_mv(max_series=MAX_SERIES)
        print(f"  {{len(train_series_list)}} series, shape[0] example: "
              f"{{train_series_list[0].shape if train_series_list else 'N/A'}}")

        # ── Pool windows cross-channel (all channels pooled together) ───────
        print("Pooling cross-channel windows …")
        pool_kwargs = dict(
            context_length=CONTEXT_LENGTH,
            horizon_length=horizon_length,
            stride=horizon_length,
            max_windows_per_series=MAX_WINDOWS_PER_SERIES,
            normalization=NORMALIZATION,{extra_kwargs}
            rng=rng,
        )
        ctx_train, tgt_train, meta_train, n_channels = pool_windows_mv(
            train_series_list, **pool_kwargs
        )
        print(f"  pooled {{ctx_train.shape[0]}} windows across {{n_channels}} channels")
        print(f"  context shape: {{ctx_train.shape}}, target shape: {{tgt_train.shape}}")

        # Per-channel window counts
        ch_idx = [m["channel_idx"] for m in meta_train]
        for ch in range(n_channels):
            cnt = sum(c == ch for c in ch_idx)
            print(f"    channel {{ch}}: {{cnt}} windows")

        # ── Layer 1: fit shared lengthscale field ───────────────────────────
        print("\\nFitting Layer 1 (shared Gibbs lengthscale field) …")
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
        print(f"  anchors (M_1)  : {{ell_field.n_anchors}}")
        print(f"  meta_bandwidth : {{ell_field.meta_bandwidth:.4f}}")
        sample_ell = ell_field.predict_ell(ctx_train[:5])
        print(f"  sample ℓ̃ values: {{sample_ell}}")
        """
    ).strip()


def fit_l2_cell() -> str:
    return dedent(
        """
        # ── Layer 2: fit MultivariateNestedKrigingForecaster ────────────────
        print("Fitting Layer 2 (MultivariateNestedKrigingForecaster) …")
        mv_forecaster = MultivariateNestedKrigingForecaster(
            ell_field,
            n_channels=n_channels,
            block_size=BLOCK_SIZE_L2,
            max_blocks=MAX_BLOCKS_L2,
            sigma_n=SIGMA_N_L2,
            amplitude=AMPLITUDE_L2,
            rng=rng,
        ).fit(ctx_train, tgt_train, meta_train)

        for ch, fc in enumerate(mv_forecaster._forecasters):
            if fc._fitted_blocks:
                n_blk = len(fc._fitted_blocks)
                sizes = [len(b.indices) for b in fc._fitted_blocks]
                print(f"  channel {ch}: {n_blk} blocks, "
                      f"sizes min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f}")
            else:
                print(f"  channel {ch}: no blocks fitted (insufficient windows)")
        """
    ).strip()


def short_term_cell() -> str:
    return dedent(
        """
        # ── Short-term evaluation: train → validation\\train ──────────────────
        print("Short-term evaluation (train → validation\\\\train) …")
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)
        n_series_total = len(dataset.hf_dataset)

        all_errors = []     # (channel, series) relative L2 errors
        plotted = False

        for eval_idx in _evaluation_series_indices(n_series_total):
            train_arr = _series_from_entry_mv(_get_train_entry(eval_idx))   # (T, p)
            val_arr   = _series_from_entry_mv(_get_validation_entry(eval_idx))
            T_train = train_arr.shape[0]
            val_extra = val_arr[T_train:]                               # (val_extra, p)

            if T_train < CONTEXT_LENGTH or val_extra.shape[0] < horizon_length:
                continue

            p = train_arr.shape[1]
            # Build (p, L) query from last L steps of each training channel
            query = np.stack(
                [train_arr[-CONTEXT_LENGTH:, ch] for ch in range(p)], axis=0
            )   # (p, L)

            # Normalise: per-channel, using training stats
            query_norm = np.empty_like(query)
            locs, scales = [], []
            for ch in range(p):
                ch_arr = train_arr[:, ch]
                if NORMALIZATION == "robust_series":
                    loc = float(np.median(ch_arr))
                    q75, q25 = float(np.percentile(ch_arr, 75)), float(np.percentile(ch_arr, 25))
                    scale = max((q75 - q25) / 1.349, float(np.std(ch_arr)), 1e-6)
                else:
                    loc, scale = 0.0, 1.0
                query_norm[ch] = (query[ch] - loc) / scale
                locs.append(loc); scales.append(scale)

            pred_norm = mv_forecaster.predict(query_norm)   # (p, H)

            series_errors = []
            for ch in range(p):
                pred_ch = _denorm(pred_norm[ch], locs[ch], scales[ch])
                true_ch = val_extra[:horizon_length, ch]
                err = relative_l2_error(true_ch, pred_ch)
                series_errors.append(err)
                all_errors.append(err)

            if (not EVALUATE_ALL_SERIES and eval_idx == SERIES_INDEX) or (
                EVALUATE_ALL_SERIES and not plotted
            ):
                ch_plot = min(CHANNEL_INDEX, p - 1)
                _plot_channel_forecast(
                    val_extra[:horizon_length, ch_plot],
                    _denorm(pred_norm[ch_plot], locs[ch_plot], scales[ch_plot]),
                    channel=ch_plot,
                    title=f"{DATASET_NAME} short-term (series {eval_idx}): train → val\\\\train",
                )
                plotted = True

        if all_errors:
            arr = np.asarray(all_errors, dtype=float)
            print(f"Short-term evaluated : {len(all_errors)} (series × channel) pairs")
            print(f"Mean  relative L2    : {float(np.nanmean(arr)):.6f}")
            print(f"Median relative L2   : {float(np.nanmedian(arr)):.6f}")
        else:
            print("No short-term forecasts could be evaluated.")
        """
    ).strip()


def refit_val_cell(has_missing: bool) -> str:
    extra_kwargs = (
        "\n    min_observed_fraction=MIN_OBS_FRACTION,"
        if has_missing
        else ""
    )
    return dedent(
        f"""
        # ── Re-fit on validation split (for test evaluation) ────────────────
        print("Collecting validation series …")
        val_series_list = _collect_val_series_mv(max_series=MAX_SERIES)
        print(f"  {{len(val_series_list)}} series")
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)

        print("Pooling validation windows …")
        pool_kwargs_val = dict(
            context_length=CONTEXT_LENGTH,
            horizon_length=horizon_length,
            stride=horizon_length,
            max_windows_per_series=MAX_WINDOWS_PER_SERIES,
            normalization=NORMALIZATION,{extra_kwargs}
            rng=rng,
        )
        ctx_val, tgt_val, meta_val, _ = pool_windows_mv(
            val_series_list, **pool_kwargs_val
        )
        print(f"  pooled {{ctx_val.shape[0]}} windows")

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
        print(f"  anchors: {{ell_field_val.n_anchors}}")

        print("Fitting Layer 2 on validation split …")
        mv_forecaster_val = MultivariateNestedKrigingForecaster(
            ell_field_val,
            n_channels=n_channels,
            block_size=BLOCK_SIZE_L2,
            max_blocks=MAX_BLOCKS_L2,
            sigma_n=SIGMA_N_L2,
            amplitude=AMPLITUDE_L2,
            rng=rng,
        ).fit(ctx_val, tgt_val, meta_val)
        print("Done.")
        """
    ).strip()


def test_eval_cell() -> str:
    return dedent(
        """
        # ── Long-term test evaluation (test window 0) ─────────────────────────
        print("Long-term test evaluation …")
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)
        n_series_total = len(dataset.hf_dataset)

        all_test_errors = []
        plotted = False

        for eval_idx in _evaluation_series_indices(n_series_total):
            val_arr = _series_from_entry_mv(_get_validation_entry(eval_idx))

            if val_arr.shape[0] < CONTEXT_LENGTH:
                continue

            try:
                test_ctx_entry, test_lbl_entry = _get_test_pair(eval_idx, 0)
            except IndexError as e:
                print(f"  skip series {eval_idx}: {e}")
                continue

            test_ctx_mv = _series_from_entry_mv(test_ctx_entry)   # (T_ctx, p)
            test_lbl_mv = _series_from_entry_mv(test_lbl_entry)   # (H_lbl, p)

            p = test_ctx_mv.shape[1]
            if test_ctx_mv.shape[0] < CONTEXT_LENGTH or test_lbl_mv.shape[0] < horizon_length:
                continue

            # Per-channel normalisation using validation stats
            query_norm = np.empty((p, CONTEXT_LENGTH), dtype=float)
            locs, scales = [], []
            for ch in range(p):
                ch_val = val_arr[:, ch]
                if NORMALIZATION == "robust_series":
                    loc = float(np.median(ch_val))
                    q75, q25 = float(np.percentile(ch_val, 75)), float(np.percentile(ch_val, 25))
                    scale = max((q75 - q25) / 1.349, float(np.std(ch_val)), 1e-6)
                else:
                    loc, scale = 0.0, 1.0
                raw_q = test_ctx_mv[-CONTEXT_LENGTH:, ch]
                query_norm[ch] = (raw_q - loc) / scale
                locs.append(loc); scales.append(scale)

            pred_norm = mv_forecaster_val.predict(query_norm)   # (p, H)

            for ch in range(p):
                pred_ch = _denorm(pred_norm[ch], locs[ch], scales[ch])
                true_ch = test_lbl_mv[:horizon_length, ch]
                err = relative_l2_error(true_ch, pred_ch)
                all_test_errors.append(err)

            if (not EVALUATE_ALL_SERIES and eval_idx == SERIES_INDEX) or (
                EVALUATE_ALL_SERIES and not plotted
            ):
                ch_plot = min(CHANNEL_INDEX, p - 1)
                _plot_channel_forecast(
                    test_lbl_mv[:horizon_length, ch_plot],
                    _denorm(pred_norm[ch_plot], locs[ch_plot], scales[ch_plot]),
                    channel=ch_plot,
                    title=f"{DATASET_NAME} test window 0 (series {eval_idx})",
                )
                plotted = True

        if all_test_errors:
            arr = np.asarray(all_test_errors, dtype=float)
            print(f"Test evaluated       : {len(all_test_errors)} (series × channel) pairs")
            print(f"Mean  relative L2    : {float(np.nanmean(arr)):.6f}")
            print(f"Median relative L2   : {float(np.nanmedian(arr)):.6f}")
        else:
            print("No test forecasts could be evaluated.")
        """
    ).strip()


def uncertainty_cell() -> str:
    return dedent(
        """
        # ── Posterior uncertainty for selected series + channel ───────────────
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)
        print(f"Posterior uncertainty: series {SERIES_INDEX}, channel {CHANNEL_INDEX} …")

        val_arr = _series_from_entry_mv(_get_validation_entry(SERIES_INDEX))
        p = val_arr.shape[1]
        ch = min(CHANNEL_INDEX, p - 1)

        try:
            test_ctx_entry, test_lbl_entry = _get_test_pair(SERIES_INDEX, 0)
        except IndexError as e:
            print(f"Cannot compute uncertainty: {e}")
        else:
            test_ctx_mv = _series_from_entry_mv(test_ctx_entry)
            test_lbl_mv = _series_from_entry_mv(test_lbl_entry)

            # Per-channel normalisation using validation stats
            query_norm = np.empty((p, CONTEXT_LENGTH), dtype=float)
            locs, scales = [], []
            for c in range(p):
                ch_val = val_arr[:, c]
                if NORMALIZATION == "robust_series":
                    loc = float(np.median(ch_val))
                    q75, q25 = float(np.percentile(ch_val, 75)), float(np.percentile(ch_val, 25))
                    scale = max((q75 - q25) / 1.349, float(np.std(ch_val)), 1e-6)
                else:
                    loc, scale = 0.0, 1.0
                raw_q = test_ctx_mv[-CONTEXT_LENGTH:, c]
                query_norm[c] = (raw_q - loc) / scale
                locs.append(loc); scales.append(scale)

            # means: (p, H), variances: (p,)
            means_norm, variances = mv_forecaster_val.predict_with_uncertainty(query_norm)
            pred_ch = _denorm(means_norm[ch], locs[ch], scales[ch])
            std_denorm = float(np.sqrt(max(variances[ch], 0.0))) * scales[ch]
            true_ch = test_lbl_mv[:horizon_length, ch]

            print(f"  posterior std (denormalised): {std_denorm:.4f}")

            t = np.arange(1, horizon_length + 1)
            plt.figure(figsize=(10, 4))
            plt.plot(t, true_ch, label="true", linewidth=2)
            plt.plot(t, pred_ch, label="predicted", linewidth=2)
            plt.fill_between(
                t,
                pred_ch - 2 * std_denorm,
                pred_ch + 2 * std_denorm,
                alpha=0.25,
                label="±2σ",
            )
            plt.title(f"{DATASET_NAME}  series {SERIES_INDEX}  channel {ch} — posterior uncertainty")
            plt.xlabel("horizon step")
            plt.ylabel("value")
            plt.legend()
            plt.grid(alpha=0.25)
            plt.show()
        """
    ).strip()


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------


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


def make_notebook(
    dataset_name: str,
    target_dim: int,
    n_series: int,
    pred_len_short: int,
    has_missing: bool,
) -> dict:
    title = f"# Multivariate Nested Kriging Playground: `{dataset_name}`"
    description = dedent(
        f"""
        {title}

        **Dataset:** `{dataset_name}` | **target_dim (p):** {target_dim} |
        **prediction length (short):** {pred_len_short}

        **Architecture:** Two-layer nested kriging with a *shared* Gibbs
        lengthscale field across all {target_dim} output channels.

        - **Layer 1** (`fit_gibbs_lengthscale_field`): fitted on pooled cross-channel
          windows → shared ℓ̃(x) field.
        - **Layer 2** (`MultivariateNestedKrigingForecaster`): one
          `NestedKrigingForecaster` per output channel, each using the shared ℓ̃
          field via the Rullière et al. aggregation.

        Edit the parameter cell freely. Cells are separated so you can re-run
        only the parts you want.
        """
    ).strip()

    cells = [
        _md_cell(description),
        _code_cell(setup_cell()),
        _code_cell(import_cell(has_missing)),
        _code_cell(
            parameter_cell(
                dataset_name, target_dim, n_series, pred_len_short, has_missing
            )
        ),
        _code_cell(metadata_cell()),
        _code_cell(helper_cell()),
        _code_cell(preview_cell()),
        _md_cell("## Layer 1 — Fit Shared Lengthscale Field (training split)"),
        _code_cell(fit_l1_cell(has_missing)),
        _md_cell("## Layer 2 — Fit MultivariateNestedKrigingForecaster (training split)"),
        _code_cell(fit_l2_cell()),
        _md_cell("## Short-Term Evaluation: train → validation \\ train"),
        _code_cell(short_term_cell()),
        _md_cell("## Re-Fit on Validation Split (for test evaluation)"),
        _code_cell(refit_val_cell(has_missing)),
        _md_cell("## Long-Term Evaluation: direct forecast on test window 0"),
        _code_cell(test_eval_cell()),
        _md_cell("## Posterior Uncertainty: selected series + channel"),
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


def notebook_filename(dataset_name: str) -> str:
    safe = dataset_name.replace("/", "__").replace(" ", "_").replace("-", "_")
    return f"{safe}_mv_nested_kriging.ipynb"


def make_readme(entries: list[tuple]) -> str:
    lines = [
        "# Multivariate Nested Kriging Playgrounds",
        "",
        "Generated by `experiments/generate_multivariate_nested_kriging_playgrounds.py`.",
        "",
        "Each notebook uses `src/nested_kriging.py` — two-layer nested kriging with a",
        "shared Gibbs lengthscale field and one `NestedKrigingForecaster` per output channel.",
        "",
        "## Notebooks",
        "",
    ]
    for name, target_dim, _, pred_len_s, _ in entries:
        lines.append(
            f"- [{name}](./{notebook_filename(name)}) "
            f"(p={target_dim}, pred_len={pred_len_s})"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    for name, target_dim, n_series, freq_key, has_missing in MULTIVARIATE_DATASETS:
        H = pred_len(freq_key)
        nb = make_notebook(name, target_dim, n_series, H, has_missing)
        path = OUTPUT_DIR / notebook_filename(name)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        generated.append((name, target_dim, n_series, H, has_missing))
        print(f"  wrote {path.name}")

    README_PATH.write_text(make_readme(generated))
    print(f"\nGenerated {len(generated)} notebooks in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
