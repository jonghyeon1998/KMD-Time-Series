"""
Generate Jupyter notebooks for univariate GIFT-Eval datasets that contain
missing values, using the nested kriging forecaster (src/nested_kriging.py).

Two distinct notebook templates are used:

1. car_parts_with_missing  →  TwoModelIntermittentForecaster
   Bernoulli indicator GP (Model 1) + log-Gaussian magnitude GP (Model 2).
   Uses pool_intermittent_windows.

2. All other missing datasets  →  NestedKrigingForecaster
   NaN gaps are linearly imputed before windowing.
   Uses pool_windows_from_series_with_missing.

Usage:
    python experiments/generate_univariate_missing_nested_kriging_playgrounds.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from textwrap import dedent

ROOT = Path("/Users/jonghyeonlee/KMD_TS")
AUDIT_JSON       = ROOT / "experiments" / "gift_eval_univariate_split_audit.json"
KRR_PRESETS_JSON = ROOT / "experiments" / "window_krr_default_presets.json"
OUTPUT_DIR       = ROOT / "experiments" / "univariate_missing_nested_kriging_playgrounds"
README_PATH      = OUTPUT_DIR / "README.md"

# Datasets treated as intermittent demand (two-model approach)
INTERMITTENT = {"car_parts_with_missing"}


# ---------------------------------------------------------------------------
# Notebook helpers
# ---------------------------------------------------------------------------

def notebook_filename(dataset_name: str) -> str:
    safe = dataset_name.replace("/", "__").replace(" ", "_").replace("-", "_")
    return f"{safe}_nested_kriging.ipynb"


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src}


def _code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8],
            "metadata": {}, "outputs": [], "source": src}


# ---------------------------------------------------------------------------
# Shared cells (both templates)
# ---------------------------------------------------------------------------

def setup_cell() -> str:
    return dedent("""
        import os, sys
        from pathlib import Path

        def _find_repo_root() -> Path:
            cwd = Path.cwd().resolve()
            for candidate in [cwd, *cwd.parents]:
                if (candidate / "src").exists() and (candidate / "gift_eval").exists():
                    return candidate
            raise RuntimeError("Could not locate KMD_TS repo root.")

        REPO_ROOT = _find_repo_root()
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        os.environ.setdefault("GIFT_EVAL", str(REPO_ROOT / "GiftEval_data"))
        print("Repo root:", REPO_ROOT)
    """).strip()


def metadata_cell() -> str:
    return dedent("""
        dataset = Dataset(
            name=DATASET_NAME, term=TERM, to_univariate=TO_UNIVARIATE,
            storage_env_var="GIFT_EVAL",
        )
        print("dataset_name       :", DATASET_NAME)
        print("frequency          :", dataset.freq)
        print("prediction_length  :", dataset.prediction_length)
        print("test windows/series:", dataset.windows)
        print("n_raw_series       :", len(dataset.hf_dataset))
        print("target_dim         :", dataset.target_dim)
    """).strip()


def shared_helpers_cell() -> str:
    return dedent("""
        def _nth(it, i):
            for j, x in enumerate(it):
                if j == i: return x
            raise IndexError(i)

        def _series(entry):
            return np.asarray(entry["target"], dtype=float).reshape(-1)

        def _train(i):   return _series(_nth(dataset.training_dataset,    i))
        def _val(i):     return _series(_nth(dataset.validation_dataset,   i))
        def _test(i, w):
            fi = i * dataset.windows + w
            ctx = _series(_nth(dataset.test_data.input, fi))
            lbl = _series(_nth(dataset.test_data.label, fi))
            return ctx, lbl

        def relative_l2(true, pred):
            t, p = np.asarray(true, float).ravel(), np.asarray(pred, float).ravel()
            d = float(np.sum(t**2))
            return float(np.sum((t-p)**2)/d) if d > 1e-12 else float("nan")

        def _plot(y_true, y_pred, *, title):
            t = np.arange(1, len(y_true)+1)
            plt.figure(figsize=(10,4))
            plt.plot(t, y_true, label="true", lw=2)
            plt.plot(t, y_pred, label="pred",  lw=2)
            plt.title(title); plt.xlabel("step"); plt.legend(); plt.grid(alpha=.25); plt.show()

        def _collect_series(split, *, max_s):
            it = dataset.training_dataset if split == "train" else dataset.validation_dataset
            out = []
            for i, entry in enumerate(it):
                if max_s is not None and i >= max_s: break
                out.append(_series(entry))
            return out

        def _eval_indices(n_total):
            if not EVALUATE_ALL_SERIES: return [SERIES_INDEX]
            lim = n_total if EVAL_SERIES_LIMIT is None else min(EVAL_SERIES_LIMIT, n_total)
            return list(range(max(1, lim)))
    """).strip()


# ---------------------------------------------------------------------------
# GAP-MISSING template cells
# ---------------------------------------------------------------------------

def gap_import_cell() -> str:
    return dedent("""
        import numpy as np
        import matplotlib.pyplot as plt
        from gift_eval.data import Dataset
        from src.nested_kriging import (
            LengthscaleField, NestedKrigingForecaster,
            fit_gibbs_lengthscale_field,
            pool_windows_from_series_with_missing,
        )
    """).strip()


def gap_parameter_cell(dataset_name: str, row: dict, krr_preset: dict) -> str:
    ctx = int(krr_preset.get("context_length", 2 * int(row["prediction_length"])))
    max_s = int(krr_preset.get("max_series", 200))
    max_w = int(krr_preset.get("max_windows_per_series", 20))
    bs_l1 = max(10, min(80, max_s * max_w // 30))
    return dedent(f"""
        DATASET_NAME = "{dataset_name}"
        TERM = "short"
        TO_UNIVARIATE = False
        SERIES_INDEX = 0

        MAX_SERIES              = {max_s}
        MAX_WINDOWS_PER_SERIES  = {max_w}
        MIN_OBSERVED_FRACTION   = 0.5   # min fraction of non-NaN entries in context

        CONTEXT_LENGTH = {ctx}
        HORIZON_LENGTH = None           # defaults to dataset.prediction_length

        NORMALIZATION = "robust_series"

        # Layer 1 (lengthscale field)
        BLOCK_SIZE_L1    = {bs_l1}
        SIGMA_N_L1       = 0.1
        AMPLITUDE_L1     = 1.0
        LOG_ELL_LO       = -3.0
        LOG_ELL_HI       = 5.0
        META_AMPLITUDE   = 1.0
        META_BANDWIDTH   = None

        # Layer 2 (nested kriging regression)
        BLOCK_SIZE_L2 = 100
        MAX_BLOCKS_L2 = 30
        SIGMA_N_L2    = 0.1
        AMPLITUDE_L2  = 1.0

        EVALUATE_ALL_SERIES = False
        EVAL_SERIES_LIMIT   = None
    """).strip()


def gap_fit_l1_cell() -> str:
    return dedent("""
        rng = np.random.default_rng(42)
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)

        print("Collecting training series …")
        train_series = _collect_series("train", max_s=MAX_SERIES)
        print(f"  {len(train_series)} series")

        print("Pooling windows (with NaN imputation) …")
        ctx_train, tgt_train, meta_train = pool_windows_from_series_with_missing(
            train_series,
            context_length=CONTEXT_LENGTH,
            horizon_length=horizon_length,
            stride=horizon_length,
            max_windows_per_series=MAX_WINDOWS_PER_SERIES,
            normalization=NORMALIZATION,
            min_observed_fraction=MIN_OBSERVED_FRACTION,
            rng=rng,
        )
        print(f"  pooled {ctx_train.shape[0]} windows, shape {ctx_train.shape}")

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
        print(f"  anchors (M_1): {ell_field.n_anchors}")
        print(f"  meta_bandwidth: {ell_field.meta_bandwidth:.4f}")
        print(f"  sample ℓ̃: {ell_field.predict_ell(ctx_train[:5]).round(4)}")
    """).strip()


def gap_fit_l2_cell() -> str:
    return dedent("""
        print("Fitting Layer 2 (nested kriging) …")
        forecaster = NestedKrigingForecaster(
            ell_field,
            block_size=BLOCK_SIZE_L2,
            max_blocks=MAX_BLOCKS_L2,
            sigma_n=SIGMA_N_L2,
            amplitude=AMPLITUDE_L2,
            rng=rng,
        ).fit(ctx_train, tgt_train)

        sizes = [len(b.indices) for b in forecaster._fitted_blocks]
        print(f"  blocks: {len(sizes)}, sizes: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f}")
    """).strip()


def gap_short_term_cell() -> str:
    return dedent("""
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)
        n_total = len(dataset.hf_dataset)
        errors, skipped, plotted = [], [], False

        for idx in _eval_indices(n_total):
            tr = _train(idx);  va = _val(idx)
            extra = va[len(tr):]
            if len(tr) < CONTEXT_LENGTH or len(extra) < horizon_length:
                skipped.append((idx, "too short")); continue

            obs = tr[np.isfinite(tr)]
            if len(obs) < 2: skipped.append((idx, "all NaN train")); continue
            if NORMALIZATION == "robust_series":
                loc = float(np.median(obs))
                q75, q25 = float(np.percentile(obs,75)), float(np.percentile(obs,25))
                scale = max((q75-q25)/1.349, float(np.std(obs)), 1e-6)
            else:
                loc, scale = 0.0, 1.0

            from src.nested_kriging import _linear_impute
            query_raw = tr[-CONTEXT_LENGTH:]
            query_norm = (_linear_impute(query_raw) - loc) / scale
            pred_norm = forecaster.predict(query_norm)
            pred = pred_norm * scale + loc

            true_h = extra[:horizon_length]
            obs_h = np.isfinite(true_h)
            if not obs_h.any(): skipped.append((idx, "all NaN val-extra")); continue
            err = relative_l2(true_h[obs_h], pred.ravel()[:horizon_length][obs_h])
            errors.append(err)

            if (not EVALUATE_ALL_SERIES and idx == SERIES_INDEX) or (EVALUATE_ALL_SERIES and not plotted):
                _plot(true_h, pred.ravel()[:horizon_length],
                      title=f"{DATASET_NAME} short-term (series {idx})")
                plotted = True

        if errors:
            a = np.asarray(errors, float)
            print(f"Evaluated: {len(errors)}, skipped: {len(skipped)}")
            print(f"Mean relL2: {np.mean(a):.6f}  Median: {np.median(a):.6f}")
        else:
            print("No short-term forecasts evaluated."); [print(" ", s) for s in skipped]
    """).strip()


def gap_refit_val_cell() -> str:
    return dedent("""
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)
        print("Collecting validation series …")
        val_series = _collect_series("validation", max_s=MAX_SERIES)

        print("Pooling validation windows …")
        ctx_val, tgt_val, meta_val = pool_windows_from_series_with_missing(
            val_series,
            context_length=CONTEXT_LENGTH,
            horizon_length=horizon_length,
            stride=horizon_length,
            max_windows_per_series=MAX_WINDOWS_PER_SERIES,
            normalization=NORMALIZATION,
            min_observed_fraction=MIN_OBSERVED_FRACTION,
            rng=rng,
        )
        print(f"  {ctx_val.shape[0]} windows")

        print("Fitting Layer 1 on validation …")
        ell_field_val = fit_gibbs_lengthscale_field(
            ctx_val, tgt_val,
            block_size_l1=BLOCK_SIZE_L1, sigma_n=SIGMA_N_L1, amplitude=AMPLITUDE_L1,
            log_ell_bounds=(LOG_ELL_LO, LOG_ELL_HI),
            meta_amplitude=META_AMPLITUDE, meta_bandwidth=META_BANDWIDTH, rng=rng,
        )
        print("Fitting Layer 2 on validation …")
        forecaster_val = NestedKrigingForecaster(
            ell_field_val, block_size=BLOCK_SIZE_L2, max_blocks=MAX_BLOCKS_L2,
            sigma_n=SIGMA_N_L2, amplitude=AMPLITUDE_L2, rng=rng,
        ).fit(ctx_val, tgt_val)
        print("Done.")
    """).strip()


def gap_test_eval_cell() -> str:
    return dedent("""
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)
        n_total = len(dataset.hf_dataset)
        errors, skipped, plotted = [], [], False

        from src.nested_kriging import _linear_impute
        for idx in _eval_indices(n_total):
            va = _val(idx)
            if len(va) < CONTEXT_LENGTH: skipped.append((idx, "val too short")); continue
            obs = va[np.isfinite(va)]
            if len(obs) < 2: skipped.append((idx, "all NaN val")); continue
            if NORMALIZATION == "robust_series":
                loc = float(np.median(obs))
                q75, q25 = float(np.percentile(obs,75)), float(np.percentile(obs,25))
                scale = max((q75-q25)/1.349, float(np.std(obs)), 1e-6)
            else: loc, scale = 0.0, 1.0

            try: ctx_e, lbl_e = _test(idx, 0)
            except IndexError as e: skipped.append((idx, str(e))); continue
            if len(ctx_e) < CONTEXT_LENGTH or len(lbl_e) < horizon_length:
                skipped.append((idx, "test ctx/lbl too short")); continue

            query_raw = ctx_e[-CONTEXT_LENGTH:]
            query_norm = (_linear_impute(query_raw) - loc) / scale
            pred_norm = forecaster_val.predict(query_norm)
            pred = pred_norm * scale + loc

            true_h = lbl_e[:horizon_length]
            obs_h = np.isfinite(true_h)
            if not obs_h.any(): skipped.append((idx, "all NaN test label")); continue
            errors.append(relative_l2(true_h[obs_h], pred.ravel()[:horizon_length][obs_h]))

            if (not EVALUATE_ALL_SERIES and idx == SERIES_INDEX) or (EVALUATE_ALL_SERIES and not plotted):
                _plot(true_h, pred.ravel()[:horizon_length],
                      title=f"{DATASET_NAME} test window 0 (series {idx})")
                plotted = True

        if errors:
            a = np.asarray(errors, float)
            print(f"Evaluated: {len(errors)}, skipped: {len(skipped)}")
            print(f"Mean relL2: {np.mean(a):.6f}  Median: {np.median(a):.6f}")
        else:
            print("No test forecasts evaluated."); [print(" ", s) for s in skipped]
    """).strip()


# ---------------------------------------------------------------------------
# INTERMITTENT (car_parts) template cells
# ---------------------------------------------------------------------------

def int_import_cell() -> str:
    return dedent("""
        import numpy as np
        import matplotlib.pyplot as plt
        from gift_eval.data import Dataset
        from src.nested_kriging import (
            LengthscaleField,
            BernoulliKrigingClassifier, NestedKrigingForecaster,
            TwoModelIntermittentForecaster,
            fit_bernoulli_lengthscale_field, fit_gibbs_lengthscale_field,
            pool_intermittent_windows,
        )
    """).strip()


def int_parameter_cell(dataset_name: str, row: dict, krr_preset: dict) -> str:
    ctx = int(krr_preset.get("context_length", 2 * int(row["prediction_length"])))
    max_s = int(krr_preset.get("max_series", 500))
    max_w = int(krr_preset.get("max_windows_per_series", 20))
    return dedent(f"""
        DATASET_NAME = "{dataset_name}"
        TERM = "short"
        TO_UNIVARIATE = False
        SERIES_INDEX = 0

        MAX_SERIES              = {max_s}
        MAX_WINDOWS_PER_SERIES  = {max_w}
        MIN_OBSERVED_FRACTION   = 0.5

        CONTEXT_LENGTH = {ctx}
        HORIZON_LENGTH = None           # defaults to dataset.prediction_length

        # ── Layer 1 indicator (Bernoulli MLE) ──────────────────────────────
        BLOCK_SIZE_L1_IND = 30          # smaller blocks → more anchors for binary surface
        AMPLITUDE_IND     = 1.0
        LOG_ELL_LO        = -3.0
        LOG_ELL_HI        = 5.0
        N_ITER_LAPLACE    = 20
        META_AMPLITUDE_IND = 1.0
        META_BANDWIDTH_IND = None

        # ── Layer 2 indicator (BernoulliKrigingClassifier) ─────────────────
        BLOCK_SIZE_L2_IND = 80
        MAX_BLOCKS_IND    = 30

        # ── Layer 1 magnitude (Gaussian MLE on log y) ─────────────────────
        BLOCK_SIZE_L1_MAG = 20          # magnitude pool is smaller; use small blocks
        SIGMA_N_MAG       = 0.1
        AMPLITUDE_MAG     = 1.0
        META_AMPLITUDE_MAG = 1.0
        META_BANDWIDTH_MAG = None

        # ── Layer 2 magnitude (NestedKrigingForecaster on log y) ───────────
        BLOCK_SIZE_L2_MAG = 50
        MAX_BLOCKS_MAG    = 20
        SIGMA_N_L2_MAG    = 0.1

        EVALUATE_ALL_SERIES = False
        EVAL_SERIES_LIMIT   = None
    """).strip()


def int_pool_cell() -> str:
    return dedent("""
        rng = np.random.default_rng(42)
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)

        print("Collecting training series …")
        train_raw = []
        for i, entry in enumerate(dataset.training_dataset):
            if i >= MAX_SERIES: break
            train_raw.append(np.asarray(entry["target"], dtype=float).reshape(-1))
        print(f"  {len(train_raw)} series")

        print("Pooling intermittent windows …")
        ind_ctx, ind_lab, mag_ctx, mag_log, ind_meta, mag_meta = pool_intermittent_windows(
            train_raw,
            context_length=CONTEXT_LENGTH,
            horizon_length=horizon_length,
            stride=horizon_length,
            max_windows_per_series=MAX_WINDOWS_PER_SERIES,
            min_observed_fraction=MIN_OBSERVED_FRACTION,
            rng=rng,
        )
        print(f"  Indicator pool: {ind_ctx.shape[0]} windows")
        print(f"  Magnitude pool: {mag_ctx.shape[0]} windows  (all-positive targets)")
        print(f"  Mean demand rate in indicator pool: {ind_lab.mean():.3f}")
    """).strip()


def int_fit_indicator_cell() -> str:
    return dedent("""
        print("\\nFitting Layer 1 — indicator (Bernoulli MLE) …")
        ell_field_ind = fit_bernoulli_lengthscale_field(
            ind_ctx, ind_lab,
            block_size_l1=BLOCK_SIZE_L1_IND,
            amplitude=AMPLITUDE_IND,
            log_ell_bounds=(LOG_ELL_LO, LOG_ELL_HI),
            n_iter_laplace=N_ITER_LAPLACE,
            meta_amplitude=META_AMPLITUDE_IND,
            meta_bandwidth=META_BANDWIDTH_IND,
            rng=rng,
        )
        print(f"  anchors: {ell_field_ind.n_anchors}, bandwidth: {ell_field_ind.meta_bandwidth:.4f}")

        print("Fitting Layer 2 — indicator (BernoulliKrigingClassifier) …")
        clf = BernoulliKrigingClassifier(
            ell_field_ind,
            block_size=BLOCK_SIZE_L2_IND,
            max_blocks=MAX_BLOCKS_IND,
            amplitude=AMPLITUDE_IND,
            n_iter_laplace=N_ITER_LAPLACE,
            rng=rng,
        ).fit(ind_ctx, ind_lab)
        print(f"  blocks: {len(clf._fitted_blocks)}")
    """).strip()


def int_fit_magnitude_cell() -> str:
    return dedent("""
        if mag_ctx.shape[0] < 5:
            print("WARNING: magnitude pool has fewer than 5 windows — Model 2 will be unreliable.")
            print("Consider reducing CONTEXT_LENGTH or increasing MAX_SERIES.")
            mag_forecaster = None
        else:
            print("\\nFitting Layer 1 — magnitude (Gaussian MLE on log y) …")
            ell_field_mag = fit_gibbs_lengthscale_field(
                mag_ctx, mag_log,
                block_size_l1=BLOCK_SIZE_L1_MAG,
                sigma_n=SIGMA_N_MAG,
                amplitude=AMPLITUDE_MAG,
                meta_amplitude=META_AMPLITUDE_MAG,
                meta_bandwidth=META_BANDWIDTH_MAG,
                rng=rng,
            )
            print(f"  anchors: {ell_field_mag.n_anchors}")

            print("Fitting Layer 2 — magnitude (NestedKrigingForecaster on log y) …")
            mag_forecaster = NestedKrigingForecaster(
                ell_field_mag,
                block_size=BLOCK_SIZE_L2_MAG,
                max_blocks=MAX_BLOCKS_MAG,
                sigma_n=SIGMA_N_L2_MAG,
                amplitude=AMPLITUDE_MAG,
                rng=rng,
            ).fit(mag_ctx, mag_log)
            print(f"  blocks: {len(mag_forecaster._fitted_blocks)}")

        two_model = (
            TwoModelIntermittentForecaster(indicator=clf, magnitude=mag_forecaster)
            if mag_forecaster is not None else None
        )
        print("\\nTwo-model forecaster ready:", two_model is not None)
    """).strip()


def int_short_term_cell() -> str:
    return dedent("""
        if two_model is None:
            print("Skipping evaluation: magnitude model unavailable.")
        else:
            horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)
            n_total = len(dataset.hf_dataset)
            errors, skipped, plotted = [], [], False
            from src.nested_kriging import _linear_impute

            for idx in _eval_indices(n_total):
                tr = _train(idx);  va = _val(idx)
                extra = va[len(tr):]
                if len(tr) < CONTEXT_LENGTH or len(extra) < horizon_length:
                    skipped.append((idx, "too short")); continue
                pos = tr[np.isfinite(tr) & (tr > 0)]
                if len(pos) == 0: skipped.append((idx, "no positive train values")); continue
                scale = max(float(np.std(pos)), 1e-6)

                query_norm = _linear_impute(tr[-CONTEXT_LENGTH:]) / scale
                # E[y] in normalised units; multiply by scale to get original units
                E_y_norm = two_model.predict(query_norm)          # (H,)
                E_y = E_y_norm * scale

                true_h = extra[:horizon_length]
                obs_h = np.isfinite(true_h) & (true_h >= 0)
                if not obs_h.any(): skipped.append((idx, "no observed val-extra")); continue
                errors.append(relative_l2(true_h[obs_h], E_y[:horizon_length][obs_h]))

                if (not EVALUATE_ALL_SERIES and idx == SERIES_INDEX) or (EVALUATE_ALL_SERIES and not plotted):
                    _plot(true_h, E_y[:horizon_length],
                          title=f"{DATASET_NAME} short-term E[y] (series {idx})")
                    plotted = True

            if errors:
                a = np.asarray(errors, float)
                print(f"Evaluated: {len(errors)}, skipped: {len(skipped)}")
                print(f"Mean relL2: {np.mean(a):.6f}  Median: {np.median(a):.6f}")
            else:
                print("No short-term forecasts evaluated."); [print(" ", s) for s in skipped]
    """).strip()


def int_refit_val_cell() -> str:
    return dedent("""
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)
        print("Collecting validation series …")
        val_raw = []
        for i, entry in enumerate(dataset.validation_dataset):
            if i >= MAX_SERIES: break
            val_raw.append(np.asarray(entry["target"], dtype=float).reshape(-1))

        print("Pooling validation intermittent windows …")
        ind_ctx_v, ind_lab_v, mag_ctx_v, mag_log_v, _, _ = pool_intermittent_windows(
            val_raw,
            context_length=CONTEXT_LENGTH, horizon_length=horizon_length,
            stride=horizon_length, max_windows_per_series=MAX_WINDOWS_PER_SERIES,
            min_observed_fraction=MIN_OBSERVED_FRACTION, rng=rng,
        )
        print(f"  Indicator: {ind_ctx_v.shape[0]} windows  Magnitude: {mag_ctx_v.shape[0]} windows")

        ell_ind_v = fit_bernoulli_lengthscale_field(
            ind_ctx_v, ind_lab_v, block_size_l1=BLOCK_SIZE_L1_IND,
            amplitude=AMPLITUDE_IND, log_ell_bounds=(LOG_ELL_LO, LOG_ELL_HI),
            n_iter_laplace=N_ITER_LAPLACE, rng=rng,
        )
        clf_v = BernoulliKrigingClassifier(
            ell_ind_v, block_size=BLOCK_SIZE_L2_IND, max_blocks=MAX_BLOCKS_IND,
            amplitude=AMPLITUDE_IND, n_iter_laplace=N_ITER_LAPLACE, rng=rng,
        ).fit(ind_ctx_v, ind_lab_v)

        if mag_ctx_v.shape[0] >= 5:
            ell_mag_v = fit_gibbs_lengthscale_field(
                mag_ctx_v, mag_log_v, block_size_l1=BLOCK_SIZE_L1_MAG,
                sigma_n=SIGMA_N_MAG, amplitude=AMPLITUDE_MAG, rng=rng,
            )
            mag_v = NestedKrigingForecaster(
                ell_mag_v, block_size=BLOCK_SIZE_L2_MAG, max_blocks=MAX_BLOCKS_MAG,
                sigma_n=SIGMA_N_L2_MAG, amplitude=AMPLITUDE_MAG, rng=rng,
            ).fit(mag_ctx_v, mag_log_v)
            two_model_val = TwoModelIntermittentForecaster(indicator=clf_v, magnitude=mag_v)
        else:
            print("WARNING: validation magnitude pool < 5 windows — using train model.")
            two_model_val = two_model
        print("Validation model ready.")
    """).strip()


def int_test_eval_cell() -> str:
    return dedent("""
        if two_model_val is None:
            print("Skipping: two-model forecaster unavailable.")
        else:
            horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)
            n_total = len(dataset.hf_dataset)
            errors, skipped, plotted = [], [], False
            from src.nested_kriging import _linear_impute

            for idx in _eval_indices(n_total):
                va = _val(idx)
                pos = va[np.isfinite(va) & (va > 0)]
                if len(pos) == 0: skipped.append((idx, "no positive val values")); continue
                scale = max(float(np.std(pos)), 1e-6)
                if len(va) < CONTEXT_LENGTH: skipped.append((idx, "val too short")); continue

                try: ctx_e, lbl_e = _test(idx, 0)
                except IndexError as e: skipped.append((idx, str(e))); continue
                if len(ctx_e) < CONTEXT_LENGTH or len(lbl_e) < horizon_length:
                    skipped.append((idx, "test ctx/lbl too short")); continue

                query_norm = _linear_impute(ctx_e[-CONTEXT_LENGTH:]) / scale
                E_y_norm = two_model_val.predict(query_norm)
                E_y = E_y_norm * scale

                true_h = lbl_e[:horizon_length]
                obs_h = np.isfinite(true_h) & (true_h >= 0)
                if not obs_h.any(): skipped.append((idx, "all NaN test label")); continue
                errors.append(relative_l2(true_h[obs_h], E_y[:horizon_length][obs_h]))

                if (not EVALUATE_ALL_SERIES and idx == SERIES_INDEX) or (EVALUATE_ALL_SERIES and not plotted):
                    _plot(true_h, E_y[:horizon_length],
                          title=f"{DATASET_NAME} test window 0 E[y] (series {idx})")
                    plotted = True

            if errors:
                a = np.asarray(errors, float)
                print(f"Evaluated: {len(errors)}, skipped: {len(skipped)}")
                print(f"Mean relL2: {np.mean(a):.6f}  Median: {np.median(a):.6f}")
            else:
                print("No test forecasts evaluated."); [print(" ", s) for s in skipped]
    """).strip()


def int_demand_rate_cell() -> str:
    return dedent("""
        # ── Inspect demand rates from the indicator model ─────────────────
        print("Sample P(demand > 0 | x*) for 5 validation-end contexts of series", SERIES_INDEX)
        from src.nested_kriging import _linear_impute
        horizon_length = dataset.prediction_length if HORIZON_LENGTH is None else int(HORIZON_LENGTH)
        va = _val(SERIES_INDEX)
        pos = va[np.isfinite(va) & (va > 0)]
        scale = max(float(np.std(pos)), 1e-6) if len(pos) > 0 else 1.0
        n_inspect = min(5, len(va) // CONTEXT_LENGTH)
        for i in range(n_inspect):
            end = (i + 1) * CONTEXT_LENGTH
            if end + horizon_length > len(va): break
            q_norm = _linear_impute(va[end - CONTEXT_LENGTH: end]) / scale
            p_z = clf_v.predict_proba(q_norm)    # (H,)
            print(f"  context ending at t={end}: P(z=1) per step = {p_z.round(3)}")
    """).strip()


# ---------------------------------------------------------------------------
# Notebook builders
# ---------------------------------------------------------------------------

def make_gap_notebook(dataset_name: str, row: dict, krr_preset: dict) -> dict:
    cells = [
        _md(f"# Nested Kriging (Gap-Missing): `{dataset_name}`\n\n"
            "NaN gaps (sensor outages / calendar holes) are linearly imputed before "
            "windowing. The model is otherwise identical to the no-missing procedure: "
            "Layer 1 learns ℓ̃(x) via Gaussian MLE per block; Layer 2 runs nested "
            "kriging with the Gibbs kernel."),
        _code(setup_cell()),
        _code(gap_import_cell()),
        _code(gap_parameter_cell(dataset_name, row, krr_preset)),
        _code(metadata_cell()),
        _code(shared_helpers_cell()),
        _md("## Layer 1 — Fit Lengthscale Field (training split)"),
        _code(gap_fit_l1_cell()),
        _md("## Layer 2 — Fit Nested Kriging Regression (training split)"),
        _code(gap_fit_l2_cell()),
        _md("## Short-Term Evaluation: train → validation \\ train"),
        _code(gap_short_term_cell()),
        _md("## Re-Fit on Validation Split"),
        _code(gap_refit_val_cell()),
        _md("## Long-Term Evaluation: test window 0"),
        _code(gap_test_eval_cell()),
    ]
    return _notebook(cells)


def make_intermittent_notebook(dataset_name: str, row: dict, krr_preset: dict) -> dict:
    cells = [
        _md(f"# Nested Kriging (Intermittent Demand): `{dataset_name}`\n\n"
            "Two-model approach for intermittent demand:\n"
            "- **Model 1** (BernoulliKrigingClassifier): P(z_h=1|x*) where z_h = 1[y_h>0]. "
            "Layer 1 uses Laplace-approximated Bernoulli MLE per block.\n"
            "- **Model 2** (NestedKrigingForecaster on log y): E[log y_h | z_h=1, x*]. "
            "Fitted only on windows where all H target steps are strictly positive.\n\n"
            "Prediction: E[y_h|x*] = P(z_h=1|x*) · exp(μ_h(x*) + σ²(x*)/2)"),
        _code(setup_cell()),
        _code(int_import_cell()),
        _code(int_parameter_cell(dataset_name, row, krr_preset)),
        _code(metadata_cell()),
        _code(shared_helpers_cell()),
        _md("## Pool Intermittent Windows (indicator + magnitude)"),
        _code(int_pool_cell()),
        _md("## Model 1 — Indicator: fit Bernoulli lengthscale field + classifier"),
        _code(int_fit_indicator_cell()),
        _md("## Model 2 — Magnitude: fit Gaussian lengthscale field + regression on log y"),
        _code(int_fit_magnitude_cell()),
        _md("## Short-Term Evaluation: train → validation \\ train"),
        _code(int_short_term_cell()),
        _md("## Re-Fit Both Models on Validation Split"),
        _code(int_refit_val_cell()),
        _md("## Long-Term Evaluation: test window 0"),
        _code(int_test_eval_cell()),
        _md("## Inspect Demand Rates P(z=1|x*) from the Indicator Model"),
        _code(int_demand_rate_cell()),
    ]
    return _notebook(cells)


def _notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def make_readme(intermittent_names: list[str], gap_names: list[str]) -> str:
    lines = [
        "# Univariate Missing-Data Nested Kriging Playgrounds",
        "",
        "Generated by `experiments/generate_univariate_missing_nested_kriging_playgrounds.py`.",
        "",
        "## Intermittent demand (two-model: Bernoulli + log-Gaussian)",
        "",
    ]
    for ds in intermittent_names:
        lines.append(f"- [{ds}](./{notebook_filename(ds)})")
    lines += ["", "## Gap-missing (linear imputation + standard nested kriging)", ""]
    for ds in gap_names:
        lines.append(f"- [{ds}](./{notebook_filename(ds)})")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows: list[dict] = json.loads(AUDIT_JSON.read_text())
    krr_presets: dict = json.loads(KRR_PRESETS_JSON.read_text())

    rows = [r for r in all_rows if r["any_missing_raw"] and int(r["target_dim"]) == 1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    intermittent_names, gap_names = [], []

    for row in rows:
        ds = row["dataset_name"]
        preset = krr_presets.get(ds, {
            "context_length": 2 * int(row["prediction_length"]),
            "max_series": 200,
            "max_windows_per_series": 20,
        })
        path = OUTPUT_DIR / notebook_filename(ds)

        if ds in INTERMITTENT:
            nb = make_intermittent_notebook(ds, row, preset)
            intermittent_names.append(ds)
        else:
            nb = make_gap_notebook(ds, row, preset)
            gap_names.append(ds)

        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        print(f"  wrote {path.name}")

    README_PATH.write_text(make_readme(intermittent_names, gap_names))
    print(f"\nGenerated {len(rows)} notebooks in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
