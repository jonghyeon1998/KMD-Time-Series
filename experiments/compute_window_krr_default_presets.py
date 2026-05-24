from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gift_eval.data import Dataset
from src.window_krr import (
    KernelConfig,
    WindowKernelRidgeForecaster,
    build_gap_deltas_from_mask,
    build_regular_deltas,
    make_windowed_forecasting_dataset,
    relative_l2_error,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_JSON = ROOT / "experiments" / "gift_eval_univariate_split_audit.json"
OUTPUT_JSON = ROOT / "experiments" / "window_krr_default_presets.json"


def _freq_base_context(freq: str) -> int:
    freq = freq.upper()
    if freq == "5T":
        return 288
    if freq == "10T":
        return 144
    if freq == "15T":
        return 96
    if freq == "H":
        return 168
    if freq == "D":
        return 180
    if freq.startswith("W"):
        return 52
    if freq == "M":
        return 48
    if freq.startswith("Q"):
        return 24
    if freq.startswith("A"):
        return 12
    return 4


def _series_from_entry(entry) -> np.ndarray:
    return np.asarray(entry["target"], dtype=float).reshape(-1)


def _train_entries(dataset: Dataset) -> list:
    return list(dataset.training_dataset)


def _validation_entries(dataset: Dataset) -> list:
    return list(dataset.validation_dataset)


def _validation_minus_train(train_entry, validation_entry) -> np.ndarray:
    train = _series_from_entry(train_entry)
    val = _series_from_entry(validation_entry)
    return val[len(train) :]


def _build_deltas(values: np.ndarray, representation: str) -> np.ndarray:
    observed_mask = np.isfinite(values)
    if representation == "masked":
        return build_gap_deltas_from_mask(observed_mask, step=1.0)
    return build_regular_deltas(len(values), step=1.0)


def _choose_context_length(train_lengths: np.ndarray, freq: str, prediction_length: int) -> int:
    base = _freq_base_context(freq)
    q25 = int(np.percentile(train_lengths, 25))
    preferred_cap = q25 - 2 * prediction_length
    fallback_cap = q25 - prediction_length
    if preferred_cap >= max(4, prediction_length // 2):
        cap = preferred_cap
    else:
        cap = fallback_cap
    floor = max(4, min(base, max(1, prediction_length // 2)))
    context = max(floor, min(base, cap))
    return int(context)


def _sample_bank(
    entries: list,
    *,
    context_length: int,
    horizon_length: int,
    representation: str,
    max_series: int = 50,
    max_windows_per_series: int = 10,
) -> dict:
    contexts = []
    targets = []
    context_masks = []
    target_masks = []
    context_deltas = []
    used_series = 0

    for entry in entries:
        if used_series >= max_series:
            break
        values = _series_from_entry(entry)
        observed_mask = np.isfinite(values)
        try:
            ds = make_windowed_forecasting_dataset(
                values,
                context_length=context_length,
                horizon_length=horizon_length,
                observed_mask=observed_mask,
                deltas=_build_deltas(values, representation),
                stride=horizon_length,
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

    if not contexts:
        raise ValueError("No windows could be built for the sampled bank.")

    return {
        "contexts": np.vstack(contexts),
        "targets": np.vstack(targets),
        "context_masks": np.vstack(context_masks),
        "target_masks": np.vstack(target_masks),
        "context_deltas": np.vstack(context_deltas),
        "used_series": used_series,
        "n_windows": int(sum(x.shape[0] for x in contexts)),
    }


def _pairwise_distance_scale(
    contexts: np.ndarray,
    masks: np.ndarray,
    *,
    representation: str,
    max_samples: int = 150,
) -> float:
    rng = np.random.default_rng(0)
    n = contexts.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X = contexts[idx]
        M = masks[idx]
    else:
        X = contexts
        M = masks

    dists = []
    for i in range(X.shape[0]):
        xi = np.where(M[i], X[i], 0.0)
        for j in range(i + 1, X.shape[0]):
            xj = np.where(M[j], X[j], 0.0)
            if representation == "masked":
                overlap = M[i] & M[j]
                if not overlap.any():
                    continue
                d = float(np.linalg.norm(xi[overlap] - xj[overlap]))
            else:
                d = float(np.linalg.norm(xi - xj))
            if np.isfinite(d) and d > 0:
                dists.append(d)
    if not dists:
        return 10.0
    q75 = float(np.quantile(np.asarray(dists, dtype=float), 0.75))
    return max(5.0, q75)


def _evaluate_short_term_mean(
    dataset: Dataset,
    train_entries: list,
    val_entries: list,
    model: WindowKernelRidgeForecaster,
    *,
    context_length: int,
    horizon_length: int,
    representation: str,
    eval_series_limit: int = 20,
) -> float:
    errors = []
    for train_entry, val_entry in list(zip(train_entries, val_entries))[:eval_series_limit]:
        train = _series_from_entry(train_entry)
        val_extra = _validation_minus_train(train_entry, val_entry)
        if len(train) < context_length or len(val_extra) < horizon_length:
            continue
        query = train[-context_length:]
        query_mask = np.isfinite(query)
        pred = model.forecast(
            query,
            strategy="direct",
            total_horizon=horizon_length,
            query_masks=query_mask,
            query_deltas=_build_deltas(query, representation),
        ).reshape(-1)
        truth = val_extra[:horizon_length]
        truth_mask = np.isfinite(truth)
        if not truth_mask.any():
            continue
        errors.append(relative_l2_error(truth, pred, mask=truth_mask))
    if not errors:
        return float("inf")
    return float(np.mean(errors))


def _select_defaults_for_dataset(record: dict) -> dict:
    dataset_name = record["dataset_name"]
    dataset = Dataset(name=dataset_name, term="short", to_univariate=False, storage_env_var="GIFT_EVAL")
    train_entries = _train_entries(dataset)
    val_entries = _validation_entries(dataset)
    prediction_length = int(dataset.prediction_length)
    freq = str(dataset.freq)
    representation = "masked" if bool(record["any_missing_raw"]) else "plain"

    train_lengths = np.asarray([len(_series_from_entry(entry)) for entry in train_entries], dtype=int)
    context_length = _choose_context_length(train_lengths, freq, prediction_length)
    first_train_len = len(_series_from_entry(train_entries[0]))
    context_length = min(context_length, max(1, first_train_len - prediction_length))
    bank = _sample_bank(
        train_entries,
        context_length=context_length,
        horizon_length=prediction_length,
        representation=representation,
    )
    base_scale = _pairwise_distance_scale(
        bank["contexts"],
        bank["context_masks"],
        representation=representation,
    )

    scale_candidates = sorted({round(base_scale * factor, 6) for factor in (2.0, 4.0)})
    reg_candidates = [1e-2, 1e-1]
    if representation == "masked":
        missing_pct = float(record["raw_missing_pct"])
        if missing_pct >= 10.0:
            decay_candidates = [1.0, 0.97, 0.93]
        elif missing_pct >= 3.0:
            decay_candidates = [1.0, 0.98, 0.95]
        else:
            decay_candidates = [1.0, 0.99, 0.97]
    else:
        decay_candidates = [1.0]

    best = None
    for length_scale in scale_candidates:
        for regularization in reg_candidates:
            for mask_decay_gamma in decay_candidates:
                config = KernelConfig(
                    family="rbf",
                    amplitude=1.0,
                    length_scale=float(length_scale),
                    mask_decay_gamma=float(mask_decay_gamma),
                )
                model = WindowKernelRidgeForecaster(
                    kernel_config=config,
                    representation=representation,
                    regularization=float(regularization),
                )
                model.fit(
                    bank["contexts"],
                    bank["targets"],
                    context_masks=bank["context_masks"],
                    target_masks=bank["target_masks"],
                    context_deltas=bank["context_deltas"],
                )
                score = _evaluate_short_term_mean(
                    dataset,
                    train_entries,
                    val_entries,
                    model,
                    context_length=context_length,
                    horizon_length=prediction_length,
                    representation=representation,
                )
                candidate = {
                    "score": score,
                    "length_scale": float(length_scale),
                    "regularization": float(regularization),
                    "mask_decay_gamma": float(mask_decay_gamma),
                }
                if best is None or candidate["score"] < best["score"]:
                    best = candidate

    assert best is not None
    return {
        "representation": representation,
        "context_length": int(context_length),
        "length_scale": float(best["length_scale"]),
        "regularization": float(best["regularization"]),
        "mask_decay_gamma": float(best["mask_decay_gamma"]),
        "recursive_chunk_length": int(prediction_length),
        "max_series": 200,
        "max_windows_per_series": 20,
        "selection_score": float(best["score"]),
    }


def main() -> None:
    audit_records = json.loads(AUDIT_JSON.read_text())
    presets = {}
    for record in audit_records:
        dataset_name = record["dataset_name"]
        print(f"Selecting defaults for {dataset_name}...")
        presets[dataset_name] = _select_defaults_for_dataset(record)
        print(dataset_name, presets[dataset_name])
    OUTPUT_JSON.write_text(json.dumps(presets, indent=2, sort_keys=True))
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
