"""
audit_gifteval_multivariate_splits.py
======================================
Mirrors audit_gifteval_univariate_splits.py but targets multivariate datasets
(target_dim > 1).

Key differences from the univariate version
--------------------------------------------
* ``target_dim > 1`` filter (instead of == 1).
* Target arrays have shape ``(p, T)`` rather than ``(T,)``, so series length
  is ``raw.shape[-1]`` (the last axis is always time).
* Missing-data counts span all p × T elements.
* Prefix / suffix equality checks compare along the time axis:
  ``val[..., :T_train] == train``.

Run from the KMD_TS repository root:
    python experiments/audit_gifteval_multivariate_splits.py

Outputs
-------
experiments/gift_eval_multivariate_split_audit.csv
experiments/gift_eval_multivariate_split_audit.json
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from gift_eval.data import Dataset


ROOT = Path("/Users/jonghyeonlee/KMD_TS/GiftEval_data")
CSV_PATH = Path("/Users/jonghyeonlee/KMD_TS/experiments/gift_eval_multivariate_split_audit.csv")
JSON_PATH = Path("/Users/jonghyeonlee/KMD_TS/experiments/gift_eval_multivariate_split_audit.json")


@dataclass
class AuditRow:
    dataset_name: str
    freq: str
    prediction_length: int
    windows: int
    n_raw_series: int
    n_train_series: int
    n_val_series: int
    target_dim: int
    raw_min_len: int
    raw_max_len: int
    train_min_len: int
    train_max_len: int
    val_min_len: int
    val_max_len: int
    test_context_min_len: int
    test_context_max_len: int
    n_test_windows_per_series: int
    train_is_validation_prefix_all: bool
    validation_is_first_test_context_all: bool
    validation_is_prefix_of_all_test_contexts_all: bool
    test_contexts_match_raw_prefix_all: bool
    test_labels_match_raw_suffix_all: bool
    any_missing_raw: bool
    raw_missing_pct: float
    train_missing_pct: float
    validation_missing_pct: float
    test_context_missing_pct: float
    test_label_missing_pct: float


def dataset_names(root: Path) -> list[str]:
    """Enumerate all dataset/freq combinations present on disk."""
    names: list[str] = []
    for dataset_dir in sorted(root.iterdir()):
        if dataset_dir.name.startswith(".") or not dataset_dir.is_dir():
            continue
        freq_dirs = [d for d in sorted(dataset_dir.iterdir()) if d.is_dir()]
        if freq_dirs:
            for freq_dir in freq_dirs:
                names.append(f"{dataset_dir.name}/{freq_dir.name}")
        else:
            names.append(dataset_dir.name)
    return names


def _series_len(arr: np.ndarray) -> int:
    """Return the time-axis length for univariate (T,) or multivariate (p, T) arrays."""
    return int(arr.shape[-1])


def _equal(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def _missing_pct(missing_count: int, total_count: int) -> float:
    if total_count == 0:
        return 0.0
    return 100.0 * float(missing_count) / float(total_count)


def audit_dataset(name: str) -> AuditRow | None:
    ds = Dataset(name=name, term="short", to_univariate=False, storage_env_var="GIFT_EVAL")

    # Skip univariate datasets
    if ds.target_dim == 1:
        return None

    train_it = iter(ds.training_dataset)
    val_it = iter(ds.validation_dataset)
    test_input_it = iter(ds.test_data.input)
    test_label_it = iter(ds.test_data.label)

    n_raw_series = len(ds.hf_dataset)
    raw_lengths: list[int] = []
    train_lengths: list[int] = []
    val_lengths: list[int] = []
    test_context_lengths: list[int] = []

    raw_missing = train_missing = val_missing = 0
    test_context_missing = test_label_missing = 0
    raw_total = train_total = val_total = 0
    test_context_total = test_label_total = 0

    train_is_val_prefix_all = True
    val_is_first_test_context_all = True
    val_is_prefix_of_all_test_contexts_all = True
    test_contexts_match_raw_prefix_all = True
    test_labels_match_raw_suffix_all = True

    n_train_series = 0
    n_val_series = 0

    for series_idx in range(n_raw_series):
        # Raw series: (p, T) for multivariate
        raw = np.asarray(ds.hf_dataset[series_idx]["target"], dtype=float)
        train_entry = next(train_it)
        val_entry = next(val_it)
        n_train_series += 1
        n_val_series += 1

        train = np.asarray(train_entry["target"], dtype=float)
        val = np.asarray(val_entry["target"], dtype=float)

        T_train = _series_len(train)
        T_val = _series_len(val)

        raw_lengths.append(_series_len(raw))
        train_lengths.append(T_train)
        val_lengths.append(T_val)

        raw_missing += int(np.isnan(raw).sum())
        train_missing += int(np.isnan(train).sum())
        val_missing += int(np.isnan(val).sum())
        raw_total += int(raw.size)
        train_total += int(train.size)
        val_total += int(val.size)

        # Check that train is a prefix of val along the time axis
        train_is_val_prefix_all &= _equal(val[..., :T_train], train)

        for window_idx in range(ds.windows):
            test_input = np.asarray(next(test_input_it)["target"], dtype=float)
            test_label = np.asarray(next(test_label_it)["target"], dtype=float)

            T_ctx = _series_len(test_input)
            test_context_lengths.append(T_ctx)

            expected_context_end = T_val + window_idx * ds.prediction_length
            expected_label_start = expected_context_end
            expected_label_end = expected_label_start + ds.prediction_length

            if window_idx == 0:
                val_is_first_test_context_all &= _equal(test_input, val)

            val_is_prefix_of_all_test_contexts_all &= _equal(
                test_input[..., :T_val], val
            )
            test_contexts_match_raw_prefix_all &= _equal(
                test_input, raw[..., :expected_context_end]
            )
            test_labels_match_raw_suffix_all &= _equal(
                test_label,
                raw[..., expected_label_start:expected_label_end],
            )

            test_context_missing += int(np.isnan(test_input).sum())
            test_label_missing += int(np.isnan(test_label).sum())
            test_context_total += int(test_input.size)
            test_label_total += int(test_label.size)

    return AuditRow(
        dataset_name=name,
        freq=ds.freq,
        prediction_length=int(ds.prediction_length),
        windows=int(ds.windows),
        n_raw_series=n_raw_series,
        n_train_series=n_train_series,
        n_val_series=n_val_series,
        target_dim=int(ds.target_dim),
        raw_min_len=min(raw_lengths),
        raw_max_len=max(raw_lengths),
        train_min_len=min(train_lengths),
        train_max_len=max(train_lengths),
        val_min_len=min(val_lengths),
        val_max_len=max(val_lengths),
        test_context_min_len=min(test_context_lengths),
        test_context_max_len=max(test_context_lengths),
        n_test_windows_per_series=int(ds.windows),
        train_is_validation_prefix_all=train_is_val_prefix_all,
        validation_is_first_test_context_all=val_is_first_test_context_all,
        validation_is_prefix_of_all_test_contexts_all=val_is_prefix_of_all_test_contexts_all,
        test_contexts_match_raw_prefix_all=test_contexts_match_raw_prefix_all,
        test_labels_match_raw_suffix_all=test_labels_match_raw_suffix_all,
        any_missing_raw=raw_missing > 0,
        raw_missing_pct=_missing_pct(raw_missing, raw_total),
        train_missing_pct=_missing_pct(train_missing, train_total),
        validation_missing_pct=_missing_pct(val_missing, val_total),
        test_context_missing_pct=_missing_pct(test_context_missing, test_context_total),
        test_label_missing_pct=_missing_pct(test_label_missing, test_label_total),
    )


def main() -> None:
    rows: list[AuditRow] = []
    all_names = dataset_names(ROOT)
    print(f"Found {len(all_names)} dataset/freq combinations on disk.")

    for name in all_names:
        print(f"  auditing {name} …", end=" ", flush=True)
        try:
            row = audit_dataset(name)
            if row is None:
                print("skipped (univariate)")
            else:
                rows.append(row)
                print(f"OK  target_dim={row.target_dim}  n_series={row.n_raw_series}"
                      f"  T_train=[{row.train_min_len},{row.train_max_len}]"
                      f"  missing={row.raw_missing_pct:.1f}%")
        except Exception as exc:
            print(f"ERROR: {exc}")

    rows.sort(key=lambda r: r.dataset_name)

    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    JSON_PATH.write_text(json.dumps([asdict(row) for row in rows], indent=2))
    print(f"\nSaved {len(rows)} multivariate dataset audits to:")
    print(f"  {CSV_PATH}")
    print(f"  {JSON_PATH}")


if __name__ == "__main__":
    main()
