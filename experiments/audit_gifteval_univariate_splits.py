from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from gift_eval.data import Dataset


ROOT = Path("/Users/jonghyeonlee/KMD_TS/GiftEval_data")
CSV_PATH = Path("/Users/jonghyeonlee/KMD_TS/experiments/gift_eval_univariate_split_audit.csv")
JSON_PATH = Path("/Users/jonghyeonlee/KMD_TS/experiments/gift_eval_univariate_split_audit.json")


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
    train_is_validation_prefix_all: bool
    validation_is_first_test_context_all: bool
    test_contexts_match_raw_prefix_all: bool
    test_labels_match_raw_suffix_all: bool
    any_missing_raw: bool
    raw_missing_pct: float
    train_missing_pct: float
    validation_missing_pct: float
    test_context_missing_pct: float
    test_label_missing_pct: float


def dataset_names(root: Path) -> list[str]:
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


def _equal(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def _missing_pct(missing_count: int, total_count: int) -> float:
    if total_count == 0:
        return 0.0
    return 100.0 * float(missing_count) / float(total_count)


def audit_dataset(name: str) -> AuditRow | None:
    ds = Dataset(name=name, term="short", to_univariate=False, storage_env_var="GIFT_EVAL")
    if ds.target_dim != 1:
        return None

    train_it = iter(ds.training_dataset)
    val_it = iter(ds.validation_dataset)
    test_input_it = iter(ds.test_data.input)
    test_label_it = iter(ds.test_data.label)

    n_raw_series = len(ds.hf_dataset)
    raw_lengths: list[int] = []
    train_lengths: list[int] = []
    val_lengths: list[int] = []

    raw_missing = train_missing = val_missing = 0
    test_context_missing = test_label_missing = 0
    raw_total = train_total = val_total = 0
    test_context_total = test_label_total = 0

    train_is_val_prefix_all = True
    val_is_first_test_context_all = True
    test_contexts_match_raw_prefix_all = True
    test_labels_match_raw_suffix_all = True

    n_train_series = 0
    n_val_series = 0

    for series_idx in range(n_raw_series):
        raw = np.asarray(ds.hf_dataset[series_idx]["target"], dtype=float)
        train_entry = next(train_it)
        val_entry = next(val_it)
        n_train_series += 1
        n_val_series += 1

        train = np.asarray(train_entry["target"], dtype=float)
        val = np.asarray(val_entry["target"], dtype=float)

        raw_lengths.append(int(raw.shape[0]))
        train_lengths.append(int(train.shape[0]))
        val_lengths.append(int(val.shape[0]))

        raw_missing += int(np.isnan(raw).sum())
        train_missing += int(np.isnan(train).sum())
        val_missing += int(np.isnan(val).sum())
        raw_total += int(raw.size)
        train_total += int(train.size)
        val_total += int(val.size)

        train_is_val_prefix_all &= _equal(val[: len(train)], train)

        for window_idx in range(ds.windows):
            test_input = np.asarray(next(test_input_it)["target"], dtype=float)
            test_label = np.asarray(next(test_label_it)["target"], dtype=float)

            expected_context_end = len(val) + window_idx * ds.prediction_length
            expected_label_start = expected_context_end
            expected_label_end = expected_label_start + ds.prediction_length

            if window_idx == 0:
                val_is_first_test_context_all &= _equal(test_input, val)

            test_contexts_match_raw_prefix_all &= _equal(test_input, raw[:expected_context_end])
            test_labels_match_raw_suffix_all &= _equal(
                test_label,
                raw[expected_label_start:expected_label_end],
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
        train_is_validation_prefix_all=train_is_val_prefix_all,
        validation_is_first_test_context_all=val_is_first_test_context_all,
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
    for name in dataset_names(ROOT):
        row = audit_dataset(name)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda r: r.dataset_name)

    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    JSON_PATH.write_text(json.dumps([asdict(row) for row in rows], indent=2))
    print(f"Saved {len(rows)} univariate dataset audits to {CSV_PATH}")
    print(f"Saved JSON summary to {JSON_PATH}")


if __name__ == "__main__":
    main()
