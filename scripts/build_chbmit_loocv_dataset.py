#!/usr/bin/env python3
"""Build patient-specific seizure-level LOOCV CHB-MIT datasets.

For each subject and each held-out seizure:
  * train positives: preictal windows from the same subject's other seizures
  * test positives: all preictal windows for the held-out seizure
  * train negatives: interictal windows not reserved for testing, undersampled
  * test negatives: reserved interictal windows, kept unbalanced by default

The generated window metadata is written next to each NPZ so event-level
metrics and false-alarm-rate estimates can be computed later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from build_chbmit_baseline_dataset import (
    COMMON_CHANNELS,
    close_edf,
    load_edf_window,
    open_edf,
    read_csv_rows,
    zscore,
)


def event_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["subject"], row["file_name"], row["seizure_index_in_file"]


def matched_event_key(row: dict[str, str]) -> tuple[str, str, str] | None:
    matched = row["matched_seizure_index_in_file"]
    if not matched:
        return None
    seizure_file = row.get("matched_seizure_file_name", "").strip() or row["file_name"]
    return row["subject"], seizure_file, matched


def stable_seed(base_seed: int, key: tuple[str, ...]) -> int:
    digest = hashlib.sha1("|".join(key).encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def sample_rows(rows: list[dict[str, str]], n: int | None, seed: int) -> list[dict[str, str]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if n is not None:
        rows = rows[:n]
    return rows


def row_id(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["file_name"], row["window_start_sec"], row["window_end_sec"], row["label"]


def contiguous_interictal_blocks(rows: list[dict[str, str]], max_gap_sec: int) -> list[list[dict[str, str]]]:
    rows = sorted(rows, key=lambda r: (r["file_name"], int(r["window_start_sec"]), int(r["window_end_sec"])))
    blocks: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    previous: dict[str, str] | None = None
    for row in rows:
        if (
            previous is None
            or row["file_name"] != previous["file_name"]
            or int(row["window_start_sec"]) > int(previous["window_end_sec"]) + max_gap_sec
        ):
            if current:
                blocks.append(current)
            current = [row]
        else:
            current.append(row)
        previous = row
    if current:
        blocks.append(current)
    return blocks


def split_interictal_rows(
    rows: list[dict[str, str]],
    heldout_key: tuple[str, str, str],
    seed: int,
    test_frac: float,
    max_test_interictal: int | None,
    test_interictal_mode: str = "random",
    contiguous_gap_sec: int = 0,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = sorted(rows, key=lambda r: (r["file_name"], int(r["window_start_sec"]), int(r["window_end_sec"])))
    test_n = int(round(len(rows) * test_frac))
    test_n = max(1, test_n) if rows else 0
    if max_test_interictal is not None:
        test_n = min(test_n, max_test_interictal)

    if test_interictal_mode == "contiguous":
        blocks = contiguous_interictal_blocks(rows, max_gap_sec=contiguous_gap_sec)
        candidates = [block for block in blocks if len(block) >= test_n]
        if candidates:
            rng = random.Random(stable_seed(seed, heldout_key))
            block = sorted(candidates, key=lambda item: (-len(item), item[0]["file_name"], int(item[0]["window_start_sec"])))[
                rng.randrange(len(candidates))
            ]
            max_start = len(block) - test_n
            start_index = rng.randrange(max_start + 1) if max_start > 0 else 0
            test_rows = block[start_index : start_index + test_n]
        else:
            block = max(blocks, key=len) if blocks else []
            test_rows = block[:test_n]
    else:
        rng = random.Random(stable_seed(seed, heldout_key))
        shuffled = list(rows)
        rng.shuffle(shuffled)
        test_rows = shuffled[:test_n]

    test_ids = {row_id(r) for r in test_rows}
    test_rows = [r for r in rows if row_id(r) in test_ids]
    train_rows = [r for r in rows if row_id(r) not in test_ids]
    return train_rows, test_rows


def select_fold_rows(
    subject: str,
    heldout: dict[str, str],
    subject_windows: list[dict[str, str]],
    seed: int,
    max_train_per_class: int | None,
    max_test_interictal: int | None,
    test_interictal_frac: float,
    test_interictal_mode: str,
    contiguous_interictal_gap_sec: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    heldout_key = event_key(heldout)
    train_preictal = []
    test_preictal = []
    interictal = []
    for row in subject_windows:
        if row["state"] == "interictal":
            interictal.append(row)
            continue
        if row["state"] != "preictal":
            continue
        if matched_event_key(row) == heldout_key:
            test_preictal.append(row)
        else:
            train_preictal.append(row)

    train_interictal_pool, test_interictal = split_interictal_rows(
        interictal,
        heldout_key=heldout_key,
        seed=seed,
        test_frac=test_interictal_frac,
        max_test_interictal=max_test_interictal,
        test_interictal_mode=test_interictal_mode,
        contiguous_gap_sec=contiguous_interictal_gap_sec,
    )

    train_target = min(len(train_preictal), len(train_interictal_pool))
    if max_train_per_class is not None:
        train_target = min(train_target, max_train_per_class)
    fold_seed = stable_seed(seed, heldout_key)
    train_rows = sample_rows(train_preictal, train_target, fold_seed + 1)
    train_rows.extend(sample_rows(train_interictal_pool, train_target, fold_seed + 2))
    random.Random(fold_seed + 3).shuffle(train_rows)

    test_rows = sorted(
        test_preictal + test_interictal,
        key=lambda r: (r["state"] != "preictal", r["file_name"], int(r["window_start_sec"]), int(r["window_end_sec"])),
    )
    train_preictal_pool = sorted(
        train_preictal,
        key=lambda r: (r["file_name"], int(r["window_start_sec"]), int(r["window_end_sec"])),
    )
    train_interictal_pool = sorted(
        train_interictal_pool,
        key=lambda r: (r["file_name"], int(r["window_start_sec"]), int(r["window_end_sec"])),
    )
    return train_rows, test_rows, train_preictal_pool, train_interictal_pool


def write_rows_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows_to_arrays(
    rows: list[dict[str, str]],
    data_root: Path,
    channels: list[str],
    split: str,
    target_samples: int | None = None,
    normalize_window: bool = True,
) -> tuple[object, object, object, list[dict[str, str]], list[dict[str, object]]]:
    import numpy as np

    def resample_time_axis(x, samples: int):
        if samples is None or x.shape[0] == samples:
            return x
        if x.shape[0] <= 1 or samples <= 1:
            raise ValueError(f"Cannot resample window with shape {x.shape} to {samples} samples")
        old_grid = np.linspace(0.0, 1.0, x.shape[0], endpoint=False)
        new_grid = np.linspace(0.0, 1.0, samples, endpoint=False)
        return np.stack(
            [np.interp(new_grid, old_grid, x[:, channel]) for channel in range(x.shape[1])],
            axis=1,
        ).astype("float32")

    x_parts = []
    y_parts = []
    subject_parts = []
    loaded_rows = []
    skipped_files = []
    rows_by_file: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_file[(row["subject"], row["file_name"])].append(row)

    loaded_count = 0
    for (subject, file_name), file_rows in sorted(rows_by_file.items()):
        edf_path = data_root / subject / file_name
        if not edf_path.exists():
            flat_path = data_root / file_name
            if flat_path.exists():
                edf_path = flat_path
        opened = open_edf(edf_path)
        skipped_reason = None
        try:
            for row in file_rows:
                try:
                    x = load_edf_window(
                        opened,
                        edf_path,
                        int(row["window_start_sec"]),
                        int(row["window_end_sec"]),
                        channels,
                    )
                except ValueError as exc:
                    skipped_reason = str(exc)
                    break
                x = resample_time_axis(x, target_samples) if target_samples is not None else x
                x_parts.append(zscore(x) if normalize_window else x.astype("float32"))
                y_parts.append(int(row["label"]))
                subject_parts.append(row["subject"])
                loaded_rows.append(row)
                loaded_count += 1
                if loaded_count % 500 == 0:
                    print(f"{split}: loaded {loaded_count}/{len(rows)} windows")
        finally:
            close_edf(opened)
        if skipped_reason:
            skipped_files.append(
                {
                    "split": split,
                    "subject": subject,
                    "file_name": file_name,
                    "windows": len(file_rows),
                    "reason": skipped_reason,
                }
            )
            print(f"{split}: skipped {subject}/{file_name}: {skipped_reason}")

    if not x_parts:
        raise ValueError(f"No {split} windows could be loaded.")
    return (
        np.stack(x_parts).astype("float32"),
        np.asarray(y_parts, dtype="int64"),
        np.asarray(subject_parts),
        loaded_rows,
        skipped_files,
    )


def build_fold_id(heldout: dict[str, str]) -> str:
    subject, file_name, seizure_idx = event_key(heldout)
    return f"{subject}_{Path(file_name).stem}_sz{int(seizure_idx):02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/chbmit/chb-mit-scalp-eeg-database-1.0.0"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/chbmit/annotations"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/chbmit/loocv"))
    parser.add_argument("--channels", default=",".join(COMMON_CHANNELS))
    parser.add_argument("--subjects", default="", help="Comma-separated subjects, e.g. chb01,chb03. Empty means all.")
    parser.add_argument("--limit-subjects", type=int, default=0)
    parser.add_argument("--limit-folds", type=int, default=0)
    parser.add_argument("--max-train-per-class", type=int, default=800)
    parser.add_argument("--max-test-interictal", type=int, default=2000)
    parser.add_argument("--test-interictal-frac", type=float, default=0.3)
    parser.add_argument("--test-interictal-mode", choices=("random", "contiguous"), default="random")
    parser.add_argument(
        "--contiguous-interictal-gap-sec",
        type=int,
        default=0,
        help="Maximum gap allowed between adjacent interictal test windows in contiguous mode.",
    )
    parser.add_argument("--min-test-positive", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    channels = [ch.strip() for ch in args.channels.split(",") if ch.strip()]
    max_train_per_class = args.max_train_per_class if args.max_train_per_class > 0 else None
    max_test_interictal = args.max_test_interictal if args.max_test_interictal > 0 else None

    seizure_rows = read_csv_rows(args.annotations_dir / "selected_subject_seizures.csv")
    window_rows = read_csv_rows(args.annotations_dir / "selected_subject_windows.csv")

    by_subject_events: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_subject_windows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in seizure_rows:
        by_subject_events[row["subject"]].append(row)
    for row in window_rows:
        by_subject_windows[row["subject"]].append(row)

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = sorted(by_subject_events)
    if args.limit_subjects:
        subjects = subjects[: args.limit_subjects]

    summary = {
        "data_root": str(args.data_root),
        "annotations_dir": str(args.annotations_dir),
        "channels": channels,
        "split_rule": "patient-specific seizure-level leave-one-out cross-validation",
        "train_positive_policy": "preictal windows from non-held-out seizures of the same subject",
        "test_positive_policy": "all preictal windows from held-out seizure",
        "train_interictal_policy": "reserved interictal windows excluded, then undersampled to balance train positives",
        "test_interictal_policy": f"deterministic {args.test_interictal_mode} subject-specific interictal reserve, unbalanced",
        "test_interictal_frac": args.test_interictal_frac,
        "test_interictal_mode": args.test_interictal_mode,
        "contiguous_interictal_gap_sec": args.contiguous_interictal_gap_sec,
        "max_train_per_class": max_train_per_class,
        "max_test_interictal": max_test_interictal,
        "min_test_positive": args.min_test_positive,
        "seed": args.seed,
        "folds": [],
        "skipped_folds": [],
    }

    total_folds = 0
    for subject in subjects:
        events = sorted(by_subject_events[subject], key=lambda r: (r["file_name"], int(r["seizure_start_sec"]), int(r["seizure_index_in_file"])))
        for heldout in events:
            if args.limit_folds and total_folds >= args.limit_folds:
                break
            fold_id = build_fold_id(heldout)
            fold_dir = args.out_dir / subject / fold_id
            train_rows, test_rows, train_preictal_pool, train_interictal_pool = select_fold_rows(
                subject=subject,
                heldout=heldout,
                subject_windows=by_subject_windows[subject],
                seed=args.seed,
                max_train_per_class=max_train_per_class,
                max_test_interictal=max_test_interictal,
                test_interictal_frac=args.test_interictal_frac,
                test_interictal_mode=args.test_interictal_mode,
                contiguous_interictal_gap_sec=args.contiguous_interictal_gap_sec,
            )
            test_positive = sum(int(r["label"]) == 1 for r in test_rows)
            if test_positive < args.min_test_positive:
                summary["skipped_folds"].append(
                    {
                        "fold_id": fold_id,
                        "subject": subject,
                        "heldout_file": heldout["file_name"],
                        "heldout_seizure_index_in_file": int(heldout["seizure_index_in_file"]),
                        "test_positive": test_positive,
                        "reason": f"test positive windows below min_test_positive={args.min_test_positive}",
                    }
                )
                print(f"Skipped fold {fold_id}: test_positive={test_positive}")
                continue
            fold_manifest = {
                "fold_id": fold_id,
                "subject": subject,
                "heldout_event": heldout,
                "selected_before_channel_skips": {
                    "train": {
                        "windows": len(train_rows),
                        "positive": sum(int(r["label"]) == 1 for r in train_rows),
                        "negative": sum(int(r["label"]) == 0 for r in train_rows),
                    },
                    "test": {
                        "windows": len(test_rows),
                        "positive": sum(int(r["label"]) == 1 for r in test_rows),
                        "negative": sum(int(r["label"]) == 0 for r in test_rows),
                    },
                    "train_preictal_pool": {
                        "windows": len(train_preictal_pool),
                        "positive": sum(int(r["label"]) == 1 for r in train_preictal_pool),
                        "negative": sum(int(r["label"]) == 0 for r in train_preictal_pool),
                    },
                    "train_interictal_pool": {
                        "windows": len(train_interictal_pool),
                        "positive": sum(int(r["label"]) == 1 for r in train_interictal_pool),
                        "negative": sum(int(r["label"]) == 0 for r in train_interictal_pool),
                    },
                },
                "skipped_files": [],
            }
            fold_dir.mkdir(parents=True, exist_ok=True)
            if args.metadata_only:
                write_rows_csv(fold_dir / "train_windows.csv", train_rows)
                write_rows_csv(fold_dir / "train_preictal_pool_windows.csv", train_preictal_pool)
                write_rows_csv(fold_dir / "train_interictal_pool_windows.csv", train_interictal_pool)
                write_rows_csv(fold_dir / "test_windows.csv", test_rows)
            else:
                x_train, y_train, subjects_train, loaded_train_rows, skipped_train = load_rows_to_arrays(
                    train_rows, args.data_root, channels, "train"
                )
                x_test, y_test, subjects_test, loaded_test_rows, skipped_test = load_rows_to_arrays(
                    test_rows, args.data_root, channels, "test"
                )
                fold_manifest["skipped_files"] = skipped_train + skipped_test
                fold_manifest["loaded_counts"] = {
                    "train": {
                        "windows": int(y_train.shape[0]),
                        "positive": int((y_train == 1).sum()),
                        "negative": int((y_train == 0).sum()),
                    },
                    "test": {
                        "windows": int(y_test.shape[0]),
                        "positive": int((y_test == 1).sum()),
                        "negative": int((y_test == 0).sum()),
                    },
                }
                write_rows_csv(fold_dir / "train_windows.csv", loaded_train_rows)
                write_rows_csv(fold_dir / "train_preictal_pool_windows.csv", train_preictal_pool)
                write_rows_csv(fold_dir / "train_interictal_pool_windows.csv", train_interictal_pool)
                write_rows_csv(fold_dir / "test_windows.csv", loaded_test_rows)
                import numpy as np

                np.savez_compressed(
                    fold_dir / "dataset.npz",
                    x_train=x_train,
                    y_train=y_train,
                    subjects_train=subjects_train,
                    x_test=x_test,
                    y_test=y_test,
                    subjects_test=subjects_test,
                )

            (fold_dir / "manifest.json").write_text(json.dumps(fold_manifest, indent=2) + "\n")
            summary["folds"].append(
                {
                    "fold_id": fold_id,
                    "subject": subject,
                    "dir": str(fold_dir),
                    "heldout_file": heldout["file_name"],
                    "heldout_seizure_index_in_file": int(heldout["seizure_index_in_file"]),
                    "selected_before_channel_skips": fold_manifest["selected_before_channel_skips"],
                    "loaded_counts": fold_manifest.get("loaded_counts"),
                    "skipped_files": len(fold_manifest["skipped_files"]),
                }
            )
            total_folds += 1
            print(f"Wrote fold {fold_id} -> {fold_dir}")
        if args.limit_folds and total_folds >= args.limit_folds:
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "loocv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}")
    print(json.dumps({"folds": len(summary["folds"]), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
