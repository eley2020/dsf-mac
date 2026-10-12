#!/usr/bin/env python3
"""Build non-federated CHB-MIT train/test arrays from generated window labels.

The split is patient-wise and event-wise:
  * train positives: preictal windows linked to all but the last seizure
  * test positives: preictal windows linked to the last seizure
  * train/test negatives: interictal windows before/after the split boundary

The output is a compact NPZ file that can be consumed by
``train_chbmit_baseline.py``. This script requires numpy and an EDF reader
(`mne` preferred, `pyedflib` fallback).
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path


COMMON_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
]

CHANNEL_ALIASES = {
    "T7": ["T7", "T3"],
    "T8": ["T8", "T4"],
    "P7": ["P7", "T5"],
    "P8": ["P8", "T6"],
    "T3": ["T3", "T7"],
    "T4": ["T4", "T8"],
    "T5": ["T5", "P7"],
    "T6": ["T6", "P8"],
}


def normalize_channel_name(name: str) -> str:
    normalized = (
        name.upper()
        .replace("EEG ", "")
        .replace("-REF", "")
        .replace("-LE", "")
        .replace("-CS2", "")
        .replace(".", "")
        .strip()
    )
    if normalized == "01":
        normalized = "O1"
    elif normalized == "02":
        normalized = "O2"
    # Legacy paper-protocol typo: CHB-MIT uses the P4-O2 derivation; there is
    # no FP2-O2 channel in the database montage.  Normalize at the EDF
    # boundary so all callers (including older command lines) remain robust.
    if normalized == "FP2-O2":
        normalized = "P4-O2"
    # MNE disambiguates duplicate EDF labels as NAME-0, NAME-1. Treat those
    # suffixes as the same bipolar channel when matching the CHB-MIT montage.
    if normalized.endswith("-0") or normalized.endswith("-1"):
        normalized = normalized[:-2]
    return normalized


def channel_key_candidates(name: str) -> list[str]:
    key = normalize_channel_name(name)
    return CHANNEL_ALIASES.get(key, [key])


def resolve_single_channel(name: str, available: dict[str, object]) -> object | None:
    for key in channel_key_candidates(name):
        if key in available:
            return available[key]
    return None


def is_seizeit2_generic_channel(name: str) -> bool:
    key = normalize_channel_name(name).replace(" ", "_").replace("-", "_")
    return key in {"SEIZEIT2_EEG1", "SEIZEIT2_EEG2"}


def seizeit2_generic_index(name: str) -> int:
    key = normalize_channel_name(name).replace(" ", "_").replace("-", "_")
    if key == "SEIZEIT2_EEG1":
        return 0
    if key == "SEIZEIT2_EEG2":
        return 1
    raise ValueError(f"Not a SeizeIT2 generic channel: {name}")


def non_annotation_signal_names(names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if "ANNOTATION" not in normalize_channel_name(name)
    ]


def open_edf(path: Path):
    try:
        import mne

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Channel names are not unique.*", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="Scaling factor is not defined.*", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="Physical range is not defined.*", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="Digital range is not defined.*", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="Channels contain different highpass filters.*", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="Channels contain different lowpass filters.*", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="Highpass cutoff frequency .* is greater than lowpass cutoff frequency .*", category=RuntimeWarning)
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        return "mne", raw
    except ImportError:
        import pyedflib

        return "pyedflib", pyedflib.EdfReader(str(path))


def close_edf(opened) -> None:
    kind, reader = opened
    if kind == "pyedflib":
        reader.close()


def load_edf_window(opened, path: Path, start_sec: int, end_sec: int, channels: list[str]) -> "object":
    kind, reader = opened
    if kind == "mne":
        import numpy as np

        raw = reader
        available: dict[str, str] = {}
        for ch in raw.ch_names:
            available.setdefault(normalize_channel_name(ch), ch)
        generic_picks = non_annotation_signal_names(raw.ch_names)
        exact_picks = []
        derived_pairs = []
        missing = []
        for ch in channels:
            key = normalize_channel_name(ch)
            if is_seizeit2_generic_channel(ch):
                generic_idx = seizeit2_generic_index(ch)
                if generic_idx < len(generic_picks):
                    exact_picks.append((ch, generic_picks[generic_idx]))
                else:
                    missing.append(ch)
            elif key in available:
                exact_picks.append((ch, available[key]))
            elif "-" in key:
                left, right = key.split("-", 1)
                left_pick = resolve_single_channel(left, available)
                right_pick = resolve_single_channel(right, available)
                if left_pick is not None and right_pick is not None:
                    derived_pairs.append((ch, left_pick, right_pick))
                else:
                    missing.append(ch)
            else:
                missing.append(ch)
        if missing:
            raise ValueError(f"{path} missing channels: {missing}")
        start_sample = int(start_sec * raw.info["sfreq"])
        stop_sample = int(end_sec * raw.info["sfreq"])
        channel_arrays = {}
        needed_picks = [pick for _, pick in exact_picks]
        needed_picks.extend([pick for _, left, right in derived_pairs for pick in (left, right)])
        needed_picks = list(dict.fromkeys(needed_picks))
        raw_data = raw.get_data(picks=needed_picks, start=start_sample, stop=stop_sample)
        pick_to_array = {pick: raw_data[index] for index, pick in enumerate(needed_picks)}
        for requested, pick in exact_picks:
            channel_arrays[requested] = pick_to_array[pick]
        for requested, left_pick, right_pick in derived_pairs:
            channel_arrays[requested] = pick_to_array[left_pick] - pick_to_array[right_pick]
        data = np.stack([channel_arrays[ch] for ch in channels], axis=1).astype("float32")
        expected = int((end_sec - start_sec) * raw.info["sfreq"])
        if data.shape[0] != expected:
            raise ValueError(f"{path} returned {data.shape[0]} samples, expected {expected}")
        return data
    else:
        import numpy as np

        labels: dict[str, int] = {}
        raw_labels = []
        for i in range(reader.signals_in_file):
            raw_label = reader.getLabel(i)
            raw_labels.append(raw_label)
            labels.setdefault(normalize_channel_name(raw_label), i)
        generic_idxs = [
            idx
            for idx, label in enumerate(raw_labels)
            if "ANNOTATION" not in normalize_channel_name(label)
        ]
        exact_idxs = []
        derived_pairs = []
        missing = []
        for ch in channels:
            key = normalize_channel_name(ch)
            if is_seizeit2_generic_channel(ch):
                generic_idx = seizeit2_generic_index(ch)
                if generic_idx < len(generic_idxs):
                    exact_idxs.append((ch, generic_idxs[generic_idx]))
                else:
                    missing.append(ch)
            elif key in labels:
                exact_idxs.append((ch, labels[key]))
            elif "-" in key:
                left, right = key.split("-", 1)
                left_idx = resolve_single_channel(left, labels)
                right_idx = resolve_single_channel(right, labels)
                if left_idx is not None and right_idx is not None:
                    derived_pairs.append((ch, left_idx, right_idx))
                else:
                    missing.append(ch)
            else:
                missing.append(ch)
        if missing:
            raise ValueError(f"{path} missing channels: {missing}")
        all_idxs = [idx for _, idx in exact_idxs]
        all_idxs.extend([idx for _, left, right in derived_pairs for idx in (left, right)])
        all_idxs = list(dict.fromkeys(all_idxs))
        fs = int(reader.getSampleFrequency(all_idxs[0]))
        start = start_sec * fs
        count = (end_sec - start_sec) * fs
        idx_to_array = {idx: reader.readSignal(idx, start=start, n=count) for idx in all_idxs}
        channel_arrays = {}
        for requested, idx in exact_idxs:
            channel_arrays[requested] = idx_to_array[idx]
        for requested, left_idx, right_idx in derived_pairs:
            channel_arrays[requested] = idx_to_array[left_idx] - idx_to_array[right_idx]
        return np.asarray([channel_arrays[ch] for ch in channels], dtype="float32").T


def zscore(x: "object") -> "object":
    import numpy as np

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def build_split_metadata(seizure_rows: list[dict[str, str]]) -> dict[str, tuple[str, int, int]]:
    by_subject: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for row in seizure_rows:
        subject = row["subject"]
        file_name = row["file_name"]
        seizure_start = int(row["seizure_start_sec"])
        seizure_index = int(row["seizure_index_in_file"])
        by_subject[subject].append((file_name, seizure_start, seizure_index))

    split: dict[str, tuple[str, int, int]] = {}
    for subject, events in by_subject.items():
        events = sorted(events, key=lambda x: (x[0], x[1], x[2]))
        split[subject] = events[-1]
    return split


def assign_split(row: dict[str, str], last_event: tuple[str, int, int]) -> str:
    file_name = row["file_name"]
    start = int(row["window_start_sec"])
    state = row["state"]
    last_file, last_start, last_index = last_event

    if state == "preictal":
        matched = row["matched_seizure_index_in_file"]
        if file_name == last_file and matched and int(matched) == last_index:
            return "test"
        return "train"

    if file_name > last_file or (file_name == last_file and start >= max(0, last_start - 35 * 60)):
        return "test"
    return "train"


def balanced_take(rows: list[dict[str, str]], seed: int, max_per_class: int | None) -> list[dict[str, str]]:
    import random

    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    if len(by_label) < 2:
        return rows
    target = min(len(v) for v in by_label.values())
    if max_per_class is not None:
        target = min(target, max_per_class)
    sampled = []
    for label_rows in by_label.values():
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        sampled.extend(label_rows[:target])
    rng.shuffle(sampled)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/chbmit/chb-mit-scalp-eeg-database-1.0.0"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/chbmit/annotations"))
    parser.add_argument("--out", type=Path, default=Path("data/chbmit/baseline/chbmit_baseline_leave_last.npz"))
    parser.add_argument("--channels", default=",".join(COMMON_CHANNELS))
    parser.add_argument("--max-train-per-class-per-subject", type=int, default=800)
    parser.add_argument("--max-test-per-class-per-subject", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-subjects", type=int, default=0, help="Debug only: process first N selected subjects.")
    args = parser.parse_args()

    import numpy as np

    channels = [ch.strip() for ch in args.channels.split(",") if ch.strip()]
    seizure_rows = read_csv_rows(args.annotations_dir / "selected_subject_seizures.csv")
    window_rows = read_csv_rows(args.annotations_dir / "selected_subject_windows.csv")
    last_events = build_split_metadata(seizure_rows)
    subjects = sorted(last_events)
    if args.limit_subjects:
        subjects = subjects[: args.limit_subjects]
    subjects_set = set(subjects)

    split_rows = {"train": [], "test": []}
    for row in window_rows:
        subject = row["subject"]
        if subject not in subjects_set:
            continue
        split = assign_split(row, last_events[subject])
        split_rows[split].append(row)

    selected_rows: dict[str, list[dict[str, str]]] = {}
    for split, rows in split_rows.items():
        by_subject = defaultdict(list)
        for row in rows:
            by_subject[row["subject"]].append(row)
        selected = []
        for subject in sorted(by_subject):
            max_per_class = args.max_train_per_class_per_subject if split == "train" else args.max_test_per_class_per_subject
            selected.extend(balanced_take(by_subject[subject], seed=args.seed, max_per_class=max_per_class))
        selected_rows[split] = selected

    arrays = {}
    manifest = {
        "data_root": str(args.data_root),
        "annotations_dir": str(args.annotations_dir),
        "channels": channels,
        "split_rule": "leave-last-seizure-out by subject",
        "incompatible_edf_policy": "skip files that cannot provide all requested channels",
        "subjects": subjects,
        "max_train_per_class_per_subject": args.max_train_per_class_per_subject,
        "max_test_per_class_per_subject": args.max_test_per_class_per_subject,
        "seed": args.seed,
        "counts": {},
        "skipped_files": [],
    }

    for split, rows in selected_rows.items():
        x_parts = []
        y_parts = []
        subject_parts = []
        rows_by_file: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            rows_by_file[(row["subject"], row["file_name"])].append(row)
        loaded_count = 0
        for (subject, file_name), file_rows in sorted(rows_by_file.items()):
            edf_path = args.data_root / subject / file_name
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
                    x_parts.append(zscore(x))
                    y_parts.append(int(row["label"]))
                    subject_parts.append(row["subject"])
                    loaded_count += 1
                    if loaded_count % 500 == 0:
                        print(f"{split}: loaded {loaded_count}/{len(rows)} windows")
            finally:
                close_edf(opened)
            if skipped_reason:
                manifest["skipped_files"].append(
                    {
                        "split": split,
                        "subject": subject,
                        "file_name": file_name,
                        "windows": len(file_rows),
                        "reason": skipped_reason,
                    }
                )
                print(f"{split}: skipped {subject}/{file_name}: {skipped_reason}")
        arrays[f"x_{split}"] = np.stack(x_parts).astype("float32")
        arrays[f"y_{split}"] = np.asarray(y_parts, dtype="int64")
        arrays[f"subjects_{split}"] = np.asarray(subject_parts)
        manifest["counts"][split] = {
            "selected_windows_before_skips": len(rows),
            "loaded_windows": int(arrays[f"y_{split}"].shape[0]),
            "positive": int((arrays[f"y_{split}"] == 1).sum()),
            "negative": int((arrays[f"y_{split}"] == 0).sum()),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    manifest_path = args.out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.out}")
    print(f"Wrote {manifest_path}")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
