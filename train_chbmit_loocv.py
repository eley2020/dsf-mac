#!/usr/bin/env python3
"""Train and evaluate CNN-GRU baseline across CHB-MIT seizure-level LOOCV folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path

from build_chbmit_baseline_dataset import COMMON_CHANNELS
from build_chbmit_baseline_dataset import close_edf, load_edf_window, open_edf, zscore
from build_chbmit_loocv_dataset import contiguous_interictal_blocks, load_rows_to_arrays
from chbmit_models import build_model
from evaluate_chbmit_loocv_event_level import evaluate_fold as evaluate_alarm_fold
from evaluate_chbmit_loocv_event_level import time_end, time_start
from train_chbmit_baseline import metrics_from_outputs


# Exact descriptor rows are stored in this channel order.  A length-only
# check is insufficient because cache misses are recomputed from raw EDFs.
DCSA_EXACT_CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3",
    "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rows_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scored_rows(rows: list[dict[str, str]], scores) -> list[dict[str, str]]:
    """Attach model scores to metadata rows for sequential alarm evaluation."""
    return [
        {**row, "score_preictal": str(float(score))}
        for row, score in zip(rows, scores, strict=True)
    ]


def validation_alarm_selection_key(
    alarm_metrics: dict[str, object],
    window_metrics: dict[str, object],
) -> tuple[float, float, float, float]:
    """Rank checkpoints by a pre-registered event-first validation objective."""
    warning = alarm_metrics.get("event_warning_time_min", "")
    try:
        warning_value = float(warning)
        if not math.isfinite(warning_value):
            warning_value = -float("inf")
    except (TypeError, ValueError):
        warning_value = -float("inf")
    auc = float(window_metrics.get("auc", float("nan")))
    if not math.isfinite(auc):
        auc = -float("inf")
    return (
        float(alarm_metrics.get("event_detected", 0.0)),
        -float(alarm_metrics.get("false_alarm_rate_per_hour", float("inf"))),
        warning_value,
        auc,
    )


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def validation_event_key(row: dict[str, str]) -> tuple[str, str] | None:
    if row.get("state") != "preictal" or int(row.get("label", 0)) != 1:
        return None
    matched = row.get("matched_seizure_index_in_file", "")
    if not matched:
        return None
    seizure_file = row.get("matched_seizure_file_name", "").strip() or row["file_name"]
    return seizure_file, matched


def validation_event_start(rows: list[dict[str, str]]) -> int:
    """Return the earliest window start for one validation seizure event."""
    return min((time_start(row) for row in rows), default=0)


def validation_event_order(key: tuple[str, str], rows: list[dict[str, str]]) -> tuple[tuple[object, ...], int, str, int]:
    """Order events by recording-file number, then onset within that file.

    CHB-MIT splits long recordings across EDF files, so an EDF-local second
    cannot be compared across files. Natural filename ordering preserves the
    recording timeline (for example, ``chb12_09.edf`` precedes
    ``chb12_38.edf``).
    """
    file_name, seizure_index = key
    file_order = tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", file_name)
        if part
    )
    return file_order, validation_event_start(rows), file_name, int(seizure_index)


def window_row_id(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["file_name"],
        str(row["window_start_sec"]),
        str(row["window_end_sec"]),
        str(row.get("label", "")),
    )


def select_temporal_consistency_rows(
    candidate_rows: list[dict[str, str]],
    excluded_rows: list[dict[str, str]],
    fold_id: str,
    seed: int,
    max_windows: int,
    max_gap_sec: int,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    if max_windows <= 0:
        return [], {
            "enabled": False,
            "reason": "max_windows <= 0",
        }
    excluded_ids = {window_row_id(row) for row in excluded_rows}
    eligible_rows = [
        row for row in candidate_rows
        if row.get("state") == "interictal"
        and int(row.get("label", 0)) == 0
        and window_row_id(row) not in excluded_ids
    ]
    blocks = contiguous_interictal_blocks(eligible_rows, max_gap_sec=max_gap_sec)
    if not blocks:
        return [], {
            "enabled": True,
            "created": False,
            "reason": "no eligible contiguous interictal blocks",
            "candidate_rows": len(candidate_rows),
            "eligible_rows": len(eligible_rows),
            "max_windows": max_windows,
        }
    block = sorted(
        blocks,
        key=lambda item: (-len(item), item[0]["file_name"], int(item[0]["window_start_sec"])),
    )[0]
    selected_n = min(len(block), max_windows)
    rng = random.Random(stable_seed(seed, f"{fold_id}|temporal_consistency"))
    max_start = len(block) - selected_n
    start = rng.randint(0, max_start) if max_start > 0 else 0
    selected_rows = block[start : start + selected_n]
    return selected_rows, {
        "enabled": True,
        "created": True,
        "source": "interictal_pool",
        "candidate_rows": len(candidate_rows),
        "eligible_rows": len(eligible_rows),
        "blocks": len(blocks),
        "largest_block_windows": len(block),
        "selected_windows": len(selected_rows),
        "selected_file": selected_rows[0]["file_name"] if selected_rows else "",
        "selected_start_sec": int(selected_rows[0]["window_start_sec"]) if selected_rows else "",
        "selected_end_sec": int(selected_rows[-1]["window_end_sec"]) if selected_rows else "",
        "max_windows": max_windows,
        "gap_sec": max_gap_sec,
    }


def split_train_validation_rows(
    train_rows: list[dict[str, str]],
    fold_id: str,
    seed: int,
    enabled: bool,
    min_positive_windows: int,
    event_count: int,
    negative_ratio: float,
    min_train_negative_ratio: float,
    negative_mode: str = "random",
    contiguous_gap_sec: int = 0,
    event_selection: str = "random",
    validation_positive_rows: list[dict[str, str]] | None = None,
    validation_negative_rows: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    if not enabled:
        return train_rows, [], {"enabled": False}

    positive_source_rows = validation_positive_rows if validation_positive_rows is not None else train_rows
    negative_source_rows = validation_negative_rows if validation_negative_rows is not None else train_rows

    positive_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    interictal_rows = []
    for row in positive_source_rows:
        key = validation_event_key(row)
        if key is None:
            continue
        positive_groups.setdefault(key, []).append(row)
    for row in negative_source_rows:
        if row.get("state") == "interictal" and int(row.get("label", 0)) == 0:
            interictal_rows.append(row)

    candidate_keys = [
        key for key, rows in positive_groups.items()
        if len(rows) >= min_positive_windows
    ]

    if not candidate_keys or not interictal_rows:
        return train_rows, [], {
            "enabled": True,
            "created": False,
            "reason": "need at least one sufficiently large training seizure group and interictal rows",
            "min_positive_windows": min_positive_windows,
            "event_count": event_count,
            "negative_ratio": negative_ratio,
            "min_train_negative_ratio": min_train_negative_ratio,
            "available_positive_groups": {
                f"{key[0]}#{key[1]}": len(rows)
                for key, rows in sorted(positive_groups.items(), key=lambda item: (item[0][0], int(item[0][1])))
            },
        }

    rng = random.Random(stable_seed(seed, fold_id))
    candidate_keys = sorted(
        candidate_keys,
        key=lambda key: validation_event_order(key, positive_groups[key]),
    )
    requested_event_count = max(1, event_count)
    selected_count = min(requested_event_count, len(candidate_keys))
    if event_selection == "random":
        selected_keys = rng.sample(candidate_keys, selected_count)
        selected_keys = sorted(
            selected_keys,
            key=lambda key: validation_event_order(key, positive_groups[key]),
        )
    elif event_selection == "chronological_first":
        selected_keys = candidate_keys[:selected_count]
    elif event_selection == "chronological_last":
        selected_keys = candidate_keys[-selected_count:]
    else:
        raise ValueError(f"Unsupported validation event selection: {event_selection}")
    val_positive_rows = [
        row
        for selected_key in selected_keys
        for row in positive_groups[selected_key]
    ]
    val_positive_ids = {
        (row["file_name"], row["window_start_sec"], row["window_end_sec"], row["label"])
        for row in val_positive_rows
    }

    default_negative_target = min(
        len(val_positive_ids),
        max(selected_count, len(interictal_rows) * selected_count // max(len(candidate_keys), 1)),
    )
    if negative_ratio <= 1.0:
        val_negative_target = default_negative_target
    else:
        ratio_negative_target = max(selected_count, int(round(len(val_positive_ids) * negative_ratio)))
        val_negative_target = min(len(interictal_rows), max(default_negative_target, ratio_negative_target))

    remaining_train_positive = sum(len(rows) for key, rows in positive_groups.items() if key not in selected_keys)
    min_train_negative_target = int(round(remaining_train_positive * max(0.0, min_train_negative_ratio)))
    max_validation_negative_by_train_floor = max(1, len(interictal_rows) - min_train_negative_target)
    val_negative_target = min(val_negative_target, max_validation_negative_by_train_floor)

    if negative_mode == "contiguous":
        blocks = contiguous_interictal_blocks(interictal_rows, max_gap_sec=contiguous_gap_sec)
        candidate_blocks = [block for block in blocks if len(block) >= val_negative_target]
        if candidate_blocks:
            block = sorted(
                candidate_blocks,
                key=lambda item: (-len(item), item[0]["file_name"], int(item[0]["window_start_sec"])),
            )[0]
            max_start = len(block) - val_negative_target
            start = rng.randint(0, max_start) if max_start > 0 else 0
            val_negative_rows = block[start : start + val_negative_target]
        elif blocks:
            block = sorted(
                blocks,
                key=lambda item: (-len(item), item[0]["file_name"], int(item[0]["window_start_sec"])),
            )[0]
            val_negative_rows = list(block)
        else:
            val_negative_rows = []
    elif negative_mode == "random":
        shuffled_interictal = list(interictal_rows)
        rng.shuffle(shuffled_interictal)
        val_negative_rows = shuffled_interictal[:val_negative_target]
    else:
        raise ValueError(f"Unsupported validation negative mode: {negative_mode}")

    val_negative_ids = {
        (row["file_name"], row["window_start_sec"], row["window_end_sec"], row["label"])
        for row in val_negative_rows
    }

    new_train_rows = []
    for row in train_rows:
        row_id = (row["file_name"], row["window_start_sec"], row["window_end_sec"], row["label"])
        if row_id not in val_positive_ids and row_id not in val_negative_ids:
            new_train_rows.append(row)

    val_rows = sorted(
        val_positive_rows + val_negative_rows,
        key=lambda row: (
            row["state"] != "preictal",
            row["file_name"],
            int(row["window_start_sec"]),
            int(row["window_end_sec"]),
        ),
    )

    train_pos = sum(int(row["label"]) == 1 for row in new_train_rows)
    train_neg = sum(int(row["label"]) == 0 for row in new_train_rows)
    val_pos = sum(int(row["label"]) == 1 for row in val_rows)
    val_neg = sum(int(row["label"]) == 0 for row in val_rows)
    if min(train_pos, train_neg, val_pos, val_neg) == 0:
        return train_rows, [], {
            "enabled": True,
            "created": False,
            "reason": "split would remove a required class",
            "selected_events": [
                {"file_name": key[0], "seizure_index_in_file": key[1]}
                for key in selected_keys
            ],
        }

    return new_train_rows, val_rows, {
        "enabled": True,
        "created": True,
        "selected_events": [
            {"file_name": key[0], "seizure_index_in_file": key[1]}
            for key in selected_keys
        ],
        "min_positive_windows": min_positive_windows,
        "requested_event_count": requested_event_count,
        "selected_event_count": selected_count,
        "negative_ratio": negative_ratio,
        "min_train_negative_ratio": min_train_negative_ratio,
        "negative_mode": negative_mode,
        "event_selection": event_selection,
        "contiguous_gap_sec": contiguous_gap_sec,
        "positive_source": "pool" if validation_positive_rows is not None else "train_windows",
        "negative_source": "pool" if validation_negative_rows is not None else "train_windows",
        "default_negative_target": default_negative_target,
        "min_train_negative_target": min_train_negative_target,
        "max_validation_negative_by_train_floor": max_validation_negative_by_train_floor,
        "selected_negative_target": val_negative_target,
        "selected_negative_actual": len(val_negative_rows),
        "eligible_positive_groups": {
            f"{key[0]}#{key[1]}": len(positive_groups[key])
            for key in candidate_keys
        },
        "counts": {
            "train": {"windows": len(new_train_rows), "positive": train_pos, "negative": train_neg},
            "validation": {"windows": len(val_rows), "positive": val_pos, "negative": val_neg},
        },
    }


def load_or_build_npz(
    fold_meta_dir: Path,
    fold_work_dir: Path,
    data_root: Path,
    channels: list[str],
    rebuild: bool,
    validation_mode: str = "none",
    seed: int = 0,
    validation_min_positive: int = 30,
    validation_event_count: int = 1,
    validation_negative_ratio: float = 1.0,
    validation_min_train_negative_ratio: float = 1.0,
    validation_negative_mode: str = "random",
    validation_contiguous_gap_sec: int = 0,
    validation_event_selection: str = "random",
    multiscale_5s: bool = False,
    multiscale_scales_sec: tuple[int, ...] = (5, 10, 15),
    multiscale_required_context_windows: int = 3,
    temporal_consistency_source: str = "train_windows",
    temporal_consistency_pool_windows: int = 0,
    temporal_consistency_gap_sec: int = 0,
    target_samples: int | None = None,
    emd_features: bool = False,
    emd_components: int = 3,
    emd_downsample: int = 2,
    emd_workers: int = 1,
    emd_channel_indices: tuple[int, ...] = (0, 4, 8, 12),
    normalize_window: bool = True,
    dcsa_feature_cache: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    import numpy as np

    dataset_path = fold_work_dir / "dataset.npz"
    manifest_path = fold_work_dir / "manifest.json"
    if dataset_path.exists() and manifest_path.exists() and not rebuild:
        cached_manifest = json.loads(manifest_path.read_text())
        cached_emd = cached_manifest.get("emd", {})
        if (
            bool(cached_emd.get("enabled", False)) == bool(emd_features)
            and int(cached_emd.get("components", 0)) == (emd_components if emd_features else 0)
            and int(cached_emd.get("downsample", 0)) == (emd_downsample if emd_features else 0)
            and tuple(cached_emd.get("channel_indices", [])) == (emd_channel_indices if emd_features else ())
            and bool(cached_manifest.get("window_zscore", True)) == bool(normalize_window)
            and bool(cached_manifest.get("dcsa_exact_feature_cache", {}).get("enabled", False))
            == bool(dcsa_feature_cache is not None)
        ):
            return dataset_path, cached_manifest

    train_rows = read_rows(fold_meta_dir / "train_windows.csv")
    validation_positive_rows = (
        read_rows(fold_meta_dir / "train_preictal_pool_windows.csv")
        if (fold_meta_dir / "train_preictal_pool_windows.csv").exists()
        else None
    )
    validation_negative_rows = (
        read_rows(fold_meta_dir / "train_interictal_pool_windows.csv")
        if (fold_meta_dir / "train_interictal_pool_windows.csv").exists()
        else None
    )
    train_rows, val_rows, validation_manifest = split_train_validation_rows(
        train_rows=train_rows,
        fold_id=fold_work_dir.name,
        seed=seed,
        enabled=validation_mode == "event",
        min_positive_windows=validation_min_positive,
        event_count=validation_event_count,
        negative_ratio=validation_negative_ratio,
        min_train_negative_ratio=validation_min_train_negative_ratio,
        negative_mode=validation_negative_mode,
        contiguous_gap_sec=validation_contiguous_gap_sec,
        event_selection=validation_event_selection,
        validation_positive_rows=validation_positive_rows,
        validation_negative_rows=validation_negative_rows,
    )
    test_rows = read_rows(fold_meta_dir / "test_windows.csv")
    temporal_consistency_rows: list[dict[str, str]] = []
    temporal_consistency_manifest: dict[str, object] = {"enabled": False, "source": temporal_consistency_source}
    if temporal_consistency_source == "interictal_pool" and validation_negative_rows is not None:
        temporal_consistency_rows, temporal_consistency_manifest = select_temporal_consistency_rows(
            candidate_rows=validation_negative_rows,
            excluded_rows=val_rows,
            fold_id=fold_work_dir.name,
            seed=seed,
            max_windows=temporal_consistency_pool_windows,
            max_gap_sec=temporal_consistency_gap_sec,
        )

    multiscale_manifest: dict[str, object] = {"enabled": False}
    if multiscale_5s:
        train_rows, train_multiscale_manifest = build_multiscale_5s_label_rows(
            train_rows,
            required_context_windows=multiscale_required_context_windows,
        )
        test_rows, test_multiscale_manifest = build_multiscale_5s_label_rows(
            test_rows,
            required_context_windows=multiscale_required_context_windows,
        )
        if val_rows:
            val_rows, val_multiscale_manifest = build_multiscale_5s_label_rows(
                val_rows,
                required_context_windows=multiscale_required_context_windows,
            )
        else:
            val_multiscale_manifest = {"enabled": True, "input_rows": 0, "output_rows": 0}
        multiscale_manifest = {
            "enabled": True,
            "scales_sec": list(multiscale_scales_sec),
            "required_context_windows": multiscale_required_context_windows,
            "label_rule": (
                "target is positive iff all required contiguous 5-second labels "
                "ending at the current time are positive"
            ),
            "train": train_multiscale_manifest,
            "validation": val_multiscale_manifest,
            "test": test_multiscale_manifest,
        }
        loader = lambda rows, split: load_rows_to_multiscale_arrays(
            rows,
            data_root,
            channels,
            split,
            scales_sec=multiscale_scales_sec,
        )
    else:
        loader = lambda rows, split: load_rows_to_arrays(
            rows,
            data_root,
            channels,
            split,
            target_samples=target_samples,
            normalize_window=normalize_window,
        )

    x_train, y_train, subjects_train, loaded_train_rows, skipped_train = loader(train_rows, "train")
    x_val = y_val = subjects_val = None
    loaded_val_rows = []
    skipped_val = []
    if val_rows:
        x_val, y_val, subjects_val, loaded_val_rows, skipped_val = loader(val_rows, "validation")
    x_test, y_test, subjects_test, loaded_test_rows, skipped_test = loader(test_rows, "test")
    x_temporal_consistency = None
    loaded_temporal_consistency_rows = []
    skipped_temporal_consistency = []
    if temporal_consistency_rows:
        try:
            (
                x_temporal_consistency,
                _y_temporal_consistency,
                _subjects_temporal_consistency,
                loaded_temporal_consistency_rows,
                skipped_temporal_consistency,
            ) = loader(temporal_consistency_rows, "temporal_consistency")
        except ValueError as exc:
            # A pool can become empty after the common-channel filter. The
            # training loop already supports train-window pairs, so retain the
            # fold and record the fallback instead of dropping it.
            temporal_consistency_manifest = {
                **temporal_consistency_manifest,
                "pool_fallback": True,
                "pool_fallback_reason": str(exc),
            }
            x_temporal_consistency = None
            loaded_temporal_consistency_rows = []
            skipped_temporal_consistency = []

    # Optional exact DC-SA descriptor cache.  The cache is global across
    # folds; rows are looked up by subject/file/start/end, while labels remain
    # owned by the fold metadata.  This keeps test descriptors independent of
    # fold standardization and avoids recomputing nonlinear features.
    feature_cache_manifest: dict[str, object] = {"enabled": False}
    if dcsa_feature_cache is not None:
        import numpy as np
        cache_npz = dcsa_feature_cache / "features.npz"
        if not cache_npz.exists():
            raise FileNotFoundError(f"Exact DC-SA feature cache not found: {cache_npz}")
        cache = np.load(cache_npz, allow_pickle=True)
        cache_keys = [tuple(str(v) for v in key) for key in cache["keys"]]
        cache_index = {key: i for i, key in enumerate(cache_keys)}
        subject = fold_meta_dir.parent.name
        def cached(rows, expected, split_name):
            keys = [(subject, r["file_name"], str(r["window_start_sec"]), str(r["window_end_sec"])) for r in rows]
            missing_indices = [i for i, key in enumerate(keys) if key not in cache_index]
            values = np.empty((len(rows), 18, 24), dtype=np.float32)
            present_indices = [i for i, key in enumerate(keys) if key in cache_index]
            if present_indices:
                values[np.asarray(present_indices, dtype=np.int64)] = cache["features"][
                    np.asarray([cache_index[keys[i]] for i in present_indices], dtype=np.int64)
                ].reshape(len(present_indices), 18, 24).astype("float32", copy=False)
            if missing_indices:
                # Validation pools were not part of the initial global cache
                # extraction.  Compute only those few descriptors from the
                # already loaded raw windows; this keeps an interrupted/cache
                # partial run resumable without silently dropping a fold.
                import scipy.signal as scipy_signal
                import torch
                from extract_dcsa_exact_features import exact_batch
                hp_sos = scipy_signal.butter(4, 0.5, btype="highpass", fs=256, output="sos")
                notch_ba = scipy_signal.iirnotch(60, Q=30, fs=256)
                print(
                    f"{split_name}: exact cache missing {len(missing_indices)} windows; "
                    "computing descriptors from loaded raw windows",
                    flush=True,
                )
                raw = expected[np.asarray(missing_indices, dtype=np.int64)]
                computed = []
                for start in range(0, len(raw), 64):
                    computed.append(
                        exact_batch(
                            raw[start : start + 64],
                            hp_sos,
                            notch_ba,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            fuzzy_chunk_size=128,
                        )
                    )
                values[np.asarray(missing_indices, dtype=np.int64)] = np.concatenate(computed, axis=0)
            values = values.reshape(len(rows), 18, 24).astype("float32", copy=False)
            if values.shape[0] != expected.shape[0]:
                raise ValueError("Exact cache row count does not match loaded metadata")
            return values
        x_train = cached(loaded_train_rows, x_train, "train")
        if x_val is not None:
            x_val = cached(loaded_val_rows, x_val, "validation")
        x_test = cached(loaded_test_rows, x_test, "test")
        if x_temporal_consistency is not None:
            x_temporal_consistency = cached(
                loaded_temporal_consistency_rows,
                x_temporal_consistency,
                "temporal_consistency",
            )
        feature_cache_manifest = {"enabled": True, "path": str(cache_npz), "shape": [18, 24], "interpretation": "exact reconstructed-subband HFD/FuzzyEn/RMS/STD"}

    fold_work_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x_train": x_train,
        "y_train": y_train,
        "subjects_train": subjects_train,
        "x_test": x_test,
        "y_test": y_test,
        "subjects_test": subjects_test,
    }
    if x_val is not None:
        arrays.update({"x_val": x_val, "y_val": y_val, "subjects_val": subjects_val})
    if x_temporal_consistency is not None:
        arrays.update({"x_temporal_consistency": x_temporal_consistency})

    emd_manifest: dict[str, object] = {"enabled": False}
    if emd_features:
        x_train = append_emd_features(
            x_train, components=emd_components, downsample=emd_downsample, split="train", workers=emd_workers, channel_indices=emd_channel_indices
        )
        if x_val is not None:
            x_val = append_emd_features(
                x_val, components=emd_components, downsample=emd_downsample, split="validation", workers=emd_workers, channel_indices=emd_channel_indices
            )
        x_test = append_emd_features(
            x_test, components=emd_components, downsample=emd_downsample, split="test", workers=emd_workers, channel_indices=emd_channel_indices
        )
        if x_temporal_consistency is not None:
            x_temporal_consistency = append_emd_features(
                x_temporal_consistency,
                components=emd_components,
                downsample=emd_downsample,
                split="temporal_consistency",
                workers=emd_workers,
                channel_indices=emd_channel_indices,
            )
        arrays = {
            "x_train": x_train,
            "y_train": y_train,
            "subjects_train": subjects_train,
            "x_test": x_test,
            "y_test": y_test,
            "subjects_test": subjects_test,
        }
        if x_val is not None:
            arrays.update({"x_val": x_val, "y_val": y_val, "subjects_val": subjects_val})
        if x_temporal_consistency is not None:
            arrays.update({"x_temporal_consistency": x_temporal_consistency})
        emd_manifest = {
            "enabled": True,
            "components": emd_components,
            "downsample": emd_downsample,
            "workers": emd_workers,
            "channel_indices": list(emd_channel_indices),
            "construction": "independent per-window, per-channel EMD; first components appended after raw channels",
        }
    np.savez_compressed(dataset_path, **arrays)
    write_rows_csv(fold_work_dir / "train_windows.csv", loaded_train_rows)
    if loaded_val_rows:
        write_rows_csv(fold_work_dir / "val_windows.csv", loaded_val_rows)
    if loaded_temporal_consistency_rows:
        write_rows_csv(fold_work_dir / "temporal_consistency_windows.csv", loaded_temporal_consistency_rows)
    write_rows_csv(fold_work_dir / "test_windows.csv", loaded_test_rows)
    manifest = {
        "dataset": str(dataset_path),
        "source_fold_meta_dir": str(fold_meta_dir),
        "validation": validation_manifest,
        "multiscale_5s": multiscale_manifest,
        "temporal_consistency": temporal_consistency_manifest,
        "emd": emd_manifest,
        "dcsa_exact_feature_cache": feature_cache_manifest,
        "window_zscore": bool(normalize_window),
        "loaded_counts": {
            "train": {
                "windows": int(y_train.shape[0]),
                "positive": int((y_train == 1).sum()),
                "negative": int((y_train == 0).sum()),
            },
            "validation": {
                "windows": int(y_val.shape[0]) if y_val is not None else 0,
                "positive": int((y_val == 1).sum()) if y_val is not None else 0,
                "negative": int((y_val == 0).sum()) if y_val is not None else 0,
            },
            "test": {
                "windows": int(y_test.shape[0]),
                "positive": int((y_test == 1).sum()),
                "negative": int((y_test == 0).sum()),
            },
            "temporal_consistency": {
                "windows": int(x_temporal_consistency.shape[0]) if x_temporal_consistency is not None else 0,
                "positive": 0,
                "negative": int(x_temporal_consistency.shape[0]) if x_temporal_consistency is not None else 0,
            },
        },
        "skipped_files": skipped_train + skipped_val + skipped_test + skipped_temporal_consistency,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return dataset_path, manifest


def evaluate_event(
    test_rows: list[dict[str, str]],
    y_true,
    y_score,
    threshold: float,
    fallback_window_sec: float = 7.0,
) -> dict[str, object]:
    false_alarm_windows = 0
    interictal_windows = 0
    interictal_seconds = 0.0
    preictal_scores = []
    seizure_start = None
    first_alarm_start = None
    for row, label, score in zip(test_rows, y_true, y_score, strict=True):
        score = float(score)
        if int(label) == 0:
            interictal_windows += 1
            try:
                interictal_seconds += max(
                    0.0,
                    float(row["window_end_sec"]) - float(row["window_start_sec"]),
                )
            except (KeyError, TypeError, ValueError):
                interictal_seconds += float(fallback_window_sec)
            if score >= threshold:
                false_alarm_windows += 1
        elif int(label) == 1:
            preictal_scores.append(score)
            if first_alarm_start is None and score >= threshold:
                first_alarm_start = time_start(row)
            # The seizure itself starts SOP+SPH after preictal start only indirectly.
            # For each held-out fold, using the last preictal end plus SPH gives the
            # seizure onset under the labels generated in this repository.
            sph_sec = int(row["sph_sec"])
            candidate_start = time_end(row) + sph_sec
            seizure_start = max(seizure_start or candidate_start, candidate_start)

    detected = int(first_alarm_start is not None)
    warning_time_sec = (seizure_start - first_alarm_start) if detected and seizure_start is not None else None
    interictal_hours = interictal_seconds / 3600
    return {
        "event_detected": detected,
        "event_preictal_windows": len(preictal_scores),
        "event_first_alarm_start_sec": first_alarm_start if first_alarm_start is not None else "",
        "event_warning_time_sec": warning_time_sec if warning_time_sec is not None else "",
        "event_warning_time_min": round(warning_time_sec / 60, 3) if warning_time_sec is not None else "",
        "false_alarm_windows": false_alarm_windows,
        "interictal_windows": interictal_windows,
        "estimated_interictal_hours": interictal_hours,
        "false_alarm_rate_per_hour": false_alarm_windows / max(interictal_hours, 1e-12),
        "max_preictal_score": max(preictal_scores) if preictal_scores else float("nan"),
        "mean_preictal_score": sum(preictal_scores) / len(preictal_scores) if preictal_scores else float("nan"),
    }


def make_context_array(
    x,
    rows: list[dict[str, str]],
    context_windows: int,
    require_contiguous: bool = False,
):
    """Build causal context tensors while preserving the original row order.

    For each target window, the context contains previous available windows from
    the same subject/file plus the current window. Missing early context is
    padded by repeating the earliest available window in that file. When
    ``require_contiguous`` is enabled, history stops at a temporal gap instead
    of joining non-adjacent rows across a train/validation split.
    """

    import numpy as np

    if context_windows <= 1:
        return x
    if x.shape[0] != len(rows):
        raise ValueError(f"row/window count mismatch: x={x.shape[0]} rows={len(rows)}")

    grouped_indices: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (row.get("subject", ""), row["file_name"])
        grouped_indices.setdefault(key, []).append(index)

    context_indices = np.empty((x.shape[0], context_windows), dtype=np.int64)
    for indices in grouped_indices.values():
        sorted_indices = sorted(
            indices,
            key=lambda i: (
                int(float(rows[i]["window_start_sec"])),
                int(float(rows[i]["window_end_sec"])),
                i,
            ),
        )
        for position, original_index in enumerate(sorted_indices):
            history = [original_index]
            history_position = position
            while len(history) < context_windows and history_position > 0:
                previous_index = sorted_indices[history_position - 1]
                if require_contiguous:
                    previous_end = int(float(rows[previous_index]["window_end_sec"]))
                    current_start = int(float(rows[history[0]]["window_start_sec"]))
                    if previous_end != current_start:
                        break
                history.insert(0, previous_index)
                history_position -= 1
            if len(history) < context_windows:
                history = [history[0]] * (context_windows - len(history)) + history
            context_indices[original_index] = history

    return x[context_indices]


def build_multiscale_5s_label_rows(
    rows: list[dict[str, str]],
    required_context_windows: int = 3,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Right-align 5-second rows and relabel by a strict all-positive rule.

    The target row stays anchored at the current 5-second interval. A positive
    label is kept only when the current and previous two contiguous 5-second
    rows from the same file are all labeled preictal. This matches the
    5/10/15-second multi-input interpretation where the 15-second branch
    contains three aligned 5-second windows.
    """

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row.get("subject", ""), row["file_name"]), []).append(row)

    relabeled_rows: list[dict[str, str]] = []
    skipped_no_context = 0
    original_positive = 0
    relabeled_positive = 0
    for group_rows in grouped.values():
        sorted_rows = sorted(group_rows, key=lambda r: (int(r["window_start_sec"]), int(r["window_end_sec"])))
        by_start = {int(r["window_start_sec"]): r for r in sorted_rows}
        for row in sorted_rows:
            original_label = int(row["label"])
            original_positive += int(original_label == 1)
            start = int(row["window_start_sec"])
            end = int(row["window_end_sec"])
            window_sec = end - start
            if window_sec <= 0:
                skipped_no_context += 1
                continue
            context_starts = [start - window_sec * offset for offset in range(required_context_windows - 1, -1, -1)]
            new_row = dict(row)
            context_rows = [by_start.get(context_start) for context_start in context_starts]
            context_labels = [
                int(context_row["label"]) if context_row is not None else 0
                for context_row in context_rows
            ]
            new_label = int(all(label == 1 for label in context_labels))
            if original_label == 1 and any(context_row is None for context_row in context_rows):
                skipped_no_context += 1
                continue
            new_row["label"] = str(new_label)
            new_row["multiscale_context_start_sec"] = str(context_starts[0])
            new_row["multiscale_context_end_sec"] = str(end)
            new_row["multiscale_context_labels"] = ",".join(str(label) for label in context_labels)
            relabeled_rows.append(new_row)
            relabeled_positive += int(new_label == 1)

    return relabeled_rows, {
        "enabled": True,
        "required_context_windows": required_context_windows,
        "input_scales_sec": [5, 10, 15],
        "label_rule": "positive iff current and previous two contiguous 5-second labels are all positive",
        "input_rows": len(rows),
        "output_rows": len(relabeled_rows),
        "skipped_no_context": skipped_no_context,
        "original_positive": original_positive,
        "relabeled_positive": relabeled_positive,
    }


def load_rows_to_multiscale_arrays(
    rows: list[dict[str, str]],
    data_root: Path,
    channels: list[str],
    split: str,
    scales_sec: tuple[int, ...] = (5, 10, 15),
):
    import numpy as np

    max_sec = max(scales_sec)
    x_parts = []
    y_parts = []
    subject_parts = []
    loaded_rows = []
    skipped_files = []
    rows_by_file: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        rows_by_file.setdefault((row["subject"], row["file_name"]), []).append(row)

    loaded_count = 0
    for (subject, file_name), file_rows in sorted(rows_by_file.items()):
        edf_path = data_root / subject / file_name
        opened = open_edf(edf_path)
        file_skipped = 0
        first_skipped_reason = ""
        try:
            for row in file_rows:
                try:
                    end_sec = int(row["window_end_sec"])
                    scale_arrays = []
                    for scale_sec in scales_sec:
                        start_sec = end_sec - int(scale_sec)
                        if start_sec < 0:
                            raise ValueError(f"not enough history for {scale_sec}s scale at {subject}/{file_name}:{end_sec}")
                        x_scale = load_edf_window(opened, edf_path, start_sec, end_sec, channels)
                        x_scale = zscore(x_scale)
                        pad_samples = int(round((max_sec - int(scale_sec)) * (x_scale.shape[0] / int(scale_sec))))
                        if pad_samples > 0:
                            x_scale = np.pad(x_scale, ((pad_samples, 0), (0, 0)), mode="constant")
                        scale_arrays.append(x_scale)
                    x_parts.append(np.stack(scale_arrays).astype("float32"))
                except ValueError as exc:
                    file_skipped += 1
                    if not first_skipped_reason:
                        first_skipped_reason = str(exc)
                    continue
                y_parts.append(int(row["label"]))
                subject_parts.append(row["subject"])
                loaded_rows.append(row)
                loaded_count += 1
                if loaded_count % 500 == 0:
                    print(f"{split}: loaded {loaded_count}/{len(rows)} multiscale windows")
        finally:
            close_edf(opened)
        if file_skipped:
            skipped_files.append(
                {
                    "split": split,
                    "subject": subject,
                    "file_name": file_name,
                    "windows": file_skipped,
                    "reason": first_skipped_reason,
                }
            )
            print(
                f"{split}: skipped {file_skipped}/{len(file_rows)} windows in "
                f"{subject}/{file_name}: {first_skipped_reason}"
            )

    if not x_parts:
        raise ValueError(f"No {split} multiscale windows could be loaded.")
    return (
        np.stack(x_parts).astype("float32"),
        np.asarray(y_parts, dtype="int64"),
        np.asarray(subject_parts),
        loaded_rows,
        skipped_files,
    )


def _emd_window_worker(payload):
    import numpy as np
    from PyEMD import EMD

    window, components, downsample, channel_indices = payload
    time_steps, channels = window.shape
    downsample = max(int(downsample), 1)
    source_grid = np.arange(time_steps, dtype=np.float32)
    reduced_grid = np.arange(0, time_steps, downsample, dtype=np.float32)
    if reduced_grid[-1] != time_steps - 1:
        reduced_grid = np.append(reduced_grid, time_steps - 1)
    output = np.zeros((time_steps, len(channel_indices) * components), dtype=np.float32)
    for output_channel_index, channel_index in enumerate(channel_indices):
        signal = np.asarray(window[:, channel_index], dtype=np.float64)
        reduced_signal = np.interp(reduced_grid, source_grid, signal)
        try:
            imfs = EMD(extrema_detection="parabol").emd(
                reduced_signal,
                max_imf=max(components - 1, 0),
            )
        except Exception:
            imfs = np.empty((0, reduced_signal.size), dtype=np.float64)
        if imfs.ndim == 1:
            imfs = imfs[None, :]
        for component_index in range(min(components, imfs.shape[0])):
            output[:, output_channel_index * components + component_index] = np.interp(
                source_grid, reduced_grid, imfs[component_index]
            ).astype(np.float32)
    return output


def append_emd_features(
    x,
    components: int = 3,
    downsample: int = 2,
    split: str = "split",
    workers: int = 1,
    channel_indices: tuple[int, ...] | None = None,
):
    """Append per-window EMD components without crossing window boundaries."""

    import multiprocessing as mp
    import numpy as np

    if x.ndim != 3:
        raise ValueError(
            "EMD augmentation currently expects [windows, time, channels], "
            f"got shape={tuple(x.shape)}"
        )
    if components <= 0:
        raise ValueError(f"EMD components must be positive, got {components}")
    downsample = max(int(downsample), 1)
    windows, time_steps, channels = x.shape
    if channel_indices is None:
        channel_indices = tuple(range(channels))
    channel_indices = tuple(int(index) for index in channel_indices)
    if not channel_indices or any(index < 0 or index >= channels for index in channel_indices):
        raise ValueError(f"Invalid EMD channel indices {channel_indices} for {channels} channels")
    emd_output = np.zeros(
        (windows, time_steps, len(channel_indices) * components), dtype=np.float32
    )
    payloads = ((x[index], components, downsample, channel_indices) for index in range(windows))
    workers = max(int(workers), 1)
    if workers == 1 or windows < 2:
        outputs = map(_emd_window_worker, payloads)
        for window_index, output in enumerate(outputs):
            emd_output[window_index] = output
            if (window_index + 1) % 500 == 0 or window_index + 1 == windows:
                print(f"{split}: EMD features {window_index + 1}/{windows}")
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=workers) as pool:
            outputs = pool.imap(_emd_window_worker, payloads, chunksize=4)
            for window_index, output in enumerate(outputs):
                emd_output[window_index] = output
                if (window_index + 1) % 500 == 0 or window_index + 1 == windows:
                    print(f"{split}: EMD features {window_index + 1}/{windows}")
    return np.concatenate([x.astype(np.float32, copy=False), emd_output], axis=2)


def supervised_contrastive_loss(features, labels, temperature: float):
    import torch
    import torch.nn.functional as F

    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    features = F.normalize(features, dim=1)
    labels = labels.reshape(-1, 1)
    positive_mask = torch.eq(labels, labels.T).to(features.device)
    self_mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive_mask = positive_mask & ~self_mask

    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0
    if not torch.any(valid_anchor):
        return features.sum() * 0.0

    logits = torch.matmul(features, features.T) / max(temperature, 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    mean_log_prob_pos = positive_log_prob.sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob_pos[valid_anchor].mean()


def extract_model_features(model, xb):
    module = model.module if hasattr(model, "module") else model
    if hasattr(module, "extract_features"):
        return module.extract_features(xb)
    if hasattr(module, "encoder") and hasattr(module.encoder, "extract_features"):
        if xb.ndim == 4:
            batch, context, time_steps, channels = xb.shape
            flat = xb.reshape(batch * context, time_steps, channels)
            features = module.encoder.extract_features(flat).reshape(batch, context, -1)
            return features[:, -1, :]
        return module.encoder.extract_features(xb)
    return None


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def synthesize_preictal_mixup_training_samples(
    x_train,
    y_train,
    train_rows: list[dict[str, str]],
    ratio: float,
    alpha: float,
    noise_std: float,
    seed: int,
    fold_key: str,
):
    """Append training-only synthetic preictal samples by mixing positive windows."""

    import numpy as np

    if ratio <= 0:
        return x_train, y_train, train_rows, {"enabled": False}

    positive_indices = np.flatnonzero(y_train.astype("int64") == 1)
    if positive_indices.size < 2:
        return x_train, y_train, train_rows, {
            "enabled": True,
            "created": False,
            "reason": "need at least two preictal training windows",
            "positive_windows": int(positive_indices.size),
        }

    synthetic_count = int(round(float(positive_indices.size) * ratio))
    if synthetic_count <= 0:
        return x_train, y_train, train_rows, {
            "enabled": True,
            "created": False,
            "reason": "synthetic count rounded to zero",
            "positive_windows": int(positive_indices.size),
            "ratio": ratio,
        }

    rng = np.random.default_rng(stable_seed(seed, f"preictal_mixup:{fold_key}"))
    first = rng.choice(positive_indices, size=synthetic_count, replace=True)
    second = rng.choice(positive_indices, size=synthetic_count, replace=True)
    if positive_indices.size > 1:
        same = first == second
        while np.any(same):
            second[same] = rng.choice(positive_indices, size=int(np.sum(same)), replace=True)
            same = first == second

    alpha = max(float(alpha), 1e-6)
    lam_shape = (synthetic_count,) + (1,) * (x_train.ndim - 1)
    lam = rng.beta(alpha, alpha, size=lam_shape).astype(x_train.dtype, copy=False)
    synthetic_x = lam * x_train[first] + (1.0 - lam) * x_train[second]
    if noise_std > 0:
        synthetic_x = synthetic_x + rng.normal(
            loc=0.0,
            scale=float(noise_std),
            size=synthetic_x.shape,
        ).astype(x_train.dtype, copy=False)
    synthetic_y = np.ones(synthetic_count, dtype=y_train.dtype)

    synthetic_rows: list[dict[str, str]] = []
    for synth_index, source_index in enumerate(first):
        source_row = dict(train_rows[int(source_index)]) if train_rows else {}
        if source_row:
            source_row["state"] = "preictal_synthetic_mixup"
            source_row["file_name"] = f"synthetic_mixup/{fold_key}"
            source_row["window_start_sec"] = str(synth_index)
            source_row["window_end_sec"] = str(synth_index + 1)
            source_row["matched_seizure_index_in_file"] = (
                source_row.get("matched_seizure_index_in_file", "") or "synthetic"
            )
            source_row["label"] = "1"
        synthetic_rows.append(source_row)

    return (
        np.concatenate([x_train, synthetic_x], axis=0),
        np.concatenate([y_train, synthetic_y], axis=0),
        train_rows + synthetic_rows,
        {
            "enabled": True,
            "created": True,
            "ratio": ratio,
            "alpha": alpha,
            "noise_std": noise_std,
            "original_train_windows": int(y_train.shape[0]),
            "original_preictal_windows": int(positive_indices.size),
            "synthetic_preictal_windows": int(synthetic_count),
            "augmented_train_windows": int(y_train.shape[0] + synthetic_count),
        },
    )


def event_balanced_sample_weights(rows: list[dict[str, str]], labels):
    """Weight samples so preictal events and interictal files contribute evenly."""

    import numpy as np

    if len(rows) != len(labels):
        raise ValueError(f"row/label count mismatch: rows={len(rows)} labels={len(labels)}")

    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, (row, label) in enumerate(zip(rows, labels, strict=True)):
        label = int(label)
        if label == 1:
            event_id = row.get("matched_seizure_index_in_file", "") or "unknown"
            group_key = ("preictal", row["file_name"], event_id)
        else:
            group_key = ("interictal", row["file_name"], "interictal")
        groups.setdefault(group_key, []).append(index)

    positive_groups = [key for key in groups if key[0] == "preictal"]
    negative_groups = [key for key in groups if key[0] == "interictal"]
    weights = np.zeros(len(labels), dtype="float32")

    for class_groups in (positive_groups, negative_groups):
        if not class_groups:
            continue
        class_mass = 0.5
        group_mass = class_mass / len(class_groups)
        for key in class_groups:
            indices = groups[key]
            sample_weight = group_mass / max(len(indices), 1)
            weights[indices] = sample_weight

    if not np.any(weights):
        weights[:] = 1.0 / max(len(weights), 1)
    else:
        weights = weights / weights.sum()
    return weights


def temporal_pair_indices(
    rows: list[dict[str, str]],
    labels,
    max_gap_sec: int = 0,
    label_mode: str = "interictal",
):
    """Return row-index pairs for adjacent windows used by stability loss."""

    import numpy as np

    if len(rows) != len(labels):
        raise ValueError(f"row/label count mismatch: rows={len(rows)} labels={len(labels)}")
    indexed_rows = []
    for index, row in enumerate(rows):
        try:
            indexed_rows.append(
                (
                    row.get("subject", ""),
                    row["file_name"],
                    int(row["window_start_sec"]),
                    int(row["window_end_sec"]),
                    int(labels[index]),
                    index,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    indexed_rows.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[5]))

    pairs = []
    previous = None
    for current in indexed_rows:
        if previous is not None:
            same_stream = current[0] == previous[0] and current[1] == previous[1]
            contiguous = current[2] <= previous[3] + max(0, max_gap_sec)
            if same_stream and contiguous:
                prev_label = previous[4]
                curr_label = current[4]
                include = False
                if label_mode == "all":
                    include = True
                elif label_mode == "same":
                    include = prev_label == curr_label
                elif label_mode == "interictal":
                    include = prev_label == 0 and curr_label == 0
                elif label_mode == "preictal":
                    include = prev_label == 1 and curr_label == 1
                else:
                    raise ValueError(f"Unsupported temporal consistency label mode: {label_mode}")
                if include:
                    pairs.append((previous[5], current[5]))
        previous = current
    return np.asarray(pairs, dtype=np.int64)


def train_one_fold(dataset_path: Path, out_dir: Path, args) -> dict[str, object]:
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loaded = np.load(dataset_path, allow_pickle=True)
    x_train = loaded["x_train"]
    y_train = loaded["y_train"]
    x_val = loaded["x_val"] if "x_val" in loaded.files else None
    y_val = loaded["y_val"] if "y_val" in loaded.files else None
    x_test = loaded["x_test"]
    y_test = loaded["y_test"]
    x_temporal_consistency = (
        loaded["x_temporal_consistency"] if "x_temporal_consistency" in loaded.files else None
    )

    if args.context_windows > 1:
        x_train = make_context_array(
            x_train,
            read_rows(out_dir / "train_windows.csv"),
            args.context_windows,
            require_contiguous=args.context_contiguous,
        )
        if x_val is not None:
            x_val = make_context_array(
                x_val,
                read_rows(out_dir / "val_windows.csv"),
                args.context_windows,
                require_contiguous=args.context_contiguous,
            )
        x_test = make_context_array(
            x_test,
            read_rows(out_dir / "test_windows.csv"),
            args.context_windows,
            require_contiguous=args.context_contiguous,
        )
        if x_temporal_consistency is not None:
            x_temporal_consistency = make_context_array(
                x_temporal_consistency,
                read_rows(out_dir / "temporal_consistency_windows.csv"),
                args.context_windows,
                require_contiguous=args.context_contiguous,
            )

    train_rows_for_sampling = read_rows(out_dir / "train_windows.csv")
    x_train, y_train, train_rows_for_sampling, synthetic_preictal_mixup = (
        synthesize_preictal_mixup_training_samples(
            x_train=x_train,
            y_train=y_train,
            train_rows=train_rows_for_sampling,
            ratio=args.synthetic_preictal_mixup_ratio,
            alpha=args.synthetic_preictal_mixup_alpha,
            noise_std=args.synthetic_preictal_mixup_noise_std,
            seed=args.seed,
            fold_key=out_dir.name,
        )
    )
    if synthetic_preictal_mixup.get("enabled"):
        print(f"synthetic_preictal_mixup={synthetic_preictal_mixup}")

    temporal_pairs = None
    temporal_pair_x = x_train
    temporal_pair_manifest = {"enabled": False}
    if args.temporal_consistency_weight > 0:
        if x_temporal_consistency is not None and (out_dir / "temporal_consistency_windows.csv").exists():
            temporal_rows_for_pairs = read_rows(out_dir / "temporal_consistency_windows.csv")
            temporal_labels_for_pairs = np.zeros(len(temporal_rows_for_pairs), dtype="int64")
            temporal_pair_x = x_temporal_consistency
            temporal_source = "interictal_pool"
        else:
            temporal_rows_for_pairs = train_rows_for_sampling
            temporal_labels_for_pairs = y_train
            temporal_source = "train_windows"
        temporal_pairs = temporal_pair_indices(
            temporal_rows_for_pairs,
            temporal_labels_for_pairs,
            max_gap_sec=args.temporal_consistency_gap_sec,
            label_mode=args.temporal_consistency_label_mode,
        )
        temporal_pair_manifest = {
            "enabled": True,
            "source": temporal_source,
            "source_windows": int(temporal_pair_x.shape[0]),
            "pairs": int(temporal_pairs.shape[0]),
            "label_mode": args.temporal_consistency_label_mode,
            "gap_sec": args.temporal_consistency_gap_sec,
            "weight": args.temporal_consistency_weight,
            "max_pairs_per_batch": args.temporal_consistency_max_pairs_per_batch,
        }
        print(f"temporal_consistency={temporal_pair_manifest}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    val_loader = None
    if x_val is not None and y_val is not None:
        val_ds = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
    train_sampler = None
    train_shuffle = True
    if args.train_sampler == "event_balanced":
        train_rows = train_rows_for_sampling
        sample_weights = event_balanced_sample_weights(train_rows, y_train)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False
        print(
            "event_balanced_sampler="
            f"groups weighted from {len(train_rows)} rows; "
            f"positive_windows={int((y_train == 1).sum())} "
            f"negative_windows={int((y_train == 0).sum())}"
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        # The published DC-SA classifier uses BatchNorm in its 128-unit head;
        # a singleton tail batch is invalid for BatchNorm training.
        drop_last=(
            args.model in {"dc_sa_ebilstm_faithful", "faithful_dc_sa_ebilstm", "dcsa_ebilstm_faithful"}
            and len(train_ds) % args.batch_size == 1
        ),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    validation_rows = read_rows(out_dir / "val_windows.csv") if x_val is not None else []
    test_rows = read_rows(out_dir / "test_windows.csv")

    progressive_blocks = tuple(
        block.strip() for block in args.progressive_blocks.split(",") if block.strip()
    )
    # Raw windows are [N,time,channels], while exact DC-SA descriptors are
    # [N,channels,24].  The model's channel count must follow the second axis
    # for descriptor caches and the last axis for waveform inputs.
    model_channels = x_train.shape[1] if (
        args.model in {"dc_sa_ebilstm_faithful", "faithful_dc_sa_ebilstm", "dcsa_ebilstm_faithful"}
        and x_train.ndim == 3 and x_train.shape[-1] == 24
    ) else x_train.shape[-1]
    model = build_model(
        args.model,
        channels=model_channels,
        progressive_blocks=progressive_blocks,
        internal_stage=args.progressive_internal_stage,
    )
    if args.pretrained_encoder:
        checkpoint = torch.load(args.pretrained_encoder, map_location="cpu")
        state_dict = checkpoint.get("encoder_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"loaded pretrained encoder={args.pretrained_encoder} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    if args.init_from_result_root:
        previous_checkpoint = (
            args.init_from_result_root
            / out_dir.parent.name
            / out_dir.name
            / "best_model.pt"
        )
        if not previous_checkpoint.exists():
            raise FileNotFoundError(
                "Progressive initialization checkpoint not found: "
                f"{previous_checkpoint}"
            )
        checkpoint = torch.load(previous_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Some CHB-MIT files expose one fewer common channel after EDF channel
        # filtering.  Retain compatible pretrained tensors and skip only
        # shape-mismatched input stems instead of aborting the fold.
        model_state = model.state_dict()
        state_dict = {
            key: value for key, value in state_dict.items()
            if key in model_state and getattr(value, "shape", None) == model_state[key].shape
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"loaded previous growth stage={previous_checkpoint} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    model = model.to(device)
    # DC-SA-EBiLSTM uses handcrafted DWT descriptors.  Fit their
    # standardization from the current training fold only, then keep the
    # transform inside the checkpoint so test windows cannot influence it.
    if hasattr(model, "fit_feature_standardizer"):
        model.fit_feature_standardizer(
            torch.from_numpy(x_train).float().to(device),
            batch_size=max(int(args.batch_size), 1),
        )
        print("dc_sa_ebilstm_feature_standardizer=fitted_on_training_fold")
    if args.freeze_encoder:
        # Keep the transferred representation and its batch-normalization
        # statistics fixed; only the task head adapts to the patient.
        for name, parameter in model.named_parameters():
            # FeatureResidualTriBandDSF exposes a compact statistics branch;
            # keep it trainable during frozen-DSF adaptation.
            parameter.requires_grad = name.startswith("classifier.") or name.startswith("band_stat_")
        print("parameter_efficient_adaptation=freeze_encoder_train_classifier")
    feature_adapter = None
    if args.residual_adapter:
        with torch.no_grad():
            feature_probe = extract_model_features(
                model,
                torch.from_numpy(x_train[:1]).float().to(device),
            )
        if feature_probe is None or feature_probe.ndim != 2:
            raise ValueError("Residual adapter requires a 2-D model feature representation")
        feature_dim = int(feature_probe.shape[-1])
        bottleneck = min(max(int(args.residual_adapter_bottleneck), 4), feature_dim)
        feature_adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, feature_dim),
        ).to(device)
        # Start as an identity residual branch so adaptation cannot erase the
        # transferred representation at initialization.
        nn.init.zeros_(feature_adapter[-1].weight)
        nn.init.zeros_(feature_adapter[-1].bias)
        if args.freeze_encoder:
            for parameter in feature_adapter.parameters():
                parameter.requires_grad = True
        print(
            "residual_adapter="
            f"enabled feature_dim={feature_dim} bottleneck={bottleneck} "
            f"freeze_encoder={args.freeze_encoder}"
        )
    freq_aux_frequencies = parse_float_list(args.freq_aux_frequencies)
    freq_aux_head = None
    if args.freq_aux_weight > 0:
        if not freq_aux_frequencies:
            raise ValueError("--freq-aux-weight requires at least one --freq-aux-frequencies value")
        with torch.no_grad():
            feature_probe = extract_model_features(
                model,
                torch.from_numpy(x_train[:1]).float().to(device),
            )
        if feature_probe is None:
            raise ValueError("Frequency auxiliary task requires a model exposing extract_features")
        freq_aux_head = nn.Linear(int(feature_probe.shape[-1]), len(freq_aux_frequencies)).to(device)
        print(
            "freq_aux="
            f"weight={args.freq_aux_weight} "
            f"frequencies={freq_aux_frequencies} "
            f"amplitude={args.freq_aux_amplitude}"
        )

    optimizer_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if feature_adapter is not None:
        optimizer_params += list(feature_adapter.parameters())
    if freq_aux_head is not None:
        optimizer_params += list(freq_aux_head.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
        )
        if args.lr_step_size > 0
        else None
    )
    class_weights = None
    if args.loss in {"weighted_ce", "focal"}:
        counts = np.bincount(y_train, minlength=2).astype("float32")
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"class_weights={class_weights.detach().cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history = []
    best = {"selection_metric": -1.0, "epoch": 0}
    best_selection_key = None
    best_val_scores = None
    epochs_without_improvement = 0

    def augment_batch(xb):
        if args.channel_dropout_prob > 0:
            if xb.ndim == 3:
                keep_shape = (xb.shape[0], 1, xb.shape[2])
            elif xb.ndim == 4:
                keep_shape = (xb.shape[0], 1, 1, xb.shape[3])
            else:
                raise ValueError(f"Unsupported batch shape for channel dropout: {tuple(xb.shape)}")
            keep = torch.rand(keep_shape, device=xb.device) >= args.channel_dropout_prob
            keep = keep.to(dtype=xb.dtype)
            xb = xb * keep / max(1.0 - args.channel_dropout_prob, 1e-6)
        if args.channel_noise_std > 0:
            xb = xb + torch.randn_like(xb) * args.channel_noise_std
        return xb

    def synthesize_frequency_batch(xb):
        if freq_aux_head is None:
            return None, None
        if xb.ndim == 3:
            batch, time_steps, _ = xb.shape
            target = torch.randint(len(freq_aux_frequencies), (batch,), device=xb.device)
            time = torch.arange(time_steps, device=xb.device, dtype=xb.dtype) / (
                time_steps / max(args.freq_aux_window_sec, 1e-6)
            )
            frequencies = torch.tensor(freq_aux_frequencies, device=xb.device, dtype=xb.dtype)[target]
            sine = torch.sin(2.0 * math.pi * frequencies[:, None] * time[None, :])
            return xb + args.freq_aux_amplitude * sine[:, :, None], target
        if xb.ndim == 4:
            batch, context, time_steps, _ = xb.shape
            target = torch.randint(len(freq_aux_frequencies), (batch,), device=xb.device)
            time = torch.arange(time_steps, device=xb.device, dtype=xb.dtype) / (
                time_steps / max(args.freq_aux_window_sec, 1e-6)
            )
            frequencies = torch.tensor(freq_aux_frequencies, device=xb.device, dtype=xb.dtype)[target]
            sine = torch.sin(2.0 * math.pi * frequencies[:, None, None] * time[None, None, :])
            return xb + args.freq_aux_amplitude * sine[:, :, :, None], target
        raise ValueError(f"Unsupported batch shape for frequency auxiliary task: {tuple(xb.shape)}")

    def classifier_from_features(features):
        module = model.module if hasattr(model, "module") else model
        if not hasattr(module, "classifier"):
            raise ValueError("Feature-space mixup requires a model with a classifier head")
        if feature_adapter is not None:
            features = features + feature_adapter(features)
        return module.classifier(features)

    def model_logits(xb):
        if feature_adapter is None:
            return model(xb)
        return classifier_from_features(extract_model_features(model, xb))

    def feature_space_preictal_mixup_loss(features, yb):
        if args.feature_preictal_mixup_weight <= 0:
            return torch.tensor(0.0, device=features.device)
        positive_indices = torch.nonzero(yb == 1, as_tuple=False).flatten()
        if positive_indices.numel() < 2:
            return torch.tensor(0.0, device=features.device)

        sample_count = max(1, int(round(float(positive_indices.numel()) * args.feature_preictal_mixup_ratio)))
        first = positive_indices[torch.randint(positive_indices.numel(), (sample_count,), device=features.device)]
        second = positive_indices[torch.randint(positive_indices.numel(), (sample_count,), device=features.device)]
        if positive_indices.numel() > 1:
            same = first == second
            while torch.any(same):
                second[same] = positive_indices[
                    torch.randint(positive_indices.numel(), (int(same.sum().item()),), device=features.device)
                ]
                same = first == second

        alpha = max(float(args.feature_preictal_mixup_alpha), 1e-6)
        beta = torch.distributions.Beta(alpha, alpha)
        lam_shape = (sample_count,) + (1,) * (features.ndim - 1)
        lam = beta.sample(lam_shape).to(device=features.device, dtype=features.dtype)
        mixed_features = lam * features[first] + (1.0 - lam) * features[second]
        mixed_logits = classifier_from_features(mixed_features)
        mixed_target = torch.ones(sample_count, dtype=torch.long, device=features.device)
        return F.cross_entropy(mixed_logits, mixed_target)

    def hard_negative_contrastive_loss(features, yb):
        if args.hard_negative_weight <= 0:
            return torch.tensor(0.0, device=features.device)
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        positive = F.normalize(features[yb == 1], dim=1)
        negative = F.normalize(features[yb == 0], dim=1)
        if positive.shape[0] == 0 or negative.shape[0] == 0:
            return torch.tensor(0.0, device=features.device)

        similarity = torch.matmul(positive, negative.T)
        topk = min(max(1, args.hard_negative_topk), similarity.shape[1])
        hard_similarity = similarity.topk(k=topk, dim=1).values
        scaled_margin_violation = (
            hard_similarity - args.hard_negative_margin
        ) / max(args.hard_negative_temperature, 1e-6)
        return F.softplus(scaled_margin_violation).mean()

    def compute_loss(logits, yb, features=None):
        if args.loss == "focal":
            ce = F.cross_entropy(logits, yb, weight=class_weights, reduction="none")
            pt = torch.exp(-ce)
            cls_loss = ((1.0 - pt) ** args.focal_gamma * ce).mean()
        else:
            cls_loss = criterion(logits, yb)
        if args.supcon_weight <= 0:
            return cls_loss, cls_loss.detach(), torch.tensor(0.0, device=logits.device)
        if features is None:
            return cls_loss, cls_loss.detach(), torch.tensor(0.0, device=logits.device)
        con_loss = supervised_contrastive_loss(features, yb, args.supcon_temperature)
        return cls_loss + args.supcon_weight * con_loss, cls_loss.detach(), con_loss.detach()

    def temporal_consistency_loss():
        if (
            args.temporal_consistency_weight <= 0
            or temporal_pairs is None
            or temporal_pairs.shape[0] == 0
        ):
            return torch.tensor(0.0, device=device)
        pair_count = min(
            int(args.temporal_consistency_max_pairs_per_batch),
            int(temporal_pairs.shape[0]),
        )
        pair_count = max(pair_count, 1)
        selected = np.random.randint(0, temporal_pairs.shape[0], size=pair_count)
        pair_indices = temporal_pairs[selected]
        first = torch.from_numpy(temporal_pair_x[pair_indices[:, 0]]).float().to(device)
        second = torch.from_numpy(temporal_pair_x[pair_indices[:, 1]]).float().to(device)
        first_prob = torch.softmax(model_logits(first), dim=1)
        second_prob = torch.softmax(model_logits(second), dim=1)
        return F.mse_loss(first_prob, second_prob)

    def evaluate(loader):
        model.eval()
        scores = []
        labels = []
        with torch.no_grad():
            for xb, yb in loader:
                logits = model_logits(xb.to(device))
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                scores.append(prob)
                labels.append(yb.numpy())
        y_score = np.concatenate(scores)
        y_true = np.concatenate(labels)
        return metrics_from_outputs(y_true, y_score, threshold=args.threshold), y_score, y_true

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_encoder:
            for name, module in model.named_children():
                if name != "classifier":
                    module.eval()
            model.classifier.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_supcon_loss = 0.0
        total_freq_aux_loss = 0.0
        total_feature_mixup_loss = 0.0
        total_hard_negative_loss = 0.0
        total_temporal_consistency_loss = 0.0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = augment_batch(xb)
            optimizer.zero_grad()
            xb_current = xb
            need_features = (
                args.supcon_weight > 0
                or args.feature_preictal_mixup_weight > 0
                or args.hard_negative_weight > 0
                or args.residual_adapter
            )
            if need_features:
                features = extract_model_features(model, xb)
                if features is None:
                    raise ValueError("Selected feature-space loss requires a model exposing extract_features")
                logits = classifier_from_features(features)
            else:
                logits = model_logits(xb)
                features = None
            loss, cls_loss, supcon_loss = compute_loss(logits, yb, features)
            feature_mixup_loss = torch.tensor(0.0, device=device)
            if args.feature_preictal_mixup_weight > 0:
                feature_mixup_loss = feature_space_preictal_mixup_loss(features, yb)
                loss = loss + args.feature_preictal_mixup_weight * feature_mixup_loss
            hard_negative_loss = torch.tensor(0.0, device=device)
            if args.hard_negative_weight > 0:
                hard_negative_loss = hard_negative_contrastive_loss(features, yb)
                loss = loss + args.hard_negative_weight * hard_negative_loss
            freq_aux_loss = torch.tensor(0.0, device=device)
            if freq_aux_head is not None:
                xb_freq, freq_target = synthesize_frequency_batch(xb)
                freq_features = extract_model_features(model, xb_freq)
                freq_logits = freq_aux_head(freq_features)
                freq_aux_loss = F.cross_entropy(freq_logits, freq_target)
                loss = loss + args.freq_aux_weight * freq_aux_loss
            temporal_loss = torch.tensor(0.0, device=device)
            if args.temporal_consistency_weight > 0:
                temporal_loss = temporal_consistency_loss()
                loss = loss + args.temporal_consistency_weight * temporal_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_params, 10.0)
            optimizer.step()
            total_loss += loss.item() * yb.numel()
            total_cls_loss += float(cls_loss.item()) * yb.numel()
            total_supcon_loss += float(supcon_loss.item()) * yb.numel()
            total_freq_aux_loss += float(freq_aux_loss.item()) * yb.numel()
            total_feature_mixup_loss += float(feature_mixup_loss.item()) * yb.numel()
            total_hard_negative_loss += float(hard_negative_loss.item()) * yb.numel()
            total_temporal_consistency_loss += float(temporal_loss.item()) * yb.numel()
            total += yb.numel()
        val_metrics = None
        val_score = None
        val_true = None
        validation_alarm_metrics = None
        if val_loader is not None:
            val_metrics, val_score, val_true = evaluate(val_loader)
            if args.checkpoint_selection == "validation_event":
                validation_alarm_metrics = evaluate_alarm_fold(
                    fold_dir=out_dir,
                    pred_rows=scored_rows(validation_rows, val_score),
                    window_rows=validation_rows,
                    threshold=args.validation_alarm_threshold,
                    merge_gap_sec=args.validation_alarm_merge_gap_sec,
                    refractory_sec=int(round(args.validation_alarm_refractory_min * 60.0)),
                    window_sec=int(round(args.window_sec)),
                    min_alarm_windows=args.validation_alarm_min_windows,
                )

        metrics = dict(val_metrics) if val_metrics is not None else {}
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(total, 1)
        metrics["train_cls_loss"] = total_cls_loss / max(total, 1)
        metrics["train_supcon_loss"] = total_supcon_loss / max(total, 1)
        metrics["train_freq_aux_loss"] = total_freq_aux_loss / max(total, 1)
        metrics["train_feature_mixup_loss"] = total_feature_mixup_loss / max(total, 1)
        metrics["train_hard_negative_loss"] = total_hard_negative_loss / max(total, 1)
        metrics["train_temporal_consistency_loss"] = total_temporal_consistency_loss / max(total, 1)
        if val_metrics is not None:
            for key, value in val_metrics.items():
                metrics[f"val_{key}"] = value
        if validation_alarm_metrics is not None:
            for key in (
                "event_detected",
                "event_warning_time_min",
                "false_alarm_rate_per_hour",
                "false_alarm_events",
            ):
                metrics[f"validation_alarm_{key}"] = validation_alarm_metrics.get(key, "")
        history.append(metrics)

        if args.checkpoint_selection == "validation_event" and validation_alarm_metrics is not None:
            selection_key = validation_alarm_selection_key(validation_alarm_metrics, val_metrics or {})
            selection_value = selection_key[0]
        else:
            selection_metrics = val_metrics
            if selection_metrics is None:
                selection_key = (0.0, 0.0, -float("inf"), -float("inf"))
            else:
                selection_value = selection_metrics["auc"] if not math.isnan(selection_metrics["auc"]) else -1.0
                selection_key = (selection_value, 0.0, -float("inf"), -float("inf"))
        if best_selection_key is None or selection_key > best_selection_key:
            best = dict(metrics)
            best["selection_metric"] = list(selection_key)
            best["selection_source"] = "validation_event" if args.checkpoint_selection == "validation_event" else "validation_auc"
            best_selection_key = selection_key
            best_val_scores = (val_score, val_true) if val_score is not None else None
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            if feature_adapter is not None:
                torch.save(feature_adapter.state_dict(), out_dir / "best_feature_adapter.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        print(
            f"epoch={epoch:03d} loss={metrics['train_loss']:.4f} "
            + (
                f"val_auc={metrics['auc']:.4f} val_f1={metrics['f1']:.4f}"
                if val_metrics is not None
                else "no_validation"
            )
            + (
                f" event={metrics['validation_alarm_event_detected']:.4f}"
                f" far={metrics['validation_alarm_false_alarm_rate_per_hour']:.4f}"
                if validation_alarm_metrics is not None
                else ""
            )
        )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"early_stopping epoch={epoch} "
                f"patience={args.early_stopping_patience} best_epoch={best.get('epoch', 0)}"
            )
            break

    if best_selection_key is None:
        model.eval()
        best = {"epoch": args.epochs, "selection_metric": [], "selection_source": "none"}
        best_val_scores = None

    checkpoint_path = out_dir / "best_model.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics, y_score, y_true = evaluate(test_loader)
    best.update(test_metrics)
    if best_val_scores is not None:
        best["val_score_available"] = True
    return {
        "best": best,
        "history": history,
        "y_score": y_score,
        "y_true": y_true,
        "val_score": best_val_scores[0] if best_val_scores is not None else None,
        "val_true": best_val_scores[1] if best_val_scores is not None else None,
        "train_shape": list(x_train.shape),
        "val_shape": list(x_val.shape) if x_val is not None else None,
        "test_shape": list(x_test.shape),
        "device": str(device),
        "synthetic_preictal_mixup": synthetic_preictal_mixup,
        "temporal_consistency": temporal_pair_manifest,
    }


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = [
        "acc",
        "auc",
        "f1",
        "sensitivity",
        "specificity",
        "event_detected",
        "false_alarm_rate_per_hour",
    ]
    result = {"folds": len(rows)}
    for key in numeric_keys:
        vals = [float(r[key]) for r in rows if r.get(key) not in ("", None) and not math.isnan(float(r[key]))]
        if vals:
            result[f"{key}_mean"] = sum(vals) / len(vals)
            result[f"{key}_std"] = math.sqrt(sum((v - result[f"{key}_mean"]) ** 2 for v in vals) / len(vals))
    return result


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    result = summarize_rows(rows)
    by_subject: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_subject.setdefault(str(row["subject"]), []).append(row)
    result["subjects"] = {
        subject: summarize_rows(subject_rows)
        for subject, subject_rows in sorted(by_subject.items())
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loocv-dir", type=Path, default=Path("data/chbmit/loocv_meta"))
    parser.add_argument("--data-root", type=Path, default=Path("data/chbmit/chb-mit-scalp-eeg-database-1.0.0"))
    parser.add_argument(
        "--dcsa-feature-cache",
        type=Path,
        default=None,
        help=(
            "Global exact DC-SA handcrafted feature cache containing features.npz. "
            "When set, folds use [channels,24] descriptors instead of raw windows."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/baseline_cnn_gru_loocv"))
    parser.add_argument("--channels", default=",".join(COMMON_CHANNELS))
    parser.add_argument("--subjects", default="")
    parser.add_argument(
        "--fold-id",
        default="",
        help="Optional exact LOOCV fold_id to run; useful for fold-specific pretraining.",
    )
    parser.add_argument("--limit-folds", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=0,
        help="Epoch interval for StepLR; zero disables scheduling.",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        help="Multiplicative StepLR decay factor.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without validation improvement; zero disables it.",
    )
    parser.add_argument(
        "--loss",
        default="ce",
        choices=["ce", "weighted_ce", "focal"],
        help="Training loss: plain cross entropy, class-weighted CE, or focal loss.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--train-sampler",
        default="shuffle",
        choices=["shuffle", "event_balanced"],
        help="Training sampler: ordinary shuffled batches or event/file-balanced weighted sampling.",
    )
    parser.add_argument(
        "--supcon-weight",
        type=float,
        default=0.0,
        help="Weight for supervised contrastive auxiliary loss on model embeddings.",
    )
    parser.add_argument(
        "--supcon-temperature",
        type=float,
        default=0.2,
        help="Temperature for supervised contrastive auxiliary loss.",
    )
    parser.add_argument(
        "--channel-dropout-prob",
        type=float,
        default=0.0,
        help="Training-only probability of dropping each EEG channel for robustness.",
    )
    parser.add_argument(
        "--channel-noise-std",
        type=float,
        default=0.0,
        help="Training-only Gaussian noise standard deviation after per-window z-score.",
    )
    parser.add_argument(
        "--freq-aux-weight",
        type=float,
        default=0.0,
        help="Weight for synthetic sine-frequency identification auxiliary loss.",
    )
    parser.add_argument(
        "--freq-aux-frequencies",
        default="3,6,10,20",
        help="Comma-separated sine frequencies in Hz for the auxiliary task.",
    )
    parser.add_argument(
        "--freq-aux-amplitude",
        type=float,
        default=0.25,
        help="Amplitude of the synthetic sine signal added for frequency auxiliary training.",
    )
    parser.add_argument(
        "--freq-aux-window-sec",
        type=float,
        default=7.0,
        help="Window duration used to synthesize auxiliary sine waves.",
    )
    parser.add_argument(
        "--synthetic-preictal-mixup-ratio",
        type=float,
        default=0.0,
        help=(
            "Training-only synthetic preictal sample count as a ratio of real "
            "training preictal windows. Validation and test splits are unchanged."
        ),
    )
    parser.add_argument(
        "--synthetic-preictal-mixup-alpha",
        type=float,
        default=0.4,
        help="Beta distribution alpha for training-only synthetic preictal mixup.",
    )
    parser.add_argument(
        "--synthetic-preictal-mixup-noise-std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise std added to synthetic preictal mixup samples.",
    )
    parser.add_argument(
        "--feature-preictal-mixup-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for training-only feature-space preictal mixup loss. "
            "Uses only embeddings from positive training windows in each batch."
        ),
    )
    parser.add_argument(
        "--feature-preictal-mixup-ratio",
        type=float,
        default=0.5,
        help="Synthetic feature count as a ratio of positive samples in each training batch.",
    )
    parser.add_argument(
        "--feature-preictal-mixup-alpha",
        type=float,
        default=0.4,
        help="Beta distribution alpha for feature-space preictal mixup.",
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for feature-space hard-negative contrastive loss. "
            "Penalizes interictal embeddings that are too similar to preictal embeddings."
        ),
    )
    parser.add_argument(
        "--hard-negative-topk",
        type=int,
        default=8,
        help="Number of most preictal-like interictal embeddings used per positive sample.",
    )
    parser.add_argument(
        "--hard-negative-margin",
        type=float,
        default=0.2,
        help="Cosine-similarity margin for hard-negative contrastive loss.",
    )
    parser.add_argument(
        "--hard-negative-temperature",
        type=float,
        default=0.1,
        help="Temperature scaling for hard-negative contrastive loss margin violations.",
    )
    parser.add_argument(
        "--temporal-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for training-only temporal consistency loss between adjacent "
            "windows. The first use case is stabilizing interictal predictions."
        ),
    )
    parser.add_argument(
        "--temporal-consistency-label-mode",
        choices=["interictal", "preictal", "same", "all"],
        default="interictal",
        help="Which adjacent training-window pairs contribute to temporal consistency.",
    )
    parser.add_argument(
        "--temporal-consistency-source",
        choices=["train_windows", "interictal_pool"],
        default="train_windows",
        help=(
            "Source for temporal consistency pairs. interictal_pool uses an auxiliary "
            "contiguous non-test interictal segment that is not part of CE training."
        ),
    )
    parser.add_argument(
        "--temporal-consistency-pool-windows",
        type=int,
        default=0,
        help="Maximum auxiliary interictal-pool windows per fold for temporal consistency.",
    )
    parser.add_argument(
        "--temporal-consistency-gap-sec",
        type=int,
        default=0,
        help="Maximum allowed gap between adjacent temporal-consistency windows.",
    )
    parser.add_argument(
        "--temporal-consistency-max-pairs-per-batch",
        type=int,
        default=64,
        help="Number of adjacent pairs sampled per training batch for consistency loss.",
    )
    parser.add_argument(
        "--model",
        default="cnn_gru",
        choices=[
            "cnn_gru",
            "dual_stream_freq",
            "attnsleep",
            "tri_band_dsf",
            "residual_tri_band_dsf",
            "same_domain_residual_tri_band_dsf",
            "feature_residual_tri_band_dsf",
            "graph_residual_dsf",
            "dufgnet",
            "s_dufgnet",
            "progressive_dsf",
            "progressive_internal_dsf",
            "fft_emd_dsf",
            "adaptive_spectral_dsf",
            "channel_attention_dsf",
            "channel_token_self_attention_dsf",
            "eegnet_dsf",
            "conv3d_dsf",
            "dwt_dual_stream",
            "dwt_db4_dsf",
            "dwt_sym5_dsf",
            "cwt_scalogram_dsf",
            "learnable_wavelet_dsf",
            "fft_dwt_gated_dsf",
            "temporal_context_dual_stream",
            "mamba_context_dsf",
            "multi_scale_window_dsf",
            "dynamic_graph_attention_dsf",
            "graph_freq",
            "pretrained_biot",
            "dc_sa_ebilstm",
            "dc_sa_ebilstm_faithful",
            "transformer_rnn_fusion",
        ],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--progressive-blocks",
        default="",
        help=(
            "Comma-separated ordered atoms for progressive_dsf. Supported atoms: "
            "temporal, frequency, channel, attention."
        ),
    )
    parser.add_argument(
        "--progressive-internal-stage",
        type=int,
        default=0,
        choices=list(range(6)),
        help=(
            "Cumulative internal DSF growth stage for progressive_internal_dsf: "
            "0 temporal stem, 1 temporal block, 2 FFT branch, 3 frequency block, "
            "4 fusion, 5 GRU."
        ),
    )
    parser.add_argument(
        "--context-windows",
        type=int,
        default=1,
        help=(
            "Number of causal consecutive 7-second windows per sample. "
            "Use >1 with temporal_context_dual_stream or mamba_context_dsf."
        ),
    )
    parser.add_argument(
        "--context-contiguous",
        action="store_true",
        help=(
            "Stop causal context at missing time intervals and left-pad from the "
            "earliest contiguous window instead of crossing temporal gaps."
        ),
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=7.0,
        help="Nominal window duration in seconds for event-level FAR/h fallback calculations.",
    )
    parser.add_argument(
        "--emd-components",
        type=int,
        default=3,
        help="Number of per-channel EMD components appended for fft_emd_dsf.",
    )
    parser.add_argument(
        "--emd-downsample",
        type=int,
        default=2,
        help="Downsample factor used only during per-window EMD decomposition.",
    )
    parser.add_argument(
        "--emd-workers",
        type=int,
        default=8,
        help="Number of CPU workers for per-window EMD feature extraction.",
    )
    parser.add_argument(
        "--emd-channel-indices",
        default="0,4,8,12",
        help="Comma-separated raw-channel indices used for EMD; empty means all channels.",
    )
    parser.add_argument(
        "--resample-sfreq",
        type=float,
        default=0.0,
        help=(
            "Optional target sampling rate used to resample loaded EDF windows to "
            "a fixed sample count. Use 0 to keep native sampling."
        ),
    )
    parser.add_argument(
        "--multiscale-5s",
        action="store_true",
        help=(
            "Build right-aligned 5/10/15-second multi-input samples from 5-second "
            "metadata and relabel positives only when all context labels are positive."
        ),
    )
    parser.add_argument(
        "--multiscale-scales-sec",
        default="5,10,15",
        help="Comma-separated input durations in seconds for --multiscale-5s.",
    )
    parser.add_argument(
        "--multiscale-required-context-windows",
        type=int,
        default=3,
        help="Number of contiguous 5-second labels required by the all-positive rule.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint-selection",
        choices=["validation_auc", "validation_event"],
        default="validation_auc",
        help=(
            "Checkpoint objective. validation_event uses validation event detection, "
            "then FAR/h, warning time, and AUC; test predictions are not evaluated "
            "until the selected checkpoint is frozen."
        ),
    )
    parser.add_argument(
        "--validation-alarm-threshold",
        type=float,
        default=0.5,
        help="Fixed validation threshold used by validation_event checkpoint selection.",
    )
    parser.add_argument(
        "--validation-alarm-min-windows",
        type=int,
        default=1,
        help="Minimum consecutive validation alarm windows for checkpoint selection.",
    )
    parser.add_argument(
        "--validation-alarm-merge-gap-sec",
        type=int,
        default=7,
        help="Maximum gap used to merge validation alarm windows.",
    )
    parser.add_argument(
        "--validation-alarm-refractory-min",
        type=float,
        default=30.0,
        help="Refractory period used by validation_event checkpoint selection.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "Seed used only for the fold-internal validation split. When "
            "omitted, --seed preserves the historical behavior. Fix this "
            "value when measuring model-training seed stability."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--validation-mode",
        default="none",
        choices=["none", "event"],
        help="Optional fold-internal validation split from non-held-out training seizures.",
    )
    parser.add_argument(
        "--validation-min-positive",
        type=int,
        default=30,
        help="Minimum preictal windows required for a seizure to be used as validation.",
    )
    parser.add_argument(
        "--validation-event-count",
        type=int,
        default=1,
        help="Number of eligible non-held-out seizure events to reserve for validation.",
    )
    parser.add_argument(
        "--validation-event-selection",
        choices=["random", "chronological_first", "chronological_last"],
        default="random",
        help=(
            "How to choose validation seizures from the non-held-out pool. "
            "chronological_first/last select a contiguous time-ordered block; "
            "random preserves the historical behavior."
        ),
    )
    parser.add_argument(
        "--validation-negative-ratio",
        type=float,
        default=1.0,
        help=(
            "Validation interictal target as a multiple of validation preictal windows. "
            "Values <=1 preserve the original negative sampling behavior."
        ),
    )
    parser.add_argument(
        "--validation-min-train-negative-ratio",
        type=float,
        default=1.0,
        help=(
            "Minimum remaining train interictal windows as a multiple of remaining "
            "train preictal windows when validation-negative-ratio is increased."
        ),
    )
    parser.add_argument(
        "--validation-negative-mode",
        choices=["random", "contiguous"],
        default="random",
        help=(
            "How to reserve validation interictal windows. Use contiguous for "
            "event-level calibration so validation false alarms are measured on a time stream."
        ),
    )
    parser.add_argument(
        "--validation-contiguous-gap-sec",
        type=int,
        default=0,
        help=(
            "Maximum gap allowed between adjacent validation interictal windows "
            "when --validation-negative-mode=contiguous."
        ),
    )
    parser.add_argument(
        "--pretrained-encoder",
        type=Path,
        default=None,
        help="Optional self-supervised encoder checkpoint to initialize compatible model weights.",
    )
    parser.add_argument(
        "--init-from-result-root",
        type=Path,
        default=None,
        help=(
            "Previous progressive stage result root. Each fold loads "
            "<root>/<subject>/<fold_id>/best_model.pt before training."
        ),
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze all encoder parameters and train only the classifier head for patient adaptation.",
    )
    parser.add_argument(
        "--residual-adapter",
        action="store_true",
        help="Add a bottleneck residual adapter in feature space before classification.",
    )
    parser.add_argument(
        "--residual-adapter-bottleneck",
        type=int,
        default=32,
        help="Bottleneck width for --residual-adapter.",
    )
    parser.add_argument("--rebuild-datasets", action="store_true")
    parser.add_argument("--keep-datasets", action="store_true")
    parser.add_argument(
        "--disable-window-zscore",
        action="store_true",
        help=(
            "Preserve physical window amplitudes. Intended for handcrafted RMS/STD "
            "features whose fold-level descriptors are standardized later."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip folds that already contain test_predictions.csv and metrics.json. "
            "Use this to continue an interrupted full LOOCV run."
        ),
    )
    args = parser.parse_args()
    if args.split_seed is None:
        args.split_seed = args.seed

    import numpy as np

    channels = [ch.strip() for ch in args.channels.split(",") if ch.strip()]
    # A prior paper-protocol launch used the common typo ``FP2-O2``.  The
    # CHB-MIT montage names this bipolar derivation ``P4-O2``; normalize the
    # legacy spelling at the boundary so a bad launch cannot drop every fold.
    channel_aliases = {"FP2-O2": "P4-O2"}
    normalized_channels = [channel_aliases.get(channel, channel) for channel in channels]
    if normalized_channels != channels:
        print(
            "WARNING: normalized channel aliases: "
            + ", ".join(f"{old}->{new}" for old, new in zip(channels, normalized_channels) if old != new),
            flush=True,
        )
        channels = normalized_channels
    if args.dcsa_feature_cache is not None and tuple(channels) != DCSA_EXACT_CHANNELS:
        raise ValueError(
            "--dcsa-feature-cache requires the exact cached channel names and order: "
            + ",".join(DCSA_EXACT_CHANNELS)
        )
    progressive_blocks = tuple(
        block.strip() for block in args.progressive_blocks.split(",") if block.strip()
    )
    emd_channel_indices = tuple(
        int(item.strip()) for item in args.emd_channel_indices.split(",") if item.strip()
    )
    multiscale_scales_sec = tuple(int(item) for item in parse_float_list(args.multiscale_scales_sec))
    target_samples = (
        int(round(float(args.window_sec) * float(args.resample_sfreq)))
        if float(args.resample_sfreq) > 0.0
        else None
    )
    if args.multiscale_5s and len(multiscale_scales_sec) < 2:
        raise ValueError("--multiscale-5s requires at least two --multiscale-scales-sec values")
    if args.model == "mamba_context_dsf" and args.context_windows <= 1:
        raise ValueError("mamba_context_dsf requires --context-windows greater than 1")
    summary = json.loads((args.loocv_dir / "loocv_summary.json").read_text())
    selected_subjects = {s.strip() for s in args.subjects.split(",") if s.strip()}
    folds = [
        fold for fold in summary["folds"]
        if not selected_subjects or fold["subject"] in selected_subjects
    ]
    if args.fold_id:
        folds = [fold for fold in folds if fold["fold_id"] == args.fold_id]
        if not folds:
            raise ValueError(f"Requested --fold-id not found: {args.fold_id}")
    if args.limit_folds:
        folds = folds[: args.limit_folds]

    fold_results = []
    skipped_results = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, fold in enumerate(folds, start=1):
        fold_id = fold["fold_id"]
        subject = fold["subject"]
        fold_out_dir = args.out_dir / subject / fold_id
        if args.resume:
            required_outputs = [
                fold_out_dir / "test_predictions.csv",
                fold_out_dir / "metrics.json",
            ]
            if all(path.exists() for path in required_outputs):
                print(f"=== Fold {index}/{len(folds)}: {fold_id} (resume: existing, skipped) ===")
                continue
        print(f"\n=== Fold {index}/{len(folds)}: {fold_id} ===")
        fold_meta_dir = Path(fold["dir"])
        try:
            dataset_path, dataset_manifest = load_or_build_npz(
                fold_meta_dir=fold_meta_dir,
                fold_work_dir=fold_out_dir,
                data_root=args.data_root,
                channels=channels,
                rebuild=args.rebuild_datasets,
                validation_mode=args.validation_mode,
                seed=args.split_seed,
                validation_min_positive=args.validation_min_positive,
                validation_event_count=args.validation_event_count,
                validation_event_selection=args.validation_event_selection,
                validation_negative_ratio=args.validation_negative_ratio,
                validation_min_train_negative_ratio=args.validation_min_train_negative_ratio,
                validation_negative_mode=args.validation_negative_mode,
                validation_contiguous_gap_sec=args.validation_contiguous_gap_sec,
                multiscale_5s=args.multiscale_5s,
                multiscale_scales_sec=multiscale_scales_sec,
                multiscale_required_context_windows=args.multiscale_required_context_windows,
                temporal_consistency_source=args.temporal_consistency_source,
                temporal_consistency_pool_windows=args.temporal_consistency_pool_windows,
                temporal_consistency_gap_sec=args.temporal_consistency_gap_sec,
                target_samples=target_samples,
                emd_features=args.model == "fft_emd_dsf",
                emd_components=args.emd_components,
                emd_downsample=args.emd_downsample,
                emd_workers=args.emd_workers,
                emd_channel_indices=emd_channel_indices,
                normalize_window=not args.disable_window_zscore,
                dcsa_feature_cache=args.dcsa_feature_cache,
            )
        except Exception as exc:
            skipped_results.append(
                {
                    "fold_id": fold_id,
                    "subject": subject,
                    "heldout_file": fold["heldout_file"],
                    "heldout_seizure_index_in_file": fold["heldout_seizure_index_in_file"],
                    "reason": f"dataset build failed: {exc}",
                }
            )
            print(f"Skipped fold {fold_id}: dataset build failed: {exc}")
            continue

        counts = dataset_manifest.get("loaded_counts", {})
        train_counts = counts.get("train", {})
        test_counts = counts.get("test", {})
        if (
            int(train_counts.get("positive", 0)) == 0
            or int(train_counts.get("negative", 0)) == 0
            or int(test_counts.get("positive", 0)) == 0
            or int(test_counts.get("negative", 0)) == 0
        ):
            skipped_results.append(
                {
                    "fold_id": fold_id,
                    "subject": subject,
                    "heldout_file": fold["heldout_file"],
                    "heldout_seizure_index_in_file": fold["heldout_seizure_index_in_file"],
                    "reason": "loaded fold lacks both classes after channel/file skips",
                    "loaded_counts": json.dumps(counts),
                    "skipped_files": len(dataset_manifest.get("skipped_files", [])),
                }
            )
            print(f"Skipped fold {fold_id}: invalid loaded class counts {counts}")
            dataset_path.unlink(missing_ok=True)
            continue

        train_result = train_one_fold(dataset_path, fold_out_dir, args)
        test_rows = read_rows(fold_out_dir / "test_windows.csv")
        event = evaluate_event(
            test_rows,
            train_result["y_true"],
            train_result["y_score"],
            threshold=args.threshold,
            fallback_window_sec=args.window_sec,
        )

        prediction_rows = []
        y_pred = (train_result["y_score"] >= args.threshold).astype("int64")
        for i, (row, label, pred, score) in enumerate(zip(test_rows, train_result["y_true"], y_pred, train_result["y_score"], strict=True)):
            prediction_rows.append(
                {
                    "index": i,
                    "subject": row["subject"],
                    "file_name": row["file_name"],
                    "window_start_sec": int(row["window_start_sec"]),
                    "window_end_sec": int(row["window_end_sec"]),
                    "state": row["state"],
                    "label": int(label),
                    "pred": int(pred),
                    "score_preictal": float(score),
                    "matched_seizure_index_in_file": row["matched_seizure_index_in_file"],
                    "matched_seizure_file_name": row.get("matched_seizure_file_name", ""),
                    "sph_sec": row.get("sph_sec", ""),
                    "sop_sec": row.get("sop_sec", ""),
                    "source_dataset": row.get("source_dataset", ""),
                    "source_split": row.get("source_split", ""),
                    "session": row.get("session", ""),
                    "montage": row.get("montage", ""),
                    "sequence_id": row.get("sequence_id", ""),
                    "abs_window_start_sec": row.get("abs_window_start_sec", ""),
                    "abs_window_end_sec": row.get("abs_window_end_sec", ""),
                }
            )
        write_csv(
            fold_out_dir / "test_predictions.csv",
            prediction_rows,
            [
                "index",
                "subject",
                "file_name",
                "window_start_sec",
                "window_end_sec",
                "state",
                "label",
                "pred",
                "score_preictal",
                "matched_seizure_index_in_file",
                "matched_seizure_file_name",
                "sph_sec",
                "sop_sec",
                "source_dataset",
                "source_split",
                "session",
                "montage",
                "sequence_id",
                "abs_window_start_sec",
                "abs_window_end_sec",
            ],
        )

        if train_result["val_score"] is not None and (fold_out_dir / "val_windows.csv").exists():
            val_rows = read_rows(fold_out_dir / "val_windows.csv")
            val_pred = (train_result["val_score"] >= args.threshold).astype("int64")
            val_prediction_rows = []
            for i, (row, label, pred, score) in enumerate(
                zip(val_rows, train_result["val_true"], val_pred, train_result["val_score"], strict=True)
            ):
                val_prediction_rows.append(
                    {
                        "index": i,
                        "subject": row["subject"],
                        "file_name": row["file_name"],
                        "window_start_sec": int(row["window_start_sec"]),
                        "window_end_sec": int(row["window_end_sec"]),
                        "state": row["state"],
                        "label": int(label),
                        "pred": int(pred),
                        "score_preictal": float(score),
                        "matched_seizure_index_in_file": row["matched_seizure_index_in_file"],
                        "matched_seizure_file_name": row.get("matched_seizure_file_name", ""),
                        "sph_sec": row.get("sph_sec", ""),
                        "sop_sec": row.get("sop_sec", ""),
                        "source_dataset": row.get("source_dataset", ""),
                        "source_split": row.get("source_split", ""),
                        "session": row.get("session", ""),
                        "montage": row.get("montage", ""),
                        "sequence_id": row.get("sequence_id", ""),
                        "abs_window_start_sec": row.get("abs_window_start_sec", ""),
                        "abs_window_end_sec": row.get("abs_window_end_sec", ""),
                    }
                )
            write_csv(
                fold_out_dir / "val_predictions.csv",
                val_prediction_rows,
                [
                    "index",
                    "subject",
                    "file_name",
                    "window_start_sec",
                    "window_end_sec",
                    "state",
                    "label",
                    "pred",
                    "score_preictal",
                    "matched_seizure_index_in_file",
                    "matched_seizure_file_name",
                    "sph_sec",
                    "sop_sec",
                    "source_dataset",
                    "source_split",
                    "session",
                    "montage",
                    "sequence_id",
                    "abs_window_start_sec",
                    "abs_window_end_sec",
                ],
            )

        best = train_result["best"]
        fold_row = {
            "fold_id": fold_id,
            "subject": subject,
            "heldout_file": fold["heldout_file"],
            "heldout_seizure_index_in_file": fold["heldout_seizure_index_in_file"],
            "epoch": best["epoch"],
            "acc": best["acc"],
            "auc": best["auc"],
            "f1": best["f1"],
            "sensitivity": best["sensitivity"],
            "specificity": best["specificity"],
            "val_acc": best.get("val_acc", ""),
            "val_auc": best.get("val_auc", ""),
            "val_f1": best.get("val_f1", ""),
            "val_sensitivity": best.get("val_sensitivity", ""),
            "val_specificity": best.get("val_specificity", ""),
            "validation_alarm_event_detected": best.get("validation_alarm_event_detected", ""),
            "validation_alarm_event_warning_time_min": best.get(
                "validation_alarm_event_warning_time_min", ""
            ),
            "validation_alarm_false_alarm_rate_per_hour": best.get(
                "validation_alarm_false_alarm_rate_per_hour", ""
            ),
            "checkpoint_selection": best.get("selection_source", ""),
            "tp": best["tp"],
            "tn": best["tn"],
            "fp": best["fp"],
            "fn": best["fn"],
            **event,
            "train_windows": train_result["train_shape"][0],
            "val_windows": train_result["val_shape"][0] if train_result["val_shape"] else 0,
            "test_windows": train_result["test_shape"][0],
            "skipped_files": len(dataset_manifest.get("skipped_files", [])),
        }
        fold_results.append(fold_row)
        (fold_out_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "dataset_manifest": dataset_manifest,
                    "best": best,
                    "history": train_result["history"],
                    "event": event,
                    "train_shape": train_result["train_shape"],
                    "val_shape": train_result["val_shape"],
                    "test_shape": train_result["test_shape"],
                    "device": train_result["device"],
                    "synthetic_preictal_mixup": train_result["synthetic_preictal_mixup"],
                    "temporal_consistency": train_result["temporal_consistency"],
                },
                indent=2,
            )
            + "\n"
        )
        if not args.keep_datasets:
            dataset_path.unlink(missing_ok=True)

    fieldnames = [
        "fold_id",
        "subject",
        "heldout_file",
        "heldout_seizure_index_in_file",
        "epoch",
        "acc",
        "auc",
        "f1",
        "sensitivity",
        "specificity",
        "val_acc",
        "val_auc",
        "val_f1",
        "val_sensitivity",
        "val_specificity",
        "validation_alarm_event_detected",
        "validation_alarm_event_warning_time_min",
        "validation_alarm_false_alarm_rate_per_hour",
        "checkpoint_selection",
        "tp",
        "tn",
        "fp",
        "fn",
        "event_detected",
        "event_preictal_windows",
        "event_first_alarm_start_sec",
        "event_warning_time_sec",
        "event_warning_time_min",
        "false_alarm_windows",
        "interictal_windows",
        "estimated_interictal_hours",
        "false_alarm_rate_per_hour",
        "max_preictal_score",
        "mean_preictal_score",
        "train_windows",
        "val_windows",
        "test_windows",
        "skipped_files",
    ]
    write_csv(args.out_dir / "fold_metrics.csv", fold_results, fieldnames)
    write_csv(
        args.out_dir / "skipped_folds.csv",
        skipped_results,
        [
            "fold_id",
            "subject",
            "heldout_file",
            "heldout_seizure_index_in_file",
            "reason",
            "loaded_counts",
            "skipped_files",
        ],
    )
    aggregate_summary = aggregate(fold_results)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "loocv_dir": str(args.loocv_dir),
                "folds_requested": len(folds),
                "folds_completed": len(fold_results),
                "folds_skipped": len(skipped_results),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "loss": args.loss,
                "focal_gamma": args.focal_gamma,
                "train_sampler": args.train_sampler,
                "supcon_weight": args.supcon_weight,
                "supcon_temperature": args.supcon_temperature,
                "channel_dropout_prob": args.channel_dropout_prob,
                "channel_noise_std": args.channel_noise_std,
                "freq_aux_weight": args.freq_aux_weight,
                "freq_aux_frequencies": args.freq_aux_frequencies,
                "freq_aux_amplitude": args.freq_aux_amplitude,
                "freq_aux_window_sec": args.freq_aux_window_sec,
                "synthetic_preictal_mixup_ratio": args.synthetic_preictal_mixup_ratio,
                "synthetic_preictal_mixup_alpha": args.synthetic_preictal_mixup_alpha,
                "synthetic_preictal_mixup_noise_std": args.synthetic_preictal_mixup_noise_std,
                "feature_preictal_mixup_weight": args.feature_preictal_mixup_weight,
                "feature_preictal_mixup_ratio": args.feature_preictal_mixup_ratio,
                "feature_preictal_mixup_alpha": args.feature_preictal_mixup_alpha,
                "hard_negative_weight": args.hard_negative_weight,
                "hard_negative_topk": args.hard_negative_topk,
                "hard_negative_margin": args.hard_negative_margin,
                "hard_negative_temperature": args.hard_negative_temperature,
                "temporal_consistency_weight": args.temporal_consistency_weight,
                "temporal_consistency_label_mode": args.temporal_consistency_label_mode,
                "temporal_consistency_source": args.temporal_consistency_source,
                "temporal_consistency_pool_windows": args.temporal_consistency_pool_windows,
                "temporal_consistency_gap_sec": args.temporal_consistency_gap_sec,
                "temporal_consistency_max_pairs_per_batch": args.temporal_consistency_max_pairs_per_batch,
                "validation_mode": args.validation_mode,
                "validation_min_positive": args.validation_min_positive,
                "validation_event_count": args.validation_event_count,
                "validation_event_selection": args.validation_event_selection,
                "validation_negative_ratio": args.validation_negative_ratio,
                "validation_min_train_negative_ratio": args.validation_min_train_negative_ratio,
                "validation_negative_mode": args.validation_negative_mode,
                "validation_contiguous_gap_sec": args.validation_contiguous_gap_sec,
                "model": args.model,
                "checkpoint_selection": args.checkpoint_selection,
                "validation_alarm_threshold": args.validation_alarm_threshold,
                "validation_alarm_min_windows": args.validation_alarm_min_windows,
                "validation_alarm_merge_gap_sec": args.validation_alarm_merge_gap_sec,
                "validation_alarm_refractory_min": args.validation_alarm_refractory_min,
            "progressive_blocks": list(progressive_blocks),
                "progressive_internal_stage": args.progressive_internal_stage,
                "channels": channels,
                "context_windows": args.context_windows,
                "window_sec": args.window_sec,
                "multiscale_5s": args.multiscale_5s,
                "multiscale_scales_sec": list(multiscale_scales_sec),
                "multiscale_required_context_windows": args.multiscale_required_context_windows,
                "pretrained_encoder": str(args.pretrained_encoder) if args.pretrained_encoder else "",
                "init_from_result_root": str(args.init_from_result_root) if args.init_from_result_root else "",
                "freeze_encoder": args.freeze_encoder,
                "residual_adapter": args.residual_adapter,
                "residual_adapter_bottleneck": args.residual_adapter_bottleneck,
                "threshold": args.threshold,
                "aggregate": aggregate_summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(aggregate_summary, indent=2))


if __name__ == "__main__":
    main()
