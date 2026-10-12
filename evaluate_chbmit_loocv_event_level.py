#!/usr/bin/env python3
"""Event-level post-processing evaluation for CHB-MIT LOOCV predictions.

This script does not retrain models. It reads fold-level ``test_predictions.csv``
files, merges positive windows into alarm events, applies a refractory period,
and reports false alarm rate at the alarm-event level.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_thresholds(raw: str) -> list[float]:
    raw = raw.strip()
    if ":" not in raw:
        return [float(item) for item in raw.split(",") if item.strip()]
    start, stop, step = [float(item) for item in raw.split(":")]
    values = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def sequence_key(row: dict[str, str] | dict[str, object]) -> str:
    return str(row.get("sequence_id", "")).strip() or str(row["file_name"])


def time_start(row: dict[str, str]) -> int:
    return int(float(row.get("abs_window_start_sec", "") or row["window_start_sec"]))


def time_end(row: dict[str, str]) -> int:
    return int(float(row.get("abs_window_end_sec", "") or row["window_end_sec"]))


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, object]:
    preds = [int(score >= threshold) for score in scores]
    tp = sum(1 for y, p in zip(labels, preds, strict=True) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds, strict=True) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds, strict=True) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds, strict=True) if y == 1 and p == 0)
    total = len(labels)
    precision = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    return {
        "sample_acc": (tp + tn) / max(total, 1),
        "sample_f1": f1,
        "sample_sensitivity": sensitivity,
        "sample_specificity": specificity,
        "sample_tp": tp,
        "sample_tn": tn,
        "sample_fp": fp,
        "sample_fn": fn,
    }


def window_hours(rows: list[dict[str, str]], fallback_window_sec: int) -> float:
    seconds = 0.0
    for row in rows:
        start = safe_float(row.get("window_start_sec"))
        end = safe_float(row.get("window_end_sec"))
        if math.isfinite(start) and math.isfinite(end) and end > start:
            seconds += end - start
        else:
            seconds += fallback_window_sec
    return seconds / 3600.0


def collect_alarm_candidates(
    rows: list[dict[str, str]],
    threshold: float,
    label: int | None = None,
) -> list[dict[str, object]]:
    candidates = []
    for row in rows:
        row_label = int(row["label"])
        if label is not None and row_label != label:
            continue
        score = safe_float(row["score_preictal"])
        if score < threshold:
            continue
        candidates.append(
            {
                "subject": row["subject"],
                "file_name": row["file_name"],
                "sequence_id": sequence_key(row),
                "start": time_start(row),
                "end": time_end(row),
                "score": score,
                "label": row_label,
                "matched_seizure_index_in_file": row.get("matched_seizure_index_in_file", ""),
                "matched_seizure_file_name": row.get("matched_seizure_file_name", ""),
            }
        )
    return candidates


def merge_consecutive_alarms(
    candidates: list[dict[str, object]],
    max_gap_sec: int,
    min_windows: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for alarm in candidates:
        grouped[(str(alarm["subject"]), sequence_key(alarm))].append(alarm)

    merged = []
    for (subject, file_name), alarms in grouped.items():
        alarms.sort(key=lambda x: (int(x["start"]), int(x["end"])))
        current: dict[str, object] | None = None
        for alarm in alarms:
            if current is None:
                current = {
                    "subject": subject,
                    "file_name": str(alarm["file_name"]),
                    "sequence_id": file_name,
                    "start": int(alarm["start"]),
                    "end": int(alarm["end"]),
                    "max_score": float(alarm["score"]),
                    "windows": 1,
                }
                continue
            if int(alarm["start"]) <= int(current["end"]) + max_gap_sec:
                current["end"] = max(int(current["end"]), int(alarm["end"]))
                current["max_score"] = max(float(current["max_score"]), float(alarm["score"]))
                current["windows"] = int(current["windows"]) + 1
            else:
                if int(current["windows"]) >= min_windows:
                    merged.append(current)
                current = {
                    "subject": subject,
                    "file_name": str(alarm["file_name"]),
                    "sequence_id": file_name,
                    "start": int(alarm["start"]),
                    "end": int(alarm["end"]),
                    "max_score": float(alarm["score"]),
                    "windows": 1,
                }
        if current is not None:
            if int(current["windows"]) >= min_windows:
                merged.append(current)
    return merged


def apply_refractory(
    alarms: list[dict[str, object]],
    refractory_sec: int,
) -> list[dict[str, object]]:
    by_file: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for alarm in alarms:
        by_file[(str(alarm["subject"]), sequence_key(alarm))].append(alarm)

    kept = []
    for key in sorted(by_file):
        next_allowed = -1
        for alarm in sorted(by_file[key], key=lambda x: (int(x["start"]), int(x["end"]))):
            start = int(alarm["start"])
            if start < next_allowed:
                continue
            kept.append(alarm)
            next_allowed = start + refractory_sec
    return kept


def estimate_seizure_start(preictal_rows: list[dict[str, str]]) -> int | None:
    if not preictal_rows:
        return None
    starts = [time_start(row) for row in preictal_rows]
    ends = [time_end(row) for row in preictal_rows]
    sph_values = [int(float(row.get("sph_sec", 300))) for row in preictal_rows if row.get("sph_sec")]
    sph_sec = sph_values[0] if sph_values else 300
    return max(ends) + sph_sec


def group_preictal_events(
    positive_rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    """Group positive windows into distinct seizure events.

    Validation splits may contain more than one non-held-out seizure. Treating
    all positive rows as a single event overstates validation sensitivity
    because one detected seizure can mask another missed seizure.
    """

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in positive_rows:
        event_index = row.get("matched_seizure_index_in_file", "").strip()
        if not event_index:
            event_index = "unknown"
        seizure_file = row.get("matched_seizure_file_name", "").strip() or row["file_name"]
        key = (row.get("subject", ""), seizure_file, event_index)
        grouped[key].append(row)
    return grouped


def evaluate_preictal_events(
    positive_rows: list[dict[str, str]],
    threshold: float,
    merge_gap_sec: int,
    refractory_sec: int,
    min_alarm_windows: int,
) -> list[dict[str, object]]:
    event_results = []
    for event_key, event_rows in sorted(group_preictal_events(positive_rows).items()):
        seizure_start = estimate_seizure_start(event_rows)
        candidates = collect_alarm_candidates(event_rows, threshold, label=1)
        alarms = apply_refractory(
            merge_consecutive_alarms(candidates, merge_gap_sec, min_alarm_windows),
            refractory_sec,
        )
        first_alarm = min((int(alarm["start"]) for alarm in alarms), default=None)
        detected = int(first_alarm is not None)
        warning_time_sec = (
            seizure_start - first_alarm
            if detected and seizure_start is not None
            else None
        )
        event_results.append(
            {
                "event_key": "#".join(event_key),
                "detected": detected,
                "alarm_events": len(alarms),
                "first_alarm_start_sec": first_alarm,
                "warning_time_sec": warning_time_sec,
                "preictal_windows": len(event_rows),
            }
        )
    return event_results


def evaluate_fold(
    fold_dir: Path,
    pred_rows: list[dict[str, str]],
    window_rows: list[dict[str, str]],
    threshold: float,
    merge_gap_sec: int,
    refractory_sec: int,
    window_sec: int,
    min_alarm_windows: int,
) -> dict[str, object]:
    labels = [int(row["label"]) for row in pred_rows]
    scores = [safe_float(row["score_preictal"], 0.0) for row in pred_rows]
    metrics = binary_metrics(labels, scores, threshold)

    positive_rows = [row for row in pred_rows if int(row["label"]) == 1]
    negative_rows = [row for row in pred_rows if int(row["label"]) == 0]
    event_results = evaluate_preictal_events(
        positive_rows=positive_rows,
        threshold=threshold,
        merge_gap_sec=merge_gap_sec,
        refractory_sec=refractory_sec,
        min_alarm_windows=min_alarm_windows,
    )
    event_count = len(event_results)
    event_detected_count = sum(int(row["detected"]) for row in event_results)
    event_detected = event_detected_count / event_count if event_count else 0.0
    first_preictal_alarm = min(
        (
            int(row["first_alarm_start_sec"])
            for row in event_results
            if row["first_alarm_start_sec"] is not None
        ),
        default=None,
    )
    warning_times = [
        int(row["warning_time_sec"])
        for row in event_results
        if row["warning_time_sec"] is not None
    ]
    warning_time_sec = mean(warning_times) if warning_times else None

    false_candidates = collect_alarm_candidates(pred_rows, threshold, label=0)
    false_alarm_events = apply_refractory(
        merge_consecutive_alarms(false_candidates, merge_gap_sec, min_alarm_windows),
        refractory_sec,
    )
    interictal_hours = window_hours(negative_rows, window_sec)

    subject = pred_rows[0]["subject"] if pred_rows else fold_dir.parent.name
    return {
        "fold_id": fold_dir.name,
        "subject": subject,
        "threshold": threshold,
        "min_alarm_windows": min_alarm_windows,
        "min_alarm_duration_sec": min_alarm_windows * window_sec,
        **metrics,
        "event_detected": event_detected,
        "event_detected_count": event_detected_count,
        "event_count": event_count,
        "event_preictal_windows": len(positive_rows),
        "event_alarm_events": sum(int(row["alarm_events"]) for row in event_results),
        "event_first_alarm_start_sec": first_preictal_alarm if first_preictal_alarm is not None else "",
        "event_warning_time_sec": warning_time_sec if warning_time_sec is not None else "",
        "event_warning_time_min": round(warning_time_sec / 60, 3) if warning_time_sec is not None else "",
        "false_alarm_windows": sum(1 for row in negative_rows if safe_float(row["score_preictal"], 0.0) >= threshold),
        "false_alarm_events": len(false_alarm_events),
        "interictal_windows": len(negative_rows),
        "estimated_interictal_hours": interictal_hours,
        "false_alarm_rate_per_hour": len(false_alarm_events) / max(interictal_hours, 1e-12),
        "max_preictal_score": max((safe_float(row["score_preictal"], 0.0) for row in positive_rows), default=float("nan")),
        "mean_preictal_score": mean([safe_float(row["score_preictal"], 0.0) for row in positive_rows]) if positive_rows else float("nan"),
        "prediction_rows": len(pred_rows),
        "has_window_rows": int(bool(window_rows)),
        "fold_dir": str(fold_dir),
    }


def summarize(rows: list[dict[str, object]], prefix: str = "") -> dict[str, object]:
    metric_keys = [
        "sample_acc",
        "sample_f1",
        "sample_sensitivity",
        "sample_specificity",
        "event_detected",
        "false_alarm_windows",
        "false_alarm_events",
        "false_alarm_rate_per_hour",
        "event_warning_time_min",
    ]
    out: dict[str, object] = {f"{prefix}folds": len(rows)}
    for key in metric_keys:
        values = [safe_float(row.get(key)) for row in rows]
        values = [value for value in values if math.isfinite(value)]
        if values:
            out[f"{prefix}{key}_mean"] = mean(values)
            out[f"{prefix}{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-roots",
        nargs="+",
        type=Path,
        default=[
            Path("results/baseline_cnn_gru_loocv_full_gpu0"),
            Path("results/baseline_cnn_gru_loocv_full_gpu1"),
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/baseline_cnn_gru_loocv_event_level"))
    parser.add_argument("--thresholds", default="0.05:0.95:0.05")
    parser.add_argument(
        "--min-alarm-windows",
        default="1,2,3,4,5",
        help="Comma-separated minimum consecutive alarm-window counts.",
    )
    parser.add_argument("--merge-gap-sec", type=int, default=7)
    parser.add_argument("--refractory-min", type=float, default=30.0)
    parser.add_argument("--window-sec", type=int, default=7)
    args = parser.parse_args()

    fold_dirs = []
    for root in args.result_roots:
        fold_dirs.extend(path.parent for path in root.glob("*/*/test_predictions.csv"))
    fold_dirs = sorted(set(fold_dirs))
    if not fold_dirs:
        raise SystemExit("No test_predictions.csv files found.")

    thresholds = parse_thresholds(args.thresholds)
    min_alarm_windows_values = [int(item) for item in args.min_alarm_windows.split(",") if item.strip()]
    refractory_sec = int(round(args.refractory_min * 60))

    fold_cache = []
    for fold_dir in fold_dirs:
        prediction_path = fold_dir / "test_predictions.csv"
        window_path = fold_dir / "test_windows.csv"
        fold_cache.append(
            {
                "fold_dir": fold_dir,
                "pred_rows": read_csv(prediction_path),
                "window_rows": read_csv(window_path) if window_path.exists() else [],
            }
        )

    fold_rows = []
    for min_alarm_windows in min_alarm_windows_values:
        for threshold in thresholds:
            for cached in fold_cache:
                fold_rows.append(
                    evaluate_fold(
                        fold_dir=cached["fold_dir"],
                        pred_rows=cached["pred_rows"],
                        window_rows=cached["window_rows"],
                        threshold=threshold,
                        merge_gap_sec=args.merge_gap_sec,
                        refractory_sec=refractory_sec,
                        window_sec=args.window_sec,
                        min_alarm_windows=min_alarm_windows,
                    )
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "fold_event_metrics.csv", fold_rows)

    sweep_rows = []
    for min_alarm_windows in min_alarm_windows_values:
        for threshold in thresholds:
            rows = [
                row
                for row in fold_rows
                if safe_float(row["threshold"]) == threshold
                and int(row["min_alarm_windows"]) == min_alarm_windows
            ]
            rec = {
                "threshold": threshold,
                "min_alarm_windows": min_alarm_windows,
                "min_alarm_duration_sec": min_alarm_windows * args.window_sec,
                **summarize(rows),
            }
            sweep_rows.append(rec)
    write_csv(args.out_dir / "threshold_sweep_event_level.csv", sweep_rows)

    per_subject_rows = []
    for min_alarm_windows in min_alarm_windows_values:
        for threshold in thresholds:
            threshold_rows = [
                row
                for row in fold_rows
                if safe_float(row["threshold"]) == threshold
                and int(row["min_alarm_windows"]) == min_alarm_windows
            ]
            for subject in sorted({str(row["subject"]) for row in threshold_rows}):
                subject_rows = [row for row in threshold_rows if row["subject"] == subject]
                per_subject_rows.append(
                    {
                        "threshold": threshold,
                        "min_alarm_windows": min_alarm_windows,
                        "min_alarm_duration_sec": min_alarm_windows * args.window_sec,
                        "subject": subject,
                        **summarize(subject_rows),
                    }
                )
    write_csv(args.out_dir / "per_subject_event_metrics.csv", per_subject_rows)

    preferred = min(
        sweep_rows,
        key=lambda row: (
            abs(safe_float(row.get("event_detected_mean")) - 0.95),
            safe_float(row.get("false_alarm_rate_per_hour_mean"), float("inf")),
        ),
    )
    summary = {
        "result_roots": [str(root) for root in args.result_roots],
        "folds": len(fold_dirs),
        "thresholds": thresholds,
        "min_alarm_windows": min_alarm_windows_values,
        "merge_gap_sec": args.merge_gap_sec,
        "refractory_min": args.refractory_min,
        "window_sec": args.window_sec,
        "preferred_threshold_by_event_sensitivity_0.95": preferred,
        "outputs": {
            "fold_event_metrics": str(args.out_dir / "fold_event_metrics.csv"),
            "threshold_sweep_event_level": str(args.out_dir / "threshold_sweep_event_level.csv"),
            "per_subject_event_metrics": str(args.out_dir / "per_subject_event_metrics.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
