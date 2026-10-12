#!/usr/bin/env python3
"""Validation-calibrated event-level evaluation for CHB-MIT LOOCV predictions.

Each fold must contain both ``val_predictions.csv`` and ``test_predictions.csv``.
Alarm parameters are selected on validation predictions from non-held-out
training seizures, then applied once to the held-out test seizure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from evaluate_chbmit_loocv_event_level import (
    evaluate_fold,
    parse_thresholds,
    read_csv,
    safe_float,
    sequence_key,
    time_end,
    time_start,
    summarize,
    write_csv,
)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def empirical_cdf(value: float, sorted_values: list[float]) -> float:
    if not sorted_values:
        return 0.5
    return bisect_right(sorted_values, value) / len(sorted_values)


def transform_score(value: float, params: dict[str, object], method: str) -> float:
    if method == "none":
        return value
    if method == "validation_zscore_sigmoid":
        mean = float(params["mean"])
        std = max(float(params["std"]), 1e-6)
        return 1.0 / (1.0 + math.exp(-((value - mean) / std)))
    if method == "validation_minmax":
        low = float(params["min"])
        high = float(params["max"])
        if high <= low:
            return 0.5
        return clip01((value - low) / (high - low))
    if method == "validation_robust_minmax":
        low = float(params["p05"])
        high = float(params["p95"])
        if high <= low:
            return 0.5
        return clip01((value - low) / (high - low))
    if method == "validation_ecdf":
        return empirical_cdf(value, params["scores"])  # type: ignore[arg-type]
    if method == "validation_interictal_ecdf":
        return empirical_cdf(value, params["interictal_scores"])  # type: ignore[arg-type]
    raise ValueError(f"Unsupported score_normalization: {method}")


def normalize_prediction_rows(
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    method: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, float | str]]:
    """Fit score normalization on validation rows only and apply to val/test rows."""
    if method == "none":
        return val_rows, test_rows, {"score_normalization": method}

    scores = sorted(safe_float(row.get("score_preictal"), 0.0) for row in val_rows)
    scores = [score for score in scores if math.isfinite(score)]
    if not scores:
        raise ValueError("Cannot normalize scores: validation predictions are empty.")

    score_mean = sum(scores) / len(scores)
    score_std = math.sqrt(sum((score - score_mean) ** 2 for score in scores) / max(len(scores), 1))
    params: dict[str, float] = {
        "mean": score_mean,
        "std": score_std,
        "min": scores[0],
        "max": scores[-1],
        "p05": percentile(scores, 0.05),
        "p95": percentile(scores, 0.95),
    }
    transform_params: dict[str, object] = {
        **params,
        "scores": scores,
        "interictal_scores": [
            safe_float(row.get("score_preictal"), 0.0)
            for row in val_rows
            if int(row.get("label", 0)) == 0 and math.isfinite(safe_float(row.get("score_preictal"), 0.0))
        ],
    }
    transform_params["interictal_scores"] = sorted(transform_params["interictal_scores"])  # type: ignore[arg-type]
    if method == "validation_interictal_ecdf" and not transform_params["interictal_scores"]:
        raise ValueError("Cannot fit validation_interictal_ecdf: no validation interictal scores.")

    def apply(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized = []
        for row in rows:
            out = dict(row)
            raw_score = safe_float(out.get("score_preictal"), 0.0)
            out["raw_score_preictal"] = out.get("score_preictal", "")
            out["score_preictal"] = str(transform_score(raw_score, transform_params, method))
            normalized.append(out)
        return normalized

    metadata: dict[str, float | str] = {
        "score_normalization": method,
        **params,
        "validation_score_count": len(scores),
        "validation_interictal_score_count": len(transform_params["interictal_scores"]),  # type: ignore[arg-type]
    }
    return apply(val_rows), apply(test_rows), metadata


def smooth_prediction_rows(
    pred_rows: list[dict[str, str]],
    smoothing_windows: int,
) -> list[dict[str, str]]:
    if smoothing_windows <= 1:
        return pred_rows

    grouped: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(pred_rows):
        grouped[(row["subject"], sequence_key(row))].append((index, row))

    smoothed = [dict(row) for row in pred_rows]
    for group_rows in grouped.values():
        group_rows.sort(
            key=lambda item: (
                time_start(item[1]),
                time_end(item[1]),
                item[0],
            )
        )
        rolling_scores: list[float] = []
        for original_index, row in group_rows:
            rolling_scores.append(safe_float(row["score_preictal"], 0.0))
            if len(rolling_scores) > smoothing_windows:
                rolling_scores.pop(0)
            smoothed[original_index]["score_preictal"] = str(sum(rolling_scores) / len(rolling_scores))
    return smoothed


def mean_metric(rows: list[dict[str, object]], key: str) -> float:
    values = [safe_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else float("nan")


def select_from_validation(
    validation_rows: list[dict[str, object]],
    target_event_sensitivity: float,
    selection_strategy: str,
    validation_far_cap: float | None,
) -> dict[str, object]:
    grouped: dict[tuple[float, int], list[dict[str, object]]] = defaultdict(list)
    for row in validation_rows:
        grouped[(safe_float(row["threshold"]), int(row["min_alarm_windows"]))].append(row)

    candidates = []
    for (threshold, min_alarm_windows), rows in grouped.items():
        candidates.append(
            {
                "threshold": threshold,
                "min_alarm_windows": min_alarm_windows,
                "min_alarm_duration_sec": int(rows[0]["min_alarm_duration_sec"]),
                "validation_folds": len(rows),
                "validation_event_detected_mean": mean_metric(rows, "event_detected"),
                "validation_false_alarm_rate_per_hour_mean": mean_metric(rows, "false_alarm_rate_per_hour"),
                "validation_false_alarm_events_mean": mean_metric(rows, "false_alarm_events"),
                "validation_sample_acc_mean": mean_metric(rows, "sample_acc"),
                "validation_sample_f1_mean": mean_metric(rows, "sample_f1"),
                "validation_sample_sensitivity_mean": mean_metric(rows, "sample_sensitivity"),
                "validation_sample_specificity_mean": mean_metric(rows, "sample_specificity"),
            }
        )

    eligible = [
        row for row in candidates
        if safe_float(row["validation_event_detected_mean"]) >= target_event_sensitivity
    ]
    if eligible:
        capped = eligible
        if validation_far_cap is not None and math.isfinite(validation_far_cap):
            capped = [
                row for row in eligible
                if safe_float(row["validation_false_alarm_rate_per_hour_mean"], float("inf"))
                <= validation_far_cap
            ]
        selection_pool = capped or eligible
        if selection_strategy == "min_far":
            selected = min(
                selection_pool,
                key=lambda row: (
                    safe_float(row["validation_false_alarm_rate_per_hour_mean"], float("inf")),
                    safe_float(row["validation_false_alarm_events_mean"], float("inf")),
                    -safe_float(row["validation_sample_specificity_mean"], -float("inf")),
                    -safe_float(row["threshold"], -float("inf")),
                    safe_float(row["min_alarm_windows"], float("inf")),
                ),
            )
        elif selection_strategy == "max_sensitivity_under_far_cap":
            selected = max(
                selection_pool,
                key=lambda row: (
                    safe_float(row["validation_sample_sensitivity_mean"], -float("inf")),
                    -safe_float(row["min_alarm_windows"], float("inf")),
                    -safe_float(row["threshold"], float("inf")),
                    -safe_float(row["validation_false_alarm_rate_per_hour_mean"], float("inf")),
                    safe_float(row["validation_sample_specificity_mean"], -float("inf")),
                ),
            )
        elif selection_strategy == "balanced_event_far":
            selected = max(
                selection_pool,
                key=lambda row: (
                    safe_float(row["validation_sample_f1_mean"], -float("inf")),
                    safe_float(row["validation_sample_sensitivity_mean"], -float("inf")),
                    -safe_float(row["validation_false_alarm_rate_per_hour_mean"], float("inf")),
                    safe_float(row["validation_sample_specificity_mean"], -float("inf")),
                    -safe_float(row["validation_false_alarm_events_mean"], float("inf")),
                    safe_float(row["min_alarm_windows"], float("inf")),
                    safe_float(row["threshold"], -float("inf")),
                ),
            )
        else:
            raise ValueError(f"Unsupported selection_strategy: {selection_strategy}")
        selected["selection_status"] = "met_target_event_sensitivity"
        selected["selection_strategy"] = selection_strategy
        selected["validation_far_cap"] = validation_far_cap if validation_far_cap is not None else ""
        selected["validation_far_cap_met"] = int(selected in capped)
        return selected

    selected = min(
        candidates,
        key=lambda row: (
            -safe_float(row["validation_event_detected_mean"], -float("inf")),
            safe_float(row["validation_false_alarm_rate_per_hour_mean"], float("inf")),
            -safe_float(row["validation_sample_specificity_mean"], -float("inf")),
        ),
    )
    selected["selection_status"] = "target_not_met_best_available"
    selected["selection_strategy"] = selection_strategy
    selected["validation_far_cap"] = validation_far_cap if validation_far_cap is not None else ""
    selected["validation_far_cap_met"] = 0
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.05:0.95:0.05")
    parser.add_argument("--min-alarm-windows", default="1,2,3,4,5,6,8,10")
    parser.add_argument("--merge-gap-sec", type=int, default=7)
    parser.add_argument("--refractory-min", type=float, default=30.0)
    parser.add_argument("--window-sec", type=int, default=7)
    parser.add_argument("--target-event-sensitivity", type=float, default=0.95)
    parser.add_argument(
        "--score-smoothing-windows",
        type=int,
        default=1,
        help=(
            "Causal rolling mean length for score_preictal before event evaluation. "
            "A value of 1 disables smoothing."
        ),
    )
    parser.add_argument(
        "--score-normalization",
        default="none",
        choices=[
            "none",
            "validation_zscore_sigmoid",
            "validation_minmax",
            "validation_robust_minmax",
            "validation_ecdf",
            "validation_interictal_ecdf",
        ],
        help=(
            "Optional validation-only score transformation before threshold selection. "
            "Parameters are fit on val_predictions.csv within each fold, then applied "
            "to both validation and held-out test predictions."
        ),
    )
    parser.add_argument(
        "--selection-strategy",
        default="min_far",
        choices=["min_far", "max_sensitivity_under_far_cap", "balanced_event_far"],
        help=(
            "How to select alarm parameters on validation predictions. "
            "min_far matches the original behavior. max_sensitivity_under_far_cap "
            "prioritizes validation sample sensitivity among event-detecting candidates, "
            "optionally restricted by --validation-far-cap. balanced_event_far selects "
            "among event-detecting candidates by validation F1 first, then sensitivity "
            "and false-alarm rate."
        ),
    )
    parser.add_argument(
        "--validation-far-cap",
        type=float,
        default=None,
        help="Optional maximum validation false-alarm rate per hour used by sensitivity-first calibration.",
    )
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

    validation_sweep_rows = []
    calibrated_rows = []
    selected_rows = []
    skipped_rows = []

    for fold_dir in fold_dirs:
        val_prediction_path = fold_dir / "val_predictions.csv"
        test_prediction_path = fold_dir / "test_predictions.csv"
        if not val_prediction_path.exists():
            skipped_rows.append(
                {
                    "fold_id": fold_dir.name,
                    "subject": fold_dir.parent.name,
                    "reason": "missing val_predictions.csv",
                    "fold_dir": str(fold_dir),
                }
            )
            continue

        val_rows = smooth_prediction_rows(
            read_csv(val_prediction_path),
            smoothing_windows=args.score_smoothing_windows,
        )
        test_rows = smooth_prediction_rows(
            read_csv(test_prediction_path),
            smoothing_windows=args.score_smoothing_windows,
        )
        try:
            val_rows, test_rows, normalization_metadata = normalize_prediction_rows(
                val_rows=val_rows,
                test_rows=test_rows,
                method=args.score_normalization,
            )
        except ValueError as exc:
            skipped_rows.append(
                {
                    "fold_id": fold_dir.name,
                    "subject": fold_dir.parent.name,
                    "reason": str(exc),
                    "fold_dir": str(fold_dir),
                }
            )
            continue
        val_window_rows = read_csv(fold_dir / "val_windows.csv") if (fold_dir / "val_windows.csv").exists() else []
        test_window_rows = read_csv(fold_dir / "test_windows.csv") if (fold_dir / "test_windows.csv").exists() else []

        fold_val_sweep = []
        for min_alarm_windows in min_alarm_windows_values:
            for threshold in thresholds:
                row = evaluate_fold(
                    fold_dir=fold_dir,
                    pred_rows=val_rows,
                    window_rows=val_window_rows,
                    threshold=threshold,
                    merge_gap_sec=args.merge_gap_sec,
                    refractory_sec=refractory_sec,
                    window_sec=args.window_sec,
                    min_alarm_windows=min_alarm_windows,
                )
                fold_val_sweep.append(row)
                validation_sweep_rows.append(row)

        selected = select_from_validation(
            validation_rows=fold_val_sweep,
            target_event_sensitivity=args.target_event_sensitivity,
            selection_strategy=args.selection_strategy,
            validation_far_cap=args.validation_far_cap,
        )
        threshold = safe_float(selected["threshold"])
        min_alarm_windows = int(selected["min_alarm_windows"])
        test_eval = evaluate_fold(
            fold_dir=fold_dir,
            pred_rows=test_rows,
            window_rows=test_window_rows,
            threshold=threshold,
            merge_gap_sec=args.merge_gap_sec,
            refractory_sec=refractory_sec,
            window_sec=args.window_sec,
            min_alarm_windows=min_alarm_windows,
        )
        test_eval.update(
            {
                "selected_threshold": threshold,
                "selected_min_alarm_windows": min_alarm_windows,
                "selected_min_alarm_duration_sec": selected["min_alarm_duration_sec"],
                "selection_status": selected["selection_status"],
                "selection_strategy": selected["selection_strategy"],
                "validation_far_cap": selected["validation_far_cap"],
                "validation_far_cap_met": selected["validation_far_cap_met"],
                "validation_event_detected_mean": selected["validation_event_detected_mean"],
                "validation_false_alarm_rate_per_hour_mean": selected[
                    "validation_false_alarm_rate_per_hour_mean"
                ],
                "validation_sample_acc_mean": selected["validation_sample_acc_mean"],
                "validation_sample_f1_mean": selected["validation_sample_f1_mean"],
                "validation_sample_sensitivity_mean": selected["validation_sample_sensitivity_mean"],
                "validation_sample_specificity_mean": selected["validation_sample_specificity_mean"],
                **normalization_metadata,
            }
        )
        calibrated_rows.append(test_eval)
        selected_rows.append(
            {
                "fold_id": test_eval["fold_id"],
                "subject": test_eval["subject"],
                "selected_threshold": threshold,
                "selected_min_alarm_windows": min_alarm_windows,
                "selected_min_alarm_duration_sec": selected["min_alarm_duration_sec"],
                "selection_status": selected["selection_status"],
                "selection_strategy": selected["selection_strategy"],
                "validation_far_cap": selected["validation_far_cap"],
                "validation_far_cap_met": selected["validation_far_cap_met"],
                "validation_event_detected_mean": selected["validation_event_detected_mean"],
                "validation_false_alarm_rate_per_hour_mean": selected[
                    "validation_false_alarm_rate_per_hour_mean"
                ],
                **normalization_metadata,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "validation_parameter_sweep.csv", validation_sweep_rows)
    write_csv(args.out_dir / "validation_calibrated_fold_event_metrics.csv", calibrated_rows)
    write_csv(args.out_dir / "selected_operating_points.csv", selected_rows)
    write_csv(args.out_dir / "skipped_folds.csv", skipped_rows)

    per_subject_rows = []
    for subject in sorted({str(row["subject"]) for row in calibrated_rows}):
        subject_rows = [row for row in calibrated_rows if str(row["subject"]) == subject]
        per_subject_rows.append({"subject": subject, **summarize(subject_rows)})
    write_csv(args.out_dir / "validation_calibrated_per_subject_event_metrics.csv", per_subject_rows)

    summary = {
        "result_roots": [str(root) for root in args.result_roots],
        "folds_found": len(fold_dirs),
        "folds_evaluated": len(calibrated_rows),
        "folds_skipped": len(skipped_rows),
        "thresholds": thresholds,
        "min_alarm_windows": min_alarm_windows_values,
        "merge_gap_sec": args.merge_gap_sec,
        "refractory_min": args.refractory_min,
        "window_sec": args.window_sec,
        "target_event_sensitivity": args.target_event_sensitivity,
        "score_smoothing_windows": args.score_smoothing_windows,
        "score_normalization": args.score_normalization,
        "selection_strategy": args.selection_strategy,
        "validation_far_cap": args.validation_far_cap,
        "validation_calibrated": summarize(calibrated_rows),
        "outputs": {
            "validation_calibrated_fold_event_metrics": str(
                args.out_dir / "validation_calibrated_fold_event_metrics.csv"
            ),
            "selected_operating_points": str(args.out_dir / "selected_operating_points.csv"),
            "validation_calibrated_per_subject_event_metrics": str(
                args.out_dir / "validation_calibrated_per_subject_event_metrics.csv"
            ),
            "skipped_folds": str(args.out_dir / "skipped_folds.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
