#!/usr/bin/env python3
"""Select multiscale DSF fusion weights from validation streams per fold.

This script creates one fused prediction stream per fold. It tries a small set
of prespecified 5s/7s/9s weight candidates, selects the candidate on validation
event-level behavior, writes the selected raw fused val/test predictions, and
also reports the selected candidate's non-controller test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from evaluate_chbmit_loocv_event_level import evaluate_fold, parse_thresholds, read_csv, safe_float, summarize, write_csv
from evaluate_chbmit_loocv_validation_calibrated_event_level import normalize_prediction_rows, select_from_validation
from fuse_multiscale_dsf_predictions import fuse_split


DEFAULT_CANDIDATES: dict[str, dict[str, float]] = {
    "7only": {"5s": 0.0, "7s": 1.0, "9s": 0.0},
    "equal": {"5s": 1.0, "7s": 1.0, "9s": 1.0},
    "7heavy": {"5s": 0.5, "7s": 2.0, "9s": 0.5},
    "5_7": {"5s": 1.0, "7s": 1.0, "9s": 0.0},
    "7_9": {"5s": 0.0, "7s": 1.0, "9s": 1.0},
    "5heavy": {"5s": 2.0, "7s": 1.0, "9s": 0.5},
    "9heavy": {"5s": 0.5, "7s": 1.0, "9s": 2.0},
    "5_7heavy": {"5s": 1.0, "7s": 2.0, "9s": 0.0},
    "7_9heavy": {"5s": 0.0, "7s": 2.0, "9s": 1.0},
}


def parse_scale_spec(raw: str) -> tuple[str, list[Path]]:
    if ":" not in raw:
        raise ValueError(f"Expected SCALE:ROOT[,ROOT...] spec, got {raw!r}")
    scale, roots = raw.split(":", 1)
    paths = [Path(item) for item in roots.split(",") if item]
    if not scale or not paths:
        raise ValueError(f"Invalid scale spec: {raw!r}")
    return scale, paths


def parse_candidates(raw: str) -> dict[str, dict[str, float]]:
    if not raw:
        return DEFAULT_CANDIDATES
    candidates: dict[str, dict[str, float]] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        name, weights_raw = item.split("=", 1)
        weights: dict[str, float] = {}
        for pair in weights_raw.split(","):
            scale, value = pair.split(":", 1)
            weights[scale.strip()] = float(value)
        candidates[name.strip()] = weights
    if not candidates:
        raise ValueError("No valid candidates parsed.")
    return candidates


def scan_fold_dirs(roots: list[Path]) -> dict[str, Path]:
    folds: dict[str, Path] = {}
    for root in roots:
        for pred_path in root.glob("*/*/test_predictions.csv"):
            folds[pred_path.parent.name] = pred_path.parent
    return folds


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def select_candidate(candidate_rows: list[dict[str, object]], target_event_sensitivity: float) -> dict[str, object]:
    eligible = [
        row for row in candidate_rows
        if safe_float(row.get("validation_event_detected_mean")) >= target_event_sensitivity
    ]
    if eligible:
        return max(
            eligible,
            key=lambda row: (
                safe_float(row.get("validation_sample_f1_mean"), -float("inf")),
                safe_float(row.get("validation_sample_sensitivity_mean"), -float("inf")),
                -safe_float(row.get("validation_false_alarm_rate_per_hour_mean"), float("inf")),
                safe_float(row.get("validation_sample_specificity_mean"), -float("inf")),
                -safe_float(row.get("validation_false_alarm_events_mean"), float("inf")),
            ),
        )
    return max(
        candidate_rows,
        key=lambda row: (
            safe_float(row.get("validation_event_detected_mean"), -float("inf")),
            -safe_float(row.get("validation_false_alarm_rate_per_hour_mean"), float("inf")),
            safe_float(row.get("validation_sample_specificity_mean"), -float("inf")),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="SCALE:ROOT[,ROOT...] entries.")
    parser.add_argument("--base-scale", default="7s")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidates", default="", help="Optional 'name=5s:1,7s:1,9s:1;...' override.")
    parser.add_argument("--thresholds", default="0.05:0.95:0.05")
    parser.add_argument("--min-alarm-windows", default="1,2,3,4,5,6,8,10")
    parser.add_argument("--merge-gap-sec", type=int, default=7)
    parser.add_argument("--refractory-min", type=float, default=30.0)
    parser.add_argument("--window-sec", type=int, default=7)
    parser.add_argument("--score-normalization", default="validation_interictal_ecdf")
    parser.add_argument("--target-event-sensitivity", type=float, default=0.95)
    parser.add_argument("--selection-strategy", default="balanced_event_far")
    args = parser.parse_args()

    scale_roots = dict(parse_scale_spec(item) for item in args.inputs)
    candidates = parse_candidates(args.candidates)
    if args.base_scale not in scale_roots:
        raise SystemExit(f"Base scale {args.base_scale!r} missing from inputs.")

    fold_dirs_by_scale = {scale: scan_fold_dirs(roots) for scale, roots in scale_roots.items()}
    base_folds = fold_dirs_by_scale[args.base_scale]
    thresholds = parse_thresholds(args.thresholds)
    min_alarm_windows_values = [int(item) for item in args.min_alarm_windows.split(",") if item.strip()]
    refractory_sec = int(round(args.refractory_min * 60))

    selected_rows = []
    candidate_sweep_rows = []
    selected_test_metric_rows = []
    skipped_rows = []

    for fold_id, base_fold_dir in sorted(base_folds.items()):
        subject = base_fold_dir.parent.name
        missing = [scale for scale, folds in fold_dirs_by_scale.items() if fold_id not in folds]
        if missing:
            skipped_rows.append({"fold_id": fold_id, "subject": subject, "reason": f"missing scales: {','.join(missing)}"})
            continue
        try:
            raw_by_split = {"val": {}, "test": {}}
            for scale, fold_dirs in fold_dirs_by_scale.items():
                raw_by_split["val"][scale] = read_csv(fold_dirs[fold_id] / "val_predictions.csv")
                raw_by_split["test"][scale] = read_csv(fold_dirs[fold_id] / "test_predictions.csv")

            candidate_payload: dict[str, dict[str, object]] = {}
            candidate_selection_rows = []
            for name, weights in candidates.items():
                candidate_weights = {scale: weights.get(scale, 0.0) for scale in scale_roots}
                fused_val_raw = fuse_split(
                    base_rows=raw_by_split["val"][args.base_scale],
                    scale_rows=raw_by_split["val"],
                    base_scale=args.base_scale,
                    scale_weights=candidate_weights,
                )
                fused_test_raw = fuse_split(
                    base_rows=raw_by_split["test"][args.base_scale],
                    scale_rows=raw_by_split["test"],
                    base_scale=args.base_scale,
                    scale_weights=candidate_weights,
                )
                fused_val_norm, fused_test_norm, normalization_metadata = normalize_prediction_rows(
                    fused_val_raw,
                    fused_test_raw,
                    method=args.score_normalization,
                )
                fold_val_sweep = []
                for min_alarm_windows in min_alarm_windows_values:
                    for threshold in thresholds:
                        row = evaluate_fold(
                            fold_dir=base_fold_dir,
                            pred_rows=fused_val_norm,
                            window_rows=read_csv(base_fold_dir / "val_windows.csv") if (base_fold_dir / "val_windows.csv").exists() else [],
                            threshold=threshold,
                            merge_gap_sec=args.merge_gap_sec,
                            refractory_sec=refractory_sec,
                            window_sec=args.window_sec,
                            min_alarm_windows=min_alarm_windows,
                        )
                        row.update({"candidate": name, "weights": json.dumps(candidate_weights, sort_keys=True)})
                        fold_val_sweep.append(row)
                        candidate_sweep_rows.append(row)
                selected = select_from_validation(
                    validation_rows=fold_val_sweep,
                    target_event_sensitivity=args.target_event_sensitivity,
                    selection_strategy=args.selection_strategy,
                    validation_far_cap=None,
                )
                test_eval = evaluate_fold(
                    fold_dir=base_fold_dir,
                    pred_rows=fused_test_norm,
                    window_rows=read_csv(base_fold_dir / "test_windows.csv") if (base_fold_dir / "test_windows.csv").exists() else [],
                    threshold=float(selected["threshold"]),
                    merge_gap_sec=args.merge_gap_sec,
                    refractory_sec=refractory_sec,
                    window_sec=args.window_sec,
                    min_alarm_windows=int(selected["min_alarm_windows"]),
                )
                selected_record = {
                    "fold_id": fold_id,
                    "subject": subject,
                    "candidate": name,
                    "weights": json.dumps(candidate_weights, sort_keys=True),
                    "selected_threshold": selected["threshold"],
                    "selected_min_alarm_windows": selected["min_alarm_windows"],
                    "selection_status": selected["selection_status"],
                    "validation_event_detected_mean": selected["validation_event_detected_mean"],
                    "validation_false_alarm_rate_per_hour_mean": selected["validation_false_alarm_rate_per_hour_mean"],
                    "validation_false_alarm_events_mean": selected["validation_false_alarm_events_mean"],
                    "validation_sample_f1_mean": selected["validation_sample_f1_mean"],
                    "validation_sample_sensitivity_mean": selected["validation_sample_sensitivity_mean"],
                    "validation_sample_specificity_mean": selected["validation_sample_specificity_mean"],
                    "test_event_detected": test_eval["event_detected"],
                    "test_false_alarm_rate_per_hour": test_eval["false_alarm_rate_per_hour"],
                    "test_false_alarm_events": test_eval["false_alarm_events"],
                    "test_event_warning_time_min": test_eval["event_warning_time_min"],
                    **normalization_metadata,
                }
                candidate_selection_rows.append(selected_record)
                candidate_payload[name] = {
                    "weights": candidate_weights,
                    "fused_val_raw": fused_val_raw,
                    "fused_test_raw": fused_test_raw,
                    "fused_test_norm": fused_test_norm,
                    "selected": selected_record,
                    "test_eval": test_eval,
                }

            chosen = select_candidate(candidate_selection_rows, args.target_event_sensitivity)
            chosen_name = str(chosen["candidate"])
            payload = candidate_payload[chosen_name]
            out_fold_dir = args.out_dir / subject / fold_id
            write_csv(out_fold_dir / "val_predictions.csv", payload["fused_val_raw"])  # type: ignore[arg-type]
            write_csv(out_fold_dir / "test_predictions.csv", payload["fused_test_raw"])  # type: ignore[arg-type]
            copy_if_exists(base_fold_dir / "val_windows.csv", out_fold_dir / "val_windows.csv")
            copy_if_exists(base_fold_dir / "test_windows.csv", out_fold_dir / "test_windows.csv")
            copy_if_exists(base_fold_dir / "train_windows.csv", out_fold_dir / "train_windows.csv")

            selected_out = dict(payload["selected"])  # type: ignore[arg-type]
            selected_out["chosen_by_weight_selector"] = 1
            selected_out["out_dir"] = str(out_fold_dir)
            selected_rows.append(selected_out)
            test_eval = dict(payload["test_eval"])  # type: ignore[arg-type]
            test_eval.update(
                {
                    "candidate": chosen_name,
                    "weights": json.dumps(payload["weights"], sort_keys=True),  # type: ignore[arg-type]
                    "selected_threshold": selected_out["selected_threshold"],
                    "selected_min_alarm_windows": selected_out["selected_min_alarm_windows"],
                    "selection_status": selected_out["selection_status"],
                }
            )
            selected_test_metric_rows.append(test_eval)
        except Exception as exc:
            skipped_rows.append({"fold_id": fold_id, "subject": subject, "reason": str(exc)})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "selected_fusion_weights.csv", selected_rows)
    write_csv(args.out_dir / "candidate_validation_sweep.csv", candidate_sweep_rows)
    write_csv(args.out_dir / "selected_weight_fold_event_metrics.csv", selected_test_metric_rows)
    write_csv(args.out_dir / "skipped_folds.csv", skipped_rows)
    per_subject_rows = []
    for subject in sorted({str(row["subject"]) for row in selected_test_metric_rows}):
        subject_rows = [row for row in selected_test_metric_rows if str(row["subject"]) == subject]
        per_subject_rows.append({"subject": subject, **summarize(subject_rows)})
    write_csv(args.out_dir / "selected_weight_per_subject_event_metrics.csv", per_subject_rows)

    candidate_counts: dict[str, int] = {}
    for row in selected_rows:
        candidate_counts[str(row["candidate"])] = candidate_counts.get(str(row["candidate"]), 0) + 1

    summary = {
        "inputs": {scale: [str(root) for root in roots] for scale, roots in scale_roots.items()},
        "base_scale": args.base_scale,
        "candidates": candidates,
        "thresholds": thresholds,
        "min_alarm_windows": min_alarm_windows_values,
        "score_normalization": args.score_normalization,
        "selection_strategy": args.selection_strategy,
        "target_event_sensitivity": args.target_event_sensitivity,
        "folds_evaluated": len(selected_test_metric_rows),
        "folds_skipped": len(skipped_rows),
        "candidate_counts": candidate_counts,
        "selected_weight_validation_calibrated": summarize(selected_test_metric_rows),
        "outputs": {
            "selected_fusion_weights": str(args.out_dir / "selected_fusion_weights.csv"),
            "selected_weight_fold_event_metrics": str(args.out_dir / "selected_weight_fold_event_metrics.csv"),
            "candidate_validation_sweep": str(args.out_dir / "candidate_validation_sweep.csv"),
            "skipped_folds": str(args.out_dir / "skipped_folds.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
