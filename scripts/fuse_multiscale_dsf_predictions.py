#!/usr/bin/env python3
"""Fuse CHB-MIT fold predictions from different window lengths on one timeline.

The output keeps a base scale's prediction rows/windows (7s for the current
main protocol) and replaces ``score_preictal`` with an equal-weighted mean of
available scale scores. Non-base scales are mapped to each base row by
overlap-weighted averaging within the same EDF file.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_scale_spec(raw: str) -> tuple[str, list[Path]]:
    if ":" not in raw:
        raise ValueError(f"Expected SCALE:ROOT[,ROOT...] spec, got {raw!r}")
    name, roots = raw.split(":", 1)
    paths = [Path(item) for item in roots.split(",") if item]
    if not name or not paths:
        raise ValueError(f"Invalid scale spec: {raw!r}")
    return name, paths


def scan_fold_dirs(roots: list[Path]) -> dict[str, Path]:
    folds: dict[str, Path] = {}
    for root in roots:
        for pred_path in root.glob("*/*/test_predictions.csv"):
            fold_dir = pred_path.parent
            folds[fold_dir.name] = fold_dir
    return folds


def rows_by_file(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["file_name"], []).append(row)
    for file_rows in grouped.values():
        file_rows.sort(key=lambda r: (float(r["window_start_sec"]), float(r["window_end_sec"])))
    return grouped


def overlap_weighted_score(
    target: dict[str, str],
    grouped_rows: dict[str, list[dict[str, str]]],
) -> tuple[float | None, int, float]:
    target_start = float(target["window_start_sec"])
    target_end = float(target["window_end_sec"])
    total_weight = 0.0
    weighted_score = 0.0
    overlaps = 0

    for row in grouped_rows.get(target["file_name"], []):
        row_start = float(row["window_start_sec"])
        row_end = float(row["window_end_sec"])
        if row_end <= target_start:
            continue
        if row_start >= target_end:
            break
        weight = max(0.0, min(target_end, row_end) - max(target_start, row_start))
        if weight <= 0:
            continue
        weighted_score += weight * float(row["score_preictal"])
        total_weight += weight
        overlaps += 1

    if total_weight <= 0:
        return None, 0, 0.0
    return weighted_score / total_weight, overlaps, total_weight


def fuse_split(
    base_rows: list[dict[str, str]],
    scale_rows: dict[str, list[dict[str, str]]],
    base_scale: str,
    scale_weights: dict[str, float],
) -> list[dict[str, object]]:
    grouped_by_scale = {
        scale: rows_by_file(rows)
        for scale, rows in scale_rows.items()
        if scale != base_scale
    }
    fused_rows: list[dict[str, object]] = []
    for index, base_row in enumerate(base_rows):
        row = dict(base_row)
        weighted_sum = float(row["score_preictal"]) * scale_weights.get(base_scale, 1.0)
        weight_sum = scale_weights.get(base_scale, 1.0)
        available = [base_scale]
        row[f"score_{base_scale}"] = float(row["score_preictal"])
        row[f"overlap_windows_{base_scale}"] = 1
        row[f"overlap_seconds_{base_scale}"] = float(row["window_end_sec"]) - float(row["window_start_sec"])

        for scale, grouped_rows in grouped_by_scale.items():
            score, overlap_count, overlap_seconds = overlap_weighted_score(base_row, grouped_rows)
            row[f"score_{scale}"] = "" if score is None else score
            row[f"overlap_windows_{scale}"] = overlap_count
            row[f"overlap_seconds_{scale}"] = overlap_seconds
            if score is None:
                continue
            weight = scale_weights.get(scale, 1.0)
            weighted_sum += score * weight
            weight_sum += weight
            available.append(scale)

        row["index"] = index
        row["score_preictal"] = weighted_sum / max(weight_sum, 1e-12)
        row["fusion_available_scales"] = ",".join(available)
        row["fusion_available_scale_count"] = len(available)
        fused_rows.append(row)
    return fused_rows


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="SCALE:ROOT[,ROOT...] entries.")
    parser.add_argument("--base-scale", default="7s")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weights", default="", help="Optional comma list like 5s=1,7s=1,9s=1.")
    args = parser.parse_args()

    scale_roots = dict(parse_scale_spec(item) for item in args.inputs)
    if args.base_scale not in scale_roots:
        raise SystemExit(f"Base scale {args.base_scale!r} missing from --inputs.")

    scale_weights = {scale: 1.0 for scale in scale_roots}
    if args.weights:
        for item in args.weights.split(","):
            if not item:
                continue
            scale, value = item.split("=", 1)
            scale_weights[scale] = float(value)

    fold_dirs_by_scale = {scale: scan_fold_dirs(roots) for scale, roots in scale_roots.items()}
    base_folds = fold_dirs_by_scale[args.base_scale]
    summary_rows = []
    skipped_rows = []

    for fold_id, base_fold_dir in sorted(base_folds.items()):
        missing = [scale for scale, folds in fold_dirs_by_scale.items() if fold_id not in folds]
        subject = base_fold_dir.parent.name
        out_fold_dir = args.out_dir / subject / fold_id
        if missing:
            skipped_rows.append({"fold_id": fold_id, "subject": subject, "reason": f"missing scales: {','.join(missing)}"})
            continue

        for split in ("val", "test"):
            base_path = base_fold_dir / f"{split}_predictions.csv"
            if not base_path.exists():
                skipped_rows.append({"fold_id": fold_id, "subject": subject, "reason": f"missing base {split}_predictions.csv"})
                continue
            prediction_rows = {
                scale: read_csv(fold_dirs_by_scale[scale][fold_id] / f"{split}_predictions.csv")
                for scale in scale_roots
            }
            fused_rows = fuse_split(
                base_rows=prediction_rows[args.base_scale],
                scale_rows=prediction_rows,
                base_scale=args.base_scale,
                scale_weights=scale_weights,
            )
            write_csv(out_fold_dir / f"{split}_predictions.csv", fused_rows)
            copy_if_exists(base_fold_dir / f"{split}_windows.csv", out_fold_dir / f"{split}_windows.csv")

        copy_if_exists(base_fold_dir / "train_windows.csv", out_fold_dir / "train_windows.csv")
        summary_rows.append(
            {
                "fold_id": fold_id,
                "subject": subject,
                "base_scale": args.base_scale,
                "scales": ",".join(scale_roots),
                "weights": json.dumps(scale_weights, sort_keys=True),
                "out_dir": str(out_fold_dir),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "fusion_fold_summary.csv", summary_rows)
    write_csv(args.out_dir / "skipped_folds.csv", skipped_rows)
    summary = {
        "base_scale": args.base_scale,
        "inputs": {scale: [str(root) for root in roots] for scale, roots in scale_roots.items()},
        "weights": scale_weights,
        "folds_written": len(summary_rows),
        "folds_skipped": len(skipped_rows),
        "outputs": {
            "fusion_fold_summary": str(args.out_dir / "fusion_fold_summary.csv"),
            "skipped_folds": str(args.out_dir / "skipped_folds.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
