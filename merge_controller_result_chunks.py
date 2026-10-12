#!/usr/bin/env python3
"""Merge fold-level learnable alarm controller result chunks.

The chunks are produced by independent subject/GPU jobs. This utility keeps
their row-level audit tables intact and recomputes the aggregate summary from
the merged fold table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    fold_rows: list[dict[str, str]] = []
    parameter_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []
    chunk_summaries = []
    for chunk in args.chunks:
        summary_path = chunk / "summary.json"
        if not summary_path.exists():
            raise SystemExit(f"Missing summary.json: {chunk}")
        chunk_summaries.append(json.loads(summary_path.read_text()))
        fold_rows.extend(read_rows(chunk / "learnable_alarm_controller_fold_event_metrics.csv"))
        parameter_rows.extend(read_rows(chunk / "learnable_alarm_controller_selected_params.csv"))
        skipped_rows.extend(read_rows(chunk / "skipped_folds.csv"))

    fold_rows.sort(key=lambda row: (row.get("subject", ""), row.get("fold_id", "")))
    parameter_rows.sort(key=lambda row: (row.get("subject", ""), row.get("fold_id", "")))
    skipped_rows.sort(key=lambda row: (row.get("subject", ""), row.get("fold_id", "")))
    fold_ids = [row.get("fold_id", "") for row in fold_rows]
    duplicate_ids = sorted({fold_id for fold_id in fold_ids if fold_ids.count(fold_id) > 1})
    if duplicate_ids:
        raise SystemExit(f"Duplicate fold IDs: {duplicate_ids[:10]}")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_chbmit_loocv_event_level import summarize

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "learnable_alarm_controller_fold_event_metrics.csv", fold_rows)
    write_rows(args.out_dir / "learnable_alarm_controller_selected_params.csv", parameter_rows)
    write_rows(args.out_dir / "skipped_folds.csv", skipped_rows)

    per_subject = []
    for subject in sorted({row["subject"] for row in fold_rows}):
        subject_rows = [row for row in fold_rows if row["subject"] == subject]
        per_subject.append({"subject": subject, **summarize(subject_rows)})
    write_rows(args.out_dir / "learnable_alarm_controller_per_subject_event_metrics.csv", per_subject)

    base = chunk_summaries[0] if chunk_summaries else {}
    summary = {
        "chunk_dirs": [str(path) for path in args.chunks],
        "result_roots": base.get("result_roots", []),
        "subjects": sorted({row["subject"] for row in fold_rows}),
        "folds_found": sum(int(item.get("folds_found", 0)) for item in chunk_summaries),
        "folds_evaluated": len(fold_rows),
        "folds_skipped": len(skipped_rows),
        "duplicate_fold_ids": duplicate_ids,
        "controller": base.get("controller"),
        "controller_variant": base.get("controller_variant"),
        "score_normalization": base.get("score_normalization"),
        "epochs": base.get("epochs"),
        "lr": base.get("lr"),
        "event_target_alarm": base.get("event_target_alarm"),
        "false_alarm_margin": base.get("false_alarm_margin"),
        "false_weight": base.get("false_weight"),
        "false_topk_weight": base.get("false_topk_weight"),
        "cooldown_decay_floor": base.get("cooldown_decay_floor"),
        "alarm_threshold": base.get("alarm_threshold"),
        "min_alarm_windows": base.get("min_alarm_windows"),
        "post_alarm_cooldown_min": base.get("post_alarm_cooldown_min"),
        "merge_gap_sec": base.get("merge_gap_sec"),
        "refractory_min": base.get("refractory_min"),
        "window_sec": base.get("window_sec"),
        "seed": base.get("seed"),
        "seed_skip_controller_inits": 0,
        "learnable_alarm_controller": summarize(fold_rows),
        "outputs": {
            "fold_event_metrics": str(args.out_dir / "learnable_alarm_controller_fold_event_metrics.csv"),
            "selected_params": str(args.out_dir / "learnable_alarm_controller_selected_params.csv"),
            "per_subject_event_metrics": str(args.out_dir / "learnable_alarm_controller_per_subject_event_metrics.csv"),
            "skipped_folds": str(args.out_dir / "skipped_folds.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
