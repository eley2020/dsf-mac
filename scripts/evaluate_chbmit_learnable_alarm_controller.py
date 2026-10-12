#!/usr/bin/env python3
"""Learnable validation-time alarm controller for CHB-MIT LOOCV predictions.

This script keeps the EEG model fixed. For each fold it trains a small
differentiable alarm controller on validation prediction streams, then applies
the learned controller once to the held-out test stream. The controller learns
patient/fold-specific thresholding, score accumulation, and soft cooldown from
validation data instead of selecting threshold/min-alarm rules by grid search.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from evaluate_chbmit_loocv_event_level import (
    evaluate_fold,
    read_csv,
    safe_float,
    sequence_key,
    time_end,
    time_start,
    summarize,
    write_csv,
)
from evaluate_chbmit_loocv_validation_calibrated_event_level import normalize_prediction_rows


def group_indices(rows: list[dict[str, str]]) -> list[list[int]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row.get("subject", ""), sequence_key(row))].append(index)
    groups = []
    for indices in grouped.values():
        groups.append(
            sorted(
                indices,
                key=lambda i: (
                    time_start(rows[i]),
                    time_end(rows[i]),
                    i,
                ),
            )
        )
    return groups


def event_groups(rows: list[dict[str, str]]) -> list[list[int]]:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if int(row.get("label", 0)) != 1:
            continue
        event_id = row.get("matched_seizure_index_in_file", "").strip() or "unknown"
        seizure_file = row.get("matched_seizure_file_name", "").strip() or row["file_name"]
        grouped[(row.get("subject", ""), seizure_file, event_id)].append(index)
    return [
        sorted(
            indices,
            key=lambda i: (
                time_start(rows[i]),
                time_end(rows[i]),
                i,
            ),
        )
        for indices in grouped.values()
    ]


class LearnableAlarmController(nn.Module):
    """Tiny causal controller over a risk-score sequence.

    Variants let us ablate whether the gain comes from a learned threshold
    alone, temporal evidence accumulation, or the soft cooldown gate. The
    multi-timescale variant keeps several causal evidence traces with learned
    decay rates before applying the same cooldown gate. The tiny-TCN variant
    uses causal dilated convolutions over recent risk scores, with a residual
    raw-score path so it starts close to the fixed risk estimator. The gated
    TCN-suppression variant keeps the multi-timescale controller as the base
    alarm path and lets a small causal TCN only suppress alarms. The hard
    cooldown-update variant keeps the multi-timescale alarm path but only lets
    near-confirmed alarms update the cooldown state.
    """

    def __init__(self, variant: str, cooldown_decay_floor: float = 0.0) -> None:
        super().__init__()
        if variant not in {
            "threshold",
            "threshold_accum",
            "threshold_accum_cooldown",
            "multi_timescale_cooldown",
            "multi_timescale_hard_cooldown_update",
            "tiny_tcn_cooldown",
            "multi_timescale_tcn_suppression_cooldown",
        }:
            raise ValueError(f"Unsupported controller variant: {variant}")
        if not 0.0 <= cooldown_decay_floor < 1.0:
            raise ValueError("--cooldown-decay-floor must be in [0, 1).")
        self.variant = variant
        self.cooldown_decay_floor = float(cooldown_decay_floor)
        self.theta_logit = nn.Parameter(torch.tensor(1.3862944))  # sigmoid ~= 0.80
        self.alpha_logit = nn.Parameter(torch.tensor(0.8472979))  # sigmoid ~= 0.70
        self.scale_log = nn.Parameter(torch.tensor(2.0))
        self.cooldown_decay_logit = nn.Parameter(torch.tensor(2.1972246))  # sigmoid ~= 0.90
        self.cooldown_strength_log = nn.Parameter(torch.tensor(1.0))
        self.cooldown_threshold_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid ~= 0.50
        self.cooldown_update_threshold_logit = nn.Parameter(torch.tensor(-0.4054651))  # sigmoid ~= 0.40
        self.cooldown_update_strength_log = nn.Parameter(torch.tensor(2.0))
        # Initial alphas roughly correspond to short / medium / long memory.
        self.multi_alpha_logit = nn.Parameter(torch.tensor([-0.6190392, 0.8472979, 2.9444390]))
        self.multi_evidence_weight_logit = nn.Parameter(torch.zeros(3))
        self.tcn_hidden_channels = 4
        self.tcn_kernel_size = 3
        self.tcn_dilations = (1, 2, 4)
        self.tcn_convs = nn.ModuleList(
            [
                nn.Conv1d(1, self.tcn_hidden_channels, kernel_size=self.tcn_kernel_size, dilation=self.tcn_dilations[0]),
                nn.Conv1d(
                    self.tcn_hidden_channels,
                    self.tcn_hidden_channels,
                    kernel_size=self.tcn_kernel_size,
                    dilation=self.tcn_dilations[1],
                ),
                nn.Conv1d(
                    self.tcn_hidden_channels,
                    self.tcn_hidden_channels,
                    kernel_size=self.tcn_kernel_size,
                    dilation=self.tcn_dilations[2],
                ),
            ]
        )
        self.tcn_out = nn.Conv1d(self.tcn_hidden_channels, 1, kernel_size=1)
        self.tcn_suppress_out = nn.Conv1d(self.tcn_hidden_channels, 1, kernel_size=1)
        self.tcn_delta_scale_log = nn.Parameter(torch.tensor(-1.0))
        self.tcn_max_suppression_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid ~= 0.50
        nn.init.zeros_(self.tcn_out.weight)
        nn.init.zeros_(self.tcn_out.bias)
        nn.init.zeros_(self.tcn_suppress_out.weight)
        nn.init.constant_(self.tcn_suppress_out.bias, -4.0)

    def cooldown_decay(self) -> torch.Tensor:
        raw_decay = torch.sigmoid(self.cooldown_decay_logit)
        if self.cooldown_decay_floor <= 0.0:
            return raw_decay
        return self.cooldown_decay_floor + (1.0 - self.cooldown_decay_floor) * raw_decay

    def hard_cooldown_update(self, alarm: torch.Tensor) -> torch.Tensor:
        update_threshold = torch.sigmoid(self.cooldown_update_threshold_logit)
        update_strength = F.softplus(self.cooldown_update_strength_log) + 5.0
        update_gate = torch.sigmoid(update_strength * (alarm - update_threshold))
        return alarm * update_gate

    def tcn_receptive_field_windows(self) -> int:
        return 1 + (self.tcn_kernel_size - 1) * sum(self.tcn_dilations)

    def tcn_features(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return scores.new_empty((1, self.tcn_hidden_channels, 0))
        x = scores.reshape(1, 1, -1)
        for conv in self.tcn_convs:
            pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
            x = F.pad(x, (pad, 0))
            x = F.relu(conv(x))
        return x

    def tcn_decision_evidence(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return scores.new_empty((0,))
        x = self.tcn_features(scores)
        delta = self.tcn_out(x).reshape(-1)
        raw_logit = torch.logit(scores.clamp(1e-4, 1.0 - 1e-4))
        delta_scale = F.softplus(self.tcn_delta_scale_log)
        return torch.sigmoid(raw_logit + delta_scale * delta)

    def tcn_suppression_gate(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return scores.new_empty((0,))
        x = self.tcn_features(scores)
        suppress_prob = torch.sigmoid(self.tcn_suppress_out(x).reshape(-1))
        max_suppression = torch.sigmoid(self.tcn_max_suppression_logit)
        return 1.0 - max_suppression * suppress_prob

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        theta = torch.sigmoid(self.theta_logit)
        alpha = torch.sigmoid(self.alpha_logit)
        multi_alpha = torch.sigmoid(self.multi_alpha_logit).to(dtype=scores.dtype, device=scores.device)
        multi_weights = torch.softmax(
            self.multi_evidence_weight_logit.to(dtype=scores.dtype, device=scores.device),
            dim=0,
        )
        scale = F.softplus(self.scale_log) + 1.0
        cooldown_decay = self.cooldown_decay()
        cooldown_strength = F.softplus(self.cooldown_strength_log) + 1.0
        cooldown_threshold = torch.sigmoid(self.cooldown_threshold_logit)
        tcn_suppression_gate = (
            self.tcn_suppression_gate(scores)
            if self.variant == "multi_timescale_tcn_suppression_cooldown"
            else None
        )
        if self.variant == "tiny_tcn_cooldown":
            decision_evidence_series = self.tcn_decision_evidence(scores)
            cooldown = scores.new_tensor(0.0)
            alarms = []
            for decision_evidence in decision_evidence_series:
                base_alarm = torch.sigmoid(scale * (decision_evidence - theta))
                cooldown_gate = torch.sigmoid(cooldown_strength * (cooldown_threshold - cooldown))
                alarm = base_alarm * cooldown_gate
                alarms.append(alarm)
                cooldown = cooldown_decay * cooldown + alarm
            return torch.stack(alarms) if alarms else scores.new_empty((0,))

        evidence = scores.new_tensor(0.0)
        multi_evidence = scores.new_zeros(multi_alpha.shape)
        cooldown = scores.new_tensor(0.0)
        alarms = []
        for step, score in enumerate(scores):
            if self.variant == "threshold":
                evidence = score
                decision_evidence = evidence
            elif self.variant in {
                "multi_timescale_cooldown",
                "multi_timescale_hard_cooldown_update",
                "multi_timescale_tcn_suppression_cooldown",
            }:
                multi_evidence = multi_alpha * multi_evidence + (1.0 - multi_alpha) * score
                decision_evidence = torch.sum(multi_weights * multi_evidence)
            else:
                evidence = alpha * evidence + (1.0 - alpha) * score
                decision_evidence = evidence
            base_alarm = torch.sigmoid(scale * (decision_evidence - theta))
            if self.variant in {
                "threshold_accum_cooldown",
                "multi_timescale_cooldown",
                "multi_timescale_hard_cooldown_update",
                "multi_timescale_tcn_suppression_cooldown",
            }:
                cooldown_gate = torch.sigmoid(cooldown_strength * (cooldown_threshold - cooldown))
                alarm = base_alarm * cooldown_gate
                if tcn_suppression_gate is not None:
                    alarm = alarm * tcn_suppression_gate[step]
            else:
                alarm = base_alarm
            alarms.append(alarm)
            if self.variant in {
                "threshold_accum_cooldown",
                "multi_timescale_cooldown",
                "multi_timescale_hard_cooldown_update",
                "multi_timescale_tcn_suppression_cooldown",
            }:
                if self.variant == "multi_timescale_hard_cooldown_update":
                    cooldown_update = self.hard_cooldown_update(alarm)
                else:
                    cooldown_update = alarm
                cooldown = cooldown_decay * cooldown + cooldown_update
        return torch.stack(alarms) if alarms else scores.new_empty((0,))

    def params_dict(self) -> dict[str, float | int | str]:
        with torch.no_grad():
            multi_alpha = torch.sigmoid(self.multi_alpha_logit)
            multi_weights = torch.softmax(self.multi_evidence_weight_logit, dim=0)
            tcn_param_count = sum(parameter.numel() for parameter in self.tcn_convs.parameters())
            tcn_param_count += sum(parameter.numel() for parameter in self.tcn_out.parameters())
            tcn_suppress_param_count = sum(parameter.numel() for parameter in self.tcn_convs.parameters())
            tcn_suppress_param_count += sum(parameter.numel() for parameter in self.tcn_suppress_out.parameters())
            return {
                "controller_variant": self.variant,
                "uses_accumulation": int(self.variant != "threshold"),
                "uses_cooldown": int(
                    self.variant
                    in {
                        "threshold_accum_cooldown",
                        "multi_timescale_cooldown",
                        "multi_timescale_hard_cooldown_update",
                        "tiny_tcn_cooldown",
                        "multi_timescale_tcn_suppression_cooldown",
                    }
                ),
                "uses_multi_timescale_evidence": int(
                    self.variant
                    in {
                        "multi_timescale_cooldown",
                        "multi_timescale_hard_cooldown_update",
                        "multi_timescale_tcn_suppression_cooldown",
                    }
                ),
                "uses_hard_cooldown_update": int(self.variant == "multi_timescale_hard_cooldown_update"),
                "uses_tcn_history": int(
                    self.variant in {"tiny_tcn_cooldown", "multi_timescale_tcn_suppression_cooldown"}
                ),
                "uses_tcn_suppression": int(self.variant == "multi_timescale_tcn_suppression_cooldown"),
                "cooldown_decay_floor": self.cooldown_decay_floor,
                "learned_threshold": float(torch.sigmoid(self.theta_logit).item()),
                "learned_accumulation_alpha": float(torch.sigmoid(self.alpha_logit).item()),
                "learned_multiscale_alpha_short": float(multi_alpha[0].item()),
                "learned_multiscale_alpha_mid": float(multi_alpha[1].item()),
                "learned_multiscale_alpha_long": float(multi_alpha[2].item()),
                "learned_multiscale_weight_short": float(multi_weights[0].item()),
                "learned_multiscale_weight_mid": float(multi_weights[1].item()),
                "learned_multiscale_weight_long": float(multi_weights[2].item()),
                "learned_scale": float((F.softplus(self.scale_log) + 1.0).item()),
                "learned_cooldown_decay": float(self.cooldown_decay().item()),
                "learned_cooldown_decay_raw": float(torch.sigmoid(self.cooldown_decay_logit).item()),
                "learned_cooldown_strength": float((F.softplus(self.cooldown_strength_log) + 1.0).item()),
                "learned_cooldown_threshold": float(torch.sigmoid(self.cooldown_threshold_logit).item()),
                "learned_cooldown_update_threshold": float(torch.sigmoid(self.cooldown_update_threshold_logit).item()),
                "learned_cooldown_update_strength": float(
                    (F.softplus(self.cooldown_update_strength_log) + 5.0).item()
                ),
                "learned_tcn_delta_scale": float(F.softplus(self.tcn_delta_scale_log).item()),
                "learned_tcn_max_suppression": float(torch.sigmoid(self.tcn_max_suppression_logit).item()),
                "learned_tcn_suppression_bias": float(self.tcn_suppress_out.bias.detach().mean().item()),
                "tcn_hidden_channels": self.tcn_hidden_channels,
                "tcn_kernel_size": self.tcn_kernel_size,
                "tcn_dilations": "/".join(str(item) for item in self.tcn_dilations),
                "tcn_receptive_field_windows": self.tcn_receptive_field_windows(),
                "tcn_param_count": tcn_param_count,
                "tcn_suppress_param_count": tcn_suppress_param_count,
            }


def controller_outputs(
    rows: list[dict[str, str]],
    controller: LearnableAlarmController,
    device: torch.device,
) -> torch.Tensor:
    out = torch.zeros(len(rows), dtype=torch.float32, device=device)
    for indices in group_indices(rows):
        scores = torch.tensor(
            [safe_float(rows[i].get("score_preictal"), 0.0) for i in indices],
            dtype=torch.float32,
            device=device,
        )
        out[torch.tensor(indices, dtype=torch.long, device=device)] = controller(scores)
    return out


def train_controller(
    rows: list[dict[str, str]],
    device: torch.device,
    epochs: int,
    lr: float,
    event_target_alarm: float,
    false_alarm_margin: float,
    false_weight: float,
    false_topk_weight: float,
    controller_variant: str,
    cooldown_decay_floor: float,
    cooldown_decay_reg_target: float,
    cooldown_decay_reg_weight: float,
) -> tuple[LearnableAlarmController, dict[str, float]]:
    controller = LearnableAlarmController(
        variant=controller_variant,
        cooldown_decay_floor=cooldown_decay_floor,
    ).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
    labels = torch.tensor([int(row.get("label", 0)) for row in rows], dtype=torch.float32, device=device)
    neg_mask = labels == 0
    events = event_groups(rows)
    if not events:
        raise ValueError("Cannot train alarm controller: validation rows contain no preictal event.")
    if int(neg_mask.sum().item()) == 0:
        raise ValueError("Cannot train alarm controller: validation rows contain no interictal windows.")

    final_metrics: dict[str, float] = {}
    for _epoch in range(epochs):
        optimizer.zero_grad()
        alarm = controller_outputs(rows, controller, device)
        event_losses = []
        event_maxima = []
        for indices in events:
            event_alarm = alarm[torch.tensor(indices, dtype=torch.long, device=device)]
            max_alarm = torch.max(event_alarm)
            event_maxima.append(max_alarm)
            event_losses.append(torch.relu(event_target_alarm - max_alarm).pow(2))
        event_loss = torch.stack(event_losses).mean()

        neg_alarm = alarm[neg_mask]
        false_margin_loss = torch.relu(neg_alarm - false_alarm_margin).pow(2).mean()
        topk = min(20, int(neg_alarm.numel()))
        topk_loss = torch.topk(neg_alarm, k=topk).values.mean() if topk > 0 else neg_alarm.mean()
        if (
            controller.variant
            in {
                "threshold_accum_cooldown",
                "multi_timescale_cooldown",
                "multi_timescale_hard_cooldown_update",
                "tiny_tcn_cooldown",
                "multi_timescale_tcn_suppression_cooldown",
            }
            and cooldown_decay_reg_weight > 0.0
            and cooldown_decay_reg_target > 0.0
        ):
            target = alarm.new_tensor(cooldown_decay_reg_target)
            cooldown_decay_reg_loss = torch.relu(target - controller.cooldown_decay()).pow(2)
        else:
            cooldown_decay_reg_loss = alarm.new_tensor(0.0)
        loss = (
            event_loss
            + false_weight * false_margin_loss
            + false_topk_weight * topk_loss
            + cooldown_decay_reg_weight * cooldown_decay_reg_loss
        )
        loss.backward()
        optimizer.step()
        final_metrics = {
            "train_loss": float(loss.detach().cpu().item()),
            "train_event_loss": float(event_loss.detach().cpu().item()),
            "train_false_margin_loss": float(false_margin_loss.detach().cpu().item()),
            "train_false_topk_loss": float(topk_loss.detach().cpu().item()),
            "train_cooldown_decay_reg_loss": float(cooldown_decay_reg_loss.detach().cpu().item()),
            "train_effective_cooldown_decay": float(controller.cooldown_decay().detach().cpu().item()),
            "train_min_event_max_alarm": float(torch.stack(event_maxima).min().detach().cpu().item()),
            "train_max_interictal_alarm": float(neg_alarm.max().detach().cpu().item()),
            "train_mean_interictal_alarm": float(neg_alarm.mean().detach().cpu().item()),
        }
    return controller, final_metrics


def rows_with_controller_score(
    rows: list[dict[str, str]],
    controller: LearnableAlarmController,
    device: torch.device,
) -> list[dict[str, str]]:
    with torch.no_grad():
        alarm_scores = controller_outputs(rows, controller, device).detach().cpu().numpy()
    out = []
    for row, alarm_score in zip(rows, alarm_scores, strict=True):
        new_row = dict(row)
        new_row["risk_score_preictal"] = new_row.get("score_preictal", "")
        new_row["score_preictal"] = str(float(alarm_score))
        out.append(new_row)
    return out


def apply_post_alarm_cooldown(
    rows: list[dict[str, str]],
    alarm_threshold: float,
    cooldown_sec: int,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Causally suppress score output after threshold-crossing alarms."""

    if cooldown_sec <= 0:
        return rows, {
            "post_alarm_cooldown_suppressed_windows": 0,
            "post_alarm_cooldown_suppressed_alarm_windows": 0,
        }

    out = [dict(row) for row in rows]
    suppressed_windows = 0
    suppressed_alarm_windows = 0
    for indices in group_indices(out):
        cooldown_until = -1
        for index in indices:
            row = out[index]
            start = time_start(row)
            score = safe_float(row.get("score_preictal"), 0.0)
            row["pre_post_alarm_cooldown_score_preictal"] = str(float(score))
            if start < cooldown_until:
                suppressed_windows += 1
                if score >= alarm_threshold:
                    suppressed_alarm_windows += 1
                row["score_preictal"] = "0.0"
                row["post_alarm_cooldown_suppressed"] = "1"
                continue
            row["post_alarm_cooldown_suppressed"] = "0"
            if score >= alarm_threshold:
                cooldown_until = start + cooldown_sec
    return out, {
        "post_alarm_cooldown_suppressed_windows": suppressed_windows,
        "post_alarm_cooldown_suppressed_alarm_windows": suppressed_alarm_windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subjects", default="", help="Comma-separated subjects to evaluate. Empty means all.")
    parser.add_argument("--limit-folds", type=int, default=0)
    parser.add_argument(
        "--fold-ids-file",
        type=Path,
        default=None,
        help="Optional newline-delimited fold IDs. Only these folds are evaluated.",
    )
    parser.add_argument(
        "--score-normalization",
        default="validation_interictal_ecdf",
        choices=[
            "none",
            "validation_zscore_sigmoid",
            "validation_minmax",
            "validation_robust_minmax",
            "validation_ecdf",
            "validation_interictal_ecdf",
        ],
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--event-target-alarm", type=float, default=0.8)
    parser.add_argument("--false-alarm-margin", type=float, default=0.05)
    parser.add_argument("--false-weight", type=float, default=8.0)
    parser.add_argument("--false-topk-weight", type=float, default=0.5)
    parser.add_argument(
        "--controller-variant",
        default="threshold_accum_cooldown",
        choices=(
            "threshold",
            "threshold_accum",
            "threshold_accum_cooldown",
            "multi_timescale_cooldown",
            "multi_timescale_hard_cooldown_update",
            "tiny_tcn_cooldown",
            "multi_timescale_tcn_suppression_cooldown",
        ),
        help=(
            "Alarm-policy ablation: learned threshold only, threshold+accumulation, "
            "full controller, multi-timescale evidence plus cooldown, hard-confirmed cooldown updates, "
            "tiny causal TCN plus cooldown, or multi-timescale evidence with a gated TCN suppression branch."
        ),
    )
    parser.add_argument(
        "--cooldown-decay-floor",
        type=float,
        default=0.0,
        help="Hard lower bound for the full controller's effective cooldown decay.",
    )
    parser.add_argument(
        "--cooldown-decay-reg-target",
        type=float,
        default=0.0,
        help="Soft regularization target for cooldown decay; active only when weight > 0.",
    )
    parser.add_argument(
        "--cooldown-decay-reg-weight",
        type=float,
        default=0.0,
        help="Penalty weight for ReLU(target - cooldown_decay)^2.",
    )
    parser.add_argument("--alarm-threshold", type=float, default=0.5)
    parser.add_argument("--min-alarm-windows", type=int, default=1)
    parser.add_argument(
        "--post-alarm-cooldown-min",
        type=float,
        default=0.0,
        help="Optional causal score suppression duration after any threshold-crossing alarm.",
    )
    parser.add_argument("--merge-gap-sec", type=int, default=7)
    parser.add_argument("--refractory-min", type=float, default=30.0)
    parser.add_argument("--window-sec", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=3, help="Torch seed for stochastic controller variants.")
    parser.add_argument(
        "--seed-skip-controller-inits",
        type=int,
        default=0,
        help=(
            "Advance the seeded controller-initialization RNG by this many folds before evaluation. "
            "This supports contiguous parallel chunks that exactly reproduce one serial seeded run."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.seed_skip_controller_inits < 0:
        raise SystemExit("--seed-skip-controller-inits must be non-negative.")
    for _ in range(args.seed_skip_controller_inits):
        # Controller modules initialize on CPU before being moved to the selected
        # device, matching the construction path used inside train_controller().
        LearnableAlarmController(
            variant=args.controller_variant,
            cooldown_decay_floor=args.cooldown_decay_floor,
        )

    fold_dirs = []
    for root in args.result_roots:
        fold_dirs.extend(path.parent for path in root.glob("*/*/test_predictions.csv"))
    fold_dirs = sorted(set(fold_dirs))
    if args.fold_ids_file is not None:
        requested_fold_ids = {
            line.strip()
            for line in args.fold_ids_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        fold_dirs = [fold_dir for fold_dir in fold_dirs if fold_dir.name in requested_fold_ids]
    selected_subjects = {item.strip() for item in args.subjects.split(",") if item.strip()}
    if selected_subjects:
        fold_dirs = [fold_dir for fold_dir in fold_dirs if fold_dir.parent.name in selected_subjects]
    if args.limit_folds:
        fold_dirs = fold_dirs[: args.limit_folds]
    if not fold_dirs:
        raise SystemExit("No test_predictions.csv files found.")

    refractory_sec = int(round(args.refractory_min * 60))
    calibrated_rows = []
    selected_rows = []
    skipped_rows = []

    for fold_dir in fold_dirs:
        val_prediction_path = fold_dir / "val_predictions.csv"
        test_prediction_path = fold_dir / "test_predictions.csv"
        if not val_prediction_path.exists() or not test_prediction_path.exists():
            skipped_rows.append({"fold_id": fold_dir.name, "subject": fold_dir.parent.name, "reason": "missing predictions"})
            continue
        try:
            val_rows, test_rows, normalization_metadata = normalize_prediction_rows(
                val_rows=read_csv(val_prediction_path),
                test_rows=read_csv(test_prediction_path),
                method=args.score_normalization,
            )
            controller, train_metrics = train_controller(
                rows=val_rows,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                event_target_alarm=args.event_target_alarm,
                false_alarm_margin=args.false_alarm_margin,
                false_weight=args.false_weight,
                false_topk_weight=args.false_topk_weight,
                controller_variant=args.controller_variant,
                cooldown_decay_floor=args.cooldown_decay_floor,
                cooldown_decay_reg_target=args.cooldown_decay_reg_target,
                cooldown_decay_reg_weight=args.cooldown_decay_reg_weight,
            )
            controlled_val_rows = rows_with_controller_score(val_rows, controller, device)
            controlled_test_rows = rows_with_controller_score(test_rows, controller, device)
            controlled_val_rows, val_cooldown_metrics = apply_post_alarm_cooldown(
                controlled_val_rows,
                alarm_threshold=args.alarm_threshold,
                cooldown_sec=int(round(args.post_alarm_cooldown_min * 60)),
            )
            controlled_test_rows, test_cooldown_metrics = apply_post_alarm_cooldown(
                controlled_test_rows,
                alarm_threshold=args.alarm_threshold,
                cooldown_sec=int(round(args.post_alarm_cooldown_min * 60)),
            )
        except Exception as exc:
            skipped_rows.append({"fold_id": fold_dir.name, "subject": fold_dir.parent.name, "reason": str(exc)})
            continue

        val_window_rows = read_csv(fold_dir / "val_windows.csv") if (fold_dir / "val_windows.csv").exists() else []
        test_window_rows = read_csv(fold_dir / "test_windows.csv") if (fold_dir / "test_windows.csv").exists() else []
        val_eval = evaluate_fold(
            fold_dir=fold_dir,
            pred_rows=controlled_val_rows,
            window_rows=val_window_rows,
            threshold=args.alarm_threshold,
            merge_gap_sec=args.merge_gap_sec,
            refractory_sec=refractory_sec,
            window_sec=args.window_sec,
            min_alarm_windows=args.min_alarm_windows,
        )
        test_eval = evaluate_fold(
            fold_dir=fold_dir,
            pred_rows=controlled_test_rows,
            window_rows=test_window_rows,
            threshold=args.alarm_threshold,
            merge_gap_sec=args.merge_gap_sec,
            refractory_sec=refractory_sec,
            window_sec=args.window_sec,
            min_alarm_windows=args.min_alarm_windows,
        )
        params = controller.params_dict()
        test_eval.update(
            {
                "controller": "learnable_alarm_controller_v1",
                "controller_variant": args.controller_variant,
                "alarm_threshold": args.alarm_threshold,
                "min_alarm_windows": args.min_alarm_windows,
                "post_alarm_cooldown_min": args.post_alarm_cooldown_min,
                "validation_event_detected": val_eval["event_detected"],
                "validation_false_alarm_rate_per_hour": val_eval["false_alarm_rate_per_hour"],
                "validation_false_alarm_events": val_eval["false_alarm_events"],
                "validation_sample_f1": val_eval["sample_f1"],
                **{f"validation_{key}": value for key, value in val_cooldown_metrics.items()},
                **test_cooldown_metrics,
                **params,
                **train_metrics,
                **normalization_metadata,
            }
        )
        calibrated_rows.append(test_eval)
        selected_rows.append(
            {
                "fold_id": fold_dir.name,
                "subject": fold_dir.parent.name,
                "controller": "learnable_alarm_controller_v1",
                "controller_variant": args.controller_variant,
                "alarm_threshold": args.alarm_threshold,
                "min_alarm_windows": args.min_alarm_windows,
                "post_alarm_cooldown_min": args.post_alarm_cooldown_min,
                **params,
                **train_metrics,
                "validation_event_detected": val_eval["event_detected"],
                "validation_false_alarm_rate_per_hour": val_eval["false_alarm_rate_per_hour"],
                "validation_false_alarm_events": val_eval["false_alarm_events"],
                "validation_sample_f1": val_eval["sample_f1"],
                **{f"validation_{key}": value for key, value in val_cooldown_metrics.items()},
                **test_cooldown_metrics,
                **normalization_metadata,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "learnable_alarm_controller_fold_event_metrics.csv", calibrated_rows)
    write_csv(args.out_dir / "learnable_alarm_controller_selected_params.csv", selected_rows)
    write_csv(args.out_dir / "skipped_folds.csv", skipped_rows)

    per_subject_rows = []
    for subject in sorted({str(row["subject"]) for row in calibrated_rows}):
        subject_rows = [row for row in calibrated_rows if str(row["subject"]) == subject]
        per_subject_rows.append({"subject": subject, **summarize(subject_rows)})
    write_csv(args.out_dir / "learnable_alarm_controller_per_subject_event_metrics.csv", per_subject_rows)

    summary = {
        "result_roots": [str(root) for root in args.result_roots],
        "subjects": sorted(selected_subjects),
        "limit_folds": args.limit_folds,
        "fold_ids_file": str(args.fold_ids_file) if args.fold_ids_file is not None else "",
        "folds_found": len(fold_dirs),
        "folds_evaluated": len(calibrated_rows),
        "folds_skipped": len(skipped_rows),
        "controller": "learnable_alarm_controller_v1",
        "controller_variant": args.controller_variant,
        "score_normalization": args.score_normalization,
        "epochs": args.epochs,
        "lr": args.lr,
        "event_target_alarm": args.event_target_alarm,
        "false_alarm_margin": args.false_alarm_margin,
        "false_weight": args.false_weight,
        "false_topk_weight": args.false_topk_weight,
        "cooldown_decay_floor": args.cooldown_decay_floor,
        "cooldown_decay_reg_target": args.cooldown_decay_reg_target,
        "cooldown_decay_reg_weight": args.cooldown_decay_reg_weight,
        "alarm_threshold": args.alarm_threshold,
        "min_alarm_windows": args.min_alarm_windows,
        "post_alarm_cooldown_min": args.post_alarm_cooldown_min,
        "merge_gap_sec": args.merge_gap_sec,
        "refractory_min": args.refractory_min,
        "window_sec": args.window_sec,
        "seed": args.seed,
        "seed_skip_controller_inits": args.seed_skip_controller_inits,
        "learnable_alarm_controller": summarize(calibrated_rows),
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
