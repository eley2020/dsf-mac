#!/usr/bin/env python3
"""Train a non-federated CNN-GRU baseline for CHB-MIT seizure prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch import nn


def metrics_from_outputs(y_true, y_score, threshold=0.5):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

    y_pred = (y_score >= threshold).astype("int64")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    return {
        "acc": float(acc),
        "auc": float(auc),
        "f1": float(f1),
        "sensitivity": float(sen),
        "specificity": float(spe),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


class CNNGRU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.gru = nn.GRU(128, 128, batch_first=False)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        _, h = self.gru(x)
        return self.fc(h[-1])



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chbmit/baseline/chbmit_baseline_leave_last.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/baseline_cnn_gru"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loaded = np.load(args.data, allow_pickle=True)
    x_train = loaded["x_train"]
    y_train = loaded["y_train"]
    x_test = loaded["x_test"]
    y_test = loaded["y_test"]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = CNNGRU(channels=x_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best = {"auc": -1.0, "epoch": 0}
    history = []

    def evaluate():
        model.eval()
        scores = []
        labels = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                scores.append(prob)
                labels.append(yb.numpy())
        return metrics_from_outputs(np.concatenate(labels), np.concatenate(scores))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item() * yb.numel()
            total += yb.numel()
        metrics = evaluate()
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(total, 1)
        history.append(metrics)
        print(
            f"epoch={epoch:03d} loss={metrics['train_loss']:.4f} "
            f"acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} "
            f"f1={metrics['f1']:.4f} sen={metrics['sensitivity']:.4f} spe={metrics['specificity']:.4f}"
        )
        if metrics["auc"] > best["auc"]:
            best = dict(metrics)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "data": str(args.data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "best": best,
        "history": history,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Best:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
