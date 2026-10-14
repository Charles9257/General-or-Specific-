"""Train the shadow attacker and evaluate gender inference on private clients."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def load_updates(path: Path) -> dict[int, dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    result: dict[int, dict[str, object]] = {}
    for actor in np.unique(data["actor_id"]):
        mask = data["actor_id"] == actor
        result[int(actor)] = {
            "gender": int(data["gender"][np.flatnonzero(mask)[0]]),
            "weight": data["first_weight"][mask].astype(np.float32),
            "bias": data["first_bias"][mask].astype(np.float32),
        }
    return result


def samples_by_actor(data: dict[int, dict[str, object]], n: int, seed: int, repeats: int = 20):
    weights, biases, labels = [], [], []
    rng = np.random.default_rng(seed)
    for actor in sorted(data):
        actor_data = data[actor]
        available = len(actor_data["weight"])
        if available == 0:
            continue
        take = min(n, available)
        for _ in range(repeats):
            indices = rng.choice(available, size=take, replace=False)
            weights.append(actor_data["weight"][indices].mean(axis=0))
            biases.append(actor_data["bias"][indices].mean(axis=0))
            labels.append(actor_data["gender"])
    return np.stack(weights), np.stack(biases), np.asarray(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, choices=(5, 10, 100), default=5)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--fast", action="store_true",
                        help="Downsample the first-layer update map 2x4 before the reference CNN; this is a runtime deviation for CPU-only runs")
    parser.add_argument("--output", type=Path, default=Path("outputs/attack_results.json"))
    args = parser.parse_args()

    upstream_model = Path(__file__).resolve().parents[1] / "upstream" / "fed-ser-leakage" / "model"
    if not upstream_model.joinpath("attack_model.py").is_file():
        raise FileNotFoundError(f"Expected authors' attack model at {upstream_model}")
    sys.path.insert(0, str(upstream_model))
    from attack_model import attack_model  # uses the original first-layer CNN

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    shadow = load_updates(args.shadow)
    private = load_updates(args.private)
    train_w, train_b, train_y = samples_by_actor(shadow, args.num_samples, args.seed)
    test_w, test_b, test_y = samples_by_actor(private, args.num_samples, args.seed + 1)

    # Match the reference implementation's per-element z-normalization.
    weight_mean, weight_std = train_w.mean(axis=0), train_w.std(axis=0)
    bias_mean, bias_std = train_b.mean(axis=0), train_b.std(axis=0)
    train_w = (train_w - weight_mean) / (weight_std + 1e-5)
    test_w = (test_w - weight_mean) / (weight_std + 1e-5)
    train_b = (train_b - bias_mean) / (bias_std + 1e-5)
    test_b = (test_b - bias_mean) / (bias_std + 1e-5)
    train_w = np.nan_to_num(train_w, nan=0.0, posinf=0.0, neginf=0.0)
    test_w = np.nan_to_num(test_w, nan=0.0, posinf=0.0, neginf=0.0)
    train_b = np.nan_to_num(train_b, nan=0.0, posinf=0.0, neginf=0.0)
    test_b = np.nan_to_num(test_b, nan=0.0, posinf=0.0, neginf=0.0)

    order = np.random.default_rng(args.seed).permutation(len(train_y))
    split_at = max(1, int(0.8 * len(order)))
    train_idx, val_idx = order[:split_at], order[split_at:]
    if len(val_idx) == 0:
        val_idx = train_idx[-1:]
    train_data = TensorDataset(
        torch.from_numpy(train_w[train_idx]).float(),
        torch.from_numpy(train_b[train_idx]).float(),
        torch.from_numpy(train_y[train_idx]).long(),
    )
    val_data = TensorDataset(
        torch.from_numpy(train_w[val_idx]).float(),
        torch.from_numpy(train_b[val_idx]).float(),
        torch.from_numpy(train_y[val_idx]).long(),
    )
    test_data = TensorDataset(
        torch.from_numpy(test_w).float(),
        torch.from_numpy(test_b).float(),
        torch.from_numpy(test_y).long(),
    )
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=20, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=20, shuffle=False)

    model = attack_model("first", "emobase")
    if args.fast:
        # The paper does not disclose the attack-network architecture. This
        # preserves the reference CNN blocks while reducing their spatial cost.
        model.conv0 = nn.Sequential(nn.AvgPool2d(kernel_size=(2, 4)), model.conv0)
        model.dense1 = nn.Linear(640, 256)
        nn.init.xavier_uniform_(model.dense1.weight)
        nn.init.constant_(model.dense1.bias, 0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.NLLLoss()
    best_val = float("inf")
    best_state = None
    patience = 6
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        for weights, bias, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(weights.unsqueeze(1), bias)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for weights, bias, labels in val_loader:
                val_losses.append(float(loss_fn(model(weights.unsqueeze(1), bias), labels)))
        mean_val = float(np.mean(val_losses))
        print(f"attack samples={args.num_samples} epoch={epoch+1}/{args.epochs} val_loss={mean_val:.5f}", flush=True)
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for weights, bias, labels in test_loader:
            logits = model(weights.unsqueeze(1), bias)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).numpy().tolist())
    result = {
        "num_samples_per_client": args.num_samples,
        "attack_model_downsampled": bool(args.fast),
        "shadow_actors": len(shadow),
        "private_actors": len(private),
        "shadow_update_events": sum(len(x["weight"]) for x in shadow.values()),
        "private_update_events": sum(len(x["weight"]) for x in private.values()),
        "success_rate": float(accuracy_score(y_true, y_pred)),
        "unweighted_average_success_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "best_validation_loss": best_val,
        "epochs_run": epoch + 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
