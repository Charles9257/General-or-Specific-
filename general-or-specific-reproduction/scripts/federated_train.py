"""Train one shadow or private FedAvg run and save first-layer update snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SERClassifier(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.dense1 = nn.Linear(input_size, 256)
        self.dense2 = nn.Linear(256, 128)
        self.pred_layer = nn.Linear(128, 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.relu(self.dense1(x.float())))
        x = self.dropout(torch.relu(self.dense2(x)))
        return torch.log_softmax(self.pred_layer(x), dim=1)


def read_manifest(path: Path, fold: int, role: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [r for r in csv.DictReader(stream) if int(r["fold"]) == fold and r["role"] == role]


def score(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    recall = []
    for label in range(4):
        mask = y == label
        if mask.any():
            recall.append(float((pred[mask] == y[mask]).float().mean()))
    return {
        "accuracy": float((pred == y).float().mean()),
        "uar": float(np.mean(recall)) if recall else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifest.csv"))
    parser.add_argument("--features", type=Path, default=Path("outputs/emobase.npz"))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--role", choices=("shadow", "private"), required=True)
    parser.add_argument("--defense", choices=("none", "udp", "voice-ind", "gender-ind"), default="none")
    parser.add_argument("--feature-privacy", choices=("plain", "voice-ind", "gender-ind"), default="plain",
                        help="Privacy transformation already applied to the audio used for these features")
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--participation", type=float, default=0.1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--clip", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    args = parser.parse_args()

    if args.defense != "none" and args.epsilon <= 0:
        parser.error("epsilon must be positive for privacy defenses")
    expected_feature_privacy = args.defense if args.defense in {"voice-ind", "gender-ind"} else "plain"
    if args.feature_privacy != expected_feature_privacy:
        parser.error(f"{args.defense} requires --feature-privacy {expected_feature_privacy}; extract features from the corresponding transformed audio first")
    torch.manual_seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    random.seed(args.seed + args.fold)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cpu")

    rows = read_manifest(args.manifest, args.fold, args.role)
    if not rows:
        raise ValueError(f"No rows for fold={args.fold}, role={args.role}")
    feature_file = np.load(args.features, allow_pickle=False)
    feature_lookup = {name: i for i, name in enumerate(feature_file["filenames"].tolist())}
    raw_x = feature_file["X"]
    label_order = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3}

    client_rows: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        client_rows.setdefault(int(row["actor_id"]), []).append(row)
    clients: dict[int, dict[str, object]] = {}
    for actor_id, actor_rows in client_rows.items():
        train_rows = [r for r in actor_rows if r["local_partition"] == "train"]
        val_rows = [r for r in actor_rows if r["local_partition"] == "validation"]
        if not train_rows or not val_rows:
            continue
        train_x = np.stack([raw_x[feature_lookup[r["filename"]]] for r in train_rows]).astype(np.float32)
        val_x = np.stack([raw_x[feature_lookup[r["filename"]]] for r in val_rows]).astype(np.float32)
        mean = np.nanmean(train_x, axis=0)
        std = np.nanstd(train_x, axis=0)
        train_x = np.nan_to_num((train_x - mean) / (std + 1e-5), nan=0.0, posinf=0.0, neginf=0.0)
        val_x = np.nan_to_num((val_x - mean) / (std + 1e-5), nan=0.0, posinf=0.0, neginf=0.0)
        label = lambda r: label_order[r["label"]]
        clients[actor_id] = {
            "gender": 0 if actor_rows[0]["gender"] == "F" else 1,
            "train_x": torch.from_numpy(train_x),
            "train_y": torch.tensor([label(r) for r in train_rows], dtype=torch.long),
            "val_x": torch.from_numpy(val_x),
            "val_y": torch.tensor([label(r) for r in val_rows], dtype=torch.long),
        }

    client_ids = sorted(clients)
    input_size = int(raw_x.shape[1])
    global_model = SERClassifier(input_size).to(device)
    criterion = nn.NLLLoss()
    saved_updates: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    round_metrics: list[dict[str, float | int]] = []

    for round_index in range(args.rounds):
        sample_n = max(1, int(args.participation * len(client_ids)))
        rng = random.Random(args.seed + args.fold * 100_000 + round_index)
        chosen = rng.sample(client_ids, min(sample_n, len(client_ids)))
        old_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
        uploads: list[dict[str, torch.Tensor]] = []
        upload_sizes: list[int] = []

        for actor_id in chosen:
            client = clients[actor_id]
            local_model = SERClassifier(input_size).to(device)
            local_model.load_state_dict(old_state)
            optimizer = torch.optim.Adam(
                local_model.parameters(), lr=args.learning_rate, weight_decay=1e-4,
                betas=(0.9, 0.98), eps=1e-9,
            )
            loader = DataLoader(
                TensorDataset(client["train_x"], client["train_y"]),
                batch_size=args.batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(args.seed + round_index + actor_id),
            )
            local_model.train()
            for _ in range(args.local_epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(local_model(batch_x), batch_y)
                    loss.backward()
                    if args.defense == "udp":
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), args.clip)
                    optimizer.step()

            local_state = {k: v.detach().clone() for k, v in local_model.state_dict().items()}
            n_local = int(client["train_y"].shape[0])
            noise_scale = 0.0
            if args.defense == "udp":
                sensitivity = 2 * args.clip / n_local
                noise_scale = sensitivity * math.sqrt(
                    2 * args.participation * args.rounds * math.log(1 / args.delta)
                ) / args.epsilon
                for key in local_state:
                    local_state[key] += torch.randn_like(local_state[key]) * noise_scale

            # The server can reconstruct the first-layer pseudo-gradient from the
            # shared FedAvg local model and the global model snapshot.
            batches_per_epoch = int(n_local / args.batch_size) + 1
            divisor = args.learning_rate * batches_per_epoch * args.local_epochs
            first_weight = (old_state["dense1.weight"] - local_state["dense1.weight"]) / divisor
            first_bias = (old_state["dense1.bias"] - local_state["dense1.bias"]) / divisor
            saved_updates.append(
                (round_index, actor_id, int(client["gender"]),
                 first_weight.cpu().numpy().astype(np.float16),
                 first_bias.cpu().numpy().astype(np.float16))
            )
            uploads.append(local_state)
            upload_sizes.append(n_local)

        total = sum(upload_sizes)
        new_state = {}
        for key in old_state:
            new_state[key] = sum(
                upload[key] * (size / total)
                for upload, size in zip(uploads, upload_sizes)
            )
        global_model.load_state_dict(new_state)

        # Validation over every client's held-out 20%, weighted by utterance count.
        val_x = torch.cat([clients[c]["val_x"] for c in client_ids])
        val_y = torch.cat([clients[c]["val_y"] for c in client_ids])
        metrics = score(global_model, val_x, val_y)
        round_metrics.append({"round": round_index + 1, **metrics})
        if (round_index + 1) % 10 == 0:
            print(f"fold={args.fold} role={args.role} defense={args.defense} round={round_index+1}/{args.rounds} val_acc={metrics['accuracy']:.4f} val_uar={metrics['uar']:.4f}", flush=True)

    output_dir = args.output_root / f"fold{args.fold}" / args.role / args.defense
    if args.defense != "none":
        output_dir = output_dir / f"epsilon_{args.epsilon:g}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds = np.asarray([item[0] for item in saved_updates], dtype=np.int16)
    actor_ids = np.asarray([item[1] for item in saved_updates], dtype=np.int16)
    genders = np.asarray([item[2] for item in saved_updates], dtype=np.int8)
    first_weights = np.stack([item[3] for item in saved_updates])
    first_biases = np.stack([item[4] for item in saved_updates])
    # Keep gradient snapshots uncompressed: they are high-entropy and the
    # compression cost is substantial for little space saved.
    np.savez(
        output_dir / "updates.npz", round=rounds, actor_id=actor_ids, gender=genders,
        first_weight=first_weights, first_bias=first_biases,
    )
    torch.save(global_model.state_dict(), output_dir / "ser_model.pt")
    with (output_dir / "ser_validation.json").open("w", encoding="utf-8") as stream:
        json.dump({"rounds": round_metrics, "final": round_metrics[-1]}, stream, indent=2)
    run_meta = {
        "fold": args.fold, "role": args.role, "defense": args.defense,
        "feature_privacy": args.feature_privacy,
        "epsilon": args.epsilon if args.defense != "none" else None,
        "clients": len(client_ids), "rounds": args.rounds,
        "participation": args.participation, "local_epochs": args.local_epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "clip": args.clip if args.defense == "udp" else None,
        "delta": args.delta if args.defense == "udp" else None,
        "udp_noise_scale_range": [
            (2 * args.clip / len(clients[c]["train_y"]))
            * math.sqrt(2 * args.participation * args.rounds * math.log(1 / args.delta))
            / args.epsilon
            for c in client_ids
        ] if args.defense == "udp" else None,
        "input_size": input_size,
        "features": str(args.features.resolve()),
    }
    with (output_dir / "run.json").open("w", encoding="utf-8") as stream:
        json.dump(run_meta, stream, indent=2)
    print(f"Saved {len(saved_updates)} updates to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
