"""Train a gender-supervised embedding transform for the Gender-Ind pilot.

The paper specifies an x-vector model trained on gender IDs but does not publish
the checkpoint or identify its public gender-embedding dataset. This script
uses the extracted SpeechBrain x-vectors as inputs and CREMA-D actor sex labels
as supervision, keeping actors disjoint between encoder training and validation.
The resulting 512-D hidden representation is used only as the Gender-Ind
mechanism's input/candidate embedding; it is not the SER model or attack model.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class GenderEmbedding(nn.Module):
    def __init__(self, dimensions: int = 512) -> None:
        super().__init__()
        self.projection = nn.Linear(dimensions, dimensions)
        self.classifier = nn.Linear(dimensions, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = torch.tanh(self.projection(x))
        return embedding, self.classifier(embedding)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, default=Path("outputs/manifest.csv"))
    parser.add_argument("embeddings", type=Path, default=Path("outputs/embeddings/utterance_xvectors.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/embeddings/utterance_gender_embeddings.npz"))
    parser.add_argument("--model-output", type=Path, default=Path("outputs/embeddings/gender_embedding_model.pt"))
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = {r["filename"]: r for r in csv.DictReader(stream)}
    data = np.load(args.embeddings, allow_pickle=False)
    filenames = data["filenames"].astype(str)
    x = np.asarray(data["vectors"], dtype=np.float32)
    actors = np.asarray([int(manifest[name]["actor_id"]) for name in filenames], dtype=np.int32)
    # F=0, M=1, matching the gender labels used by the update attack.
    y = np.asarray([0 if manifest[name]["gender"] == "F" else 1 for name in filenames], dtype=np.int64)

    unique_actor_gender = {int(a): int(y[np.flatnonzero(actors == a)[0]]) for a in np.unique(actors)}
    rng = np.random.default_rng(args.seed)
    train_actors: set[int] = set()
    for gender in (0, 1):
        members = np.asarray([a for a, label in unique_actor_gender.items() if label == gender])
        rng.shuffle(members)
        train_actors.update(members[: max(1, int(0.8 * len(members)))].tolist())
    train_mask = np.asarray([int(a) in train_actors for a in actors])
    val_mask = ~train_mask
    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0)
    x = np.nan_to_num((x - mean) / (std + 1e-5), nan=0.0, posinf=0.0, neginf=0.0)

    train = TensorDataset(torch.from_numpy(x[train_mask]), torch.from_numpy(y[train_mask]))
    val_x, val_y = torch.from_numpy(x[val_mask]), torch.from_numpy(y[val_mask])
    loader = DataLoader(train, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = GenderEmbedding(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            _, val_logits = model(val_x)
            accuracy = float((val_logits.argmax(1) == val_y).float().mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"epoch={epoch+1}/{args.epochs} held-out-actor gender accuracy={accuracy:.4f}", flush=True)

    if best_state is None:
        raise RuntimeError("Gender embedding training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vectors = model(torch.from_numpy(x))[0].numpy()
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, filenames=filenames, vectors=vectors.astype(np.float32), actor_ids=actors,
        gender_labels=y, method=np.asarray("gender-supervised-hidden-xvector"),
        heldout_actor_accuracy=np.float64(best_accuracy),
    )
    pool_actors = np.unique(actors)
    pool = np.stack([vectors[actors == actor].mean(axis=0) for actor in pool_actors])
    pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
    pool_path = args.output.with_name("cremad_gender_embedding_pool.npz")
    np.savez_compressed(pool_path, vectors=pool.astype(np.float32), actor_ids=pool_actors.astype(str))
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean, "std": std, "heldout_actor_accuracy": best_accuracy}, args.model_output)
    print(f"Wrote gender embeddings {vectors.shape} and candidate pool {pool.shape}; held-out actor accuracy={best_accuracy:.4f}")


if __name__ == "__main__":
    main()
