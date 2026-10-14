"""Apply the exponential-mechanism embedding replacement used by Voice/Gender-Ind.

This stage expects embeddings from the selected extractor. It does not run the
speaker/gender encoder or the ESPnet speech synthesizer; those are separate
versioned model assets because the 2023 paper does not identify their exact
checkpoint files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_embeddings", type=Path, help="NPZ with filenames and vectors")
    parser.add_argument("public_pool", type=Path, help="NPY/NPZ matrix of public candidate vectors")
    parser.add_argument("--method", choices=("voice-ind", "gender-ind"), required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.epsilon <= 0:
        parser.error("epsilon must be positive")

    source_data = np.load(args.source_embeddings, allow_pickle=False)
    names = source_data["filenames"].astype(str)
    source = np.asarray(source_data["vectors"], dtype=np.float64)
    pool_data = np.load(args.public_pool, allow_pickle=False)
    pool = np.asarray(pool_data["vectors"] if isinstance(pool_data, np.lib.npyio.NpzFile) else pool_data, dtype=np.float64)
    if source.ndim != 2 or pool.ndim != 2 or source.shape[1] != pool.shape[1]:
        raise ValueError(f"Expected source [n,d] and pool [m,d] with shared d; got {source.shape} and {pool.shape}")
    if len(names) != len(source) or len(pool) == 0:
        raise ValueError("Embedding filenames must align with source vectors and the public pool cannot be empty")

    if args.method == "voice-ind":
        if "actor_ids" in source_data:
            source_actor_ids = source_data["actor_ids"].astype(str)
        else:
            source_actor_ids = np.asarray([name.split("_", 1)[0] for name in names], dtype=str)
        if not isinstance(pool_data, np.lib.npyio.NpzFile) or "actor_ids" not in pool_data:
            raise ValueError("Voice-Ind needs public_pool.npz with vectors and per-speaker actor_ids for self-exclusion")
        pool_actor_ids = pool_data["actor_ids"].astype(str)
        if len(source_actor_ids) != len(source):
            raise ValueError("source actor_ids must align with source vectors")
        unique_actors = np.unique(source_actor_ids)
        actor_to_vector = {
            actor: source[source_actor_ids == actor].mean(axis=0)
            for actor in unique_actors
        }
        mechanism_source = np.stack([actor_to_vector[actor] for actor in unique_actors])
    else:
        source_actor_ids = np.asarray([name.split("_", 1)[0] for name in names], dtype=str)
        unique_actors = None
        pool_actor_ids = None
        mechanism_source = source

    mechanism_source /= np.maximum(np.linalg.norm(mechanism_source, axis=1, keepdims=True), 1e-12)
    pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
    cosine = np.clip(mechanism_source @ pool.T, -1.0, 1.0)
    angular_distance = np.arccos(cosine) / np.pi
    logits = -args.epsilon * angular_distance
    if args.method == "voice-ind":
        logits[unique_actors[:, None] == pool_actor_ids[None, :]] = -np.inf
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(args.seed)
    selected_by_source = np.asarray([
        rng.choice(len(pool), p=probabilities[index])
        for index in range(len(mechanism_source))
    ], dtype=np.int32)
    if args.method == "voice-ind":
        actor_to_index = dict(zip(unique_actors, selected_by_source))
        selected = np.asarray([actor_to_index[actor] for actor in source_actor_ids], dtype=np.int32)
        chosen_distance = np.asarray([
            angular_distance[np.flatnonzero(unique_actors == actor)[0], actor_to_index[actor]]
            for actor in source_actor_ids
        ])
    else:
        selected = selected_by_source
        chosen_distance = angular_distance[np.arange(len(source)), selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        filenames=names,
        vectors=pool[selected].astype(np.float32),
        public_pool_index=selected,
        angular_distance=chosen_distance.astype(np.float32),
        method=np.asarray(args.method),
        epsilon=np.float64(args.epsilon),
        seed=np.int64(args.seed),
    )
    print(f"Selected replacement embeddings for {len(names)} utterances: {args.output.resolve()}")


if __name__ == "__main__":
    main()
