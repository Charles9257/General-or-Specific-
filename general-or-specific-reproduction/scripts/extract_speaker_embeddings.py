"""Extract CREMA-D x-vectors with the public SpeechBrain VoxCeleb model.

The paper does not identify its embedding checkpoint. This is an explicit
substitute for preparing a reproducible Voice-Ind/Gender-Ind pilot. The
resulting pool is CREMA-D actors, not an independent public speaker corpus.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, default=Path("outputs/manifest.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/embeddings"))
    parser.add_argument("--model", default="speechbrain/spkrec-xvect-voxceleb")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        import torch
        import torchaudio
        import soundfile as sf
        from scipy.signal import resample_poly
        # torchaudio 2.11 removed this query; SpeechBrain 1.0.3 still calls it.
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
    except ImportError as exc:
        raise SystemExit(
            "Install the pinned optional dependencies from requirements-embeddings.txt"
        ) from exc

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    unique = {row["filename"]: row for row in rows}
    encoder = EncoderClassifier.from_hparams(
        source=args.model,
        savedir=str(args.output_root / "speechbrain-xvect-model"),
        run_opts={"device": args.device},
        local_strategy=LocalStrategy.COPY,
    )
    names: list[str] = []
    actors: list[str] = []
    vectors: list[np.ndarray] = []
    ordered_rows = sorted(unique.items())
    for item_index, (filename, row) in enumerate(ordered_rows, start=1):
        waveform_np, sample_rate = sf.read(row["audio_path"], dtype="float32", always_2d=True)
        waveform_np = waveform_np.mean(axis=1)
        if sample_rate != 16000:
            from math import gcd
            divisor = gcd(int(sample_rate), 16000)
            waveform_np = resample_poly(waveform_np, 16000 // divisor, int(sample_rate) // divisor)
        waveform = torch.from_numpy(np.ascontiguousarray(waveform_np)).unsqueeze(0)
        with torch.inference_mode():
            embedding = encoder.encode_batch(waveform.to(args.device)).squeeze().cpu().numpy()
        vectors.append(np.asarray(embedding, dtype=np.float32).reshape(-1))
        names.append(filename)
        actors.append(str(row["actor_id"]))
        if item_index % 250 == 0 or item_index == len(ordered_rows):
            print(f"Encoded {item_index}/{len(ordered_rows)} clips", flush=True)

    matrix = np.stack(vectors)
    actor_array = np.asarray(actors, dtype=str)
    args.output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_root / "utterance_xvectors.npz",
        filenames=np.asarray(names, dtype=str), actor_ids=actor_array, vectors=matrix,
    )
    actor_ids = np.unique(actor_array)
    actor_vectors = np.stack([matrix[actor_array == actor].mean(axis=0) for actor in actor_ids])
    actor_vectors /= np.maximum(np.linalg.norm(actor_vectors, axis=1, keepdims=True), 1e-12)
    np.savez_compressed(
        args.output_root / "cremad_actor_pool.npz",
        actor_ids=actor_ids, vectors=actor_vectors.astype(np.float32),
    )
    print(f"Saved {len(names)} utterance x-vectors ({matrix.shape[1]} dimensions)")
    print(f"Saved {len(actor_ids)} actor-mean candidates to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
