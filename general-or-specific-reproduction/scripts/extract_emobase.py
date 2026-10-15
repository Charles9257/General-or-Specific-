"""Extract the EmoBase functionals used by the authors' public reference code."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, default=Path("outputs/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/emobase.npz"))
    parser.add_argument("--audio-root", type=Path,
                        help="Use synthesized <filename>.wav files from this directory")
    parser.add_argument("--feature-cache", type=Path,
                        help="Optional NPZ cache keyed by source-audio SHA-256, useful across synthesis sweeps")
    args = parser.parse_args()

    try:
        import opensmile
    except ImportError as exc:
        raise SystemExit("Install OpenSMILE first: python -m pip install opensmile==2.2.0") from exc

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    unique = {
        row["filename"]: (args.audio_root / f"{row['filename']}.wav")
        if args.audio_root else Path(row["audio_path"])
        for row in rows
    }

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    feature_cache: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    if args.feature_cache and args.feature_cache.is_file():
        cached = np.load(args.feature_cache, allow_pickle=False)
        feature_cache = {
            key: values.astype(np.float32, copy=False)
            for key, values in zip(cached["audio_sha256"].astype(str), cached["X"])
        }
        feature_names = cached["feature_names"].astype(str).tolist()

    features: list[np.ndarray] = []
    names: list[str] = []
    new_features: dict[str, np.ndarray] = {}
    for filename, audio_path in sorted(unique.items()):
        audio_hash = hashlib.sha256()
        with audio_path.open("rb") as audio_stream:
            for chunk in iter(lambda: audio_stream.read(1024 * 1024), b""):
                audio_hash.update(chunk)
        cache_key = audio_hash.hexdigest()
        vector = feature_cache.get(cache_key)
        if vector is None:
            frame = smile.process_file(str(audio_path))
            if frame.shape[0] != 1:
                raise ValueError(f"Expected one functionals row for {audio_path}; got {frame.shape[0]}")
            if feature_names is None:
                feature_names = [str(name) for name in frame.columns]
            vector = frame.to_numpy(dtype=np.float32)[0]
            new_features[cache_key] = vector
        features.append(vector)
        names.append(filename)
        if len(names) % 250 == 0:
            print(f"Processed {len(names)}/{len(unique)} clips; extracted {len(new_features)} new feature vectors", flush=True)

    matrix = np.stack(features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=matrix,
        filenames=np.asarray(names, dtype=str),
        feature_names=np.asarray(feature_names or [], dtype=str),
    )
    if args.feature_cache:
        feature_cache.update(new_features)
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        cache_keys = np.asarray(sorted(feature_cache), dtype=str)
        np.savez_compressed(
            args.feature_cache,
            audio_sha256=cache_keys,
            X=np.stack([feature_cache[key] for key in cache_keys]),
            feature_names=np.asarray(feature_names or [], dtype=str),
        )
        print(f"Feature cache now contains {len(feature_cache)} unique audio vectors")
    print(f"Saved {matrix.shape[0]} utterances x {matrix.shape[1]} features to {args.output.resolve()}")


if __name__ == "__main__":
    main()
