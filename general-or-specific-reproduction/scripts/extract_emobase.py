"""Extract the EmoBase functionals used by the authors' public reference code."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, default=Path("outputs/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/emobase.npz"))
    parser.add_argument("--audio-root", type=Path,
                        help="Use synthesized <filename>.wav files from this directory")
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
    features: list[np.ndarray] = []
    names: list[str] = []
    feature_names: list[str] | None = None
    for filename, audio_path in sorted(unique.items()):
        frame = smile.process_file(str(audio_path))
        if frame.shape[0] != 1:
            raise ValueError(f"Expected one functionals row for {audio_path}; got {frame.shape[0]}")
        if feature_names is None:
            feature_names = [str(name) for name in frame.columns]
        features.append(frame.to_numpy(dtype=np.float32)[0])
        names.append(filename)

    matrix = np.stack(features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=matrix,
        filenames=np.asarray(names, dtype=str),
        feature_names=np.asarray(feature_names or [], dtype=str),
    )
    print(f"Saved {matrix.shape[0]} utterances x {matrix.shape[1]} features to {args.output.resolve()}")


if __name__ == "__main__":
    main()
