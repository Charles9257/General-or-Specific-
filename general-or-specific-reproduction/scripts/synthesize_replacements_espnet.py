"""Condition ESPnet VITS on the sampled replacement x-vector for each clip.

This uses the public LibriTTS x-vector VITS checkpoint as an explicit
substitute for the paper's unspecified ESPnet-TTS checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PHRASES = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, default=Path("outputs/manifest.csv"))
    parser.add_argument("replacements", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="kan-bayashi/libritts_xvector_vits")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="Optional pilot limit; default synthesizes all rows")
    args = parser.parse_args()
    try:
        import soundfile as sf
        import torch
        from espnet2.bin.tts_inference import Text2Speech
    except ImportError as exc:
        raise SystemExit("Install ESPnet and its inference dependencies to synthesize audio") from exc

    replacement_data = np.load(args.replacements, allow_pickle=False)
    replacement_lookup = {
        filename: vector
        for filename, vector in zip(
            replacement_data["filenames"].astype(str), replacement_data["vectors"]
        )
    }
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows_by_name = {row["filename"]: row for row in csv.DictReader(stream)}
    names = sorted(replacement_lookup)
    if args.limit is not None:
        names = names[:args.limit]
    missing = set(names) - set(rows_by_name)
    if missing:
        raise ValueError(f"{len(missing)} replacement filenames are absent from the manifest")

    tts = Text2Speech.from_pretrained(args.model, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, filename in enumerate(names, start=1):
        phrase_code = filename.split("_")[1]
        if phrase_code not in PHRASES:
            raise ValueError(f"Unknown CREMA-D phrase code: {phrase_code}")
        vector = torch.as_tensor(replacement_lookup[filename], dtype=torch.float32, device=args.device)
        with torch.inference_mode():
            result = tts(PHRASES[phrase_code], spembs=vector)
        waveform = result["wav"].detach().cpu().numpy().reshape(-1)
        sf.write(args.output_dir / f"{filename}.wav", waveform, tts.fs, subtype="PCM_16")
        if index % 100 == 0 or index == len(names):
            print(f"Synthesized {index}/{len(names)} clips", flush=True)
    print(f"Wrote resynthesized audio to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
