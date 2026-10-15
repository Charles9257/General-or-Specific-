"""Condition ESPnet VITS on the sampled replacement x-vector for each clip.

This uses the public LibriTTS x-vector VITS checkpoint as an explicit
substitute for the paper's unspecified ESPnet-TTS checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
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
    parser.add_argument(
        "--source-conditioned", action="store_true",
        help="Use source waveform as teacher-forced acoustic features while replacing the speaker vector",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/audio/_synthesis_cache"),
        help="Cache deterministic outputs by phrase and replacement embedding across epsilon sweeps",
    )
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

    # A fixed seed makes identical phrase/embedding inputs reproducible and
    # allows outputs to be shared across epsilon sweeps. The replacement pool
    # has only a finite number of vectors, so this avoids resynthesizing the
    # same phrase/voice pair thousands of times.
    tts = Text2Speech.from_pretrained(
        args.model, device=args.device, always_fix_seed=True,
        use_teacher_forcing=args.source_conditioned,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    synthesized = 0
    reused = 0
    for index, filename in enumerate(names, start=1):
        phrase_code = filename.split("_")[1]
        if phrase_code not in PHRASES:
            raise ValueError(f"Unknown CREMA-D phrase code: {phrase_code}")
        vector = torch.as_tensor(replacement_lookup[filename], dtype=torch.float32, device=args.device)
        vector_bytes = vector.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
        mode = b"source-conditioned" if args.source_conditioned else b"transcript-conditioned"
        cache_key = hashlib.sha256(mode + b"\0" + phrase_code.encode("utf-8") + b"\0" + vector_bytes).hexdigest()
        output_path = args.output_dir / f"{filename}.wav"
        cache_path = args.cache_dir / f"{cache_key}.wav"
        if not cache_path.exists() and output_path.exists():
            # Preserve valid work from an interrupted earlier invocation.
            shutil.copy2(output_path, cache_path)
        if cache_path.exists():
            if output_path != cache_path:
                shutil.copyfile(cache_path, output_path)
            reused += 1
            continue
        with torch.inference_mode():
            if args.source_conditioned:
                from math import gcd

                from scipy.signal import resample_poly

                source, source_fs = sf.read(rows_by_name[filename]["audio_path"], dtype="float32")
                if source.ndim == 2:
                    source = source.mean(axis=1)
                if source_fs != tts.fs:
                    divisor = gcd(int(source_fs), int(tts.fs))
                    source = resample_poly(
                        source, int(tts.fs) // divisor, int(source_fs) // divisor,
                    ).astype(np.float32, copy=False)
                result = tts(PHRASES[phrase_code], speech=source, spembs=vector)
            else:
                result = tts(PHRASES[phrase_code], spembs=vector)
        waveform = result["wav"].detach().cpu().numpy().reshape(-1)
        sf.write(output_path, waveform, tts.fs, subtype="PCM_16")
        shutil.copyfile(output_path, cache_path)
        synthesized += 1
        if index % 100 == 0 or index == len(names):
            print(
                f"Processed {index}/{len(names)} clips "
                f"(new synthesis={synthesized}, cache reuse={reused})",
                flush=True,
            )
    print(
        f"Wrote resynthesized audio to {args.output_dir.resolve()} "
        f"(new synthesis={synthesized}, cache reuse={reused})"
    )


if __name__ == "__main__":
    main()
