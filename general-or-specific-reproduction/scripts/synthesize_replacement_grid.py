"""Synthesize one Voice-Ind or Gender-Ind epsilon grid with one ESPnet load."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from math import gcd
from pathlib import Path

import numpy as np

from synthesize_replacements_espnet import PHRASES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, default=Path("outputs/manifest.csv"))
    parser.add_argument("--method", choices=("voice-ind", "gender-ind"), required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=(1, 5, 10, 25, 50))
    parser.add_argument("--embedding-dir", type=Path, default=Path("outputs/embeddings"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/audio"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/audio/_source-conditioning-cache"))
    parser.add_argument("--model", default="kan-bayashi/libritts_xvector_vits")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="Optional clips per epsilon for a smoke run")
    args = parser.parse_args()
    try:
        import soundfile as sf
        import torch
        from espnet2.bin.tts_inference import Text2Speech
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise SystemExit("Install ESPnet, SciPy and soundfile in the synthesis environment") from exc

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows_by_name = {row["filename"]: row for row in csv.DictReader(stream)}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tts = Text2Speech.from_pretrained(
        args.model, device=args.device, always_fix_seed=True, use_teacher_forcing=True,
    )
    mode = b"source-conditioned"
    for epsilon in args.epsilons:
        tag = f"{epsilon:g}"
        replacement_path = args.embedding_dir / f"{args.method}-epsilon-{tag}.npz"
        if not replacement_path.is_file():
            raise FileNotFoundError(replacement_path)
        replacement_data = np.load(replacement_path, allow_pickle=False)
        names = replacement_data["filenames"].astype(str).tolist()
        vectors = replacement_data["vectors"]
        if args.limit is not None:
            names = names[:args.limit]
        replacement_lookup = dict(zip(replacement_data["filenames"].astype(str), vectors))
        missing = set(names) - rows_by_name.keys()
        if missing:
            raise ValueError(f"{len(missing)} replacement filenames are absent from the manifest")
        output_dir = args.output_root / f"{args.method}-e{tag}"
        output_dir.mkdir(parents=True, exist_ok=True)
        synthesized = reused = 0
        for index, filename in enumerate(names, start=1):
            phrase_code = filename.split("_")[1]
            if phrase_code not in PHRASES:
                raise ValueError(f"Unknown CREMA-D phrase code: {phrase_code}")
            vector = torch.as_tensor(replacement_lookup[filename], dtype=torch.float32, device=args.device)
            vector_bytes = vector.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
            cache_key = hashlib.sha256(
                mode + b"\0" + phrase_code.encode("utf-8") + b"\0" + vector_bytes
            ).hexdigest()
            output_path = output_dir / f"{filename}.wav"
            cache_path = args.cache_dir / f"{cache_key}.wav"
            if not cache_path.exists() and output_path.exists():
                shutil.copy2(output_path, cache_path)
            if cache_path.exists():
                if output_path != cache_path:
                    shutil.copyfile(cache_path, output_path)
                reused += 1
                continue

            source, source_fs = sf.read(rows_by_name[filename]["audio_path"], dtype="float32")
            if source.ndim == 2:
                source = source.mean(axis=1)
            if source_fs != tts.fs:
                divisor = gcd(int(source_fs), int(tts.fs))
                source = resample_poly(
                    source, int(tts.fs) // divisor, int(source_fs) // divisor,
                ).astype(np.float32, copy=False)
            with torch.inference_mode():
                result = tts(PHRASES[phrase_code], speech=source, spembs=vector)
            waveform = result["wav"].detach().cpu().numpy().reshape(-1)
            sf.write(output_path, waveform, tts.fs, subtype="PCM_16")
            shutil.copyfile(output_path, cache_path)
            synthesized += 1
            if index % 100 == 0 or index == len(names):
                print(
                    f"{args.method} epsilon={tag}: {index}/{len(names)} "
                    f"(new synthesis={synthesized}, cache reuse={reused})",
                    flush=True,
                )
        print(
            f"Completed {args.method} epsilon={tag}: new synthesis={synthesized}; "
            f"cache reuse={reused}; output={output_dir.resolve()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
