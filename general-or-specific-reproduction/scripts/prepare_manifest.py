"""Build deterministic speaker-level folds from the CREMA-D release."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


LABEL_MAP = {"N": "neutral", "S": "sad", "H": "happy", "A": "angry"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Extracted CREMA-D root containing AudioWAV/")
    parser.add_argument("--output", type=Path, default=Path("outputs/manifest.csv"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--shadow-actors", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    audio_dir = dataset / "AudioWAV"
    demographics_path = dataset / "VideoDemographics.csv"
    votes_path = dataset / "processedResults" / "summaryTable.csv"
    for required in (audio_dir, demographics_path, votes_path):
        if not required.exists():
            raise FileNotFoundError(f"Required CREMA-D path not found: {required}")

    demographics = {
        int(row["ActorID"]): row["Sex"].strip().strip('"')
        for row in read_rows(demographics_path)
    }
    vote_rows = read_rows(votes_path)
    utterances: list[dict[str, object]] = []
    skipped_label = 0
    missing_audio = 0
    for row in vote_rows:
        filename = row["FileName"].strip().strip('"')
        vote = row["MultiModalVote"].strip().strip('"')
        if vote not in LABEL_MAP:
            skipped_label += 1
            continue
        try:
            actor_id = int(filename.split("_", maxsplit=1)[0])
        except ValueError as exc:
            raise ValueError(f"Unexpected CREMA-D filename: {filename}") from exc
        audio_path = audio_dir / f"{filename}.wav"
        if not audio_path.is_file():
            missing_audio += 1
            continue
        sex = demographics.get(actor_id)
        if sex not in {"Male", "Female"}:
            raise ValueError(f"Missing or unsupported sex metadata for actor {actor_id}: {sex!r}")
        utterances.append(
            {
                "actor_id": actor_id,
                "gender": "M" if sex == "Male" else "F",
                "filename": filename,
                "audio_path": str(audio_path),
                "label_code": vote,
                "label": LABEL_MAP[vote],
            }
        )

    actors = sorted({int(row["actor_id"]) for row in utterances})
    if args.shadow_actors >= len(actors):
        raise ValueError("shadow-actors must be smaller than the available actor count")

    output_rows: list[dict[str, object]] = []
    for fold in range(1, args.folds + 1):
        rng = random.Random(args.seed + fold - 1)
        shuffled = actors[:]
        rng.shuffle(shuffled)
        shadow = set(shuffled[: args.shadow_actors])
        for actor_id in actors:
            actor_rows = [row for row in utterances if int(row["actor_id"]) == actor_id]
            actor_rng = random.Random(args.seed + fold * 100_000 + actor_id)
            actor_rng.shuffle(actor_rows)
            train_n = int(0.8 * len(actor_rows))
            role = "shadow" if actor_id in shadow else "private"
            for index, row in enumerate(actor_rows):
                output_rows.append(
                    {
                        "fold": fold,
                        "role": role,
                        "local_partition": "train" if index < train_n else "validation",
                        **row,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold", "role", "actor_id", "gender", "filename", "audio_path",
        "label_code", "label", "local_partition",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} utterance rows to {args.output.resolve()}")
    print(f"Usable actors: {len(actors)}; shadow per fold: {args.shadow_actors}; private per fold: {len(actors)-args.shadow_actors}")
    print(f"Skipped vote labels: {skipped_label}; missing audio: {missing_audio}")
    print("Labels: " + ", ".join(f"{code}={name}" for code, name in LABEL_MAP.items()))


if __name__ == "__main__":
    main()
