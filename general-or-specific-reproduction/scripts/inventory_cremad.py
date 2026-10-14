"""Inventory CREMA-D audio without changing the downloaded dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re


CREMAD_NAME = re.compile(
    r"^(?P<actor>\d{4})_(?P<phrase>[A-Z]{3})_(?P<emotion>[A-Z]{3})_"
    r"(?P<intensity>[A-Z]{2})\.wav$",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="CREMA-D root or extracted AudioWAV folder")
    args = parser.parse_args()

    files = sorted(args.dataset.rglob("*.wav"))
    actors: set[str] = set()
    emotions: Counter[str] = Counter()
    unmatched: list[str] = []
    for wav in files:
        match = CREMAD_NAME.match(wav.name)
        if match is None:
            unmatched.append(str(wav.relative_to(args.dataset)))
            continue
        actors.add(match["actor"])
        emotions[match["emotion"].upper()] += 1

    print(f"Dataset root: {args.dataset.resolve()}")
    print(f"WAV files: {len(files)}")
    print(f"Parsed CREMA-D filenames: {len(files) - len(unmatched)}")
    print(f"Unique actors: {len(actors)}")
    print("Emotion-code counts:")
    for emotion, count in sorted(emotions.items()):
        print(f"  {emotion}: {count}")
    if unmatched:
        print(f"Unparsed WAV filenames: {len(unmatched)}")
        for item in unmatched[:20]:
            print(f"  {item}")


if __name__ == "__main__":
    main()
