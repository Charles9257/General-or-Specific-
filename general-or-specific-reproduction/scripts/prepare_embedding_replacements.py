"""Generate Voice-Ind and Gender-Ind replacement vectors over the paper grid."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs" / "embeddings" / "utterance_xvectors.npz"
    pool = root / "outputs" / "embeddings" / "cremad_actor_pool.npz"
    sampler = root / "scripts" / "sample_privacy_embeddings.py"
    if not source.is_file() or not pool.is_file():
        raise SystemExit("Run extract_speaker_embeddings.py first")
    for method in ("voice-ind", "gender-ind"):
        for epsilon in (1, 5, 10, 25, 50):
            output = root / "outputs" / "embeddings" / f"{method}-epsilon-{epsilon}.npz"
            command = [
                sys.executable, str(sampler), str(source), str(pool),
                "--method", method, "--epsilon", str(epsilon),
                "--seed", "2023", "--output", str(output),
            ]
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
