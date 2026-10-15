"""Write compact, versionable comparison tables from local run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PAPER_BASELINE = {
    1: {"sr": 0.837, "uasr": 0.829, "acc": 0.663, "uar": 0.595},
    2: {"sr": 0.847, "uasr": 0.838, "acc": 0.666, "uar": 0.601},
    3: {"sr": 0.817, "uasr": 0.791, "acc": 0.656, "uar": 0.619},
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    project = args.project.resolve()
    runs = project / "outputs" / "runs"
    output = args.output_dir if args.output_dir.is_absolute() else project / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    ser_rows: list[dict] = []
    for fold in (1, 2, 3):
        for defense_dir in sorted((runs / f"fold{fold}" / "private").glob("*")):
            if not defense_dir.is_dir():
                continue
            for validation_path in sorted(defense_dir.rglob("ser_validation.json")):
                meta_path = validation_path.with_name("run.json")
                meta = load_json(meta_path) if meta_path.is_file() else {}
                final = load_json(validation_path).get("final", {})
                ser_rows.append({
                    "source": "reproduction",
                    "fold": fold,
                    "defense": meta.get("defense", defense_dir.name),
                    "epsilon": meta.get("epsilon"),
                    "accuracy": final.get("accuracy"),
                    "uar": final.get("uar"),
                    "round": final.get("round"),
                    "normalization": meta.get("normalization"),
                    "artifact": validation_path.relative_to(project).as_posix(),
                })

    for fold, values in PAPER_BASELINE.items():
        ser_rows.append({
            "source": "paper",
            "fold": fold,
            "defense": "none",
            "epsilon": None,
            "accuracy": values["acc"],
            "uar": values["uar"],
            "round": None,
            "normalization": "paper-reported",
            "artifact": "attached paper Table 1",
        })

    ser_fields = ["source", "fold", "defense", "epsilon", "accuracy", "uar", "round", "normalization", "artifact"]
    with (output / "ser_fold_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ser_fields)
        writer.writeheader()
        writer.writerows(ser_rows)

    attack_rows: list[dict] = []
    for attack_path in sorted(runs.glob("fold*/attack*.json")):
        result = load_json(attack_path)
        count_private = result.get("observed_samples_per_private_client", {})
        count_shadow = result.get("observed_samples_per_shadow_client", {})
        attack_rows.append({
            "source": "reproduction",
            "fold": result.get("fold"),
            "defense": result.get("defense"),
            "epsilon": result.get("epsilon"),
            "requested_updates": result.get("num_samples_per_client"),
            "attack_model_downsampled": result.get("attack_model_downsampled"),
            "success_rate": result.get("success_rate"),
            "uasr": result.get("unweighted_average_success_recall"),
            "private_updates_min": count_private.get("min"),
            "private_updates_mean": count_private.get("mean"),
            "private_updates_max": count_private.get("max"),
            "shadow_updates_min": count_shadow.get("min"),
            "shadow_updates_mean": count_shadow.get("mean"),
            "shadow_updates_max": count_shadow.get("max"),
            "artifact": attack_path.relative_to(project).as_posix(),
        })
    attack_fields = [
        "source", "fold", "defense", "epsilon", "requested_updates", "attack_model_downsampled",
        "success_rate", "uasr", "private_updates_min", "private_updates_mean", "private_updates_max",
        "shadow_updates_min", "shadow_updates_mean", "shadow_updates_max", "artifact",
    ]
    with (output / "attack_fold_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=attack_fields)
        writer.writeheader()
        writer.writerows(attack_rows)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in ser_rows:
        if row["source"] == "reproduction":
            groups[(row["defense"], row["epsilon"])].append(row)
    lines = [
        "# Reproduction results summary",
        "",
        "Generated from local run JSON artifacts. Fold assignments, model substitutions, and the UDP accounting assumptions are described in `reproduction_decisions.md`.",
        "",
        "## SER private-client validation",
        "",
        "| Defense | Epsilon | Folds present | Mean ACC | SD ACC | Mean UAR | SD UAR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (defense, epsilon), rows in sorted(groups.items(), key=lambda item: (item[0][0], float(item[0][1] or 0))):
        acc = [float(row["accuracy"]) for row in rows if row["accuracy"] is not None]
        uar = [float(row["uar"]) for row in rows if row["uar"] is not None]
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        sd = lambda xs: (sum((x - mean(xs)) ** 2 for x in xs) / len(xs)) ** 0.5 if xs else float("nan")
        lines.append(f"| {defense} | {epsilon if epsilon is not None else '—'} | {len(rows)} | {mean(acc):.4f} | {sd(acc):.4f} | {mean(uar):.4f} | {sd(uar):.4f} |")

    paper_acc = sum(item["acc"] for item in PAPER_BASELINE.values()) / 3
    paper_uar = sum(item["uar"] for item in PAPER_BASELINE.values()) / 3
    lines.extend([
        "",
        f"Paper baseline three-fold mean: ACC={paper_acc:.4f}, UAR={paper_uar:.4f}.",
        "",
        "## Gender inference attack",
        "",
        f"Attack artifacts currently present: {len(attack_rows)}. Review `attack_fold_comparison.csv` for requested and actual update counts; the 100-requested setting can be capped by the available per-client history.",
        "",
        "Paper unprotected folds (SR/UASR): " + "; ".join(
            f"F{fold} {v['sr']:.3f}/{v['uasr']:.3f}" for fold, v in PAPER_BASELINE.items()
        ) + ".",
        "",
        "## Interpretation",
        "",
        "Numerical differences are not automatically defence effects. The paper does not publish exact fold seeds or the Voice-Ind/Gender-Ind embedding/TTS checkpoints and public pools. Local substitutions are identified in the methods and decisions files.",
        "",
    ])
    (output / "reproduction_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output / 'ser_fold_comparison.csv'}")
    print(f"Wrote {output / 'attack_fold_comparison.csv'}")
    print(f"Wrote {output / 'reproduction_summary.md'}")


if __name__ == "__main__":
    main()
