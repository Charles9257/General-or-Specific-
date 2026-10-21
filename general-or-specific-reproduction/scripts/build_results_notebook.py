"""Build the portable notebook used to inspect reproduction run artifacts."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "notebooks" / "results_viewer.ipynb"

cells = [
    nbf.v4.new_markdown_cell(
        """# Results viewer: *General or Specific?*

This notebook is the results dashboard for the three-fold CREMA-D reproduction. It loads local JSON artifacts when present and falls back to the committed fold/attack CSVs when run outputs are not available. Run it from the repository root (or a child directory); paths are resolved automatically.

## Current evidence at a glance

The current run has all three baseline and UDP SER folds, all 15 UDP settings, a five-epsilon source-conditioned audio grid for each targeted defense, and baseline/UDP attack artifacts. Baseline SER mean ACC/UAR is 0.7136/0.6808, compared with the paper's 0.6617/0.6050. UDP mean SER ACC/UAR ranges from 0.5462/0.5053 (ε=1) to 0.7051/0.6771 (ε=50). Voice-Ind/Gender-Ind audio feature extraction and their FL/attack evaluations are not yet in `outputs/runs/`; audio generation is not an evaluated defense. Local attack CNN results use a CPU-friendly downsampled diagnostic and are not directly comparable to the paper's full attack model. The generated audio itself is kept locally; its 10-setting clip counts are in [`results/audio_generation_manifest.csv`](../results/audio_generation_manifest.csv).

| Fold | Paper baseline ACC/UAR | Reproduction ACC/UAR |
|---|---:|---:|
| 1 | 0.663 / 0.595 | 0.7063 / 0.7033 |
| 2 | 0.666 / 0.601 | 0.7126 / 0.6706 |
| 3 | 0.656 / 0.619 | 0.7220 / 0.6684 |

Exact fold seeds and some method assets are unavailable in the paper, so these are reproduction comparisons, not a claim of exact numerical replication. See [`results/reproduction_summary.md`](../results/reproduction_summary.md) and [`results/attack_fold_comparison.csv`](../results/attack_fold_comparison.csv) for the generated tables.

## Methods and provenance

The target study uses CREMA-D, actor-level clients, FedAvg, a two-hidden-layer SER network, a shadow-client gender attack on first-layer updates, and UDP, Voice-Ind, and Gender-Ind. Per-speaker EmoBase normalization follows the cited reference implementation. The project details and deviations are in [`paper_analysis_for_phd.md`](../paper_analysis_for_phd.md) and [`reproduction_decisions.md`](../reproduction_decisions.md). Outputs are generated locally under `outputs/` (ignored by Git); this notebook reads matching artifacts when run alongside them. Do not interpret the fast attack diagnostic as the paper's reference-CNN result.
"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

def locate_project(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "outputs" / "runs").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError("Could not find project outputs/runs and scripts. Open this notebook inside the reproduction project.")

PROJECT_ROOT = locate_project(Path.cwd())
RUNS_ROOT = PROJECT_ROOT / "outputs" / "runs"
PAPER_BASELINE = pd.DataFrame([
    {"fold": 1, "attack_sr": 0.837, "attack_uasr": 0.829, "ser_acc": 0.663, "ser_uar": 0.595},
    {"fold": 2, "attack_sr": 0.847, "attack_uasr": 0.838, "ser_acc": 0.666, "ser_uar": 0.601},
    {"fold": 3, "attack_sr": 0.817, "attack_uasr": 0.791, "ser_acc": 0.656, "ser_uar": 0.619},
])
plt.style.use("seaborn-v0_8-whitegrid")
print(f"Project: {PROJECT_ROOT}")
print(f"Run-artifact directory: {RUNS_ROOT}")
print(f"Paper baseline folds: {len(PAPER_BASELINE)}")"""
    ),
nbf.v4.new_markdown_cell("## 1. Inventory available experiment artifacts\n\nThe inventory is generated from saved JSON files. A listed file indicates an artifact exists; it does not imply that the run matches the paper exactly. The audio inventory checks the expected five replacement-audio folders for each targeted defence; audio completion does not mean SER utility or privacy evaluation is complete."),
    nbf.v4.new_code_cell(
        """def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)

validation_files = sorted(RUNS_ROOT.glob("fold*/**/ser_validation.json"))
run_files = sorted(RUNS_ROOT.glob("fold*/**/run.json"))
attack_files = sorted(path for path in RUNS_ROOT.glob("fold*/attack*.json") if "pilot" not in path.stem)

validation_rows = []
for path in validation_files:
    run = read_json(path)
    meta_path = path.with_name("run.json")
    meta = read_json(meta_path) if meta_path.exists() else {}
    final = run.get("final", {})
    validation_rows.append({
        "fold": meta.get("fold", path.parts[-4] if len(path.parts) >= 4 else "?"),
        "role": meta.get("role", "unknown"),
        "defense": meta.get("defense", "unknown"),
        "epsilon": meta.get("epsilon"),
        "round": final.get("round"),
        "SER ACC": final.get("accuracy"),
        "SER UAR": final.get("uar"),
        "artifact": path.relative_to(PROJECT_ROOT).as_posix(),
    })

validation_df = pd.DataFrame(validation_rows)
if validation_df.empty:
    cached_path = PROJECT_ROOT / "results" / "ser_fold_comparison.csv"
    if cached_path.is_file():
        cached = pd.read_csv(cached_path)
        cached = cached[cached["source"] == "reproduction"].copy()
        validation_df = cached.rename(columns={"accuracy": "SER ACC", "uar": "SER UAR"})
        validation_df["role"] = "private"
        print("Using committed fold summary CSV; local round-history JSON files are not present.")
if not validation_df.empty:
    display(validation_df.sort_values(["fold", "role", "defense", "epsilon"], na_position="first").reset_index(drop=True).round(4))
else:
    print("No ser_validation.json files found yet.")

attack_rows = []
for path in attack_files:
    result = read_json(path)
    fold = result.get("fold")
    if fold is None:
        for part in path.parts:
            if part.startswith("fold") and part[4:].isdigit():
                fold = int(part[4:])
                break
    attack_rows.append({
        "run": path.stem,
        "updates per client": result.get("num_samples_per_client"),
        "fold": fold,
        "defense": result.get("defense"),
        "epsilon": result.get("epsilon"),
        "private update count min/mean/max": result.get("observed_samples_per_private_client"),
        "downsampled attack": result.get("attack_model_downsampled"),
        "shadow actors": result.get("shadow_actors"),
        "private actors": result.get("private_actors"),
        "attack SR": result.get("success_rate"),
        "attack UASR": result.get("unweighted_average_success_recall"),
        "artifact": path.relative_to(PROJECT_ROOT).as_posix(),
    })
attack_df = pd.DataFrame(attack_rows)
if attack_df.empty:
    cached_path = PROJECT_ROOT / "results" / "attack_fold_comparison.csv"
    if cached_path.is_file():
        cached = pd.read_csv(cached_path)
        attack_df = cached.rename(columns={
            "requested_updates": "updates per client",
            "attack_model_downsampled": "downsampled attack",
            "success_rate": "attack SR",
            "uasr": "attack UASR",
        })
        attack_df["run"] = attack_df["artifact"].fillna("").map(lambda value: Path(value).stem)
        print("Using committed attack summary CSV; local attack JSON files are not present.")
if not attack_df.empty:
    display(attack_df.round(4))
else:
    print("No attack result JSON files found yet.")"""
    ),
    nbf.v4.new_markdown_cell("## 1a. Defense pipeline completion status\n\nThis inventory distinguishes generated replacement audio from evaluated defense runs. Each audio setting should contain 4,799 clips. Feature extraction and three-fold SER/attack evaluation are required before Voice-Ind and Gender-Ind can be compared with UDP."),
    nbf.v4.new_code_cell(
        """audio_rows = []
for defense in ["voice-ind", "gender-ind"]:
    for epsilon in [1, 5, 10, 25, 50]:
        folder = PROJECT_ROOT / "outputs" / "audio" / f"{defense}-e{epsilon}"
        disk_count = len(list(folder.glob("*.wav"))) if folder.is_dir() else 0
        manifest_path = PROJECT_ROOT / "results" / "audio_generation_manifest.csv"
        manifest = pd.read_csv(manifest_path) if manifest_path.is_file() else pd.DataFrame()
        recorded = manifest[(manifest["defense"] == defense) & (manifest["epsilon"] == epsilon)] if not manifest.empty else pd.DataFrame()
        actual = disk_count if disk_count else (int(recorded.iloc[0]["generated_clips"]) if not recorded.empty else 0)
        audio_rows.append({
            "defense": defense,
            "epsilon": epsilon,
            "audio clips generated": actual,
            "expected clips": 4799,
            "audio complete": actual == 4799,
        })

audio_status_df = pd.DataFrame(audio_rows)
validation_defenses = set(validation_df["defense"].dropna().unique()) if not validation_df.empty else set()
audio_status_df["FL SER evaluated"] = audio_status_df["defense"].isin(validation_defenses)
display(audio_status_df)
print("Completed SER defense runs:", ", ".join(sorted(validation_defenses)) or "none")
print("Audio generation alone does not establish privacy or utility; transformed features, client training, and attack evaluation are required.")"""
    ),
    nbf.v4.new_markdown_cell("## 2. SER utility over federated rounds\n\nThe chart shows private-client validation metrics from every available fold/defense artifact. The final-round table and the paper comparison below are the primary summary; the curves show training dynamics."),
    nbf.v4.new_code_cell(
        """def load_rounds(path: Path):
    payload = read_json(path)
    frame = pd.DataFrame(payload.get("rounds", []))
    if frame.empty:
        return frame
    frame["artifact"] = path.relative_to(PROJECT_ROOT).as_posix()
    meta_path = path.with_name("run.json")
    meta = read_json(meta_path) if meta_path.exists() else {}
    frame["defense"] = meta.get("defense", "unknown")
    frame["epsilon"] = meta.get("epsilon")
    frame["role"] = meta.get("role", "unknown")
    frame["fold"] = meta.get("fold", "?")
    return frame

round_frames = [load_rounds(path) for path in validation_files]
round_frames = [frame for frame in round_frames if not frame.empty and (frame["role"] == "private").all()]
round_df = pd.concat(round_frames, ignore_index=True) if round_frames else pd.DataFrame()

if round_df.empty:
    print("No per-round private-client validation series are available.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for (fold, defense, epsilon), group in round_df.groupby(["fold", "defense", "epsilon"], dropna=False):
        label = f"Fold {fold} · {defense}" + (f" ε={epsilon:g}" if pd.notna(epsilon) else "")
        group = group.sort_values("round")
        axes[0].plot(group["round"], group["accuracy"], label=label, linewidth=2)
        axes[1].plot(group["round"], group["uar"], label=label, linewidth=2)
    axes[0].set(title="SER validation accuracy", xlabel="FedAvg round", ylabel="Accuracy")
    axes[1].set(title="SER validation unweighted average recall", xlabel="FedAvg round", ylabel="UAR")
    axes[1].legend(frameon=True, fontsize=9)
    fig.suptitle("Available private-client runs by fold and defense", y=1.03, fontsize=14)
    fig.tight_layout()
    plt.show()"""
    ),
    nbf.v4.new_markdown_cell("## 3. Paper baseline versus reproduction runs\n\nThe published values below are reported in the target paper's baseline table. Each local fold uses the recorded deterministic actor split and the preprocessing choices documented in `reproduction_decisions.md`; the paper's exact fold seeds and Voice-Ind/Gender-Ind assets are unavailable, so interpret gaps as reproduction differences rather than exact replication error."),
    nbf.v4.new_code_cell(
        """paper_mean = PAPER_BASELINE[["ser_acc", "ser_uar"]].mean()
local_baseline = validation_df[(validation_df["role"] == "private") & (validation_df["defense"] == "none")]
local_udp10 = validation_df[(validation_df["role"] == "private") & (validation_df["defense"] == "udp") & (validation_df["epsilon"] == 10)]

comparison_rows = []
for _, row in PAPER_BASELINE.iterrows():
    comparison_rows.append({"source": f"Paper · fold {row['fold']}", "fold": row["fold"], "SER ACC": row["ser_acc"], "SER UAR": row["ser_uar"]})
for frame, label in [(local_baseline, "Local · no protection"), (local_udp10, "Local · UDP ε=10")]:
    for _, row in frame.iterrows():
        comparison_rows.append({"source": f"{label} · fold {row['fold']}", "fold": row["fold"], "SER ACC": row["SER ACC"], "SER UAR": row["SER UAR"]})
comparison_rows.append({"source": "Paper · 3-fold mean", "fold": "mean", "SER ACC": paper_mean["ser_acc"], "SER UAR": paper_mean["ser_uar"]})
comparison_df = pd.DataFrame(comparison_rows)
display(comparison_df.drop(columns="fold").round(4))

fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(comparison_df))
width = 0.36
ax.bar(x - width/2, comparison_df["SER ACC"], width, label="Accuracy", color="#3b82f6")
ax.bar(x + width/2, comparison_df["SER UAR"], width, label="UAR", color="#f59e0b")
ax.set_xticks(x, comparison_df["source"], rotation=12, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Paper baseline and local three-fold SER metrics")
ax.legend()
fig.tight_layout()
plt.show()"""
    ),
    nbf.v4.new_markdown_cell("## 4. Gender-inference attack results\n\nThe target paper reports unprotected SR/UASR for each fold and privacy curves by requested attack-gradient count. Local attack files include the fold, defense, requested sample count, attack model type, and actual update-count distribution; keep fast/downsampled CNN results distinct from the reference CNN."),
    nbf.v4.new_code_cell(
        """paper_attack = PAPER_BASELINE[["fold", "attack_sr", "attack_uasr"]].copy()
paper_attack["source"] = "Paper baseline"
attack_comparison = paper_attack.melt(id_vars=["fold", "source"], value_vars=["attack_sr", "attack_uasr"], var_name="metric", value_name="score")

if not attack_df.empty:
    for _, row in attack_df.iterrows():
        model_tag = "fast" if row["downsampled attack"] else "reference"
        source = f"Local {row['defense']} ε={row['epsilon']} n={row['updates per client']} ({model_tag})"
        attack_comparison = pd.concat([attack_comparison, pd.DataFrame([{
            "fold": row["fold"], "source": source,
            "metric": "attack_sr", "score": row["attack SR"]
        }, {
            "fold": row["fold"], "source": source,
            "metric": "attack_uasr", "score": row["attack UASR"]
        }])], ignore_index=True)

display(attack_comparison.round(4))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, metric, title in zip(axes, ["attack_sr", "attack_uasr"], ["Attack success rate", "Unweighted average success recall"]):
    frame = attack_comparison[attack_comparison["metric"] == metric]
    for source, group in frame.groupby("source", sort=False):
        ax.scatter(group["fold"].astype(str), group["score"], s=70, label=source)
    ax.set(title=title, xlabel="Fold / diagnostic", ylabel="Score", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=20)
axes[1].legend(frameon=True, fontsize=8)
fig.suptitle("Published attack baseline and local defense runs", y=1.03)
fig.tight_layout()
plt.show()"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Interpretation and reproduction caveats

Compare the local values fold by fold with the paper table and figures. Report discrepancies as evidence about the reproduction pipeline; do not attribute them to a defense without controlling the feature, label, actor-split, checkpoint, and attack-model settings. The attack artifact records the actual number of available updates per client. When requested n=100 exceeds a client's observed history, the reference procedure uses that client's full available history.

For UDP, ε labels follow the cited reference formula and its δ=0.5 assumption; they are not presented as independently audited formal guarantees. For Voice-Ind and Gender-Ind, distinguish the paper's exact algorithms from local embedding/checkpoint substitutions. All three-fold averages should be accompanied by fold-level values and the split seed.

### Current completion checklist

- [x] Run baseline and UDP SER training for all three folds; UDP covers ε = 1, 5, 10, 25, 50.
- [x] Generate source-conditioned Voice-Ind and Gender-Ind audio for all five epsilon settings (4,799 clips per setting).
- [x] Run baseline/UDP attack diagnostics for requested n = 5, 10, 100 across folds; these use the downsampled CPU attack model.
- [ ] Extract EmoBase features from replacement audio, train the Voice-Ind/Gender-Ind SER models for all folds, then run their attacks.
- [ ] Run the full reference attack CNN or GPU equivalent before claiming direct paper-level attack-score comparability.
- [x] Keep exact-vs-inferred decisions current in `reproduction_decisions.md`.
- [ ] Add class-wise/subgroup and audio-quality analyses for the thesis study.

### Reading the paper-to-thesis analysis

See [`paper_analysis_for_phd.md`](../paper_analysis_for_phd.md) for the detailed methods, materials, system architecture, experimental design, findings, discussion, limitations, research gaps, and suggested PhD study plan. Current project status and deviations are recorded in [`reproduction_decisions.md`](../reproduction_decisions.md) and [`results/reproduction_summary.md`](../results/reproduction_summary.md).
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
