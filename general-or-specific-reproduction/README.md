# Reproduction: General or Specific?

Working reproduction of Tan, Cao, Li and Yoshikawa, ICASSP 2023, *General or Specific? Investigating Effective Privacy Protection in Federated Learning for Speech Emotion Recognition*.

## Target experiment

Reproduce the client gender inference attack against FedAvg SER and compare the unprotected baseline with:

1. User-level differential privacy (UDP).
2. Voice-Ind (voiceprint anonymisation followed by speech synthesis).
3. Gender-Ind (sample a replacement gender embedding from a public pool with probability proportional to `exp(-epsilon * angular_distance)`, then resynthesise the clip).

The result figures compare all three methods at `epsilon = 1, 5, 10, 25, 50`; attack evaluation also varies the number of observed gradients/updates (5, 10, 100).

The attacker is an honest-but-curious server. It observes client updates, trains on shadow clients, and predicts gender from the first layer. Report attack success rate (SR) and unweighted average success recall (UASR); report SER accuracy (ACC) and unweighted average recall (UAR).

## Analysis and results notebook

- Open [`notebooks/results_viewer.ipynb`](notebooks/results_viewer.ipynb) to view the currently available SER and attack results. The notebook loads local run JSON files and plots available results; rerun it after adding new runs.
- Read [`paper_analysis_for_phd.md`](paper_analysis_for_phd.md) for a detailed account of the paper's methods, materials, threat model, system architecture, experimental design, findings, limitations, and research gaps mapped to the PhD topic.
- Rebuild the notebook with `python scripts/build_results_notebook.py`.

Run outputs under `outputs/` are local and excluded from version control. Use `scripts/summarize_reproduction.py` to write versionable fold-comparison tables under `results/`, then rebuild and run the notebook to inspect them.

## Published settings to reproduce

- Dataset: CREMA-D; actors are the clients.
- SER labels: use CREMA-D `MultiModalVote` labels `N/S/H/A` (neutral, sad, happy, angry), matching the authors' cited attack/UDP reference implementation. The 2023 paper names the four labels differently (“neutral, pleased, joyful, and furious”), so this mapping is recorded as a reproducibility deviation rather than silently equating those terms.
- SER network: two fully connected hidden layers (256 and 128), ReLU, dropout 0.2.
- FL: FedAvg; 200 rounds; 10% client participation per round; one local epoch; batch size 20; learning rate 0.0005.
- Per client, split 80% train / 20% validation.
- 30 actors are shadow clients; the rest are private/test clients.
- Evaluate the attack using 5, 10, and 100 observed first-layer gradients/updates.

The unprotected table reports (attack SR/UASR; SER ACC/UAR): Fold 1 `0.837/0.829; 0.663/0.595`, Fold 2 `0.847/0.838; 0.666/0.601`, Fold 3 `0.817/0.791; 0.656/0.619`. These are the comparison targets for the first baseline run.

## Important implementation gaps

The paper does not publish the three fold actor assignments/seeds or the exact Voice-Ind/Gender-Ind embedding datasets and ESPnet checkpoints. The cited reference resolves the EmoBase representation, per-speaker normalization, attack network, and UDP clipping/noise calculation. All remaining pool/checkpoint and split substitutions are recorded in `reproduction_decisions.md` rather than silently treated as exact replication.

The Kaggle download is included at `../data/crema-d/content/CREMA-D/`: 7,442 WAV clips across 91 actors, with `VideoDemographics.csv` and `processedResults/summaryTable.csv`. The dataset remains under its original ODbL/Database Contents License terms (see `../data/README.md`). The official reference implementation uses the multimodal crowd-vote labels, not the emotion code embedded in the filename; its map retains N/S/H/A and filters other labels. We follow that evidence-based mapping and record the paper's terminology discrepancy.

## Data

The Kaggle dataset is downloaded to `../data/crema-d/content/CREMA-D/`. It contains audio and video; the experiments use audio only. This repository tracks the dataset with Git LFS. Do not commit model checkpoints, extracted features, virtual environments, credentials, or run outputs.

The project `.venv` inherits the existing Anaconda CPU PyTorch/torchaudio packages and isolates OpenSMILE 2.2.0. Use `.venv/Scripts/python.exe` on Windows. This machine currently has no CUDA-enabled PyTorch device, so full FL sweeps will run on CPU.

To download yourself with KaggleHub from this directory:

```powershell
python -m pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('orvile/crema-d-emotional-multimodal-dataset', output_dir='../data/crema-d'))"
```

The initial manifest has already been generated. After feature extraction finishes, the baseline run commands are:

```powershell
.\.venv\Scripts\python.exe scripts\federated_train.py --fold 1 --role shadow --defense none
.\.venv\Scripts\python.exe scripts\federated_train.py --fold 1 --role private --defense none
.\.venv\Scripts\python.exe scripts\attack_eval.py --shadow outputs\runs\fold1\shadow\none\updates.npz --private outputs\runs\fold1\private\none\updates.npz --num-samples 5
```

UDP uses the same training command with `--defense udp --epsilon 1` (repeat for 5, 10, 25, 50). Voice-Ind and Gender-Ind require separate transformed-audio features for each epsilon before training; the runner refuses those methods unless `--feature-privacy` matches the requested defense.

## Reference implementation

The authors' [public SER attribute-inference and UDP reference implementation](https://github.com/usc-sail/fed-ser-leakage) informs the SER architecture, per-speaker EmoBase normalization, UDP assumptions, first-layer attack CNN, and update-count cap when fewer than 5/10/100 observations are available. Run `scripts/run_core_grid.ps1 -Force` to execute the unprotected and UDP settings across all three folds. The per-run metadata records the training and privacy parameters.

To prepare speaker and gender embeddings, install the optional extractor and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-embeddings.txt
.\.venv\Scripts\python.exe scripts\extract_speaker_embeddings.py outputs\manifest.csv
.\.venv\Scripts\python.exe scripts\train_gender_embedding.py outputs\manifest.csv outputs\embeddings\utterance_xvectors.npz
.\.venv\Scripts\python.exe scripts\prepare_embedding_replacements.py
```

Voice-Ind uses speaker x-vectors. Gender-Ind uses a distinct 512-dimensional gender-supervised embedding trained from those x-vectors and CREMA-D actor sex labels. The held-out-actor gender classification score is saved with the embedding artifact. Both mechanisms select from CREMA-D actor-mean candidate pools with angular distance and exponential sampling at epsilon 1/5/10/25/50; the source actor is excluded. These public pool/model choices substitute for assets the paper does not identify.

The isolated TTS environment uses NumPy 2 and SciPy 1.15+, separate from the main SER/OpenSMILE environment. The source-conditioned synthesis grid uses each original waveform as teacher-forced acoustic features while replacing the speaker vector:

```powershell
C:\Users\charl\anaconda3\python.exe -m venv --system-site-packages .venv-espnet
.venv-espnet\Scripts\python.exe -m pip install -c constraints-espnet.txt -r requirements-espnet.txt
.venv-espnet\Scripts\python.exe scripts\synthesize_replacement_grid.py outputs\manifest.csv --method voice-ind --epsilons 1 5 10 25 50
.\.venv\Scripts\python.exe scripts\extract_emobase.py outputs\manifest.csv --audio-root outputs\audio\voice-ind-e10 --output outputs\features\voice-ind-e10.npz --feature-cache outputs\features\synth-emobase-cache.npz
```

This is closer to the paper's source-acoustic-feature plus replacement-embedding pipeline than transcript-only TTS, but it substitutes a public LibriTTS VITS checkpoint for the paper's ESPnet `egs/libritts/tts1` model. Run the same synthesis and feature extraction for Gender-Ind and each epsilon. Exact checkpoints, vocoder settings, and the external public embedding databases are not specified in the paper, so this remains a documented approximation rather than an exact numerical reproduction.

Reference sources: [target 2023 paper (DOI)](https://doi.org/10.1109/ICASSP49357.2023.10096844), [authors' public SER attack/UDP implementation](https://github.com/usc-sail/fed-ser-leakage), [UDP paper](https://arxiv.org/abs/2204.02500), and [Voice-Ind paper](https://arxiv.org/abs/2004.07442).

## Status

- [x] Locate and inspect the target paper.
- [x] Identify and clone the public attack/UDP reference implementation.
- [x] Download CREMA-D and inventory the audio/metadata.
- [x] Prepare a deterministic three-fold manifest with 30 shadow actors per fold, the rest private, and each actor's 80/20 local train/validation partition.
- [x] Extract EmoBase features (4,799 x 988) and create the adapted FedAvg/UDP and attack runners.
- [x] Reconcile reference model, UDP, update-observation, and label choices with the target paper; document inferred details.
- [x] Extract SpeechBrain speaker x-vectors; train a separate held-out-actor gender embedding model; sample replacement vectors over the epsilon grid.
- [x] Validate one source-conditioned ESPnet VITS resynthesis clip.
- [ ] Complete source-conditioned Voice-Ind/Gender-Ind synthesis and feature extraction over the epsilon grid.
- [ ] Evaluate SER and gender attack metrics across all three folds and defense settings.
- [ ] Compare per-fold outputs with the paper and document nonrecoverable split/checkpoint details, including actual update counts for the 100-requested setting.
