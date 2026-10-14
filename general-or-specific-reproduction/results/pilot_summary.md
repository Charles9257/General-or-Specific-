# Pilot results and current deviations

## Data and setup

- CREMA-D: 7,442 WAV files, 91 actors, Kaggle archive stored at `../data/crema-d/`.
- The reference code's `MultiModalVote` mapping retains N/S/H/A (neutral, happy, sad, angry), leaving 4,799 clips. This differs from the 2023 paper's printed class names.
- EmoBase functionals: 4,799 clips x 988 features.
- Fold 1: 30 shadow actors; 61 private actors. Each actor has an 80/20 local train/validation split.
- FedAvg: 200 rounds, 10% clients per round, one local epoch, batch 20, Adam learning rate 0.0005.

## First fold

| Setting | Private-client validation accuracy | UAR |
|---|---:|---:|
| No protection | 0.7063 | 0.7009 |
| UDP, epsilon 10 | 0.6867 | 0.6815 |

The paper's fold-1 unprotected table reports SER accuracy 0.663 and UAR 0.595. The current fold is higher; likely contributors include the inferred feature/normalization recipe and the random actor split. It is not yet a reproduction match.

## Gender attack

The reference attack CNN on full 256x988 update maps did not complete within a practical CPU-only run (no CUDA device is available). The explicitly downsampled CPU pilot, after 40 epochs, gave SR 0.6492 and UASR 0.6352 for five updates per client. This is a runtime-deviation diagnostic only and must not be compared as the paper's result. The attack script's default remains the full reference CNN.

## Still needed

- Run all three folds and the full attack metrics for 5, 10, and 100 updates.
- Sweep UDP at epsilon 1, 5, 10, 25, and 50.
- SpeechBrain VoxCeleb x-vector extraction has now completed for all 4,799 clips (512 dimensions); actor means for 91 actors are saved as candidate vectors. Replacement embeddings are sampled for Voice-Ind and Gender-Ind at epsilon 1, 5, 10, 25, and 50.
- Synthesize Voice-Ind and Gender-Ind audio, re-extract features, and train/evaluate those settings. ESPnet imports in an isolated environment, but its 375 MB public LibriTTS x-vector VITS checkpoint downloaded at about 30–70 kB/s and was stopped before completion. The script uses transcript-conditioned synthesis, an explicit deviation. Audio-generation results are not available yet.
- Resolve the emotion-name and feature/normalization discrepancies against author materials if available.
