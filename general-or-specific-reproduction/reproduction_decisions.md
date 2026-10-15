# Reproduction decisions and deviations

This file separates the target paper's reported setup from implementation choices inferred from the paper and the cited public reference code. The attached paper's detailed setup appears on pages 2–4; the reference implementation is the USC SAIL `fed-ser-leakage` repository.

| Topic | Paper/reference evidence | Reproduction decision | Status |
|---|---|---|---|
| Dataset | CREMA-D, 7,442 original clips, 91 actors | Audio-only; `MultiModalVote` labels, retaining N/S/H/A; 4,799 usable clips, 91 actors | Prepared |
| Emotion names | Paper calls the four classes neutral, pleased, joyful, furious; reference maps N/S/H/A to neutral, sad, happy, angry | Follow the reference mapping and retain the terminology mismatch as a deviation | Documented |
| Features | Paper does not state the SER feature representation; cited reference uses OpenSMILE EmoBase | 988-dimensional EmoBase functionals | Inferred from reference |
| Normalization | Reference preprocess code uses speaker-wise z-normalization calculated from the speaker's clips before local train/validation split | Match per-speaker all-clip statistics; local model fitting still uses the paper's 80/20 split | Aligned to reference |
| SER model | Two hidden layers, 256 and 128; ReLU; dropout 0.2 | Four-class MLP with the stated architecture | Implemented |
| FL | FedAvg; 200 rounds; 10% clients each round; one local epoch; batch 20; learning rate 0.0005 | Same settings, actor as client; 30 shadow actors, remainder private per fold | Implemented |
| Folds | Paper reports Fold 1/2/3 but does not publish the split seeds/actor IDs | Three deterministic actor-level splits with seed 2023; compare as independent fold runs, not identical assignments | Implemented; split deviation disclosed |
| UDP | Gaussian noise; equation uses sensitivity, q, T, delta, epsilon. Reference code clips gradients at 0.25, uses delta=0.5, sensitivity `2*clip/local_data_size`, and noise scale `sensitivity*sqrt(2*q*T*ln(1/delta))/epsilon` | Match reference calculation, add noise to client model parameters before sample-weighted FedAvg, and derive first-layer pseudo-gradients from those noisy uploads | Implemented; epsilon is not claimed as a verified DP guarantee |
| Voice-Ind | Default Voice-Ind; x-vector distance mechanism; ESPnet source-acoustic-feature + replacement-voice synthesis | SpeechBrain VoxCeleb speaker x-vectors; actor-mean CREMA-D candidate pool; angular distance `acos(cosine)/pi`; epsilon 1/5/10/25/50; source-conditioned ESPnet VITS synthesis | Approximation; public pool and checkpoint differ/are unspecified |
| Gender-Ind | Gender embedding is an x-vector model trained on gender IDs rather than speaker IDs; exponential selection by angular distance; source speech plus protected vector is synthesized | Train a distinct 512-D gender-supervised projection on speaker x-vectors and CREMA-D sex labels, hold out actors for encoder validation, build actor-mean candidate vectors, exclude source actor, sample over epsilon 1/5/10/25/50 | Approximation; no author checkpoint/public gender dataset identified |
| Attack sample counts | Requested 5/10/100 first-layer gradients; upstream evaluation uses all available observations if fewer than requested and repeats random selection trials | Match the upstream cap and 10 repeated per-client samples; every attack artifact records requested count and observed update min/max/mean for shadow/private actors | Implemented |
| Attack input | Reference “fake” gradients for FedAvg are computed from global-vs-client parameter difference divided by learning rate and estimated local steps | Store first-layer weight/bias pseudo-gradients from each actor upload; attack CNN is the reference architecture unless marked `fast` | Implemented; fast CNN is diagnostic only |

## Interpretation constraints

- The three fold seed/speaker assignments are not available in the paper, so matching the published per-fold values exactly cannot be guaranteed even with the same high-level setup.
- The emotion label names in the paper and the referenced CREMA-D `MultiModalVote` mapping do not match. The reproduction follows the public reference implementation and reports the mismatch.
- The Voice-Ind and Gender-Ind mechanisms depend on public embedding sets and trained checkpoints that the paper does not fully identify. The local pools/checkpoints are named substitutes, not the original assets.
- Source-conditioned VITS inference retains source acoustic features through teacher forcing and replaces speaker conditioning, but the public LibriTTS VITS checkpoint is not necessarily the paper's ESPnet `egs/libritts/tts1` checkpoint/vocoder.
- UDP uses the reference implementation's `delta=0.5` and sensitivity/noise calculation. The result is a method replication of those settings, not a validated privacy guarantee: adjacency, clipping interpretation, sampling, and composition assumptions require separate audit.
- With 200 rounds and 10% participation, a client usually contributes far fewer than 100 updates. A “100 requested” attack therefore uses each client's full available history when below 100, exactly as in the cited reference evaluation; the artifact reports the actual count distribution.
- Validation accuracy and UAR are computed after every global round on the local held-out portions of clients. They are not a pooled external test set.

## Reporting plan

For every fold and privacy setting, report private-client SER ACC/UAR and gender attack SR/UASR for requested 5/10/100 updates, alongside actual update counts. Compare each local fold against the paper's reported three-fold unprotected table and defense curves. Keep the `fast` attack diagnostic separate. Report mean and spread over folds/seeds only after all runs exist; identify all unavailable or substituted assets instead of treating local results as exact replication.
