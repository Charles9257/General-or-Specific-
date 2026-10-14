# Reproduction decisions and deviations

| Topic | Paper specifies | Decision / evidence | Status |
|---|---|---|---|
| FL method | FedAvg | Use the paper's stated FedAvg setting; compare implementation against the upstream USC SAIL reference | Pending source review |
| SER input | DNN; two hidden layers, 256 and 128 | Use OpenSMILE EmoBase functionals (988 dimensions), the default feature representation in the authors' cited public reference code; z-normalize per actor using the actor's local training subset. The target paper does not state this input choice. | Chosen, report as inferred |
| Emotion labels | “neutral, pleased, joyful, furious” | The authors' cited public reference implementation uses `processedResults/summaryTable.csv` → `MultiModalVote`, mapping N/S/H/A to neutral/sad/happy/angry and filtering other votes. Use this documented mapping; flag the paper wording mismatch as a deviation. | Chosen |
| Client identity | User/client; 30 shadow speakers | Treat each actor as one client; record actor split and seed | Planned |
| UDP | Gaussian perturbation, epsilon sweep, formula includes sensitivity, sampling ratio, rounds, delta | Reference implementation clips each minibatch's full-model gradient norm at 0.25, uses `delta=0.5`, and sets sensitivity to `2*clip_threshold/local_data_size`, then scales by `sqrt(2*q*T*ln(1/delta))/epsilon`. It adds Gaussian noise to each uploaded local parameter tensor before sample-weighted FedAvg and derives the observed pseudo-gradient from the noisy client update. | Follow reference implementation; disclose these inherited assumptions |
| Voice-Ind | Voice-Ind using an exponential mechanism over speaker embeddings; target paper references ESPnet-TTS | Extract 512-D SpeechBrain VoxCeleb x-vectors; use actor-mean x-vectors from CREMA-D as a candidate pool and exclude the source actor; angular distance is arccos(cosine)/pi; use epsilon 1/5/10/25/50. ESPnet VITS transcript-conditioned synthesis script added. | Approximation prepared; ESPnet execution not yet validated |
| Gender-Ind | Gender embedding replacement using angular distance and parameter epsilon; target paper references ESPnet-TTS | Apply the same exponential mechanism to utterance x-vectors and the CREMA-D actor-mean pool with epsilon 1/5/10/25/50. The paper does not specify the public pool/checkpoint in enough detail to recover these exactly. | Approximation prepared; synthesis and gender-attack evaluation remain |
| Attack outputs | First-layer gradients/updates, 5/10/100 samples | Clarify whether snapshots are gradients or FedAvg parameter deltas in the implementation | Open |

## Planned reporting

For every defense, report attack SR/UASR and SER ACC/UAR, across the three speaker folds if recoverable, plus mean and spread across seeds. For UDP, report the precise accountant/mechanism assumptions and do not present epsilon as a guarantee unless the implementation's adjacency, clipping, sampling, and composition assumptions are verified.
