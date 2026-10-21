# Paper analysis for the PhD research

## Purpose and scope

This note explains Tan, Cao, Li and Yoshikawa's ICASSP 2023 paper, *General or Specific? Investigating Effective Privacy Protection in Federated Learning for Speech Emotion Recognition*, and relates it to the proposed PhD research, **“Paralinguistic Inference Attacks on Emotion Recognition Systems: Threat Modelling and Federated Privacy-Preserving Defences.”** It distinguishes the paper's reported method and findings from choices in this working reproduction. It is a research synthesis, not a claim that the current implementation has reproduced every result.

The paper's central question is whether a federated speech emotion recognition (SER) system leaks a client's gender through the model updates visible to an honest-but-curious server, and how three protections compare: user-level differential privacy (UDP), Voice-Indistinguishability (Voice-Ind), and the authors' Gender-Indistinguishability (Gender-Ind). Its main argument is that a defence tailored to the sensitive attribute may preserve SER utility better than perturbing all model parameters, while reducing the gender inference attack.

## 1. Research problem, motivation, and threat model

Federated learning keeps raw training examples on clients, but each client still transmits model information. The paper treats that information as a possible side channel: a server that follows the aggregation protocol can inspect an individual client's model gradients or updates and infer a sensitive property that was not the SER target. This is a **property-inference** or **attribute-inference** threat, distinct from reconstructing a particular utterance or deciding whether a record was in the training set.

The adversary is an **honest-but-curious central server**. It receives client updates during ordinary horizontal FL and tries to infer client gender. It has a shadow population whose gender labels and local training data are available to it; it trains an attack model on shadow clients' updates and labels, then applies that model to updates from private clients. The attack therefore assumes that shadow data and private data are sufficiently similar for transfer. The paper also notes that an external observer could conceivably obtain update information, but its experiments focus on the server-side scenario.

For the PhD, this gives a useful baseline threat-model template:

| Threat-model element | Paper's setting | Thesis implication |
|---|---|---|
| Protected subject | A speaker/client | State whether privacy is utterance-, session-, or person-level. |
| Adversary | Honest-but-curious FL server | Model its protocol access, visibility of per-client updates, auxiliary data, and compute. |
| Sensitive property | Gender | Extend to emotion-related traits and other paralinguistic attributes, with careful construct definitions. |
| Observation | First-layer gradients/updates over training | Inventory every observable: individual update, aggregate, multiple rounds, metadata, and model outputs. |
| Attack learning | Shadow-client supervised inference | Test transfer across actors, datasets, collection settings, and model families. |
| Utility | SER accuracy and UAR | Measure task utility and subgroup performance alongside privacy. |

The threat model should avoid treating demographic gender labels as a simple biological ground truth. CREMA-D records binary actor sex categories; this is a dataset attribute, not a complete account of gender identity. A thesis should use precise operational language and discuss the consequences of its label source.

## 2. Materials and dataset

### CREMA-D

The paper uses the Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D): 7,442 original audiovisual clips from 91 actors (48 male and 43 female actors, as described in the paper). Actors speak fixed sentences with acted emotional expressions. The source dataset includes multiple emotion categories and intensity levels. The paper's experiment selects four emotion labels, which it calls **neutral, pleased, joyful, and furious**.

The working project uses audio WAV recordings for SER and the bundled demographic and crowd-vote tables. It follows the label mapping used by the cited public attack implementation: `N/S/H/A` mapped to `neutral/sad/happy/angry`, with other vote labels excluded. The resulting manifest contains 4,799 clips from 91 actors. This differs from the target paper's printed emotion names and is a material reproducibility caveat: the paper does not document a mapping that resolves the terminology. The project records this discrepancy instead of implying that `sad/happy/angry` exactly mean `pleased/joyful/furious`.

### Features and preprocessing

The paper describes a two-hidden-layer feed-forward SER network but does not fully specify its input representation or feature extraction. The reproduction uses OpenSMILE EmoBase functionals (988 dimensions), based on the cited public SER attack implementation. That reference code normalizes within each speaker using all clips assigned to the speaker before the 80/20 local train/validation split, so the current training implementation follows that rule and replaces non-finite standardized values with zero. The target paper's exact three fold seeds/splits remain unavailable; the reproduction records its deterministic actor split and seed rather than claiming the same random folds.

For Voice-Ind, the paper's conceptual pipeline extracts an x-vector, selects a replacement from a public x-vector pool, then synthesizes speech using both source acoustic features and the replacement vector. The reproduction uses SpeechBrain VoxCeleb x-vectors and CREMA-D actor-mean candidates, then uses a public ESPnet VITS model in teacher-forcing mode with the original waveform as the acoustic source and the replacement vector as speaker conditioning. For Gender-Ind, the paper explicitly calls for an x-vector model trained with gender IDs rather than speaker IDs. The reproduction therefore trains a separate gender-supervised 512-D projection on the SpeechBrain x-vectors and CREMA-D actor-sex metadata, validates it on held-out actors, and constructs a gender-embedding candidate pool. These are documented substitutes: the paper does not identify its public embedding dataset, gender encoder checkpoint, or complete ESPnet model/vocoder checkpoint configuration.

## 3. System architecture and implementation

### Federated SER model

The paper uses horizontal FL with **FedAvg**. Clients have the same feature space and emotion-label space; each actor is treated as one client in this reproduction. The SER classifier has two fully connected hidden layers of 256 and 128 units, ReLU activations, dropout 0.2, and an output layer for four emotions. The local learning rate is 0.0005; each sampled client trains for one local epoch with batch size 20. The server aggregates local models into a global model. The experiment uses 200 rounds and samples 10% of clients per round; 30 actors form the shadow group and the rest the private group.

Each global round samples 10% of the participating client population (`q = 0.1`) and the experiment runs for `T = 200` rounds. In the paper's split, 30 speakers are used for shadow training and the remaining speakers for private training. Within each client, 80% of the clips are used for local training and 20% for validation. The reproduction builds three deterministic actor folds, while preserving the 30-shadow-actor setup per fold.

The server-side attack is motivated by the first-layer gradient's sensitivity to input distributions. The paper says it uses the attack model from Feng et al. and focuses on first-layer gradients; it evaluates requested sample counts 5, 10, and 100. The upstream reference chooses up to the requested number of available per-speaker observations, so a 100-requested result can use fewer than 100 when the client was sampled in fewer rounds. It uses repeated random selections per client. The working code adapts the upstream reference CNN, groups client observations by actor, normalizes gradient maps, and trains a binary gender classifier on shadow clients. Under 200 rounds and 10% participation, it records and reports actual update availability so 100 requested is not mistaken for 100 observed updates. For FedAvg, these are the reference's “fake” gradients reconstructed from parameter deltas, not raw per-minibatch gradients.

### Three protection mechanisms

**1. User-level differential privacy (UDP).** The client perturbs its update with Gaussian noise. In broad form, an update `g_k` becomes `g_k + N(0, sigma_k^2 I)`, with noise calibrated from a sensitivity bound, sampling rate, total rounds, `delta`, and target `epsilon`. UDP is general-purpose: it perturbs the whole update rather than a single attribute representation. Smaller epsilon usually means more noise and stronger nominal privacy, subject to the validity of the accounting assumptions. The paper uses the UDP method from Feng et al. The local reproduction adopts assumptions from that reference implementation (including gradient clipping and a specific sensitivity/noise calculation); its chosen `delta` and resulting guarantee must be audited before calling the run a formally valid DP guarantee. Merely naming a run “epsilon = 10” does not establish a correct person-level guarantee.

**2. Voice-Ind.** Voice-Ind was originally proposed to protect speaker voiceprints in speech release. The mechanism extracts a speaker x-vector, selects a replacement vector from a public pool with probability that decreases exponentially with angular distance, then synthesizes speech conditioned on the replacement voice representation. The target paper uses its default Voice-Ind configuration. Because voiceprints encode more than gender (for example speaker identity and demographic correlates), this is a broader representation-level defence than a gender-only mechanism. The paper's explanation suggests its protection against gender inference is weaker than UDP at comparable plotted settings, while SER utility is less affected than with strong UDP.

**3. Gender-Ind.** The paper proposes a gender-specific indistinguishability notion. A gender embedding is extracted from speech; a replacement is sampled from public gender embeddings with probability proportional to `exp(-epsilon * angular_distance)`; speech is resynthesized using the selected embedding and the original speech content. The paper gives an algorithm with an embedding extractor and ESPnet speech synthesizer. The intention is to perturb the gender-bearing representation while retaining other speech information relevant to emotion. Its formal definition bounds the output likelihood ratio for two possible input gender embeddings as a function of epsilon and angular distance. This is a proposed metric-based privacy definition; it is not automatically interchangeable with standard DP, and its privacy interpretation depends on the candidate space, distance, embedding model, pool composition, and adversary knowledge.

All three methods alter information at different points: UDP perturbs model updates, while Voice-Ind and Gender-Ind transform speech before SER training. Consequently, an epsilon value across methods is not a common privacy-strength unit. The paper itself cautions that equal epsilon values correspond to different intensities for different methods.

### Software path in this reproduction

The project uses scripts to (1) create actor-based folds/manifests, (2) extract EmoBase features, (3) train shadow and private FedAvg runs and save first-layer update snapshots, (4) train/evaluate the gender attack, (5) extract and sample replacement embeddings, and (6) synthesize transformed audio and re-extract features for Voice-Ind/Gender-Ind. UDP is applied during local training. Source-conditioned synthesis, feature extraction, and SER utility sweeps have completed for all three folds and five epsilon settings. The remaining attack evaluations use the CPU-downsampled diagnostic network because the full reference CNN is impractical on this CPU-only host.

## 4. Experimental design and outcome measures

The paper compares an unprotected FL-SER baseline with UDP, Voice-Ind, and Gender-Ind. UDP is evaluated on epsilon values `{1, 5, 10, 25, 50}`; the plotted privacy-strength sweeps are compared with attack observations based on 5, 10, and 100 gradients/updates. The baseline includes three folds. The paper reports attack **success rate (SR)** and **unweighted average success recall (UASR)**, plus SER **accuracy (ACC)** and **unweighted average recall (UAR)**.

- **SR / attack accuracy:** fraction of test clients whose gender is correctly inferred.
- **UASR / macro recall:** average of male recall and female recall, so the majority class does not dominate the score.
- **ACC:** fraction of held-out utterances with the correct emotion prediction.
- **UAR:** mean recall over the four emotion classes, giving each class equal weight.

The paper's unprotected results are:

| Fold | Gender attack SR | Gender attack UASR | SER ACC | SER UAR |
|---|---:|---:|---:|---:|
| 1 | 0.837 | 0.829 | 0.663 | 0.595 |
| 2 | 0.847 | 0.838 | 0.666 | 0.601 |
| 3 | 0.817 | 0.791 | 0.656 | 0.619 |

The paper's figures show the privacy-utility curves. In its discussion, it reports that Voice-Ind's attack success is about two percentage points higher than UDP in a comparison, and that Gender-Ind's attack success is lower than the other defences through epsilon at most 25. SER accuracy is described as being affected only slightly by Voice-Ind and Gender-Ind; strong UDP noise can damage utility substantially. At epsilon 50, attack success is higher than at more protective settings, illustrating that epsilon is method-specific and that weaker perturbation permits more leakage. The figures' point values should be transcribed directly from the original plots before using them as numerical claims; this note does not invent values that are not clearly tabulated in the paper text.

The working implementation logs per-round SER validation metrics and writes attack evaluation JSON. The results notebook reads the complete three-fold baseline/UDP/Voice-Ind/Gender-Ind grid and summarizes SER utility and gender-attack outcomes by fold, epsilon, and requested update count. It marks the downsampled CPU attacker and substitutes explicitly.

## 5. What the current reproduction has and has not established

The complete local no-protection runs report three-fold mean SER ACC/UAR of **0.7136/0.6808**, compared with **0.6617/0.6050** in the paper's table. UDP mean ACC/UAR ranges from **0.5462/0.5053** at ε=1 to **0.7051/0.6771** at ε=50. Voice-Ind utility is much lower: across ε=1/5/10/25/50 its mean ACC ranges **0.2986–0.3360** and UAR **0.1961–0.2210**. Gender-Ind mean ACC ranges **0.3202–0.3471** and UAR **0.2552–0.2839**. Fold values, standard deviations, and all attack counts are in `results/ser_fold_comparison.csv`, `results/attack_fold_comparison.csv`, and the results notebook.

Across epsilon and requested update-count settings, the fast/downsampled local attacker's mean SR/UASR is **0.6291/0.6242** on the local no-protection runs, **0.5907/0.5888** for UDP, **0.5936/0.5916** for Voice-Ind, and **0.6176/0.6188** for Gender-Ind. These averages summarize different epsilon settings and are only diagnostics: the attack CNN was downsampled for CPU, whereas the paper's mean unprotected SR/UASR is **0.8337/0.8193** using its reference attack. Do not compare those local and paper attack scores as like-for-like results.

The three-defense grid is complete as a **close methodological reproduction**, not an exact numerical reproduction. The paper's split seeds/actor assignments and embedding/TTS checkpoints are unavailable; the local targeted defenses use a CREMA-D candidate pool and a public LibriTTS VITS checkpoint. Low Voice-Ind/Gender-Ind utility suggests the resynthesis/feature pipeline or model recipe has a substantial domain or implementation mismatch; the present runs do not identify which factor caused it. The full reference CNN remains impractical on this CPU-only host. No claim is made that the paper's headline privacy-utility ranking has been reproduced.

## 6. Discussion and interpretation

The paper's theoretical and empirical distinction is between a general mechanism and a task-specific mechanism. UDP tries to bound information from the whole client contribution, which can suppress useful emotion information as well as gender information. Voice-Ind aims at speaker representation and therefore protects a broader attribute set. Gender-Ind targets the representation used for gender inference, with the intended advantage of preserving emotion-relevant content. The reported utility advantage is plausible, but depends on embedding disentanglement, synthesizer fidelity, the threat model, and how privacy strength is calibrated.

For the PhD, the most valuable reading is not “specific is always better.” The defensible hypothesis is: **a defence aligned to a precisely defined sensitive property may improve the privacy-utility frontier for a specified attacker and task**. That proposition must be tested against adaptive attackers and multiple paralinguistic targets. A defence may lower one gender classifier's score while leaving age, identity, accent, health, or emotion inference intact; it may also shift speech quality or group performance. Report those as separate outcomes.

The attack result also depends on the attacker’s auxiliary population. Actor-disjoint shadow/private splits are useful, but an attacker with a larger or more diverse public speech corpus may generalize better. The number and timing of observed rounds matter; cumulative exposure may make a seemingly weak per-round leak useful over time. The paper observes a rise in attack success with more updates, making temporal exposure and aggregation policy important experimental dimensions.

## 7. Limitations and research gaps

The paper does not contain a standalone limitations section. The following are limitations stated or implied by its scope, separated from the paper's own empirical findings:

1. **Narrow task and sensitive attribute.** The study evaluates SER and gender inference on one acted dataset. It does not establish generality to spontaneous speech, other languages, devices, populations, or paralinguistic attributes.
2. **Small speaker population and shadow transfer.** There are 91 actors, with 30 used for shadow training. Results may depend on a relatively small, controlled actor pool and how similar the shadow and private populations are.
3. **Construct and label ambiguity.** The emotion names printed by the paper (“pleased, joyful, furious”) do not transparently match the common CREMA-D vote labels. Gender is represented as two dataset categories. Both choices affect interpretation and external validity.
4. **Incomplete method specification.** Reproducing the exact feature pipeline, model initialization, split seeds, attack architecture/training, full Voice-Ind configuration, gender-embedding extractor/pool, and synthesizer checkpoint is difficult from the paper alone.
5. **Privacy guarantee comparability.** UDP's formal guarantee requires precise adjacency, clipping, sampling, composition/accounting, and delta choices. Gender-Ind uses a metric over embedding space, not the same guarantee as standard DP. The same epsilon does not mean equal risk across mechanisms.
6. **Limited utility measurement.** ACC/UAR do not cover calibration, robustness, class-wise degradation, subgroup parity, speech naturalness, intelligibility, or speaker/emotion embedding quality after synthesis.
7. **Limited attacker evaluation.** One gender attack and a restricted observation are not a broad privacy audit. Adaptive, cross-model, cross-dataset, or multimodal attackers may behave differently.
8. **Operational costs.** The paper discusses processing time as a possible trade-off but does not fully characterize deployment latency, communication overhead, energy use, synthesizer failure, or user experience.

These gaps align directly with the proposed PhD. A contribution could define a systematic threat model for paralinguistic inference in FL-SER, build an attack suite that varies protected property, observation surface, rounds, auxiliary data, model family and client partition, then compare defences under common, clearly documented threat conditions. Evaluate privacy and utility jointly, include fairness and audio quality, and quantify uncertainty across seeds/folds. For each method, state exactly what property is protected and what remains outside the claim.

## 8. Suggested experimental programme for the PhD

1. **Threat-model matrix:** server vs external observer; per-client update vs aggregate; one vs many rounds; known vs unknown architecture; shadow-data overlap and domain shift.
2. **Attribute inventory:** gender/sex as dataset-coded labels, age, speaker identity, accent/region, health-related speech markers where ethically justified, and emotion-related attributes. Pre-register operational labels and exclusions.
3. **Attack baselines:** supervised shadow attacks on first-layer and full updates; update trajectory models; attacks on aggregates and model outputs; compare against majority/prior and demographic-only baselines.
4. **Defence baselines:** no protection, carefully accounted user-level DP, voiceprint anonymisation, attribute-targeted embedding perturbation, and combinations. Keep mechanisms and epsilon semantics separate.
5. **Utility and fairness:** SER ACC/UAR and class recalls; subgroup metrics; audio intelligibility/naturalness for resynthesized speech; latency and bytes sent; confidence intervals over actor splits and training seeds.
6. **Robustness:** evaluate unseen speakers and datasets, controlled-to-natural speech shift, alternate SER encoders, adaptive attack retraining, and composition over repeated rounds.
7. **Privacy reporting:** include attack SR/UASR and per-class confusion matrices, calibration, attacker advantage over prior, uncertainty intervals, and a guarantee audit for mechanisms that claim formal DP.

## 9. Conclusion

Tan et al. demonstrate that keeping raw audio local does not eliminate sensitive-attribute leakage from federated SER updates. Their study compares a general client-level DP approach with two speech-representation approaches and proposes Gender-Ind as an attribute-specific defence. They report a better utility/privacy balance for Gender-Ind in their CREMA-D experiment, while emphasizing that the protection definition should match the sensitive information of concern.

For this PhD, the paper is a strong motivating case study and a reusable experimental pattern: explicit attacker, shadow clients, update-based attribute inference, defence sweeps, and paired privacy/utility metrics. It is not yet a general answer for paralinguistic privacy. The thesis opportunity is to make the threat model broader and more realistic, make the claims and guarantees comparable, and test whether targeted defences remain effective across attributes, attackers, data domains, and operational constraints.

## Source

Tan, C., Cao, Y., Li, S., & Yoshikawa, M. (2023). “General or Specific? Investigating Effective Privacy Protection in Federated Learning for Speech Emotion Recognition.” *Proceedings of ICASSP 2023*. DOI: [10.1109/ICASSP49357.2023.10096844](https://doi.org/10.1109/ICASSP49357.2023.10096844).
