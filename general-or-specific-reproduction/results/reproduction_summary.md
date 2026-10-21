# Reproduction results summary

Generated from local run JSON artifacts. Fold assignments, model substitutions, and the UDP accounting assumptions are described in `reproduction_decisions.md`.

## SER private-client validation

| Defense | Epsilon | Folds present | Mean ACC | SD ACC | Mean UAR | SD UAR |
|---|---:|---:|---:|---:|---:|---:|
| gender-ind | 1.0 | 3 | 0.3471 | 0.0125 | 0.2839 | 0.0040 |
| gender-ind | 5.0 | 3 | 0.3202 | 0.0217 | 0.2552 | 0.0079 |
| gender-ind | 10.0 | 3 | 0.3310 | 0.0250 | 0.2687 | 0.0186 |
| gender-ind | 25.0 | 3 | 0.3232 | 0.0092 | 0.2600 | 0.0210 |
| gender-ind | 50.0 | 3 | 0.3390 | 0.0249 | 0.2731 | 0.0114 |
| none | — | 3 | 0.7136 | 0.0064 | 0.6808 | 0.0160 |
| udp | 1.0 | 3 | 0.5462 | 0.0177 | 0.5053 | 0.0098 |
| udp | 5.0 | 3 | 0.6957 | 0.0078 | 0.6529 | 0.0199 |
| udp | 10.0 | 3 | 0.7026 | 0.0128 | 0.6660 | 0.0149 |
| udp | 25.0 | 3 | 0.7032 | 0.0101 | 0.6770 | 0.0147 |
| udp | 50.0 | 3 | 0.7051 | 0.0135 | 0.6771 | 0.0101 |
| voice-ind | 1.0 | 3 | 0.2986 | 0.0357 | 0.1961 | 0.0147 |
| voice-ind | 5.0 | 3 | 0.3190 | 0.0321 | 0.2104 | 0.0101 |
| voice-ind | 10.0 | 3 | 0.3261 | 0.0205 | 0.2156 | 0.0108 |
| voice-ind | 25.0 | 3 | 0.3360 | 0.0279 | 0.2210 | 0.0124 |
| voice-ind | 50.0 | 3 | 0.3335 | 0.0209 | 0.2181 | 0.0052 |

Paper baseline three-fold mean: ACC=0.6617, UAR=0.6050.

## Gender-inference attack (three folds)

Attack evaluation artifacts completed: 145. The table reports means and population SD across the three folds; review `attack_fold_comparison.csv` for per-fold scores and actual update counts. The 100-requested setting is capped by each client's available history when necessary.

| Defense | Epsilon | Requested updates | Attack CNN | Folds | Mean SR | SD SR | Mean UASR | SD UASR |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| gender-ind | 1.0 | 5 | downsampled | 3 | 0.5962 | 0.0160 | 0.5998 | 0.0177 |
| gender-ind | 1.0 | 10 | downsampled | 3 | 0.5929 | 0.0199 | 0.5975 | 0.0200 |
| gender-ind | 1.0 | 100 | downsampled | 3 | 0.6120 | 0.0309 | 0.6173 | 0.0289 |
| gender-ind | 5.0 | 5 | downsampled | 3 | 0.6235 | 0.0159 | 0.6238 | 0.0129 |
| gender-ind | 5.0 | 10 | downsampled | 3 | 0.6186 | 0.0253 | 0.6165 | 0.0283 |
| gender-ind | 5.0 | 100 | downsampled | 3 | 0.6284 | 0.0337 | 0.6296 | 0.0336 |
| gender-ind | 10.0 | 5 | downsampled | 3 | 0.6016 | 0.0241 | 0.6047 | 0.0258 |
| gender-ind | 10.0 | 10 | downsampled | 3 | 0.6131 | 0.0460 | 0.6120 | 0.0483 |
| gender-ind | 10.0 | 100 | downsampled | 3 | 0.6120 | 0.0409 | 0.6101 | 0.0432 |
| gender-ind | 25.0 | 5 | downsampled | 3 | 0.6208 | 0.0274 | 0.6233 | 0.0216 |
| gender-ind | 25.0 | 10 | downsampled | 3 | 0.6311 | 0.0478 | 0.6310 | 0.0473 |
| gender-ind | 25.0 | 100 | downsampled | 3 | 0.6557 | 0.0535 | 0.6602 | 0.0515 |
| gender-ind | 50.0 | 5 | downsampled | 3 | 0.6235 | 0.0443 | 0.6239 | 0.0421 |
| gender-ind | 50.0 | 10 | downsampled | 3 | 0.6120 | 0.0332 | 0.6085 | 0.0289 |
| gender-ind | 50.0 | 100 | downsampled | 3 | 0.6230 | 0.0232 | 0.6241 | 0.0212 |
| none | — | 5 | reference | 1 | 0.6328 | 0.0000 | 0.6267 | 0.0000 |
| none | — | 5 | downsampled | 3 | 0.6432 | 0.0135 | 0.6376 | 0.0054 |
| none | — | 10 | downsampled | 3 | 0.6213 | 0.0269 | 0.6144 | 0.0243 |
| none | — | 100 | downsampled | 3 | 0.6230 | 0.0354 | 0.6205 | 0.0293 |
| udp | 1.0 | 5 | downsampled | 3 | 0.5301 | 0.0112 | 0.5284 | 0.0153 |
| udp | 1.0 | 10 | downsampled | 3 | 0.5514 | 0.0107 | 0.5502 | 0.0061 |
| udp | 1.0 | 100 | downsampled | 3 | 0.5847 | 0.0155 | 0.5793 | 0.0104 |
| udp | 5.0 | 5 | downsampled | 3 | 0.5284 | 0.0031 | 0.5272 | 0.0067 |
| udp | 5.0 | 10 | downsampled | 3 | 0.5639 | 0.0163 | 0.5642 | 0.0125 |
| udp | 5.0 | 100 | downsampled | 3 | 0.5792 | 0.0337 | 0.5749 | 0.0322 |
| udp | 10.0 | 5 | downsampled | 3 | 0.5530 | 0.0164 | 0.5534 | 0.0166 |
| udp | 10.0 | 10 | downsampled | 3 | 0.5995 | 0.0137 | 0.5993 | 0.0147 |
| udp | 10.0 | 100 | downsampled | 3 | 0.6175 | 0.0309 | 0.6120 | 0.0317 |
| udp | 25.0 | 5 | downsampled | 3 | 0.5689 | 0.0280 | 0.5716 | 0.0243 |
| udp | 25.0 | 10 | downsampled | 3 | 0.6180 | 0.0469 | 0.6157 | 0.0502 |
| udp | 25.0 | 100 | downsampled | 3 | 0.6393 | 0.0354 | 0.6378 | 0.0366 |
| udp | 50.0 | 5 | downsampled | 3 | 0.6208 | 0.0225 | 0.6190 | 0.0187 |
| udp | 50.0 | 10 | downsampled | 3 | 0.6443 | 0.0348 | 0.6403 | 0.0387 |
| udp | 50.0 | 100 | downsampled | 3 | 0.6612 | 0.0279 | 0.6585 | 0.0280 |
| voice-ind | 1.0 | 5 | downsampled | 3 | 0.5699 | 0.0483 | 0.5693 | 0.0455 |
| voice-ind | 1.0 | 10 | downsampled | 3 | 0.5508 | 0.0402 | 0.5487 | 0.0386 |
| voice-ind | 1.0 | 100 | downsampled | 3 | 0.5628 | 0.0386 | 0.5604 | 0.0348 |
| voice-ind | 5.0 | 5 | downsampled | 3 | 0.6153 | 0.0074 | 0.6137 | 0.0081 |
| voice-ind | 5.0 | 10 | downsampled | 3 | 0.6191 | 0.0155 | 0.6173 | 0.0159 |
| voice-ind | 5.0 | 100 | downsampled | 3 | 0.6120 | 0.0204 | 0.6137 | 0.0219 |
| voice-ind | 10.0 | 5 | downsampled | 3 | 0.6055 | 0.0384 | 0.6057 | 0.0408 |
| voice-ind | 10.0 | 10 | downsampled | 3 | 0.6230 | 0.0377 | 0.6235 | 0.0427 |
| voice-ind | 10.0 | 100 | downsampled | 3 | 0.6066 | 0.0402 | 0.6069 | 0.0431 |
| voice-ind | 25.0 | 5 | downsampled | 3 | 0.6087 | 0.0491 | 0.6058 | 0.0484 |
| voice-ind | 25.0 | 10 | downsampled | 3 | 0.6224 | 0.0559 | 0.6186 | 0.0558 |
| voice-ind | 25.0 | 100 | downsampled | 3 | 0.6175 | 0.0386 | 0.6129 | 0.0393 |
| voice-ind | 50.0 | 5 | downsampled | 3 | 0.5836 | 0.0335 | 0.5783 | 0.0312 |
| voice-ind | 50.0 | 10 | downsampled | 3 | 0.5497 | 0.0295 | 0.5453 | 0.0260 |
| voice-ind | 50.0 | 100 | downsampled | 3 | 0.5574 | 0.0483 | 0.5537 | 0.0479 |

The full reference CNN has so far been run only for the unprotected fold-1, 5-update configuration as a runtime benchmark; do not treat that single result as a completed full-CNN comparison. Downsampled CPU results are diagnostics and are not directly comparable with paper full-CNN scores. The paper's unprotected folds (SR/UASR): F1 0.837/0.829; F2 0.847/0.838; F3 0.817/0.791.

## Interpretation

UDP retains near-baseline SER utility from epsilon 5 upward, while its epsilon 1 setting incurs a clear utility drop. Voice-Ind and Gender-Ind have low SER utility in this implementation, indicating a substantial transformed-speech/model mismatch. Reduced local attack scores do not by themselves establish privacy improvement because the attack CNN is downsampled, the score aggregates are not paper-matched, and transformed-audio attack results are sensitive to the substitute embeddings/TTS. The paper does not publish exact fold seeds or the Voice-Ind/Gender-Ind embedding/TTS checkpoints and public pools; see `reproduction_decisions.md` for all deviations.
