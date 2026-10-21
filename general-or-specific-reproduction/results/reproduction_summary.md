# Reproduction results summary

Generated from local run JSON artifacts. Fold assignments, model substitutions, and the UDP accounting assumptions are described in `reproduction_decisions.md`.

## SER private-client validation

| Defense | Epsilon | Folds present | Mean ACC | SD ACC | Mean UAR | SD UAR |
|---|---:|---:|---:|---:|---:|---:|
| none | — | 3 | 0.7136 | 0.0064 | 0.6808 | 0.0160 |
| udp | 1.0 | 3 | 0.5462 | 0.0177 | 0.5053 | 0.0098 |
| udp | 5.0 | 3 | 0.6957 | 0.0078 | 0.6529 | 0.0199 |
| udp | 10.0 | 3 | 0.7026 | 0.0128 | 0.6660 | 0.0149 |
| udp | 25.0 | 3 | 0.7032 | 0.0101 | 0.6770 | 0.0147 |
| udp | 50.0 | 3 | 0.7051 | 0.0135 | 0.6771 | 0.0101 |

Paper baseline three-fold mean: ACC=0.6617, UAR=0.6050.

## Gender inference attack

Attack artifacts currently present: 55. Review `attack_fold_comparison.csv` for requested and actual update counts; the 100-requested setting can be capped by the available per-client history.

Paper unprotected folds (SR/UASR): F1 0.837/0.829; F2 0.847/0.838; F3 0.817/0.791.

## Interpretation

Numerical differences are not automatically defence effects. The paper does not publish exact fold seeds or the Voice-Ind/Gender-Ind embedding/TTS checkpoints and public pools. Local substitutions are identified in the methods and decisions files.
