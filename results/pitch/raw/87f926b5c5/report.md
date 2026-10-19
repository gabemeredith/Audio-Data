# pitch / raw

- Test set: 400 clips (1600 train), split by 2000 room/source groups.
- Best layer for regression: **layer 00**, R² = **0.997** (Spearman 0.998, MAE 10.7 Hz).
- Best layer for 4-bin classification: **layer 01**, accuracy **0.993** (chance 0.235).
- Hand-crafted baseline (autocorrelation pitch estimate (log Hz)): R² 1.000, accuracy 0.990.
- Audio-LLM: not run yet (model download held back; see DECISIONS.md), so the gap is not computed.
