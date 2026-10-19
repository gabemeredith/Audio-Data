# distance / loudness_normalized

- Test set: 400 clips (1600 train), split by 10 room/source groups.
- Best layer for regression: **layer 04**, R² = **0.851** (Spearman 0.927, MAE 0.894 m).
- Best layer for 4-bin classification: **layer 03**, accuracy **0.677** (chance 0.367).
- Hand-crafted baseline (RMS + oracle DRR): R² 0.666, accuracy 0.588.
- Audio-LLM: not run yet (model download held back; see DECISIONS.md), so the gap is not computed.
