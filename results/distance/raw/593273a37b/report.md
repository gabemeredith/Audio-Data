# distance / raw

- Test set: 400 clips (1600 train), split by 10 room/source groups.
- Best layer for regression: **layer 04**, R² = **0.925** (Spearman 0.961, MAE 0.68 m).
- Best layer for 4-bin classification: **layer 03**, accuracy **0.767** (chance 0.367).
- Hand-crafted baseline (RMS + oracle DRR): R² 0.894, accuracy 0.807.
- Audio-LLM: not run yet (model download held back; see DECISIONS.md), so the gap is not computed.
