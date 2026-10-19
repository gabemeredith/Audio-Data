# distance / raw

- Test set: 400 clips (1600 train), split by 10 room/source groups.
- Best layer for regression: **layer 04**, R² = **0.928** (Spearman 0.963, MAE 0.657 m).
- Best layer for 4-bin classification: **layer 02**, accuracy **0.770** (chance 0.367).
- Hand-crafted baseline (RMS + oracle DRR): R² 0.894, accuracy 0.807.
- Audio-LLM: not run yet (model download held back; see DECISIONS.md), so the gap is not computed.
