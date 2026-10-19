# highband / raw

- Test set: 400 clips (1600 train), split by 2000 room/source groups.
- Best layer for regression: **layer 07**, R² = **-0.000** (Spearman 0.017, MAE 12.2 dBFS).
- Best layer for 4-bin classification: **layer 08**, accuracy **0.273** (chance 0.228).
- Hand-crafted baseline (overall level + 7.5-8 kHz level): R² -0.001, accuracy 0.228.
- Audio-LLM: not run yet (model download held back; see DECISIONS.md), so the gap is not computed.
