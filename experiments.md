# Experiment log

Append one row after every meaningful run (see `CLAUDE.md`).

| date-time | who | what changed | val F0.5 | blocking recall | avg cands | public LB (if submitted) |
|---|---|---|---|---|---|---|
| 2026-09-26 | Claude (A) | baseline v0, sample (ER_DATA_DIR=dataset_sample): 3-channel blocking (name/ctx/addr, 2-channel recall was 0.907), LightGBM 35 feats, OOF-tuned threshold 0.45 | 0.9850 (OPTIMISTIC: sample train pool = true matches + random distractors) | 0.9950 | 53.4 | - |
