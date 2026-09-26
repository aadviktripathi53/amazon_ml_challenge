# Experiment log

Append one row after every meaningful run (see `CLAUDE.md`).

| date-time | who | what changed | val F0.5 | blocking recall | avg cands | public LB (if submitted) |
|---|---|---|---|---|---|---|
| 2026-09-26 | Claude (A) | baseline v0, sample (ER_DATA_DIR=dataset_sample): 3-channel blocking (name/ctx/addr, 2-channel recall was 0.907), LightGBM 35 feats, OOF-tuned threshold 0.45 | 0.9850 (OPTIMISTIC: sample train pool = true matches + random distractors) | 0.9950 | 53.4 | - |
| 2026-09-26 | Claude (A) | baseline v0, realistic 5% (ER_DATA_DIR=dataset ER_TRAIN_S1_FRAC=0.05 ER_MAX_MEM_GB=8): 88,274 train / 22,068 val S1 vs FULL train pool (10.3M S2+S3); sharded memory-bounded blocking, 3 channels (max_df cap 0.2% of partition docs), LightGBM, OOF threshold 0.65. Recall 0.583 (India 0.549, US 0.606), full true set captured 0.409; OOF AUC 0.9997; val precision 0.976 / recall 0.548; singletons 0.940, non-singletons 0.638; India 0.624, US 0.675. Cause of low recall: search-time n-gram cap empties name/ctx queries (37% of S1 get 0 name candidates) - NOT fixed yet | 0.6546 | 0.5830 | 44.9 | - |
