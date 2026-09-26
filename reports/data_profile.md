# Data profile

Data dir: `/home/sagemaker-user/amazon_ml_challenge/dataset`  
Streaming chunk size: 200,000 rows. IDs compared via 64-bit hashes.

## Files

| file | size | parsed rows | raw data lines | columns |
|---|---|---|---|---|
| train/train_ground_truth.tsv | 121.1 MB | 2,206,821 | 2,206,821 | source1_entity_id, matched_entity_ids |
| train/train_source1.tsv | 200.3 MB | 2,206,821 | 2,206,821 | entity_id, business_name, business_address, country |
| train/train_source2.tsv | 466.6 MB | 5,034,616 | 5,034,616 | entity_id, business_name, business_address, country |
| train/train_source3.tsv | 480.4 MB | 5,285,603 | 5,285,603 | entity_id, business_name, business_address, country |
| test/test_source1.tsv | 166.9 MB | 1,732,544 | 1,732,544 | entity_id, business_name, business_address, country |
| test/test_source2.tsv | 485.9 MB | 4,887,273 | 4,887,273 | entity_id, business_name, business_address, country |
| test/test_source3.tsv | 482.6 MB | 5,082,316 | 5,082,316 | entity_id, business_name, business_address, country |

## Integrity checks

- OK: parsed row counts equal physical line counts; IDs are clean.

## Source files

### train/train_source1.tsv

- rows: 2,206,821; duplicate entity_ids: 0; ids with unexpected prefix: 0
- empty business_name: 0 (0.00%); empty business_address: 0 (0.00%)

| country | rows | share |
|---|---|---|
| US | 1,323,633 | 59.98% |
| India | 883,188 | 40.02% |

### train/train_source2.tsv

- rows: 5,034,616; duplicate entity_ids: 0; ids with unexpected prefix: 0
- empty business_name: 0 (0.00%); empty business_address: 168,967 (3.36%)

| country | rows | share |
|---|---|---|
| US | 3,016,817 | 59.92% |
| India | 2,017,799 | 40.08% |

### train/train_source3.tsv

- rows: 5,285,603; duplicate entity_ids: 0; ids with unexpected prefix: 0
- empty business_name: 0 (0.00%); empty business_address: 175,916 (3.33%)

| country | rows | share |
|---|---|---|
| US | 3,170,056 | 59.98% |
| India | 2,115,547 | 40.02% |

### test/test_source1.tsv

- rows: 1,732,544; duplicate entity_ids: 0; ids with unexpected prefix: 0
- empty business_name: 0 (0.00%); empty business_address: 0 (0.00%)

| country | rows | share |
|---|---|---|
| India | 809,986 | 46.75% |
| US | 663,106 | 38.27% |
| France | 259,452 | 14.98% |

### test/test_source2.tsv

- rows: 4,887,273; duplicate entity_ids: 0; ids with unexpected prefix: 0
- empty business_name: 0 (0.00%); empty business_address: 129,408 (2.65%)

| country | rows | share |
|---|---|---|
| India | 2,312,565 | 47.32% |
| US | 1,871,330 | 38.29% |
| France | 703,378 | 14.39% |

### test/test_source3.tsv

- rows: 5,082,316; duplicate entity_ids: 0; ids with unexpected prefix: 0
- empty business_name: 0 (0.00%); empty business_address: 136,098 (2.68%)

| country | rows | share |
|---|---|---|
| India | 2,405,000 | 47.32% |
| US | 1,945,701 | 38.28% |
| France | 731,615 | 14.40% |

## Ground truth (`train/train_ground_truth.tsv`)

- S1 rows: 2,206,821; duplicate source1_entity_id rows: 0
- singleton rate (no matches): 123,247 (5.58%)
- S1 with matches from both S2 and S3: 1,776,047 (80.48%)
- total matches: 7,638,365 - from S2: 3,693,619 (48.36%), from S3: 3,944,746 (51.64%); IDs with another prefix: 0
- S2/S3 IDs appearing in more than one ground-truth row: **0** (max rows per ID: 1)

Matches per S1 entity:

| matches | S1 entities | share |
|---|---|---|
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10+ | 571 | 0.03% |

Ground truth vs Source 1 (train):

- ground-truth rows: 2,206,821; distinct source1_entity_id: 2,206,821
- train_source1.tsv: parsed rows 2,206,821; distinct entity_ids 2,206,821
- distinct ground-truth S1 IDs absent from train_source1.tsv: 0
- train S1 IDs without a ground-truth row: 0

Existence of ground-truth IDs in the train source files (mentions / distinct IDs):

- S1: 0 missing mentions / 0 distinct
- S2: 0 missing mentions / 0 distinct
- S3: 0 missing mentions / 0 distinct

**Every ground-truth ID exists in the source files: YES**

