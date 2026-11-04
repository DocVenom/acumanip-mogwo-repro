# data/

Everything here is **derived**, never copied from a restricted source.

## splits/

| File | Meaning |
|---|---|
| `ucihar_index.csv` | 10,299 rows. Canonical sample order: all of official *train* (rows 0-7351) then official *test* (rows 7352-10298). Columns `part,row,subject,label,activity`. Every index in the other UCI files points at a row of this table. |
| `ucihar_search_split.json` | Model-selection stage. 3,000 samples drawn from official *train* (stratified by label, seed 42), then split 8:2 into 2,400 search-train / 600 validation. The official test set is excluded by construction. |
| `ucihar_full_split.json` | Final retraining stage. Official *train* split 85/15 (seed 42) -> 6,249 train / 1,103 validation. |
| `ucihar_loso_folds.json` | 30 leave-one-subject-out folds: held-out subject, its sample count, and the size of the remaining pool. Inside each fold, 10% of the pool is used for early stopping (seed 42). |
| `wisdm_split.json` | Subject-disjoint split: subjects 1-27 train (12,090 windows), 28-36 test (4,300 windows). |

`scripts/regenerate_splits.py` recomputes all of the above from the public archives and
asserts equality, so these files can be checked without trusting them.

## cards/

Per-dataset cards: channel names, sampling rate, window length, class distribution,
subject ranges, and the normalisation rule. Aggregates only — no per-sample records.

## audit/

`audit_table.json` is the evidence behind the data-availability discussion. For each of
the four acupuncture corpora it records the trial count, the number of distinct subjects,
the sampling rate, the per-class counts where labels exist, and whether the corpus can
support time-series classification at all. The `reason` field states, per corpus, why it
can or cannot be used.

No subject name, file path, or per-trial identifier appears in this file. One field
(`subjects_unique_names`) is an aggregate count of distinct subject names in a corpus; it
exists to document that the corpus is *not* anonymised, which is itself a reason the
corpus cannot be redistributed.

## What is deliberately absent

- Raw sensor signals from any source.
- The NBSDC corpora, in any form, including derived per-trial features.
- Model checkpoints and training logs.

See the top-level `README.md` for how to obtain the restricted corpora.
