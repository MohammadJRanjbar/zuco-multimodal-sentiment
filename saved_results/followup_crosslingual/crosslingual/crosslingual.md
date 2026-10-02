# English vs Persian EEG directions in a shared text space

labse layer 2 (768 dimensions), 32 EEG components per language, within-sentence centered.

## Overlap of the two EEG subspaces

Mean squared cosine of the principal angles (0 = unrelated, 1 = identical).

| | real | null mean | null 95th percentile | p |
|---|---:|---:|---:|---:|
| all directions | 0.1120 | 0.0964 | 0.1064 | 0.010 |
| without word-feature directions | 0.1060 | 0.0934 | 0.1037 | 0.030 |

Random subspaces would give about 0.0417. Share of each language's word-feature directions inside its own EEG subspace: English (ZuCo) 0.14, Persian (TeCo) 0.15.

## Transfer: predicting one language's EEG through the other's directions

**Persian (TeCo)** (held-out R², within-sentence)

| features | R² [95% CI] |
|---|---|
| full text space | 0.0026 [0.0020, 0.0032] |
| English EEG subspace | 0.0030 [0.0024, 0.0037] |
| English EEG subspace without word features | 0.0023 [0.0017, 0.0030] |
| English shuffled-EEG subspace | 0.0020 [0.0016, 0.0024] |
| English word-feature subspace | 0.0027 [0.0022, 0.0032] |
| random subspace (mean of 5) | 0.0010 |

**English (ZuCo)** (held-out R², within-sentence)

| features | R² [95% CI] |
|---|---|
| full text space | 0.0031 [0.0026, 0.0036] |
| Persian EEG subspace | 0.0030 [0.0025, 0.0034] |
| Persian EEG subspace without word features | 0.0020 [0.0016, 0.0024] |
| Persian shuffled-EEG subspace | 0.0026 [0.0021, 0.0030] |
| Persian word-feature subspace | 0.0036 [0.0030, 0.0042] |
| random subspace (mean of 5) | 0.0008 |

## Reading

* **Shared EEG directions:** the overlap exceeds the shuffled-EEG null by 0.0156 (p = 0.010).
* **Beyond word features:** not supported: the overlap without word features is only slightly above its null, and transfer without word features does not beat the shuffled-EEG subspace (CIs overlap).
* Compare the word-feature subspace rows: a few word-feature directions of the other language predict EEG about as well as the full text space.
