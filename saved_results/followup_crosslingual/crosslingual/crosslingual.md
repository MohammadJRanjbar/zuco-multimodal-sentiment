# English vs Persian EEG directions in a shared text space

labse layer 2 (768 dimensions), 32 EEG components per language, within-sentence centered.

## Overlap of the two EEG subspaces

Mean squared cosine of the principal angles (0 = unrelated, 1 = identical).

| | real | null mean | null 95th percentile |
|---|---:|---:|---:|
| all directions | 0.1120 | 0.0969 | 0.1061 |
| without word-feature directions | 0.1073 | 0.0943 | 0.1032 |

Random subspaces would give about 0.0417. Share of each language's word-feature directions inside its own EEG subspace: English (ZuCo) 0.18, Persian (TeCo) 0.20.

## Transfer: predicting one language's EEG through the other's directions

**Persian (TeCo)** (held-out R², within-sentence)

| features | R² [95% CI] |
|---|---|
| full text space | 0.0026 [0.0020, 0.0032] |
| English EEG subspace | 0.0030 [0.0024, 0.0037] |
| English EEG subspace without word features | 0.0027 [0.0021, 0.0034] |
| English shuffled-EEG subspace | 0.0020 [0.0016, 0.0024] |
| English word-feature subspace | 0.0025 [0.0019, 0.0032] |
| random subspace (mean of 5) | 0.0008 |

**English (ZuCo)** (held-out R², within-sentence)

| features | R² [95% CI] |
|---|---|
| full text space | 0.0031 [0.0026, 0.0036] |
| Persian EEG subspace | 0.0030 [0.0025, 0.0034] |
| Persian EEG subspace without word features | 0.0022 [0.0018, 0.0026] |
| Persian shuffled-EEG subspace | 0.0027 [0.0023, 0.0032] |
| Persian word-feature subspace | 0.0028 [0.0024, 0.0033] |
| random subspace (mean of 5) | 0.0010 |

## Reading

* **Shared EEG directions:** the overlap exceeds the shuffled-EEG null.
* **Beyond word features:** the overlap remains after removing length, frequency and position.
