# Decoding the read word from EEG (zuco_frp)

5,574 words with EEG, 2,483 word types, 5 folds split by sentence. Target: labse input embeddings (word identity, no position). No teacher forcing: every word is decoded from its own EEG only.

## 2-vs-2 accuracy (chance 0.5)

| input | all pairs | pairs matched on length, frequency and position |
|---|---|---|
| eeg | 0.515 [0.506, 0.523] | 0.505 [0.495, 0.516] |
| shuffled_eeg | 0.506 [0.498, 0.515] | 0.503 [0.492, 0.514] |
| noise | 0.499 [0.491, 0.507] | 0.499 [0.489, 0.509] |
| word_features | 0.958 [0.955, 0.961] | 0.729 [0.720, 0.739] |
| word_features+eeg | 0.958 [0.955, 0.961] | 0.728 [0.719, 0.738] |

| difference | all pairs | matched pairs |
|---|---|---|
| eeg - shuffled_eeg | 0.009 [-0.003, 0.020] | 0.003 [-0.011, 0.018] |
| eeg - noise | 0.016 [0.004, 0.028] | 0.007 [-0.008, 0.022] |
| word_features+eeg - word_features | 0.000 [-0.001, 0.001] | -0.001 [-0.003, 0.002] |

Pairs: 20,000 (all), 20,000 (matched).

## Retrieval among the test vocabulary

Chance mean reciprocal rank 0.0102.

| input | top-1 | top-5 | mean reciprocal rank |
|---|---:|---:|---|
| eeg | 0.0160 | 0.0305 | 0.0292 [0.0260, 0.0325] |
| shuffled_eeg | 0.0079 | 0.0239 | 0.0221 [0.0197, 0.0250] |
| noise | 0.0084 | 0.0248 | 0.0225 [0.0199, 0.0255] |
| word_features | 0.0572 | 0.1916 | 0.1284 [0.1219, 0.1349] |
| word_features+eeg | 0.0574 | 0.1914 | 0.1286 [0.1221, 0.1351] |

EEG minus shuffled EEG (reciprocal rank): 0.0070 [0.0031, 0.0115]

## Sentiment of the decoded text (macro-F1, chance about 0.33)

| text given to the classifier | macro-F1 [95% CI] |
|---|---|
| real text (same words) | 0.527 [0.475, 0.575] |
| decoded from eeg | 0.316 [0.275, 0.360] |
| decoded from shuffled_eeg | 0.332 [0.283, 0.374] |
| decoded from noise | 0.301 [0.254, 0.348] |
| decoded from word_features | 0.376 [0.328, 0.424] |
| decoded from word_features+eeg | 0.372 [0.325, 0.421] |

## Reading

* **EEG carries information about the read word:** not detected (EEG ~ shuffled EEG).
* **Beyond word features** (EEG fitted to what length, frequency and position leave unexplained): EEG adds nothing on all pairs; nothing on matched pairs.
* **Sentiment from decoded text:** not above the shuffled-EEG and noise controls.

Examples: `decoded_examples_zuco_frp.csv`.
