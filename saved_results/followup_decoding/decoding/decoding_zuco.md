# Decoding the read word from EEG (zuco)

7,033 words with EEG, 2,630 word types, 5 folds split by sentence. Target: labse input embeddings (word identity, no position). No teacher forcing: every word is decoded from its own EEG only.

## 2-vs-2 accuracy (chance 0.5)

| input | all pairs | pairs matched on length, frequency and position |
|---|---|---|
| eeg | 0.574 [0.566, 0.583] | 0.514 [0.504, 0.525] |
| shuffled_eeg | 0.494 [0.486, 0.502] | 0.499 [0.490, 0.509] |
| noise | 0.499 [0.491, 0.508] | 0.509 [0.499, 0.520] |
| word_features | 0.959 [0.956, 0.961] | 0.733 [0.725, 0.741] |
| word_features+eeg | 0.959 [0.956, 0.961] | 0.734 [0.725, 0.742] |

| difference | all pairs | matched pairs |
|---|---|---|
| eeg - shuffled_eeg | 0.080 [0.069, 0.091] | 0.015 [0.002, 0.029] |
| eeg - noise | 0.075 [0.063, 0.087] | 0.005 [-0.009, 0.020] |
| word_features+eeg - word_features | -0.000 [-0.001, 0.000] | 0.000 [-0.003, 0.004] |

Pairs: 20,000 (all), 20,000 (matched).

## Retrieval among the test vocabulary

Chance mean reciprocal rank 0.0095.

| input | top-1 | top-5 | mean reciprocal rank |
|---|---:|---:|---|
| eeg | 0.0186 | 0.0584 | 0.0440 [0.0406, 0.0477] |
| shuffled_eeg | 0.0166 | 0.0427 | 0.0346 [0.0312, 0.0381] |
| noise | 0.0148 | 0.0418 | 0.0333 [0.0301, 0.0365] |
| word_features | 0.0631 | 0.2295 | 0.1471 [0.1407, 0.1535] |
| word_features+eeg | 0.0633 | 0.2301 | 0.1472 [0.1409, 0.1536] |

EEG minus shuffled EEG (reciprocal rank): 0.0094 [0.0052, 0.0143]

## Sentiment of the decoded text (macro-F1, chance about 0.33)

| text given to the classifier | macro-F1 [95% CI] |
|---|---|
| real text (same words) | 0.519 [0.467, 0.567] |
| decoded from eeg | 0.308 [0.268, 0.348] |
| decoded from shuffled_eeg | 0.284 [0.238, 0.329] |
| decoded from noise | 0.325 [0.279, 0.368] |
| decoded from word_features | 0.313 [0.272, 0.356] |
| decoded from word_features+eeg | 0.329 [0.283, 0.370] |

## Reading

* **EEG carries information about the read word:** yes, real EEG beats shuffled EEG in 2-vs-2.
* **Beyond word features** (EEG fitted to what length, frequency and position leave unexplained): EEG adds nothing on all pairs; nothing on matched pairs.
* **Sentiment from decoded text:** not above the shuffled-EEG and noise controls.

Examples: `decoded_examples_zuco.csv`.
