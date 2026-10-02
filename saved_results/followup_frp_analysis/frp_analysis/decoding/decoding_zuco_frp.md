# Decoding the read word from EEG (zuco_frp)

6,929 words with EEG, 2,623 word types, 5 folds split by sentence. Target: labse input embeddings (word identity, no position). No teacher forcing: every word is decoded from its own EEG only.

## 2-vs-2 accuracy (chance 0.5)

| input | all pairs | pairs matched on length, frequency and position |
|---|---|---|
| eeg | 0.544 [0.535, 0.553] | 0.511 [0.501, 0.522] |
| shuffled_eeg | 0.499 [0.491, 0.508] | 0.496 [0.487, 0.506] |
| noise | 0.507 [0.499, 0.515] | 0.511 [0.501, 0.521] |
| word_features | 0.958 [0.955, 0.961] | 0.733 [0.725, 0.740] |
| word_features+eeg | 0.958 [0.955, 0.961] | 0.731 [0.724, 0.739] |

| difference | all pairs | matched pairs |
|---|---|---|
| eeg - shuffled_eeg | 0.044 [0.033, 0.056] | 0.015 [0.001, 0.030] |
| eeg - noise | 0.037 [0.024, 0.048] | 0.000 [-0.014, 0.015] |
| word_features+eeg - word_features | 0.000 [-0.001, 0.001] | -0.001 [-0.005, 0.003] |

Pairs: 20,000 (all), 20,000 (matched).

## Retrieval among the test vocabulary

Chance mean reciprocal rank 0.0095.

| input | top-1 | top-5 | mean reciprocal rank |
|---|---:|---:|---|
| eeg | 0.0263 | 0.0576 | 0.0466 [0.0430, 0.0504] |
| shuffled_eeg | 0.0149 | 0.0390 | 0.0322 [0.0287, 0.0356] |
| noise | 0.0137 | 0.0390 | 0.0317 [0.0290, 0.0346] |
| word_features | 0.0641 | 0.2270 | 0.1477 [0.1412, 0.1543] |
| word_features+eeg | 0.0638 | 0.2273 | 0.1475 [0.1411, 0.1541] |

EEG minus shuffled EEG (reciprocal rank): 0.0144 [0.0096, 0.0191]

## Sentiment of the decoded text (macro-F1, chance about 0.33)

| text given to the classifier | macro-F1 [95% CI] |
|---|---|
| real text (same words) | 0.518 [0.468, 0.568] |
| decoded from eeg | 0.315 [0.271, 0.361] |
| decoded from shuffled_eeg | 0.290 [0.244, 0.334] |
| decoded from noise | 0.373 [0.329, 0.418] |
| decoded from word_features | 0.294 [0.255, 0.334] |
| decoded from word_features+eeg | 0.296 [0.257, 0.334] |

## Reading

* **EEG carries information about the read word:** yes, real EEG beats shuffled EEG in 2-vs-2.
* **Beyond word features** (EEG fitted to what length, frequency and position leave unexplained): EEG adds nothing on all pairs; nothing on matched pairs.
* **Sentiment from decoded text:** not above the shuffled-EEG and noise controls.

Examples: `decoded_examples_zuco_frp.csv`.
