# Decoding the read word from EEG (zuco)

7,033 words with EEG, 2,630 word types, 5 folds split by sentence. Target: labse layer 0. No teacher forcing: every word is decoded from its own EEG only.

## 2-vs-2 accuracy (chance 0.5)

| input | all pairs | pairs matched on length and frequency |
|---|---|---|
| eeg | 0.630 [0.622, 0.638] | 0.609 [0.599, 0.618] |
| shuffled_eeg | 0.497 [0.489, 0.506] | 0.496 [0.487, 0.506] |
| noise | 0.501 [0.493, 0.510] | 0.506 [0.497, 0.516] |
| word_features | 0.978 [0.976, 0.980] | 0.880 [0.874, 0.886] |
| eeg+word_features | 0.924 [0.920, 0.928] | 0.778 [0.771, 0.786] |

| difference | all pairs | matched pairs |
|---|---|---|
| eeg - shuffled_eeg | 0.133 [0.121, 0.144] | 0.112 [0.098, 0.125] |
| eeg - noise | 0.129 [0.117, 0.140] | 0.102 [0.087, 0.116] |
| eeg+word_features - word_features | -0.054 [-0.058, -0.050] | -0.101 [-0.109, -0.093] |

Pairs: 20,000 (all), 20,000 (matched).

## Retrieval among the test vocabulary

Chance mean reciprocal rank 0.0095.

| input | top-1 | top-5 | mean reciprocal rank |
|---|---:|---:|---|
| eeg | 0.0178 | 0.0537 | 0.0416 [0.0384, 0.0451] |
| shuffled_eeg | 0.0155 | 0.0421 | 0.0337 [0.0305, 0.0371] |
| noise | 0.0152 | 0.0409 | 0.0330 [0.0299, 0.0362] |
| word_features | 0.0677 | 0.2568 | 0.1571 [0.1509, 0.1636] |
| eeg+word_features | 0.0634 | 0.1766 | 0.1277 [0.1220, 0.1340] |

EEG minus shuffled EEG (reciprocal rank): 0.0080 [0.0041, 0.0126]

## Sentiment of the decoded text (macro-F1, chance about 0.33)

| text given to the classifier | macro-F1 [95% CI] |
|---|---|
| real text (same words) | 0.511 [0.461, 0.562] |
| decoded from eeg | 0.348 [0.301, 0.394] |
| decoded from shuffled_eeg | 0.273 [0.230, 0.316] |
| decoded from noise | 0.311 [0.265, 0.355] |
| decoded from word_features | 0.340 [0.291, 0.389] |
| decoded from eeg+word_features | 0.317 [0.268, 0.358] |

## Reading

* **EEG carries information about the read word:** yes, real EEG beats shuffled EEG in 2-vs-2.
* **Beyond length and frequency:** not detected; on length- and frequency-matched pairs EEG adds nothing to word features.
* **Sentiment from decoded text:** not above the shuffled-EEG control.

Examples: `decoded_examples_zuco.csv`.
