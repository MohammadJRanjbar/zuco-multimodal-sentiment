# Decoding the read word from EEG (teco)

2,999 words with EEG, 1,238 word types, 5 folds split by sentence. Target: labse layer 0. No teacher forcing: every word is decoded from its own EEG only.

## 2-vs-2 accuracy (chance 0.5)

| input | all pairs | pairs matched on length and frequency |
|---|---|---|
| eeg | 0.626 [0.616, 0.636] | 0.600 [0.589, 0.611] |
| shuffled_eeg | 0.503 [0.494, 0.513] | 0.506 [0.493, 0.519] |
| noise | 0.512 [0.501, 0.521] | 0.500 [0.489, 0.510] |
| word_features | 0.969 [0.966, 0.972] | 0.880 [0.873, 0.886] |
| eeg+word_features | 0.946 [0.943, 0.950] | 0.804 [0.795, 0.814] |

| difference | all pairs | matched pairs |
|---|---|---|
| eeg - shuffled_eeg | 0.122 [0.109, 0.136] | 0.094 [0.078, 0.110] |
| eeg - noise | 0.114 [0.101, 0.129] | 0.100 [0.085, 0.115] |
| eeg+word_features - word_features | -0.023 [-0.026, -0.020] | -0.075 [-0.085, -0.066] |

Pairs: 20,000 (all), 20,000 (matched).

## Retrieval among the test vocabulary

Chance mean reciprocal rank 0.0182.

| input | top-1 | top-5 | mean reciprocal rank |
|---|---:|---:|---|
| eeg | 0.0250 | 0.0854 | 0.0639 [0.0585, 0.0694] |
| shuffled_eeg | 0.0157 | 0.0530 | 0.0419 [0.0370, 0.0470] |
| noise | 0.0243 | 0.0714 | 0.0531 [0.0477, 0.0588] |
| word_features | 0.1224 | 0.2958 | 0.2074 [0.1972, 0.2181] |
| eeg+word_features | 0.0974 | 0.2741 | 0.1878 [0.1782, 0.1982] |

EEG minus shuffled EEG (reciprocal rank): 0.0220 [0.0153, 0.0290]

## Sentiment of the decoded text (macro-F1, chance about 0.33)

| text given to the classifier | macro-F1 [95% CI] |
|---|---|
| real text (same words) | 0.507 [0.438, 0.582] |
| decoded from eeg | 0.218 [0.165, 0.272] |
| decoded from shuffled_eeg | 0.330 [0.260, 0.401] |
| decoded from noise | 0.358 [0.288, 0.426] |
| decoded from word_features | 0.282 [0.227, 0.334] |
| decoded from eeg+word_features | 0.279 [0.214, 0.340] |

## Reading

* **EEG carries information about the read word:** yes, real EEG beats shuffled EEG in 2-vs-2.
* **Beyond length and frequency:** not detected; on length- and frequency-matched pairs EEG adds nothing to word features.
* **Sentiment from decoded text:** not above the shuffled-EEG control.

Examples: `decoded_examples_teco.csv`.
