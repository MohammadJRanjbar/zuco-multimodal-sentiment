# Decoding the read word from EEG (teco)

2,999 words with EEG, 1,238 word types, 5 folds split by sentence. Target: labse input embeddings (word identity, no position). No teacher forcing: every word is decoded from its own EEG only.

## 2-vs-2 accuracy (chance 0.5)

| input | all pairs | pairs matched on length, frequency and position |
|---|---|---|
| eeg | 0.578 [0.567, 0.588] | 0.519 [0.506, 0.534] |
| shuffled_eeg | 0.508 [0.499, 0.517] | 0.524 [0.509, 0.540] |
| noise | 0.519 [0.509, 0.529] | 0.510 [0.496, 0.522] |
| word_features | 0.944 [0.941, 0.948] | 0.734 [0.724, 0.746] |
| word_features+eeg | 0.944 [0.941, 0.948] | 0.729 [0.719, 0.741] |

| difference | all pairs | matched pairs |
|---|---|---|
| eeg - shuffled_eeg | 0.070 [0.055, 0.084] | -0.005 [-0.025, 0.013] |
| eeg - noise | 0.059 [0.046, 0.073] | 0.010 [-0.008, 0.028] |
| word_features+eeg - word_features | 0.000 [-0.001, 0.001] | -0.005 [-0.012, 0.002] |

Pairs: 20,000 (all), 20,000 (matched).

## Retrieval among the test vocabulary

Chance mean reciprocal rank 0.0182.

| input | top-1 | top-5 | mean reciprocal rank |
|---|---:|---:|---|
| eeg | 0.0240 | 0.0870 | 0.0637 [0.0584, 0.0696] |
| shuffled_eeg | 0.0173 | 0.0540 | 0.0435 [0.0384, 0.0487] |
| noise | 0.0237 | 0.0724 | 0.0535 [0.0481, 0.0590] |
| word_features | 0.1220 | 0.2978 | 0.2038 [0.1937, 0.2141] |
| word_features+eeg | 0.1214 | 0.2978 | 0.2033 [0.1933, 0.2136] |

EEG minus shuffled EEG (reciprocal rank): 0.0203 [0.0124, 0.0276]

## Sentiment of the decoded text (macro-F1, chance about 0.33)

| text given to the classifier | macro-F1 [95% CI] |
|---|---|
| real text (same words) | 0.464 [0.395, 0.539] |
| decoded from eeg | 0.198 [0.154, 0.246] |
| decoded from shuffled_eeg | 0.329 [0.258, 0.401] |
| decoded from noise | 0.310 [0.243, 0.381] |
| decoded from word_features | 0.289 [0.233, 0.342] |
| decoded from word_features+eeg | 0.275 [0.224, 0.329] |

## Reading

* **EEG carries information about the read word:** yes, real EEG beats shuffled EEG in 2-vs-2.
* **Beyond word features** (EEG fitted to what length, frequency and position leave unexplained): EEG adds nothing on all pairs; nothing on matched pairs.
* **Sentiment from decoded text:** not above the shuffled-EEG and noise controls.

Examples: `decoded_examples_teco.csv`.
