# EEG-to-text generation (English ZuCo, Persian TeCo)

Test sentences were never seen in training (by any reader). *Teacher-forced*: each token is predicted from the true previous tokens (how many published EEG-to-text results were scored). *Free-running*: the model writes the whole sentence from its own previous tokens. Every input condition has the same sentence length and fixation pattern; `word_vectors` and `mbart_vectors` are positive controls (the input contains the text). Conditions ending in `_avg` are tested on the reader average of each test sentence.

## Positive control (must pass before the EEG rows mean anything)

A control model must write unseen sentences better than the noise model (free-running BLEU-4, paired by trial). If it does not, the generator is not using its input, and EEG = noise says nothing about the EEG.

| setting | encoder | input | language | test | BLEU-4 difference [95% CI] | verdict |
|---|---|---|---|---|---|---|
| en | continuous | mbart_vectors | en | single reader | +0.1819 [+0.1564, +0.2105] | PASS |
| en | continuous | mbart_vectors | en | reader average | +0.4739 [+0.4240, +0.5249] | PASS |
| en | continuous | word_vectors | en | single reader | -0.0013 [-0.0048, +0.0019] | FAIL |
| en | continuous | word_vectors | en | reader average | -0.0023 [-0.0068, +0.0019] | FAIL |

## English (ZuCo)

| setting | encoder | trained_on | tested_on | n_trials | tf_accuracy | tf_bleu4 | free_bleu1 | free_bleu4 | rouge1 | wer | sentiment_f1_generated | sentiment_f1_real_text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en | continuous | mbart_vectors | mbart_vectors | 942 | 0.6557 | 0.2880 | 0.4858 | 0.1990 | 0.5083 | 0.6767 | 0.5226 | 0.6108 |
| en | continuous | mbart_vectors | mbart_vectors_avg | 80 | 0.8235 | 0.5868 | 0.7050 | 0.5092 | 0.7478 | 0.3684 | 0.4938 | 0.6116 |
| en | continuous | noise | noise | 942 | 0.3802 | 0.0423 | 0.1731 | 0.0077 | 0.1844 | 0.9285 | 0.1663 | 0.6108 |
| en | continuous | noise | noise_avg | 80 | 0.3838 | 0.0451 | 0.1717 | 0.0075 | 0.1779 | 0.9349 | 0.1489 | 0.6116 |
| en | continuous | word_vectors | word_vectors | 942 | 0.3708 | 0.0385 | 0.1812 | 0.0096 | 0.1736 | 0.9588 | 0.2819 | 0.6108 |
| en | continuous | word_vectors | word_vectors_avg | 80 | 0.3727 | 0.0388 | 0.1728 | 0.0065 | 0.1740 | 0.9426 | 0.2760 | 0.6116 |

## Paired comparisons (same test trials; 95% CI from a sentence bootstrap)

| comparison | language | metric | difference [95% CI] |
|---|---|---|---|
| en_continuous: positive control, mbart_vectors model - noise model (single reader) | en | tf_accuracy | +0.2754 [+0.2569, +0.2946] |
| en_continuous: positive control, mbart_vectors model - noise model (single reader) | en | bleu4 | +0.1819 [+0.1564, +0.2105] |
| en_continuous: positive control, mbart_vectors model - noise model (single reader) | en | rouge1 | +0.3239 [+0.2948, +0.3558] |
| en_continuous: positive control, mbart_vectors model - noise model (reader average) | en | tf_accuracy | +0.4397 [+0.4145, +0.4656] |
| en_continuous: positive control, mbart_vectors model - noise model (reader average) | en | bleu4 | +0.4739 [+0.4240, +0.5249] |
| en_continuous: positive control, mbart_vectors model - noise model (reader average) | en | rouge1 | +0.5699 [+0.5325, +0.6082] |
| en_continuous: positive control, word_vectors model - noise model (single reader) | en | tf_accuracy | -0.0094 [-0.0203, +0.0004] |
| en_continuous: positive control, word_vectors model - noise model (single reader) | en | bleu4 | -0.0013 [-0.0048, +0.0019] |
| en_continuous: positive control, word_vectors model - noise model (single reader) | en | rouge1 | -0.0108 [-0.0203, -0.0018] |
| en_continuous: positive control, word_vectors model - noise model (reader average) | en | tf_accuracy | -0.0112 [-0.0240, +0.0012] |
| en_continuous: positive control, word_vectors model - noise model (reader average) | en | bleu4 | -0.0023 [-0.0068, +0.0019] |
| en_continuous: positive control, word_vectors model - noise model (reader average) | en | rouge1 | -0.0039 [-0.0188, +0.0111] |

## How to read

* Read the EEG rows only where the positive control passes in the same setting.
* EEG helps only if the EEG model beats the **noise** and **shuffled-EEG** models in free-running generation (CI above 0), per reader or on the reader average.
* If the EEG model's output barely changes when it is fed noise at test time, it ignores the EEG (the check of Jo et al.).
* Teacher-forced scores are high for every input because the language model predicts the next word from the true previous words; compare them across inputs, not with free-running scores.
* Multilingual training helps the EEG only if the interaction row is above 0; a joint-minus-single gain that is equal for EEG and noise comes from the shared language model, not from the brain signal.
