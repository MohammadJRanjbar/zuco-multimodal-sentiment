# EEG-to-text generation (English ZuCo, Persian TeCo)

Test sentences were never seen in training (by any reader). *Teacher-forced*: each token is predicted from the true previous tokens (how many published EEG-to-text results were scored). *Free-running*: the model writes the whole sentence from its own previous tokens. Every input condition has the same sentence length and fixation pattern; `word_vectors` and `mbart_vectors` are positive controls (the input contains the text). Conditions ending in `_avg` are tested on the reader average of each test sentence.

## Positive control (must pass before the EEG rows mean anything)

A control model must write unseen sentences better than the noise model (free-running BLEU-4, paired by trial). If it does not, the generator is not using its input, and EEG = noise says nothing about the EEG.

| setting | encoder | input | language | test | BLEU-4 difference [95% CI] | verdict |
|---|---|---|---|---|---|---|
| en | continuous | mbart_vectors | en | single reader | +0.5422 [+0.5123, +0.5715] | PASS |
| en | continuous | mbart_vectors | en | reader average | +0.8264 [+0.7814, +0.8663] | PASS |
| en | continuous | word_vectors | en | single reader | +0.3766 [+0.3425, +0.4095] | PASS |
| en | continuous | word_vectors | en | reader average | +0.1630 [+0.1002, +0.2268] | PASS |

## Input -> mBART word embedding (ridge map, validation + test words)

How well each input predicts the read word's mBART embedding (the generator's input). Cosine 0 and R² <= 0 mean the input carries no information about the word's embedding.

| run | language/input | words | mean cosine | R² |
|---|---|---:|---:|---:|
| en_continuous_mbart_vectors | en/mbart_vectors | 16582 | 1.000 | 1.000 |
| en_continuous_noise | en/noise | 16582 | 0.796 | -0.001 |
| en_continuous_word_vectors | en/word_vectors | 16582 | 0.909 | 0.544 |

## English (ZuCo)

| setting | encoder | trained_on | tested_on | n_trials | tf_accuracy | tf_bleu4 | free_bleu1 | free_bleu4 | rouge1 | wer | sentiment_f1_generated | sentiment_f1_real_text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en | continuous | mbart_vectors | mbart_vectors | 942 | 0.8306 | 0.5746 | 0.7520 | 0.5329 | 0.7646 | 0.3231 | 0.5730 | 0.6108 |
| en | continuous | mbart_vectors | mbart_vectors_avg | 80 | 0.9467 | 0.8300 | 0.8880 | 0.8038 | 0.9197 | 0.1530 | 0.5974 | 0.6116 |
| en | continuous | noise | noise | 942 | 0.3626 | 0.0354 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1590 | 0.6108 |
| en | continuous | noise | noise_avg | 80 | 0.3617 | 0.0274 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1587 | 0.6116 |
| en | continuous | word_vectors | word_vectors | 942 | 0.7354 | 0.4095 | 0.6337 | 0.3628 | 0.6372 | 0.4811 | 0.5780 | 0.6108 |
| en | continuous | word_vectors | word_vectors_avg | 80 | 0.8120 | 0.5949 | 0.0707 | 0.0510 | 0.2272 | 0.8093 | 0.3414 | 0.6116 |

## Paired comparisons (same test trials; 95% CI from a sentence bootstrap)

| comparison | language | metric | difference [95% CI] |
|---|---|---|---|
| en_continuous: positive control, mbart_vectors model - noise model (single reader) | en | tf_accuracy | +0.4681 [+0.4498, +0.4861] |
| en_continuous: positive control, mbart_vectors model - noise model (single reader) | en | bleu4 | +0.5422 [+0.5123, +0.5715] |
| en_continuous: positive control, mbart_vectors model - noise model (single reader) | en | rouge1 | +0.7646 [+0.7463, +0.7821] |
| en_continuous: positive control, mbart_vectors model - noise model (reader average) | en | tf_accuracy | +0.5850 [+0.5612, +0.6103] |
| en_continuous: positive control, mbart_vectors model - noise model (reader average) | en | bleu4 | +0.8264 [+0.7814, +0.8663] |
| en_continuous: positive control, mbart_vectors model - noise model (reader average) | en | rouge1 | +0.9197 [+0.8913, +0.9423] |
| en_continuous: positive control, word_vectors model - noise model (single reader) | en | tf_accuracy | +0.3728 [+0.3500, +0.3953] |
| en_continuous: positive control, word_vectors model - noise model (single reader) | en | bleu4 | +0.3766 [+0.3425, +0.4095] |
| en_continuous: positive control, word_vectors model - noise model (single reader) | en | rouge1 | +0.6372 [+0.6070, +0.6647] |
| en_continuous: positive control, word_vectors model - noise model (reader average) | en | tf_accuracy | +0.4502 [+0.4187, +0.4809] |
| en_continuous: positive control, word_vectors model - noise model (reader average) | en | bleu4 | +0.1630 [+0.1002, +0.2268] |
| en_continuous: positive control, word_vectors model - noise model (reader average) | en | rouge1 | +0.2272 [+0.1466, +0.3045] |

## How to read

* Read the EEG rows only where the positive control passes in the same setting.
* EEG helps only if the EEG model beats the **noise** and **shuffled-EEG** models in free-running generation (CI above 0), per reader or on the reader average.
* If the EEG model's output barely changes when it is fed noise at test time, it ignores the EEG (the check of Jo et al.).
* Teacher-forced scores are high for every input because the language model predicts the next word from the true previous words; compare them across inputs, not with free-running scores.
* Multilingual training helps the EEG only if the interaction row is above 0; a joint-minus-single gain that is equal for EEG and noise comes from the shared language model, not from the brain signal.
