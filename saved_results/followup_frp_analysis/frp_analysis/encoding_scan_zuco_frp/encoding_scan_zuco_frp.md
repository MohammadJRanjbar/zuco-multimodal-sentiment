# Do word vectors predict word-level EEG? (zuco_frp)

400 sentences, 6929 words with EEG (832 EEG features per word, averaged over readers). Targets: top 32 EEG principal components (share of EEG variance: centered 87%, raw 87%). 5 folds split by sentence; the ridge penalty is chosen inside the training sentences only.

**How to read:** R² is measured on sentences the model never saw. Above 0 means the word vectors predict EEG better than the average does. `shuffled` is the same pipeline with EEG targets shuffled across words (should be about 0). *centered*: only word-to-word differences within a sentence; *raw*: also differences between sentences. A layer **passes** when both its R² and its margin over shuffled have 95% CIs above 0.

## Verdict

* **centered: PASS** — best labse layer 5/12: R² 0.0024 [0.0020, 0.0028], above shuffled by 0.0024 [0.0020, 0.0028]; 92 of 92 model-layers pass.
* **raw: PASS** — best labse layer 5/12: R² 0.0013 [0.0001, 0.0012], above shuffled by 0.0023 [0.0019, 0.0026]; 1 of 92 model-layers pass.

> The best layer is the maximum over 92 model-layers, so a single layer barely above 0 is weak evidence. Convincing: several neighbouring layers pass, and the same models and depths also pass on the other language.

## Best layer per model

| model | mode | best layer | R² [95% CI] | shuffled R² | R² − shuffled [95% CI] | layers passing | penalty at grid max |
|---|---|---:|---|---:|---|---:|---:|
| labse | centered | 5/12 | 0.0024 [0.0020, 0.0028] | 0.0000 | 0.0024 [0.0020, 0.0028] | 13/13 | 0% |
| labse | raw | 5/12 | 0.0013 [0.0001, 0.0012] | -0.0010 | 0.0023 [0.0019, 0.0026] | 1/13 | 0% |
| xlmr-large | centered | 8/24 | 0.0016 [0.0011, 0.0021] | 0.0000 | 0.0016 [0.0011, 0.0021] | 25/25 | 0% |
| xlmr-large | raw | 8/24 | 0.0006 [-0.0006, 0.0005] | -0.0010 | 0.0016 [0.0011, 0.0020] | 0/25 | 0% |
| me5-large | centered | 8/24 | 0.0017 [0.0013, 0.0022] | 0.0000 | 0.0017 [0.0013, 0.0022] | 25/25 | 0% |
| me5-large | raw | 8/24 | 0.0006 [-0.0006, 0.0005] | -0.0010 | 0.0016 [0.0012, 0.0020] | 0/25 | 0% |
| qwen2.5-1.5b | centered | 28/28 | 0.0017 [0.0014, 0.0020] | -0.0000 | 0.0017 [0.0014, 0.0020] | 29/29 | 0% |
| qwen2.5-1.5b | raw | 28/28 | 0.0006 [-0.0006, 0.0004] | -0.0010 | 0.0016 [0.0013, 0.0019] | 0/29 | 0% |

*Penalty at grid max*: share of folds where the strongest penalty won, i.e. the model chose to predict almost nothing (expected when there is no signal).

All layers: one CSV per model next to this report; plot `encoding_scan_zuco_frp.png`. Runtime 20.1 min.

## Is it more than word length, frequency and reading behaviour? (centered)

**Noise ceiling:** 2.8% of the word-to-word EEG variance repeats across readers (split-half reliability of the reader average, mean over the 32 components; best component 10.6%). No model can explain more than this.

* **lexical features alone** (log_length, zipf, zipf_sq, relative_position, is_first, is_last, punctuation): R² 0.0025 [0.0021, 0.0030]
* **lexical+reading features alone** (log_length, zipf, zipf_sq, relative_position, is_first, is_last, punctuation, share_fixated, n_fixations, log_trt, log_ffd): R² 0.0030 [0.0025, 0.0036]
* **lexical+surprisal features alone** (log_length, zipf, zipf_sq, relative_position, is_first, is_last, punctuation, surprisal): R² 0.0026 [0.0021, 0.0031]
* **lexical+reading+surprisal features alone** (log_length, zipf, zipf_sq, relative_position, is_first, is_last, punctuation, share_fixated, n_fixations, log_trt, log_ffd, surprisal): R² 0.0030 [0.0025, 0.0036]

| model | layer | text vectors R² | beyond lexical R² [95% CI] | beyond lexical+reading R² [95% CI] | beyond lexical+surprisal R² [95% CI] | beyond lexical+reading+surprisal R² [95% CI] | share of ceiling |
|---|---:|---|---|---|---|---|---:|
| labse | 5/12 | 0.0024 [0.0020, 0.0028] | 0.00004 [-0.00000, 0.00009] (2% of text vectors) | 0.00002 [-0.00003, 0.00006] (1% of text vectors) | 0.00004 [-0.00001, 0.00009] (2% of text vectors) | 0.00002 [-0.00003, 0.00006] (1% of text vectors) | 9% |
| xlmr-large | 8/24 | 0.0016 [0.0011, 0.0021] | 0.00001 [0.00001, 0.00002] (1% of text vectors) | 0.00001 [0.00000, 0.00002] (1% of text vectors) | 0.00001 [0.00001, 0.00002] (1% of text vectors) | 0.00001 [0.00000, 0.00002] (1% of text vectors) | 6% |
| me5-large | 8/24 | 0.0017 [0.0013, 0.0022] | 0.00002 [0.00001, 0.00003] (1% of text vectors) | 0.00001 [0.00001, 0.00002] (1% of text vectors) | 0.00001 [0.00001, 0.00002] (1% of text vectors) | 0.00001 [0.00001, 0.00002] (1% of text vectors) | 6% |
| qwen2.5-1.5b | 28/28 | 0.0017 [0.0014, 0.0020] | 0.00001 [0.00001, 0.00002] (1% of text vectors) | 0.00001 [0.00000, 0.00002] (1% of text vectors) | 0.00001 [0.00001, 0.00002] (1% of text vectors) | 0.00001 [0.00000, 0.00002] (1% of text vectors) | 6% |

* **lexical features alone predict EEG at least as well as every text model** (R² 0.0025 vs best text model 0.0024).
* **lexical+reading features alone predict EEG at least as well as every text model** (R² 0.0030 vs best text model 0.0024).
* **lexical+surprisal features alone predict EEG at least as well as every text model** (R² 0.0026 vs best text model 0.0024).
* **lexical+reading+surprisal features alone predict EEG at least as well as every text model** (R² 0.0030 vs best text model 0.0024).
* **Beyond lexical: nothing remains** — what the text vectors predict about EEG is accounted for by lexical features.
* **Beyond lexical+reading: nothing remains** — what the text vectors predict about EEG is accounted for by lexical+reading features.
* **Beyond lexical+surprisal: nothing remains** — what the text vectors predict about EEG is accounted for by lexical+surprisal features.
* **Beyond lexical+reading+surprisal: nothing remains** — what the text vectors predict about EEG is accounted for by lexical+reading+surprisal features.

R² per EEG component (labse layer 5; component 1 = largest EEG variance): #3 0.0180, #4 0.0172, #8 0.0077, #1 0.0067, #7 0.0064; 5 of 32 components above 0.005.

*Beyond* = R² for EEG with the control features' (training-set) linear prediction removed, as a share of what remains.
