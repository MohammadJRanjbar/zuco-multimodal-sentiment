# Do word vectors predict word-level EEG? (zuco_frp)

400 sentences, 5574 words with EEG (832 EEG features per word, averaged over readers). Targets: top 32 EEG principal components (share of EEG variance: centered 86%, raw 86%). 5 folds split by sentence; the ridge penalty is chosen inside the training sentences only.

**How to read:** R² is measured on sentences the model never saw. Above 0 means the word vectors predict EEG better than the average does. `shuffled` is the same pipeline with EEG targets shuffled across words (should be about 0). *centered*: only word-to-word differences within a sentence; *raw*: also differences between sentences. A layer **passes** when both its R² and its margin over shuffled have 95% CIs above 0.

## Verdict

* **centered: PASS** — best labse layer 8/12: R² 0.0006 [0.0003, 0.0010], above shuffled by 0.0007 [0.0003, 0.0010]; 58 of 92 model-layers pass.
* **raw: FAIL** — best labse layer 8/12: R² -0.0002 [-0.0012, -0.0005], above shuffled by 0.0005 [0.0003, 0.0007]; 0 of 92 model-layers pass.

> The best layer is the maximum over 92 model-layers, so a single layer barely above 0 is weak evidence. Convincing: several neighbouring layers pass, and the same models and depths also pass on the other language.

## Best layer per model

| model | mode | best layer | R² [95% CI] | shuffled R² | R² − shuffled [95% CI] | layers passing | penalty at grid max |
|---|---|---:|---|---:|---|---:|---:|
| labse | centered | 8/12 | 0.0006 [0.0003, 0.0010] | -0.0000 | 0.0007 [0.0003, 0.0010] | 13/13 | 0% |
| labse | raw | 8/12 | -0.0002 [-0.0012, -0.0005] | -0.0007 | 0.0005 [0.0003, 0.0007] | 0/13 | 0% |
| xlmr-large | centered | 0/24 | 0.0005 [0.0003, 0.0007] | -0.0000 | 0.0005 [0.0004, 0.0007] | 20/25 | 0% |
| xlmr-large | raw | 0/24 | -0.0002 [-0.0012, -0.0006] | -0.0007 | 0.0005 [0.0003, 0.0006] | 0/25 | 0% |
| me5-large | centered | 0/24 | 0.0005 [0.0004, 0.0007] | -0.0000 | 0.0005 [0.0004, 0.0007] | 21/25 | 0% |
| me5-large | raw | 0/24 | -0.0002 [-0.0012, -0.0006] | -0.0007 | 0.0005 [0.0003, 0.0006] | 0/25 | 2% |
| qwen2.5-1.5b | centered | 28/28 | 0.0005 [0.0003, 0.0007] | -0.0000 | 0.0005 [0.0003, 0.0007] | 4/29 | 8% |
| qwen2.5-1.5b | raw | 28/28 | -0.0004 [-0.0014, -0.0007] | -0.0007 | 0.0003 [0.0001, 0.0005] | 0/29 | 13% |

*Penalty at grid max*: share of folds where the strongest penalty won, i.e. the model chose to predict almost nothing (expected when there is no signal).

All layers: one CSV per model next to this report; plot `encoding_scan_zuco_frp.png`. Runtime 1.7 min.

## Is it more than word length, frequency and reading behaviour? (centered)

* **lexical features alone** (log_length, zipf, zipf_sq, relative_position, is_first, is_last, punctuation): R² 0.0008 [0.0005, 0.0011]
* **lexical+reading features alone** (log_length, zipf, zipf_sq, relative_position, is_first, is_last, punctuation, share_fixated, n_fixations, log_trt, log_ffd): R² 0.0014 [0.0011, 0.0018]
* **lexical+surprisal features alone** (log_length, zipf, zipf_sq, relative_position, is_first, is_last, punctuation, surprisal): R² 0.0008 [0.0005, 0.0012]
* **lexical+reading+surprisal features alone** (log_length, zipf, zipf_sq, relative_position, is_first, is_last, punctuation, share_fixated, n_fixations, log_trt, log_ffd, surprisal): R² 0.0014 [0.0010, 0.0018]

| model | layer | text vectors R² | beyond lexical R² [95% CI] | beyond lexical+reading R² [95% CI] | beyond lexical+surprisal R² [95% CI] | beyond lexical+reading+surprisal R² [95% CI] | share of ceiling |
|---|---:|---|---|---|---|---|---:|
| labse | 8/12 | 0.0006 [0.0003, 0.0010] | 0.00001 [0.00000, 0.00001] (1% of text vectors) | 0.00001 [0.00000, 0.00001] (1% of text vectors) | 0.00001 [0.00000, 0.00001] (1% of text vectors) | 0.00001 [0.00000, 0.00001] (1% of text vectors) | — |
| xlmr-large | 0/24 | 0.0005 [0.0003, 0.0007] | 0.00001 [0.00001, 0.00002] (2% of text vectors) | 0.00001 [0.00001, 0.00002] (2% of text vectors) | 0.00001 [0.00001, 0.00002] (3% of text vectors) | 0.00001 [0.00001, 0.00002] (2% of text vectors) | — |
| me5-large | 0/24 | 0.0005 [0.0004, 0.0007] | 0.00001 [0.00001, 0.00002] (2% of text vectors) | 0.00001 [0.00001, 0.00002] (2% of text vectors) | 0.00001 [0.00001, 0.00002] (2% of text vectors) | 0.00001 [0.00001, 0.00002] (2% of text vectors) | — |
| qwen2.5-1.5b | 28/28 | 0.0005 [0.0003, 0.0007] | 0.00001 [0.00000, 0.00001] (2% of text vectors) | 0.00001 [-0.00000, 0.00001] (1% of text vectors) | 0.00001 [-0.00000, 0.00001] (1% of text vectors) | 0.00001 [-0.00000, 0.00001] (1% of text vectors) | — |

* **lexical features alone predict EEG at least as well as every text model** (R² 0.0008 vs best text model 0.0006).
* **lexical+reading features alone predict EEG at least as well as every text model** (R² 0.0014 vs best text model 0.0006).
* **lexical+surprisal features alone predict EEG at least as well as every text model** (R² 0.0008 vs best text model 0.0006).
* **lexical+reading+surprisal features alone predict EEG at least as well as every text model** (R² 0.0014 vs best text model 0.0006).
* **Beyond lexical: nothing remains** — what the text vectors predict about EEG is accounted for by lexical features.
* **Beyond lexical+reading: nothing remains** — what the text vectors predict about EEG is accounted for by lexical+reading features.
* **Beyond lexical+surprisal: nothing remains** — what the text vectors predict about EEG is accounted for by lexical+surprisal features.
* **Beyond lexical+reading+surprisal: nothing remains** — what the text vectors predict about EEG is accounted for by lexical+reading+surprisal features.

R² per EEG component (labse layer 8; component 1 = largest EEG variance): #3 0.0078, #1 0.0061, #4 0.0032, #32 0.0019, #19 0.0017; 2 of 32 components above 0.005.

*Beyond* = R² for EEG with the control features' (training-set) linear prediction removed, as a share of what remains.
