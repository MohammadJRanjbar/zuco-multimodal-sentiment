# Where does EEG + text sentiment fail? Research plan

## Question

Text alone classifies ZuCo sentence sentiment well (LaBSE ≈ 0.68 macro-F1).
Every EEG addition so far has given no alignment-specific gain:

- gated LaBSE fusion was matched by shuffled, noise, and zero EEG;
- frozen NeuroLM sentence embeddings decode sentiment at chance while
  identifying the reader 81% of the time.

Instead of searching architectures, this phase traces the sentiment
information from the brain to the prediction. The aim is to find where it is
lost and to choose the fix that stage calls for.

## Stages

Each stage pairs the sentiment question with *positive controls*: properties
known to modulate reading-related EEG. A failure can therefore be attributed
either to the data or to the model.

| stage | question | measurements | positive controls |
|---|---|---|---|
| 1 Signal | Is reliable, sentiment-related information present in word-level EEG? | variance shares (reader / word / residual); split-half reliability across readers (the ceiling for any model); ridge probes with label-permutation nulls on single-reader and reader-averaged word EEG; per-band probes; scalp correlation maps; valence beyond lexical covariates | word length, frequency, surprisal, reading time, content vs function word |
| 2 Representation | Which sentence-level representation keeps it? | the same probes on word-EEG means, NeuroLM embeddings, and handcrafted features, single-reader and reader-averaged | sentence length, mean frequency, mean surprisal, lexicon valence |
| 3 Fusion | Does the LoRA model read its EEG tokens? | attention from the answer to EEG vs word tokens per layer; gradient × input shares; predictions under shuffled, other-reader, and feature-free EEG; probes of raw input, projector output, and hidden EEG-slot states | the synthetic test, where EEG is the only signal |
| 4 Errors | Where does the text model fail, and does EEG help there? | arm comparisons within negation, contrast, lexicon-conflict, length, label, and low-confidence subsets; per-reader effects; whether EEG predicts text-model errors | — |

## Decision guide

| finding | implication | next step |
|---|---|---|
| Positive controls are decodable, sentiment is not | EEG carries reading information but not sentiment; the ceiling is in the data and labels | report the controlled negative result; consider reading targets or eye tracking |
| Sentiment is decodable only after averaging over readers | single-reader signal-to-noise is too low | reader-averaged EEG tokens |
| Reader variance dominates | reader identity masks content | per-reader alignment; reader-adversarial loss |
| Signal is present in raw EEG but lost in the projector or hidden states | the model fails to read it | contrastive EEG–word pretraining; word dropout |
| The model reacts to EEG but aligned ≈ shuffled | non-specific use (regularization) | as above, with controls |
| EEG predicts text-model errors | EEG is an uncertainty signal, not a feature source | confidence or routing models |

All tests use p < 0.05 against permutation nulls (50 permutations for
reader-averaged probes, 20 for single-reader probes). Folds are grouped by
sentence. Every gain must beat the shuffled-EEG control.

## Running

`notebooks/eeg_text_diagnostics_colab.ipynb` runs extraction, stages 1–2,
LoRA training with saved weights, stage 3, stage 4, and builds
`reports/eeg_text_diagnostics.md`.
