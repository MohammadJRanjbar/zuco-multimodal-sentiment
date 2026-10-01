"""EEG-shaped word representations for sentiment (EEG used only during training).

A (static): frozen contextual word vectors; the directions that predict
word-level EEG are amplified before a sentiment classifier.
B (brain-tuning): a LoRA LLM is trained for sentiment while an auxiliary head
predicts each word's EEG from its hidden state.

Controls in both: no EEG objective, EEG targets shuffled across words, and
random targets. EEG helps only if real targets beat the shuffled ones.
Nothing here depends on the language or the EEG montage: each dataset only
needs per-word EEG vectors, so Persian data can use its own targets.
"""
