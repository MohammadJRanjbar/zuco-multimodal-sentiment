# Brain-tuned LoRA LLM (EEG as training signal) — results

Qwen/Qwen2.5-1.5B-Instruct, auxiliary head on layer 14, 32 EEG components, weight 0.5; 1 fold(s), 80 unseen test sentences. Text-only inference.

| arm | macro-F1 | 95% CI | accuracy | aux R² on real held-out EEG |
|---|---:|---|---:|---:|
| text_only | 0.802 | [0.709, 0.882] | 0.800 | — |
| eeg | 0.802 | [0.709, 0.882] | 0.800 | -0.118 |
| shuffled_eeg | 0.802 | [0.709, 0.882] | 0.800 | -0.132 |
| random_targets | 0.815 | [0.723, 0.892] | 0.812 | -0.133 |

| comparison | Δ macro-F1 | 95% CI | p |
|---|---:|---|---:|
| eeg − shuffled_eeg | 0.000 | [0.000, 0.000] | 1.000 |
| eeg − random_targets | -0.012 | [-0.040, 0.000] | 1.000 |
| eeg − text_only | 0.000 | [0.000, 0.000] | 1.000 |

EEG helps only if `eeg` beats `shuffled_eeg` and `random_targets` (CI above 0). The aux R² column checks that the model actually learned to predict real EEG.
