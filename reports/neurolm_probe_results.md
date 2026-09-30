# Frozen NeuroLM/LaBraM EEG-only sentiment probe — results

**No results yet.** The pipeline is implemented and tested offline, but it has not been run on the real
ZuCo EEG. This file is overwritten by `scripts/build_neurolm_report.py` (step 12 of
`notebooks/neurolm_probe_colab.ipynb`) once extraction, probes, and sanity checks have run on Colab.

Predeclared before any result (see `configs/neurolm_probe.yaml`):

* primary: view `tenten6_chunk1024` (42 channels), feature `tokenizer__mean` (frozen NeuroLM VQ encoder,
  mean over tokens), logistic regression, **unseen subject + unseen sentence** protocol, 5 seeds;
* above chance ⇔ sentence-level label-permutation p < 0.05, macro-F1 ≥ null mean + 0.02, and ≥ 4/5 seeds
  above the null mean;
* NeuroLM beats handcrafted ⇔ paired sentence-cluster 95% CI of Δ macro-F1 excludes 0 and paired
  permutation p < 0.05 (same trials, same seeds, same classifier);
* recommendation: *stop* if the primary test fails; *fine-tune the encoder* if it passes; fusion only
  after a fine-tuned EEG-only model is meaningfully predictive.
