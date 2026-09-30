"""Frozen NeuroLM/LaBraM EEG-only sentiment probe for raw ZuCo EEG.

This package is separate from the LaBSE + classical-feature pipeline in ``src``.
Modules avoid importing torch or the NeuroLM repository at import time so the
mapping, preprocessing, split, and statistics code can be tested without a GPU
or the pretrained checkpoint.
"""
