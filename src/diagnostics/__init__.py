"""Diagnostics: where EEG + text sentiment models gain or lose information.

Stage 1 (signal): is reliable, sentiment-related information present in
word-level EEG? Stage 2 (representation): which feature representation keeps
it? Stage 3 (fusion): does the LoRA model read its EEG tokens? Stage 4
(errors): where does the text model fail, and does EEG help there?
"""
