"""Word-level EEG targets and contextual word vectors shared by both versions."""

import hashlib
import json
import os

import numpy as np
import pandas as pd

from ..config import LABEL_TO_ID
from ..diagnostics import signal

TARGET_ARMS = ("eeg", "shuffled_eeg", "random_targets")


def sentence_table(trials):
    """One row per sentence: id, words, label id (all readers read the same words)."""
    seen = {}
    for trial in trials:
        seen.setdefault(trial["sentence_id"], (trial["words"], trial["label"]))
    rows = [{"sentence_id": sid, "words": words, "label_id": LABEL_TO_ID[label]}
            for sid, (words, label) in sorted(seen.items())]
    return pd.DataFrame(rows)


def item_eeg(trials):
    """Reader-averaged, per-reader-normalized word EEG: one row per (sentence, word)."""
    meta, X, _ = signal.long_word_table(trials)
    Z = signal.zscore_per_reader(meta, X)
    del X
    items, averaged = signal.reader_average(meta, Z)
    return items[["sentence_id", "word_index", "n_readers"]].reset_index(drop=True), averaged


def fit_targets(train_eeg, k=32):
    """PCA of training-item EEG (centered); returns a transform to k standardized scores."""
    mean = train_eeg.mean(axis=0)
    _, singular, vt = np.linalg.svd(train_eeg - mean, full_matrices=False)
    components = vt[:k]
    scale = (singular[:k] / np.sqrt(max(len(train_eeg) - 1, 1)))
    scale[scale < 1e-8] = 1.0

    def transform(eeg):
        return ((eeg - mean) @ components.T / scale).astype(np.float32)

    explained = float((singular[:k] ** 2).sum() / (singular ** 2).sum())
    return transform, explained


def arm_targets(arm, targets, train_rows, rng):
    """Targets for one control arm; only training rows are ever used for fitting."""
    out = targets.copy()
    if arm == "eeg":
        return out
    if arm == "shuffled_eeg":
        out[train_rows] = targets[rng.permutation(train_rows)]
        return out
    if arm == "random_targets":
        return rng.standard_normal(targets.shape).astype(np.float32)
    raise ValueError(f"unknown target arm {arm!r}")


def contextual_word_vectors(sentences, model_name, device="cpu", batch_size=32, cache_path=None, layer=-1):
    """Mean of sub-word hidden states per word, from a frozen encoder (e.g. LaBSE).

    ``sentences``: list of word lists. Returns a list of ``[n_words, dim]`` arrays.
    Cached to ``cache_path`` keyed by model, layer, and the exact word lists.
    """
    key = hashlib.sha256(json.dumps([model_name, layer, sentences]).encode()).hexdigest()[:16]
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["key"]) == key:
            offsets = cached["offsets"]
            return [cached["vectors"][offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    vectors = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        encoded = tokenizer(batch, is_split_into_words=True, padding=True, truncation=True,
                            max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in encoded.items()}, output_hidden_states=True)
        hidden = out.hidden_states[layer].float().cpu().numpy()
        for b, words in enumerate(batch):
            word_ids = encoded.word_ids(b)
            pooled = np.zeros((len(words), hidden.shape[-1]), dtype=np.float32)
            counts = np.zeros(len(words))
            for position, word in enumerate(word_ids):
                if word is not None:
                    pooled[word] += hidden[b, position]
                    counts[word] += 1
            pooled /= np.maximum(counts, 1)[:, None]
            vectors.append(pooled)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        offsets = np.cumsum([0] + [len(v) for v in vectors])
        np.savez(cache_path, key=key, offsets=offsets, vectors=np.concatenate(vectors))
    return vectors
