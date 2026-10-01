"""Stage 2: which sentence-level EEG representation keeps which information?

Every representation is probed for the same sentence-level targets: sentiment
(the target of interest) and text-derived positive controls (sentence length,
mean word frequency, mean surprisal, mean lexicon valence). Single-reader
representations are evaluated with folds grouped by sentence; reader-averaged
ones have one row per sentence. Each representation is z-scored per reader
(label-free) before pooling.
"""

import numpy as np
import pandas as pd

from .signal import ridge_probe

SENTENCE_TARGETS = {
    "sentence_label": "multiclass",
    "n_words": "regression",
    "mean_zipf": "regression",
    "mean_surprisal": "regression",
    "mean_valence": "regression",
}


def _zscore_rows(frame, X):
    Z = np.empty_like(X, dtype=np.float32)
    for _, rows in frame.groupby("reader").indices.items():
        block = X[rows].astype(np.float64)
        block = np.where(np.isfinite(block), block, np.nanmean(block, axis=0))
        block = np.nan_to_num(block)
        std = block.std(axis=0)
        std[std < 1e-8] = 1.0
        Z[rows] = (block - block.mean(axis=0)) / std
    return Z


def word_eeg_sentence_means(meta, Z):
    """Mean word EEG per (reader, sentence)."""
    keys = meta["reader"].astype(str) + "|" + meta["sentence_id"].astype(str)
    codes, uniques = pd.factorize(keys)
    sums = np.zeros((len(uniques), Z.shape[1]))
    np.add.at(sums, codes, Z)
    X = sums / np.bincount(codes)[:, None]
    readers, sentences = zip(*(k.split("|") for k in uniques))
    return pd.DataFrame({"reader": readers, "sentence_id": np.array(sentences, dtype=int)}), X.astype(np.float32)


def average_over_readers(frame, X):
    codes, sentences = pd.factorize(frame["sentence_id"])
    sums = np.zeros((len(sentences), X.shape[1]))
    np.add.at(sums, codes, X)
    return pd.DataFrame({"sentence_id": sentences}), (sums / np.bincount(codes)[:, None]).astype(np.float32)


def build_representations(meta, Z, extra=None):
    """``extra``: {name: (frame with reader/sentence_id, X)} for NeuroLM, handcrafted, ..."""
    out = {}
    frame, X = word_eeg_sentence_means(meta, Z)
    out["word_eeg_mean · single reader"] = (frame, X)
    out["word_eeg_mean · reader-averaged"] = average_over_readers(frame, X)
    for name, (frame, X) in (extra or {}).items():
        Zx = _zscore_rows(frame, X)
        out[f"{name} · single reader"] = (frame, Zx)
        out[f"{name} · reader-averaged"] = average_over_readers(frame, Zx)
    return out


def probe_representations(representations, sentence_table, n_perm=50, seed=0, targets=SENTENCE_TARGETS):
    rows = []
    table = sentence_table.set_index("sentence_id")
    for name, (frame, X) in representations.items():
        sentence_ids = frame["sentence_id"].to_numpy()
        for target, task in targets.items():
            if target not in table or table[target].isna().all():
                continue
            y = table.loc[sentence_ids, target].to_numpy()
            if task == "multiclass":
                y = y.astype(int)
            result = ridge_probe(X, y, groups=sentence_ids, task=task, n_perm=n_perm,
                                 perm_units=sentence_ids, seed=seed)
            rows.append({"representation": name, "target": target, "rows": result["n"], "metric": result["metric"],
                         "score": result["score"], "null_mean": result.get("null_mean"),
                         "null_q95": result.get("null_q95"), "p_value": result.get("p_value")})
    return pd.DataFrame(rows)
