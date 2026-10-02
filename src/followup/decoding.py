"""Decoding the read word from EEG: a controlled form of "EEG-to-text".

A ridge map from a word's reader-averaged EEG to the word's text-model vector
(by default the model's non-contextual input layer, i.e. word identity) is fit
on training sentences and evaluated on unseen sentences:

* 2-vs-2: for two test words, does pairing each prediction with its own word
  beat the swapped pairing? Chance is 50%.
* matched 2-vs-2: the same, using only pairs of words with equal length and
  similar frequency, so that length and frequency cannot decide.
* retrieval: rank of the true word among all word types in the test
  sentences; the top-ranked word per position gives the "decoded text".

The same pipeline runs on four inputs: real EEG; EEG shuffled across training
words (the map is learned without pairing); Gaussian noise; and word features
alone (length, frequency, position) in place of EEG. EEG plus word features is
also compared with word features alone. Nothing is ever conditioned on the
true previous word, so there is no teacher forcing.
"""

import numpy as np
import torch

from ..brainshaping.data import grouped_splits
from ..brainshaping.encoding import ALPHAS, ridge_predictor


def fit_predict(X_train, Y_train, X_test, groups_train, n_inner=4, alphas=ALPHAS, device="cpu", seed=0):
    """Ridge with the penalty chosen by sentence-grouped inner CV; returns test predictions."""
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)  # noqa: E731
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    Xtr, Xte = t((X_train - mean) / std), t((X_test - mean) / std)
    Ytr = t(Y_train)
    alphas_t = t(alphas)
    sse = torch.zeros(len(alphas), dtype=torch.float64, device=device)
    for a, b in grouped_splits(groups_train, n_inner, seed):
        predict = ridge_predictor(Xtr[a], Xtr[b])
        sse += ((predict(Ytr[a], alphas_t) - Ytr[b][None]) ** 2).sum(dim=(1, 2))
    best = int(sse.argmin())
    return ridge_predictor(Xtr, Xte)(Ytr, alphas_t[best:best + 1])[0].cpu().numpy(), float(alphas[best])


def _unit(a):
    a = a - a.mean(axis=1, keepdims=True)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def sample_pairs(word_ids, rng, n_pairs, lengths=None, zipf=None, max_zipf_gap=0.5):
    """Random pairs of items with different words; matched pairs share length and similar frequency."""
    n = len(word_ids)
    if lengths is None:
        i = rng.integers(0, n, size=4 * n_pairs)
        j = rng.integers(0, n, size=4 * n_pairs)
        keep = word_ids[i] != word_ids[j]
        return i[keep][:n_pairs], j[keep][:n_pairs]
    pairs = []
    by_length = {}
    for index, length in enumerate(lengths):
        by_length.setdefault(int(length), []).append(index)
    groups = [np.array(g) for g in by_length.values() if len(g) > 1]
    if not groups:
        return np.zeros(0, int), np.zeros(0, int)
    sizes = np.array([len(g) for g in groups], dtype=float)
    for _ in range(20 * n_pairs):
        g = groups[rng.choice(len(groups), p=sizes / sizes.sum())]
        i, j = rng.choice(g, 2, replace=False)
        if word_ids[i] != word_ids[j] and abs(zipf[i] - zipf[j]) <= max_zipf_gap:
            pairs.append((i, j))
            if len(pairs) >= n_pairs:
                break
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int)
    i, j = np.array(pairs).T
    return i, j


def two_vs_two(P, T, i, j):
    """Per pair: True when matched pairing is more similar than the swapped one (correlation)."""
    P, T = _unit(P), _unit(T)
    right = (P[i] * T[i]).sum(1) + (P[j] * T[j]).sum(1)
    wrong = (P[i] * T[j]).sum(1) + (P[j] * T[i]).sum(1)
    return right > wrong


def retrieval_ranks(P, type_vectors, type_of_item):
    """1-based rank of each item's own word type among ``type_vectors`` (cosine)."""
    scores = _unit(P) @ _unit(type_vectors).T
    own = scores[np.arange(len(P)), type_of_item]
    return 1 + (scores > own[:, None]).sum(axis=1), scores.argmax(axis=1)


def cluster_bootstrap_mean(values, clusters, n_boot=1000, seed=0):
    """Mean and 95% CI, resampling clusters (sentences)."""
    values = np.asarray(values, dtype=float)
    codes = np.unique(clusters, return_inverse=True)[1]
    n = codes.max() + 1
    sums = np.bincount(codes, weights=values, minlength=n)
    counts = np.bincount(codes, minlength=n).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(n, np.full(n, 1 / n), size=n_boot)
    boot = (w @ sums) / np.maximum(w @ counts, 1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_difference(a, b, clusters, n_boot=1000, seed=0):
    """Mean of a - b (per-pair outcomes) with a sentence-cluster bootstrap CI."""
    return cluster_bootstrap_mean(np.asarray(a, float) - np.asarray(b, float), clusters, n_boot, seed)
