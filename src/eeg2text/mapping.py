"""Closed-form ridge map from word inputs (EEG or a control) to mBART word embeddings.

In the v2 check, the generator wrote unseen sentences from mBART's own word
embeddings, but the end-to-end input layer could not learn to read LaBSE word
vectors from ~280 training sentences (that control stayed at noise level). The
mapping is therefore fitted separately: ridge regression from each fixated
word's input to that word's mBART embedding, with the penalty chosen by
sentence-grouped inner cross-validation. The generator is then trained on the
predicted embeddings. Training trials get out-of-fold predictions (sentence
folds), so the generator sees the same kind of imperfect predictions in
training as at test time; validation and test trials are predicted by a map
fitted on all training trials. The same map is reused for test-time swaps.
"""

import numpy as np
import torch

from ..brainshaping.data import grouped_splits
from ..brainshaping.encoding import ALPHAS


class RidgeMap:
    """Ridge on standardized inputs; ``fit`` picks the penalty by sentence-grouped inner CV."""

    def __init__(self, n_inner=4, alphas=ALPHAS, device="cpu", seed=0):
        self.n_inner, self.alphas, self.device, self.seed = n_inner, np.asarray(alphas), device, seed

    def _t(self, a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float64, device=self.device)

    def fit(self, X, Y, groups):
        self.mean, self.std = X.mean(axis=0), X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        Xs, Yt = self._t((X - self.mean) / self.std), self._t(Y)
        sse = np.zeros(len(self.alphas))
        for a, b in grouped_splits(groups, self.n_inner, self.seed):
            x_mean, y_mean = Xs[a].mean(dim=0), Yt[a].mean(dim=0)
            Xa, Xb = Xs[a] - x_mean, Xs[b] - x_mean
            eigvals, vecs = torch.linalg.eigh(Xa.T @ Xa)
            eigvals = eigvals.clamp_min(0)
            projected, left, target = vecs.T @ (Xa.T @ (Yt[a] - y_mean)), Xb @ vecs, Yt[b] - y_mean
            for k, alpha in enumerate(self.alphas):  # one penalty at a time: one prediction matrix in memory
                sse[k] += float(((left @ (projected / (eigvals + alpha)[:, None]) - target) ** 2).sum())
        self.alpha = float(self.alphas[int(sse.argmin())])
        self.x_mean, self.y_mean = Xs.mean(dim=0), Yt.mean(dim=0)
        Xc = Xs - self.x_mean
        gram = Xc.T @ Xc + self.alpha * torch.eye(Xc.shape[1], dtype=Xc.dtype, device=Xc.device)
        self.weights = torch.linalg.solve(gram, Xc.T @ (Yt - self.y_mean))
        return self

    def predict(self, X):
        Xs = self._t((X - self.mean) / self.std)
        return ((Xs - self.x_mean) @ self.weights + self.y_mean).float().cpu().numpy()


def _rows(inputs):
    """Stacked fixated-word inputs with (trial, word position) of each row."""
    trial = np.concatenate([np.full(int(m.sum()), i) for i, (_, m) in enumerate(inputs)])
    position = np.concatenate([np.flatnonzero(m) for _, m in inputs])
    X = np.concatenate([x[m] for x, m in inputs]).astype(np.float64)
    return X, trial, position


def _assemble(inputs, predicted, trial, position, dim):
    out = [(np.zeros((len(m), dim), np.float32), m) for _, m in inputs]
    for row, (i, j) in enumerate(zip(trial, position)):
        out[i][0][j] = predicted[row]
    return out


def quality(mapped, targets, held_out):
    """Mean cosine and R² of predicted vs true embeddings over fixated words of ``held_out`` trials."""
    pred = [x[m] for (x, m), keep in zip(mapped, held_out) if keep]
    true = [t[m] for (_, m), t, keep in zip(mapped, targets, held_out) if keep]
    if not pred:
        return None
    P, T = np.concatenate(pred), np.concatenate(true)
    ok = np.isfinite(T).all(axis=1)
    P, T = P[ok].astype(np.float64), T[ok].astype(np.float64)
    cosine = (P * T).sum(1) / np.maximum(np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1), 1e-12)
    r2 = 1 - ((P - T) ** 2).sum() / max(((T - T.mean(axis=0)) ** 2).sum(), 1e-12)
    return {"words": int(len(P)), "cosine": float(cosine.mean()), "r2": float(r2)}


def map_inputs(inputs, targets, part, sentence_id, n_folds=5, device="cpu", seed=0):
    """Replace each fixated word's input by its predicted mBART embedding.

    ``inputs``: per trial ``(x [n_words, F], fixated)``; ``targets``: per trial [n_words, D] embeddings (NaN rows
    for words without one). Returns the mapped inputs, the map fitted on all training trials, and the mapping
    quality on validation + test words.
    """
    X, trial, position = _rows(inputs)
    Y = np.concatenate([t[m] for (_, m), t in zip(inputs, targets)]).astype(np.float64)
    has_target = np.isfinite(Y).all(axis=1)
    groups = np.asarray(sentence_id)[trial]
    train = np.asarray(part)[trial] == "train"
    predicted = np.zeros_like(Y, dtype=np.float32)
    train_rows = np.flatnonzero(train)
    for a, b in grouped_splits(groups[train_rows], n_folds, seed):  # out-of-fold predictions for training rows
        fit_rows = train_rows[a][has_target[train_rows[a]]]
        fold_map = RidgeMap(device=device, seed=seed).fit(X[fit_rows], Y[fit_rows], groups[fit_rows])
        predicted[train_rows[b]] = fold_map.predict(X[train_rows[b]])
    fit_rows = train_rows[has_target[train_rows]]
    full = RidgeMap(device=device, seed=seed).fit(X[fit_rows], Y[fit_rows], groups[fit_rows])
    if (~train).any():
        predicted[~train] = full.predict(X[~train])
    mapped = _assemble(inputs, predicted, trial, position, Y.shape[1])
    return mapped, full, quality(mapped, targets, np.asarray(part) != "train")


def apply_map(ridge_map, inputs):
    """Inputs mapped by an already fitted map (test-time swaps)."""
    X, trial, position = _rows(inputs)
    return _assemble(inputs, ridge_map.predict(X), trial, position, ridge_map.weights.shape[1])
