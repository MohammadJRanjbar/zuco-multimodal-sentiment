"""Stage 1: is there reliable, target-related information in word-level EEG?

Word EEG = ZuCo band power over each word's reading time (8 bands x 104
electrodes; Cz dropped), log-transformed when positive. Analyses:

* variance components (reader / word item / residual) on the raw features;
* split-half reliability across readers: an upper bound on what any model can
  extract from the reader-averaged signal, and the implied single-reader bound;
* cross-validated ridge probes (grouped by sentence, so no sentence is split
  across train/test) with label-permutation nulls, for positive-control and
  sentiment targets, on single-reader and reader-averaged EEG;
* per-band probes and univariate per-channel correlation maps.

Per-reader z-scoring is label-free and uses each reader's own words only.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from ..config import ZUCO_REFERENCE_CHANNEL_INDEX
from ..fusion.word_eeg import BANDS

ALPHAS = np.logspace(-1, 6, 15)


# ----------------------------------------------------------------------------
# Data assembly
# ----------------------------------------------------------------------------


def long_word_table(trials, drop_channels=(ZUCO_REFERENCE_CHANNEL_INDEX,), log_power="auto"):
    """Rows = (reader, sentence, word) with EEG; returns (meta DataFrame, X float32)."""
    keep = None
    meta, blocks = [], []
    for trial in trials:
        block = np.asarray(trial["features"], dtype=np.float32)
        if keep is None:
            keep = [c for c in range(block.shape[2]) if c not in set(drop_channels)]
            n_channels = len(keep)
        flat = block[:, :, keep].reshape(len(block), -1)
        present = np.isfinite(flat).all(axis=1)
        for index in np.flatnonzero(present):
            meta.append({"reader": trial["subject_id"], "sentence_id": trial["sentence_id"],
                         "word_index": int(index), "label": trial["label"],
                         "trt_ms": float(trial.get("trt_ms", np.full(len(block), np.nan))[index]),
                         "n_fixations": float(trial["fixations"][index])})
        blocks.append(flat[present])
    X = np.concatenate(blocks).astype(np.float32)
    use_log = log_power is True or (log_power == "auto" and (X > 0).all())
    if use_log:
        X = np.log(X)
    frame = pd.DataFrame(meta)
    frame["item"] = frame["sentence_id"].astype(str) + ":" + frame["word_index"].astype(str)
    return frame, X, {"log_transformed": bool(use_log), "n_channels": n_channels, "n_bands": len(BANDS)}


def zscore_per_reader(meta, X):
    Z = np.empty_like(X)
    for _, rows in meta.groupby("reader").indices.items():
        block = X[rows]
        std = block.std(axis=0)
        std[std < 1e-8] = 1.0
        Z[rows] = (block - block.mean(axis=0)) / std
    return Z


def reader_average(meta, Z):
    """Average z-scored word EEG over readers: one row per (sentence, word) item."""
    codes, items = pd.factorize(meta["item"])
    sums = np.zeros((len(items), Z.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, Z)
    counts = np.bincount(codes, minlength=len(items))
    averaged = (sums / counts[:, None]).astype(np.float32)
    first = meta.drop_duplicates("item").set_index("item").loc[items]
    item_meta = pd.DataFrame({"item": items, "sentence_id": first["sentence_id"].to_numpy(),
                              "word_index": first["word_index"].to_numpy(), "label": first["label"].to_numpy(),
                              "n_readers": counts,
                              "trt_ms": meta.groupby("item")["trt_ms"].mean().loc[items].to_numpy()})
    return item_meta, averaged


# ----------------------------------------------------------------------------
# Variance components and reliability
# ----------------------------------------------------------------------------


def variance_components(meta, X):
    """Per-feature share of variance: reader means, item means (within reader), residual."""
    total = X.var(axis=0)
    reader_codes = pd.factorize(meta["reader"])[0]
    reader_means = np.zeros((reader_codes.max() + 1, X.shape[1]))
    np.add.at(reader_means, reader_codes, X)
    reader_means /= np.bincount(reader_codes)[:, None]
    within = X - reader_means[reader_codes]
    item_codes = pd.factorize(meta["item"])[0]
    item_means = np.zeros((item_codes.max() + 1, X.shape[1]))
    np.add.at(item_means, item_codes, within)
    item_means /= np.bincount(item_codes)[:, None]
    reader_var = (reader_means[reader_codes] - X.mean(axis=0)).var(axis=0)
    item_var = item_means[item_codes].var(axis=0)
    residual = (within - item_means[item_codes]).var(axis=0)
    safe = np.where(total > 0, total, 1.0)
    return pd.DataFrame({"reader": reader_var / safe, "word_item": item_var / safe, "residual": residual / safe})


def split_half_reliability(meta, Z, n_splits=20, min_readers=4, seed=0):
    """Per-feature split-half correlation of reader-averaged word EEG across items.

    Returns the mean split-half r, its Spearman-Brown projection to all
    readers (reliability of the full reader average), and the implied
    single-reader reliability.
    """
    rng = np.random.default_rng(seed)
    groups = {item: rows for item, rows in meta.groupby("item").indices.items() if len(rows) >= min_readers}
    items = list(groups)
    rs, half_sizes = [], []
    for _ in range(n_splits):
        a = np.empty((len(items), Z.shape[1]))
        b = np.empty_like(a)
        for i, item in enumerate(items):
            rows = rng.permutation(groups[item])
            half = len(rows) // 2
            a[i] = Z[rows[:half]].mean(axis=0)
            b[i] = Z[rows[half:2 * half]].mean(axis=0)
            half_sizes.append(half)
        a -= a.mean(axis=0)
        b -= b.mean(axis=0)
        denominator = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
        rs.append(np.where(denominator > 0, (a * b).sum(axis=0) / np.maximum(denominator, 1e-12), 0.0))
    r_half = np.mean(rs, axis=0)
    k = np.mean(half_sizes)
    full = np.array([len(groups[i]) for i in items]).mean()
    clipped = np.clip(r_half, -0.99, 0.999)
    r_full = (full / k) * clipped / (1 + (full / k - 1) * clipped)
    r_single = clipped / (k - (k - 1) * clipped)
    return pd.DataFrame({"split_half_r": r_half, "reliability_all_readers": r_full,
                         "reliability_single_reader": r_single}), {"items": len(items), "readers_per_half": k,
                                                                    "readers_per_item": full}


# ----------------------------------------------------------------------------
# Ridge probes with permutation nulls
# ----------------------------------------------------------------------------


def _ridge_fold(X_train, Y_train, X_test, alphas):
    """Eigen-decomposed ridge for one fold; returns a solver for any target.

    With ``X'X = V diag(l) V'`` and ``b = V'X'y``, the coefficients are
    ``b / (l + alpha)`` and the training residual is
    ``|y|^2 - 2 c'b + c' diag(l) c``, so generalized cross-validation over all
    alphas costs O(features) per target; a permutation adds one ``X'y``.
    """
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    Xtr = (X_train - mean) / std
    Xte = (X_test - mean) / std
    eigvals, V = np.linalg.eigh(Xtr.T @ Xtr)
    eigvals = np.clip(eigvals, 0, None)
    XteV = Xte @ V
    n = len(Xtr)

    def solve(Y):
        offset = Y.mean(axis=0)
        Yc = Y - offset
        b = V.T @ (Xtr.T @ Yc)
        total = (Yc ** 2).sum()
        best = None
        for alpha in alphas:
            coef = b / (eigvals + alpha)[:, None]
            residual = total - 2 * (coef * b).sum() + (eigvals[:, None] * coef ** 2).sum()
            dof = (eigvals / (eigvals + alpha)).sum()
            gcv = max(residual, 0.0) / max(n - dof, 1.0) ** 2
            if best is None or gcv < best[0]:
                best = (gcv, alpha, coef)
        return XteV @ best[2] + offset, best[1]

    return solve, None


def _score(task, y_true, prediction, classes=None):
    if task == "regression":
        return float(r2_score(y_true, prediction[:, 0]))
    if task == "binary":
        return float(roc_auc_score(y_true, prediction[:, 1] - prediction[:, 0]))
    return float(f1_score(y_true, classes[prediction.argmax(axis=1)], average="macro"))


def _encode(task, y):
    if task == "regression":
        return y.astype(np.float64)[:, None], None
    classes = np.unique(y)
    return (y[:, None] == classes[None, :]).astype(np.float64), classes


def ridge_probe(X, y, groups, task="regression", n_splits=5, n_perm=0, perm_units=None, seed=0,
                alphas=ALPHAS):
    """Cross-validated ridge probe; returns score, chance, p-value, and null summary.

    Scores: R^2 (regression), AUC (binary), macro-F1 (multiclass). Folds are
    grouped by ``groups``. Permutations shuffle the target among
    ``perm_units`` (e.g. items or sentences), keeping repeated measurements
    of one unit on one label; each permutation re-selects alpha.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    keep = np.isfinite(y) if np.issubdtype(y.dtype, np.floating) else np.ones(len(y), dtype=bool)
    X, y, groups = X[keep], y[keep], np.asarray(groups)[keep]
    units = np.asarray(perm_units)[keep] if perm_units is not None else np.arange(len(y))
    metric = {"regression": "R2", "binary": "AUC", "multiclass": "macro_F1"}[task]
    if len(y) < 20 or len(np.unique(groups)) < 2 or (task != "regression" and len(np.unique(y)) < 2):
        return {"task": task, "n": int(len(y)), "score": float("nan"), "metric": metric,
                "skipped": "too few rows, groups, or classes"}
    Y, classes = _encode(task, y)
    folds = list(GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(X, groups=groups))
    rng = np.random.default_rng(seed)
    permutations = []
    unit_ids, unit_first, unit_index = np.unique(units, return_index=True, return_inverse=True)
    for _ in range(n_perm):
        order = rng.permutation(len(unit_ids))
        permutations.append(unit_first[order][unit_index])
    predictions = np.zeros_like(Y)
    null_predictions = [np.zeros_like(Y) for _ in permutations]
    alphas_used = []
    for train, test in folds:
        solve, _ = _ridge_fold(X[train], Y[train], X[test], alphas)
        predictions[test], alpha = solve(Y[train])
        alphas_used.append(float(alpha))
        for p, source in enumerate(permutations):
            null_predictions[p][test], _ = solve(Y[source][train])
    observed = _score(task, y, predictions, classes)
    out = {"task": task, "n": int(len(y)), "score": observed, "alphas": alphas_used, "metric": metric}
    if permutations:
        null = np.array([_score(task, y[source], pred, classes) for source, pred in zip(permutations, null_predictions)])
        out.update({"null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
                    "p_value": float((1 + (null >= observed).sum()) / (1 + len(null)))})
    return out


# ----------------------------------------------------------------------------
# Univariate maps
# ----------------------------------------------------------------------------


def feature_correlations(X, y):
    """Pearson r of every feature with a continuous target -> ``[bands, channels]``."""
    keep = np.isfinite(y)
    Xc = X[keep] - X[keep].mean(axis=0)
    yc = y[keep] - y[keep].mean()
    denominator = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    r = (Xc * yc[:, None]).sum(axis=0) / np.maximum(denominator, 1e-12)
    return r.reshape(len(BANDS), -1)


def band_slices(n_channels):
    return {band: slice(i * n_channels, (i + 1) * n_channels) for i, band in enumerate(BANDS)}
