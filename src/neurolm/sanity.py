"""Mandatory controls for the EEG-only probe.

* label permutation (sentence-level and trial-level) and its empirical null;
* shuffled sentence association: EEG rows are re-assigned to other sentences
  of the same subject inside each split partition, so labels stay balanced and
  no row crosses a split boundary;
* subject-identity decoding from the same embeddings;
* Gaussian random features of the same dimensionality;
* label-distribution baselines;
* embedding collapse diagnostics;
* a duration-only baseline (trial length is a potential sentence-level confound).
"""

import numpy as np
from sklearn.metrics import f1_score

from .evaluation import run_probe


def permute_labels(samples, rng, level="sentence", column="label_id"):
    """Return a copy whose ``column`` holds permuted labels.

    Sentence-level permutation keeps every subject's trials of one sentence on
    the same (new) label, which is the correct null for "is sentiment
    decodable", because it preserves the sentence structure of the data.
    ``label_id`` itself is left untouched when ``column`` differs, so splits
    (stratified on the true sentence labels) stay identical to the observed run.
    """
    permuted = samples.copy()
    if level == "trial":
        permuted[column] = rng.permutation(samples["label_id"].to_numpy())
        return permuted
    if level != "sentence":
        raise ValueError("level must be 'sentence' or 'trial'")
    table = samples.drop_duplicates("sentence_id").set_index("sentence_id")["label_id"]
    shuffled = dict(zip(table.index, rng.permutation(table.to_numpy())))
    permuted[column] = samples["sentence_id"].map(shuffled).astype(int)
    return permuted


def permutation_null(X, samples, *, protocol, classifier, seeds, n_permutations, observed,
                     level="sentence", protocol_params=None, grid=None, seed=0):
    """Empirical null of macro-F1 under permuted labels.

    Each permutation re-runs the full pipeline, including hyperparameter
    selection. Permutation k uses the single seed ``seeds[k % len(seeds)]``;
    a single-seed null is wider than the seed-averaged ``observed`` value, so
    the p-value is conservative.
    """
    rng = np.random.default_rng(seed)
    values = []
    for k in range(n_permutations):
        permuted = permute_labels(samples, rng, level, column="label_permuted")
        result = run_probe(
            X, permuted, protocol=protocol, classifier=classifier, seeds=[seeds[k % len(seeds)]],
            protocol_params=protocol_params, grid=grid, target="label_permuted", keep_predictions=False,
        )
        values.append(result["summary"]["macro_f1"]["mean"])
    values = np.asarray(values)
    return {
        "level": level,
        "protocol": protocol,
        "classifier": classifier,
        "n_permutations": int(n_permutations),
        "observed_seed_mean": float(observed),
        "null_mean": float(values.mean()),
        "null_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "null_q95": float(np.quantile(values, 0.95)),
        "p_value": float((1 + np.sum(values >= observed)) / (1 + len(values))),
        "null_values": values.tolist(),
    }


def shuffle_sentence_association(X, samples, split, rng):
    """Permute EEG rows among one subject's trials within each partition."""
    shuffled = X.copy()
    subjects = samples["subject_id"].to_numpy()
    for part in (split.train, split.val, split.test):
        part = np.asarray(part)
        for subject in np.unique(subjects[part]):
            rows = part[subjects[part] == subject]
            shuffled[rows] = X[rng.permutation(rows)]
    return shuffled


def gaussian_features(n_rows, dim, seed=0):
    return np.random.default_rng(seed).standard_normal((n_rows, dim)).astype(np.float32)


def label_distribution_baselines(labels, n_classes=3, n_simulations=500, seed=0):
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=n_classes)
    prior = counts / counts.sum()
    majority = int(counts.argmax())
    majority_pred = np.full_like(labels, majority)
    rng = np.random.default_rng(seed)
    uniform_f1, prior_f1 = [], []
    for _ in range(n_simulations):
        uniform_f1.append(f1_score(labels, rng.integers(0, n_classes, len(labels)), average="macro",
                                   labels=list(range(n_classes)), zero_division=0))
        prior_f1.append(f1_score(labels, rng.choice(n_classes, len(labels), p=prior), average="macro",
                                 labels=list(range(n_classes)), zero_division=0))
    return {
        "class_counts": counts.tolist(),
        "class_prior": prior.tolist(),
        "majority_class": majority,
        "majority_accuracy": float(prior[majority]),
        "majority_macro_f1": float(f1_score(labels, majority_pred, average="macro",
                                            labels=list(range(n_classes)), zero_division=0)),
        "uniform_random_expected_accuracy": 1.0 / n_classes,
        "uniform_random_expected_macro_f1": float(np.mean(uniform_f1)),
        "prior_matched_random_expected_accuracy": float((prior ** 2).sum()),
        "prior_matched_random_expected_macro_f1": float(np.mean(prior_f1)),
        "balanced_accuracy_chance": 1.0 / n_classes,
    }


def collapse_diagnostics(X, n_pairs=20000, seed=0):
    """Variance, cosine spread, effective rank, and duplicate embeddings."""
    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    rng = np.random.default_rng(seed)
    variances = X.var(axis=0)
    centered = X - X.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    p = energy / energy.sum() if energy.sum() > 0 else np.ones_like(energy) / len(energy)
    entropy_rank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    participation = float(energy.sum() ** 2 / (energy ** 2).sum()) if energy.sum() > 0 else 0.0
    unit = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    i = rng.integers(0, len(X), n_pairs)
    j = rng.integers(0, len(X), n_pairs)
    keep = i != j
    cos = (unit[i[keep]] * unit[j[keep]]).sum(axis=1)
    centered_unit = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    centered_cos = (centered_unit[i[keep]] * centered_unit[j[keep]]).sum(axis=1)
    rounded = np.round(X, 6)
    _, first = np.unique(rounded, axis=0, return_index=True)
    exact_duplicates = int(len(X) - len(first))
    report = {
        "n_rows": int(len(X)),
        "dim": int(X.shape[1]),
        "variance_min": float(variances.min()),
        "variance_median": float(np.median(variances)),
        "n_near_zero_variance_dims": int((variances < 1e-8 * max(variances.max(), 1e-30)).sum()),
        "cosine_mean": float(cos.mean()),
        "cosine_q05_q50_q95": [float(q) for q in np.quantile(cos, [0.05, 0.5, 0.95])],
        "centered_cosine_mean": float(centered_cos.mean()),
        "effective_rank_entropy": entropy_rank,
        "participation_ratio": participation,
        "top1_variance_fraction": float(p[0]) if len(p) else float("nan"),
        "exact_duplicate_rows": exact_duplicates,
    }
    collapse = []
    if report["n_near_zero_variance_dims"] > 0.5 * report["dim"]:
        collapse.append("more than half of the dimensions have ~zero variance")
    if entropy_rank < 5:
        collapse.append("effective rank below 5")
    if report["top1_variance_fraction"] > 0.9:
        collapse.append("one direction carries more than 90% of the variance")
    if exact_duplicates > 0:
        collapse.append(f"{exact_duplicates} duplicate embedding rows")
    notes = []
    if report["cosine_mean"] > 0.99:
        # A large shared mean vector is common in transformer features; the
        # centered statistics above decide whether the spread has collapsed.
        notes.append("raw embeddings are nearly parallel (mean cosine > 0.99): strong shared component")
    report["warnings"] = collapse + notes
    report["collapsed"] = bool(collapse)
    return report


def duration_features(metadata):
    duration = metadata["duration_s"].to_numpy(dtype=np.float64)
    return np.stack([duration, np.log(np.maximum(duration, 1e-3))], axis=1).astype(np.float32)
