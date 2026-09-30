"""Split-aware probes, metrics, and cluster-level uncertainty.

Classifiers are deliberately small. Every fitted statistic (imputation,
standardization, weights, hyperparameter choice) uses the training rows or the
validation rows of the current split only; the test rows are scored once.
"""

import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .splits import make_splits

SENTIMENT_CLASSES = ["negative", "neutral", "positive"]

DEFAULT_GRIDS = {
    "logreg": [{"C": c} for c in (1e-3, 1e-2, 1e-1, 1.0)],
    "svm": [{"C": c} for c in (1e-4, 1e-3, 1e-2, 1e-1)],
    "linear": [{"weight_decay": w} for w in (1e-1, 1e-2, 1e-3)],
    "mlp": [{"weight_decay": w} for w in (1e-1, 1e-2, 1e-3)],
}


# ----------------------------------------------------------------------------
# Classifiers
# ----------------------------------------------------------------------------


def _softmax(scores):
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


class SklearnProbe:
    """Median imputation + standardization + a linear sklearn model."""

    def __init__(self, kind, params, seed, n_classes):
        self.kind = kind
        self.n_classes = n_classes
        if kind == "logreg":
            model = LogisticRegression(
                C=params["C"], max_iter=3000, class_weight="balanced", random_state=seed
            )
        elif kind == "svm":
            model = LinearSVC(
                C=params["C"], class_weight="balanced", max_iter=20000, random_state=seed
            )
        else:
            raise ValueError(kind)
        self.pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)

    def fit(self, X, y, X_val=None, y_val=None):
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.classes_
        return self

    def predict_proba(self, X):
        if self.kind == "logreg":
            proba = self.pipeline.predict_proba(X)
        else:
            # LinearSVC has no probabilities; a softmax over its decision
            # scores gives an uncalibrated ranking used only for the CSV.
            scores = self.pipeline.decision_function(X)
            if scores.ndim == 1:
                scores = np.stack([-scores, scores], axis=1)
            proba = _softmax(scores)
        full = np.zeros((len(X), self.n_classes))
        full[:, self.classes_] = proba
        return full


class TorchProbe:
    """Linear layer or one-hidden-layer MLP with validation early stopping."""

    def __init__(self, kind, params, seed, n_classes, hidden=256, dropout=0.5,
                 max_epochs=300, min_epochs=30, patience=30, lr=3e-3, batch_size=128):
        self.kind = kind
        self.params = params
        self.seed = seed
        self.n_classes = n_classes
        self.hidden = hidden
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.patience = patience
        self.lr = lr
        self.batch_size = batch_size
        self.prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    def _net(self, dim):
        import torch.nn as nn

        if self.kind == "linear":
            return nn.Linear(dim, self.n_classes)
        return nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.n_classes),
        )

    def fit(self, X, y, X_val, y_val):
        import torch

        torch.manual_seed(self.seed)
        Xt = torch.from_numpy(self.prep.fit_transform(X).astype(np.float32))
        Xv = torch.from_numpy(self.prep.transform(X_val).astype(np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        weights = torch.tensor(
            np.where(counts > 0, counts.sum() / (self.n_classes * np.maximum(counts, 1)), 0.0),
            dtype=torch.float32,
        )
        net = self._net(Xt.shape[1])
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.params["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        generator = torch.Generator().manual_seed(self.seed)
        best_score, best_state, stale = -1.0, None, 0
        for epoch in range(self.max_epochs):
            net.train()
            order = torch.randperm(len(Xt), generator=generator)
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                optimizer.zero_grad()
                loss_fn(net(Xt[batch]), yt[batch]).backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                predictions = net(Xv).argmax(dim=1).numpy()
            score = f1_score(y_val, predictions, average="macro", labels=list(range(self.n_classes)), zero_division=0)
            if score > best_score + 1e-9:
                best_score, stale = score, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience and epoch + 1 >= self.min_epochs:
                    break
        net.load_state_dict(best_state)
        self.net = net.eval()
        return self

    def predict_proba(self, X):
        import torch

        with torch.no_grad():
            logits = self.net(torch.from_numpy(self.prep.transform(X).astype(np.float32)))
        return torch.softmax(logits, dim=1).numpy()


def make_probe(kind, params, seed, n_classes):
    if kind in {"logreg", "svm"}:
        return SklearnProbe(kind, params, seed, n_classes)
    if kind in {"linear", "mlp"}:
        return TorchProbe(kind, params, seed, n_classes)
    raise ValueError(f"unknown classifier {kind!r}")


def select_and_fit(kind, X_train, y_train, X_val, y_val, seed, n_classes, grid=None):
    """Pick hyperparameters by validation macro-F1, return the fitted model."""
    grid = grid or DEFAULT_GRIDS[kind]
    best = None
    scores = []
    for params in grid:
        model = make_probe(kind, params, seed, n_classes).fit(X_train, y_train, X_val, y_val)
        predictions = model.predict_proba(X_val).argmax(axis=1)
        score = f1_score(y_val, predictions, average="macro", labels=list(range(n_classes)), zero_division=0)
        scores.append({"params": params, "val_macro_f1": float(score)})
        if best is None or score > best[0] + 1e-9:
            best = (score, params, model)
    return best[2], {"selected": best[1], "val_macro_f1": float(best[0]), "grid": scores}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------


def compute_metrics(y_true, y_pred, class_names=SENTIMENT_CLASSES):
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": list(class_names),
        "n_predicted_classes": int(len(np.unique(y_pred))),
    }


def aggregate_seeds(seed_metrics):
    keys = ["macro_f1", "accuracy", "balanced_accuracy", "weighted_f1"]
    out = {}
    for key in keys:
        values = np.array([m[key] for m in seed_metrics])
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "values": values.tolist()}
    names = seed_metrics[0]["class_names"]
    out["per_class_f1_mean"] = {
        name: float(np.mean([m["per_class"][name]["f1"] for m in seed_metrics])) for name in names
    }
    out["per_class_recall_mean"] = {
        name: float(np.mean([m["per_class"][name]["recall"] for m in seed_metrics])) for name in names
    }
    out["per_class_precision_mean"] = {
        name: float(np.mean([m["per_class"][name]["precision"] for m in seed_metrics])) for name in names
    }
    out["confusion_matrix_sum"] = np.sum([m["confusion_matrix"] for m in seed_metrics], axis=0).tolist()
    return out


# ----------------------------------------------------------------------------
# Protocol runner
# ----------------------------------------------------------------------------


def run_probe(
    X,
    samples,
    *,
    protocol,
    classifier,
    seeds,
    protocol_params=None,
    target="label_id",
    class_names=SENTIMENT_CLASSES,
    grid=None,
    feature_transform=None,
    keep_predictions=True,
):
    """Evaluate one feature matrix under one protocol for several seeds.

    ``feature_transform(X, samples, split, rng)`` may return a modified copy of
    X for a split (used by the shuffled-association control).
    """
    samples = samples.reset_index(drop=True)
    y_all = samples[target].to_numpy().astype(int)
    n_classes = len(class_names)
    per_seed = []
    for seed in seeds:
        start = time.time()
        splits = make_splits(protocol, samples, seed, protocol_params)
        proba = np.full((len(samples), n_classes), np.nan)
        fold_log = []
        for index, split in enumerate(splits):
            rng = np.random.default_rng(seed * 7919 + index)
            X_split = feature_transform(X, samples, split, rng) if feature_transform else X
            model, selection = select_and_fit(
                classifier,
                X_split[split.train], y_all[split.train],
                X_split[split.val], y_all[split.val],
                seed, n_classes, grid,
            )
            proba[split.test] = model.predict_proba(X_split[split.test])
            test_metrics = compute_metrics(y_all[split.test], proba[split.test].argmax(axis=1), class_names)
            fold_log.append({**split.describe(), "selection": selection,
                             "test_macro_f1": test_metrics["macro_f1"],
                             "test_accuracy": test_metrics["accuracy"]})
        if np.isnan(proba).any():
            raise RuntimeError("some trials were never tested")
        predicted = proba.argmax(axis=1)
        record = {
            "seed": int(seed),
            "metrics": compute_metrics(y_all, predicted, class_names),
            "folds": fold_log,
            "runtime_s": float(time.time() - start),
        }
        if keep_predictions:
            frame = pd.DataFrame({
                "sample_id": samples["sample_id"],
                "subject_id": samples["subject_id"],
                "sentence_id": samples["sentence_id"],
                "true_label": [class_names[i] for i in y_all],
                "predicted_label": [class_names[i] for i in predicted],
                "true_id": y_all,
                "predicted_id": predicted,
                "seed": seed,
            })
            for i, name in enumerate(class_names):
                frame[f"prob_{name}"] = proba[:, i]
            record["predictions"] = frame
        per_seed.append(record)
    return {
        "protocol": protocol,
        "classifier": classifier,
        "seeds": [int(s) for s in seeds],
        "per_seed": per_seed,
        "summary": aggregate_seeds([r["metrics"] for r in per_seed]),
    }


# ----------------------------------------------------------------------------
# Cluster bootstrap and paired tests
# ----------------------------------------------------------------------------


def _cluster_confusions(predictions, cluster, n_classes, true_col="true_id", pred_col="predicted_id"):
    """``[seeds, clusters, K, K]`` confusion counts and the cluster labels."""
    seeds = sorted(predictions["seed"].unique())
    clusters = np.array(sorted(predictions[cluster].unique()))
    lookup = {c: i for i, c in enumerate(clusters)}
    counts = np.zeros((len(seeds), len(clusters), n_classes, n_classes))
    for s_index, seed in enumerate(seeds):
        part = predictions[predictions["seed"] == seed]
        c_index = part[cluster].map(lookup).to_numpy()
        np.add.at(counts, (s_index, c_index, part[true_col].to_numpy(), part[pred_col].to_numpy()), 1)
    return counts, clusters


def _macro_f1_from_confusion(conf):
    """Macro-F1 over the last two axes (true x predicted), classes with no
    support and no predictions scoring 0 as sklearn does with fixed labels."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    fp = conf.sum(axis=-2) - tp
    fn = conf.sum(axis=-1) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    return f1.mean(axis=-1)


def _accuracy_from_confusion(conf):
    return np.diagonal(conf, axis1=-2, axis2=-1).sum(axis=-1) / conf.sum(axis=(-2, -1))


def cluster_bootstrap(predictions, cluster="sentence_id", n_classes=3, n_boot=2000, seed=0):
    """Seed-averaged macro-F1/accuracy with a cluster-bootstrap 95% interval.

    Clusters (sentences or subjects) are resampled with replacement; all trials
    and all seeds belonging to a drawn cluster enter together, so repeated
    measurements of one sentence are not treated as independent.
    """
    counts, clusters = _cluster_confusions(predictions, cluster, n_classes)
    rng = np.random.default_rng(seed)
    observed_conf = counts.sum(axis=1)
    observed = {
        "macro_f1": float(_macro_f1_from_confusion(observed_conf).mean()),
        "accuracy": float(_accuracy_from_confusion(observed_conf).mean()),
    }
    draws = {"macro_f1": [], "accuracy": []}
    for _ in range(n_boot):
        weights = np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters))
        conf = np.einsum("c,sckl->skl", weights, counts)
        draws["macro_f1"].append(_macro_f1_from_confusion(conf).mean())
        draws["accuracy"].append(_accuracy_from_confusion(conf).mean())
    out = {"cluster": cluster, "n_clusters": int(len(clusters)), "n_boot": n_boot}
    for key, values in draws.items():
        low, high = np.percentile(values, [2.5, 97.5])
        out[key] = {"estimate": observed[key], "ci95": [float(low), float(high)]}
    return out


def paired_comparison(baseline, candidate, cluster="sentence_id", n_classes=3, n_boot=2000, n_perm=5000, seed=0):
    """Candidate minus baseline macro-F1 on identical trials and seeds.

    Returns a cluster-bootstrap 95% interval and a paired cluster permutation
    p-value (each cluster's predictions are swapped between systems with
    probability 1/2, preserving within-cluster dependence).
    """
    keys = ["sample_id", "seed"]
    merged = baseline[keys + [cluster, "true_id", "predicted_id"]].merge(
        candidate[keys + ["predicted_id"]], on=keys, suffixes=("_a", "_b"), validate="one_to_one"
    )
    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise ValueError("paired comparison needs predictions for the same trials and seeds")
    counts_a, clusters = _cluster_confusions(merged, cluster, n_classes, pred_col="predicted_id_a")
    counts_b, _ = _cluster_confusions(merged, cluster, n_classes, pred_col="predicted_id_b")

    def delta(conf_a, conf_b):
        return float(_macro_f1_from_confusion(conf_b).mean() - _macro_f1_from_confusion(conf_a).mean())

    observed = delta(counts_a.sum(axis=1), counts_b.sum(axis=1))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters))
        boot.append(delta(np.einsum("c,sckl->skl", w, counts_a), np.einsum("c,sckl->skl", w, counts_b)))
    null = []
    for _ in range(n_perm):
        flip = rng.random(len(clusters)) < 0.5
        a = np.where(flip[None, :, None, None], counts_b, counts_a).sum(axis=1)
        b = np.where(flip[None, :, None, None], counts_a, counts_b).sum(axis=1)
        null.append(delta(a, b))
    null = np.abs(np.asarray(null))
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "delta_macro_f1": observed,
        "ci95": [float(low), float(high)],
        "p_value_two_sided": float((1 + np.sum(null >= abs(observed) - 1e-12)) / (1 + n_perm)),
        "cluster": cluster,
        "n_clusters": int(len(clusters)),
        "n_trials": int(len(merged) // max(1, merged["seed"].nunique())),
    }


def sentence_aggregate(predictions, class_names=SENTIMENT_CLASSES):
    """Secondary analysis: average held-out probabilities over subjects per sentence.

    Every probability is still an out-of-fold prediction for its own trial, so
    aggregation adds no training information; it only asks whether pooling
    readers makes the sentence-level decision more reliable.
    """
    columns = [f"prob_{name}" for name in class_names]
    per_seed = []
    for _, part in predictions.groupby("seed"):
        grouped = part.groupby("sentence_id")
        mean = grouped[columns].mean()
        truth = grouped["true_id"].first().loc[mean.index].to_numpy()
        per_seed.append(compute_metrics(truth, mean.to_numpy().argmax(axis=1), class_names))
    return aggregate_seeds(per_seed)


def predictions_frame(result):
    return pd.concat([r["predictions"] for r in result["per_seed"]], ignore_index=True)
