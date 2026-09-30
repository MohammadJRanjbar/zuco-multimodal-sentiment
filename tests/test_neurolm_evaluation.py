import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.neurolm.evaluation import (
    _macro_f1_from_confusion, cluster_bootstrap, compute_metrics, make_probe, paired_comparison,
    predictions_frame, run_probe,
)
from tests.neurolm_helpers import signal_features, synthetic_samples


@pytest.fixture(scope="module")
def samples():
    return synthetic_samples(n_subjects=8, n_sentences=45)


@pytest.mark.parametrize("kind,params", [
    ("logreg", {"C": 1.0}), ("svm", {"C": 0.1}), ("linear", {"weight_decay": 1e-2}), ("mlp", {"weight_decay": 1e-2}),
])
def test_classifier_forward_pass(kind, params):
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(3), 100)
    X = rng.standard_normal((300, 10)).astype(np.float32)
    X[np.arange(300), y] += 3.0
    X[5, 4] = np.nan  # imputation path
    model = make_probe(kind, params, seed=0, n_classes=3).fit(X[:200], y[:200], X[200:250], y[200:250])
    proba = model.predict_proba(X[250:])
    assert proba.shape == (50, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba.argmax(axis=1) == y[250:]).mean() > 0.85


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 3, 200), rng.integers(0, 3, 200)
    metrics = compute_metrics(y, p)
    assert metrics["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    conf = np.array(metrics["confusion_matrix"])
    assert _macro_f1_from_confusion(conf) == pytest.approx(metrics["macro_f1"])
    assert set(metrics["per_class"]) == {"negative", "neutral", "positive"}
    assert {"precision", "recall", "f1", "support"} <= set(metrics["per_class"]["neutral"])


def test_run_probe_predictions_and_signal(samples):
    X = signal_features(samples, strength=1.5)
    result = run_probe(X, samples, protocol="joint", classifier="logreg", seeds=[1, 2])
    assert result["summary"]["macro_f1"]["mean"] > 0.7
    frame = predictions_frame(result)
    expected = {"sample_id", "subject_id", "sentence_id", "true_label", "predicted_label",
                "prob_positive", "prob_neutral", "prob_negative"}
    assert expected <= set(frame.columns)
    assert frame.groupby("seed")["sample_id"].nunique().eq(len(samples)).all()
    fold = result["per_seed"][0]["folds"][0]
    assert not set(fold["subjects"]["train"]) & set(fold["subjects"]["test"])
    assert not set(fold["sentences"]["train"]) & set(fold["sentences"]["test"])


def test_cluster_bootstrap_and_paired_comparison(samples):
    strong = run_probe(signal_features(samples, strength=1.5), samples, protocol="text",
                       classifier="logreg", seeds=[1, 2])
    weak = run_probe(signal_features(samples, strength=0.0, seed=3), samples, protocol="text",
                     classifier="logreg", seeds=[1, 2])
    strong_frame, weak_frame = predictions_frame(strong), predictions_frame(weak)
    boot = cluster_bootstrap(strong_frame, n_boot=200)
    assert boot["macro_f1"]["estimate"] == pytest.approx(strong["summary"]["macro_f1"]["mean"])
    low, high = boot["macro_f1"]["ci95"]
    assert low <= boot["macro_f1"]["estimate"] <= high
    same = paired_comparison(strong_frame, strong_frame, n_boot=100, n_perm=200)
    assert same["delta_macro_f1"] == 0 and same["p_value_two_sided"] == 1.0
    better = paired_comparison(weak_frame, strong_frame, n_boot=200, n_perm=500)
    assert better["delta_macro_f1"] > 0.2 and better["ci95"][0] > 0 and better["p_value_two_sided"] < 0.01
