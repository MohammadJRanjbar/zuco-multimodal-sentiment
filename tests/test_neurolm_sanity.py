import numpy as np
import pytest

from src.neurolm import sanity
from src.neurolm.evaluation import run_probe
from src.neurolm.splits import make_splits
from tests.neurolm_helpers import signal_features, synthetic_samples


@pytest.fixture(scope="module")
def samples():
    return synthetic_samples(n_subjects=8, n_sentences=45)


def test_sentence_level_permutation_keeps_sentences_consistent(samples):
    permuted = sanity.permute_labels(samples, np.random.default_rng(0), "sentence")
    assert (permuted.groupby("sentence_id")["label_id"].nunique() == 1).all()
    before = samples.drop_duplicates("sentence_id")["label_id"].value_counts().sort_index()
    after = permuted.drop_duplicates("sentence_id")["label_id"].value_counts().sort_index()
    assert before.equals(after)
    assert (permuted["label_id"] != samples["label_id"]).any()


def test_trial_level_permutation_preserves_counts(samples):
    permuted = sanity.permute_labels(samples, np.random.default_rng(0), "trial")
    assert sorted(permuted["label_id"]) == sorted(samples["label_id"])


def test_shuffled_association_stays_inside_subject_and_partition(samples):
    X = np.arange(len(samples), dtype=float)[:, None]
    split = make_splits("joint", samples, seed=1)[0]
    shuffled = sanity.shuffle_sentence_association(X, samples, split, np.random.default_rng(0))
    subjects = samples["subject_id"].to_numpy()
    for part in (split.train, split.val, split.test):
        for subject in np.unique(subjects[part]):
            rows = part[subjects[part] == subject]
            assert sorted(shuffled[rows, 0]) == sorted(X[rows, 0])
    untouched = np.setdiff1d(np.arange(len(samples)), np.concatenate([split.train, split.val, split.test]))
    np.testing.assert_array_equal(shuffled[untouched], X[untouched])


def test_shuffled_labels_and_random_features_fall_to_chance(samples):
    X = signal_features(samples, strength=1.5)
    real = run_probe(X, samples, protocol="joint", classifier="logreg", seeds=[1], keep_predictions=False)
    assert real["summary"]["macro_f1"]["mean"] > 0.7
    null = sanity.permutation_null(X, samples, protocol="joint", classifier="logreg", seeds=[1, 2],
                                   n_permutations=4, observed=real["summary"]["macro_f1"]["mean"])
    assert null["null_mean"] < 0.45 and null["p_value"] == pytest.approx(1 / 5)
    random_X = sanity.gaussian_features(len(samples), X.shape[1], seed=0)
    assert random_X.shape == X.shape and random_X.dtype == np.float32
    rand = run_probe(random_X, samples, protocol="joint", classifier="logreg", seeds=[1], keep_predictions=False)
    assert rand["summary"]["macro_f1"]["mean"] < 0.45


def test_label_distribution_baselines():
    labels = np.array([0] * 50 + [1] * 30 + [2] * 20)
    out = sanity.label_distribution_baselines(labels, n_simulations=200)
    assert out["majority_accuracy"] == 0.5 and out["majority_class"] == 0
    assert out["majority_macro_f1"] == pytest.approx((2 * 50 / 150) / 3)
    assert out["prior_matched_random_expected_accuracy"] == pytest.approx(0.25 + 0.09 + 0.04)
    assert 0.25 < out["uniform_random_expected_macro_f1"] < 0.42


def test_collapse_diagnostics():
    rng = np.random.default_rng(0)
    healthy = sanity.collapse_diagnostics(rng.standard_normal((300, 32)))
    assert not healthy["collapsed"] and healthy["effective_rank_entropy"] > 20
    direction = rng.standard_normal(32)
    collapsed = sanity.collapse_diagnostics(np.outer(rng.standard_normal(300), direction) + 1e-6 * rng.standard_normal((300, 32)))
    assert collapsed["collapsed"]
    duplicates = sanity.collapse_diagnostics(np.vstack([rng.standard_normal((50, 8))] * 2))
    assert duplicates["exact_duplicate_rows"] == 50
