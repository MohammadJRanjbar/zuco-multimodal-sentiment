import numpy as np
import pytest

from src.neurolm.splits import Split, assert_split, make_splits
from neurolm_helpers import synthetic_samples


@pytest.fixture(scope="module")
def samples():
    return synthetic_samples()


def sets(samples, rows):
    return set(samples["subject_id"].iloc[rows]), set(samples["sentence_id"].iloc[rows])


def test_text_protocol_is_sentence_disjoint(samples):
    for split in make_splits("text", samples, seed=42):
        train_subj, train_sent = sets(samples, split.train)
        test_subj, test_sent = sets(samples, split.test)
        _, val_sent = sets(samples, split.val)
        assert not train_sent & test_sent and not train_sent & val_sent and not val_sent & test_sent
        assert train_subj == test_subj  # every subject is seen


def test_subject_protocol_is_subject_disjoint(samples):
    splits = make_splits("subject", samples, seed=42)
    assert len(splits) == samples["subject_id"].nunique()
    for split in splits:
        train_subj, train_sent = sets(samples, split.train)
        test_subj, test_sent = sets(samples, split.test)
        val_subj, _ = sets(samples, split.val)
        assert not train_subj & test_subj and not train_subj & val_subj and not val_subj & test_subj
        assert len(test_subj) == 1


def test_joint_protocol_is_disjoint_in_subjects_and_sentences(samples):
    splits = make_splits("joint", samples, seed=42)
    assert len(splits) == 20
    for split in splits:
        train_subj, train_sent = sets(samples, split.train)
        val_subj, val_sent = sets(samples, split.val)
        test_subj, test_sent = sets(samples, split.test)
        assert not train_subj & test_subj and not train_sent & test_sent
        assert not train_subj & val_subj and not train_sent & val_sent
        assert not val_subj & test_subj and not val_sent & test_sent


@pytest.mark.parametrize("protocol", ["text", "subject", "joint"])
def test_every_trial_is_tested_once_and_splits_are_deterministic(samples, protocol):
    first = make_splits(protocol, samples, seed=7)
    counts = np.zeros(len(samples), int)
    for split in first:
        counts[split.test] += 1
    assert (counts == 1).all()
    second = make_splits(protocol, samples, seed=7)
    assert all(np.array_equal(a.test, b.test) for a, b in zip(first, second))
    other = make_splits(protocol, samples, seed=8)
    if protocol != "subject" or len(first) > 1:
        assert any(not np.array_equal(a.train, b.train) for a, b in zip(first, other))


def test_assertions_reject_leaky_splits(samples):
    subject_rows = np.flatnonzero(samples["subject_id"] == "S00")
    leaky = Split("subject", "bad", train=subject_rows[:10], val=np.array([], int), test=subject_rows[10:20])
    with pytest.raises(AssertionError):
        assert_split(leaky, samples)
    sentence_rows = np.flatnonzero(samples["sentence_id"] == 3)
    leaky = Split("joint", "bad", train=sentence_rows[:3], val=np.array([], int), test=sentence_rows[3:6])
    with pytest.raises(AssertionError):
        assert_split(leaky, samples)
