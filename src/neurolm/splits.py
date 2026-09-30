"""Subject- and sentence-aware evaluation splits with hard disjointness checks.

``samples`` is a table with one row per subject x sentence trial and the
columns ``subject_id``, ``sentence_id``, and ``label_id``.

* ``text``    unseen sentences: stratified sentence folds; every subject is in
  train and test, but no sentence crosses train/validation/test.
* ``subject`` unseen readers: leave one subject out; the sentence pool is shared.
* ``joint``   unseen readers AND unseen sentences (primary): subject groups x
  sentence folds; train = train subjects x train sentences, validation =
  validation subjects x validation sentences, test = test subjects x test
  sentences. Iterating over every (subject group, sentence fold) cell tests
  each trial exactly once per seed.

Random trial-level splitting is deliberately not offered.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

PROTOCOLS = ("text", "subject", "joint")


@dataclass
class Split:
    protocol: str
    fold: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    subjects: dict = field(default_factory=dict)
    sentences: dict = field(default_factory=dict)

    def describe(self):
        return {
            "protocol": self.protocol,
            "fold": self.fold,
            "n_train": int(len(self.train)),
            "n_val": int(len(self.val)),
            "n_test": int(len(self.test)),
            "subjects": {k: sorted(map(str, v)) for k, v in self.subjects.items()},
            "sentences": {k: sorted(int(s) for s in v) for k, v in self.sentences.items()},
        }


def require(condition, message):
    """Hard check that survives ``python -O``."""
    if not condition:
        raise AssertionError(message)


def sentence_labels(samples):
    grouped = samples.groupby("sentence_id")["label_id"].nunique()
    require(bool((grouped == 1).all()), "a sentence carries more than one label")
    first = samples.drop_duplicates("sentence_id").set_index("sentence_id")["label_id"]
    return first.sort_index()


def _sentence_folds(labels, n_folds, seed):
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ids = labels.index.to_numpy()
    return [ids[test] for _, test in splitter.split(np.zeros(len(ids)), labels.to_numpy())]


def _val_sentences(labels, candidates, fraction, seed):
    candidates = np.asarray(sorted(candidates))
    if fraction <= 0:
        return np.asarray([], dtype=candidates.dtype)
    _, val = train_test_split(
        candidates,
        test_size=fraction,
        stratify=labels.loc[candidates].to_numpy(),
        random_state=seed,
    )
    return np.asarray(sorted(val))


def _rows(samples, subjects=None, sentences=None):
    mask = np.ones(len(samples), dtype=bool)
    if subjects is not None:
        mask &= samples["subject_id"].isin(list(subjects)).to_numpy()
    if sentences is not None:
        mask &= samples["sentence_id"].isin(list(sentences)).to_numpy()
    return np.flatnonzero(mask)


def text_holdout(samples, n_folds=5, val_fraction=0.15, seed=42):
    labels = sentence_labels(samples)
    all_sentences = set(labels.index)
    splits = []
    for fold, test_sentences in enumerate(_sentence_folds(labels, n_folds, seed), start=1):
        rest = sorted(all_sentences - set(test_sentences))
        val_sentences = _val_sentences(labels, rest, val_fraction, seed + fold)
        train_sentences = sorted(set(rest) - set(val_sentences))
        subjects = sorted(samples["subject_id"].unique())
        splits.append(Split(
            protocol="text",
            fold=f"s{fold}",
            train=_rows(samples, sentences=train_sentences),
            val=_rows(samples, sentences=val_sentences),
            test=_rows(samples, sentences=test_sentences),
            subjects={"train": subjects, "val": subjects, "test": subjects},
            sentences={"train": train_sentences, "val": list(val_sentences), "test": list(test_sentences)},
        ))
    return splits


def subject_holdout(samples, n_val_subjects=1, seed=42):
    subjects = sorted(samples["subject_id"].unique())
    order = list(np.random.default_rng(seed).permutation(subjects))
    sentences = sorted(samples["sentence_id"].unique())
    splits = []
    for position, test_subject in enumerate(order):
        others = order[position + 1:] + order[:position]
        val_subjects = others[:n_val_subjects]
        train_subjects = sorted(set(others) - set(val_subjects))
        splits.append(Split(
            protocol="subject",
            fold=f"subj-{test_subject}",
            train=_rows(samples, subjects=train_subjects),
            val=_rows(samples, subjects=val_subjects),
            test=_rows(samples, subjects=[test_subject]),
            subjects={"train": train_subjects, "val": sorted(val_subjects), "test": [test_subject]},
            sentences={"train": sentences, "val": sentences, "test": sentences},
        ))
    return splits


def joint_holdout(samples, n_subject_folds=4, n_sentence_folds=5, val_fraction=0.15, n_val_subjects=1, seed=42):
    labels = sentence_labels(samples)
    all_sentences = set(labels.index)
    subjects = sorted(samples["subject_id"].unique())
    rng = np.random.default_rng(seed)
    groups = [sorted(group) for group in np.array_split(rng.permutation(subjects), n_subject_folds)]
    sentence_folds = _sentence_folds(labels, n_sentence_folds, seed)
    splits = []
    for a, test_subjects in enumerate(groups, start=1):
        remaining = [s for s in subjects if s not in set(test_subjects)]
        for b, test_sentences in enumerate(sentence_folds, start=1):
            cell_rng = np.random.default_rng(seed * 1000 + a * 100 + b)
            val_subjects = sorted(cell_rng.choice(remaining, size=n_val_subjects, replace=False).tolist())
            train_subjects = sorted(set(remaining) - set(val_subjects))
            rest = sorted(all_sentences - set(test_sentences))
            val_sentences = _val_sentences(labels, rest, val_fraction, seed + 100 * a + b)
            train_sentences = sorted(set(rest) - set(val_sentences))
            splits.append(Split(
                protocol="joint",
                fold=f"g{a}-s{b}",
                train=_rows(samples, train_subjects, train_sentences),
                val=_rows(samples, val_subjects, val_sentences),
                test=_rows(samples, test_subjects, test_sentences),
                subjects={"train": train_subjects, "val": val_subjects, "test": list(test_subjects)},
                sentences={"train": train_sentences, "val": list(val_sentences), "test": list(test_sentences)},
            ))
    return splits


def make_splits(protocol, samples, seed, params=None):
    params = dict(params or {})
    if protocol == "text":
        splits = text_holdout(samples, seed=seed, **params)
    elif protocol == "subject":
        splits = subject_holdout(samples, seed=seed, **params)
    elif protocol == "joint":
        splits = joint_holdout(samples, seed=seed, **params)
    else:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    for split in splits:
        assert_split(split, samples)
    assert_coverage(splits, len(samples))
    return splits


def assert_split(split, samples):
    """Hard assertions on the rows actually selected, not only the ID lists."""
    train, val, test = (set(map(int, part)) for part in (split.train, split.val, split.test))
    require(len(train) > 0 and len(test) > 0, f"{split.fold}: empty train or test")
    require(not (train & test) and not (train & val) and not (val & test), f"{split.fold}: rows overlap")
    subj = {name: set(samples["subject_id"].iloc[list(part)]) for name, part in
            (("train", train), ("val", val), ("test", test))}
    sent = {name: set(samples["sentence_id"].iloc[list(part)]) for name, part in
            (("train", train), ("val", val), ("test", test))}
    if split.protocol in {"text", "joint"}:
        require(not (sent["train"] & sent["test"]), f"{split.fold}: train/test sentences overlap")
        require(not (sent["train"] & sent["val"]), f"{split.fold}: train/val sentences overlap")
        require(not (sent["val"] & sent["test"]), f"{split.fold}: val/test sentences overlap")
    if split.protocol in {"subject", "joint"}:
        require(not (subj["train"] & subj["test"]), f"{split.fold}: train/test subjects overlap")
        require(not (subj["train"] & subj["val"]), f"{split.fold}: train/val subjects overlap")
        require(not (subj["val"] & subj["test"]), f"{split.fold}: val/test subjects overlap")


def assert_coverage(splits, n_rows):
    """Every trial is tested exactly once per seed."""
    counts = np.zeros(n_rows, dtype=int)
    for split in splits:
        counts[split.test] += 1
    require(bool((counts == 1).all()), f"test coverage is not exactly once: {np.bincount(counts)}")
