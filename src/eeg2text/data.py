"""Trials for EEG-to-text: one reader's word-level EEG sequence for one sentence -> the sentence.

Every trial keeps the sentence's full word sequence; words the reader did not
fixate have no EEG and get a learned "missing" input. Input conditions share
the sentence length and the fixation pattern and differ only in what sits at
the fixated positions:

* ``eeg``: the reader's word EEG, z-scored per reader on training sentences;
* ``shuffled_eeg``: EEG vectors of random fixated words of the same reader's
  training sentences (pairing with the text destroyed);
* ``noise``: standard Gaussian vectors;
* ``word_vectors``: the word's identity vector (positive control: the input
  contains the text, so a working pipeline must decode it).

Sentences are split into train / validation / test so that no test sentence
is ever seen in training, by any reader.
"""

from dataclasses import dataclass, field

import numpy as np

from ..config import LABEL_TO_ID, ZUCO_REFERENCE_CHANNEL_INDEX

INPUTS = ("eeg", "shuffled_eeg", "noise", "word_vectors")


@dataclass
class Corpus:
    lang: str
    features: list            # per trial [n_words, F] float32, NaN rows = not fixated
    words: list               # per trial list of words
    sentence_id: np.ndarray
    reader: np.ndarray
    label: np.ndarray         # 0 negative, 1 neutral, 2 positive
    part: np.ndarray = None   # "train" / "val" / "test" per trial
    static: dict = field(default_factory=dict)  # word -> identity vector (positive control)

    @property
    def texts(self):
        return [" ".join(w) for w in self.words]

    @property
    def dim(self):
        return int(self.features[0].shape[1])


def _from_trials(lang, trials, flatten):
    features, words, sentence_id, reader, label = [], [], [], [], []
    for trial in trials:
        f = flatten(np.asarray(trial["features"], dtype=np.float32))
        features.append(f)
        words.append([str(w) for w in trial["words"]])
        sentence_id.append(int(trial["sentence_id"]))
        reader.append(str(trial["subject_id"]))
        label.append(LABEL_TO_ID[int(trial["label"])])
    finite = np.concatenate([f[np.isfinite(f).all(axis=1)] for f in features])
    if len(finite) and (finite > 0).all():  # band power -> log scale
        features = [np.where(np.isfinite(f), np.log(np.where(f > 0, f, 1.0)), np.nan).astype(np.float32)
                    for f in features]
    return Corpus(lang, features, words, np.array(sentence_id), np.array(reader), np.array(label))


def load_zuco(word_eeg_dir):
    from ..fusion.word_eeg import load_word_eeg

    def flatten(block):  # [n, bands, 105] -> [n, bands * 104] without the flat Cz reference
        keep = [c for c in range(block.shape[2]) if c != ZUCO_REFERENCE_CHANNEL_INDEX]
        return block[:, :, keep].reshape(len(block), -1)

    return _from_trials("en", load_word_eeg(word_eeg_dir), flatten)


def load_teco(trt_dir, labels_csv):
    from ..brainshaping.teco import load_teco_trials

    return _from_trials("fa", load_teco_trials(trt_dir, labels_csv), lambda block: block.reshape(len(block), -1))


def assign_parts(corpus, seed=0, test=0.2, val=0.1, fold=None, n_folds=5):
    """Sentence-disjoint train / val / test (``fold`` rotates the test fifth)."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.unique(corpus.sentence_id))
    if fold is None:
        n_test = max(1, int(round(test * len(ids))))
        test_ids = set(ids[:n_test])
        rest = ids[n_test:]
    else:
        chunks = np.array_split(ids, n_folds)
        test_ids = set(chunks[fold])
        rest = np.concatenate([c for i, c in enumerate(chunks) if i != fold])
    n_val = max(1, int(round(val * len(ids))))
    val_ids = set(rest[:n_val])
    corpus.part = np.array(["test" if s in test_ids else "val" if s in val_ids else "train"
                            for s in corpus.sentence_id])
    return corpus


def normalize(corpus):
    """Z-score each reader's word EEG with that reader's fixated training words."""
    out = []
    for r in np.unique(corpus.reader):
        rows = [i for i in np.flatnonzero(corpus.reader == r) if corpus.part[i] == "train"]
        pool = np.concatenate([corpus.features[i] for i in rows]) if rows else np.zeros((0, corpus.dim))
        pool = pool[np.isfinite(pool).all(axis=1)]
        mean = pool.mean(axis=0) if len(pool) else np.zeros(corpus.dim)
        std = pool.std(axis=0) if len(pool) else np.ones(corpus.dim)
        std[std < 1e-8] = 1.0
        for i in np.flatnonzero(corpus.reader == r):
            out.append((i, ((corpus.features[i] - mean) / std).astype(np.float32)))
    for i, f in out:
        corpus.features[i] = f
    return corpus


def build_inputs(corpus, condition, seed=0):
    """Per trial ``(x [n_words, F], fixated [n_words])`` for one input condition."""
    rng = np.random.default_rng(seed)
    fixated = [np.isfinite(f).all(axis=1) for f in corpus.features]
    if condition == "word_vectors":
        dim = len(next(iter(corpus.static.values())))
        xs = [np.stack([corpus.static.get(w, np.zeros(dim)) for w in words]).astype(np.float32)
              for words in corpus.words]
        return [(np.where(m[:, None], x, 0.0).astype(np.float32), m) for x, m in zip(xs, fixated)]
    pools = {}
    if condition == "shuffled_eeg":
        for r in np.unique(corpus.reader):
            rows = [i for i in np.flatnonzero(corpus.reader == r) if corpus.part[i] == "train"]
            pools[r] = np.concatenate([corpus.features[i][fixated[i]] for i in rows])
    out = []
    for i, (f, m) in enumerate(zip(corpus.features, fixated)):
        x = np.zeros_like(f, dtype=np.float32)
        if condition == "eeg":
            x[m] = f[m]
        elif condition == "shuffled_eeg":
            pool = pools[corpus.reader[i]]
            x[m] = pool[rng.integers(0, len(pool), size=int(m.sum()))]
        elif condition == "noise":
            x[m] = rng.standard_normal((int(m.sum()), f.shape[1]))
        else:
            raise ValueError(f"unknown input condition {condition!r}")
        out.append((x, m))
    return out


def attach_static_vectors(corpus, model_name):
    """Word-identity vectors (input embeddings of a multilingual encoder) for the positive control."""
    from ..brainshaping.encoding import static_word_vectors

    unique = sorted({w for words in corpus.words for w in words})
    vectors, found = static_word_vectors([unique], np.zeros(len(unique), int), np.arange(len(unique)), model_name)
    mean, std = vectors[found].mean(axis=0), vectors[found].std(axis=0)
    std[std < 1e-8] = 1.0
    corpus.static = {w: ((v - mean) / std).astype(np.float32) for w, v, ok in zip(unique, vectors, found) if ok}
    return corpus
