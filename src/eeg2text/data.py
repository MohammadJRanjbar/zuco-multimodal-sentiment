"""Trials for EEG-to-text: one reader's word-level EEG sequence for one sentence -> the sentence.

Every trial keeps the sentence's full word sequence; words the reader did not
fixate have no EEG and get a learned "missing" input. Input conditions share
the sentence length and the fixation pattern and differ only in what sits at
the fixated positions:

* ``eeg``: the reader's word EEG, z-scored per reader on training sentences;
* ``shuffled_eeg``: EEG vectors of random fixated words of the same reader's
  training sentences (pairing with the text destroyed);
* ``noise``: standard Gaussian vectors;
* ``word_vectors``: the word's identity vector from LaBSE (positive control:
  the input contains the text, so a working pipeline must decode it; like EEG,
  it needs a learned mapping into mBART);
* ``mbart_vectors``: the word's own mBART token embeddings (easiest positive
  control: checks that the generator uses its input at all).

Sentences are split into train / validation / test so that no test sentence
is ever seen in training, by any reader. Recordings of other ZuCo tasks (NR,
TSR) can be added as extra training data; any of their sentences that is in
validation or test is dropped.
"""

from dataclasses import dataclass, field

import numpy as np

from ..config import LABEL_TO_ID, ZUCO_REFERENCE_CHANNEL_INDEX

INPUTS = ("eeg", "shuffled_eeg", "noise", "word_vectors", "mbart_vectors")
VECTOR_INPUTS = ("word_vectors", "mbart_vectors")


@dataclass
class Corpus:
    lang: str
    features: list            # per trial [n_words, F] float32, NaN rows = not fixated
    words: list               # per trial list of words
    sentence_id: np.ndarray
    reader: np.ndarray
    label: np.ndarray         # 0 negative, 1 neutral, 2 positive, -1 unlabelled
    part: np.ndarray = None   # "train" / "val" / "test" per trial
    static: dict = field(default_factory=dict)  # input name -> {word: vector} (positive controls)
    session: np.ndarray = None  # recording per trial ("SR"; extra training data e.g. "NR", "TSR")

    @property
    def texts(self):
        return [" ".join(w) for w in self.words]

    @property
    def dim(self):
        return int(self.features[0].shape[1])


def _from_trials(lang, trials, flatten, session=""):
    features, words, sentence_id, reader, label = [], [], [], [], []
    for trial in trials:
        f = flatten(np.asarray(trial["features"], dtype=np.float32))
        features.append(f)
        words.append([str(w) for w in trial["words"]])
        sentence_id.append(int(trial["sentence_id"]))
        reader.append(str(trial["subject_id"]))
        label.append(LABEL_TO_ID.get(int(trial["label"]), -1))
    finite = np.concatenate([f[np.isfinite(f).all(axis=1)] for f in features])
    generic = np.asarray(trials[0]["features"]).ndim == 2 if trials else False
    if len(finite) and not generic and (finite > 0).all():  # band power -> log scale
        features = [np.where(np.isfinite(f), np.log(np.where(f > 0, f, 1.0)), np.nan).astype(np.float32)
                    for f in features]
    return Corpus(lang, features, words, np.array(sentence_id), np.array(reader), np.array(label),
                  session=np.full(len(words), session))


def load_zuco(word_eeg_dir, session="SR"):
    from ..fusion.word_eeg import load_word_eeg

    def flatten(block):  # [n, bands, 105] -> [n, bands * 104] without the flat Cz reference; [n, F] as is
        if block.ndim == 2:
            return block
        keep = [c for c in range(block.shape[2]) if c != ZUCO_REFERENCE_CHANNEL_INDEX]
        return block[:, :, keep].reshape(len(block), -1)

    return _from_trials("en", load_word_eeg(word_eeg_dir), flatten, session)


def load_teco(trt_dir, labels_csv):
    from ..brainshaping.teco import load_teco_trials

    return _from_trials("fa", load_teco_trials(trt_dir, labels_csv), lambda block: block.reshape(len(block), -1),
                        "TeCo")


def add_training_data(corpus, extra):
    """Append ``extra`` trials (another recording of the same language) as training data.

    Call after ``assign_parts`` and before ``normalize``. Trials whose sentence is in the validation or
    test part are dropped, so held-out sentences stay unseen. Returns the number of dropped trials.
    """
    from ..labels import normalize_text

    if extra.dim != corpus.dim:
        raise ValueError(f"extra data has {extra.dim} features per word, the corpus {corpus.dim}")
    held_out = {normalize_text(" ".join(w)) for w, part in zip(corpus.words, corpus.part) if part != "train"}
    keep = [i for i, w in enumerate(extra.words) if normalize_text(" ".join(w)) not in held_out]
    corpus.features = corpus.features + [extra.features[i] for i in keep]
    corpus.words = corpus.words + [extra.words[i] for i in keep]
    for name in ("sentence_id", "reader", "label", "session"):
        setattr(corpus, name, np.concatenate([getattr(corpus, name), getattr(extra, name)[keep]]))
    corpus.part = np.concatenate([corpus.part, np.full(len(keep), "train")])
    return len(extra.words) - len(keep)


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
    """Z-score each reader's word EEG (per recording) with that reader's fixated training words."""
    session = corpus.session if corpus.session is not None else np.full(len(corpus.words), "")
    group = np.char.add(np.char.add(corpus.reader.astype(str), "|"), session.astype(str))
    out = []
    for r in np.unique(group):
        rows = [i for i in np.flatnonzero(group == r) if corpus.part[i] == "train"]
        pool = np.concatenate([corpus.features[i] for i in rows]) if rows else np.zeros((0, corpus.dim))
        pool = pool[np.isfinite(pool).all(axis=1)]
        mean = pool.mean(axis=0) if len(pool) else np.zeros(corpus.dim)
        std = pool.std(axis=0) if len(pool) else np.ones(corpus.dim)
        std[std < 1e-8] = 1.0
        for i in np.flatnonzero(group == r):
            out.append((i, ((corpus.features[i] - mean) / std).astype(np.float32)))
    for i, f in out:
        corpus.features[i] = f
    return corpus


def build_inputs(corpus, condition, seed=0):
    """Per trial ``(x [n_words, F], fixated [n_words])`` for one input condition."""
    rng = np.random.default_rng(seed)
    fixated = [np.isfinite(f).all(axis=1) for f in corpus.features]
    if condition in VECTOR_INPUTS:
        table = corpus.static[condition]
        dim = len(next(iter(table.values())))
        xs = [np.stack([table.get(w, np.zeros(dim)) for w in words]).astype(np.float32)
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


def vectors_by_word(corpus, name):
    """Per trial [n_words, D] vectors of input ``name`` (NaN rows for words without one)."""
    table = corpus.static[name]
    dim = len(next(iter(table.values())))
    missing = np.full(dim, np.nan, np.float32)
    return [np.stack([table.get(w, missing) for w in words]).astype(np.float32) for words in corpus.words]


def unique_words(corpus):
    return sorted({w for words in corpus.words for w in words})


def attach_vectors(corpus, name, words, vectors, found, standardize=True):
    """Store word vectors as ``name`` (z-scored per dimension unless ``standardize`` is False)."""
    vectors, found = np.asarray(vectors, dtype=np.float64), np.asarray(found, dtype=bool)
    mean, std = vectors[found].mean(axis=0), vectors[found].std(axis=0)
    std[std < 1e-8] = 1.0
    if not standardize:
        mean, std = np.zeros_like(mean), np.ones_like(std)
    corpus.static[name] = {w: ((v - mean) / std).astype(np.float32) for w, v, ok in zip(words, vectors, found) if ok}
    return corpus


def attach_static_vectors(corpus, model_name):
    """Word-identity vectors (input embeddings of a multilingual encoder) for the ``word_vectors`` control."""
    from ..brainshaping.encoding import static_word_vectors

    unique = unique_words(corpus)
    vectors, found = static_word_vectors([unique], np.zeros(len(unique), int), np.arange(len(unique)), model_name)
    return attach_vectors(corpus, "word_vectors", unique, vectors, found)
