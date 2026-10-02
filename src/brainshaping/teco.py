"""TeCo (Persian) word-level EEG in the same trial format as the ZuCo cache.

Reads the ``<Name>_trt_total.pickle`` files (one per participant) used by the
reproduce-teco-results repository: a dict of trials, each with ``sentenceId``
(1-based), ``persian_sentence`` and ``word`` (word index -> dict with a
``TRT_total`` vector; un-fixated words hold a short stub). Labels come from
``teco_sentiment_labels_task1.csv`` (``sentence``, ``sentiment_label`` in
{-1, 0, 1}), matched by sentence text as in that repository.

Each trial gets ``features`` of shape ``[n_words, 1, n_values]`` (NaN for
un-fixated words) so the ZuCo helpers (per-reader z-scoring, reader averaging)
apply unchanged.
"""

import glob
import os
import pickle
import re

import numpy as np

WORD_TEXT_KEYS = ("content", "word", "text", "persian_word", "token")


def normalize_sentence(text):
    """Key used to match label rows to pickles: no whitespace, no zero-width non-joiner."""
    return re.sub(r"\s+", "", str(text)).replace("‌", "")


def load_labels(labels_csv):
    import pandas as pd

    table = pd.read_csv(labels_csv)
    return dict(zip(table["sentence"].map(normalize_sentence), table["sentiment_label"].astype(int)))


def _word_texts(sentence, word_dicts):
    for key in WORD_TEXT_KEYS:
        values = [w.get(key) if isinstance(w, dict) else None for w in word_dicts]
        if all(isinstance(v, str) and v.strip() for v in values):
            return [v.strip() for v in values]
    split = str(sentence.get("persian_sentence", "")).split()
    if len(split) == len(word_dicts):
        return split
    first = word_dicts[0] if word_dicts else {}
    raise ValueError(
        f"cannot recover word text for sentenceId {sentence.get('sentenceId')}: {len(word_dicts)} words with EEG "
        f"but {len(split)} whitespace tokens in persian_sentence, and no text key among {WORD_TEXT_KEYS} "
        f"(word keys: {sorted(first) if isinstance(first, dict) else type(first).__name__})")


def describe_pickle(path, max_words=3):
    """Short structural summary of one participant pickle (for checking the format in Colab)."""
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    key = next(iter(data))
    trial = data[key]
    lines = [f"{os.path.basename(path)}: {type(data).__name__} with {len(data)} trials; first key {key!r}",
             f"trial keys: {sorted(trial)}"]
    for index in list(trial["word"])[:max_words]:
        word = trial["word"][index]
        parts = []
        for name, value in sorted(word.items()):
            array = np.asarray(value) if not isinstance(value, str) else None
            parts.append(f"{name}={value!r}" if array is None or array.size <= 3 else f"{name}: shape {array.shape}")
        lines.append(f"  word {index!r}: " + ", ".join(parts))
    return "\n".join(lines)


def load_teco_trials(trt_dir, labels_csv=None, suffix="_trt_total.pickle"):
    paths = sorted(glob.glob(os.path.join(trt_dir, f"*{suffix}")))
    if not paths:
        raise FileNotFoundError(f"no *{suffix} files in {trt_dir}")
    label_of = load_labels(labels_csv) if labels_csv else None
    participants = []
    for path in paths:
        with open(path, "rb") as handle:
            participants.append((os.path.basename(path)[:-len(suffix)], pickle.load(handle)))
    sizes = [np.asarray(w.get("TRT_total", [])).size for _, data in participants for key in data
             for w in data[key]["word"].values()]
    n_values = max(sizes, default=0)
    if n_values <= 2:
        raise ValueError(f"no fixated words with a TRT_total vector in {trt_dir}")
    trials = []
    for subject, data in participants:
        for key in data:
            sentence = data[key]
            indices = sorted(sentence["word"], key=int)
            word_dicts = [sentence["word"][i] for i in indices]
            features = np.full((len(indices), 1, n_values), np.nan, dtype=np.float32)
            for j, word in enumerate(word_dicts):
                vector = np.asarray(word.get("TRT_total", []), dtype=np.float64).ravel()
                if vector.size == n_values and np.isfinite(vector).all():
                    features[j, 0] = vector
            text = sentence.get("persian_sentence", "")
            if label_of is None:
                label = 0
            else:
                label = label_of.get(normalize_sentence(text))
                if label is None:
                    raise ValueError(f"sentenceId {sentence.get('sentenceId')} of {subject} has no label in {labels_csv}")
            fixated = np.isfinite(features[:, 0]).all(axis=1)
            trials.append({"sample_id": f"{subject}_{int(sentence['sentenceId']):04d}", "subject_id": subject,
                           "sentence_id": int(sentence["sentenceId"]), "label": int(label),
                           "words": _word_texts(sentence, word_dicts), "features": features,
                           "fixations": fixated.astype(np.float32)})
    return trials
