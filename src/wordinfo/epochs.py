"""Raw fixation-locked EEG epochs per word: input of pretrained EEG models and of raw time-course features.

Each sentence's EEG is preprocessed as for NeuroLM (``src/neurolm/preprocess.py``: demean, 50 Hz notch,
0.1-75 Hz band-pass, average reference including the flat Cz, resampled to 200 Hz, microvolts). For every
word whose first fixation was located in the sentence EEG (``src/followup/frp.py``), a 1-s epoch from
-200 to +800 ms around the fixation onset is cut: 200 samples, one patch of CBraMod / NeuroLM / LaBraM.

Saved per subject in the word-EEG cache layout (sentence ids, labels, words, fixations, reading times;
``features`` holds NaN placeholders) plus ``epochs`` [n_epochs, 105, 200] float16 and ``epoch_word``, the
flat word index of each epoch.
"""

import glob
import os

import numpy as np

from ..fusion.word_eeg import READING_MEASURES
from ..neurolm.preprocess import PreprocessConfig, preprocess_trial

TMIN_S, TMAX_S = -0.2, 0.8


def word_epochs(raw, onsets, cfg=None, tmin=TMIN_S, tmax=TMAX_S):
    """``(epochs [n, C, samples] float32, word indices [n])`` for words with a first-fixation onset.

    ``onsets``: per word, sorted fixation onsets in samples of ``raw`` (``cfg.source_sfreq``).
    """
    cfg = cfg or PreprocessConfig()
    data, _ = preprocess_trial(raw, cfg)
    ratio = cfg.target_sfreq / cfg.source_sfreq
    start, length = int(round(tmin * cfg.target_sfreq)), int(round((tmax - tmin) * cfg.target_sfreq))
    epochs, index = [], []
    for w, word_onsets in enumerate(onsets):
        if not len(word_onsets):
            continue
        first = int(round(word_onsets[0] * ratio)) + start
        if first < 0 or first + length > data.shape[1]:
            continue
        epochs.append(data[:, first:first + length])
        index.append(w)
    if not epochs:
        return np.zeros((0, data.shape[0], length), np.float32), np.zeros(0, np.int64)
    return np.stack(epochs).astype(np.float32), np.array(index, dtype=np.int64)


def save_subject_epochs(out_dir, subject, trials):
    """``trials``: dicts with sample_id, sentence_id, label, words, fixations, *_ms, epochs, epoch_index."""
    os.makedirs(out_dir, exist_ok=True)
    offsets = np.cumsum([0] + [len(t["words"]) for t in trials]).astype(np.int64)
    n_words = int(offsets[-1])
    epochs = [t["epochs"] for t in trials if len(t["epochs"])]
    arrays = {
        "sample_id": np.array([t["sample_id"] for t in trials]),
        "sentence_id": np.array([t["sentence_id"] for t in trials], dtype=np.int64),
        "label": np.array([t["label"] for t in trials], dtype=np.int64),
        "offsets": offsets,
        "words": np.array([w for t in trials for w in t["words"]]),
        "fixations": np.concatenate([t["fixations"] for t in trials]).astype(np.float32) if trials
        else np.zeros(0, np.float32),
        "features": np.full((n_words, 1), np.nan, np.float32),
        "epochs": np.concatenate(epochs).astype(np.float16) if epochs else np.zeros((0, 105, 200), np.float16),
        "epoch_word": np.concatenate([offsets[i] + t["epoch_index"] for i, t in enumerate(trials)]).astype(np.int64)
        if trials else np.zeros(0, np.int64),
    }
    for name in READING_MEASURES:
        key = f"{name.lower()}_ms"
        arrays[key] = np.concatenate([np.asarray(t.get(key, np.full(len(t["words"]), np.nan)), dtype=np.float32)
                                      for t in trials]) if trials else np.zeros(0, np.float32)
    temporary = os.path.join(out_dir, f"{subject}.tmp.npz")
    np.savez(temporary, **arrays)  # epochs are float16 already; compression would only cost time
    os.replace(temporary, os.path.join(out_dir, f"{subject}.npz"))


def load_subject_epochs(path):
    """``(arrays dict, trials list)``: trials in the word-EEG layout (features = NaN placeholders)."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    subject = os.path.basename(path)[:-4]
    offsets, words = arrays["offsets"], [str(w) for w in arrays["words"]]
    trials = []
    for i, sid in enumerate(arrays["sample_id"]):
        start, stop = int(offsets[i]), int(offsets[i + 1])
        trials.append({"sample_id": str(sid), "subject_id": subject, "sentence_id": int(arrays["sentence_id"][i]),
                       "label": int(arrays["label"][i]), "words": words[start:stop],
                       "fixations": arrays["fixations"][start:stop], "word_offset": start,
                       **{f"{n.lower()}_ms": arrays[f"{n.lower()}_ms"][start:stop] for n in READING_MEASURES}})
    return arrays, trials


def subject_epoch_files(epochs_dir):
    return sorted(p for p in glob.glob(os.path.join(epochs_dir, "*.npz")) if not p.endswith(".tmp.npz"))
