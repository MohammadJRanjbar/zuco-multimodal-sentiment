"""Fixation-related potentials (FRPs) from ZuCo's raw sentence EEG.

ZuCo stores, per sentence, the preprocessed EEG (``rawData``, 105 channels x
samples at 500 Hz) and, per word, the EEG recorded during each of its
fixations (``word.rawEEG``, one array per fixation). The files do not store
fixation onset times, so each fixation segment is located inside ``rawData``:
first by exact match, then allowing a constant per-channel offset. The
onset of a word's first fixation then anchors an epoch from -100 to +700 ms,
which is average-referenced (this restores the flat Cz reference), low-pass
filtered at 30 Hz on the whole sentence, and baseline-corrected on -100..0 ms.

Features per word: mean amplitude per channel in eight windows (the seven
100-ms windows from 0 to 700 ms and the N400 window 300-500 ms), stored like
ZuCo's band-power features ([n_words, 8, 105]) so the existing word-EEG
analyses run unchanged. Responses to the next fixation overlap the later
windows; the time to the next fixation is saved so analyses can control it.
"""

import numpy as np

from ..fusion.word_eeg import N_CHANNELS, READING_MEASURES, read_sentence_words, read_v5_words
from ..neurolm.dataset import orient_raw

SFREQ = 500.0
PRE_MS, POST_MS = 100, 700
WINDOWS_MS = ((0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600), (600, 700), (300, 500))
WINDOW_NAMES = ("0-100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-700", "N400 300-500")
N400_INDEX = 7
# Electrode groups (GSN HydroCel labels within 25 deg of Oz / 30 deg of CPz on the ZuCo montage).
OCCIPITAL = ("E75", "E70", "E83", "E76", "E71", "E74", "E82")
CENTRO_PARIETAL = ("E55", "Cz", "E31", "E80", "E79", "E54", "E106", "E7", "E87", "E37", "E78", "E61")


# ----------------------------------------------------------------------------
# Reading per-fixation segments
# ----------------------------------------------------------------------------


def _hdf5_cells(handle, ref):
    """All 2-D arrays reachable from a MATLAB v7.3 cell reference (nested cells allowed)."""
    import h5py

    if not ref:
        return []
    obj = handle[ref]
    if not isinstance(obj, h5py.Dataset):
        return []
    if h5py.check_dtype(ref=obj.dtype) is not None:
        out = []
        for child in np.asarray(obj).reshape(-1):
            out.extend(_hdf5_cells(handle, child))
        return out
    if obj.attrs.get("MATLAB_empty", 0):
        return []
    segment = orient_raw(np.asarray(obj, dtype=np.float64), N_CHANNELS, from_hdf5=True)
    return [segment] if segment is not None else []


def _v5_cells(value):
    """Same for a scipy-loaded value (squeeze_me turns a one-fixation cell into a plain array)."""
    if value is None:
        return []
    if isinstance(value, np.ndarray) and value.dtype == object:
        out = []
        for child in value.reshape(-1):
            out.extend(_v5_cells(child))
        return out
    array = np.asarray(value)
    if array.ndim != 2 or array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return []
    segment = orient_raw(array.astype(np.float64), N_CHANNELS, from_hdf5=False)
    return [segment] if segment is not None else []


def iter_frp_sentences(path):
    """Yield ``(position, sentence text, parse)``; ``parse()`` returns a dict or ``None``.

    The dict holds ``raw`` [105, T], ``words``, ``fixations`` (counts), ``times``
    (TRT/FFD/GD in ms) and ``segments`` (per word, a list of [105, L] arrays).
    """
    try:
        import h5py

        is_hdf5 = h5py.is_hdf5(path)
    except ImportError:
        is_hdf5 = False
    if is_hdf5:
        with h5py.File(path, "r") as handle:
            data = handle["sentenceData"]
            contents = np.asarray(data["content"]).reshape(-1)
            words = np.asarray(data["word"]).reshape(-1)
            raws = np.asarray(data["rawData"]).reshape(-1) if "rawData" in data else [None] * len(contents)
            for position, (content_ref, word_ref, raw_ref) in enumerate(zip(contents, words, raws)):
                text = "".join(chr(int(c)) for c in np.asarray(handle[content_ref]).reshape(-1) if int(c) > 0)

                def parse(word_ref=word_ref, raw_ref=raw_ref):
                    parsed = read_sentence_words(handle, word_ref)
                    if parsed is None or not raw_ref:
                        return None
                    tokens, _, fixations, times = parsed
                    group = handle[word_ref]
                    if "rawEEG" not in group:
                        return {"missing_field": "rawEEG", "word_fields": sorted(group.keys())}
                    refs = np.asarray(group["rawEEG"]).reshape(-1)[:len(tokens)]
                    raw = orient_raw(np.asarray(handle[raw_ref], dtype=np.float64), N_CHANNELS, from_hdf5=True)
                    return {"raw": raw, "words": tokens, "fixations": fixations, "times": times,
                            "segments": [_hdf5_cells(handle, r) for r in refs]}

                yield position, text.strip(), parse
        return
    from scipy.io import loadmat

    data = loadmat(path, struct_as_record=False, squeeze_me=True, variable_names=["sentenceData"])
    for position, sentence in enumerate(np.atleast_1d(data["sentenceData"])):
        def parse(sentence=sentence):
            entries = getattr(sentence, "word", None)
            parsed = read_v5_words(entries)
            stored = getattr(sentence, "rawData", None)
            if parsed is None or stored is None or not np.size(stored):
                return None
            tokens, _, fixations, times = parsed
            entries = list(np.atleast_1d(entries))
            if not hasattr(entries[0], "rawEEG"):
                return {"missing_field": "rawEEG", "word_fields": sorted(getattr(entries[0], "_fieldnames", []))}
            raw = orient_raw(np.asarray(stored, dtype=np.float64), N_CHANNELS, from_hdf5=False)
            return {"raw": raw, "words": tokens, "fixations": fixations, "times": times,
                    "segments": [_v5_cells(getattr(e, "rawEEG", None)) for e in entries[:len(tokens)]]}

        yield position, str(getattr(sentence, "content", "") or "").strip(), parse


# ----------------------------------------------------------------------------
# Locating fixations in the sentence EEG
# ----------------------------------------------------------------------------


def _match_cost(R, S, n, k):
    return sum(np.abs(R[:, j:j + n] - S[:, j:j + 1]).max(axis=0) for j in range(k))


def locate_segment(raw, segment, n_channels=8, rtol=1e-4):
    """Onset sample of ``segment`` inside ``raw`` and the method used ('exact', 'offset' or failure reason)."""
    L, T = segment.shape[1], raw.shape[1]
    if L < 2:
        return None, "too_short"
    if L > T:
        return None, "longer_than_sentence"
    raw = np.nan_to_num(raw)
    segment = np.nan_to_num(segment)
    spread = segment.std(axis=1)
    channels = np.argsort(spread)[::-1][:n_channels]
    if spread[channels[0]] == 0:
        return None, "flat_segment"
    scale = max(float(np.abs(segment[channels]).max()), 1e-6)
    n = T - L + 1
    k = min(3, L)
    cost = _match_cost(raw[channels], segment[channels], n, k)
    t = int(np.argmin(cost))
    if cost[t] <= rtol * scale * k and np.allclose(raw[:, t:t + L], segment, rtol=0, atol=rtol * scale):
        return t, "exact"
    dR, dS = np.diff(raw[channels], axis=1), np.diff(segment[channels], axis=1)
    cost = _match_cost(dR, dS, n, min(3, L - 1))
    t = int(np.argmin(cost))
    difference = segment - raw[:, t:t + L]
    if cost[t] <= rtol * scale * 3 and float(difference.std(axis=1).max()) <= rtol * scale:
        return t, "offset"
    return None, "unmatched"


def locate_sentence(raw, segments):
    """Per word, the matched onsets (sorted) and per-method counts."""
    onsets, durations, counts = [], [], {}
    for word_segments in segments:
        word_onsets, word_durations = [], []
        for segment in word_segments:
            t, method = locate_segment(raw, segment)
            counts[method] = counts.get(method, 0) + 1
            if t is not None:
                word_onsets.append(t)
                word_durations.append(segment.shape[1])
        order = np.argsort(word_onsets)
        onsets.append([word_onsets[i] for i in order])
        durations.append([word_durations[i] for i in order])
    return onsets, durations, counts


# ----------------------------------------------------------------------------
# Epochs and window features
# ----------------------------------------------------------------------------


def preprocess_sentence(raw, sfreq=SFREQ, lowpass_hz=30.0):
    """Average reference (restores the flat Cz) and zero-phase 30 Hz low-pass on the whole sentence."""
    from scipy.signal import butter, filtfilt

    x = np.nan_to_num(np.asarray(raw, dtype=np.float64))
    x = x - x.mean(axis=0, keepdims=True)
    b, a = butter(4, lowpass_hz / (sfreq / 2.0))
    if x.shape[1] > 3 * max(len(a), len(b)):
        x = filtfilt(b, a, x, axis=1)
    return x


def epoch_words(x, onsets, durations, sfreq=SFREQ, pre_ms=PRE_MS, post_ms=POST_MS, windows=WINDOWS_MS):
    """Window means [n_words, len(windows), C] at each word's first fixation, plus timing per word.

    Returns ``(features, epochs, timing)``: ``epochs`` [n_valid, C, samples] for grand averages, and
    ``timing`` with onset, first-fixation duration and time to the next fixation (ms; NaN if unknown).
    """
    pre, post = int(round(pre_ms * sfreq / 1000)), int(round(post_ms * sfreq / 1000))
    n_words, n_channels = len(onsets), x.shape[0]
    features = np.full((n_words, len(windows), n_channels), np.nan, dtype=np.float32)
    timing = {key: np.full(n_words, np.nan, dtype=np.float32) for key in ("onset_ms", "first_fix_ms", "next_fix_ms")}
    all_onsets = np.sort([t for word in onsets for t in word])
    epochs = []
    for w, (word_onsets, word_durations) in enumerate(zip(onsets, durations)):
        if not word_onsets:
            continue
        t = word_onsets[0]
        timing["onset_ms"][w] = t * 1000 / sfreq
        timing["first_fix_ms"][w] = word_durations[0] * 1000 / sfreq
        later = all_onsets[all_onsets > t]
        if len(later):
            timing["next_fix_ms"][w] = (later[0] - t) * 1000 / sfreq
        if t - pre < 0 or t + post > x.shape[1]:
            continue
        epoch = x[:, t - pre:t + post]
        epoch = epoch - epoch[:, :pre].mean(axis=1, keepdims=True)
        for i, (start, stop) in enumerate(windows):
            a, b = pre + int(round(start * sfreq / 1000)), pre + int(round(stop * sfreq / 1000))
            features[w, i] = epoch[:, a:b].mean(axis=1)
        epochs.append(epoch.astype(np.float32))
    return features, (np.stack(epochs) if epochs else np.zeros((0, n_channels, pre + post), np.float32)), timing


def channel_indices(names, labels):
    lookup = {label: i for i, label in enumerate(labels)}
    return [lookup[n] for n in names if n in lookup]


def zuco_channel_labels():
    import os

    import pandas as pd

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "neurolm", "montages", "zuco_chanlocs.csv")
    return pd.read_csv(path)["labels"].tolist()


__all__ = ["READING_MEASURES", "iter_frp_sentences", "locate_segment", "locate_sentence", "preprocess_sentence",
           "epoch_words", "WINDOWS_MS", "WINDOW_NAMES", "N400_INDEX", "OCCIPITAL", "CENTRO_PARIETAL",
           "channel_indices", "zuco_channel_labels", "SFREQ", "PRE_MS", "POST_MS"]
