"""Raw ZuCo Task 1 trials and per-trial handcrafted features.

Every sample is one subject reading one sentence: ``(subject_id,
sentence_id, EEG)``. Nothing is averaged across subjects here.
"""

import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CLASS_NAMES, LABEL_TO_ID, ZUCO_REFERENCE_CHANNEL_INDEX
from ..labels import label_lookup, match_sentence

ZUCO_CHANNELS = 105
LANGUAGE = "en"


def subject_from_path(path):
    match = re.search(r"results([A-Za-z0-9]+)_SR", os.path.basename(str(path)))
    if not match:
        raise ValueError(f"cannot read a subject ID from {path}")
    return match.group(1)


def sample_id(subject_id, sentence_id):
    return f"{subject_id}_{int(sentence_id):04d}"


def orient_raw(array, expected_channels=ZUCO_CHANNELS, from_hdf5=True):
    """Return ``[channels, samples]`` or ``None`` without guessing.

    MATLAB stores ``rawData`` as channels x samples; HDF5 (v7.3) files expose
    the transpose. The channel axis is the one whose length equals the known
    channel count; if both axes match, the storage convention decides.
    """
    array = np.asarray(array)
    if array.ndim != 2 or min(array.shape) < 2:
        return None
    rows, cols = array.shape
    if rows == expected_channels and cols == expected_channels:
        return array.T if from_hdf5 else array
    if cols == expected_channels:
        return array.T
    if rows == expected_channels:
        return array
    return None


def _decode(handle, ref):
    codes = np.asarray(handle[ref]).flatten()
    return "".join(chr(int(code)) for code in codes if int(code) > 0).strip()


def iter_raw_sentences(path, expected_channels=ZUCO_CHANNELS):
    """Yield ``{"position", "content", "raw", "raw_shape"}`` for one subject file."""
    try:
        import h5py

        is_hdf5 = h5py.is_hdf5(path)
    except ImportError:
        is_hdf5 = False
    if not is_hdf5:
        yield from _iter_raw_scipy(path, expected_channels)
        return
    with h5py.File(path, "r") as handle:
        data = handle["sentenceData"]
        contents = np.asarray(data["content"]).flatten()
        raws = np.asarray(data["rawData"]).flatten() if "rawData" in data else [None] * len(contents)
        for position, (content_ref, raw_ref) in enumerate(zip(contents, raws)):
            raw_shape = None
            raw = None
            if raw_ref:
                stored = np.asarray(handle[raw_ref])
                raw_shape = tuple(stored.shape)
                raw = orient_raw(stored, expected_channels, from_hdf5=True)
            yield {
                "position": position,
                "content": _decode(handle, content_ref),
                "raw": raw,
                "raw_shape": raw_shape,
            }


def _iter_raw_scipy(path, expected_channels):
    from scipy.io import loadmat

    data = loadmat(path, struct_as_record=False, squeeze_me=True)
    for position, sentence in enumerate(np.atleast_1d(data["sentenceData"])):
        stored = getattr(sentence, "rawData", None)
        raw_shape = tuple(np.shape(stored)) if stored is not None else None
        raw = orient_raw(stored, expected_channels, from_hdf5=False) if stored is not None and np.size(stored) else None
        yield {
            "position": position,
            "content": str(getattr(sentence, "content", "") or "").strip(),
            "raw": raw,
            "raw_shape": raw_shape,
        }


@dataclass
class RawTrial:
    subject_id: str
    sentence_id: int
    label: int
    label_id: int
    content: str
    position: int
    raw: np.ndarray

    @property
    def sample_id(self):
        return sample_id(self.subject_id, self.sentence_id)


def describe_raw(raw, sfreq, reference_index=ZUCO_REFERENCE_CHANNEL_INDEX):
    finite = np.isfinite(raw)
    stds = np.nanstd(np.where(finite, raw, np.nan), axis=1)
    flat = np.flatnonzero(~(stds > 0))
    return {
        "n_channels": int(raw.shape[0]),
        "n_samples": int(raw.shape[1]),
        "duration_s": float(raw.shape[1] / sfreq),
        "nan_fraction": float(1.0 - finite.mean()),
        "n_all_nan_channels": int((~finite.any(axis=1)).sum()),
        "n_flat_channels": int(len(flat)),
        "flat_channels": ",".join(str(i) for i in flat),
        "reference_channel_flat": bool(reference_index in set(flat.tolist())),
        "median_channel_std": float(np.nanmedian(stds)),
    }


def iter_subject_trials(path, labels_csv=None, lookup=None, sfreq=500.0, limit=None, sentence_ids=None):
    """Yield ``(RawTrial or None, record)`` for every sentence in one file.

    ``record`` is a dict describing the sentence (status, shape, duration),
    written to the inspection table whether or not the trial is usable.
    """
    lookup = lookup or label_lookup(labels_csv)
    subject = subject_from_path(path)
    seen = set()
    yielded = 0
    for item in iter_raw_sentences(path):
        sentence_id, label = match_sentence(item["content"], lookup)
        record = {
            "subject_id": subject,
            "position": item["position"],
            "sentence_id": sentence_id if sentence_id is not None else -1,
            "label": label if label is not None else np.nan,
            "raw_shape": "x".join(map(str, item["raw_shape"])) if item["raw_shape"] else "",
        }
        if sentence_id is None:
            record["status"] = "unlabelled_sentence"
            yield None, record
            continue
        if sentence_ids is not None and sentence_id not in sentence_ids:
            continue
        if sentence_id in seen:
            record["status"] = "duplicate_sentence"
            yield None, record
            continue
        seen.add(sentence_id)
        if item["raw"] is None:
            shape = item["raw_shape"]
            empty = shape is None or len(shape) != 2 or min(shape) < 2
            record["status"] = "missing_raw" if empty else "unexpected_shape"
            yield None, record
            continue
        record.update(describe_raw(item["raw"], sfreq))
        if record["n_all_nan_channels"] == record["n_channels"]:
            record["status"] = "all_nan"
            yield None, record
            continue
        record["status"] = "ok"
        trial = RawTrial(
            subject_id=subject,
            sentence_id=int(sentence_id),
            label=int(label),
            label_id=LABEL_TO_ID[int(label)],
            content=item["content"],
            position=item["position"],
            raw=item["raw"],
        )
        yield trial, record
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def subject_files(mat_dir, subjects=None):
    files = sorted(glob.glob(os.path.join(mat_dir, "results*_SR.mat")))
    if subjects:
        wanted = set(subjects)
        files = [f for f in files if subject_from_path(f) in wanted]
    if not files:
        raise FileNotFoundError(f"no results*_SR.mat files in {mat_dir}")
    return files


# ----------------------------------------------------------------------------
# Handcrafted per-trial features (the existing classical cache)
# ----------------------------------------------------------------------------


def load_handcrafted_trials(features_dir, drop_reference=True):
    """Per-trial classical features ``(table, X)``; subjects are NOT averaged."""
    names = json.load(open(os.path.join(features_dir, "feature_names.json")))
    suffix = f"_ch{ZUCO_REFERENCE_CHANNEL_INDEX}"
    keep = np.array([not (drop_reference and n.endswith(suffix)) for n in names])
    rows, blocks = [], []
    for path in sorted(glob.glob(os.path.join(features_dir, "*.npz"))):
        subject = os.path.basename(path).rsplit(".", 1)[0]
        cached = np.load(path, allow_pickle=True)
        X = cached["X"][:, keep].astype(np.float32)
        usable = np.isfinite(X).any(axis=1)
        for row, (sentence_id, label) in enumerate(zip(cached["sentence_id"], cached["label"])):
            if not usable[row]:
                continue
            rows.append({
                "sample_id": sample_id(subject, sentence_id),
                "subject_id": subject,
                "sentence_id": int(sentence_id),
                "label": int(label),
                "label_id": LABEL_TO_ID[int(label)],
            })
        blocks.append(X[usable])
    table = pd.DataFrame(rows)
    X = np.vstack(blocks) if blocks else np.zeros((0, int(keep.sum())), dtype=np.float32)
    if table["sample_id"].duplicated().any():
        raise ValueError("duplicate subject x sentence rows in the handcrafted cache")
    return table, X


def label_names():
    return list(CLASS_NAMES)
