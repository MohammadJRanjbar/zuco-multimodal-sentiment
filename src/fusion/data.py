"""Word-aligned EEG inputs: per-reader normalization, controls, and batches.

Each word's EEG vector is ZuCo's 8-band power over 104 electrodes (the flat Cz
reference is dropped), log-transformed when all values are positive, then
standardized *per reader* with statistics from that reader's training-sentence
words only. That removes most reader identity (which dominated the
sentence-level features) and keeps word-to-word variation. The model input
per word is ``[832 normalized values, has_eeg flag]``; skipped words are zeros
with flag 0.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CLASS_NAMES, LABEL_TO_ID, ZUCO_REFERENCE_CHANNEL_INDEX

ARMS = ("text_only", "text_eeg", "text_shuffled_eeg", "text_fixation_only")
ARM_CONTROL = {"text_only": None, "text_eeg": "aligned", "text_shuffled_eeg": "shuffled",
               "text_fixation_only": "fixation_only"}


@dataclass
class FusionData:
    samples: pd.DataFrame
    words: list
    features: list  # per trial: [n_words, F] float32, NaN rows where the word has no EEG
    has_eeg: list  # per trial: [n_words] bool
    log_transformed: bool

    @property
    def dim(self):
        return self.features[0].shape[1] + 1


def build_fusion_data(trials, drop_channels=(ZUCO_REFERENCE_CHANNEL_INDEX,), log_power="auto"):
    rows, words, features, has_eeg = [], [], [], []
    keep = None
    for trial in trials:
        block = np.asarray(trial["features"], dtype=np.float32)  # [n_words, bands, channels]
        if keep is None:
            keep = [c for c in range(block.shape[2]) if c not in set(drop_channels)]
        flat = block[:, :, keep].reshape(len(block), -1)
        present = np.isfinite(flat).all(axis=1)
        flat[~present] = np.nan
        rows.append({"sample_id": trial["sample_id"], "subject_id": trial["subject_id"],
                     "sentence_id": trial["sentence_id"], "label_id": LABEL_TO_ID[trial["label"]],
                     "n_words": len(trial["words"]), "n_words_with_eeg": int(present.sum())})
        words.append(list(trial["words"]))
        features.append(flat)
        has_eeg.append(present)
    observed = np.concatenate([f[h] for f, h in zip(features, has_eeg)])
    use_log = log_power is True or (log_power == "auto" and len(observed) and (observed > 0).all())
    if use_log:
        features = [np.log(f) for f in features]
    samples = pd.DataFrame(rows)
    usable = samples["n_words"] > 0
    keep_rows = np.flatnonzero(usable.to_numpy())
    return FusionData(
        samples=samples.iloc[keep_rows].reset_index(drop=True),
        words=[words[i] for i in keep_rows],
        features=[features[i] for i in keep_rows],
        has_eeg=[has_eeg[i] for i in keep_rows],
        log_transformed=bool(use_log),
    )


def reader_statistics(data, train_rows):
    """Mean/std per reader from the words of that reader's training trials."""
    stats, fallback = {}, []
    subjects = data.samples["subject_id"].to_numpy()
    for subject in np.unique(subjects):
        rows = [i for i in train_rows if subjects[i] == subject]
        if not rows:
            # Unseen reader (subject-disjoint splits): label-free statistics from
            # the reader's own words. Recorded as transductive.
            rows = list(np.flatnonzero(subjects == subject))
            fallback.append(subject)
        block = np.concatenate([data.features[i][data.has_eeg[i]] for i in rows] or
                               [np.zeros((0, data.dim - 1), np.float32)])
        if len(block) < 2:
            mean, std = np.zeros(data.dim - 1), np.ones(data.dim - 1)
        else:
            mean, std = block.mean(axis=0), block.std(axis=0)
            std[~np.isfinite(std) | (std < 1e-6)] = 1.0
        stats[subject] = (mean.astype(np.float32), std.astype(np.float32))
    return stats, fallback


def model_inputs(data, stats, control="aligned", split=None, rng=None):
    """Per-trial ``[n_words, F + 1]`` inputs for one control condition."""
    subjects = data.samples["subject_id"].to_numpy()
    out = []
    for i, (block, present) in enumerate(zip(data.features, data.has_eeg)):
        mean, std = stats[subjects[i]]
        normalized = np.zeros_like(block)
        normalized[present] = (block[present] - mean) / std
        out.append(np.concatenate([normalized, present[:, None].astype(np.float32)], axis=1))
    if control == "aligned":
        return out
    if control == "fixation_only":
        for item in out:
            item[:, :-1] = 0.0
        return out
    if control != "shuffled":
        raise ValueError(f"unknown control {control!r}")
    # Re-assign word EEG vectors among the same reader's words inside each
    # partition: the flag pattern stays, word-EEG alignment is destroyed.
    for part in (split.train, split.val, split.test):
        part = np.asarray(part)
        for subject in np.unique(subjects[part]):
            rows = part[subjects[part] == subject]
            slots = [(i, w) for i in rows for w in np.flatnonzero(data.has_eeg[i])]
            if len(slots) < 2:
                continue
            vectors = np.stack([out[i][w, :-1] for i, w in slots])
            vectors = vectors[rng.permutation(len(vectors))]
            for (i, w), vector in zip(slots, vectors):
                out[i][w, :-1] = vector
    return out


def batch_inputs(rows, data, inputs):
    """Words, a padded ``[B, max_words, F + 1]`` array, and word counts."""
    words = [data.words[i] for i in rows]
    counts = np.array([len(w) for w in words])
    width = int(counts.max())
    eeg = np.zeros((len(rows), width, inputs[rows[0]].shape[1]), dtype=np.float32)
    for b, i in enumerate(rows):
        eeg[b, :counts[b]] = inputs[i]
    return words, eeg, counts


def class_names():
    return list(CLASS_NAMES)
