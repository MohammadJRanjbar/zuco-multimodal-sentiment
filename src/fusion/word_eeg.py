"""Word-level, fixation-locked EEG from ZuCo ``results*_SR.mat``.

For every reader x sentence trial this reads the sentence's word list and, for
each word, ZuCo's band power over the word's total reading time
(``TRT_t1 ... TRT_g2``: 8 frequency bands x 105 electrodes). Words the reader
skipped have no EEG and are marked as not fixated. The layout follows the ZuCo
authors' loader (zuco-benchmark ``data_loading_helpers.extract_word_level_data``):
``sentenceData/word[i]`` references a struct whose fields hold one object
reference per word.
"""

import glob
import json
import os

import numpy as np

BANDS = ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]
N_CHANNELS = 105


def decode_string(dataset):
    codes = np.asarray(dataset).flatten()
    return "".join(chr(int(c)) for c in codes if int(c) > 0)


def _vector(handle, ref, size):
    if not ref:
        return None
    array = np.asarray(handle[ref], dtype=np.float64)
    if array.ndim != 2 or array.size != size:
        return None
    return array.reshape(-1)


def _scalar(handle, ref):
    if not ref:
        return np.nan
    array = np.asarray(handle[ref], dtype=np.float64)
    return float(array.reshape(-1)[0]) if array.ndim == 2 and array.size >= 1 else np.nan


def read_sentence_words(handle, word_ref, measure="TRT", n_channels=N_CHANNELS):
    """Return ``(words, features [n_words, 8, C], n_fixations [n_words])`` or ``None``."""
    import h5py

    if not word_ref:
        return None
    group = handle[word_ref]
    if not isinstance(group, h5py.Group) or "content" not in group:
        return None
    content = np.asarray(group["content"]).reshape(-1)
    words = [decode_string(handle[ref]) for ref in content]
    features = np.full((len(words), len(BANDS), n_channels), np.nan, dtype=np.float32)
    fixations = np.zeros(len(words), dtype=np.float32)
    fields = [f"{measure}_{band}" for band in BANDS]
    if all(field in group for field in fields):
        for b, field in enumerate(fields):
            refs = np.asarray(group[field]).reshape(-1)
            for w, ref in enumerate(refs[:len(words)]):
                vector = _vector(handle, ref, n_channels)
                if vector is not None:
                    features[w, b] = vector
    if "nFixations" in group:
        refs = np.asarray(group["nFixations"]).reshape(-1)
        fixations[:] = [np.nan_to_num(_scalar(handle, ref)) for ref in refs[:len(words)]]
    return words, features, fixations


def extract_subject(path, lookup, match_sentence, measure="TRT"):
    """All labelled trials of one subject file (first occurrence of each sentence)."""
    import h5py

    from ..neurolm.dataset import sample_id, subject_from_path

    subject = subject_from_path(path)
    trials, records, seen = [], [], set()
    with h5py.File(path, "r") as handle:
        data = handle["sentenceData"]
        contents = np.asarray(data["content"]).reshape(-1)
        word_refs = np.asarray(data["word"]).reshape(-1)
        for position, (content_ref, word_ref) in enumerate(zip(contents, word_refs)):
            sentence = decode_string(handle[content_ref]).strip()
            sentence_id, label = match_sentence(sentence, lookup)
            record = {"subject_id": subject, "position": position, "sentence_id": sentence_id,
                      "label": label}
            if sentence_id is None:
                records.append({**record, "status": "unlabelled_sentence"})
                continue
            if sentence_id in seen:
                records.append({**record, "status": "duplicate_sentence"})
                continue
            seen.add(sentence_id)
            parsed = read_sentence_words(handle, word_ref, measure)
            if parsed is None:
                records.append({**record, "status": "no_word_data"})
                continue
            words, features, fixations = parsed
            has_eeg = np.isfinite(features).all(axis=(1, 2))
            records.append({**record, "status": "ok", "n_words": len(words),
                            "n_words_with_eeg": int(has_eeg.sum()), "n_words_fixated": int((fixations > 0).sum())})
            trials.append({
                "sample_id": sample_id(subject, sentence_id), "subject_id": subject,
                "sentence_id": int(sentence_id), "label": int(label),
                "words": words, "features": features, "fixations": fixations,
            })
    return trials, records


def save_subject(out_dir, subject, trials):
    """One compressed file per subject: concatenated words with offsets."""
    os.makedirs(out_dir, exist_ok=True)
    offsets = np.cumsum([0] + [len(t["words"]) for t in trials])
    arrays = {
        "sample_id": np.array([t["sample_id"] for t in trials]),
        "sentence_id": np.array([t["sentence_id"] for t in trials], dtype=np.int64),
        "label": np.array([t["label"] for t in trials], dtype=np.int64),
        "offsets": offsets.astype(np.int64),
        "words": np.array([w for t in trials for w in t["words"]]),
        "features": np.concatenate([t["features"] for t in trials]).astype(np.float32)
        if trials else np.zeros((0, len(BANDS), N_CHANNELS), np.float32),
        "fixations": np.concatenate([t["fixations"] for t in trials]).astype(np.float32)
        if trials else np.zeros(0, np.float32),
    }
    temporary = os.path.join(out_dir, f"{subject}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, os.path.join(out_dir, f"{subject}.npz"))


def load_word_eeg(cache_dir):
    """Return a list of trial dicts (subject order, then file order)."""
    trials = []
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        subject = os.path.basename(path)[:-4]
        data = np.load(path, allow_pickle=False)
        offsets = data["offsets"]
        for i, sid in enumerate(data["sample_id"]):
            start, stop = offsets[i], offsets[i + 1]
            trials.append({
                "sample_id": str(sid), "subject_id": subject,
                "sentence_id": int(data["sentence_id"][i]), "label": int(data["label"][i]),
                "words": [str(w) for w in data["words"][start:stop]],
                "features": data["features"][start:stop],
                "fixations": data["fixations"][start:stop],
            })
    if not trials:
        raise FileNotFoundError(f"no word-level EEG cache in {cache_dir}")
    return trials


def write_manifest(out_dir, records, measure):
    import pandas as pd

    table = pd.DataFrame(records)
    table.to_csv(os.path.join(out_dir, "word_eeg_trials.csv"), index=False)
    ok = table[table["status"] == "ok"]
    summary = {
        "measure": measure,
        "bands": BANDS,
        "n_channels": N_CHANNELS,
        "status_counts": table["status"].value_counts().to_dict(),
        "trials": int(len(ok)),
        "subjects": sorted(table["subject_id"].unique().tolist()),
        "words_total": int(ok["n_words"].sum()) if len(ok) else 0,
        "words_with_eeg": int(ok["n_words_with_eeg"].sum()) if len(ok) else 0,
        "fraction_words_with_eeg": float(ok["n_words_with_eeg"].sum() / max(ok["n_words"].sum(), 1)) if len(ok) else 0.0,
    }
    with open(os.path.join(out_dir, "word_eeg_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary
