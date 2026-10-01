"""Word-level EEG reader against an HDF5 file laid out like MATLAB v7.3 ZuCo files."""

import os

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

from src.fusion.word_eeg import BANDS, N_CHANNELS, extract_subject, load_word_eeg, save_subject  # noqa: E402
from src.labels import label_lookup, match_sentence  # noqa: E402


def matlab_string(handle, name, text):
    return handle.create_dataset(name, data=np.array([[ord(c)] for c in text], dtype=np.uint16)).ref


def matlab_empty(handle, name):
    dataset = handle.create_dataset(name, data=np.zeros(2, dtype=np.uint64))
    dataset.attrs["MATLAB_empty"] = 1
    return dataset.ref


def write_zuco_like(path, sentences, rng):
    ref_dtype = h5py.special_dtype(ref=h5py.Reference)
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        content_refs, word_refs = [], []
        counter = [0]

        def name():
            counter[0] += 1
            return f"r{counter[0]}"

        for s, (text, fixated) in enumerate(sentences):
            content_refs.append(matlab_string(refs, name(), text))
            words = text.split(" ")
            group = refs.create_group(name())
            group.create_dataset("content", data=np.array([[matlab_string(refs, name(), w)] for w in words],
                                                          dtype=ref_dtype))
            for band in BANDS:
                cells = []
                for w in range(len(words)):
                    if fixated[w]:
                        values = rng.uniform(1, 2, size=(N_CHANNELS, 1)) * (10 if band == "a1" else 1)
                        cells.append([refs.create_dataset(name(), data=values).ref])
                    else:
                        cells.append([matlab_empty(refs, name())])
                group.create_dataset(f"TRT_{band}", data=np.array(cells, dtype=ref_dtype))
            group.create_dataset("nFixations", data=np.array(
                [[refs.create_dataset(name(), data=np.array([[float(fixated[w])]])).ref] for w in range(len(words))],
                dtype=ref_dtype))
            word_refs.append(group.ref)
        data = f.create_group("sentenceData")
        data.create_dataset("content", data=np.array([[r] for r in content_refs], dtype=ref_dtype))
        data.create_dataset("word", data=np.array([[r] for r in word_refs], dtype=ref_dtype))


def test_word_eeg_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    sentences = [("A great film .", [1, 1, 0, 1]), ("Dull and long", [1, 0, 1]), ("Not labelled", [1, 1])]
    path = os.path.join(tmp_path, "resultsZAB_SR.mat")
    write_zuco_like(path, sentences, rng)
    labels = os.path.join(tmp_path, "labels.csv")
    pd.DataFrame({"sentence_id": [3, 8], "sentence": ["A great film .", "Dull and long"],
                  "sentiment_label": [1, -1]}).to_csv(labels, index=False)
    trials, records = extract_subject(path, label_lookup(labels), match_sentence)
    assert [r["status"] for r in records] == ["ok", "ok", "unlabelled_sentence"]
    first = trials[0]
    assert first["words"] == ["A", "great", "film", "."] and first["sample_id"] == "ZAB_0003"
    assert first["features"].shape == (4, len(BANDS), N_CHANNELS)
    assert np.isnan(first["features"][2]).all() and np.isfinite(first["features"][[0, 1, 3]]).all()
    assert first["features"][0, BANDS.index("a1")].mean() > 5
    np.testing.assert_array_equal(first["fixations"], [1, 1, 0, 1])
    out = os.path.join(tmp_path, "cache")
    save_subject(out, "ZAB", trials)
    loaded = load_word_eeg(out)
    assert [t["words"] for t in loaded] == [t["words"] for t in trials]
    np.testing.assert_array_equal(np.isnan(loaded[1]["features"]), np.isnan(trials[1]["features"]))
