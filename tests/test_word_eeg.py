"""Word-level EEG reader against an HDF5 file laid out like MATLAB v7.3 ZuCo files."""

import os

import numpy as np
import pandas as pd
import pytest

from src.fusion.word_eeg import BANDS, N_CHANNELS, extract_subject, load_word_eeg, save_subject  # noqa: E402
from src.labels import label_lookup, match_sentence  # noqa: E402


def matlab_string(handle, name, text):
    import h5py  # noqa: F401
    return handle.create_dataset(name, data=np.array([[ord(c)] for c in text], dtype=np.uint16)).ref


def matlab_empty(handle, name):
    import h5py  # noqa: F401
    dataset = handle.create_dataset(name, data=np.zeros(2, dtype=np.uint64))
    dataset.attrs["MATLAB_empty"] = 1
    return dataset.ref


def write_zuco_like(path, sentences, rng):
    import h5py

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
            for measure, scale in (("TRT", 250.0), ("FFD", 200.0), ("GD", 220.0)):
                group.create_dataset(measure, data=np.array(
                    [[refs.create_dataset(name(), data=np.array([[scale * fixated[w]]])).ref if fixated[w]
                      else matlab_empty(refs, name())] for w in range(len(words))], dtype=ref_dtype))
            group.create_dataset("nFixations", data=np.array(
                [[refs.create_dataset(name(), data=np.array([[float(fixated[w])]])).ref] for w in range(len(words))],
                dtype=ref_dtype))
            word_refs.append(group.ref)
        data = f.create_group("sentenceData")
        data.create_dataset("content", data=np.array([[r] for r in content_refs], dtype=ref_dtype))
        data.create_dataset("word", data=np.array([[r] for r in word_refs], dtype=ref_dtype))


def test_word_eeg_roundtrip(tmp_path):
    pytest.importorskip("h5py")
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
    np.testing.assert_array_equal(first["trt_ms"][[0, 1, 3]], [250.0, 250.0, 250.0])
    assert np.isnan(first["trt_ms"][2])
    out = os.path.join(tmp_path, "cache")
    save_subject(out, "ZAB", trials)
    loaded = load_word_eeg(out)
    assert [t["words"] for t in loaded] == [t["words"] for t in trials]
    np.testing.assert_array_equal(np.isnan(loaded[1]["features"]), np.isnan(trials[1]["features"]))
    np.testing.assert_array_equal(loaded[0]["trt_ms"], trials[0]["trt_ms"])


def test_word_eeg_from_matlab_v5(tmp_path):
    from scipy.io import savemat

    rng = np.random.default_rng(1)
    word_fields = ["content", "nFixations", "TRT", "FFD", "GD"] + [f"TRT_{band}" for band in BANDS]

    def words(text, fixated):
        tokens = text.split(" ")
        array = np.zeros((len(tokens),), dtype=[(f, "O") for f in word_fields])
        for w, token in enumerate(tokens):
            empty = np.zeros((0, 0))
            values = [token, float(fixated[w]), 250.0 if fixated[w] else empty, 200.0 if fixated[w] else empty,
                      220.0 if fixated[w] else empty]
            values += [rng.uniform(1, 2, size=(N_CHANNELS, 1)) if fixated[w] else empty for _ in BANDS]
            array[w] = tuple(values)
        return array

    sentences = np.zeros((2,), dtype=[("content", "O"), ("word", "O")])
    sentences[0] = ("A great film .", words("A great film .", [1, 1, 0, 1]))
    sentences[1] = ("Not labelled", words("Not labelled", [1, 1]))
    path = os.path.join(tmp_path, "resultsZDM_SR.mat")
    savemat(path, {"sentenceData": sentences})
    labels = os.path.join(tmp_path, "labels.csv")
    pd.DataFrame({"sentence_id": [3], "sentence": ["A great film ."], "sentiment_label": [1]}).to_csv(labels, index=False)
    trials, records = extract_subject(path, label_lookup(labels), match_sentence)
    assert [r["status"] for r in records] == ["ok", "unlabelled_sentence"]
    trial = trials[0]
    assert trial["words"] == ["A", "great", "film", "."] and trial["sample_id"] == "ZDM_0003"
    assert np.isnan(trial["features"][2]).all() and np.isfinite(trial["features"][[0, 1, 3]]).all()
    np.testing.assert_array_equal(trial["fixations"], [1, 1, 0, 1])
    np.testing.assert_array_equal(trial["trt_ms"][[0, 1, 3]], [250.0, 250.0, 250.0])


def test_unreadable_file_names_the_file(tmp_path):
    from src.fusion.word_eeg import iter_sentence_words

    path = os.path.join(tmp_path, "resultsZXX_SR.mat")
    with open(path, "wb") as handle:
        handle.write(b"not a mat file at all")
    with pytest.raises(RuntimeError, match="resultsZXX_SR.mat"):
        list(iter_sentence_words(path))


def test_loaded_trials_share_one_array_per_subject(tmp_path):
    rng = np.random.default_rng(0)
    trials = []
    for s in range(30):
        n = int(rng.integers(3, 8))
        trials.append({"sample_id": f"ZAB_{s:04d}", "subject_id": "ZAB", "sentence_id": s, "label": 1,
                       "words": [f"w{i}" for i in range(n)],
                       "features": rng.standard_normal((n, len(BANDS), N_CHANNELS)).astype(np.float32),
                       "fixations": np.ones(n, np.float32), "trt_ms": np.full(n, 200.0, np.float32)})
    save_subject(os.path.join(tmp_path, "cache"), "ZAB", trials)
    loaded = load_word_eeg(os.path.join(tmp_path, "cache"))
    # Re-reading the npz per trial made every trial hold its own full copy (the Colab OOM).
    bases = {id(t["features"].base) for t in loaded}
    assert len(bases) == 1
    np.testing.assert_array_equal(loaded[7]["features"], trials[7]["features"])
    assert loaded[7]["words"] == trials[7]["words"]
