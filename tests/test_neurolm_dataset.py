import os

import numpy as np
import pandas as pd
from scipy.io import savemat

from src.neurolm.dataset import iter_subject_trials, load_handcrafted_trials, orient_raw, sample_id
from src.features import N_FAMILIES, feature_names


def test_orient_raw_uses_channel_count_not_heuristics():
    assert orient_raw(np.zeros((1500, 105))).shape == (105, 1500)
    assert orient_raw(np.zeros((105, 1500))).shape == (105, 1500)
    square = np.arange(105 * 105).reshape(105, 105)
    np.testing.assert_array_equal(orient_raw(square, from_hdf5=True), square.T)
    np.testing.assert_array_equal(orient_raw(square, from_hdf5=False), square)
    # a short trial (fewer samples than channels) is still oriented correctly
    assert orient_raw(np.zeros((60, 105))).shape == (105, 60)
    assert orient_raw(np.zeros((50, 60))) is None
    assert orient_raw(np.zeros(2)) is None


def write_subject(tmp_path, subject="ZAB"):
    rng = np.random.default_rng(0)
    sentences = np.zeros((4,), dtype=[("content", "O"), ("rawData", "O")])
    sentences[0] = ("A great, moving film.", rng.standard_normal((105, 1250)))
    sentences[1] = ("Not in the label file.", rng.standard_normal((105, 900)))
    sentences[2] = ("A dull film.", np.zeros((0, 0)))
    sentences[3] = ("A great, moving film.", rng.standard_normal((105, 700)))
    path = os.path.join(tmp_path, f"results{subject}_SR.mat")
    savemat(path, {"sentenceData": sentences})
    labels = os.path.join(tmp_path, "labels.csv")
    pd.DataFrame({"sentence_id": [7, 9], "sentence": ["A great, moving film.", "A dull film."],
                  "sentiment_label": [1, -1]}).to_csv(labels, index=False)
    return path, labels


def test_raw_trials_keep_subject_and_sentence_separate(tmp_path):
    path, labels = write_subject(str(tmp_path))
    results = list(iter_subject_trials(path, labels_csv=labels))
    statuses = [record["status"] for _, record in results]
    assert statuses == ["ok", "unlabelled_sentence", "missing_raw", "duplicate_sentence"]
    trial, record = results[0]
    assert trial.sample_id == sample_id("ZAB", 7) == "ZAB_0007"
    assert trial.raw.shape == (105, 1250) and trial.label == 1 and trial.label_id == 2
    assert record["duration_s"] == 2.5 and record["n_channels"] == 105


def test_handcrafted_trials_are_per_subject_rows(tmp_path):
    names = feature_names(105)
    pd.Series(names).to_json(os.path.join(tmp_path, "feature_names.json"), orient="values")
    X = np.ones((3, 105 * N_FAMILIES), dtype=np.float32)
    X[1] = np.nan  # unusable row
    for subject in ("ZAB", "ZDM"):
        np.savez_compressed(os.path.join(tmp_path, f"{subject}.npz"), X=X,
                            sentence_id=np.array([1, 2, 3]), label=np.array([-1, 0, 1]))
    table, features = load_handcrafted_trials(str(tmp_path))
    assert len(table) == 4 and features.shape == (4, 104 * N_FAMILIES)
    assert set(table["subject_id"]) == {"ZAB", "ZDM"}
    assert table["label_id"].tolist() == [0, 2, 0, 2]
