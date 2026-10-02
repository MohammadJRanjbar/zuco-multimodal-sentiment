import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.brainshaping.data import grouped_splits  # noqa: E402
from src.brainshaping.encoding import (  # noqa: E402
    char_word_index, prepare_folds, ridge_predictor, scan_layer, summarize_layer, token_word_map,
)
from src.brainshaping.teco import describe_pickle, load_teco_trials  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_token_word_map_handles_leading_spaces_and_pieces():
    text, char_word = char_word_index(["the", "movie's", "great"])
    assert text == "the movie's great"
    # byte-level BPE style: leading spaces inside tokens, a split word, special and pad tokens
    offsets = [(0, 0), (0, 3), (3, 9), (9, 11), (11, 17), (0, 0)]
    assert token_word_map(offsets, char_word) == [-1, 0, 1, 1, 2, -1]
    # a token that covers only a space belongs to no word
    assert token_word_map([(3, 4)], char_word) == [-1]


@pytest.mark.parametrize("n_rows", [200, 40])  # primal (rows >= features) and dual form
def test_ridge_predictor_matches_sklearn(n_rows):
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    X, Y, X_test = rng.standard_normal((n_rows, 60)), rng.standard_normal((n_rows, 3)), rng.standard_normal((25, 60))
    alphas = np.array([0.5, 30.0])
    predict = ridge_predictor(torch.as_tensor(X), torch.as_tensor(X_test))
    ours = predict(torch.as_tensor(Y), torch.as_tensor(alphas)).numpy()
    for i, alpha in enumerate(alphas):
        np.testing.assert_allclose(ours[i], Ridge(alpha=alpha).fit(X, Y).predict(X_test), atol=1e-6)


def test_grouped_splits_keep_sentences_whole():
    groups = np.repeat(np.arange(30), 4)
    splits = grouped_splits(groups, 5, seed=1)
    assert sorted(np.concatenate([test for _, test in splits]).tolist()) == list(range(120))
    for train, test in splits:
        assert not set(groups[train]) & set(groups[test])


def synthetic_layers(n_sentences=150, n_words=10, dim=48, n_features=20, seed=0):
    """Layer 0: sentence-shared component only (no word-EEG link). Layer 1: carries the word code."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_sentences), n_words)
    sentence_x = rng.standard_normal((n_sentences, dim))[groups]
    sentence_eeg = rng.standard_normal((n_sentences, n_features))[groups]
    word_code = rng.standard_normal((len(groups), 6))
    eeg = sentence_eeg + word_code @ rng.standard_normal((6, n_features)) + rng.standard_normal((len(groups), n_features))
    noise_layer = 1.5 * sentence_x + rng.standard_normal((len(groups), dim))
    signal_layer = noise_layer.copy()
    signal_layer[:, :6] += 2 * word_code
    return groups, eeg.astype(np.float32), [noise_layer, signal_layer]


def test_scan_finds_signal_and_no_sentence_leak():
    groups, eeg, layers = synthetic_layers()
    prepared = prepare_folds(eeg, groups, modes=("centered", "raw"), k=4, n_folds=5, n_inner=4, seed=0)
    out = {}
    for mode in ("centered", "raw"):
        for index, X in enumerate(layers):
            folds = scan_layer(X, prepared["modes"][mode], prepared["codes"], mode)
            out[mode, index] = summarize_layer(folds, n_boot=300)
    for mode in ("centered", "raw"):
        signal = out[mode, 1]
        assert signal["r2"] > 0.2 and signal["r2_ci_low"] > 0 and signal["delta_ci_low"] > 0, (mode, signal)
        noise = out[mode, 0]
        # no word-EEG link: sentence-grouped selection keeps R^2 near 0 instead of overfitting sentences
        assert noise["r2"] > -0.03 and noise["delta_ci_low"] < 0.01, (mode, noise)


def test_residualized_targets_separate_lexical_from_other_signal():
    from src.brainshaping.encoding import residualize_folds

    rng = np.random.default_rng(3)
    n_sentences, n_words = 150, 10
    groups = np.repeat(np.arange(n_sentences), n_words)
    length = rng.integers(2, 12, len(groups)).astype(float)
    other = rng.standard_normal(len(groups))                       # word property not in the controls
    lexical = np.log1p(length)[:, None]
    vectors = rng.standard_normal((len(groups), 32))
    vectors[:, 0] += 2 * (lexical[:, 0] - lexical.mean())          # vectors encode length ...
    vectors[:, 1] += 2 * other                                       # ... and the other property
    mixing = rng.standard_normal((2, 12))
    for weight, expect_beyond in ((0.0, False), (1.0, True)):
        drivers = np.column_stack([lexical[:, 0] - lexical.mean(), weight * other])
        eeg = (drivers @ mixing + 0.8 * rng.standard_normal((len(groups), 12))).astype(np.float32)
        prepared = prepare_folds(eeg, groups, modes=("centered",), k=4, n_folds=5, n_inner=4, seed=0)
        folds, codes = prepared["modes"]["centered"], prepared["codes"]
        full = summarize_layer(scan_layer(vectors, folds, codes, "centered"), n_boot=300)
        beyond = summarize_layer(scan_layer(vectors, residualize_folds(folds, lexical, codes), codes, "centered"),
                                 n_boot=300)
        assert full["r2_ci_low"] > 0
        if expect_beyond:
            assert beyond["r2_ci_low"] > 0 and beyond["delta_ci_low"] > 0, beyond
        else:
            assert beyond["r2"] < 0.01 and beyond["delta_ci_low"] <= 0.005, beyond


def write_teco(tmp_path, n_subjects=3, n_sentences=24, n_values=126, word_text=True):
    rng = np.random.default_rng(0)
    vocab = ["خوب", "بد", "فیلم", "کتاب", "می‌خواهم", "است", "نبود", "زیبا"]
    sentences = [list(rng.choice(vocab, size=rng.integers(4, 8))) for _ in range(n_sentences)]
    labels = np.tile([-1, 0, 1], n_sentences // 3)
    trt = os.path.join(tmp_path, "TRT_Total")
    os.makedirs(trt)
    for s in range(n_subjects):
        data = {}
        for i, words in enumerate(sentences):
            entry = {}
            for j, word in enumerate(words):
                vector = (np.exp(rng.normal(0, 0.3, n_values)) * (1 + 0.2 * len(word))).tolist()
                fixated = rng.random() > 0.15
                entry[j] = {"TRT_total": vector if fixated else [0, 0], "nFixations": int(rng.integers(1, 4)) if fixated else 0}
                if word_text:
                    entry[j]["content"] = word
            data[f"trial_{i}"] = {"sentenceId": i + 1, "persian_sentence": " ".join(words), "word": entry}
        with open(os.path.join(trt, f"P{s}_trt_total.pickle"), "wb") as handle:
            pickle.dump(data, handle)
    labels_csv = os.path.join(tmp_path, "labels.csv")
    pd.DataFrame({"sentence": [" ".join(w).replace("‌", "") for w in sentences],
                  "sentiment_label": labels, "control": 0}).to_csv(labels_csv, index=False)
    return trt, labels_csv, sentences, labels


@pytest.mark.parametrize("word_text", [True, False])
def test_teco_loader(tmp_path, word_text):
    trt, labels_csv, sentences, labels = write_teco(str(tmp_path), word_text=word_text)
    trials = load_teco_trials(trt, labels_csv)
    assert len(trials) == 3 * len(sentences)
    first = [t for t in trials if t["subject_id"] == "P0" and t["sentence_id"] == 1][0]
    assert first["words"] == sentences[0] and first["label"] == labels[0]
    assert first["features"].shape == (len(sentences[0]), 1, 126)
    assert np.array_equal(np.isfinite(first["features"][:, 0]).all(axis=1), first["fixations"] > 0)
    assert first["fixations"].max() <= 3
    assert "TRT_total: shape (126,)" in describe_pickle(os.path.join(trt, "P0_trt_total.pickle"))


def test_teco_loader_skips_empty_and_inconsistent_trials(tmp_path):
    trt, labels_csv, sentences, _ = write_teco(str(tmp_path), n_subjects=3)
    path = os.path.join(trt, "P1_trt_total.pickle")
    data = pickle.load(open(path, "rb"))
    keys = list(data)
    data[keys[0]]["word"] = {}                                       # a trial with no words
    words = data[keys[1]]["word"]
    words[0] = {**words[0], "content": words[0]["content"] + "x"}    # this reader's word list disagrees
    pickle.dump(data, open(path, "wb"))
    messages = []
    trials = load_teco_trials(trt, labels_csv, log=messages.append)
    assert len(trials) == 3 * len(sentences) - 2
    assert any("no words" in m for m in messages) and any("differs" in m for m in messages)
    from src.brainshaping.data import item_eeg, sentence_table

    items, eeg = item_eeg(trials, drop_channels=())
    table = sentence_table(trials)
    assert table["words"].tolist() == sentences and len(items) == len(eeg) > 0


def test_teco_loader_explains_missing_word_text(tmp_path):
    trt, _, _, _ = write_teco(str(tmp_path), n_subjects=1, word_text=False)
    path = os.path.join(trt, "P0_trt_total.pickle")
    data = pickle.load(open(path, "rb"))
    first = next(iter(data))
    data[first]["persian_sentence"] += " اضافه"
    pickle.dump(data, open(path, "wb"))
    with pytest.raises(ValueError, match="cannot recover word text"):
        load_teco_trials(trt)


def fake_extractor(signal_layer=2, n_layers=4, dim=16):
    def extract(sentences, item_sentence, item_word, model_name, device="cpu", **kwargs):
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((n_layers, len(item_sentence), dim)).astype(np.float32)
        lengths = np.array([len(sentences[s][w]) for s, w in zip(item_sentence, item_word)], dtype=np.float32)
        vectors[signal_layer, :, 0] += 3 * (lengths - lengths.mean()) / lengths.std()
        return vectors, np.ones(len(item_sentence), bool), {"pool": "mean", "model": model_name}
    return extract


def test_scan_script_zuco_end_to_end(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_diagnostics import word_trials, write_cache

    trials, _, _ = word_trials(n_readers=4, n_sentences=60)
    cache = write_cache(str(tmp_path), trials)
    runner = load_script("scan_eeg_encoding")
    monkeypatch.setattr(runner, "extract_layer_vectors", fake_extractor())
    out = os.path.join(str(tmp_path), "scan")
    argv = ["x", "--dataset", "zuco", "--word-eeg-dir", cache, "--out-dir", out, "--models", "fake=org/fake-model",
            "--k", "4", "--n-boot", "200", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    scan_dir = os.path.join(out, "encoding_scan_zuco")
    table = pd.read_csv(os.path.join(scan_dir, "fake.csv"))
    assert len(table) == 4 * 2 and set(table["mode"]) == {"centered", "raw"}
    by_layer = table[table["mode"] == "raw"].set_index("layer")["r2"]
    assert by_layer.idxmax() == 2 and by_layer[2] > 0.05, by_layer
    assert len(json.loads(table["fold_r2"].iloc[0])) == 5
    report = open(os.path.join(scan_dir, "encoding_scan_zuco.md")).read()
    assert "## Verdict" in report and "fake | raw | 2/3" in report
    assert os.path.exists(os.path.join(scan_dir, "encoding_scan_zuco.png"))
    # synthetic EEG tracks word length only, and the fake vectors' signal is word length
    assert "**Noise ceiling:**" in report and "Beyond lexical: nothing remains" in report, report
    controls = pd.read_csv(os.path.join(scan_dir, "controls_zuco.csv"))
    assert controls["layer"].tolist() == [2] and controls["r2_vectors"].iloc[0] > 0.05
    capsys.readouterr()
    runner.main()
    assert "reusing finished scan" in capsys.readouterr().out


def test_scan_script_teco_end_to_end(tmp_path, monkeypatch):
    trt, labels_csv, _, _ = write_teco(str(tmp_path), n_subjects=3, n_sentences=30)
    runner = load_script("scan_eeg_encoding")
    monkeypatch.setattr(runner, "extract_layer_vectors", fake_extractor(signal_layer=1, n_layers=3))
    out = os.path.join(str(tmp_path), "scan")
    monkeypatch.setattr(sys, "argv", ["x", "--dataset", "teco", "--trt-dir", trt, "--labels-csv", labels_csv,
                                      "--out-dir", out, "--models", "fake=org/fake", "--k", "4", "--n-boot", "100",
                                      "--modes", "raw", "--device", "cpu"])
    runner.main()
    table = pd.read_csv(os.path.join(out, "encoding_scan_teco", "fake.csv"))
    assert table.set_index("layer")["r2"].idxmax() == 1
