import importlib.util
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

from src.followup import crosslingual as cl  # noqa: E402
from src.followup import decoding  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_extractor(dim=24, n_layers=3):
    """Word-identity vectors: one random vector per word type (plus a length dimension)."""
    def extract(sentences, item_sentence, item_word, name, device="cpu", **kwargs):
        rng = np.random.default_rng(0)
        table = {}
        out = np.zeros((n_layers, len(item_sentence), dim), np.float32)
        for row, (s, w) in enumerate(zip(item_sentence, item_word)):
            word = sentences[s][w]
            if word not in table:
                table[word] = rng.standard_normal(dim)
            out[:, row] = table[word]
            out[:, row, 0] += 3 * len(word)
        return out, np.ones(len(item_sentence), bool), {"pool": "mean", "model": name}
    return extract


def fake_static(dim=24):
    def static(sentences, item_sentence, item_word, model_name, **kwargs):
        rng = np.random.default_rng(1)
        table = {}
        out = np.zeros((len(item_sentence), dim), np.float32)
        for row, (s, w) in enumerate(zip(item_sentence, item_word)):
            word = sentences[s][w]
            if word not in table:
                table[word] = rng.standard_normal(dim)
            out[row] = table[word]
            out[row, 0] += 3 * len(word)
        return out, np.ones(len(item_sentence), bool)
    return static


def zuco_cache(tmp_path):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_diagnostics import word_trials, write_cache

    trials, _, _ = word_trials(n_readers=4, n_sentences=60)
    return write_cache(str(tmp_path), trials)


def test_overlap_and_subspace_removal():
    rng = np.random.default_rng(0)
    U = cl.orth(rng.standard_normal((30, 4)))
    assert abs(cl.overlap(U, U @ rng.standard_normal((4, 4))) - 1) < 1e-9
    W = cl.remove_subspace(rng.standard_normal((30, 4)), U)
    assert cl.overlap(U, W) < 1e-12
    assert 0 < cl.overlap(U, cl.orth(rng.standard_normal((30, 4)))) < 1


def test_two_vs_two_and_retrieval():
    rng = np.random.default_rng(1)
    T = rng.standard_normal((200, 16))
    word_ids = np.arange(200)
    i, j = decoding.sample_pairs(word_ids, rng, 2000)
    assert decoding.two_vs_two(T + 0.1 * rng.standard_normal(T.shape), T, i, j).mean() > 0.95
    assert abs(decoding.two_vs_two(rng.standard_normal(T.shape), T, i, j).mean() - 0.5) < 0.05
    ranks, top = decoding.retrieval_ranks(T, T, word_ids)
    assert (ranks == 1).all() and (top == word_ids).all()
    lengths = np.repeat([3, 4, 5, 6], 50)
    zipf = rng.uniform(3, 4, 200)
    i, j = decoding.sample_pairs(word_ids, rng, 500, lengths, zipf, max_zipf_gap=0.3)
    assert len(i) > 100 and (lengths[i] == lengths[j]).all() and (np.abs(zipf[i] - zipf[j]) <= 0.3).all()


def test_decode_script_detects_lexical_but_not_beyond(tmp_path, monkeypatch):
    cache = zuco_cache(tmp_path)
    runner = load_script("decode_eeg_to_text")
    monkeypatch.setattr(runner, "static_word_vectors", fake_static())
    out = os.path.join(str(tmp_path), "decoding")
    monkeypatch.setattr(sys, "argv", ["x", "--dataset", "zuco", "--word-eeg-dir", cache, "--out-dir", out,
                                      "--n-pairs", "4000", "--n-boot", "200", "--device", "cpu"])
    runner.main()
    summary = json.load(open(os.path.join(out, "decoding_zuco.json")))
    pairs, matched = summary["two_vs_two"]["pairs"], summary["two_vs_two"]["matched"]
    assert pairs["eeg"][0] > 0.6 and pairs["eeg - shuffled_eeg"][1] > 0  # EEG tracks word length here
    assert abs(matched["eeg"][0] - 0.5) < 0.1  # nothing beyond length
    beyond = pairs["word_features+eeg - word_features"]
    assert beyond[1] <= 0.01, beyond  # EEG only repeats word length, which word features already have
    assert set(summary["sentiment"]) >= {"real_text", "eeg", "shuffled_eeg"}
    assert os.path.exists(os.path.join(out, "decoded_examples_zuco.csv"))


def test_crosslingual_script_runs(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_encoding_scan import write_teco

    cache = zuco_cache(tmp_path)
    trt, labels, _, _ = write_teco(str(tmp_path), n_subjects=4, n_sentences=30)
    runner = load_script("crosslingual_subspaces")
    monkeypatch.setattr(runner, "extract_layer_vectors", fake_extractor())
    out = os.path.join(str(tmp_path), "crosslingual")
    monkeypatch.setattr(sys, "argv", ["x", "--zuco-word-eeg-dir", cache, "--teco-trt-dir", trt, "--teco-labels-csv",
                                      labels, "--out-dir", out, "--k", "4", "--n-null", "3", "--n-boot", "100",
                                      "--layer", "1", "--device", "cpu"])
    runner.main()
    summary = json.load(open(os.path.join(out, "crosslingual.json")))
    assert 0 <= summary["overlap"]["real"] <= 1 and set(summary["transfer"]) == {"fa", "en"}
    assert "Shared EEG directions" in open(os.path.join(out, "crosslingual.md")).read()
