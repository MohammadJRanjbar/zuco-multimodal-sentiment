"""Diagnostics on synthetic word EEG that encodes word length but not valence."""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from src.diagnostics import signal
from src.diagnostics.targets import build_word_table
from src.fusion.word_eeg import save_subject

ROOT = os.path.join(os.path.dirname(__file__), "..")
VOCAB = ["the", "a", "film", "wonderful", "terrible", "plot", "actors", "boring", "brilliant", "is", "was",
         "story", "dull", "great", "awful", "scene", "of", "and", "music", "bad"]
LEXICON = {"wonderful": 3.1, "terrible": -2.9, "boring": -1.8, "brilliant": 2.8, "dull": -1.7, "great": 3.0,
           "awful": -2.6, "bad": -2.5}


def word_trials(n_readers=6, n_sentences=36, seed=0):
    rng = np.random.default_rng(seed)
    sentences = [list(rng.choice(VOCAB, size=rng.integers(5, 9))) for _ in range(n_sentences)]
    labels = np.tile([-1, 0, 1], n_sentences // 3)
    trials = []
    for r in range(n_readers):
        offset = rng.normal(0, 1.0)
        for s, words in enumerate(sentences):
            n = len(words)
            features = np.exp(rng.normal(offset, 0.4, size=(n, 8, 105))).astype(np.float32)
            lengths = np.array([len(w) for w in words], dtype=float)
            features[:, 2, :12] *= np.exp(0.15 * lengths)[:, None]  # alpha band tracks word length
            fixated = rng.random(n) > 0.15
            features[~fixated] = np.nan
            trials.append({"sample_id": f"R{r}_{s:04d}", "subject_id": f"R{r}", "sentence_id": s,
                           "label": int(labels[s]), "words": [str(w) for w in words], "features": features,
                           "fixations": fixated.astype(np.float32),
                           "trt_ms": np.where(fixated, 120 + 25 * lengths, np.nan).astype(np.float32),
                           "ffd_ms": np.full(n, np.nan, np.float32), "gd_ms": np.full(n, np.nan, np.float32)})
    return trials, sentences, labels


def write_cache(tmp_path, trials):
    cache = os.path.join(tmp_path, "word_eeg")
    for reader in sorted({t["subject_id"] for t in trials}):
        save_subject(cache, reader, [t for t in trials if t["subject_id"] == reader])
    return cache


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_signal_components_find_controls_not_valence():
    trials, _, _ = word_trials()
    meta, X, info = signal.long_word_table(trials)
    assert info["log_transformed"] and X.shape[1] == 8 * 104
    variance = signal.variance_components(meta, X)
    assert np.allclose(variance.sum(axis=1), 1.0, atol=0.05)
    Z = signal.zscore_per_reader(meta, X)
    reliability, details = signal.split_half_reliability(meta, Z, n_splits=5)
    alpha = signal.band_slices(104)["a1"]
    assert reliability["reliability_all_readers"].to_numpy()[alpha][:12].mean() > 0.5
    assert np.median(reliability["reliability_all_readers"].to_numpy()) < 0.3
    items, A = signal.reader_average(meta, Z)
    table = items.merge(build_word_table(trials, lexicon=LEXICON, frequencies=False),
                        on=["sentence_id", "word_index"])
    length = signal.ridge_probe(A, table["length"].to_numpy(float), items["sentence_id"], "regression",
                                n_perm=10, perm_units=items["item"])
    assert length["score"] > 0.3 and length["p_value"] <= 1 / 11 + 1e-9
    valence = signal.ridge_probe(A, table["valence"].to_numpy(float), items["sentence_id"], "regression",
                                 n_perm=10, perm_units=items["item"])
    assert valence["score"] < length["score"]


def test_full_diagnostics_pipeline(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    sys.path.insert(0, os.path.dirname(__file__))
    from test_fusion import WordTokenizer, tiny_lm

    trials, sentences, labels = word_trials()
    tmp = str(tmp_path)
    cache = write_cache(tmp, trials)
    labels_csv = os.path.join(tmp, "labels.csv")
    pd.DataFrame({"sentence_id": range(len(sentences)), "sentence": [" ".join(s) for s in sentences],
                  "sentiment_label": labels}).to_csv(labels_csv, index=False)

    # Stages 1-2
    stage1 = load_script("analyze_eeg_signal")
    monkeypatch.setattr(stage1, "valence_lexicon", lambda: LEXICON)
    signal_dir = os.path.join(tmp, "signal")
    monkeypatch.setattr(sys, "argv", ["x", "--word-eeg-dir", cache, "--labels-csv", labels_csv, "--out-dir", signal_dir,
                                      "--surprisal-model", "none", "--n-perm", "5", "--n-perm-single", "3"])
    stage1.main()
    probes = pd.read_csv(os.path.join(signal_dir, "word_probes.csv"))
    averaged = probes[probes["level"] == "reader-averaged"].set_index("target")
    assert averaged.loc["length", "score"] > averaged.loc["length", "null_q95"]
    assert os.path.exists(os.path.join(signal_dir, "plots", "reliability_all_readers.png"))
    assert os.path.exists(os.path.join(signal_dir, "stage2_representations.csv"))

    # Train a tiny fusion model with saved weights
    runner = load_script("run_eeg_text_lora")
    monkeypatch.setattr(runner, "load_language_model", lambda *a: (tiny_lm(), WordTokenizer()))
    results = os.path.join(tmp, "lora")
    monkeypatch.setattr(sys, "argv", ["x", "--config", os.path.join(ROOT, "configs", "eeg_text_lora.yaml"),
                                      "--word-eeg-dir", cache, "--results-dir", results, "--run-tag", "run",
                                      "--folds", "1", "--epochs", "1", "--device", "cpu",
                                      "--report", os.path.join(tmp, "lora.md")])
    runner.main()
    run_dir = os.path.join(results, "run")
    assert os.path.exists(os.path.join(run_dir, "text_eeg", "fold_0_weights.pt"))

    # Stage 3 (eager attention so attention weights are returned)
    stage3 = load_script("analyze_fusion_model")

    def eager_loader(*_args):
        lm = tiny_lm()
        lm.config._attn_implementation = "eager"
        return lm, WordTokenizer()

    monkeypatch.setattr(stage3, "load_for_analysis", eager_loader)
    fusion_dir = os.path.join(tmp, "fusion")
    monkeypatch.setattr(sys, "argv", ["x", "--word-eeg-dir", cache, "--lora-run-dir", run_dir, "--out-dir", fusion_dir,
                                      "--word-targets", os.path.join(signal_dir, "word_targets.csv"),
                                      "--max-trials", "40", "--device", "cpu", "--n-perm", "2"])
    stage3.main()
    summary = json.load(open(os.path.join(fusion_dir, "stage3_summary.json")))
    variants = {row["variant"] for row in summary["reliance"]}
    assert {"aligned", "shuffled_within_reader", "other_reader_same_sentence", "fixation_only"} <= variants
    assert abs(sum(summary["attribution_mean_share"][k] for k in ("prompt", "word", "eeg", "answer")) - 1) < 1e-4
    attention = pd.read_csv(os.path.join(fusion_dir, "attention_by_layer.csv"))
    assert {"eeg", "word"} <= set(attention["role"])

    # Stage 4
    stage4 = load_script("analyze_errors")
    errors_dir = os.path.join(tmp, "errors")
    monkeypatch.setattr(sys, "argv", ["x", "--lora-run-dir", run_dir, "--labels-csv", labels_csv,
                                      "--word-eeg-dir", cache, "--out-dir", errors_dir, "--n-perm", "5",
                                      "--n-boot", "50"])
    stage4.main()
    subsets = pd.read_csv(os.path.join(errors_dir, "subset_effects.csv"))
    assert "all" in set(subsets["subset"])

    report = load_script("build_diagnostics_report")
    path = os.path.join(tmp, "diagnostics.md")
    monkeypatch.setattr(sys, "argv", ["x", "--signal-dir", signal_dir, "--fusion-dir", fusion_dir,
                                      "--errors-dir", errors_dir, "--report", path])
    report.main()
    text = open(path).read()
    for heading in ("Stage 1", "Stage 2", "Stage 3", "Stage 4", "Where it fails"):
        assert heading in text
