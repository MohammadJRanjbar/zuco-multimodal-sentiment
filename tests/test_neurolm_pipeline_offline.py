"""End-to-end wiring test on synthetic .mat files with a stub encoder.

Runs the real scripts: channel mapping -> extraction (stub encoder instead of
the checkpoint) -> probe (--smoke) -> sanity (--smoke) -> report.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from src.features import feature_names

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBJECTS = ["ZAA", "ZBB", "ZCC", "ZDD", "ZEE", "ZFF"]
N_SENTENCES = 30


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubEncoder:
    """Stands in for NeuroLMEncoder: hidden state = patch statistics + noise."""

    def __init__(self, *args, **kwargs):
        self.checkpoint_info = {"checkpoint_name": "stub.pt", "checkpoint_sha256": "0" * 64, "n_parameters": 0}
        self.representations = ["tokenizer", "gpt"]
        self.rng = np.random.default_rng(0)

    def encode_chunks(self, chunks, batch_size=16):
        out = []
        for x, chans, _ in chunks:
            base = np.stack([x.mean(1), x.std(1), chans / 100.0, np.abs(x).max(1)], axis=1)
            hidden = np.concatenate([base, self.rng.standard_normal((len(x), 4))], axis=1).astype(np.float32)
            out.append({"tokenizer": hidden, "gpt": hidden[:, ::-1].copy()})
        return out


def build_dataset(tmp):
    rng = np.random.default_rng(0)
    labels = np.tile([-1, 0, 1], N_SENTENCES // 3)
    sentences = [f"synthetic sentence number {i}" for i in range(N_SENTENCES)]
    pd.DataFrame({"sentence_id": np.arange(N_SENTENCES), "sentence": sentences,
                  "sentiment_label": labels}).to_csv(os.path.join(tmp, "labels.csv"), index=False)
    mat_dir = os.path.join(tmp, "mat")
    hand_dir = os.path.join(tmp, "hand")
    os.makedirs(mat_dir)
    os.makedirs(hand_dir)
    names = feature_names(105)
    json.dump(names, open(os.path.join(hand_dir, "feature_names.json"), "w"))
    for subject in SUBJECTS:
        records = np.zeros((N_SENTENCES,), dtype=[("content", "O"), ("rawData", "O")])
        for i, text in enumerate(sentences):
            n = int(rng.integers(560, 900))  # 1.1-1.8 s at 500 Hz
            raw = (8 * rng.standard_normal((105, n))).astype(np.float32)
            raw[-1] = 0.0
            raw[:10] += 3 * labels[i]
            records[i] = (text, raw)
        savemat(os.path.join(mat_dir, f"results{subject}_SR.mat"), {"sentenceData": records})
        np.savez_compressed(os.path.join(hand_dir, f"{subject}.npz"),
                            X=rng.standard_normal((N_SENTENCES, len(names))).astype(np.float32),
                            sentence_id=np.arange(N_SENTENCES), label=labels)
    return mat_dir, hand_dir, os.path.join(tmp, "labels.csv")


def test_offline_pipeline(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    tmp = str(tmp_path)
    mat_dir, hand_dir, labels_csv = build_dataset(tmp)
    config = os.path.join(tmp, "config.yaml")
    with open(config, "w") as handle:
        handle.write(
            f"base: {os.path.join(ROOT, 'configs', 'neurolm_probe.yaml')}\n"
            "probe:\n  seeds: [42, 52]\n  bootstrap: {n_boot: 50, n_perm: 50}\n"
            "  protocols:\n    joint: {n_subject_folds: 3, n_sentence_folds: 3, val_fraction: 0.2, n_val_subjects: 1}\n"
            "    text: {n_folds: 3, val_fraction: 0.2}\n    subject: {n_val_subjects: 1}\n"
        )
    mapping_dir = os.path.join(tmp, "mapping")
    cache_dir = os.path.join(tmp, "cache")
    results_dir = os.path.join(tmp, "results")
    reports = os.path.join(tmp, "reports")

    monkeypatch.setattr(sys, "argv", ["x", "--config", config, "--out-dir", mapping_dir,
                                      "--report", os.path.join(reports, "channel_mapping.md")])
    load_script("check_channel_mapping").main()
    assert os.path.exists(os.path.join(mapping_dir, "mapping_tenten6_chunk1024.json"))

    extract = load_script("extract_neurolm_features")
    monkeypatch.setattr(extract, "NeuroLMEncoder", StubEncoder)
    monkeypatch.setattr(sys, "argv", ["x", "--config", config, "--mat-dir", mat_dir, "--labels-csv", labels_csv,
                                      "--checkpoint", "stub.pt", "--neurolm-dir", tmp, "--cache-dir", cache_dir,
                                      "--mapping-dir", mapping_dir, "--device", "cpu"])
    extract.main()
    views = sorted(os.listdir(cache_dir))
    assert len(views) == 4
    meta = pd.read_csv(os.path.join(cache_dir, views[0], "metadata.csv"))
    assert len(meta) == len(SUBJECTS) * N_SENTENCES
    expected = {"sample_id", "subject_id", "sentence_id", "label", "duration_s", "num_channels",
                "language", "embedding_path"}
    assert expected <= set(meta.columns)

    common = ["--config", config, "--cache-dir", cache_dir, "--handcrafted-dir", hand_dir,
              "--results-dir", results_dir, "--run-tag", "smoke"]
    monkeypatch.setattr(sys, "argv", ["x", *common, "--smoke"])
    load_script("run_neurolm_probe").main()
    monkeypatch.setattr(sys, "argv", ["x", *common, "--smoke"])
    load_script("run_neurolm_sanity").main()
    run_dir = os.path.join(results_dir, "smoke")
    predictions = pd.read_csv(os.path.join(run_dir, "joint", "tenten6_chunk1024__tokenizer__mean__logreg",
                                           "predictions_seed_42.csv"))
    assert list(predictions.columns[:5]) == ["sample_id", "subject_id", "sentence_id", "true_label", "predicted_label"]
    assert {"prob_positive", "prob_neutral", "prob_negative"} <= set(predictions.columns)

    monkeypatch.setattr(sys, "argv", ["x", "--config", config, "--run-dir", run_dir,
                                      "--report", os.path.join(reports, "results.md")])
    load_script("build_neurolm_report").main()
    text = open(os.path.join(reports, "results.md")).read()
    assert "Q5." in text and "Recommendation" in text
