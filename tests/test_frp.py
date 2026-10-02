import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from src.followup import frp
from src.fusion.word_eeg import N_CHANNELS, load_word_eeg

ROOT = os.path.join(os.path.dirname(__file__), "..")
SENTENCES = [f"word{s} alpha beta gamma delta epsilon zeta eta" for s in range(24)]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def surprisal_of(sentence_index, word_index):
    return float((sentence_index * 7 + word_index * 3) % 5)


def synthetic_sentence(rng, s, words, offset_segments=False):
    """Sentence EEG with a lambda response (occipital, +100 ms) and an N400 scaled by surprisal."""
    labels = frp.zuco_channel_labels()
    occipital = frp.channel_indices(frp.OCCIPITAL, labels)
    parietal = frp.channel_indices(frp.CENTRO_PARIETAL, labels)
    T = 4000
    raw = rng.normal(0, 1.0, size=(N_CHANNELS, T))
    raw[-1] = 0.0  # Cz reference
    times = np.arange(T)
    segments, ffd, t = [], [], 200
    for w in range(len(words)):
        if w == 3:  # skipped word
            segments.append([])
            ffd.append(None)
            continue
        word_segments = []
        for _ in range(2 if w == 5 else 1):
            length = int(rng.integers(95, 125))
            bump = np.exp(-0.5 * ((times - (t + 50)) / 8.0) ** 2)
            raw[occipital] += 6.0 * bump
            n400 = np.exp(-0.5 * ((times - (t + 200)) / 25.0) ** 2)
            raw[parietal] -= 1.5 * surprisal_of(s, w) * n400 if not word_segments else 0.0
            word_segments.append((t, length))
            t += length + 15
        segments.append(word_segments)
        ffd.append(word_segments[0][1] * 2.0)
    out = []
    for word_segments in segments:
        cells = []
        for start, length in word_segments:
            segment = raw[:, start:start + length].copy()
            if offset_segments:
                segment += rng.normal(0, 5, size=(N_CHANNELS, 1))
            cells.append(segment)
        out.append(cells)
    return raw, out, ffd


def write_v5(path, rng):
    from scipy.io import savemat

    fields = ["content", "nFixations", "TRT", "FFD", "GD", "rawEEG"]
    sentences = np.zeros((len(SENTENCES),), dtype=[("content", "O"), ("rawData", "O"), ("word", "O")])
    for s, text in enumerate(SENTENCES):
        words = text.split(" ")
        raw, segments, ffd = synthetic_sentence(rng, s, words, offset_segments=(s % 2 == 1))
        array = np.zeros((len(words),), dtype=[(f, "O") for f in fields])
        for w, token in enumerate(words):
            empty = np.zeros((0, 0))
            cell = np.empty((len(segments[w]),), dtype=object)
            for i, segment in enumerate(segments[w]):
                cell[i] = segment
            array[w] = (token, float(len(segments[w])), 250.0 if segments[w] else empty,
                        ffd[w] if ffd[w] else empty, 230.0 if segments[w] else empty, cell if len(cell) else empty)
        sentences[s] = (text, raw, array)
    savemat(path, {"sentenceData": sentences})


def write_hdf5(path, rng):
    import h5py

    sys.path.insert(0, os.path.dirname(__file__))
    from test_word_eeg import matlab_empty, matlab_string

    ref_dtype = h5py.special_dtype(ref=h5py.Reference)
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        counter = [0]

        def name():
            counter[0] += 1
            return f"r{counter[0]}"

        content_refs, word_refs, raw_refs = [], [], []
        for s, text in enumerate(SENTENCES):
            words = text.split(" ")
            raw, segments, ffd = synthetic_sentence(rng, s, words)
            content_refs.append(matlab_string(refs, name(), text))
            raw_refs.append(refs.create_dataset(name(), data=raw.T).ref)
            group = refs.create_group(name())
            group.create_dataset("content", data=np.array([[matlab_string(refs, name(), w)] for w in words],
                                                          dtype=ref_dtype))
            cells = []
            for word_segments in segments:
                if not word_segments:
                    cells.append([matlab_empty(refs, name())])
                    continue
                inner = [[refs.create_dataset(name(), data=segment.T).ref] for segment in word_segments]
                cells.append([refs.create_dataset(name(), data=np.array(inner, dtype=ref_dtype)).ref])
            group.create_dataset("rawEEG", data=np.array(cells, dtype=ref_dtype))
            for measure, values in (("TRT", [250.0 if seg else None for seg in segments]), ("FFD", ffd),
                                    ("GD", [230.0 if seg else None for seg in segments])):
                group.create_dataset(measure, data=np.array(
                    [[refs.create_dataset(name(), data=np.array([[v]])).ref if v else matlab_empty(refs, name())]
                     for v in values], dtype=ref_dtype))
            group.create_dataset("nFixations", data=np.array(
                [[refs.create_dataset(name(), data=np.array([[float(len(seg))]])).ref] for seg in segments],
                dtype=ref_dtype))
            word_refs.append(group.ref)
        data = f.create_group("sentenceData")
        for key, values in (("content", content_refs), ("word", word_refs), ("rawData", raw_refs)):
            data.create_dataset(key, data=np.array([[r] for r in values], dtype=ref_dtype))


def labels_csv(tmp_path):
    path = os.path.join(tmp_path, "labels.csv")
    pd.DataFrame({"sentence_id": range(len(SENTENCES)), "sentence": SENTENCES,
                  "sentiment_label": np.tile([-1, 0, 1], len(SENTENCES) // 3)}).to_csv(path, index=False)
    return path


def test_locate_segment_exact_offset_and_unmatched():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(N_CHANNELS, 2000))
    segment = raw[:, 700:810].copy()
    assert frp.locate_segment(raw, segment) == (700, "exact")
    assert frp.locate_segment(raw, segment + rng.normal(0, 3, size=(N_CHANNELS, 1))) == (700, "offset")
    assert frp.locate_segment(raw, rng.normal(size=(N_CHANNELS, 110)))[0] is None


def test_epoch_window_means_and_timing():
    x = np.zeros((N_CHANNELS, 1000))
    x[:, 300 + 175:300 + 225] = 2.0  # +350..+450 ms after an onset at sample 300
    features, epochs, timing = frp.epoch_words(x, [[300], [], [10]], [[100], [], [80]])
    assert features.shape == (3, 8, N_CHANNELS) and len(epochs) == 1
    n400 = features[0, frp.N400_INDEX, 0]
    assert abs(n400 - 2.0 * 50 / 100) < 1e-6  # 50 of the 100 N400-window samples are raised
    assert np.isnan(features[1]).all() and np.isnan(features[2]).all()  # skipped word; onset too early
    assert timing["first_fix_ms"][0] == 200 and timing["next_fix_ms"][2] == (300 - 10) * 2


@pytest.mark.parametrize("fmt", ["v5", "hdf5"])
def test_extract_and_analyze_frp(tmp_path, monkeypatch, fmt):
    if fmt == "hdf5":
        pytest.importorskip("h5py")
    rng = np.random.default_rng(1)
    mat_dir = os.path.join(tmp_path, "mat")
    os.makedirs(mat_dir)
    for subject in ("ZAA", "ZAB", "ZAC"):
        path = os.path.join(mat_dir, f"results{subject}_SR.mat")
        (write_v5 if fmt == "v5" else write_hdf5)(path, rng)
    labels = labels_csv(str(tmp_path))
    out = os.path.join(tmp_path, "frp")
    extractor = load_script("extract_frp")
    monkeypatch.setattr(sys, "argv", ["x", "--mat-dir", mat_dir, "--labels-csv", labels, "--out-dir", out])
    extractor.main()
    report = json.load(open(os.path.join(out, "frp_extraction.json")))
    for summary in report["subjects"].values():
        assert summary["match_rate"] == 1.0
        assert summary["ffd_vs_segment_r"] > 0.99
    trials = load_word_eeg(out)
    assert trials[0]["features"].shape == (8, 8, N_CHANNELS)
    assert np.isnan(trials[0]["features"][3]).all()  # skipped word

    analyzer = load_script("analyze_frp")
    monkeypatch.setattr(analyzer, "lm_surprisal",
                        lambda sentences, *a, **k: [np.array([surprisal_of(int(ws[0][4:]), w) for w in range(len(ws))])
                                                    for ws in sentences])
    results = os.path.join(tmp_path, "analysis")
    monkeypatch.setattr(sys, "argv", ["x", "--frp-dir", out, "--out-dir", results, "--n-boot", "200"])
    analyzer.main()
    text = open(os.path.join(results, "frp_report.md")).read()
    assert "**Pass:**" in text, text
    coefficients = pd.read_csv(os.path.join(results, "n400_regression.csv")).set_index("predictor")
    assert coefficients.loc["surprisal", "beta_uv_per_sd"] < 0 and coefficients.loc["surprisal", "excludes_zero"]
    assert os.path.exists(os.path.join(results, "plots", "frp_grand_average.png"))


def test_sentence_sentiment_script(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    mat_dir = os.path.join(tmp_path, "mat")
    os.makedirs(mat_dir)
    for subject in ("ZAA", "ZAB", "ZAC"):
        write_v5(os.path.join(mat_dir, f"results{subject}_SR.mat"), rng)
    labels = labels_csv(str(tmp_path))
    out = os.path.join(tmp_path, "frp")
    extractor = load_script("extract_frp")
    monkeypatch.setattr(sys, "argv", ["x", "--mat-dir", mat_dir, "--labels-csv", labels, "--out-dir", out])
    extractor.main()
    trials_csv = os.path.join(tmp_path, "word_eeg_trials.csv")
    pd.DataFrame([{"subject_id": s, "position": i, "sentence_id": i, "status": "ok"}
                  for s in ("ZAA", "ZAB", "ZAC") for i in range(len(SENTENCES))]).to_csv(trials_csv, index=False)
    runner = load_script("frp_sentence_sentiment")
    monkeypatch.setattr(runner, "lm_surprisal", lambda sentences, *a, **k: [np.ones(len(s)) for s in sentences])
    results = os.path.join(tmp_path, "sentence")
    monkeypatch.setattr(sys, "argv", ["x", "--frp-dir", out, "--out-dir", results, "--trials-csv", trials_csv,
                                      "--n-perm", "50"])
    runner.main()
    table = pd.read_csv(os.path.join(results, "sentence_sentiment.csv"))
    assert {"FRP", "text form", "presentation order"} <= set(table["probe"])
    assert any(p.startswith("FRP with form") for p in table["probe"])
    assert (table["rows"] == "single reader").sum() == 1
    assert "## Reading" in open(os.path.join(results, "sentence_sentiment.md")).read()
