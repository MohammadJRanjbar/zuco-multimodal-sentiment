import importlib.util
import json
import os
import sys
import zlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.diagnostics import signal  # noqa: E402
from src.fusion.word_eeg import load_word_eeg, save_subject  # noqa: E402
from src.wordinfo import representations as reps  # noqa: E402
from src.wordinfo.epochs import load_subject_epochs, word_epochs  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(__file__))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def word_vector(word, dim=24):
    return np.random.default_rng(zlib.crc32(word.encode())).standard_normal(dim)


def fake_static(sentences, item_sentence, item_word, model_name, **kwargs):
    out = np.stack([word_vector(sentences[s][w]) for s, w in zip(item_sentence, item_word)]).astype(np.float32)
    return out, np.ones(len(out), bool)


class FakeCBraMod:
    """Stand-in for the pretrained model: per-channel mean and std of the epoch, tiled to 200 values."""

    def __init__(self, device="cpu", batch_size=256, model=None):
        self.regions = reps.scalp_regions()

    def __call__(self, epochs):
        x = np.asarray(epochs, dtype=np.float32)
        feats = np.repeat(np.stack([x.mean(axis=2), x.std(axis=2)], axis=2), 100, axis=2)  # [n, C, 200]
        return reps.region_pool(feats, self.regions)


@pytest.fixture(scope="module")
def epoch_dirs(tmp_path_factory):
    from test_frp import labels_csv, write_v5

    tmp = str(tmp_path_factory.mktemp("wordinfo"))
    rng = np.random.default_rng(3)
    mat_dir = os.path.join(tmp, "mat")
    os.makedirs(mat_dir)
    for subject in ("ZAA", "ZAB", "ZAC"):
        write_v5(os.path.join(mat_dir, f"results{subject}_SR.mat"), rng)
    labels = labels_csv(tmp)
    epochs_dir, frp_dir = os.path.join(tmp, "epochs"), os.path.join(tmp, "frp")
    argv = sys.argv
    try:
        sys.argv = ["x", "--mat-dir", mat_dir, "--labels-csv", labels, "--out-dir", epochs_dir]
        load_script("extract_word_epochs").main()
        sys.argv = ["x", "--mat-dir", mat_dir, "--labels-csv", labels, "--out-dir", frp_dir]
        load_script("extract_frp").main()
    finally:
        sys.argv = argv
    return tmp, epochs_dir, frp_dir


def test_word_epochs_cut_one_second_around_first_fixations():
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 1, (105, 5000))
    raw[-1] = 0
    epochs, index = word_epochs(raw, [[50], [], [600, 900], [4900]])  # 0.1 s: too early; 9.8 s: too late
    assert epochs.shape == (1, 105, 200) and index.tolist() == [2]


def test_extraction_keeps_fixated_words_with_aligned_lambda(epoch_dirs):
    _, epochs_dir, _ = epoch_dirs
    report = json.load(open(os.path.join(epochs_dir, "word_epochs.json")))
    assert set(report["subjects"]) == {"ZAA", "ZAB", "ZAC"} and report["sfreq"] == 200.0
    arrays, trials = load_subject_epochs(os.path.join(epochs_dir, "ZAB.npz"))
    assert arrays["epochs"].shape == (24 * 7, 105, 200) and arrays["epochs"].dtype == np.float16
    word_in_sentence = arrays["epoch_word"] - np.repeat([t["word_offset"] for t in trials], 7)
    assert 3 not in set(word_in_sentence)  # the skipped word has no epoch
    from src.followup import frp

    occipital = frp.channel_indices(frp.OCCIPITAL, frp.zuco_channel_labels())
    response = arrays["epochs"][:, occipital].astype(np.float32).mean(axis=(0, 1))
    peak = int(np.argmax(response[40:120]))  # 0..400 ms
    assert 10 <= peak <= 30, peak  # lambda response about 100 ms after fixation onset


def test_build_representations(epoch_dirs, monkeypatch):
    tmp, epochs_dir, frp_dir = epoch_dirs
    builder = load_script("build_word_representations")
    monkeypatch.setattr(builder.reps, "CBraModFeatures", FakeCBraMod)
    out = os.path.join(tmp, "reps")
    monkeypatch.setattr(sys, "argv", ["x", "--epochs-dir", epochs_dir, "--out-root", out, "--representations",
                                      "raw", "cbramod", "eye_tracking", "frp", "--frp-dir", frp_dir, "--device", "cpu"])
    builder.main()
    dims = {"raw": 105 * 16, "cbramod": 6 * 200, "eye_tracking": 4}
    for name, dim in dims.items():
        trials = load_word_eeg(os.path.join(out, name))
        assert len(trials) == 3 * 24 and trials[0]["features"].shape == (8, dim)
        assert np.isnan(trials[0]["features"][3]).all() and np.isfinite(trials[0]["features"][0]).all()
        meta, X, info = signal.long_word_table(trials)
        assert X.shape[1] == dim and not info["log_transformed"] and info["n_bands"] == 0
    frp_trials = load_word_eeg(os.path.join(out, "frp"))
    assert frp_trials[0]["features"].shape == (8, 8, 105) and np.isnan(frp_trials[0]["features"][3]).all()
    summary = json.load(open(os.path.join(out, "representations.json")))
    assert summary["frp"]["subjects"]["ZAA"]["word_count_mismatches"] == 0
    builder.main()  # resumable: nothing left to do


def informative_cache(tmp, n_sentences=60, readers=4, dim=30, noise=0.5):
    """A representation whose features are a linear image of the word's embedding plus noise."""
    rng = np.random.default_rng(5)
    vocab = [f"w{i}" for i in range(50)]
    mix = rng.standard_normal((24, dim))
    sentences = [list(rng.choice(vocab, size=int(rng.integers(6, 9)))) for _ in range(n_sentences)]
    out = os.path.join(tmp, "informative")
    for r in range(readers):
        trials = []
        for s, words in enumerate(sentences):
            features = np.stack([word_vector(w) @ mix for w in words]) + noise * rng.standard_normal((len(words), dim))
            features[rng.random(len(words)) < 0.15] = np.nan
            trials.append({"sample_id": f"R{r}_{s}", "sentence_id": s, "label": s % 3 - 1, "words": words,
                           "features": features.astype(np.float32), "fixations": np.ones(len(words), np.float32)})
        save_subject(out, f"R{r}", trials)
    return out


def test_word_information_and_summary(tmp_path, monkeypatch):
    cache = informative_cache(str(tmp_path))
    results = os.path.join(str(tmp_path), "results")
    info = load_script("word_information")
    monkeypatch.setattr(info, "static_word_vectors", fake_static)
    monkeypatch.setattr(sys, "argv", ["x", "--word-eeg-dir", cache, "--tag", "informative", "--out-dir", results,
                                      "--surprisal-model", "none", "--n-boot", "100", "--device", "cpu"])
    info.main()
    s = json.load(open(os.path.join(results, "word_info_informative.json")))
    assert s["embedding"]["eeg"]["centred_cosine"][0] > 0.5 > abs(s["embedding"]["shuffled_eeg"]["centred_cosine"][0])
    ident = s["sentence_identification"]
    assert ident["eeg"]["accuracy"] > 0.9 and ident["eeg"]["chance"] < 0.5
    assert ident["eeg - shuffled_eeg"][1] > 0
    decode = load_script("decode_eeg_to_text")
    monkeypatch.setattr(decode, "static_word_vectors", fake_static)
    monkeypatch.setattr(sys, "argv", ["x", "--dataset", "zuco", "--word-eeg-dir", cache, "--tag", "informative",
                                      "--out-dir", results, "--n-pairs", "2000", "--n-boot", "100", "--device", "cpu"])
    decode.main()
    monkeypatch.setattr(sys, "argv", ["x", "--results-dir", results])
    load_script("summarize_word_information").main()
    table = open(os.path.join(results, "word_information_summary.md")).read()
    assert "| informative | 30 |" in table


class TinyEncoder(torch.nn.Module):
    """Stand-in for CBraMod + head in the fine-tuning script."""

    def __init__(self, target_dim):
        super().__init__()
        self.backbone = torch.nn.Linear(105, 16)
        self.head = torch.nn.Sequential(torch.nn.Linear(16, target_dim))
        self.log_scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return self.head(self.backbone(x.mean(dim=2)))


def test_finetune_script_runs_real_and_shuffled(epoch_dirs, monkeypatch):
    tmp, epochs_dir, _ = epoch_dirs
    ft = load_script("finetune_eeg_encoder")
    monkeypatch.setattr(ft.Head, "build", staticmethod(lambda target_dim, device: TinyEncoder(target_dim)))
    monkeypatch.setattr(ft, "static_word_vectors", fake_static)
    out = os.path.join(tmp, "finetuned")
    monkeypatch.setattr(sys, "argv", ["x", "--epochs-dir", epochs_dir, "--out-root", out, "--folds", "3",
                                      "--max-epochs", "2", "--batch-size", "64", "--device", "cpu"])
    ft.main()
    report = json.load(open(os.path.join(out, "cbramod_ft_training.json")))
    assert set(report["runs"]) == {"cbramod_ft", "cbramod_ft_shuffled"}
    for run in report["runs"]:
        trials = load_word_eeg(os.path.join(out, run))
        assert len(trials) == 3 * 24 and trials[0]["features"].shape == (8, 24)
        assert np.isnan(trials[0]["features"][3]).all() and np.isfinite(trials[0]["features"][0]).all()
