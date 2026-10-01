import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

from src.brainshaping import static
from src.brainshaping.data import arm_targets, fit_targets

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_subspace_recovers_planted_eeg_direction():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 64))
    w = rng.standard_normal(64)
    Y = (X @ w)[:, None] + 0.5 * rng.standard_normal((2000, 1))
    mean, std = static.standardizer(X)
    Xs = (X - mean) / std
    U, ridge = static.fit_subspace(Xs, Y)
    direction = (w * std) / np.linalg.norm(w * std)
    assert abs(float(U[:, 0] @ direction)) > 0.95
    assert static.encoding_r2(ridge, Xs, Y) > 0.8
    shuffled = arm_targets("shuffled_eeg", Y.astype(np.float32), np.arange(2000), rng)
    U_shuffled, _ = static.fit_subspace(Xs, shuffled)
    assert abs(float(U_shuffled[:, 0] @ direction)) < 0.5


def test_shaping_and_selection():
    rng = np.random.default_rng(1)
    S = rng.standard_normal((300, 16))
    y = (S[:, 0] > 0.5).astype(int) + (S[:, 0] > -0.5).astype(int)
    U = np.linalg.qr(rng.standard_normal((16, 2)))[0]
    np.testing.assert_allclose(static.shape(S, U, 0.0), S)
    np.testing.assert_allclose(static.shape(S, None, 3.0), S)
    probs, choice = static.select_and_predict(S[:200], y[:200], S[200:250], y[200:250], S[250:], U)
    assert probs.shape == (50, 3) and choice["beta"] in static.BETAS
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-6)


def test_targets_and_arms():
    rng = np.random.default_rng(2)
    eeg = rng.standard_normal((500, 40)).astype(np.float32)
    transform, explained = fit_targets(eeg[:400], k=8)
    targets = transform(eeg)
    assert targets.shape == (500, 8) and 0 < explained <= 1
    np.testing.assert_allclose(targets[:400].std(axis=0, ddof=1), 1.0, atol=1e-4)
    train = np.arange(400)
    shuffled = arm_targets("shuffled_eeg", targets, train, rng)
    np.testing.assert_array_equal(shuffled[400:], targets[400:])
    assert sorted(map(tuple, shuffled[:400].round(5))) == sorted(map(tuple, targets[:400].round(5)))
    assert arm_targets("random_targets", targets, train, rng).shape == targets.shape


def test_static_script_end_to_end(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_diagnostics import word_trials, write_cache

    trials, _, _ = word_trials(n_readers=4, n_sentences=60)
    cache = write_cache(str(tmp_path), trials)
    runner = load_script("run_eeg_shaped_embeddings")
    rng = np.random.default_rng(0)
    table = {w: rng.standard_normal(24) for w in {w for t in trials for w in t["words"]}}

    def fake_vectors(sentences, *args, **kwargs):
        return [np.stack([table[w] + 0.1 * rng.standard_normal(24) for w in words]).astype(np.float32)
                for words in sentences]

    monkeypatch.setattr(runner, "contextual_word_vectors", fake_vectors)
    out = os.path.join(str(tmp_path), "static")
    monkeypatch.setattr(sys, "argv", ["x", "--word-eeg-dir", cache, "--out-dir", out, "--external", "none",
                                      "--folds", "2", "--k", "8"])
    runner.main()
    report = open(os.path.join(out, "eeg_shaped_embeddings.md")).read()
    assert "eeg − shuffled_eeg" in report and "EEG encoding R²" in report
    assert len(pd.read_csv(os.path.join(out, "zuco_predictions_eeg.csv"))) > 0


class PieceTokenizer:
    """Splits words longer than four characters into two pieces."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in text.replace("\n", " \n ").split(" "):
            if not piece:
                continue
            parts = [piece[:4], piece[4:]] if len(piece) > 4 else [piece]
            for part in parts:
                ids.append(self.vocab.setdefault(part, 1 + len(self.vocab) % 299))
        return ids


def test_brain_tuning_positions_and_learning():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    sys.path.insert(0, os.path.dirname(__file__))
    from test_fusion import PROMPT, tiny_lm

    from src.brainshaping.brain_tuning import BrainTunedClassifier, aux_r2, run_brain_fold
    from src.fusion.train import TrainConfig
    from src.neurolm.splits import make_splits

    model = BrainTunedClassifier(tiny_lm(), PieceTokenizer(), class_words=["negative", "neutral", "positive"],
                                 prompt=PROMPT, lora={"r": 4, "alpha": 8, "dropout": 0.0,
                                                      "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
                                 aux_dim=3, aux_layer=2)
    words = ["a", "wonderful", "film"]
    ids = model.build_ids(words, use_eeg=False)
    positions = model.word_positions(words)
    expected_last = [model._encode_word(w)[-1] for w in words]
    assert [ids[p] for p in positions] == expected_last

    rng = np.random.default_rng(0)
    vocab = ["the", "movie", "was", "great", "awful", "plot", "story", "dull", "bright", "scene"]
    word_lists = [list(rng.choice(vocab, size=5)) for _ in range(90)]
    labels = np.tile([0, 1, 2], 30)
    code = {w: rng.standard_normal(3).astype(np.float32) for w in vocab}
    real = [np.stack([code[w] for w in ws]) for ws in word_lists]
    samples = pd.DataFrame({"sample_id": [str(i) for i in range(90)], "subject_id": "all",
                            "sentence_id": range(90), "label_id": labels})
    split = make_splits("text", samples, 0, {"n_folds": 5, "val_fraction": 0.15})[0]
    cfg = TrainConfig(epochs=8, batch_size=8, lr_lora=1e-3, lr_projector=5e-3, evals_per_epoch=1)
    init = model.trainable_state()
    device = torch.device("cpu")
    run_brain_fold(model, init, word_lists, labels, real, split, 1.0, cfg, device, log=lambda *_: None)
    learned = aux_r2(model, split.test, word_lists, real, device, torch.float32)
    shuffled_rows = rng.permutation(len(word_lists))
    shuffled = [real[i][:len(word_lists[j])] for i, j in zip(shuffled_rows, range(len(word_lists)))]
    run_brain_fold(model, init, word_lists, labels, shuffled, split, 1.0, cfg, device, log=lambda *_: None)
    control = aux_r2(model, split.test, word_lists, real, device, torch.float32)
    assert learned > 0.1 and learned > control + 0.15, (learned, control)


def test_brain_tuning_script_end_to_end(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    sys.path.insert(0, os.path.dirname(__file__))
    from test_diagnostics import word_trials, write_cache
    from test_fusion import WordTokenizer, tiny_lm

    trials, _, _ = word_trials(n_readers=4, n_sentences=60)
    cache = write_cache(str(tmp_path), trials)
    runner = load_script("run_brain_tuning")
    monkeypatch.setattr(runner, "load_language_model", lambda *a: (tiny_lm(), WordTokenizer()))
    results = os.path.join(str(tmp_path), "bt")
    argv = ["x", "--word-eeg-dir", cache, "--results-dir", results, "--run-tag", "t", "--epochs", "2",
            "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    report = open(os.path.join(results, "t", "brain_tuning_results.md")).read()
    assert "eeg − shuffled_eeg" in report and "aux R²" in report
    capsys.readouterr()
    runner.main()
    assert capsys.readouterr().out.count("reuse saved predictions") == 4
