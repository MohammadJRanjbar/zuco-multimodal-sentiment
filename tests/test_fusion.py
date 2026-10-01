import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.fusion.data import build_fusion_data, model_inputs, reader_statistics  # noqa: E402
from src.fusion.model import FusionClassifier, LoRALinear  # noqa: E402
from src.fusion.train import TrainConfig, predict, run_fold  # noqa: E402
from src.neurolm.splits import make_splits  # noqa: E402

PROMPT = {"instruction": "Classify sentiment.\n", "sentence_label": "Sentence:", "answer_label": "\nSentiment:"}
WORDS = ["the", "film", "was", "here", "today"]


class WordTokenizer:
    """Whitespace tokenizer standing in for a Hugging Face tokenizer."""

    def __init__(self, size=300):
        self.size, self.vocab = size, {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in text.replace("\n", " \n ").split(" "):
            if piece:
                ids.append(self.vocab.setdefault(piece, 1 + len(self.vocab) % (self.size - 1)))
        return ids


def tiny_lm(seed=0):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                                      initializer_range=0.3)  # wide output range, like a pretrained LM
    return transformers.Qwen2ForCausalLM(config)


def tiny_model(eeg_dim, lora_r=4, seed=0):
    lm = tiny_lm(seed)
    return FusionClassifier(lm, WordTokenizer(), class_words=["negative", "neutral", "positive"], eeg_dim=eeg_dim,
                            prompt=PROMPT, lora={"r": lora_r, "alpha": 8, "dropout": 0.0,
                                                 "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
                            projector={"hidden": 32, "dropout": 0.0})


def synthetic_trials(n_readers=6, n_sentences=30, signal=1.5, seed=0):
    """Text is identical for every sentence; only word 2's EEG carries the label."""
    rng = np.random.default_rng(seed)
    labels = np.tile([-1, 0, 1], n_sentences // 3)
    trials = []
    for r in range(n_readers):
        offset = rng.normal(0, 2.0)  # reader-specific level, removed by per-reader normalization
        for s in range(n_sentences):
            features = np.exp(rng.normal(offset, 0.3, size=(len(WORDS), 8, 105))).astype(np.float32)
            features[2, :, :20] *= np.exp(signal * labels[s])
            features[4] = np.nan  # last word skipped (no fixation)
            trials.append({"sample_id": f"R{r}_{s:04d}", "subject_id": f"R{r}", "sentence_id": s,
                           "label": int(labels[s]), "words": list(WORDS), "features": features,
                           "fixations": np.array([1, 1, 1, 1, 0], np.float32)})
    return trials


def test_lora_starts_as_the_frozen_layer():
    base = torch.nn.Linear(8, 6)
    layer = LoRALinear(base, r=2, alpha=4)
    x = torch.randn(3, 8)
    torch.testing.assert_close(layer(x), base(x))
    layer(x).sum().backward()
    assert base.weight.grad is None and layer.lora_B.grad is not None


def test_word_slots_align_and_batching_does_not_matter():
    model = tiny_model(eeg_dim=5).eval()
    words = [["a", "b", "c"], ["a", "b", "c", "d", "e"]]
    eeg = torch.randn(2, 5, 5)
    counts = [3, 5]
    assert model.build_ids(words[0], True).count(-1) == 3
    with torch.no_grad():
        both = model(words, eeg, counts, use_eeg=True)
        alone = model(words[:1], eeg[:1, :3], [3], use_eeg=True)
        torch.testing.assert_close(both[:1], alone, atol=1e-5, rtol=1e-5)
        shift = torch.zeros_like(eeg)
        shift[:, :, 0] = 3.0  # a uniform shift must also register (no per-vector normalization)
        changed = model(words, eeg + shift, counts, use_eeg=True)
        assert not torch.allclose(both, model(words, eeg + 1.0, counts, use_eeg=True))
        assert not torch.allclose(both, changed)
        text_a = model(words, eeg, counts, use_eeg=False)
        text_b = model(words, eeg + 1.0, counts, use_eeg=False)
        torch.testing.assert_close(text_a, text_b)
    assert both.shape == (2, 3)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n or n.startswith("projector.") for n in trainable)


def test_reader_normalization_uses_training_rows_only():
    data = build_fusion_data(synthetic_trials(n_readers=2, n_sentences=9))
    assert data.log_transformed and data.dim == 8 * 104 + 1
    train_rows = [i for i in range(len(data.samples)) if data.samples["sentence_id"][i] < 6]
    stats, fallback = reader_statistics(data, train_rows)
    assert not fallback
    test_row = next(i for i in range(len(data.samples)) if data.samples["sentence_id"][i] >= 6)
    data.features[test_row][data.has_eeg[test_row]] += 1000.0
    again, _ = reader_statistics(data, train_rows)
    for reader in stats:
        np.testing.assert_array_equal(stats[reader][0], again[reader][0])


def test_controls_keep_flags_and_reader_partitions():
    data = build_fusion_data(synthetic_trials(n_readers=3, n_sentences=30))
    split = make_splits("text", data.samples, seed=1, params={"n_folds": 5, "val_fraction": 0.15})[0]
    stats, _ = reader_statistics(data, split.train)
    aligned = model_inputs(data, stats, "aligned")
    shuffled = model_inputs(data, stats, "shuffled", split, np.random.default_rng(0))
    fixation = model_inputs(data, stats, "fixation_only")
    subjects = data.samples["subject_id"].to_numpy()
    for part in (split.train, split.test):
        for reader in np.unique(subjects[part]):
            rows = [i for i in part if subjects[i] == reader]
            a = np.concatenate([aligned[i][data.has_eeg[i], :-1] for i in rows])
            s = np.concatenate([shuffled[i][data.has_eeg[i], :-1] for i in rows])
            np.testing.assert_allclose(np.sort(a, axis=0), np.sort(s, axis=0))
    for i in range(len(data.samples)):
        np.testing.assert_array_equal(shuffled[i][:, -1], aligned[i][:, -1])
        assert np.all(fixation[i][:, :-1] == 0) and np.array_equal(fixation[i][:, -1], aligned[i][:, -1])
        assert np.all(aligned[i][~data.has_eeg[i]] == 0)


def test_model_learns_eeg_only_signal_and_controls_do_not():
    data = build_fusion_data(synthetic_trials())
    split = make_splits("text", data.samples, seed=0, params={"n_folds": 5, "val_fraction": 0.15})[0]
    stats, _ = reader_statistics(data, split.train)
    labels = data.samples["label_id"].to_numpy()
    cfg = TrainConfig(epochs=6, batch_size=8, lr_lora=1e-3, lr_projector=3e-3, evals_per_epoch=2)
    model = tiny_model(eeg_dim=data.dim)
    init = model.trainable_state()
    device = torch.device("cpu")
    scores = {}
    for arm, control, use_eeg in [("text_eeg", "aligned", True), ("text_shuffled_eeg", "shuffled", True),
                                  ("text_only", "aligned", False)]:
        inputs = model_inputs(data, stats, control, split, np.random.default_rng(0))
        probs, _ = run_fold(model, init, data, inputs, split, use_eeg, cfg, device, log=lambda *_: None)
        scores[arm] = float((probs.argmax(1) == labels[split.test]).mean())
    assert scores["text_eeg"] >= 0.6, scores
    assert scores["text_eeg"] - max(scores["text_shuffled_eeg"], scores["text_only"]) >= 0.25, scores
    probs = predict(model, split.test, data, model_inputs(data, stats), True, device, torch.float32, 16)
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-5)


def test_runner_end_to_end_with_controls(tmp_path, monkeypatch, capsys):
    import importlib.util
    import os
    import sys

    from src.fusion.word_eeg import save_subject

    trials = synthetic_trials(n_readers=4, n_sentences=30)
    cache = os.path.join(tmp_path, "word_eeg")
    for reader in sorted({t["subject_id"] for t in trials}):
        save_subject(cache, reader, [t for t in trials if t["subject_id"] == reader])
    root = os.path.join(os.path.dirname(__file__), "..")
    spec = importlib.util.spec_from_file_location("run_eeg_text_lora", os.path.join(root, "scripts", "run_eeg_text_lora.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def fake_loader(name, revision, dtype, device):
        return tiny_lm(), WordTokenizer()

    monkeypatch.setattr(runner, "load_language_model", fake_loader)
    results = os.path.join(tmp_path, "results")
    argv = ["x", "--config", os.path.join(root, "configs", "eeg_text_lora.yaml"), "--word-eeg-dir", cache,
            "--results-dir", results, "--run-tag", "t", "--folds", "1", "--epochs", "4", "--device", "cpu",
            "--report", os.path.join(tmp_path, "report.md")]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    report = open(os.path.join(tmp_path, "report.md")).read()
    assert "text_eeg − text_shuffled_eeg" in report and "## Verdict" in report
    fold = pd.read_csv(os.path.join(results, "t", "text_eeg", "fold_0.csv"))
    assert {"sample_id", "subject_id", "sentence_id", "true_label", "predicted_label",
            "prob_negative", "prob_neutral", "prob_positive"} <= set(fold.columns)
    capsys.readouterr()
    runner.main()  # a second run reuses every saved fold instead of retraining
    assert capsys.readouterr().out.count("reuse saved predictions") == 4
