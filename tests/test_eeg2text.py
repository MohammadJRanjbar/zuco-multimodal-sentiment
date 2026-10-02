import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.eeg2text import data as d  # noqa: E402
from src.eeg2text.metrics import corpus_bleu, sentence_bleu, tokenize, word_error_rate  # noqa: E402
from src.eeg2text.model import EEGToText, VectorQuantizer, code_usage  # noqa: E402
from src.eeg2text.train import Settings, evaluate, train  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CharCodec:
    """Character-level stand-in for the mBART-50 tokenizer (same special-token layout)."""

    def __init__(self, alphabet):
        self.vocab = ["<s>", "<pad>", "</s>", "en_XX", "fa_IR"] + sorted(alphabet)
        self.index = {c: i for i, c in enumerate(self.vocab)}
        self.pad_id = 1

    def lang_id(self, lang):
        return self.index[{"en": "en_XX", "fa": "fa_IR"}[lang]]

    def encode(self, texts, lang, max_length=128):
        return [[self.lang_id(lang)] + [self.index[c] for c in t if c in self.index][:max_length - 2] + [2]
                for t in texts]

    def decode(self, ids):
        return ["".join(self.vocab[i] for i in seq if i > 4) for seq in ids]


def tiny_seq2seq(codec):
    from transformers import MBartConfig, MBartForConditionalGeneration

    torch.manual_seed(0)
    config = MBartConfig(vocab_size=len(codec.vocab), d_model=32, encoder_layers=1, decoder_layers=1,
                         encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64,
                         decoder_ffn_dim=64, max_position_embeddings=256, pad_token_id=1, bos_token_id=0,
                         eos_token_id=2, decoder_start_token_id=2, forced_eos_token_id=2, scale_embedding=True)
    return MBartForConditionalGeneration(config)


def test_metrics():
    assert corpus_bleu(["the film was good"], ["the film was good"]) == 1.0
    assert sentence_bleu("x y z", "the film was good") == 0.0
    assert word_error_rate("a b", "a b c") == pytest.approx(1 / 3)
    assert tokenize("فیلمی واقعاً خوب.") == ["فیلمی", "واقعاً", "خوب", "."]


def test_vector_quantizer_and_code_usage():
    vq = VectorQuantizer(n_codes=8, dim=4)
    z = torch.randn(2, 5, 4, requires_grad=True)
    q, codes, loss = vq(z)
    assert q.shape == z.shape and codes.shape == (2, 5) and loss.item() >= 0
    q.sum().backward()
    assert z.grad is not None  # straight-through gradient
    usage = code_usage({"en": torch.tensor([0, 1, 1, 2]), "fa": torch.tensor([1, 3])}, 8)
    assert usage["shared_codes"] == 1 and usage["en"]["codes_used"] == 3


def synthetic_corpus(lang, n_sentences=36, readers=3, dim=12, seed=0):
    rng = np.random.default_rng(seed)
    vocab = ["ab", "cd", "ef", "gh", "ij", "kl"]
    codes = {w: rng.standard_normal(dim) for w in vocab}
    features, words, sid, reader, label = [], [], [], [], []
    for r in range(readers):
        for s in range(n_sentences):
            ws = [vocab[(s * 7 + k * 3) % len(vocab)] for k in range(3 + s % 2)]
            f = np.stack([codes[w] + 0.1 * rng.standard_normal(dim) for w in ws]).astype(np.float32)
            f[1 if s % 3 == 0 else len(ws) + 1:2 if s % 3 == 0 else 0] = np.nan  # one skipped word sometimes
            features.append(f)
            words.append(ws)
            sid.append(s)
            reader.append(f"R{r}")
            label.append(s % 3)
    corpus = d.Corpus(lang, features, words, np.array(sid), np.array(reader), np.array(label))
    corpus.static = {w: rng.standard_normal(8).astype(np.float32) for w in vocab}
    return d.normalize(d.assign_parts(corpus, seed=0))


def test_controls_share_length_and_fixations():
    corpus = synthetic_corpus("en")
    eeg, noise, shuffled, words = (d.build_inputs(corpus, c, 0) for c in ("eeg", "noise", "shuffled_eeg",
                                                                            "word_vectors"))
    for a, b, c, e in zip(eeg, noise, shuffled, words):
        assert a[0].shape[0] == b[0].shape[0] == c[0].shape[0] == e[0].shape[0]
        assert (a[1] == b[1]).all() and (a[1] == c[1]).all() and (a[1] == e[1]).all()
    assert set(corpus.part) == {"train", "val", "test"}
    test_sentences = set(corpus.sentence_id[corpus.part == "test"])
    assert not test_sentences & set(corpus.sentence_id[corpus.part == "train"])


def test_model_learns_from_informative_input_not_noise():
    corpus = synthetic_corpus("en")
    codec = CharCodec(set("".join(" ".join(w) for w in corpus.words)))
    labels = codec.encode(corpus.texts, "en")
    accuracy = {}
    for condition in ("eeg", "noise"):
        inputs = d.build_inputs(corpus, condition, 0)
        items = {p: [{"x": x, "fixated": m, "labels": labels[i], "trial": f"en:{i}"}
                     for i, ((x, m), part) in enumerate(zip(inputs, corpus.part)) if part == p]
                 for p in ("train", "val", "test")}
        model = EEGToText(tiny_seq2seq(codec), codec, {"en": 12}, lora={"r": 4, "alpha": 8, "targets":
                                                                        ["q_proj", "k_proj", "v_proj", "out_proj"]},
                          hidden=32)
        settings = Settings(epochs=25, batch_size=8, lr=3e-3, lr_lora=3e-3, patience=25, max_new_tokens=16)
        train(model, {"en": items["train"]}, {"en": items["val"]}, settings, torch.device("cpu"), log=lambda *_: None)
        records, _ = evaluate(model, items["test"], "en", settings, torch.device("cpu"))
        accuracy[condition] = sum(r["tf_correct"] for r in records) / sum(r["tf_scored"] for r in records)
        assert all(isinstance(r["free_text"], str) for r in records)
    assert accuracy["eeg"] > accuracy["noise"] + 0.1, accuracy


def test_script_end_to_end(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_diagnostics import word_trials, write_cache
    from test_encoding_scan import write_teco

    trials, _, _ = word_trials(n_readers=3, n_sentences=30)
    cache = write_cache(str(tmp_path), trials)
    trt, labels_csv, _, _ = write_teco(str(tmp_path), n_subjects=3, n_sentences=24)
    runner = load_script("run_eeg_to_text")
    texts = [" ".join(t["words"]) for t in trials] + [" ".join(map(str, w)) for w in
                                                       pd.read_csv(labels_csv)["sentence"].str.split()]
    codec = CharCodec(set("".join(texts)))
    monkeypatch.setattr(runner, "load_seq2seq", lambda name: (tiny_seq2seq(codec), codec))

    def fake_static(corpus, model_name):
        rng = np.random.default_rng(0)
        corpus.static = {w: rng.standard_normal(8).astype(np.float32) for ws in corpus.words for w in ws}
        return corpus

    monkeypatch.setattr(runner.d, "attach_static_vectors", fake_static)
    root = os.path.join(str(tmp_path), "results")
    argv = ["x", "--zuco-word-eeg-dir", cache, "--teco-trt-dir", trt, "--teco-labels-csv", labels_csv,
            "--results-dir", root, "--run-tag", "t", "--inputs", "eeg", "noise", "word_vectors",
            "--encoders", "continuous", "vq", "--epochs", "1", "--max-new-tokens", "8", "--sentiment-model", "none",
            "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    run_root = os.path.join(root, "t")
    joint = json.load(open(os.path.join(run_root, "joint_vq_eeg", "metrics.json")))
    assert set(joint["langs"]) == {"en", "fa"} and "shared_codes" in joint["vq_code_usage"]
    generations = pd.read_csv(os.path.join(run_root, "en_continuous_eeg", "generations.csv"))
    assert set(generations["condition"]) == {"eeg", "noise", "shuffled_eeg"}  # test-time swaps
    report = open(os.path.join(run_root, "eeg_to_text_report.md")).read()
    assert "Paired comparisons" in report and "multilingual" in report.lower()
    summary = json.load(open(os.path.join(run_root, "eeg_to_text_summary.json")))
    assert summary["multilingual_interaction"], "joint vs single-language interaction missing"
    os.utime(os.path.join(run_root, "en_continuous_eeg", "metrics.json"))
    runner.main()  # finished runs are reused


def test_training_under_bf16_autocast(monkeypatch):
    """The GPU path trains under bf16 autocast (VQ codebook included)."""
    import src.eeg2text.train as t

    monkeypatch.setattr(t, "precision_for", lambda device: (torch.bfloat16, None))
    monkeypatch.setattr(t, "autocast", lambda device, dtype: torch.autocast("cpu", dtype=torch.bfloat16))
    corpus = synthetic_corpus("en")
    codec = CharCodec(set("".join(" ".join(w) for w in corpus.words)))
    labels = codec.encode(corpus.texts, "en")
    inputs = d.build_inputs(corpus, "eeg", 0)
    items = {p: [{"x": x, "fixated": m, "labels": labels[i], "trial": f"en:{i}"}
                 for i, ((x, m), part) in enumerate(zip(inputs, corpus.part)) if part == p] for p in ("train", "val", "test")}
    model = EEGToText(tiny_seq2seq(codec), codec, {"en": 12}, lora={"r": 4, "alpha": 8, "targets": ["q_proj", "v_proj"]},
                      vq={"codes": 16, "dim": 8}, hidden=32)
    settings = Settings(epochs=1, batch_size=8, max_new_tokens=8)
    t.train(model, {"en": items["train"]}, {"en": items["val"]}, settings, torch.device("cpu"), log=lambda *_: None)
    records, codes = t.evaluate(model, items["test"], "en", settings, torch.device("cpu"))
    assert records and codes is not None
