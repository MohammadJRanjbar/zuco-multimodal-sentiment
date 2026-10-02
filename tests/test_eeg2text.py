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
from src.eeg2text.augment import Augment, Augmenter, average_by_sentence  # noqa: E402
from src.eeg2text.metrics import corpus_bleu, sentence_bleu, tokenize, word_error_rate  # noqa: E402
from src.eeg2text.model import EEGToText, VectorQuantizer, code_usage, token_embeddings  # noqa: E402
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

    def word_ids(self, words):
        return [[self.index[c] for c in w if c in self.index] for w in words]

    def special_ids(self):
        return [0, 1, 2, 3, 4]


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
    corpus.static = {"word_vectors": {w: rng.standard_normal(8).astype(np.float32) for w in vocab}}
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
    # Full fine-tuning: a tiny random mBART with frozen weights and LoRA barely trains (loss stays near uniform).
    corpus = synthetic_corpus("en", n_sentences=90)
    codec = CharCodec(set("".join(" ".join(w) for w in corpus.words)))
    labels = codec.encode(corpus.texts, "en")
    accuracy = {}
    for condition in ("eeg", "noise"):
        inputs = d.build_inputs(corpus, condition, 0)
        items = {p: [{"x": x, "fixated": m, "labels": labels[i], "trial": f"en:{i}"}
                     for i, ((x, m), part) in enumerate(zip(inputs, corpus.part)) if part == p]
                 for p in ("train", "val", "test")}
        model = EEGToText(tiny_seq2seq(codec), codec, {"en": 12}, hidden=32, full_finetune=True)
        settings = Settings(epochs=20, batch_size=8, lr=1e-3, patience=20, max_new_tokens=16)
        train(model, {"en": items["train"]}, {"en": items["val"]}, settings, torch.device("cpu"), log=lambda *_: None)
        records, _ = evaluate(model, items["test"], "en", settings, torch.device("cpu"))
        accuracy[condition] = sum(r["tf_correct"] for r in records) / sum(r["tf_scored"] for r in records)
        assert all(isinstance(r["free_text"], str) for r in records)
    assert accuracy["eeg"] > accuracy["noise"] + 0.05, accuracy


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
        corpus.static["word_vectors"] = {w: rng.standard_normal(8).astype(np.float32) for ws in corpus.words
                                         for w in ws}
        return corpus

    monkeypatch.setattr(runner.d, "attach_static_vectors", fake_static)
    extra_trials, _, _ = word_trials(n_readers=2, n_sentences=12, seed=1)
    for t in extra_trials:
        t["sentence_id"], t["label"] = 10 ** 8 + t["sentence_id"], 99
    extra = write_cache(os.path.join(str(tmp_path), "NR"), extra_trials)
    root = os.path.join(str(tmp_path), "results")
    argv = ["x", "--zuco-word-eeg-dir", cache, "--zuco-extra-dirs", extra, "--teco-trt-dir", trt,
            "--teco-labels-csv", labels_csv, "--results-dir", root, "--run-tag", "t",
            "--inputs", "eeg", "noise", "word_vectors", "mbart_vectors", "--encoders", "continuous", "vq",
            "--epochs", "1", "--min-epochs", "1", "--max-new-tokens", "8", "--sentiment-model", "none",
            "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    run_root = os.path.join(root, "t")
    joint = json.load(open(os.path.join(run_root, "joint_vq_eeg", "metrics.json")))
    assert set(joint["langs"]) == {"en", "fa"} and "shared_codes" in joint["vq_code_usage"]
    generations = pd.read_csv(os.path.join(run_root, "en_continuous_eeg", "generations.csv"))
    # test-time swaps, each per reader and on the reader average
    assert set(generations["condition"]) == {"eeg", "noise", "shuffled_eeg", "eeg_avg", "noise_avg",
                                             "shuffled_eeg_avg"}
    averaged = generations[generations["condition"] == "eeg_avg"]
    assert (averaged["reader"] == "average").all() and averaged["sentence_id"].is_unique
    sentences = pd.read_csv(os.path.join(run_root, "sentences.csv"))
    assert (sentences.loc[sentences["sentence_id"] >= 10 ** 8, "part"] == "train").all()
    report = open(os.path.join(run_root, "eeg_to_text_report.md")).read()
    assert "Paired comparisons" in report and "multilingual" in report.lower() and "Positive control" in report
    summary = json.load(open(os.path.join(run_root, "eeg_to_text_summary.json")))
    assert summary["multilingual_interaction"], "joint vs single-language interaction missing"
    assert {(c["input"], c["test"]) for c in summary["positive_control"] if c["setting"] == "en"} == {
        (v, t) for v in ("word_vectors", "mbart_vectors") for t in ("single reader", "reader average")}
    eeg_metrics = json.load(open(os.path.join(run_root, "en_continuous_eeg", "metrics.json")))
    assert eeg_metrics["input_mapping"] == "ridge"
    assert set(eeg_metrics["mapping_quality"]) == {"en/eeg", "en/noise", "en/shuffled_eeg"}  # swaps reuse the EEG map
    assert "Input -> mBART word embedding" in report
    os.utime(os.path.join(run_root, "en_continuous_eeg", "metrics.json"))
    runner.main()  # finished runs are reused
    path = os.path.join(run_root, "en_continuous_eeg", "metrics.json")
    old = json.load(open(path))
    json.dump({**old, "decoding": {"text_only": False}}, open(path, "w"))  # as if evaluated with older decoding

    def no_training(*args, **kwargs):
        raise AssertionError("a run with saved weights must be evaluated again, not retrained")

    monkeypatch.setattr(runner, "train", no_training)
    runner.main()
    again = json.load(open(path))
    assert again["decoding"] == {"text_only": True, "min_text_tokens": 1} and again["train"] == old["train"]


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


def test_encoder_inputs_follow_mbart_layout_and_scale():
    corpus = synthetic_corpus("en")
    codec = CharCodec(set("".join(" ".join(w) for w in corpus.words)))
    lm = tiny_seq2seq(codec)
    model = EEGToText(lm, codec, {"en": 12}, hidden=32)
    x = torch.randn(2, 4, 12)
    fixated = torch.tensor([[True, False, True, True], [True, True, False, False]])
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    embeds, _, _ = model.embed(x, fixated, "en")
    rms = embeds.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, model.target_rms.expand_as(rms), rtol=0.02)  # same scale as mBART's token embeddings
    out, mask = model.encoder_inputs(embeds, valid, "en")
    special = token_embeddings(lm, torch.tensor([codec.lang_id("en"), 2]))
    assert out.shape == (2, 6, 32) and mask.tolist() == [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]
    assert torch.allclose(out[:, 0], special[0].expand(2, -1))  # language code first
    assert torch.allclose(out[0, 5], special[1]) and torch.allclose(out[1, 3], special[1])  # </s> after the words
    assert torch.equal(out[1, 4:], torch.zeros(2, 32))


def test_augmenter_crops_mixes_within_sentence_and_averages_readers():
    rng = np.random.default_rng(0)
    words = ["aa", "bb", "cc", "dd", "ee", "ff"]
    items = [{"x": np.full((6, 3), float(r), np.float32), "fixated": np.array([True] * 5 + [r == 0]),
              "labels": [3, 9, 2], "words": words, "sentence": 7, "trial": f"en:{r}"} for r in range(3)]
    items.append({"x": np.full((6, 3), 100.0, np.float32), "fixated": np.ones(6, bool), "labels": [3, 2],
                  "words": list("uvwxyz"), "sentence": 8, "trial": "en:3"})
    encoded = []
    crop = Augmenter(items, lambda text: encoded.append(text) or [3, 5, 2], Augment(crop=1.0), rng)
    out = crop(0)
    span = encoded[-1].split()
    assert 3 <= len(span) < 6 and len(out["x"]) == len(span) == len(out["fixated"]) and out["labels"] == [3, 5, 2]
    assert " ".join(span) in " ".join(words)
    mix = Augmenter(items, None, Augment(mix=1.0, max_mix=3), rng)
    for _ in range(20):
        mixed = mix(0)
        assert mixed["x"].max() < 100  # never mixed with another sentence
        assert mixed["fixated"][5]  # reader 0 fixated the last word
    averaged = average_by_sentence(items)
    assert len(averaged) == 2
    first = next(a for a in averaged if a["sentence"] == 7)
    assert first["trial"] == "en:0:avg" and np.allclose(first["x"][0], 1.0) and np.allclose(first["x"][5], 0.0)


def test_extra_training_data_never_contains_held_out_sentences():
    corpus = synthetic_corpus("en")
    held_out = [w for w, part in zip(corpus.words, corpus.part) if part == "test"][0]
    extra = d.Corpus("en", [np.ones((len(held_out), 12), np.float32), np.ones((2, 12), np.float32)],
                     [list(held_out), ["zz", "yy"]], np.array([10 ** 8, 10 ** 8 + 1]), np.array(["R0", "R0"]),
                     np.array([-1, -1]), session=np.array(["NR", "NR"]))
    before = len(corpus.words)
    corpus.session = np.full(before, "SR")
    assert d.add_training_data(corpus, extra) == 1
    assert len(corpus.words) == before + 1 and corpus.part[-1] == "train" and corpus.session[-1] == "NR"


def test_unlabelled_tasks_get_stable_text_ids():
    from src.labels import UNLABELLED, unlabelled_match
    from src.neurolm.dataset import subject_from_path

    a, label = unlabelled_match("Henry Ford, born in 1863, was an engineer.")
    b, _ = unlabelled_match("henry ford born in 1863 was an engineer")
    assert a == b and a >= 10 ** 8 and label == UNLABELLED
    assert subject_from_path("/x/resultsZKB_NR.mat") == "ZKB" and subject_from_path("resultsZAB_TSR.mat") == "ZAB"


def test_fetch_skips_a_failing_download_and_retries_it(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_diagnostics import word_trials

    fetch = load_script("fetch_zuco_task")
    calls = []

    def flaky_download(url, path):
        calls.append(os.path.basename(path))
        if path.endswith("resultsZDM_NR.mat") and calls.count("resultsZDM_NR.mat") == 1:
            raise IOError("HTTP Error 403: Forbidden")
        open(path, "w").close()

    trials, _, _ = word_trials(n_readers=1, n_sentences=3)
    monkeypatch.setattr(fetch, "download", flaky_download)
    monkeypatch.setattr(fetch, "extract_subject", lambda path, lookup, match, measure: (trials, [
        {"subject_id": fetch.os.path.basename(path)[7:10], "status": "ok", "n_words": 5, "n_words_with_eeg": 4}]))
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)
    out = os.path.join(str(tmp_path), "NR")
    monkeypatch.setattr(sys, "argv", ["x", "--task", "NR", "--out-dir", out, "--tmp-dir", str(tmp_path),
                                      "--subjects", "ZAB", "ZDM", "ZJS"])
    fetch.main()
    assert calls == ["resultsZAB_NR.mat", "resultsZDM_NR.mat", "resultsZJS_NR.mat", "resultsZDM_NR.mat"]
    assert sorted(f for f in os.listdir(out) if f.endswith(".npz")) == ["ZAB.npz", "ZDM.npz", "ZJS.npz"]
    assert not [f for f in os.listdir(str(tmp_path)) if f.endswith(".mat")]  # downloads deleted


def test_ridge_map_recovers_embeddings_from_informative_inputs_only():
    from src.eeg2text import mapping

    rng = np.random.default_rng(0)
    n_sentences, words_per, dim_in, dim_out = 60, 8, 20, 6
    A = rng.standard_normal((dim_in, dim_out))
    part = np.array(["train"] * 40 + ["val"] * 8 + ["test"] * 12)
    targets, informative, noise = [], [], []
    for s in range(n_sentences):
        x = rng.standard_normal((words_per, dim_in))
        fixated = rng.random(words_per) > 0.2
        targets.append((x @ A).astype(np.float32))
        informative.append(((x + 0.3 * rng.standard_normal(x.shape)) * fixated[:, None], fixated))
        noise.append((rng.standard_normal(x.shape) * fixated[:, None], fixated))
    targets[0][1] = np.nan  # a word without an embedding is skipped in fitting, still predicted
    mapped, fitted, good = mapping.map_inputs(informative, targets, part, np.arange(n_sentences))
    _, _, bad = mapping.map_inputs(noise, targets, part, np.arange(n_sentences))
    assert good["cosine"] > 0.9 and good["r2"] > 0.8, good
    assert abs(bad["cosine_centered"]) < 0.2 and bad["r2"] < 0.05, bad
    assert good["cosine_centered"] > 0.9, good
    assert all(len(x) == words_per and (x[~m] == 0).all() for x, m in mapped)
    again = mapping.apply_map(fitted, informative)
    held_out = [i for i in range(n_sentences) if part[i] != "train"]
    assert all(np.allclose(again[i][0], mapped[i][0], atol=1e-5) for i in held_out)


def test_identity_input_layer_feeds_embeddings_as_they_are():
    corpus = synthetic_corpus("en")
    codec = CharCodec(set("".join(" ".join(w) for w in corpus.words)))
    lm = tiny_seq2seq(codec)
    model = EEGToText(lm, codec, {"en": 32}, input_layer="identity")
    word = token_embeddings(lm, torch.tensor([7, 8, 9]))[None]  # three real token embeddings as word vectors
    embeds, _, _ = model.embed(word, torch.ones(1, 3, dtype=torch.bool), "en")
    cos = torch.nn.functional.cosine_similarity(embeds[0], word[0] - word[0].mean(-1, keepdim=True), dim=-1)
    assert (cos > 0.999).all()  # only re-centred and re-scaled (layer norm), not transformed
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ["missing.en"]
    with pytest.raises(ValueError):
        EEGToText(lm, codec, {"en": 12}, input_layer="identity")


def test_text_only_decoding_bans_special_tokens_and_early_stop():
    from src.eeg2text.model import text_only_processor

    processor = text_only_processor(banned=[0, 1, 3, 4], eos=2, min_text_tokens=1)[0]
    scores = torch.zeros(2, 8)
    forced_step = processor(torch.zeros(2, 1, dtype=torch.long), scores.clone())  # language token step: untouched
    assert torch.isfinite(forced_step).all()
    first = processor(torch.zeros(2, 2, dtype=torch.long), scores.clone())
    assert torch.isinf(first[:, [0, 1, 2, 3, 4]]).all() and torch.isfinite(first[:, 5:]).all()
    later = processor(torch.zeros(2, 3, dtype=torch.long), scores.clone())
    assert torch.isfinite(later[:, 2]).all() and torch.isinf(later[:, [0, 1, 3, 4]]).all()  # </s> allowed again
