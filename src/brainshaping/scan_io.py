"""Shared data and word-vector loading for the encoding scan and the follow-up analyses."""

import hashlib
import json
import os

import numpy as np
import pandas as pd

from ..diagnostics import signal
from .data import sentence_table

MODELS = {
    "labse": "sentence-transformers/LaBSE",
    "xlmr-large": "FacebookAI/xlm-roberta-large",
    "me5-large": "intfloat/multilingual-e5-large",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}
REVISIONS = {"Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}


def model_specs(entries):
    """``alias``, ``alias=name_or_path`` or a Hugging Face name -> ``[(alias, name)]``."""
    specs = []
    for entry in entries:
        if "=" in entry:
            alias, name = entry.split("=", 1)
        elif entry in MODELS:
            alias, name = entry, MODELS[entry]
        else:
            alias, name = entry.rstrip("/").split("/")[-1].lower(), entry
        specs.append((alias, name))
    return specs


def load_items(dataset, word_eeg_dir=None, trt_dir=None, labels_csv=None):
    """Reader-averaged word EEG (as ``item_eeg``) plus the per-reader table and readers per sentence."""
    if dataset == "zuco":
        from ..fusion.word_eeg import load_word_eeg

        if not word_eeg_dir:
            raise SystemExit("--word-eeg-dir is required for ZuCo")
        trials, drop = load_word_eeg(word_eeg_dir), None
    else:
        from .teco import load_teco_trials

        if not trt_dir:
            raise SystemExit("--trt-dir is required for TeCo")
        trials, drop = load_teco_trials(trt_dir, labels_csv), ()
    sentences = sentence_table(trials)
    meta, X, _ = signal.long_word_table(trials) if drop is None else signal.long_word_table(trials, drop_channels=drop)
    Z = signal.zscore_per_reader(meta, X)
    del X
    items, eeg = signal.reader_average(meta, Z)
    readers = pd.DataFrame({"sentence_id": [t["sentence_id"] for t in trials],
                            "reader": [t["subject_id"] for t in trials]})
    readers_per_sentence = readers.drop_duplicates().groupby("sentence_id").size().to_dict()
    items = items.reset_index(drop=True)
    row_of = {sid: i for i, sid in enumerate(sentences["sentence_id"])}
    items["sentence_row"] = items["sentence_id"].map(row_of)
    return {"sentences": sentences, "items": items, "eeg": eeg, "meta": meta, "Z": Z,
            "readers_per_sentence": readers_per_sentence}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def word_vectors(cache_dir, tag, alias, name, sentences, item_sentence, item_word, device, extractor,
                 batch_size=16):
    """Cached vectors [layers, items, dim] for (tag, model); extracted with ``extractor`` when missing."""
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.join(cache_dir, f"{tag}_{alias}")
    revision = REVISIONS.get(name)
    key = digest([name, revision, sentences["words"].tolist(), np.asarray(item_sentence).tolist(),
                  np.asarray(item_word).tolist()])
    if os.path.exists(stem + ".json") and os.path.exists(stem + ".npy"):
        meta = json.load(open(stem + ".json"))
        if meta["key"] == key:
            print(f"{alias}: reusing cached word vectors")
            return np.load(stem + ".npy", mmap_mode="r"), np.array(meta["found"], dtype=bool), meta["info"]
    print(f"{alias}: extracting word vectors from every layer of {name}")
    vectors, found, info = extractor(sentences["words"].tolist(), np.asarray(item_sentence), np.asarray(item_word),
                                     name, device, batch_size=batch_size, revision=revision,
                                     desc=f"{alias}: word vectors")
    np.save(stem + ".npy", vectors)
    json.dump({"key": key, "found": found.tolist(), "info": info}, open(stem + ".json", "w"))
    del vectors
    return np.load(stem + ".npy", mmap_mode="r"), found, info


def word_surprisal(sentences, model_name, device, cache_path=None):
    """Per-word surprisal (bits) for every sentence (list of word lists), cached as JSON."""
    from ..diagnostics.targets import lm_surprisal

    key = digest([model_name, sentences])
    if cache_path and os.path.exists(cache_path):
        cached = json.load(open(cache_path))
        if cached.get("key") == key:
            return [np.array(v) for v in cached["values"]]
    values = lm_surprisal(sentences, model_name, device)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        json.dump({"key": key, "values": [np.asarray(v).tolist() for v in values]}, open(cache_path, "w"))
    return values
