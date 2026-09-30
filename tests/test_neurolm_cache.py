import copy
import json
import os

import numpy as np
import pytest

from src.neurolm import cache


def payload(**changes):
    base = cache.fingerprint_payload(
        checkpoint={"checkpoint_name": "NeuroLM-B.pt", "checkpoint_sha256": "abc", "repo": "r", "revision": "v"},
        channels=[("E70", "O1"), ("Cz", "CZ")],
        preprocess={"source_sfreq": 500, "target_sfreq": 200, "notch_hz": 50},
        length={"strategy": "chunk", "max_tokens": 1024},
        representations=["tokenizer", "gpt"],
        poolings=["mean"],
        precision="float32",
    )
    out = copy.deepcopy(base)
    for path, value in changes.items():
        node = out
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return out


@pytest.mark.parametrize("change", [
    {"checkpoint.sha256": "def"},
    {"checkpoint.name": "NeuroLM-L.pt"},
    {"channels": [["E70", "O1"]]},
    {"sampling_rate.target": 250},
    {"preprocess.notch_hz": 60},
    {"poolings": ["mean", "max"]},
    {"length.strategy": "crop"},
])
def test_fingerprint_changes_with_every_relevant_setting(change):
    assert cache.fingerprint(payload(**change)) != cache.fingerprint(payload())


def test_fingerprint_is_stable():
    assert cache.fingerprint(payload()) == cache.fingerprint(payload())


def test_roundtrip_and_tamper_detection(tmp_path):
    root = str(tmp_path)
    p = payload()
    path = cache.prepare_view(root, "view", p)
    rows = [{"sample_id": f"S_{i}", "subject_id": "S", "sentence_id": i, "label_id": i % 3} for i in range(4)]
    cache.save_part(path, "S", rows, {"tokenizer__mean": np.ones((4, 3)) * np.arange(4)[:, None]})
    metadata, merged = cache.merge_parts(path)
    loaded_meta, features, stored = cache.load_view(path)
    assert loaded_meta["sample_id"].tolist() == [f"S_{i}" for i in range(4)]
    np.testing.assert_array_equal(features["tokenizer__mean"][:, 0], [0, 1, 2, 3])
    assert cache.find_view(root, "view", expected={"poolings": ["mean"]}) == path
    with pytest.raises(FileNotFoundError):
        cache.find_view(root, "view", expected={"poolings": ["mean", "max"]})
    stored_path = os.path.join(path, "fingerprint.json")
    tampered = json.load(open(stored_path))
    tampered["payload"]["preprocess"]["notch_hz"] = 60
    json.dump(tampered, open(stored_path, "w"))
    with pytest.raises(RuntimeError):
        cache.load_view(path)


def test_changed_config_uses_a_new_folder(tmp_path):
    a = cache.prepare_view(str(tmp_path), "view", payload())
    b = cache.prepare_view(str(tmp_path), "view", payload(**{"preprocess.notch_hz": 60}))
    assert a != b
