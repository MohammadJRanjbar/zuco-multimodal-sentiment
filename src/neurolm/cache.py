"""Fingerprinted cache of frozen NeuroLM trial embeddings.

A view's folder name ends with a hash of everything that changes the
embeddings: checkpoint identity, retained channels and their targets, sampling
rates, preprocessing, length handling, representations, and pooling. Any
change therefore lands in a new folder; an old cache can never be read under a
new configuration, and ``load_view`` re-checks the stored hash.
"""

import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

CACHE_SCHEMA_VERSION = 1


def fingerprint_payload(*, checkpoint, channels, preprocess, length, representations, poolings, precision):
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "checkpoint": {
            "name": checkpoint.get("checkpoint_name"),
            "sha256": checkpoint.get("checkpoint_sha256"),
            "repo": checkpoint.get("repo"),
            "revision": checkpoint.get("revision"),
        },
        "channels": [[label, target] for label, target in channels],
        "sampling_rate": {"source": preprocess["source_sfreq"], "target": preprocess["target_sfreq"]},
        "preprocess": preprocess,
        "length": length,
        "representations": list(representations),
        "poolings": list(poolings),
        "precision": precision,
    }


def fingerprint(payload):
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def view_dir(cache_root, view_name, payload):
    return os.path.join(cache_root, f"{view_name}__{fingerprint(payload)}")


def prepare_view(cache_root, view_name, payload):
    path = view_dir(cache_root, view_name, payload)
    os.makedirs(os.path.join(path, "parts"), exist_ok=True)
    stored_path = os.path.join(path, "fingerprint.json")
    if os.path.exists(stored_path):
        stored = json.load(open(stored_path))
        if stored["fingerprint"] != fingerprint(payload):
            raise RuntimeError(f"fingerprint mismatch in {path}")
    else:
        _write_json({"fingerprint": fingerprint(payload), "payload": payload}, stored_path)
    return path


def _write_json(value, path):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, default=str)
    os.replace(temporary, path)


def part_exists(path, subject):
    return os.path.exists(os.path.join(path, "parts", f"{subject}.npz"))


def save_part(path, subject, metadata_rows, features):
    """Save one subject's pooled features atomically."""
    parts = os.path.join(path, "parts")
    keys = sorted(features)
    arrays = {key: np.asarray(features[key], dtype=np.float32) for key in keys}
    arrays["sample_id"] = np.asarray([row["sample_id"] for row in metadata_rows])
    temporary = os.path.join(parts, f"{subject}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, os.path.join(parts, f"{subject}.npz"))
    frame = pd.DataFrame(metadata_rows)
    frame.to_csv(os.path.join(parts, f"{subject}.csv.tmp"), index=False)
    os.replace(os.path.join(parts, f"{subject}.csv.tmp"), os.path.join(parts, f"{subject}.csv"))


def merge_parts(path):
    """Merge subject parts into ``embeddings.npz`` and ``metadata.csv``."""
    frames, arrays = [], {}
    for npz_path in sorted(glob.glob(os.path.join(path, "parts", "*.npz"))):
        subject = os.path.basename(npz_path)[:-4]
        part = np.load(npz_path, allow_pickle=False)
        meta = pd.read_csv(os.path.join(path, "parts", f"{subject}.csv"))
        if list(part["sample_id"]) != meta["sample_id"].tolist():
            raise RuntimeError(f"sample order mismatch in part {subject}")
        frames.append(meta)
        for key in part.files:
            if key != "sample_id":
                arrays.setdefault(key, []).append(part[key])
    if not frames:
        raise RuntimeError(f"no parts in {path}")
    metadata = pd.concat(frames, ignore_index=True)
    merged = {key: np.concatenate(blocks) for key, blocks in arrays.items()}
    merged["sample_id"] = metadata["sample_id"].to_numpy().astype(str)
    embeddings_path = os.path.join(path, "embeddings.npz")
    metadata["embedding_path"] = embeddings_path
    metadata["embedding_row"] = np.arange(len(metadata))
    np.savez_compressed(embeddings_path + ".tmp.npz", **merged)
    os.replace(embeddings_path + ".tmp.npz", embeddings_path)
    metadata.to_csv(os.path.join(path, "metadata.csv"), index=False)
    return metadata, merged


def load_view(path):
    """Load a merged view after re-checking its stored fingerprint."""
    stored = json.load(open(os.path.join(path, "fingerprint.json")))
    if fingerprint(stored["payload"]) != stored["fingerprint"]:
        raise RuntimeError(f"stored fingerprint does not match payload in {path}")
    if not os.path.basename(path.rstrip("/")).endswith(stored["fingerprint"]):
        raise RuntimeError(f"folder name does not carry fingerprint {stored['fingerprint']}")
    metadata = pd.read_csv(os.path.join(path, "metadata.csv"))
    data = np.load(os.path.join(path, "embeddings.npz"), allow_pickle=False)
    features = {key: data[key] for key in data.files if key != "sample_id"}
    if list(data["sample_id"]) != metadata["sample_id"].tolist():
        raise RuntimeError("embedding rows and metadata disagree")
    return metadata, features, stored


def payload_matches(stored, expected):
    """True when every key in ``expected`` equals the stored payload value."""
    normalized = json.loads(json.dumps(expected, default=str))
    return all(stored.get(key) == value for key, value in normalized.items())


def find_view(cache_root, view_name, expected=None):
    """Return the completed cache folder for ``view_name``.

    ``expected`` is a partial payload (for example preprocessing, length,
    channels, and pooling from the current config). Folders whose stored payload
    differs are ignored, so a stale cache is never selected.
    """
    matches = sorted(glob.glob(os.path.join(cache_root, f"{view_name}__*")))
    matches = [m for m in matches if os.path.exists(os.path.join(m, "metadata.csv"))]
    if expected is not None:
        matches = [
            m for m in matches
            if payload_matches(json.load(open(os.path.join(m, "fingerprint.json")))["payload"], expected)
        ]
    if not matches:
        raise FileNotFoundError(f"no completed cache for view {view_name!r} matching the config in {cache_root}")
    if len(matches) > 1:
        raise RuntimeError(
            f"{len(matches)} caches match view {view_name!r}; pass the exact folder: {matches}"
        )
    return matches[0]
