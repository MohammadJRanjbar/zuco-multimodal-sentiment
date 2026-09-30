"""Synthetic data shared by the NeuroLM probe tests."""

import numpy as np
import pandas as pd


def synthetic_samples(n_subjects=12, n_sentences=60, missing=0.05, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(3), n_sentences // 3 + 1)[:n_sentences]
    rng.shuffle(labels)
    rows = []
    for s in range(n_subjects):
        for sentence in range(n_sentences):
            if rng.random() < missing:
                continue
            rows.append({
                "sample_id": f"S{s:02d}_{sentence:04d}",
                "subject_id": f"S{s:02d}",
                "sentence_id": sentence,
                "label_id": int(labels[sentence]),
                "duration_s": float(rng.uniform(2, 12)),
            })
    return pd.DataFrame(rows)


def signal_features(samples, dim=16, strength=1.0, seed=0):
    """Features whose first dimensions encode the label (plus subject offsets)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((len(samples), dim)).astype(np.float32)
    X[np.arange(len(samples)), samples["label_id"].to_numpy()] += strength * 2.0
    subject_codes = samples["subject_id"].astype("category").cat.codes.to_numpy()
    X[:, -1] += subject_codes * 0.5
    return X
