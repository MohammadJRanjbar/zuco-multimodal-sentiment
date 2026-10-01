"""Stage 4: where does the text model fail, and does EEG help there?

Uses the saved held-out predictions of the LoRA arms. Sentences are grouped by
linguistic properties (negation, contrast, length, neutral label, lexicon
polarity that disagrees with the label, low text-model confidence). For each
group the accuracy of text-only, text + EEG, and text + shuffled EEG is
compared with a sentence-cluster bootstrap. Also: per-reader EEG effects, and
whether EEG can predict *which* sentences the text model gets wrong.
"""

import re

import numpy as np
import pandas as pd

NEGATION = re.compile(r"\b(?:not|no|never|nothing|none|nobody|neither|nor|hardly|barely|without)\b|n't", re.I)
CONTRAST = re.compile(r"\b(?:but|although|though|however|yet|despite|whereas)\b", re.I)


def sentence_properties(sentences, text_only, lexicon=None):
    """``sentences``: DataFrame (sentence_id, sentence, label_id); ``text_only``: its predictions."""
    lexicon = lexicon or {}
    table = sentences.copy()
    table["negation"] = table["sentence"].str.contains(NEGATION)
    table["contrast"] = table["sentence"].str.contains(CONTRAST)
    words = table["sentence"].str.split()
    table["n_words"] = words.str.len()
    table["length_bin"] = pd.qcut(table["n_words"], 3, labels=["short", "medium", "long"])
    valences = words.apply(lambda ws: [lexicon[w.lower().strip(".,;:!?\"'()")] for w in ws
                                       if w.lower().strip(".,;:!?\"'()") in lexicon])
    table["lexicon_polarity"] = valences.apply(lambda v: float(np.sum(v)) if v else 0.0)
    label_sign = table["label_id"].map({0: -1, 1: 0, 2: 1})
    table["lexicon_conflict"] = (label_sign != 0) & (np.sign(table["lexicon_polarity"]) == -label_sign)
    per_sentence = text_only.groupby("sentence_id").agg(
        text_correct=("correct", "mean"),
        text_confidence=("confidence", "mean"))
    table = table.merge(per_sentence, on="sentence_id", how="left")
    table["text_low_confidence"] = table["text_confidence"] <= table["text_confidence"].quantile(0.25)
    table["text_wrong"] = table["text_correct"] < 0.5
    return table


def add_correctness(frame):
    probs = frame[[c for c in frame.columns if c.startswith("prob_")]].to_numpy()
    frame = frame.copy()
    frame["correct"] = (frame["true_id"] == frame["predicted_id"]).astype(float)
    frame["confidence"] = probs.max(axis=1)
    return frame


def accuracy_delta(candidate, baseline, sentences=None, n_boot=2000, seed=0):
    """Sentence-cluster bootstrap of the accuracy difference on shared trials."""
    merged = candidate[["sample_id", "sentence_id", "correct"]].merge(
        baseline[["sample_id", "correct"]], on="sample_id", suffixes=("_a", "_b"))
    if sentences is not None:
        merged = merged[merged["sentence_id"].isin(sentences)]
    if merged.empty:
        return {"n_trials": 0}
    per_sentence = merged.groupby("sentence_id").agg(a=("correct_a", "sum"), b=("correct_b", "sum"),
                                                     n=("correct_a", "size"))
    a, b, n = per_sentence["a"].to_numpy(), per_sentence["b"].to_numpy(), per_sentence["n"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(n), len(n))
        draws.append((a[pick].sum() - b[pick].sum()) / n[pick].sum())
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"n_sentences": int(len(n)), "n_trials": int(n.sum()), "accuracy_a": float(a.sum() / n.sum()),
            "accuracy_b": float(b.sum() / n.sum()), "delta": float((a.sum() - b.sum()) / n.sum()),
            "ci95": [round(float(low), 4), round(float(high), 4)]}


def subset_effects(arms, properties, comparisons, n_boot=2000):
    subsets = {"all": properties["sentence_id"]}
    for column in ("negation", "contrast", "lexicon_conflict", "text_low_confidence", "text_wrong"):
        subsets[column] = properties.loc[properties[column].astype(bool), "sentence_id"]
    for bin_name in ("short", "medium", "long"):
        subsets[f"length_{bin_name}"] = properties.loc[properties["length_bin"] == bin_name, "sentence_id"]
    for label, name in ((0, "negative"), (1, "neutral"), (2, "positive")):
        subsets[f"label_{name}"] = properties.loc[properties["label_id"] == label, "sentence_id"]
    rows = []
    for subset, ids in subsets.items():
        for candidate, baseline in comparisons:
            if candidate in arms and baseline in arms:
                result = accuracy_delta(arms[candidate], arms[baseline], set(ids), n_boot)
                rows.append({"subset": subset, "comparison": f"{candidate} - {baseline}", **result})
    return pd.DataFrame(rows)


def reader_effects(arms, candidate, baseline):
    rows = []
    merged = arms[candidate].merge(arms[baseline][["sample_id", "correct"]], on="sample_id", suffixes=("_a", "_b"))
    for reader, part in merged.groupby("subject_id"):
        rows.append({"reader": reader, "n": int(len(part)), "accuracy_a": float(part["correct_a"].mean()),
                     "accuracy_b": float(part["correct_b"].mean()),
                     "delta": float(part["correct_a"].mean() - part["correct_b"].mean())})
    return pd.DataFrame(rows).sort_values("delta", ascending=False)
