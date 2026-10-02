"""Sentence-label matching shared by feature extraction and training."""

import hashlib
import re

import pandas as pd


def normalize_text(text):
    text = str(text).lower().strip()
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_labels(path):
    data = pd.read_csv(path)
    required = {"sentence_id", "sentence", "sentiment_label"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"label file is missing columns: {sorted(missing)}")
    data = data[list(required)].dropna().copy()
    data["sentence_id"] = data["sentence_id"].astype(int)
    data["sentiment_label"] = data["sentiment_label"].astype(int)
    if data["sentence_id"].duplicated().any():
        raise ValueError("sentence_id must be unique")
    return data.sort_values("sentence_id").reset_index(drop=True)


def label_lookup(path):
    data = load_labels(path)
    return {
        normalize_text(row.sentence): (int(row.sentence_id), int(row.sentiment_label))
        for row in data.itertuples()
    }


def match_sentence(content, lookup):
    return lookup.get(normalize_text(content), (None, None))


UNLABELLED = 99  # label of sentences without a sentiment label (ZuCo NR and TSR tasks)
TEXT_ID_OFFSET = 10 ** 8  # text-derived sentence ids start here, far above the SR ids


def text_sentence_id(content):
    """Stable sentence id from the normalized text: the same sentence gets the same id in every reader's file."""
    digest = hashlib.md5(normalize_text(content).encode()).hexdigest()
    return TEXT_ID_OFFSET + int(digest[:7], 16)


def unlabelled_match(content, lookup=None):
    """``match_sentence`` for unlabelled tasks: every sentence is kept, with a text-derived id."""
    if not normalize_text(content):
        return None, None
    return text_sentence_id(content), UNLABELLED
