"""Training-time augmentation for EEG-to-text, and reader-averaged test items.

Augmentation is applied in the same way to every input condition (EEG, shuffled
EEG, noise, word vectors), so the comparisons between them stay fair. None of
it creates new sentences or new information; it varies how the training
sentences are presented:

* ``crop``: a contiguous span of the trial's words, input rows and target text
  cut together (shorter, new targets, so the decoder cannot simply recall a
  memorized whole sentence and has to follow its input);
* ``mix``: the average of the trial and other readers' trials of the same
  sentence (more distinct inputs, less noise per input);
* ``noise`` and ``dropout``: Gaussian noise added to, and random features
  zeroed in, the fixated inputs.

``average_by_sentence`` builds one test item per sentence from all its readers.
Single-trial word EEG is very noisy (ZuCo split-half reliability about 0.01 for
one reader), so the reader average is the fairest test of what EEG carries.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class Augment:
    crop: float = 0.0       # probability of cropping a training trial
    min_crop: float = 0.5   # a crop keeps at least this share of the words (and at least 3)
    mix: float = 0.0        # probability of averaging a trial with other readers of the same sentence
    max_mix: int = 4        # at most this many trials averaged
    noise: float = 0.0      # standard deviation of the Gaussian noise added to fixated inputs
    dropout: float = 0.0    # share of input features zeroed

    @property
    def active(self):
        return any((self.crop, self.mix, self.noise, self.dropout))


def mean_input(items):
    """Average of the inputs of trials of one sentence; a word counts as fixated if any reader fixated it."""
    x = np.stack([item["x"] for item in items])
    fixated = np.stack([item["fixated"] for item in items])
    count = fixated.sum(axis=0)
    return ((x * fixated[..., None]).sum(axis=0) / np.maximum(count, 1)[:, None]).astype(np.float32), count > 0


def _sentence_key(item):
    return item["sentence"], len(item["x"])


def average_by_sentence(items):
    """One item per sentence (trial id of its first reader + ':avg'), input averaged over its readers."""
    groups = defaultdict(list)
    for item in items:
        groups[_sentence_key(item)].append(item)
    out = []
    for members in groups.values():
        x, fixated = mean_input(members)
        out.append({**members[0], "x": x, "fixated": fixated, "trial": members[0]["trial"] + ":avg"})
    return out


class Augmenter:
    """``augmenter(i)`` returns an augmented copy of training item ``i``."""

    def __init__(self, items, encode, augment, rng):
        self.items, self.encode, self.augment, self.rng = items, encode, augment, rng
        self.groups = defaultdict(list)
        for i, item in enumerate(items):
            if "sentence" in item:
                self.groups[_sentence_key(item)].append(i)

    def __call__(self, index):
        item, a, rng = self.items[index], self.augment, self.rng
        x, fixated, labels = item["x"], item["fixated"], item["labels"]
        if a.mix and a.max_mix > 1 and "sentence" in item and rng.random() < a.mix:
            others = [j for j in self.groups[_sentence_key(item)] if j != index]
            if others:
                n = int(rng.integers(1, min(a.max_mix - 1, len(others)) + 1))
                x, fixated = mean_input([item] + [self.items[j] for j in rng.choice(others, size=n, replace=False)])
        words = item.get("words")
        if a.crop and words and len(words) > 3 and rng.random() < a.crop:
            size = int(rng.integers(max(3, math.ceil(a.min_crop * len(words))), len(words)))
            start = int(rng.integers(0, len(words) - size + 1))
            x, fixated = x[start:start + size], fixated[start:start + size]
            labels = self.encode(" ".join(words[start:start + size]))
        if a.noise:
            x = x + a.noise * rng.standard_normal(x.shape) * fixated[:, None]
        if a.dropout:
            x = x * (rng.random(x.shape) >= a.dropout) / (1 - a.dropout)
        return {**item, "x": np.asarray(x, dtype=np.float32), "fixated": fixated, "labels": labels}
