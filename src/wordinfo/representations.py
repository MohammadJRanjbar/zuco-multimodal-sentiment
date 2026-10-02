"""Per-word EEG representations computed from raw word epochs (src/wordinfo/epochs.py).

Every representation is one vector per (reader, sentence, word) and is written in the word-EEG cache
layout with 2-D features [n_words, F] (NaN rows for words without an epoch), so the existing analyses
(scripts/decode_eeg_to_text.py, scripts/scan_eeg_encoding.py, scripts/word_information.py,
scripts/run_eeg_to_text.py) run on it unchanged.

* ``raw``: baseline-corrected (-200..0 ms) epoch, mean in 16 bins of 50 ms from 0 to 800 ms, all 105
  channels (1,680 values): the EEG time course without any model.
* ``cbramod``: frozen pretrained CBraMod (braindecode, ``braindecode/cbramod-pretrained``; 1-s patch at
  200 Hz, input in units of 100 µV), encoder output averaged within 6 scalp regions (6 x 200 values).
* ``neurolm``: frozen NeuroLM-B neural tokenizer (one token per mapped 10-10 channel), mean over tokens
  (768 values).
* ``eye_tracking``: log total reading time, first-fixation duration, gaze duration and number of
  fixations, for the same words: what the eye tracker alone says about the word.
"""

import csv
import os

import numpy as np

CHANLOCS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "neurolm", "montages", "zuco_chanlocs.csv")
BASELINE_SAMPLES = 40  # -200..0 ms at 200 Hz
BIN_SAMPLES = 10       # 50 ms at 200 Hz


def scalp_regions(path=CHANLOCS):
    """Six regions (anterior / central / posterior x left / right) as lists of channel indices.

    EEGLAB coordinates: X points to the nose, Y to the left ear. Midline channels (Y = 0) count as left."""
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    x = np.array([float(r["X"]) for r in rows])
    y = np.array([float(r["Y"]) for r in rows])
    cuts = np.quantile(x, [1 / 3, 2 / 3])
    band = np.digitize(x, cuts)  # 0 posterior, 1 central, 2 anterior
    side = (y < 0).astype(int)   # 0 left (incl. midline), 1 right
    regions = [np.flatnonzero((band == b) & (side == s)) for b in range(3) for s in range(2)]
    return [r for r in regions if len(r)]


def raw_timecourse(epochs):
    """[n, C, 200] µV -> [n, C * 16]: baseline-corrected means of 50-ms bins from 0 to 800 ms."""
    x = np.asarray(epochs, dtype=np.float32)
    x = x - x[:, :, :BASELINE_SAMPLES].mean(axis=2, keepdims=True)
    post = x[:, :, BASELINE_SAMPLES:]
    n_bins = post.shape[2] // BIN_SAMPLES
    binned = post[:, :, :n_bins * BIN_SAMPLES].reshape(len(x), x.shape[1], n_bins, BIN_SAMPLES).mean(axis=3)
    return binned.reshape(len(x), -1)


def region_pool(features, regions):
    """[n, C, D] -> [n, len(regions) * D]: mean over the channels of each region."""
    return np.concatenate([features[:, r].mean(axis=1) for r in regions], axis=1)


class CBraModFeatures:
    """Frozen pretrained CBraMod; ``__call__(epochs µV [n, C, 200])`` -> [n, regions * 200]."""

    def __init__(self, device="cpu", batch_size=256, model=None):
        import torch

        if model is None:
            from braindecode.models import CBraMod

            model = CBraMod.from_pretrained("braindecode/cbramod-pretrained", return_encoder_output=True)
        self.torch, self.device, self.batch_size = torch, device, batch_size
        self.model = model.to(device).eval()
        self.regions = scalp_regions()

    def __call__(self, epochs):
        torch, out = self.torch, []
        with torch.no_grad():
            for start in range(0, len(epochs), self.batch_size):
                x = torch.as_tensor(np.asarray(epochs[start:start + self.batch_size], dtype=np.float32) / 100.0,
                                    device=self.device)
                feats = self.model(x, return_features=True)["features"]  # [B, C, patches, 200]
                out.append(region_pool(feats.mean(dim=2).float().cpu().numpy(), self.regions))
        return np.concatenate(out) if out else np.zeros((0, len(self.regions) * 200), np.float32)


class NeuroLMFeatures:
    """Frozen NeuroLM-B tokenizer; ``__call__(epochs µV [n, 105, 200])`` -> [n, 768] (mean over tokens)."""

    def __init__(self, neurolm_dir, checkpoint, device="cpu", batch_size=64):
        from ..neurolm.channel_mapping import build_channel_mapping, load_zuco_chanlocs, retained_channels
        from ..neurolm.encoder import NeuroLMEncoder
        from ..neurolm.preprocess import LengthConfig, PreprocessConfig

        rows = build_channel_mapping(load_zuco_chanlocs()[0])
        self.plan = [(int(zi), int(vi)) for zi, _, _, vi in retained_channels(rows)]
        self.cfg, self.length = PreprocessConfig(), LengthConfig()
        self.encoder = NeuroLMEncoder(neurolm_dir, checkpoint, device=device)
        self.batch_size = batch_size

    def __call__(self, epochs):
        from ..neurolm.preprocess import tokenize

        valid = np.ones(np.asarray(epochs).shape[1], dtype=bool)
        chunks = [tokenize(np.asarray(e, dtype=np.float32), valid, self.plan, self.cfg, self.length).chunks[0]
                  for e in epochs]
        hidden = self.encoder.encode_chunks(chunks, batch_size=self.batch_size)
        return np.stack([h["tokenizer"].mean(axis=0) for h in hidden]).astype(np.float32) if hidden \
            else np.zeros((0, 768), np.float32)


def eye_tracking(arrays, words):
    """[len(words), 4]: log TRT, log FFD, log GD (ms) and number of fixations at the given flat word indices
    (``arrays``: a subject's epoch file, see ``load_subject_epochs``)."""
    columns = [np.log1p(np.nan_to_num(np.asarray(arrays[f"{m}_ms"], dtype=np.float64)[words], nan=0.0))
               for m in ("trt", "ffd", "gd")]
    columns.append(np.asarray(arrays["fixations"], dtype=np.float64)[words])
    return np.stack(columns, axis=1).astype(np.float32)
