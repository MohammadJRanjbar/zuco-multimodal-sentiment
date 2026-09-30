"""Trial-local raw-EEG preprocessing matched to the released NeuroLM code.

Verified against NeuroLM (commit 0cda987):

* ``dataset_maker/*.py``: 0.1-75 Hz band-pass, 50/60 Hz notch, resample to
  200 Hz, values in microvolts;
* ``dataset.py`` / ``downstream_dataset.py``: ``X / 100`` (a fixed scale, no
  fitted statistics); 200-sample (1 s) patches; tokens ordered time-major
  (``'N (A T) -> (A N) T'``: every channel at second 0, then second 1, ...);
  ``input_time`` is the second index, ``input_chans`` the vocabulary index;
* ``prepare_TUH_pretrain.py``: pretraining samples are
  ``floor(1024 / n_channels)`` seconds long, so the context used in
  pretraining is ``block_size = 1024`` tokens.

Every operation here uses only the trial being processed; there is no fitted
normalization, so no statistic can leak from test trials.
"""

from dataclasses import asdict, dataclass, field
from fractions import Fraction

import numpy as np
from scipy import signal

UNIT_TO_MICROVOLT = {"uV": 1.0, "mV": 1e3, "V": 1e6}


@dataclass
class PreprocessConfig:
    source_sfreq: float = 500.0
    target_sfreq: float = 200.0
    input_unit: str = "uV"
    reference: str = "average"  # 'average' or 'none'
    reference_channel_index: int = 104  # ZuCo's flat Cz (recording reference)
    highpass_hz: float = 0.1
    lowpass_hz: float = 75.0
    notch_hz: float = 50.0  # ZuCo was recorded in Zurich (50 Hz mains)
    notch_quality: float = 30.0
    filter_order: int = 4
    demean: bool = True
    max_nan_fraction: float = 0.1
    scale_divisor: float = 100.0
    patch_samples: int = 200
    remainder: str = "drop"  # 'drop' the final partial second, or 'pad' it with zeros
    amplitude_guard_uv: tuple = (0.05, 2000.0)  # plausible median channel std

    def to_dict(self):
        return asdict(self)


@dataclass
class LengthConfig:
    strategy: str = "chunk"  # 'crop' or 'chunk'
    max_tokens: int = 1024  # NeuroLM pretraining context (block_size)
    max_seconds: int = 0  # 0 -> floor(max_tokens / channels)
    time_embedding_limit: int = 64  # rows of NeuralTransformer.time_embed

    def to_dict(self):
        return asdict(self)


@dataclass
class TrialTokens:
    """One trial split into model-ready chunks."""

    chunks: list = field(default_factory=list)  # [(x, chan_idx, time_idx)]
    n_channels: int = 0
    n_patches: int = 0
    n_patches_used: int = 0
    seconds_per_chunk: int = 0
    truncated: bool = False
    dropped_channels: list = field(default_factory=list)
    median_channel_std_uv: float = float("nan")
    max_abs_uv: float = float("nan")


def _interpolate_nans(row):
    missing = ~np.isfinite(row)
    if not missing.any():
        return row
    good = np.flatnonzero(~missing)
    filled = row.copy()
    filled[missing] = np.interp(np.flatnonzero(missing), good, row[good])
    return filled


def clean_channels(raw, cfg):
    """Interpolate short NaN gaps; return data and per-channel validity."""
    data = np.asarray(raw, dtype=np.float64) * UNIT_TO_MICROVOLT[cfg.input_unit]
    valid = np.ones(data.shape[0], dtype=bool)
    for channel in range(data.shape[0]):
        nan_fraction = float(np.mean(~np.isfinite(data[channel])))
        if nan_fraction > cfg.max_nan_fraction or nan_fraction == 1.0:
            valid[channel] = False
            data[channel] = 0.0
        elif nan_fraction > 0:
            data[channel] = _interpolate_nans(data[channel])
    return data, valid


def rereference(data, valid, cfg):
    if cfg.reference == "none":
        return data
    if cfg.reference != "average":
        raise ValueError("reference must be 'average' or 'none'")
    # The flat recording reference is a real zero-potential electrode, so it
    # takes part in the average and regains a signal afterwards.
    mean = data[valid].mean(axis=0, keepdims=True)
    out = data - mean
    out[~valid] = 0.0
    return out


def _padlen(n_samples, sos):
    return min(3 * (2 * len(sos) + 1), max(n_samples - 1, 0))


def filter_and_resample(data, cfg):
    """Notch, band-pass, and polyphase-resample every channel of one trial."""
    fs = cfg.source_sfreq
    n = data.shape[1]
    if cfg.demean:
        data = data - data.mean(axis=1, keepdims=True)
    if cfg.notch_hz and cfg.notch_hz < fs / 2:
        b, a = signal.iirnotch(cfg.notch_hz, cfg.notch_quality, fs=fs)
        sos = signal.tf2sos(b, a)
        data = signal.sosfiltfilt(sos, data, axis=1, padlen=_padlen(n, sos))
    low = cfg.highpass_hz or None
    high = cfg.lowpass_hz if cfg.lowpass_hz and cfg.lowpass_hz < fs / 2 else None
    if low and high:
        sos = signal.butter(cfg.filter_order, [low, high], btype="bandpass", fs=fs, output="sos")
    elif high:
        sos = signal.butter(cfg.filter_order, high, btype="lowpass", fs=fs, output="sos")
    elif low:
        sos = signal.butter(cfg.filter_order, low, btype="highpass", fs=fs, output="sos")
    else:
        sos = None
    if sos is not None and n > 1:
        data = signal.sosfiltfilt(sos, data, axis=1, padlen=_padlen(n, sos))
    ratio = Fraction(cfg.target_sfreq / fs).limit_denominator(1000)
    if ratio != 1:
        data = signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=1)
    return data


def preprocess_trial(raw, cfg):
    """Raw ``[channels, samples]`` -> microvolts at 200 Hz plus channel validity."""
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError(f"expected [channels, samples], got {raw.shape}")
    data, valid = clean_channels(raw, cfg)
    ref = cfg.reference_channel_index
    if 0 <= ref < data.shape[0] and cfg.reference == "average":
        valid[ref] = True  # a flat reference is valid (zero potential), not missing
    data = rereference(data, valid, cfg)
    data = filter_and_resample(data, cfg)
    data[~valid] = 0.0
    return data, valid


def seconds_per_chunk(n_channels, length):
    limit = length.max_seconds or (length.max_tokens // max(n_channels, 1))
    return int(max(1, min(limit, length.time_embedding_limit)))


def tokenize(data, valid, channel_plan, cfg, length):
    """Select mapped channels, scale, and cut into NeuroLM token chunks.

    ``channel_plan`` is ``[(zuco_index, vocab_index)]`` in the order tokens are
    emitted within each second. Channels that are invalid in this trial are
    left out of this trial's tokens rather than filled with zeros.
    """
    keep = [(zi, vi) for zi, vi in channel_plan if valid[zi]]
    dropped = [zi for zi, _ in channel_plan if not valid[zi]]
    out = TrialTokens(dropped_channels=dropped, n_channels=len(keep))
    if not keep:
        return out
    selected = data[[zi for zi, _ in keep]]
    stds = selected.std(axis=1)
    out.median_channel_std_uv = float(np.median(stds))
    out.max_abs_uv = float(np.abs(selected).max()) if selected.size else float("nan")
    patch = cfg.patch_samples
    n_full, rest = divmod(selected.shape[1], patch)
    if rest and cfg.remainder == "pad":
        selected = np.pad(selected, ((0, 0), (0, patch - rest)))
        n_full += 1
    elif cfg.remainder not in {"drop", "pad"}:
        raise ValueError("remainder must be 'drop' or 'pad'")
    out.n_patches = n_full
    if n_full == 0:
        return out
    scaled = (selected[:, : n_full * patch] / cfg.scale_divisor).astype(np.float32)
    # [C, A*T] -> [A, C, T]: time-major token order, as in NeuroLM's rearrange.
    patches = scaled.reshape(len(keep), n_full, patch).transpose(1, 0, 2)
    vocab = np.asarray([vi for _, vi in keep], dtype=np.int64)

    window = seconds_per_chunk(len(keep), length)
    out.seconds_per_chunk = window
    if length.strategy == "crop":
        starts = [0]
        out.truncated = n_full > window
    elif length.strategy == "chunk":
        starts = list(range(0, n_full, window))
    else:
        raise ValueError("length strategy must be 'crop' or 'chunk'")
    used = 0
    for start in starts:
        block = patches[start:start + window]
        n_seconds = block.shape[0]
        used += n_seconds
        x = block.reshape(n_seconds * len(keep), patch)
        chans = np.tile(vocab, n_seconds)
        times = np.repeat(np.arange(n_seconds, dtype=np.int64), len(keep))
        out.chunks.append((x, chans, times))
    out.n_patches_used = used
    return out


def amplitude_ok(tokens, cfg):
    low, high = cfg.amplitude_guard_uv
    value = tokens.median_channel_std_uv
    return bool(np.isfinite(value) and low <= value <= high)


def config_from_dict(values, cls):
    known = {k: v for k, v in (values or {}).items() if k in cls.__dataclass_fields__}
    if "amplitude_guard_uv" in known:
        known["amplitude_guard_uv"] = tuple(known["amplitude_guard_uv"])
    return cls(**known)
