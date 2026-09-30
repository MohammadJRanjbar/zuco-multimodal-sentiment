import numpy as np
import pytest

from src.neurolm.preprocess import (
    LengthConfig, PreprocessConfig, preprocess_trial, seconds_per_chunk, tokenize,
)


def make_raw(n_channels=6, seconds=3.3, fs=500, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    raw = 10 * rng.standard_normal((n_channels, len(t)))
    raw += 20 * np.sin(2 * np.pi * 10 * t)
    raw[-1] = 0.0  # flat recording reference, as ZuCo's Cz
    return raw


def cfg(**kwargs):
    base = dict(reference_channel_index=5)
    base.update(kwargs)
    return PreprocessConfig(**base)


def test_resampling_and_patch_count():
    data, valid = preprocess_trial(make_raw(seconds=3.3), cfg())
    assert data.shape == (6, 660)  # 3.3 s at 200 Hz
    tokens = tokenize(data, valid, [(i, 10 + i) for i in range(6)], cfg(), LengthConfig())
    assert tokens.n_patches == 3 and tokens.n_patches_used == 3  # remainder (0.3 s) dropped
    x, chans, times = tokens.chunks[0]
    assert x.shape == (18, 200)


def test_tokens_are_time_major_with_channel_and_time_indices():
    data = np.zeros((3, 600))
    for c in range(3):
        for second in range(3):
            data[c, second * 200:(second + 1) * 200] = 100 * (10 * c + second)
    valid = np.ones(3, bool)
    plan = [(0, 7), (1, 8), (2, 9)]
    x, chans, times = tokenize(data, valid, plan, cfg(), LengthConfig()).chunks[0]
    expected = [10 * c + s for s in range(3) for c in range(3)]  # all channels of second 0 first
    np.testing.assert_allclose(x[:, 0], expected)  # divided by 100
    assert chans.tolist() == [7, 8, 9] * 3
    assert times.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_average_reference_restores_the_reference_channel():
    data, valid = preprocess_trial(make_raw(), cfg(reference="average"))
    assert valid[5] and data[5].std() > 0
    np.testing.assert_allclose(data.mean(axis=0), 0.0, atol=1e-6)
    data_none, _ = preprocess_trial(make_raw(), cfg(reference="none"))
    assert np.abs(data_none[5]).max() < 1e-6


def test_notch_removes_line_noise():
    fs, t = 500, np.arange(5000) / 500
    raw = np.vstack([30 * np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 7 * t) for _ in range(3)])
    data, _ = preprocess_trial(raw, cfg(reference="none", reference_channel_index=-1))
    spectrum = np.abs(np.fft.rfft(data[0]))
    freqs = np.fft.rfftfreq(data.shape[1], 1 / 200)
    assert spectrum[np.argmin(abs(freqs - 50))] < 0.05 * spectrum[np.argmin(abs(freqs - 7))] * 30


def test_nan_channels_are_dropped_and_short_gaps_filled():
    raw = make_raw()
    raw[1, :1000] = np.nan  # > 10% missing -> invalid in this trial
    raw[2, 100:110] = np.nan  # short gap -> interpolated
    data, valid = preprocess_trial(raw, cfg())
    assert not valid[1] and valid[2] and np.isfinite(data).all()
    tokens = tokenize(data, valid, [(i, i) for i in range(6)], cfg(), LengthConfig())
    assert tokens.dropped_channels == [1] and tokens.n_channels == 5


def test_chunk_and_crop_strategies():
    data = np.random.default_rng(0).standard_normal((42, 30 * 200))
    valid = np.ones(42, bool)
    plan = [(i, i) for i in range(42)]
    chunk = tokenize(data, valid, plan, cfg(), LengthConfig(strategy="chunk", max_tokens=1024))
    assert chunk.seconds_per_chunk == 24 and [len(c[0]) for c in chunk.chunks] == [24 * 42, 6 * 42]
    assert not chunk.truncated and chunk.n_patches_used == 30
    crop = tokenize(data, valid, plan, cfg(), LengthConfig(strategy="crop", max_tokens=276))
    assert crop.seconds_per_chunk == 6 and len(crop.chunks) == 1 and crop.truncated
    assert crop.chunks[0][2].max() == 5


def test_window_never_exceeds_time_embedding():
    assert seconds_per_chunk(4, LengthConfig(max_tokens=1024)) == 64


def test_preprocessing_is_stateless():
    raw = make_raw(seed=3)
    first, _ = preprocess_trial(raw, cfg())
    preprocess_trial(make_raw(seed=4) * 1000, cfg())
    second, _ = preprocess_trial(raw, cfg())
    np.testing.assert_array_equal(first, second)


def test_rejects_wrong_dimensions():
    with pytest.raises(ValueError):
        preprocess_trial(np.zeros(100), cfg())
