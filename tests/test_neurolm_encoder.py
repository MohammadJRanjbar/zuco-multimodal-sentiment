import os

import numpy as np
import pytest

from src.neurolm.pooling import combine_chunks, pool_chunk
from src.neurolm.preprocess import TrialTokens

torch = pytest.importorskip("torch")

from src.neurolm.encoder import bidirectional_mask, pool_trials, stair_mask  # noqa: E402

NEUROLM_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "NeuroLM")


def test_stair_mask_matches_neurolm_pretraining_mask():
    n_chans, n_time, pad = 3, 4, 2
    valid_len = n_chans * n_time
    times = torch.tensor([[t for t in range(n_time) for _ in range(n_chans)] + [0] * pad])
    valid = torch.zeros(1, valid_len + pad, dtype=torch.bool)
    valid[0, :valid_len] = True
    mask = stair_mask(times, valid)[0, 0]
    # dataset.PickleLoader: tril + full blocks per second, padded keys removed
    size = valid_len + pad
    reference = torch.tril(torch.ones(size, size))
    for i in range(n_time):
        reference[i * n_chans:(i + 1) * n_chans, i * n_chans:(i + 1) * n_chans] = 1
    reference[:, valid_len:] = 0
    assert torch.equal(mask[:valid_len], reference[:valid_len].bool())
    assert mask.any(dim=1).all()  # no query row is fully masked (no NaN in attention)


def test_bidirectional_mask_hides_only_padding():
    valid = torch.tensor([[True, True, False]])
    mask = bidirectional_mask(valid)
    assert mask.shape == (1, 1, 3, 3)
    assert mask[0, 0, :, 2].sum() == 0 and mask[0, 0, :, :2].all()


def test_pooling_and_chunk_combination():
    hidden = np.arange(12, dtype=np.float32).reshape(6, 2)
    times = np.array([0, 0, 1, 1, 2, 2])
    pooled = pool_chunk(hidden, times, ["mean", "max", "last_step"])
    np.testing.assert_allclose(pooled["mean"], hidden.mean(0))
    np.testing.assert_allclose(pooled["max"], hidden.max(0))
    np.testing.assert_allclose(pooled["last_step"], hidden[4:].mean(0))
    a = {"mean": np.zeros(2), "max": np.zeros(2)}
    b = {"mean": np.ones(2), "max": np.full(2, 5.0)}
    combined = combine_chunks([a, b], [30, 10])
    np.testing.assert_allclose(combined["mean"], 0.25)
    np.testing.assert_allclose(combined["max"], 5.0)


class StubEncoder:
    """Returns hidden state = first sample of each patch, repeated."""

    def encode_chunks(self, chunks, batch_size=16):
        return [{"tokenizer": np.repeat(x[:, :1], 4, axis=1), "gpt": np.repeat(x[:, 1:2], 4, axis=1)}
                for x, _, _ in chunks]


def trial(values, seconds, channels=2):
    x = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 200, axis=1)
    return (x, np.tile(np.arange(channels), seconds), np.repeat(np.arange(seconds), channels))


def test_pool_trials_combines_chunks_per_trial():
    single = TrialTokens(chunks=[trial([1, 1, 3, 3], 2)])
    split = TrialTokens(chunks=[trial([2, 2], 1), trial([4, 4, 4, 4], 2)])
    pooled = pool_trials(StubEncoder(), [single, split, TrialTokens()], ["mean", "max"])
    np.testing.assert_allclose(pooled[0]["tokenizer__mean"], 2.0)
    np.testing.assert_allclose(pooled[1]["tokenizer__mean"], (2 * 2 + 4 * 4) / 6)
    np.testing.assert_allclose(pooled[1]["gpt__max"], 4.0)
    assert pooled[2] is None


@pytest.mark.skipif(not os.path.exists(os.path.join(NEUROLM_DIR, "model")), reason="NeuroLM repo not cloned")
def test_real_architecture_forward_is_batch_invariant():
    pytest.importorskip("einops")
    pytest.importorskip("transformers")
    from src.neurolm.encoder import NeuroLMEncoder, import_neurolm

    NeuroLM, GPTConfig = import_neurolm(NEUROLM_DIR)
    torch.manual_seed(0)
    model = NeuroLM(GPTConfig(n_layer=1, n_head=12, n_embd=768, block_size=1024, vocab_size=50304,
                              bias=False, dropout=0.0), init_from="scratch")
    encoder = NeuroLMEncoder(None, None, device="cpu", model=model, checkpoint_info={"checkpoint_name": "random"})
    rng = np.random.default_rng(0)

    def chunk(seconds):
        return (rng.standard_normal((3 * seconds, 200)).astype(np.float32),
                np.tile(np.array([0, 1, 2]), seconds), np.repeat(np.arange(seconds), 3))

    a, b, c = chunk(2), chunk(2), chunk(4)
    alone = encoder.encode_chunks([a])[0]
    mixed = encoder.encode_chunks([c, a, b])
    for name in ("tokenizer", "gpt"):
        assert alone[name].shape == (6, 768) and mixed[0][name].shape == (12, 768)
        np.testing.assert_allclose(alone[name], mixed[1][name], atol=1e-4)
    # GroupNorm in TemporalConv pools over all tokens, so zero padding would change
    # real-token outputs; this is why encode_chunks never pads.
    x, chans, times = a
    padded = encoder.forward(
        torch.from_numpy(np.vstack([x, np.zeros((6, 200), np.float32)]))[None],
        torch.from_numpy(np.concatenate([chans, np.full(6, 136)]))[None],
        torch.from_numpy(np.concatenate([times, np.zeros(6, int)]))[None],
        torch.tensor([[True] * 6 + [False] * 6]),
    )
    assert not np.allclose(padded["tokenizer"][0, :6].numpy(), alone["tokenizer"], atol=1e-4)
