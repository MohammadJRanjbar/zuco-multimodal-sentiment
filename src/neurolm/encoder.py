"""Frozen NeuroLM wrapper that returns token-level hidden states.

One checkpoint (``NeuroLM-B.pt`` from ``huggingface.co/Weibang/NeuroLM``) holds
both representations used by the probe:

* ``tokenizer``: the frozen, text-aligned VQ encoder (LaBraM-style temporal
  convolution + 12 bidirectional Transformer blocks, 768-d ``fc_norm`` output).
  NeuroLM feeds exactly these features into its language model.
* ``gpt``: NeuroLM-B's GPT-2 backbone after multi-channel autoregressive
  pretraining (final ``ln_f`` output, 768-d).

Masks follow the released code. The tokenizer attends bidirectionally over
valid tokens (``NeuralTransformer`` falls back to *causal* attention when no
mask is given, so a mask is always passed). The GPT uses the stair-stepping
mask of ``dataset.PickleLoader``: a token sees every token of its own and
earlier seconds.

Chunks are never zero-padded. ``TemporalConv`` (the patch embedder) applies
``GroupNorm`` over *all* tokens of a sample, so padding tokens would change the
normalization statistics of the real tokens, and by an amount that depends on
trial length. Chunks are therefore batched only with chunks of exactly the same
token count, which makes every embedding independent of batch composition.
NeuroLM's pretraining samples were full ``floor(1024 / C)``-second windows with
at most a few padding tokens, so unpadded full chunks match that regime.
"""

import hashlib
import json
import os
import sys

import numpy as np

from .pooling import combine_chunks, feature_key, pool_chunk


def file_sha256(path, chunk_size=1 << 24):
    sidecar = path + ".sha256.json"
    stat = os.stat(path)
    if os.path.exists(sidecar):
        cached = json.load(open(sidecar))
        if cached.get("size") == stat.st_size and cached.get("mtime") == int(stat.st_mtime):
            return cached["sha256"]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    value = digest.hexdigest()
    try:
        json.dump({"size": stat.st_size, "mtime": int(stat.st_mtime), "sha256": value}, open(sidecar, "w"))
    except OSError:
        pass
    return value


def import_neurolm(neurolm_dir):
    """Import NeuroLM's model classes from a cloned repository."""
    neurolm_dir = os.path.abspath(neurolm_dir)
    if not os.path.exists(os.path.join(neurolm_dir, "model", "model_neurolm.py")):
        raise FileNotFoundError(f"no NeuroLM repository at {neurolm_dir}")
    if neurolm_dir not in sys.path:
        sys.path.insert(0, neurolm_dir)
    from model.model import GPTConfig  # noqa: E402
    from model.model_neurolm import NeuroLM  # noqa: E402

    return NeuroLM, GPTConfig


def stair_mask(times, valid):
    """``[B, 1, N, N]`` bool: query i sees key j iff ``t_j <= t_i`` and j is valid."""
    return (times[:, None, :] <= times[:, :, None])[:, None] & valid[:, None, None, :]


def bidirectional_mask(valid):
    n = valid.shape[1]
    return valid[:, None, None, :].expand(valid.shape[0], 1, n, n)


class NeuroLMEncoder:
    def __init__(
        self,
        neurolm_dir,
        checkpoint_path,
        device="cuda",
        precision="float32",
        gpt_layers=(),
        model=None,
        checkpoint_info=None,
    ):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.precision = precision
        self.gpt_layers = tuple(int(layer) for layer in gpt_layers)
        if model is None:
            model, checkpoint_info = self._load(neurolm_dir, checkpoint_path)
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.checkpoint_info = checkpoint_info or {}
        self._captured = {}
        for layer in self.gpt_layers:
            self.model.GPT2.transformer.h[layer].register_forward_hook(self._hook(layer))

    def _hook(self, layer):
        def capture(_module, _inputs, output):
            self._captured[layer] = output
        return capture

    def _load(self, neurolm_dir, checkpoint_path):
        torch = self.torch
        NeuroLM, GPTConfig = import_neurolm(neurolm_dir)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_args = checkpoint["model_args"]
        fields = GPTConfig.__dataclass_fields__
        config = GPTConfig(**{k: v for k, v in model_args.items() if k in fields})
        model = NeuroLM(config, init_from="scratch")
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in checkpoint["model"].items()
        }
        model.load_state_dict(state, strict=True)
        info = {
            "checkpoint_name": os.path.basename(checkpoint_path),
            "checkpoint_bytes": os.path.getsize(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "model_args": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in model_args.items()},
            "checkpoint_keys": sorted(k for k in checkpoint if k not in {"model", "optimizer"}),
            "iter_num": int(checkpoint.get("iter_num", -1)),
            "epoch": int(checkpoint.get("epoch", -1)),
            "strict_state_dict_load": True,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        }
        del checkpoint
        return model, info

    @property
    def representations(self):
        return ["tokenizer", "gpt"] + [f"gpt_L{layer}" for layer in self.gpt_layers]

    def spatial_embeddings(self):
        return {
            "tokenizer.pos_embed": self.model.tokenizer.pos_embed.weight.detach().cpu().numpy(),
            "neurolm.pos_embed": self.model.pos_embed.weight.detach().cpu().numpy(),
        }

    def forward(self, x, chans, times, valid):
        """Token hidden states ``{representation: [B, N, D]}`` for one batch."""
        torch = self.torch
        autocast = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.precision == "bfloat16" and self.device.type == "cuda"
            else torch.autocast(device_type=self.device.type, enabled=False)
        )
        with torch.no_grad(), autocast:
            tokens = self.model.tokenizer(
                x, chans, times, bidirectional_mask(valid), return_all_tokens=True
            )
            eeg = self.model.encode_transform_layer(tokens) + self.model.pos_embed(chans)
            self._captured = {}
            hidden = self.model.GPT2(
                x_eeg=eeg, eeg_time_idx=times, eeg_mask=stair_mask(times, valid), lm_head=False
            )
        out = {"tokenizer": tokens.float(), "gpt": hidden.float()}
        for layer in self.gpt_layers:
            out[f"gpt_L{layer}"] = self._captured[layer].float()
        return out

    def encode_chunks(self, chunks, batch_size=16):
        """Encode ``[(x, chans, times)]``; return hidden states per chunk.

        Chunks are grouped by exact token count so no padding is ever added
        (see the module docstring on ``GroupNorm``).
        """
        torch = self.torch
        groups = {}
        for index, chunk in enumerate(chunks):
            groups.setdefault(len(chunk[0]), []).append(index)
        results = [None] * len(chunks)
        for _, members in sorted(groups.items()):
            for start in range(0, len(members), batch_size):
                batch_ids = members[start:start + batch_size]
                x = np.stack([chunks[i][0] for i in batch_ids]).astype(np.float32)
                chans = np.stack([chunks[i][1] for i in batch_ids]).astype(np.int64)
                times = np.stack([chunks[i][2] for i in batch_ids]).astype(np.int64)
                valid = np.ones(chans.shape, dtype=bool)
                hidden = self.forward(
                    torch.from_numpy(x).to(self.device),
                    torch.from_numpy(chans).to(self.device),
                    torch.from_numpy(times).to(self.device),
                    torch.from_numpy(valid).to(self.device),
                )
                for row, i in enumerate(batch_ids):
                    results[i] = {name: h[row].cpu().numpy() for name, h in hidden.items()}
        return results


def pool_trials(encoder, trial_tokens, poolings, batch_size=16):
    """Encode several trials and return one pooled feature dict per trial."""
    flat, owner = [], []
    for trial_index, tokens in enumerate(trial_tokens):
        for chunk in tokens.chunks:
            flat.append(chunk)
            owner.append(trial_index)
    hidden = encoder.encode_chunks(flat, batch_size=batch_size) if flat else []
    per_trial = [[] for _ in trial_tokens]
    for chunk, states, trial_index in zip(flat, hidden, owner):
        per_trial[trial_index].append((chunk, states))
    pooled = []
    for items in per_trial:
        if not items:
            pooled.append(None)
            continue
        features = {}
        counts = [len(chunk[0]) for chunk, _ in items]
        for representation in items[0][1]:
            chunk_pools = [pool_chunk(states[representation], chunk[2], poolings) for chunk, states in items]
            for pooling, vector in combine_chunks(chunk_pools, counts).items():
                features[feature_key(representation, pooling)] = vector
        pooled.append(features)
    return pooled
