"""Pool token-level hidden states into one vector per trial.

Pooling runs per chunk on valid tokens only, then chunks of one trial are
combined: means are token-weighted, maxima take the element-wise maximum.
``last_step`` averages the tokens of the final second; under NeuroLM's
stair-stepping mask these tokens have attended to the whole chunk, which makes
them the closest causal analogue of a CLS token.
"""

import numpy as np

POOLINGS = ("mean", "max", "last_step")


def pool_chunk(hidden, times, poolings):
    """``hidden`` [tokens, dim] (valid tokens only) -> ``{pooling: vector}``."""
    hidden = np.asarray(hidden, dtype=np.float32)
    out = {}
    for pooling in poolings:
        if pooling == "mean":
            out[pooling] = hidden.mean(axis=0)
        elif pooling == "max":
            out[pooling] = hidden.max(axis=0)
        elif pooling == "last_step":
            last = np.asarray(times) == np.max(times)
            out[pooling] = hidden[last].mean(axis=0)
        else:
            raise ValueError(f"unknown pooling {pooling!r}; choose from {POOLINGS}")
    return out


def combine_chunks(pooled_chunks, token_counts):
    """Merge per-chunk pooled vectors of one trial."""
    weights = np.asarray(token_counts, dtype=np.float64)
    weights = weights / weights.sum()
    combined = {}
    for pooling in pooled_chunks[0]:
        stack = np.stack([chunk[pooling] for chunk in pooled_chunks])
        if pooling == "max":
            combined[pooling] = stack.max(axis=0)
        else:
            combined[pooling] = (stack * weights[:, None]).sum(axis=0).astype(np.float32)
    return combined


def feature_key(representation, pooling):
    return f"{representation}__{pooling}"


def concat_features(features, keys):
    """Build derived feature sets such as mean+max by concatenation."""
    return np.concatenate([features[key] for key in keys], axis=1)
