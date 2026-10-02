"""Do a text model's word vectors predict word-level EEG on unseen sentences?

For every model and layer: ridge regression from the word vectors to the top-k
principal components of reader-averaged word EEG, scored on held-out sentences
(outer folds split by sentence). The ridge penalty is chosen by inner folds,
also split by sentence. Choosing it by leave-one-word-out instead lets words of
the same sentence inform each other: they share an EEG offset (same trials,
same moment of the recording) and similar contextual vectors, so the selection
rewards memorising sentences and the model fails on new ones.

Target modes:
  raw       word EEG as is (includes differences between sentences)
  centered  each sentence's mean removed from its words' EEG and vectors, so
            only word-to-word differences within a sentence are predicted
Control: the same pipeline with the EEG targets shuffled across training words.

Encoders and decoder-only LMs are both supported: any transformer's hidden
states can serve as word vectors. A word's vector is the mean of its sub-word
states (encoders) or the state of its last sub-word (decoders, where that is the
first position that has read the whole word).
"""

import numpy as np
import pandas as pd
import torch

from .data import fit_targets, grouped_splits

ALPHAS = np.logspace(0, 7, 15)


def _progress(iterable=None, **kwargs):
    from tqdm.auto import tqdm

    return tqdm(iterable, mininterval=1.0, dynamic_ncols=True, **kwargs)


# ----------------------------------------------------------------------------
# Word vectors from every layer
# ----------------------------------------------------------------------------


def char_word_index(words):
    """Text with single spaces between words and the word index of every character (-1 = space)."""
    text = " ".join(words)
    char_word = np.full(len(text), -1, dtype=np.int64)
    position = 0
    for index, word in enumerate(words):
        char_word[position:position + len(word)] = index
        position += len(word) + 1
    return text, char_word


def token_word_map(offsets, char_word):
    """Word index of each token: the first non-space character it covers (-1 for none)."""
    out = []
    for start, end in offsets:
        word = -1
        for c in range(start, min(end, len(char_word))):
            if char_word[c] >= 0:
                word = int(char_word[c])
                break
        out.append(word)
    return out


def extract_layer_vectors(sentences, item_sentence, item_word, model_name, device="cpu", pool="auto",
                          batch_size=16, revision=None, desc="word vectors"):
    """Word vectors of the given (sentence, word) items from every hidden layer.

    ``sentences``: list of word lists; ``item_sentence``/``item_word``: index
    into it per item. Returns ``(vectors [n_layers + 1, n_items, dim] float32,
    found [n_items] bool, info)``; ``found`` is False for words that produced no
    token (empty, or cut by truncation).
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if not tokenizer.is_fast:
        raise ValueError(f"{model_name}: needs a fast tokenizer (character offsets) to align words")
    model = AutoModel.from_pretrained(model_name, revision=revision, torch_dtype=torch.float32).to(device).eval()
    causal = any(a.endswith("ForCausalLM") for a in (getattr(model.config, "architectures", None) or []))
    pool = ("last" if causal else "mean") if pool == "auto" else pool
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    items_of = {}
    for row, (s, w) in enumerate(zip(item_sentence, item_word)):
        items_of.setdefault(int(s), []).append((row, int(w)))
    order = sorted(items_of)
    vectors, found = None, np.zeros(len(item_sentence), dtype=bool)
    for start in _progress(range(0, len(order), batch_size), desc=desc, unit="batch"):
        batch = order[start:start + batch_size]
        texts, char_maps = zip(*(char_word_index(sentences[s]) for s in batch))
        encoded = tokenizer(list(texts), return_offsets_mapping=True, padding=True, truncation=True,
                            max_length=512, return_tensors="pt")
        offsets = encoded.pop("offset_mapping").tolist()
        with torch.no_grad():
            hidden = model(**{k: v.to(device) for k, v in encoded.items()}, output_hidden_states=True).hidden_states
        hidden = torch.stack(hidden)  # [layers, batch, tokens, dim]
        if vectors is None:
            vectors = np.zeros((hidden.shape[0], len(item_sentence), hidden.shape[-1]), dtype=np.float32)
        mask = encoded["attention_mask"].bool().tolist()
        for b, s in enumerate(batch):
            positions = {}
            for t, word in enumerate(token_word_map(offsets[b], char_maps[b])):
                if word >= 0 and mask[b][t]:
                    positions.setdefault(word, []).append(t)
            for row, w in items_of[s]:
                tokens = positions.get(w)
                if not tokens:
                    continue
                state = hidden[:, b, tokens[-1]] if pool == "last" else hidden[:, b, tokens].mean(dim=1)
                vectors[:, row] = state.float().cpu().numpy()
                found[row] = True
    info = {"model": model_name, "revision": revision, "pool": pool, "causal": causal,
            "n_layers": int(vectors.shape[0]), "dim": int(vectors.shape[-1])}
    del model, hidden
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vectors, found, info


# ----------------------------------------------------------------------------
# Sentence-grouped ridge
# ----------------------------------------------------------------------------


def center_by_group(values, codes):
    """Subtract each group's mean (numpy or torch, rows grouped by integer ``codes``)."""
    if isinstance(values, torch.Tensor):
        index = torch.as_tensor(codes, device=values.device)
        n_groups = int(index.max()) + 1
        sums = torch.zeros((n_groups, values.shape[1]), dtype=values.dtype, device=values.device)
        sums.index_add_(0, index, values)
        counts = torch.bincount(index, minlength=n_groups).to(values.dtype)
        return values - (sums / counts[:, None])[index]
    n_groups = int(codes.max()) + 1
    sums = np.zeros((n_groups, values.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, values)
    counts = np.bincount(codes, minlength=n_groups)
    return (values - (sums / counts[:, None])[codes]).astype(values.dtype)


def ridge_predictor(X_train, X_test):
    """Ridge predictions for any targets and many penalties from one eigendecomposition.

    Primal form (``X'X``) when there are more rows than features, dual form
    (``XX'``) otherwise; both give the same predictions. Features are centered
    with the training mean, so the intercept is the training target mean.
    """
    mean = X_train.mean(dim=0)
    X_train, X_test = X_train - mean, X_test - mean
    if X_train.shape[0] >= X_train.shape[1]:
        eigvals, vecs = torch.linalg.eigh(X_train.T @ X_train)
        left = X_test @ vecs

        def project(Y):
            return vecs.T @ (X_train.T @ Y)
    else:
        eigvals, vecs = torch.linalg.eigh(X_train @ X_train.T)
        left = (X_test @ X_train.T) @ vecs

        def project(Y):
            return vecs.T @ Y
    eigvals = eigvals.clamp_min(0)

    def predict(Y_train, alphas):
        offset = Y_train.mean(dim=0)
        projected = project(Y_train - offset)
        scale = 1.0 / (eigvals[None, :] + alphas[:, None])  # [alphas, rank]
        return torch.einsum("mr,ar,rk->amk", left, scale, projected) + offset

    return predict


def prepare_folds(eeg, groups, modes=("centered", "raw"), k=32, n_folds=5, n_inner=4, seed=0):
    """Outer/inner sentence-grouped splits and, per mode and fold, PCA targets fit on training items."""
    codes = np.unique(groups, return_inverse=True)[1]
    outer = grouped_splits(groups, n_folds, seed)
    prepared = {"codes": codes, "modes": {}}
    for mode in modes:
        Y = center_by_group(eeg.astype(np.float64), codes) if mode == "centered" else eeg
        folds = []
        for f, (train, test) in enumerate(outer):
            transform, explained = fit_targets(Y[train], k)
            targets = transform(Y)
            rng = np.random.default_rng(seed * 1000 + f)
            inner = grouped_splits(groups[train], n_inner, seed + f + 1)
            folds.append({"train": train, "test": test, "inner": inner, "explained": explained,
                          "real_train": targets[train], "real_test": targets[test],
                          "shuffled_train": targets[train][rng.permutation(len(train))],
                          "test_codes": np.unique(codes[test], return_inverse=True)[1]})
        prepared["modes"][mode] = folds
    return prepared


def _sentence_stats(Y, P_real, P_shuffled, codes):
    """Per-sentence sums needed to recompute held-out R^2 under any resampling of sentences."""
    index = torch.as_tensor(codes, device=Y.device)
    n = int(index.max()) + 1

    def scatter(values):
        out = torch.zeros((n,) + tuple(values.shape[1:]), dtype=values.dtype, device=values.device)
        return out.index_add_(0, index, values)

    return {"n": torch.bincount(index, minlength=n).double().cpu().numpy(),
            "sum_y": scatter(Y).cpu().numpy(),
            "sum_y2": scatter((Y ** 2).sum(dim=1, keepdim=True))[:, 0].cpu().numpy(),
            "sse_real": scatter(((P_real - Y) ** 2).sum(dim=1, keepdim=True))[:, 0].cpu().numpy(),
            "sse_shuffled": scatter(((P_shuffled - Y) ** 2).sum(dim=1, keepdim=True))[:, 0].cpu().numpy()}


def scan_layer(X, folds, codes, mode, alphas=ALPHAS, device="cpu", on_fold=None):
    """Held-out-sentence encoding for one layer: per-fold sentence stats and chosen penalties."""
    alphas_t = torch.as_tensor(alphas, dtype=torch.float64, device=device)
    Xt = torch.tensor(np.asarray(X), dtype=torch.float64, device=device)  # copy: inputs may be read-only
    if mode == "centered":
        Xt = center_by_group(Xt, codes)
    results = []
    for fold in folds:
        Xtr, Xte = Xt[fold["train"]], Xt[fold["test"]]
        mean, std = Xtr.mean(dim=0), Xtr.std(dim=0).clamp_min(1e-8)
        Xtr, Xte = (Xtr - mean) / std, (Xte - mean) / std
        tensors = {key: torch.as_tensor(fold[key], dtype=torch.float64, device=device)
                   for key in ("real_train", "shuffled_train", "real_test")}
        sse = torch.zeros((2, len(alphas)), dtype=torch.float64, device=device)
        for a, b in fold["inner"]:
            predict = ridge_predictor(Xtr[a], Xtr[b])
            for row, key in enumerate(("real_train", "shuffled_train")):
                Y = tensors[key]
                sse[row] += ((predict(Y[a], alphas_t) - Y[b][None]) ** 2).sum(dim=(1, 2))
        best = sse.argmin(dim=1).tolist()
        predict = ridge_predictor(Xtr, Xte)
        P_real = predict(tensors["real_train"], alphas_t[best[0]:best[0] + 1])[0]
        P_shuffled = predict(tensors["shuffled_train"], alphas_t[best[1]:best[1] + 1])[0]
        Y = tensors["real_test"]
        stats = _sentence_stats(Y, P_real, P_shuffled, fold["test_codes"])
        results.append({"stats": stats, "alpha_real": float(alphas[best[0]]),
                        "alpha_shuffled": float(alphas[best[1]]),
                        "sse_component": ((P_real - Y) ** 2).sum(dim=0).cpu().numpy(),
                        "sst_component": ((Y - Y.mean(dim=0)) ** 2).sum(dim=0).cpu().numpy()})
        if on_fold is not None:
            on_fold()
    return results


def _r2(stats, weights):
    n = weights @ stats["n"]
    sst = weights @ stats["sum_y2"] - ((weights @ stats["sum_y"]) ** 2).sum(axis=-1) / n
    return 1 - (weights @ stats["sse_real"]) / sst, 1 - (weights @ stats["sse_shuffled"]) / sst


def summarize_layer(fold_results, n_boot=1000, seed=0, alphas=ALPHAS):
    """Mean held-out R^2 over folds, with sentence-bootstrap 95% CIs (resampled within each fold)."""
    rng = np.random.default_rng(seed)
    real_points, shuffled_points, real_boot, shuffled_boot = [], [], [], []
    for result in fold_results:
        stats = result["stats"]
        n_sentences = len(stats["n"])
        real, shuffled = _r2(stats, np.ones(n_sentences))
        real_points.append(real)
        shuffled_points.append(shuffled)
        weights = rng.multinomial(n_sentences, np.full(n_sentences, 1.0 / n_sentences), size=n_boot).astype(float)
        boot_real, boot_shuffled = _r2(stats, weights)
        real_boot.append(boot_real)
        shuffled_boot.append(boot_shuffled)
    real_boot, shuffled_boot = np.mean(real_boot, axis=0), np.mean(shuffled_boot, axis=0)
    real, shuffled = float(np.mean(real_points)), float(np.mean(shuffled_points))
    alpha_real = [r["alpha_real"] for r in fold_results]
    return {"r2": real, "r2_ci_low": float(np.percentile(real_boot, 2.5)),
            "r2_ci_high": float(np.percentile(real_boot, 97.5)),
            "r2_shuffled": shuffled, "delta": real - shuffled,
            "delta_ci_low": float(np.percentile(real_boot - shuffled_boot, 2.5)),
            "delta_ci_high": float(np.percentile(real_boot - shuffled_boot, 97.5)),
            "fold_r2": [float(v) for v in real_points],
            "component_r2": [float(v) for v in np.mean([1 - r["sse_component"] / r["sst_component"]
                                                        for r in fold_results], axis=0)],
            "alpha_real_median": float(np.median(alpha_real)),
            "alpha_shuffled_median": float(np.median([r["alpha_shuffled"] for r in fold_results])),
            "share_alpha_at_max": float(np.mean(np.isclose(alpha_real, alphas[-1])))}


# ----------------------------------------------------------------------------
# Controls: lexical and reading-behaviour features, noise ceiling
# ----------------------------------------------------------------------------


def strip_punctuation(word):
    """Remove leading/trailing non-word characters (any script); keeps inner ZWNJ and apostrophes."""
    import re

    return re.sub(r"^[^\w]+|[^\w]+$", "", str(word))


def word_controls(sentences, items, meta, readers_per_sentence, lang):
    """Per-item control features: (table, lexical columns, reading columns).

    Lexical (from text only): log length, Zipf frequency (wordfreq, ``lang``)
    and its square, relative position, first/last word, punctuation attached.
    Reading behaviour (from the eye tracker): share of readers who fixated the
    word, mean number of fixations, and log mean total reading time and
    first-fixation duration when available.
    """
    from wordfreq import zipf_frequency

    words = [sentences[s][w] for s, w in zip(items["sentence_row"], items["word_index"])]
    lengths = np.array([len(sentences[s]) for s in items["sentence_row"]])
    bare = [strip_punctuation(w) for w in words]
    zipf = np.array([zipf_frequency(b, lang) if b else 0.0 for b in bare])
    position = items["word_index"].to_numpy()
    table = pd.DataFrame({
        "log_length": np.log1p([len(b) for b in bare]),
        "zipf": zipf, "zipf_sq": zipf ** 2,
        "relative_position": position / np.maximum(lengths - 1, 1),
        "is_first": (position == 0).astype(float), "is_last": (position == lengths - 1).astype(float),
        "punctuation": np.array([b != w for b, w in zip(bare, words)], dtype=float),
    })
    lexical = list(table.columns)
    by_item = meta.groupby("item")
    keys = items["item"]
    table["share_fixated"] = (items["n_readers"].to_numpy()
                              / items["sentence_id"].map(readers_per_sentence).to_numpy())
    table["n_fixations"] = by_item["n_fixations"].mean().loc[keys].to_numpy()
    reading = ["share_fixated", "n_fixations"]
    for column, name in (("trt_ms", "log_trt"), ("ffd_ms", "log_ffd")):
        if column not in meta:
            continue
        values = by_item[column].mean().loc[keys].to_numpy()
        if np.isfinite(values).all() and (values > 0).all():
            table[name] = np.log(values)
            reading.append(name)
    return table, lexical, reading


def residualize_folds(folds, L, codes, centered=True):
    """Folds whose targets are what training-set OLS on ``L`` cannot explain (train fit only)."""
    L = center_by_group(np.asarray(L, dtype=np.float64), codes) if centered else np.asarray(L, dtype=np.float64)
    out = []
    for f, fold in enumerate(folds):
        design_train = np.column_stack([np.ones(len(fold["train"])), L[fold["train"]]])
        design_test = np.column_stack([np.ones(len(fold["test"])), L[fold["test"]]])
        beta = np.linalg.lstsq(design_train, fold["real_train"], rcond=None)[0]
        train = (fold["real_train"] - design_train @ beta).astype(np.float32)
        rng = np.random.default_rng(10_000 + f)
        out.append({**fold, "real_train": train, "real_test": (fold["real_test"] - design_test @ beta).astype(np.float32),
                    "shuffled_train": train[rng.permutation(len(train))]})
    return out


def noise_ceiling(meta, Z, eeg_items, groups, k=32, centered=True, n_splits=20, seed=0):
    """Split-half reliability (Spearman-Brown to all readers) of each EEG target component.

    The components are a PCA of the reader-averaged item EEG (all items); each
    reader's word EEG is projected onto them (and centered within that reader's
    sentence when ``centered``). The mean over standardized components bounds
    the variance-weighted R^2 any model can reach.
    """
    from ..diagnostics.signal import split_half_reliability

    codes = np.unique(groups, return_inverse=True)[1]
    Y = center_by_group(eeg_items.astype(np.float64), codes) if centered else eeg_items
    mean = Y.mean(axis=0)
    _, _, vt = np.linalg.svd(Y - mean, full_matrices=False)
    projected = (Z - Z.mean(axis=0)) @ vt[:k].T
    if centered:
        reader_sentence = pd.factorize(meta["reader"].astype(str) + "|" + meta["sentence_id"].astype(str))[0]
        projected = center_by_group(projected, reader_sentence)
    counts = meta.groupby("item").size()
    if (counts >= 4).sum() < 10:
        return None, {"items": int((counts >= 4).sum())}  # too few words read by >= 4 readers
    reliability, details = split_half_reliability(meta, projected.astype(np.float32), n_splits=n_splits, seed=seed)
    return reliability["reliability_all_readers"].to_numpy(), details
