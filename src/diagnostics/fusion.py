"""Stage 3: does the trained EEG + text LoRA model read its EEG tokens?

For held-out trials of one fold:

* attention from the answer position to prompt / word / EEG / answer tokens,
  per layer (total mass and mass per token);
* gradient x input attribution of the predicted class to each token type;
* counterfactual reliance: predictions with aligned EEG versus EEG shuffled
  among the reader's words, EEG from another reader of the same sentence, and
  feature-free EEG tokens (fixation flag only);
* what the model's EEG representations encode: ridge probes on the raw EEG
  input, the projector output, and hidden states at EEG slots, for word
  targets (valence, frequency, ...) and for reader identity.
"""

import numpy as np
import pandas as pd
import torch

from ..fusion.data import batch_inputs
from .signal import ridge_probe

ROLES = {0: "prompt", 1: "word", 2: "eeg", 3: "answer"}


def _batches(rows, size):
    for start in range(0, len(rows), size):
        yield list(rows[start:start + size])


def attention_and_states(model, rows, data, inputs, layers, batch_size=8):
    """Attention shares from the answer position, and states at EEG slots."""
    model.eval()
    attention_rows, slot_records = [], []
    with torch.no_grad():
        for batch in _batches(rows, batch_size):
            words, eeg, counts = batch_inputs(batch, data, inputs)
            out = model.analyze(words, eeg, counts, use_eeg=True, attentions=True)
            role, lengths = out["role"].cpu().numpy(), out["lengths"].cpu().numpy()
            for layer, attention in enumerate(out["attentions"]):
                weights = attention.float().mean(dim=1).cpu().numpy()  # [B, L, L] averaged over heads
                for b in range(len(batch)):
                    query = lengths[b] - 1
                    row_weights = weights[b, query, :lengths[b]]
                    for code, name in ROLES.items():
                        selected = role[b, :lengths[b]] == code
                        if selected.any():
                            attention_rows.append({"layer": layer, "role": name, "mass": float(row_weights[selected].sum()),
                                                   "per_token": float(row_weights[selected].mean())})
            soft = out["soft"].float().cpu().numpy()
            offset = 0
            for b, i in enumerate(batch):
                slot_positions = np.flatnonzero(role[b] == 2)
                for w, position in enumerate(slot_positions):
                    record = {"row": i, "word_index": w, "input": inputs[i][w], "projector": soft[offset + w]}
                    for layer in layers:
                        record[f"layer_{layer}"] = out["hidden_states"][layer][b, position].float().cpu().numpy()
                    slot_records.append(record)
                offset += len(slot_positions)
    summary = pd.DataFrame(attention_rows).groupby(["layer", "role"]).mean().reset_index()
    return summary, slot_records


def gradient_attribution(model, rows, data, inputs, batch_size=8):
    """|grad . embedding| of the predicted-class logit, shared out by token role."""
    model.eval()
    shares = []
    for batch in _batches(rows, batch_size):
        words, eeg, counts = batch_inputs(batch, data, inputs)
        prepared = model.prepare(words, eeg, counts, use_eeg=True)
        embeds = prepared["embeds"].detach().float().requires_grad_(True)
        hidden = model.decoder(inputs_embeds=embeds.to(prepared["embeds"].dtype),
                               attention_mask=prepared["mask"]).last_hidden_state
        logits = model.classify(hidden, prepared["lengths"])
        logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
        attribution = (embeds.grad * embeds).sum(-1).abs().detach().cpu().numpy()
        role = prepared["role"].cpu().numpy()
        for b in range(len(batch)):
            valid = role[b] >= 0
            total = attribution[b][valid].sum() or 1.0
            entry = {name: float(attribution[b][role[b] == code].sum() / total) for code, name in ROLES.items()}
            entry["eeg_per_token_over_word_per_token"] = float(
                attribution[b][role[b] == 2].mean() / max(attribution[b][role[b] == 1].mean(), 1e-12))
            shares.append(entry)
        model.zero_grad(set_to_none=True)
    return pd.DataFrame(shares)


def other_reader_inputs(data, inputs, rows, rng):
    """Replace each trial's EEG with another reader's EEG for the same sentence."""
    swapped = list(inputs)
    samples = data.samples
    by_sentence = samples.groupby("sentence_id").indices
    for i in rows:
        candidates = [j for j in by_sentence[samples["sentence_id"][i]] if j != i
                      and len(data.words[j]) == len(data.words[i])]
        if candidates:
            swapped[i] = inputs[int(rng.choice(candidates))]
    return swapped


def counterfactual_reliance(model, rows, data, variants, labels, batch_size=16):
    """Accuracy, flip rate, and logit change for each EEG variant vs aligned EEG."""
    model.eval()
    logits = {}
    with torch.no_grad():
        for name, inputs in variants.items():
            chunks = []
            for batch in _batches(rows, batch_size):
                words, eeg, counts = batch_inputs(batch, data, inputs)
                chunks.append(model(words, eeg, counts, use_eeg=True).float().cpu().numpy())
            logits[name] = np.concatenate(chunks)
    reference = logits["aligned"]
    out = []
    for name, values in logits.items():
        out.append({"variant": name, "accuracy": float((values.argmax(1) == labels).mean()),
                    "flip_rate_vs_aligned": float((values.argmax(1) != reference.argmax(1)).mean()),
                    "mean_abs_logit_change": float(np.abs(values - reference).mean())})
    return pd.DataFrame(out)


def probe_slot_representations(slot_records, data, word_table, layers, n_perm=20, seed=0):
    """What do raw EEG, projector output, and hidden EEG-slot states encode?"""
    samples = data.samples
    meta = pd.DataFrame({"sentence_id": [samples["sentence_id"][r["row"]] for r in slot_records],
                         "reader": [samples["subject_id"][r["row"]] for r in slot_records],
                         "word_index": [r["word_index"] for r in slot_records]})
    targets = meta.merge(word_table, on=["sentence_id", "word_index"], how="left")
    items = targets["sentence_id"].astype(str) + ":" + targets["word_index"].astype(str)
    reader_codes = pd.factorize(meta["reader"])[0]
    spaces = {"raw EEG input": np.stack([r["input"] for r in slot_records]),
              "projector output": np.stack([r["projector"] for r in slot_records])}
    for layer in layers:
        spaces[f"hidden layer {layer}"] = np.stack([r[f"layer_{layer}"] for r in slot_records])
    tasks = {"valence": "regression", "abs_valence": "regression", "zipf": "regression",
             "length": "regression", "is_content": "binary"}
    rows = []
    for space, X in spaces.items():
        for target, task in tasks.items():
            if target not in targets or targets[target].isna().all():
                continue
            y = targets[target].to_numpy()
            if task == "binary":
                y = y.astype(int)
            result = ridge_probe(X, y, groups=targets["sentence_id"], task=task, n_perm=n_perm,
                                 perm_units=items, seed=seed)
            rows.append({"space": space, "target": target, "metric": result["metric"], "score": result["score"],
                         "null_q95": result.get("null_q95"), "p_value": result.get("p_value")})
        reader = ridge_probe(X, reader_codes, groups=targets["sentence_id"], task="multiclass", seed=seed)
        rows.append({"space": space, "target": "reader identity", "metric": reader["metric"], "score": reader["score"],
                     "null_q95": None, "p_value": None})
    return pd.DataFrame(rows)
