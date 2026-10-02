"""Version B: brain-tuning a LoRA LLM (EEG is a training signal only).

The sentiment model is the text-only LoRA classifier. An auxiliary linear head
reads the hidden state at each word's last sub-word token (layer
``aux_layer``) and predicts that word's EEG target (reader-averaged EEG
principal components). Loss = cross-entropy + lambda * masked MSE. At test
time only text is used. Arms differ only in the auxiliary targets.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from ..fusion.model import FusionClassifier
from ..fusion.train import _autocast, precision_for


class BrainTunedClassifier(FusionClassifier):
    def __init__(self, lm, tokenizer, *, class_words, prompt, lora, aux_dim, aux_layer):
        super().__init__(lm, tokenizer, class_words=class_words, eeg_dim=1, prompt=prompt, lora=lora,
                         projector={"hidden": 1})
        for parameter in self.projector.parameters():
            parameter.requires_grad_(False)  # no EEG input in this model
        self.aux_head = nn.Linear(lm.get_input_embeddings().weight.shape[1], aux_dim)
        self.aux_layer = aux_layer

    def word_positions(self, words):
        position, positions = len(self.prefix_ids), []
        for word in words:
            position += len(self._encode_word(word))
            positions.append(position - 1)
        return positions

    def forward_with_aux(self, batch_words):
        prepared = self.prepare(batch_words, use_eeg=False)
        out = self.decoder(inputs_embeds=prepared["embeds"], attention_mask=prepared["mask"],
                           output_hidden_states=True)
        logits = self.classify(out.last_hidden_state, prepared["lengths"])
        hidden = out.hidden_states[self.aux_layer]
        rows = [hidden[b, self.word_positions(words)] for b, words in enumerate(batch_words)]
        return logits, self.aux_head(torch.cat(rows).float())


def _batch(rows, words, targets):
    batch_words = [words[i] for i in rows]
    target = np.concatenate([targets[i] for i in rows])
    return batch_words, target


def predict(model, rows, words, device, dtype, batch_size=32):
    model.eval()
    out = []
    with torch.no_grad(), _autocast(device, dtype):
        for start in range(0, len(rows), batch_size):
            batch = [words[i] for i in rows[start:start + batch_size]]
            out.append(torch.softmax(model(batch).float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def aux_r2(model, rows, words, real_targets, device, dtype, batch_size=32):
    """R^2 of the auxiliary head against the *real* EEG targets of held-out words."""
    model.eval()
    predictions, truth = [], []
    with torch.no_grad(), _autocast(device, dtype):
        for start in range(0, len(rows), batch_size):
            chunk = list(rows[start:start + batch_size])
            _, aux = model.forward_with_aux([words[i] for i in chunk])
            predictions.append(aux.float().cpu().numpy())
            truth.append(np.concatenate([real_targets[i] for i in chunk]))
    pred, true = np.concatenate(predictions), np.concatenate(truth)
    keep = np.isfinite(true).all(axis=1)
    if keep.sum() < 10:
        return float("nan")
    residual = ((true[keep] - pred[keep]) ** 2).sum()
    total = ((true[keep] - true[keep].mean(axis=0)) ** 2).sum()
    return float(1 - residual / total)


def run_brain_fold(model, init_state, words, labels, targets, split, aux_weight, cfg, device, log=print):
    """Train one arm on one fold; ``targets`` per sentence: [n_words, k] (NaN rows = no EEG)."""
    model.load_trainable_state(init_state)
    torch.manual_seed(cfg.seed)
    dtype, scaler = precision_for(device)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("aux_head")]
    groups = [{"params": lora, "lr": cfg.lr_lora}]
    if aux_weight > 0:
        groups.append({"params": head, "lr": cfg.lr_projector})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    train = np.asarray(split.train)
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total = max(1, int(round(cfg.epochs * steps_per_epoch)))
    warmup = max(1, int(cfg.warmup_ratio * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total - s) / max(1, total - warmup))))
    counts = np.bincount(labels[train], minlength=3).astype(np.float64)
    weight = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    eval_every = max(1, steps_per_epoch // cfg.evals_per_epoch)
    rng = np.random.default_rng(cfg.seed)
    order = np.concatenate([rng.permutation(train) for _ in range(math.ceil(cfg.epochs) + 1)])
    best = (-1.0, None, 0)
    history = []
    started = time.time()
    model.train()
    for step in range(total):
        batch = list(order[step * cfg.batch_size:(step + 1) * cfg.batch_size])
        batch_words, target = _batch(batch, words, targets)
        y = torch.as_tensor(labels[batch], device=device)
        with _autocast(device, dtype):
            logits, aux = model.forward_with_aux(batch_words)
        loss = F.cross_entropy(logits.float(), y, weight=weight)
        aux_loss = torch.zeros((), device=device)
        if aux_weight > 0:
            target = torch.as_tensor(target, device=device)
            keep = torch.isfinite(target).all(dim=1)
            if keep.any():
                # aux comes out of the autocast region in bf16/fp16; compute the loss in float32
                aux_loss = F.mse_loss(aux[keep].float(), target[keep].float())
        total_loss = loss + aux_weight * aux_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}; rerun with --base-dtype float32")
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
        else:
            total_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], cfg.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        if (step + 1) % eval_every == 0 or step + 1 == total:
            val = predict(model, split.val, words, device, dtype)
            score = f1_score(labels[split.val], val.argmax(1), average="macro", labels=[0, 1, 2], zero_division=0)
            history.append({"step": step + 1, "loss": loss.item(), "aux_loss": aux_loss.item(),
                            "val_macro_f1": float(score)})
            if score > best[0] + 1e-9:
                best = (score, model.trainable_state(), step + 1)
            model.train()
    model.load_trainable_state(best[1])
    test = predict(model, split.test, words, device, dtype)
    log(f"    best val macro-F1 {best[0]:.3f} at step {best[2]}/{total} ({time.time() - started:.0f}s)")
    return test, {"history": history, "best_step": best[2], "best_val_macro_f1": best[0], "total_steps": total}
