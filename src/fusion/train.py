"""Train and evaluate one arm on one fold.

Every arm starts from the same initial trainable weights (LoRA + projector),
sees the same batch order, and stops at the step with the best validation
macro-F1; the test sentences are scored once.
"""

import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .data import batch_inputs


@dataclass
class TrainConfig:
    epochs: float = 2.0
    batch_size: int = 16
    eval_batch_size: int = 32
    lr_lora: float = 2e-4
    lr_projector: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    evals_per_epoch: int = 4
    max_grad_norm: float = 1.0
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def precision_for(device):
    if device.type != "cuda":
        return torch.float32, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")


def _autocast(device, dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def predict(model, rows, data, inputs, use_eeg, device, dtype, batch_size):
    model.eval()
    probs = []
    with torch.no_grad(), _autocast(device, dtype):
        for start in range(0, len(rows), batch_size):
            batch = list(rows[start:start + batch_size])
            words, eeg, counts = batch_inputs(batch, data, inputs)
            logits = model(words, eeg, counts, use_eeg=use_eeg)
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs)


def run_fold(model, init_state, data, inputs, split, use_eeg, cfg, device, n_classes=3, log=print):
    """Return test probabilities and a training history for one fold."""
    model.load_trainable_state(init_state)
    torch.manual_seed(cfg.seed)
    dtype, scaler = precision_for(device)
    labels = data.samples["label_id"].to_numpy()
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    projector = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    groups = [{"params": lora, "lr": cfg.lr_lora}]
    if use_eeg:
        groups.append({"params": projector, "lr": cfg.lr_projector})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    train = np.asarray(split.train)
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total = max(1, int(round(cfg.epochs * steps_per_epoch)))
    warmup = max(1, int(cfg.warmup_ratio * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (total - s) / max(1, total - warmup))))
    counts = np.bincount(labels[train], minlength=n_classes).astype(np.float64)
    weight = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    eval_every = max(1, steps_per_epoch // cfg.evals_per_epoch)
    rng = np.random.default_rng(cfg.seed)
    order = np.concatenate([rng.permutation(train) for _ in range(math.ceil(cfg.epochs) + 1)])
    best = (-1.0, None, 0)
    history = []
    started = time.time()
    model.train()
    for step in range(total):
        batch = list(order[step * cfg.batch_size:(step + 1) * cfg.batch_size])
        words, eeg, word_counts = batch_inputs(batch, data, inputs)
        target = torch.as_tensor(labels[batch], device=device)
        with _autocast(device, dtype):
            logits = model(words, eeg, word_counts, use_eeg=use_eeg)
        loss = F.cross_entropy(logits.float(), target, weight=weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}; rerun with --base-dtype float32")
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], cfg.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        if (step + 1) % eval_every == 0 or step + 1 == total:
            val_probs = predict(model, split.val, data, inputs, use_eeg, device, dtype, cfg.eval_batch_size)
            score = f1_score(labels[split.val], val_probs.argmax(1), average="macro",
                             labels=list(range(n_classes)), zero_division=0)
            history.append({"step": step + 1, "loss": loss.item(), "val_macro_f1": float(score)})
            if score > best[0] + 1e-9:
                best = (score, model.trainable_state(), step + 1)
            model.train()
    model.load_trainable_state(best[1])
    test_probs = predict(model, split.test, data, inputs, use_eeg, device, dtype, cfg.eval_batch_size)
    log(f"    best val macro-F1 {best[0]:.3f} at step {best[2]}/{total} ({time.time() - started:.0f}s)")
    return test_probs, {"history": history, "best_step": best[2], "best_val_macro_f1": best[0],
                        "total_steps": total, "seconds": time.time() - started}
