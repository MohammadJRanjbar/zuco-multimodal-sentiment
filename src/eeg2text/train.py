"""Training and evaluation loops for EEG-to-text."""

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import partial

import numpy as np
import torch

from ..progress import progress
from .augment import Augment, Augmenter


@dataclass
class Settings:
    epochs: int = 25
    batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 3e-4          # input projections, missing vectors, EEG tokenizer
    lr_lora: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    patience: int = 4
    min_epochs: int = 6       # no early stop before this: validation loss can rise while the input layer still learns
    max_grad_norm: float = 1.0
    seed: int = 42
    max_new_tokens: int = 128
    num_beams: int = 1
    augment: Augment = field(default_factory=Augment)

    def to_dict(self):
        return asdict(self)


def precision_for(device):
    """bf16 autocast on GPUs with native bf16 (compute capability 8+: A100, L4, ...), else float32.

    fp16 training of mBART is unstable, and ``torch.cuda.is_bf16_supported()`` is also True on older GPUs
    such as the T4 that only emulate bf16 (bf16 precision without its speed)."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16, None
    return torch.float32, None


def autocast(device, dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" and dtype != torch.float32 \
        else nullcontext()


def collate(items, device):
    """Pad a list of trials: x [B, L, F], fixated/valid [B, L], labels [B, T] (-100 padding)."""
    length = max(len(item["x"]) for item in items)
    width = items[0]["x"].shape[1]
    x = np.zeros((len(items), length, width), dtype=np.float32)
    fixated = np.zeros((len(items), length), dtype=bool)
    valid = np.zeros((len(items), length), dtype=bool)
    target = max(len(item["labels"]) for item in items)
    labels = np.full((len(items), target), -100, dtype=np.int64)
    for b, item in enumerate(items):
        n = len(item["x"])
        x[b, :n], fixated[b, :n], valid[b, :n] = item["x"], item["fixated"], True
        labels[b, :len(item["labels"])] = item["labels"]
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return t(x), t(fixated), t(valid), t(labels)


def _encode_one(codec, text, lang):
    return codec.encode([text], lang)[0]


def _batches(n, size, rng):
    order = rng.permutation(n)
    return [order[i:i + size] for i in range(0, n, size)]


def epoch_plan(train_by_lang, batch_size, rng):
    """(lang, indices) batches; in joint training each language gets as many batches as the larger one."""
    plans = {lang: _batches(len(items), batch_size, rng) for lang, items in train_by_lang.items()}
    longest = max(len(p) for p in plans.values())
    for lang, plan in plans.items():
        while len(plan) < longest:  # oversample the smaller language with fresh shuffles
            plan.extend(_batches(len(train_by_lang[lang]), batch_size, rng))
        plans[lang] = plan[:longest]
    steps = []
    for i in range(longest):
        for lang in plans:
            steps.append((lang, plans[lang][i]))
    return steps


@torch.no_grad()
def validation_loss(model, items_by_lang, settings, device, dtype):
    model.eval()
    losses = []
    for lang, items in items_by_lang.items():
        total, count = 0.0, 0
        for start in range(0, len(items), settings.eval_batch_size):
            batch = items[start:start + settings.eval_batch_size]
            x, fixated, valid, labels = collate(batch, device)
            with autocast(device, dtype):
                loss, _, _ = model(x, fixated, valid, labels, lang)
            total += float(loss) * len(batch)
            count += len(batch)
        losses.append(total / max(count, 1))
    return float(np.mean(losses))


def train(model, train_by_lang, val_by_lang, settings, device, log=print):
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    dtype, scaler = precision_for(device)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    other = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    groups = [{"params": other, "lr": settings.lr}] + ([{"params": lora, "lr": settings.lr_lora}] if lora else [])
    optimizer = torch.optim.AdamW(groups, weight_decay=settings.weight_decay)
    steps_per_epoch = len(epoch_plan(train_by_lang, settings.batch_size, np.random.default_rng(0)))
    total = settings.epochs * steps_per_epoch
    warmup = max(1, int(settings.warmup_ratio * total))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / total))))
    augment_rng = np.random.default_rng(settings.seed + 1)
    augmenters = {lang: Augmenter(items, partial(_encode_one, model.codec, lang=lang), settings.augment, augment_rng)
                  for lang, items in train_by_lang.items()} if settings.augment.active else None
    best, best_state, waited, history = math.inf, model.trainable_state(), 0, []
    for epoch in range(settings.epochs):
        model.train()
        running = []
        for lang, index in progress(epoch_plan(train_by_lang, settings.batch_size, rng),
                                    desc=f"epoch {epoch + 1}/{settings.epochs}", unit="batch", leave=False):
            batch = [augmenters[lang](i) if augmenters else train_by_lang[lang][i] for i in index]
            x, fixated, valid, labels = collate(batch, device)
            with autocast(device, dtype):
                loss, _, _ = model(x, fixated, valid, labels, lang)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], settings.max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            schedule.step()
            running.append(float(loss.detach()))
        val = validation_loss(model, val_by_lang, settings, device, dtype)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(running)), "val_loss": val})
        log(f"    epoch {epoch + 1}: train {np.mean(running):.3f}, val {val:.3f}")
        if val < best - 1e-4:
            best, best_state, waited = val, model.trainable_state(), 0
        else:
            waited += 1
            if waited >= settings.patience and epoch + 1 >= settings.min_epochs:
                break
    model.load_trainable(best_state)
    return {"best_val_loss": best, "history": history}


@torch.no_grad()
def evaluate(model, items, lang, settings, device, desc="evaluate"):
    """Per trial: teacher-forced token accuracy and text, free-running text, EEG-token codes (if any)."""
    model.eval()
    dtype, _ = precision_for(device)
    records, codes = [], []
    for start in progress(range(0, len(items), settings.eval_batch_size), desc=desc, unit="batch", leave=False):
        batch = items[start:start + settings.eval_batch_size]
        x, fixated, valid, labels = collate(batch, device)
        with autocast(device, dtype):
            correct, scored, tf_texts = model.teacher_forced(x, fixated, valid, labels, lang)
            free = model.generate(x, fixated, valid, lang, settings.max_new_tokens, settings.num_beams)
            if model.vq is not None:
                _, _, batch_codes = model.embed(x, fixated, lang)
                codes.append(batch_codes[fixated].cpu())
        for item, c, s, tf, fr in zip(batch, correct.tolist(), scored.tolist(), tf_texts, free):
            records.append({"trial": item["trial"], "tf_correct": c, "tf_scored": s, "tf_text": tf, "free_text": fr})
    return records, (torch.cat(codes) if codes else None)
