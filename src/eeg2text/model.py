"""EEG sequence -> sentence with a multilingual seq2seq model (mBART-50: English and Persian).

Each word position gets one input vector, made by a per-language projection
of that word's EEG features (or a learned "missing" vector for words the
reader skipped). The vectors go to mBART's encoder as input embeddings. mBART
positional embeddings are added, and its decoder writes the sentence, starting
with the target-language token. Training uses LoRA on mBART's attention.

Optional EEG tokenizer (``vq``): the projected vector is snapped to the
nearest entry of a learned codebook (vector quantization, straight-through
gradients, commitment loss), so the encoder sees discrete "EEG tokens".
The codebook is shared by both languages in joint training.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fusion.model import add_lora

LANG_CODES = {"en": "en_XX", "fa": "fa_IR"}


class MBartCodec:
    """Target-text encoding/decoding with the mBART-50 tokenizer."""

    def __init__(self, name):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.pad_id = self.tokenizer.pad_token_id

    def lang_id(self, lang):
        return self.tokenizer.convert_tokens_to_ids(LANG_CODES[lang])

    def encode(self, texts, lang, max_length=128):
        self.tokenizer.tgt_lang = LANG_CODES[lang]
        return self.tokenizer(text_target=list(texts), truncation=True, max_length=max_length)["input_ids"]

    def decode(self, ids):
        return [t.strip() for t in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]


class VectorQuantizer(nn.Module):
    """Nearest-codebook-entry quantization with straight-through gradients."""

    def __init__(self, n_codes=512, dim=64, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)
        self.beta = beta

    def forward(self, z):
        flat = z.reshape(-1, z.shape[-1]).float()
        weight = self.codebook.weight.float()
        distance = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ weight.T + weight.pow(2).sum(1)[None]
        codes = distance.argmin(dim=1)
        quantized = weight[codes].view_as(z).to(z.dtype)
        loss = self.beta * F.mse_loss(z, quantized.detach()) + F.mse_loss(quantized, z.detach())
        return z + (quantized - z).detach(), codes.view(z.shape[:-1]), loss


class EEGToText(nn.Module):
    def __init__(self, seq2seq, codec, input_dims, *, lora=None, vq=None, hidden=1024, dropout=0.1,
                 full_finetune=False):
        super().__init__()
        self.lm, self.codec = seq2seq, codec
        d_model = seq2seq.config.d_model
        for parameter in self.lm.parameters():
            parameter.requires_grad_(full_finetune)
        if lora and lora.get("r", 0) > 0 and not full_finetune:
            add_lora(self.lm, set(lora["targets"]), lora["r"], lora["alpha"], lora.get("dropout", 0.0))
        width = vq["dim"] if vq else d_model
        self.inputs = nn.ModuleDict({lang: nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                                         nn.Linear(hidden, width))
                                     for lang, dim in input_dims.items()})
        self.missing = nn.ParameterDict({lang: nn.Parameter(torch.zeros(width)) for lang in input_dims})
        self.vq = VectorQuantizer(vq["codes"], vq["dim"], vq.get("beta", 0.25)) if vq else None
        self.up = nn.Linear(width, d_model) if vq else nn.Identity()

    def trainable_state(self):
        names = self._trainable_names()
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}

    def _trainable_names(self):
        return {name for name, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state):
        self.load_state_dict(state, strict=False)

    def embed(self, x, fixated, lang):
        h = self.inputs[lang](x)
        h = torch.where(fixated[..., None], h, self.missing[lang].to(h.dtype).expand_as(h))
        vq_loss, codes = h.new_zeros(()), None
        if self.vq is not None:
            h, codes, vq_loss = self.vq(h)
        return self.up(h), vq_loss, codes

    def forward(self, x, fixated, valid, labels, lang):
        embeds, vq_loss, codes = self.embed(x, fixated, lang)
        out = self.lm(inputs_embeds=embeds, attention_mask=valid.long(), labels=labels)
        return out.loss + vq_loss, out.logits, codes

    @torch.no_grad()
    def teacher_forced(self, x, fixated, valid, labels, lang):
        """Next-token argmax given the true previous tokens: per-token correctness and decoded text."""
        embeds, _, _ = self.embed(x, fixated, lang)
        logits = self.lm(inputs_embeds=embeds, attention_mask=valid.long(), labels=labels).logits
        predicted = logits.argmax(-1)
        scored = labels != -100
        scored[:, 0] = False  # the language token is given, not predicted from EEG
        correct = (predicted == labels) & scored
        texts = self.codec.decode(torch.where(scored, predicted, torch.full_like(predicted, self.codec.pad_id)).tolist())
        return correct.sum(1).cpu(), scored.sum(1).cpu(), texts

    @torch.no_grad()
    def generate(self, x, fixated, valid, lang, max_new_tokens=128, num_beams=1):
        """Free-running generation: every token conditioned on the model's own previous tokens."""
        embeds, _, _ = self.embed(x, fixated, lang)
        encoder = self.lm.get_encoder()(inputs_embeds=embeds, attention_mask=valid.long())
        ids = self.lm.generate(encoder_outputs=encoder, attention_mask=valid.long(), max_new_tokens=max_new_tokens,
                               num_beams=num_beams, forced_bos_token_id=self.codec.lang_id(lang), do_sample=False)
        return self.codec.decode(ids.tolist())


def code_usage(codes_by_lang, n_codes):
    """Codebook use per language and how many codes both languages use."""
    out, used = {}, {}
    for lang, codes in codes_by_lang.items():
        counts = torch.bincount(codes.reshape(-1), minlength=n_codes).float()
        p = counts / counts.sum().clamp(min=1)
        entropy = -(p[p > 0] * p[p > 0].log()).sum().item()
        used[lang] = set(torch.nonzero(counts).reshape(-1).tolist())
        out[lang] = {"codes_used": len(used[lang]), "perplexity": math.exp(entropy)}
    if len(used) == 2:
        a, b = used.values()
        out["shared_codes"] = len(a & b)
        out["jaccard"] = len(a & b) / max(len(a | b), 1)
    return out
