"""A causal LM with LoRA that reads EEG soft tokens and a sentence.

Prompt layout (EEG arms)::

    <instruction>
    Sentence: w1 [e1] w2 [e2] ... wN [eN]
    Sentiment:

``[e_i]`` is a soft token carrying the reader's EEG recorded while reading word
``w_i`` (fixation-locked band power), mapped by a small trained projector into
the LM's embedding space and scaled to the RMS of real word embeddings. The
text-only arm has no ``[e_i]`` slots. The class score is the LM's next-token
logit for the first token of " negative", " neutral", and " positive", so
training starts from the LM's own sentiment knowledge.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EEG_SLOT = -1


class LoRALinear(nn.Module):
    """``base(x) + scale * B(A(dropout(x)))`` with a frozen base layer; B starts at zero."""

    def __init__(self, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        update = F.linear(F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return self.base(x) + (self.scale * update).to(x.dtype)


def add_lora(model, targets, r, alpha, dropout):
    """Wrap every ``nn.Linear`` whose attribute name is in ``targets``."""
    replaced = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha, dropout))
                replaced += 1
    if not replaced:
        raise ValueError(f"no nn.Linear named {targets} found for LoRA")
    return replaced


class EEGProjector(nn.Module):
    """Shared MLP: one word's EEG vector -> one soft token."""

    def __init__(self, in_dim, d_model, hidden=512, dropout=0.1, target_rms=1.0):
        super().__init__()
        # No input LayerNorm: inputs are already standardized per reader, and a
        # per-vector LayerNorm would erase a word's overall power level.
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, d_model))
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.register_buffer("target_rms", torch.tensor(float(target_rms)))

    def forward(self, eeg):
        return self.out_norm(self.mlp(eeg)) * self.target_rms


class FusionClassifier(nn.Module):
    def __init__(self, lm, tokenizer, *, class_words, eeg_dim, prompt, lora=None, projector=None):
        super().__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        for parameter in self.lm.parameters():
            parameter.requires_grad_(False)
        lora = lora or {}
        if lora.get("r", 0) > 0:
            self.n_lora_layers = add_lora(self.lm, set(lora["targets"]), lora["r"], lora["alpha"],
                                          lora.get("dropout", 0.0))
        else:
            self.n_lora_layers = 0
        self.decoder = getattr(self.lm, self.lm.base_model_prefix)
        embeddings = self.lm.get_input_embeddings().weight.detach().float()
        target_rms = embeddings.pow(2).mean(dim=1).sqrt().mean().item()
        projector = projector or {}
        self.projector = EEGProjector(eeg_dim, embeddings.shape[1], projector.get("hidden", 512),
                                      projector.get("dropout", 0.1), target_rms)
        self.class_ids = [tokenizer.encode(" " + word, add_special_tokens=False)[0] for word in class_words]
        if len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError(f"class words {class_words} share a first token: {self.class_ids}")
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
        self.prefix_ids = encode(prompt["instruction"]) + encode(prompt["sentence_label"])
        self.answer_ids = encode(prompt["answer_label"])
        self.pad_id = 0
        self._word_ids = {}

    def _encode_word(self, word):
        if word not in self._word_ids:
            self._word_ids[word] = self.tokenizer.encode(" " + word, add_special_tokens=False)
        return self._word_ids[word]

    def build_ids(self, words, use_eeg):
        ids = list(self.prefix_ids)
        for word in words:
            ids += self._encode_word(word)
            if use_eeg:
                ids.append(EEG_SLOT)
        return ids + self.answer_ids

    def prepare(self, batch_words, eeg=None, counts=None, use_eeg=False):
        """Input embeddings, attention mask, token roles, and sequence lengths."""
        device = self.lm.get_input_embeddings().weight.device
        sequences, roles = [], []
        for words in batch_words:
            ids = list(self.prefix_ids)
            role = [0] * len(ids)  # 0 prompt, 1 word token, 2 EEG slot, 3 answer cue
            for word in words:
                pieces = self._encode_word(word)
                ids += pieces
                role += [1] * len(pieces)
                if use_eeg:
                    ids.append(EEG_SLOT)
                    role.append(2)
            ids += self.answer_ids
            role += [3] * len(self.answer_ids)
            sequences.append(ids)
            roles.append(role)
        lengths = torch.tensor([len(s) for s in sequences], device=device)
        width = int(lengths.max())
        raw = torch.full((len(sequences), width), self.pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
        role = torch.full((len(sequences), width), -1, dtype=torch.long, device=device)
        for row, sequence in enumerate(sequences):
            raw[row, :len(sequence)] = torch.tensor(sequence, device=device)
            mask[row, :len(sequence)] = 1
            role[row, :len(sequence)] = torch.tensor(roles[row], device=device)
        embeds = self.lm.get_input_embeddings()(raw.clamp(min=0))
        soft = None
        if use_eeg:
            eeg = torch.as_tensor(eeg, dtype=torch.float32, device=device)
            counts = torch.as_tensor(counts, device=device)
            valid = torch.arange(eeg.shape[1], device=device)[None, :] < counts[:, None]
            soft = self.projector(eeg[valid])
            slots = raw == EEG_SLOT
            if int(slots.sum()) != len(soft):
                raise RuntimeError("EEG slots and word vectors do not line up")
            embeds = embeds.clone()
            embeds[slots] = soft.to(embeds.dtype)
        return {"embeds": embeds, "mask": mask, "role": role, "lengths": lengths, "soft": soft}

    def classify(self, hidden, lengths):
        last = hidden[torch.arange(len(lengths), device=hidden.device), lengths - 1]
        head = self.lm.get_output_embeddings()
        weight = head.weight[self.class_ids]
        bias = head.bias[self.class_ids] if getattr(head, "bias", None) is not None else None
        return F.linear(last.float(), weight.float(), None if bias is None else bias.float())

    def forward(self, batch_words, eeg=None, counts=None, use_eeg=False):
        """``eeg``: ``[B, max_words, F]``; ``counts``: words per example."""
        inputs = self.prepare(batch_words, eeg, counts, use_eeg)
        hidden = self.decoder(inputs_embeds=inputs["embeds"], attention_mask=inputs["mask"]).last_hidden_state
        return self.classify(hidden, inputs["lengths"])

    def analyze(self, batch_words, eeg=None, counts=None, use_eeg=False, attentions=True, embeds=None):
        """Forward pass that also returns attentions, hidden states, and token roles.

        Attention weights need an eager attention implementation
        (``attn_implementation="eager"`` when loading the LM).
        """
        inputs = self.prepare(batch_words, eeg, counts, use_eeg)
        if embeds is not None:
            inputs["embeds"] = embeds
        out = self.decoder(inputs_embeds=inputs["embeds"], attention_mask=inputs["mask"],
                           output_attentions=attentions, output_hidden_states=True)
        inputs["logits"] = self.classify(out.last_hidden_state, inputs["lengths"])
        inputs["hidden_states"] = out.hidden_states
        inputs["attentions"] = out.attentions if attentions else None
        return inputs

    def trainable_state(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        parameters = dict(self.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value.to(parameters[name].device))

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
