"""Word- and sentence-level targets that EEG might (or should) predict.

Positive controls (known to modulate reading-related EEG): word length, word
frequency, surprisal, fixation duration, content vs function word, position.
Sentiment targets: lexicon valence of the word, its absolute value
(emotionality), and the sentence label.
"""

import math
import re

import numpy as np
import pandas as pd

FUNCTION_WORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves 's 're n't 'll 've 'd s t
""".split())


def clean(word):
    return re.sub(r"[^a-z0-9']+", "", str(word).lower())


def word_length(word):
    return len(re.sub(r"[^A-Za-z]", "", str(word)))


def is_content(word):
    token = clean(word)
    return bool(token) and token not in FUNCTION_WORDS and any(c.isalpha() for c in token)


def zipf_frequencies(words, lang="en"):
    """Zipf frequency (log10 per billion + 3) from ``wordfreq``; NaN if unavailable."""
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return np.full(len(words), np.nan)
    return np.array([zipf_frequency(clean(w), lang) if clean(w) else np.nan for w in words])


def valence_lexicon():
    """VADER word valence (-4 .. 4); empty dict if the package is unavailable."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return {}
    return dict(SentimentIntensityAnalyzer().lexicon)


def lm_surprisal(sentences, model_name="gpt2", device="cpu", batch_size=16):
    """Per-word surprisal (bits) from a causal LM, summing sub-word surprisals.

    ``sentences`` is a list of word lists; each word is conditioned on all
    words to its left (the first word on the BOS token).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    results = []
    for words in sentences:
        ids, owner = [bos], [-1]
        for index, word in enumerate(words):
            pieces = tokenizer.encode((" " if index else "") + word, add_special_tokens=False)
            ids += pieces
            owner += [index] * len(pieces)
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=device)).logits[0].float()
        logprobs = torch.log_softmax(logits[:-1], dim=-1)
        token_bits = -logprobs[torch.arange(len(ids) - 1), torch.tensor(ids[1:], device=device)] / math.log(2)
        per_word = np.zeros(len(words))
        for position, bits in enumerate(token_bits.cpu().numpy(), start=1):
            per_word[owner[position]] += bits
        results.append(per_word)
    return results


def build_word_table(trials, labels_lookup=None, lexicon=None, surprisal=None, frequencies=True):
    """One row per (sentence, word): text-derived targets shared by all readers."""
    lexicon = lexicon if lexicon is not None else {}
    sentences = {}
    for trial in trials:
        sentences.setdefault(trial["sentence_id"], (trial["words"], trial["label"]))
    rows = []
    for sentence_id, (words, label) in sorted(sentences.items()):
        for index, word in enumerate(words):
            token = clean(word)
            valence = lexicon.get(token, np.nan)
            rows.append({
                "sentence_id": sentence_id, "word_index": index, "word": word,
                "length": word_length(word), "is_content": int(is_content(word)),
                "position": index, "relative_position": index / max(len(words) - 1, 1),
                "n_words": len(words), "sentence_label": label,
                "valence": valence, "abs_valence": abs(valence) if np.isfinite(valence) else np.nan,
                "in_lexicon": int(np.isfinite(valence)),
            })
    table = pd.DataFrame(rows)
    if frequencies:
        table["zipf"] = zipf_frequencies(table["word"].tolist())
    if surprisal is not None:
        table["surprisal"] = [surprisal[s][i] for s, i in zip(table["sentence_id"], table["word_index"])]
    return table


def sentence_targets(word_table):
    grouped = word_table.groupby("sentence_id")
    out = grouped.agg(sentence_label=("sentence_label", "first"), n_words=("n_words", "first"),
                      mean_length=("length", "mean"), mean_valence=("valence", "mean"),
                      n_lexicon_words=("in_lexicon", "sum"))
    for column in ("zipf", "surprisal"):
        if column in word_table:
            out[f"mean_{column}"] = grouped[column].mean()
    return out.reset_index()
