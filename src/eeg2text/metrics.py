"""Text-generation metrics that work for English and Persian alike.

Tokens: lower-cased words with punctuation split off (Unicode-aware, so
Persian letters and the zero-width non-joiner stay inside words).
BLEU follows Papineni et al. (corpus level, brevity penalty); sentence BLEU
uses add-one smoothing for n > 1 (Lin & Och).
"""

import math
import re
from collections import Counter

# Word characters, the zero-width non-joiner (U+200C) and combining marks (e.g. Arabic-script tanvin).
TOKEN = re.compile("[\\w‌ً-ٰٟ̀-ͯ]+|[^\\w\\sً-ٰٟ]", re.UNICODE)


def tokenize(text):
    return TOKEN.findall(str(text).lower())


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _matches(hyp, ref, n):
    h, r = _ngrams(hyp, n), _ngrams(ref, n)
    return sum(min(c, r[g]) for g, c in h.items()), max(len(hyp) - n + 1, 0)


def corpus_bleu(hypotheses, references, max_n=4):
    matched, total = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        h, r = tokenize(hyp), tokenize(ref)
        hyp_len, ref_len = hyp_len + len(h), ref_len + len(r)
        for n in range(1, max_n + 1):
            m, t = _matches(h, r, n)
            matched[n - 1] += m
            total[n - 1] += t
    if hyp_len == 0 or min(matched) == 0:
        return 0.0
    log_precision = sum(math.log(m / t) for m, t in zip(matched, total)) / max_n
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return brevity * math.exp(log_precision)


def sentence_bleu(hypothesis, reference, max_n=4):
    h, r = tokenize(hypothesis), tokenize(reference)
    if not h:
        return 0.0
    log_precision = 0.0
    for n in range(1, max_n + 1):
        m, t = _matches(h, r, n)
        if n == 1:
            if m == 0:
                return 0.0
            log_precision += math.log(m / t)
        else:
            log_precision += math.log((m + 1) / (t + 1))
    brevity = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / len(h))
    return brevity * math.exp(log_precision / max_n)


def rouge_n(hypothesis, reference, n=1):
    h, r = _ngrams(tokenize(hypothesis), n), _ngrams(tokenize(reference), n)
    overlap = sum(min(c, r[g]) for g, c in h.items())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(h.values()), overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def word_error_rate(hypothesis, reference):
    h, r = tokenize(hypothesis), tokenize(reference)
    previous = list(range(len(h) + 1))
    for i, ref_token in enumerate(r, 1):
        current = [i] + [0] * len(h)
        for j, hyp_token in enumerate(h, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_token != hyp_token))
        previous = current
    return previous[-1] / max(len(r), 1)


def per_trial(hypothesis, reference):
    return {"bleu4": sentence_bleu(hypothesis, reference), "bleu1": sentence_bleu(hypothesis, reference, 1),
            "rouge1": rouge_n(hypothesis, reference, 1), "rouge2": rouge_n(hypothesis, reference, 2),
            "wer": word_error_rate(hypothesis, reference)}
