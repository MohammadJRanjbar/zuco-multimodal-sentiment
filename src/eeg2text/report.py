"""Compile EEG-to-text runs: metrics, paired controls, multilingual effect, sentiment of generated text."""

import glob
import json
import os

import numpy as np
import pandas as pd

from .metrics import corpus_bleu, per_trial

TRIAL_METRICS = ("tf_accuracy", "bleu4", "bleu1", "rouge1", "wer")
VECTOR_INPUTS = ("word_vectors", "mbart_vectors")
TEST_MODES = (("", "single reader"), ("_avg", "reader average"))


def load_runs(root):
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "generations.csv"))):
        name = os.path.basename(os.path.dirname(path))
        table = pd.read_csv(path, keep_default_na=False)
        metrics = json.load(open(os.path.join(os.path.dirname(path), "metrics.json")))
        runs[name] = (table, metrics)
    return runs


def trial_scores(table, text_column="free_text"):
    rows = [per_trial(h, r) for h, r in zip(table[text_column], table["gold"])]
    scores = pd.DataFrame(rows, index=table.index)
    scores["tf_accuracy"] = table["tf_correct"] / table["tf_scored"].clip(lower=1)
    return scores


def cluster_ci(values, clusters, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    codes = np.unique(clusters, return_inverse=True)[1]
    n = codes.max() + 1
    sums = np.bincount(codes, weights=values, minlength=n)
    counts = np.bincount(codes, minlength=n).astype(float)
    w = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=n_boot)
    boot = (w @ sums) / np.maximum(w @ counts, 1)
    return [float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def embed_texts(texts, model_name, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            batch = [t if t.strip() else "." for t in texts[start:start + 64]]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"][..., None].float()
            out.append(((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
    return np.concatenate(out)


def summarize(root, sentiment_model=None, device="cpu", n_boot=2000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    runs = load_runs(root)
    if not runs:
        return None
    sentences = pd.read_csv(os.path.join(root, "sentences.csv"), keep_default_na=False)
    classifiers = {}
    if sentiment_model:
        for lang, group in sentences.groupby("lang"):
            train = group[(group["part"] == "train") & (group["label"] >= 0)]
            clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
            clf.fit(embed_texts(train["text"].tolist(), sentiment_model, device), train["label"])
            classifiers[lang] = clf
    rows, scores, sentiment = [], {}, {}
    for name, (table, metrics) in runs.items():
        for condition, part in table.groupby("condition"):
            for lang, sub in part.groupby("lang"):
                key = (name, condition, lang)
                scores[key] = trial_scores(sub).assign(sentence_id=sub["sentence_id"].values, trial=sub["trial"].values)
                row = {"run": name, "setting": metrics["setting"], "encoder": metrics["encoder"],
                       "trained_on": metrics["input"], "tested_on": condition, "lang": lang, "n_trials": len(sub),
                       "tf_accuracy": float(scores[key]["tf_accuracy"].mean()),
                       "tf_bleu4": corpus_bleu(sub["tf_text"], sub["gold"]),
                       "free_bleu4": corpus_bleu(sub["free_text"], sub["gold"]),
                       "free_bleu1": corpus_bleu(sub["free_text"], sub["gold"], max_n=1),
                       "rouge1": float(scores[key]["rouge1"].mean()), "wer": float(scores[key]["wer"].mean())}
                if lang in classifiers:
                    predicted = classifiers[lang].predict(embed_texts(sub["free_text"].tolist(), sentiment_model, device))
                    row["sentiment_f1_generated"] = float(f1_score(sub["label"], predicted, average="macro"))
                    gold = classifiers[lang].predict(embed_texts(sub["gold"].tolist(), sentiment_model, device))
                    row["sentiment_f1_real_text"] = float(f1_score(sub["label"], gold, average="macro"))
                rows.append(row)
    table = pd.DataFrame(rows)
    comparisons = []

    def paired(a_key, b_key, label):
        a, b = scores.get(a_key), scores.get(b_key)
        if a is None or b is None:
            return
        merged = a.merge(b, on="trial", suffixes=("_a", "_b"))
        for metric in ("tf_accuracy", "bleu4", "rouge1"):
            ci = cluster_ci(merged[f"{metric}_a"] - merged[f"{metric}_b"], merged["sentence_id_a"], n_boot)
            comparisons.append({"comparison": label, "lang": a_key[2], "metric": metric, "diff": ci[0],
                                "ci_low": ci[1], "ci_high": ci[2]})

    positive = []
    for (name, condition, lang) in list(scores):
        metrics = runs[name][1]
        prefix = f"{metrics['setting']}_{metrics['encoder']}"
        for suffix, mode in TEST_MODES:
            if metrics["input"] in VECTOR_INPUTS and condition == metrics["input"] + suffix:
                before = len(comparisons)
                paired((name, condition, lang), (f"{prefix}_noise", "noise" + suffix, lang),
                       f"{prefix}: positive control, {metrics['input']} model - noise model ({mode})")
                for c in comparisons[before:]:
                    if c["metric"] == "bleu4":
                        positive.append({"setting": metrics["setting"], "encoder": metrics["encoder"],
                                         "input": metrics["input"], "lang": lang, "test": mode, **c,
                                         "passed": c["ci_low"] > 0})
            if metrics["input"] != "eeg" or condition != "eeg" + suffix:
                continue
            for control in ("noise", "shuffled_eeg"):
                paired((name, condition, lang), (f"{prefix}_{control}", control + suffix, lang),
                       f"{prefix}: EEG model - {control} model ({mode})")
                paired((name, condition, lang), (name, control + suffix, lang),
                       f"{prefix}: EEG model fed EEG - fed {control} at test ({mode})")
    for lang, setting in (("en", "en"), ("fa", "fa")):
        for encoder in sorted({m["encoder"] for _, m in runs.values()}):
            for condition in ("eeg", "noise", "shuffled_eeg") + VECTOR_INPUTS:
                for suffix, mode in TEST_MODES:
                    paired((f"joint_{encoder}_{condition}", condition + suffix, lang),
                           (f"{setting}_{encoder}_{condition}", condition + suffix, lang),
                           f"{encoder} {condition}: joint - {setting}-only training ({mode})")
    interaction = []
    for lang, setting in (("en", "en"), ("fa", "fa")):
        for encoder, (suffix, mode) in ((e, t) for e in sorted({m["encoder"] for _, m in runs.values()})
                                        for t in TEST_MODES):
            keys = {s: (f"{s}_{encoder}_eeg", "eeg" + suffix, lang) for s in ("joint", setting)}
            nkeys = {s: (f"{s}_{encoder}_noise", "noise" + suffix, lang) for s in ("joint", setting)}
            if all(k in scores for k in list(keys.values()) + list(nkeys.values())):
                gap = {}
                for s in ("joint", setting):
                    merged = scores[keys[s]].merge(scores[nkeys[s]], on="trial", suffixes=("_e", "_n"))
                    gap[s] = merged.assign(gap=merged["bleu4_e"] - merged["bleu4_n"])[["trial", "gap", "sentence_id_e"]]
                merged = gap["joint"].merge(gap[setting], on="trial", suffixes=("_j", "_m"))
                ci = cluster_ci(merged["gap_j"] - merged["gap_m"], merged["sentence_id_e_j"], n_boot)
                interaction.append({"lang": lang, "encoder": encoder, "test": mode, "metric": "bleu4",
                                    "effect": "(EEG - noise) in joint minus (EEG - noise) in single-language",
                                    "diff": ci[0], "ci_low": ci[1], "ci_high": ci[2]})
    codes = {name: m.get("vq_code_usage") for name, (_, m) in runs.items() if m.get("vq_code_usage")}
    summary = {"runs": table.to_dict("records"), "positive_control": positive, "comparisons": comparisons,
               "multilingual_interaction": interaction, "vq_code_usage": codes}
    json.dump(summary, open(os.path.join(root, "eeg_to_text_summary.json"), "w"), indent=1, default=float)
    table.to_csv(os.path.join(root, "eeg_to_text_runs.csv"), index=False)
    write_markdown(os.path.join(root, "eeg_to_text_report.md"), table, comparisons, interaction, codes, positive)
    return summary


def _fmt(row):
    return f"{row['diff']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"


def write_markdown(path, table, comparisons, interaction, codes, positive=()):
    lines = ["# EEG-to-text generation (English ZuCo, Persian TeCo)", "",
             "Test sentences were never seen in training (by any reader). *Teacher-forced*: each token is predicted "
             "from the true previous tokens (how many published EEG-to-text results were scored). *Free-running*: "
             "the model writes the whole sentence from its own previous tokens. Every input condition has the same "
             "sentence length and fixation pattern; `word_vectors` and `mbart_vectors` are positive controls (the "
             "input contains the text). Conditions ending in `_avg` are tested on the reader average of each test "
             "sentence.", ""]
    if positive:
        lines += ["## Positive control (must pass before the EEG rows mean anything)", "",
                  "A control model must write unseen sentences better than the noise model (free-running BLEU-4, "
                  "paired by trial). If it does not, the generator is not using its input, and EEG = noise says "
                  "nothing about the EEG.", "",
                  "| setting | encoder | input | language | test | BLEU-4 difference [95% CI] | verdict |",
                  "|---|---|---|---|---|---|---|"]
        for c in positive:
            lines.append(f"| {c['setting']} | {c['encoder']} | {c['input']} | {c['lang']} | {c['test']} | {_fmt(c)} | "
                         f"{'PASS' if c['passed'] else 'FAIL'} |")
        lines.append("")
    cols = ["setting", "encoder", "trained_on", "tested_on", "n_trials", "tf_accuracy", "tf_bleu4", "free_bleu1",
            "free_bleu4", "rouge1", "wer"] + [c for c in ("sentiment_f1_generated", "sentiment_f1_real_text")
                                              if c in table]
    for lang, sub in table.groupby("lang"):
        lines += [f"## {'English (ZuCo)' if lang == 'en' else 'Persian (TeCo)'}", "",
                  "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in sub.sort_values(["encoder", "setting", "trained_on", "tested_on"]).iterrows():
            lines.append("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
        lines.append("")
    if comparisons:
        lines += ["## Paired comparisons (same test trials; 95% CI from a sentence bootstrap)", "",
                  "| comparison | language | metric | difference [95% CI] |", "|---|---|---|---|"]
        for c in comparisons:
            lines.append(f"| {c['comparison']} | {c['lang']} | {c['metric']} | {_fmt(c)} |")
        lines.append("")
    if interaction:
        lines += ["## Does multilingual training help the EEG specifically?", "",
                  "| language | encoder | test | effect | difference [95% CI] |", "|---|---|---|---|---|"]
        for c in interaction:
            lines.append(f"| {c['lang']} | {c['encoder']} | {c['test']} | {c['effect']} | {_fmt(c)} |")
        lines.append("")
    if codes:
        lines += ["## EEG tokenizer (codebook) use", ""]
        for name, usage in codes.items():
            lines.append(f"* {name}: {json.dumps(usage)}")
        lines.append("")
    lines += ["## How to read", "",
              "* Read the EEG rows only where the positive control passes in the same setting.",
              "* EEG helps only if the EEG model beats the **noise** and **shuffled-EEG** models in free-running "
              "generation (CI above 0), per reader or on the reader average.",
              "* If the EEG model's output barely changes when it is fed noise at test time, it ignores the EEG "
              "(the check of Jo et al.).",
              "* Teacher-forced scores are high for every input because the language model predicts the next "
              "word from the true previous words; compare them across inputs, not with free-running scores.",
              "* Multilingual training helps the EEG only if the interaction row is above 0; a joint-minus-single "
              "gain that is equal for EEG and noise comes from the shared language model, not from the brain signal."]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
