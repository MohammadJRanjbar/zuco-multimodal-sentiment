"""Does sentence-level fixation-locked EEG predict the sentence's sentiment, beyond its text?

Sentence representation: the mean over the sentence's words of the FRP window
features (8 windows x 104 electrodes), z-scored per reader, then averaged over
readers. Ridge probes with 5 folds grouped by sentence; label-permutation nulls
re-select the penalty for every permutation.

1. FRP alone (also per time window).
2. Text form alone: number of words, mean word length, mean frequency,
   punctuation share, mean surprisal.
3. Form plus text sentiment: mean VADER valence and share of lexicon words.
4. Presentation order alone (mean position in the session), when
   --trials-csv (word_eeg_trials.csv from extract_word_eeg.py) is given.
5. FRP with form (and order) regressed out. Regressing out is label-free,
   so it uses all sentences.
6. FRP with form, text sentiment (and order) regressed out.
7. FRP per reader (single-reader rows; permutations keep each sentence's
   label across its readers).

EEG carries sentiment information beyond the text only if 5 and 6 stay above their nulls.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import LABEL_TO_ID, ZUCO_REFERENCE_CHANNEL_INDEX  # noqa: E402
from src.diagnostics import signal  # noqa: E402
from src.diagnostics.targets import build_word_table, lm_surprisal, sentence_targets, valence_lexicon  # noqa: E402
from src.followup import frp  # noqa: E402
from src.fusion.word_eeg import load_word_eeg  # noqa: E402

FORM = ["n_words", "mean_length", "mean_zipf", "punctuation_share", "mean_surprisal"]
TEXT_SENTIMENT = ["mean_valence", "lexicon_share"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frp-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--trials-csv", default=None, help="word_eeg_trials.csv with each reader's sentence order")
    parser.add_argument("--surprisal-model", default="gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sentence_frp(trials):
    """Per (reader, sentence): mean FRP features over words with an epoch -> rows, readers, sentence ids."""
    keep = [c for c in range(np.asarray(trials[0]["features"]).shape[2]) if c != ZUCO_REFERENCE_CHANNEL_INDEX]
    rows, readers, sentences = [], [], []
    for trial in trials:
        features = np.asarray(trial["features"], dtype=np.float64)[:, :, keep]
        valid = np.isfinite(features).all(axis=(1, 2))
        if valid.sum() == 0:
            continue
        rows.append(features[valid].mean(axis=0).reshape(-1))
        readers.append(trial["subject_id"])
        sentences.append(trial["sentence_id"])
    X = np.array(rows)
    readers = np.array(readers)
    for r in np.unique(readers):
        block = X[readers == r]
        sd = block.std(axis=0)
        sd[sd < 1e-12] = 1.0
        X[readers == r] = (block - block.mean(axis=0)) / sd
    return X, readers, np.array(sentences)


def residualize(X, covariates):
    C = np.column_stack([np.ones(len(X)), (covariates - covariates.mean(0)) / np.where(covariates.std(0) > 0,
                                                                                      covariates.std(0), 1)])
    beta = np.linalg.lstsq(C, X, rcond=None)[0]
    return X - C @ beta


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    trials = load_word_eeg(args.frp_dir)
    X_single, readers, sentence_of_row = sentence_frp(trials)
    ids = np.unique(sentence_of_row)
    codes = np.searchsorted(ids, sentence_of_row)
    X = np.zeros((len(ids), X_single.shape[1]))
    np.add.at(X, codes, X_single)
    X /= np.bincount(codes)[:, None]
    words_by_sentence = {}
    for trial in trials:
        words_by_sentence.setdefault(trial["sentence_id"], trial["words"])
    surprisal = None
    if args.surprisal_model != "none":
        values = lm_surprisal([words_by_sentence[i] for i in ids], args.surprisal_model, args.device)
        surprisal = dict(zip(ids, values))
    table = sentence_targets(build_word_table(trials, lexicon=valence_lexicon(), surprisal=surprisal))
    table = table.set_index("sentence_id").loc[ids]
    table["punctuation_share"] = [np.mean([not any(ch.isalnum() for ch in w[-1:]) for w in words_by_sentence[i]])
                                  for i in ids]
    table["mean_valence"] = table["mean_valence"].fillna(0.0)
    table["lexicon_share"] = table["n_lexicon_words"] / table["n_words"]
    if "mean_surprisal" not in table:
        table["mean_surprisal"] = 0.0
    labels = np.array([LABEL_TO_ID[int(v)] for v in table["sentence_label"]])
    covariate_sets = {"form": FORM, "form + text sentiment": FORM + TEXT_SENTIMENT}
    order = None
    if args.trials_csv and os.path.exists(args.trials_csv):
        records = pd.read_csv(args.trials_csv)
        records = records[records["status"] == "ok"]
        order = records.groupby("sentence_id")["position"].mean().reindex(ids)
        if order.isna().any():
            order = None
    rows = []

    def probe(name, features, kind="reader-averaged", groups=ids, units=None, n_perm=args.n_perm):
        result = signal.ridge_probe(features, labels if kind == "reader-averaged" else labels[codes], groups,
                                    "multiclass", n_perm=n_perm, perm_units=units, seed=args.seed)
        rows.append({"probe": name, "rows": kind, "macro_F1": result["score"], "null_mean": result.get("null_mean"),
                     "null_q95": result.get("null_q95"), "p_value": result.get("p_value")})
        print(f"{name} ({kind}): macro-F1 {result['score']:.3f}, null q95 {result.get('null_q95', float('nan')):.3f}, "
              f"p {result.get('p_value', float('nan')):.4f}")

    probe("FRP", X)
    for i, window in enumerate(frp.WINDOW_NAMES):
        n = X.shape[1] // len(frp.WINDOW_NAMES)
        probe(f"FRP window {window} ms", X[:, i * n:(i + 1) * n], n_perm=min(args.n_perm, 200))
    probe("text form", table[FORM].to_numpy(dtype=float))
    probe("text form + text sentiment", table[FORM + TEXT_SENTIMENT].to_numpy(dtype=float))
    extra = []
    if order is not None:
        probe("presentation order", order.to_numpy(dtype=float)[:, None])
        extra = [order.to_numpy(dtype=float)[:, None]]
    for name, columns in covariate_sets.items():
        covariates = np.hstack([table[columns].to_numpy(dtype=float)] + extra)
        suffix = " and presentation order" if extra else ""
        probe(f"FRP with {name}{suffix} regressed out", residualize(X, covariates))
    probe("FRP", X_single, kind="single reader", groups=sentence_of_row, units=sentence_of_row)
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(args.out_dir, "sentence_sentiment.csv"), index=False)
    json.dump({"n_sentences": int(len(ids)), "n_readers": int(len(np.unique(readers))),
               "label_counts": np.bincount(labels).tolist(), "n_perm": args.n_perm,
               "probes": result.to_dict("records")},
              open(os.path.join(args.out_dir, "sentence_sentiment.json"), "w"), indent=1, default=float)
    write_report(os.path.join(args.out_dir, "sentence_sentiment.md"), result, len(ids), len(np.unique(readers)),
                 np.bincount(labels), args.n_perm)
    print(open(os.path.join(args.out_dir, "sentence_sentiment.md")).read())


def write_report(path, result, n_sentences, n_readers, counts, n_perm):
    lines = ["# Sentence sentiment from fixation-locked EEG, with text controls", "",
             f"{n_sentences} sentences (labels {counts.tolist()}), {n_readers} readers; ridge probes, 5 folds by "
             f"sentence, {n_perm} label permutations (200 for single windows). Chance macro-F1 is about 0.33; the "
             "permutation null is the fair reference.", "",
             "| probe | rows | macro-F1 | null mean | null 95th pct | p |", "|---|---|---:|---:|---:|---:|"]
    for _, r in result.iterrows():
        lines.append(f"| {r['probe']} | {r['rows']} | {r['macro_F1']:.3f} | {r['null_mean']:.3f} | "
                     f"{r['null_q95']:.3f} | {r['p_value']:.4f} |")
    by = result.set_index(["probe", "rows"])
    main = by.loc[("FRP", "reader-averaged")]
    controlled = [p for p in result["probe"] if p.startswith("FRP with")]
    survives = [p for p in controlled if by.loc[(p, "reader-averaged"), "p_value"] < 0.05]
    lines += ["", "## Reading", "",
              f"* **FRP alone:** {'above' if main['p_value'] < 0.05 else 'not above'} its permutation null "
              f"(p = {main['p_value']:.4f}).",
              "* **Beyond the text:** " + ("still above the null after regressing out "
                                           + "; ".join(p.replace("FRP with ", "").replace(" regressed out", "")
                                                       for p in survives) + "."
                                           if survives else "no: once text properties are regressed out, the FRP "
                                           "no longer predicts sentiment above its null."),
              "* Text properties alone (rows 'text form...') show how much sentiment the text itself makes "
              "predictable from these simple features."]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
