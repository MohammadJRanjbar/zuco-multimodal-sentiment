"""How much does a word-level EEG representation say about the word read? (complements decode_eeg_to_text.py)

On reader-averaged words of unseen sentences (5 folds split by sentence; ridge with the penalty chosen
by sentence-grouped inner CV), for the representation itself, the same representation shuffled across
training words, and text-only word features (length, frequency, position):

1. word embedding: R² and centred cosine between the predicted and the true LaBSE input embedding;
2. sentence identification: for each held-out sentence, the predicted embeddings are compared position by
   position with those of every held-out sentence of the same length; correct if the read sentence scores
   highest (chance = 1 / number of same-length candidates);
3. word properties: held-out R² for log length, Zipf frequency, relative position and (optional) GPT-2
   surprisal.

Writes word_info_<tag>.json/.md to --out-dir.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.brainshaping import scan_io  # noqa: E402
from src.brainshaping.data import grouped_splits  # noqa: E402
from src.brainshaping.encoding import static_word_vectors, word_controls  # noqa: E402
from src.followup import decoding  # noqa: E402
from src.progress import progress  # noqa: E402

INPUTS = ("eeg", "shuffled_eeg", "word_features")
PROPERTIES = ("log_length", "zipf", "relative_position")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--word-eeg-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="labse", help="text model alias for the word embedding (scan_io.MODELS)")
    parser.add_argument("--surprisal-model", default="gpt2", help="'none' to skip the surprisal property")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def sentence_identification(pred, true, sentence_of, position_of, length_of):
    """Per held-out sentence: (correct, number of same-length candidates) from position-wise cosine."""
    by_sentence = {}
    for row, (s, p) in enumerate(zip(sentence_of, position_of)):
        by_sentence.setdefault(s, {})[p] = row
    P, T = unit(pred), unit(true)
    out = []
    for s, rows in by_sentence.items():
        candidates = [c for c in by_sentence if length_of[c] == length_of[s]]
        if len(candidates) < 2:
            continue
        scores = []
        for c in candidates:
            shared = [p for p in rows if p in by_sentence[c]]
            scores.append(np.mean([P[rows[p]] @ T[by_sentence[c][p]] for p in shared]) if shared else -np.inf)
        best = candidates[int(np.argmax(scores))]
        out.append((s, float(best == s), len(candidates)))
    return out


def main():
    args = parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    data = scan_io.load_items("zuco", args.word_eeg_dir)
    sentences, items, eeg = data["sentences"], data["items"], data["eeg"]
    item_sentence, item_word = items["sentence_row"].to_numpy(), items["word_index"].to_numpy()
    controls, lexical, _ = word_controls(sentences["words"].tolist(), items, data["meta"],
                                         data["readers_per_sentence"], "en")
    (alias, name), = scan_io.model_specs([args.model])
    vectors, found = static_word_vectors(sentences["words"].tolist(), item_sentence, item_word, name,
                                         revision=scan_io.REVISIONS.get(name))
    properties = {p: controls[p].to_numpy() for p in PROPERTIES}
    if args.surprisal_model != "none":
        surprisal = scan_io.word_surprisal(sentences["words"].tolist(), args.surprisal_model, device,
                                           os.path.join(args.out_dir, f"surprisal_{args.surprisal_model}.json"))
        properties["surprisal"] = np.array([surprisal[s][w] for s, w in zip(item_sentence, item_word)])
    keep = np.flatnonzero(np.asarray(found) & np.all([np.isfinite(v) for v in properties.values()], axis=0))
    V = np.asarray(vectors, dtype=np.float64)[keep]
    X = {"eeg": eeg[keep].astype(np.float64), "word_features": controls[lexical].to_numpy(np.float64)[keep]}
    groups = items["sentence_id"].to_numpy()[keep]
    positions = item_word[keep]
    length_of = dict(zip(sentences["sentence_id"], sentences["words"].map(len)))
    print(f"{args.tag}: {len(keep)} reader-averaged words, {eeg.shape[1]} features, {len(np.unique(groups))} sentences")

    rng = np.random.default_rng(args.seed)
    folds = grouped_splits(groups, args.folds, args.seed)
    embed = {k: {"r2_num": [], "r2_den": [], "cos": [], "cluster": []} for k in INPUTS}
    ident = {k: [] for k in INPUTS}
    prop = {k: {p: {"pred": np.zeros(len(keep)), } for p in properties} for k in INPUTS}
    bar = progress(total=len(folds) * len(INPUTS), desc=f"{args.tag}: word information", unit="fit")
    for f, (tr, te) in enumerate(folds):
        mean, std = V[tr].mean(axis=0), V[tr].std(axis=0)
        std[std < 1e-8] = 1.0
        T = (V - mean) / std
        shuffled = X["eeg"].copy()
        shuffled[tr] = X["eeg"][tr][rng.permutation(len(tr))]
        inputs = {"eeg": X["eeg"], "shuffled_eeg": shuffled, "word_features": X["word_features"]}
        for k in INPUTS:
            Y = np.column_stack([T] + [properties[p][:, None] for p in properties])
            pred, _ = decoding.fit_predict(inputs[k][tr], Y[tr], inputs[k][te], groups[tr], args.inner_folds,
                                           device=device, seed=args.seed + f)
            P, Pp = pred[:, :T.shape[1]], pred[:, T.shape[1]:]
            embed[k]["r2_num"].append(((P - T[te]) ** 2).sum(axis=1))
            embed[k]["r2_den"].append((T[te] ** 2).sum(axis=1))  # training mean is 0 after z-scoring
            embed[k]["cos"].append((unit(P) * unit(T[te])).sum(axis=1))
            embed[k]["cluster"].append(groups[te])
            for j, p in enumerate(properties):
                prop[k][p]["pred"][te] = Pp[:, j]
            ident[k] += sentence_identification(P, T[te], groups[te], positions[te], length_of)
            bar.update()
    bar.close()

    summary = {"tag": args.tag, "n_words": int(len(keep)), "n_features": int(eeg.shape[1]),
               "target": f"{alias} input embeddings", "embedding": {}, "sentence_identification": {},
               "properties": {}}
    for k in INPUTS:
        num, den = np.concatenate(embed[k]["r2_num"]), np.concatenate(embed[k]["r2_den"])
        cos, cl = np.concatenate(embed[k]["cos"]), np.concatenate(embed[k]["cluster"])
        summary["embedding"][k] = {"r2": float(1 - num.sum() / den.sum()),
                                   "centred_cosine": decoding.cluster_bootstrap_mean(cos, cl, args.n_boot, args.seed)}
        table = pd.DataFrame(ident[k], columns=["sentence", "correct", "candidates"])
        summary["sentence_identification"][k] = {
            "n_sentences": int(len(table)), "accuracy": float(table["correct"].mean()),
            "chance": float((1 / table["candidates"]).mean()), "median_candidates": float(table["candidates"].median())}
        summary["properties"][k] = {
            p: float(1 - ((prop[k][p]["pred"] - properties[p][keep]) ** 2).sum()
                     / ((properties[p][keep] - properties[p][keep].mean()) ** 2).sum()) for p in properties}
    cos = {k: np.concatenate(embed[k]["cos"]) for k in INPUTS}
    cl = np.concatenate(embed["eeg"]["cluster"])
    summary["embedding"]["eeg - shuffled_eeg (centred cosine)"] = decoding.paired_difference(
        cos["eeg"], cos["shuffled_eeg"], cl, args.n_boot, args.seed)
    a, b = (pd.DataFrame(ident[k], columns=["s", "c", "n"]).set_index("s")["c"] for k in ("eeg", "shuffled_eeg"))
    both = a.index.intersection(b.index)
    summary["sentence_identification"]["eeg - shuffled_eeg"] = decoding.paired_difference(
        a.loc[both].to_numpy(), b.loc[both].to_numpy(), both.to_numpy(), args.n_boot, args.seed) if len(both) else None
    summary["runtime_min"] = (time.time() - started) / 60
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(args.out_dir, f"word_info_{args.tag}.json"), "w"), indent=1)
    write_report(os.path.join(args.out_dir, f"word_info_{args.tag}.md"), summary)
    print(open(os.path.join(args.out_dir, f"word_info_{args.tag}.md")).read())


def ci(triple, digits=3):
    return f"{triple[0]:.{digits}f} [{triple[1]:.{digits}f}, {triple[2]:.{digits}f}]" if triple else "—"


def write_report(path, s):
    lines = [f"# What the representation says about the word ({s['tag']})", "",
             f"{s['n_words']} reader-averaged words of unseen sentences, {s['n_features']} features per word; "
             f"target: {s['target']}. `shuffled_eeg` = the same features shuffled across training words; "
             "`word_features` = length, frequency and position from the text (no EEG).", "",
             "## Word embedding", "", "| input | R² | centred cosine [95% CI] |", "|---|---:|---|"]
    for k in INPUTS:
        e = s["embedding"][k]
        lines.append(f"| {k} | {e['r2']:.4f} | {ci(e['centred_cosine'])} |")
    lines += ["", f"EEG - shuffled (centred cosine): {ci(s['embedding']['eeg - shuffled_eeg (centred cosine)'], 4)}",
              "", "## Sentence identification (among held-out sentences of the same length)", "",
              "| input | sentences | accuracy | chance | median candidates |", "|---|---:|---:|---:|---:|"]
    for k in INPUTS:
        r = s["sentence_identification"][k]
        lines.append(f"| {k} | {r['n_sentences']} | {r['accuracy']:.3f} | {r['chance']:.3f} | "
                     f"{r['median_candidates']:.0f} |")
    lines += ["", f"EEG - shuffled (accuracy): {ci(s['sentence_identification']['eeg - shuffled_eeg'])}", "",
              "## Word properties (held-out R²)", "", "| input | " + " | ".join(s["properties"]["eeg"]) + " |",
              "|---|" + "---:|" * len(s["properties"]["eeg"])]
    for k in INPUTS:
        lines.append(f"| {k} | " + " | ".join(f"{v:.4f}" for v in s["properties"][k].values()) + " |")
    lines += ["", "EEG carries information about the word only where it beats `shuffled_eeg`; it carries information "
              "beyond the word's form only where it also adds to `word_features` (see decode_eeg_to_text.py's "
              "matched pairs and stacked test)."]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
