"""Decode the read word from EEG (controlled EEG-to-text) and classify the decoded text's sentiment.

See src/followup/decoding.py for the measures. For each input (EEG, EEG
shuffled across training words, Gaussian noise, word features alone, EEG plus
word features) this reports, on unseen sentences:
- 2-vs-2 accuracy, overall and for pairs matched on length and frequency;
- retrieval of the word among the test sentences' vocabulary (top-1, top-5,
  mean reciprocal rank);
- the macro-F1 of a sentiment classifier trained on real training text and
  applied to the decoded test sentences.

Works on ZuCo word EEG (band power or fixation-related potentials) and TeCo.
Writes decoding_<tag>.md/.json and decoded_examples_<tag>.csv to --out-dir.
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src.brainshaping import scan_io  # noqa: E402
from src.brainshaping.data import grouped_splits  # noqa: E402
from src.brainshaping.encoding import extract_layer_vectors, strip_punctuation, word_controls  # noqa: E402
from src.followup import decoding  # noqa: E402
from src.progress import progress  # noqa: E402

INPUTS = ("eeg", "shuffled_eeg", "noise", "word_features", "eeg+word_features")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["zuco", "teco"])
    parser.add_argument("--word-eeg-dir")
    parser.add_argument("--trt-dir")
    parser.add_argument("--labels-csv")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vector-cache-dir", default=None)
    parser.add_argument("--model", default="labse", help="text model alias or alias=path (see scan_io.MODELS)")
    parser.add_argument("--layer", type=int, default=0, help="0 = non-contextual input layer (word identity)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--n-pairs", type=int, default=20000)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.tag = args.tag or args.dataset
    return args


def main():
    args = parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    data = scan_io.load_items(args.dataset, args.word_eeg_dir, args.trt_dir, args.labels_csv)
    sentences, items, eeg = data["sentences"], data["items"], data["eeg"]
    item_sentence, item_word = items["sentence_row"].to_numpy(), items["word_index"].to_numpy()
    lang = "en" if args.dataset == "zuco" else "fa"
    controls, lexical, _ = word_controls(sentences["words"].tolist(), items, data["meta"],
                                         data["readers_per_sentence"], lang)
    (alias, name), = scan_io.model_specs([args.model])
    cache_dir = args.vector_cache_dir or os.path.join(args.out_dir, "vectors")
    vectors, found, _ = scan_io.word_vectors(cache_dir, args.tag, alias, name, sentences, item_sentence, item_word,
                                             device, extractor=extract_layer_vectors, batch_size=args.batch_size)
    keep = np.flatnonzero(found)
    V = np.array(vectors[args.layer])[keep].astype(np.float64)
    X_eeg = eeg[keep].astype(np.float64)
    L = controls[lexical].to_numpy(dtype=np.float64)[keep]
    groups = items["sentence_id"].to_numpy()[keep]
    rows = item_sentence[keep]
    words = [sentences["words"].iloc[s][w] for s, w in zip(item_sentence[keep], item_word[keep])]
    bare = [strip_punctuation(w).lower() or w for w in words]
    type_ids, types = pd.factorize(pd.Series(bare))
    type_mean = np.zeros((len(types), V.shape[1]))
    np.add.at(type_mean, type_ids, V)
    type_mean /= np.bincount(type_ids)[:, None]
    lengths = np.round(np.expm1(controls["log_length"].to_numpy()[keep])).astype(int)
    zipf = controls["zipf"].to_numpy()[keep]
    labels = sentences["label_id"].to_numpy()
    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal(X_eeg.shape)
    print(f"{args.tag}: {len(keep)} words with EEG ({len(types)} word types) in {len(np.unique(groups))} sentences; "
          f"target = {alias} layer {args.layer}; inputs {INPUTS}")

    outcome = {kind: {name: [] for name in INPUTS} for kind in ("pairs", "matched")}
    clusters = {"pairs": [], "matched": []}
    ranks = {name: [] for name in INPUTS}
    chance_rr, rank_clusters = [], []
    sentence_rows = []
    examples = []
    folds = grouped_splits(groups, args.folds, args.seed)
    bar = progress(total=len(folds) * len(INPUTS), desc=f"{args.tag}: decoding", unit="fit")
    for f, (tr, te) in enumerate(folds):
        mean, std = V[tr].mean(axis=0), V[tr].std(axis=0)
        std[std < 1e-8] = 1.0
        T = (V - mean) / std
        types_te, local = np.unique(type_ids[te], return_inverse=True)
        type_vectors = (type_mean[types_te] - mean) / std
        perm = rng.permutation(len(tr))
        inputs = {"eeg": X_eeg, "noise": noise, "word_features": L, "eeg+word_features": np.hstack([X_eeg, L])}
        shuffled = X_eeg.copy()
        shuffled[tr] = X_eeg[tr][perm]
        inputs["shuffled_eeg"] = shuffled
        predictions = {}
        for name in INPUTS:
            predictions[name], _ = decoding.fit_predict(inputs[name][tr], T[tr], inputs[name][te], groups[tr],
                                                        args.inner_folds, device=device, seed=args.seed + f)
            bar.update()
        fold_rng = np.random.default_rng(args.seed * 100 + f)
        n_pairs = max(1, args.n_pairs // len(folds))
        for kind, (i, j) in {"pairs": decoding.sample_pairs(type_ids[te], fold_rng, n_pairs),
                             "matched": decoding.sample_pairs(type_ids[te], fold_rng, n_pairs, lengths[te],
                                                              zipf[te])}.items():
            for name in INPUTS:
                outcome[kind][name].append(decoding.two_vs_two(predictions[name], T[te], i, j))
            clusters[kind].append(groups[te][i])
        decoded = {}
        for name in INPUTS:
            rank, top = decoding.retrieval_ranks(predictions[name], type_vectors, local)
            ranks[name].append(rank)
            decoded[name] = top
        chance_rr.append(np.full(len(te), np.mean(1.0 / np.arange(1, len(types_te) + 1))))
        rank_clusters.append(groups[te])
        # sentiment: classifier on training sentences' real text, applied to decoded test sentences
        train_rows = np.unique(rows[tr])
        test_rows = np.unique(rows[te])
        features = lambda vectors, sel, rws: np.stack([vectors[sel][rows[sel] == r].mean(0) for r in rws])  # noqa
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(
            features(T, tr, train_rows), labels[train_rows])
        record = {"fold": f, "row": test_rows, "label": labels[test_rows],
                  "real_text": clf.predict(features(T, te, test_rows))}
        for name in INPUTS:
            decoded_vectors = type_vectors[decoded[name]]
            record[name] = clf.predict(np.stack([decoded_vectors[rows[te] == r].mean(0) for r in test_rows]))
        sentence_rows.append(pd.DataFrame(record))
        if f == 0:
            for r in test_rows[:15]:
                sel = rows[te] == r
                examples.append({"sentence_id": int(sentences["sentence_id"].iloc[r]), "label": int(labels[r]),
                                 "real (words with EEG)": " ".join(np.array(words, dtype=object)[te][sel]),
                                 **{f"decoded from {name}": " ".join(types[types_te[decoded[name][sel]]])
                                    for name in ("eeg", "shuffled_eeg", "noise", "word_features")}})
    bar.close()

    summary = {"tag": args.tag, "target": f"{alias} layer {args.layer}", "n_items": int(len(keep)),
               "n_types": int(len(types)), "folds": len(folds), "two_vs_two": {}, "retrieval": {}, "sentiment": {}}
    for kind in ("pairs", "matched"):
        cl = np.concatenate(clusters[kind]) if clusters[kind] else np.zeros(0)
        res = {"n_pairs": int(len(cl))}
        for name in INPUTS:
            values = np.concatenate(outcome[kind][name]).astype(float)
            res[name] = decoding.cluster_bootstrap_mean(values, cl, args.n_boot, args.seed) if len(cl) else None
        for a, b in (("eeg", "shuffled_eeg"), ("eeg", "noise"), ("eeg+word_features", "word_features")):
            if len(cl):
                res[f"{a} - {b}"] = decoding.paired_difference(np.concatenate(outcome[kind][a]),
                                                               np.concatenate(outcome[kind][b]), cl, args.n_boot,
                                                               args.seed)
        summary["two_vs_two"][kind] = res
    rc = np.concatenate(rank_clusters)
    for name in INPUTS:
        r = np.concatenate(ranks[name])
        summary["retrieval"][name] = {"top1": float((r == 1).mean()), "top5": float((r <= 5).mean()),
                                      "mrr": decoding.cluster_bootstrap_mean(1.0 / r, rc, args.n_boot, args.seed)}
    summary["retrieval"]["chance_mrr"] = float(np.concatenate(chance_rr).mean())
    summary["retrieval"]["eeg - shuffled_eeg (reciprocal rank)"] = decoding.paired_difference(
        1.0 / np.concatenate(ranks["eeg"]), 1.0 / np.concatenate(ranks["shuffled_eeg"]), rc, args.n_boot, args.seed)
    sent = pd.concat(sentence_rows, ignore_index=True)
    rng_b = np.random.default_rng(args.seed)
    for name in ("real_text",) + INPUTS:
        point = f1_score(sent["label"], sent[name], average="macro")
        boot = [f1_score(sent["label"].iloc[idx], sent[name].iloc[idx], average="macro")
                for idx in (rng_b.integers(0, len(sent), len(sent)) for _ in range(args.n_boot))]
        summary["sentiment"][name] = [float(point), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    summary["runtime_min"] = (time.time() - started) / 60
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(args.out_dir, f"decoding_{args.tag}.json"), "w"), indent=1)
    pd.DataFrame(examples).to_csv(os.path.join(args.out_dir, f"decoded_examples_{args.tag}.csv"), index=False)
    report = os.path.join(args.out_dir, f"decoding_{args.tag}.md")
    write_report(report, summary)
    print(open(report).read())


def fmt(triple, digits=3):
    return "—" if triple is None else f"{triple[0]:.{digits}f} [{triple[1]:.{digits}f}, {triple[2]:.{digits}f}]"


def write_report(path, s):
    tv, rt, se = s["two_vs_two"], s["retrieval"], s["sentiment"]
    lines = [f"# Decoding the read word from EEG ({s['tag']})", "",
             f"{s['n_items']:,} words with EEG, {s['n_types']:,} word types, {s['folds']} folds split by sentence. "
             f"Target: {s['target']}. No teacher forcing: every word is decoded from its own EEG only.", "",
             "## 2-vs-2 accuracy (chance 0.5)", "",
             "| input | all pairs | pairs matched on length and frequency |", "|---|---|---|"]
    for name in INPUTS:
        lines.append(f"| {name} | {fmt(tv['pairs'][name])} | {fmt(tv['matched'][name])} |")
    lines += ["", "| difference | all pairs | matched pairs |", "|---|---|---|"]
    for key in ("eeg - shuffled_eeg", "eeg - noise", "eeg+word_features - word_features"):
        lines.append(f"| {key} | {fmt(tv['pairs'].get(key))} | {fmt(tv['matched'].get(key))} |")
    lines += ["", f"Pairs: {tv['pairs']['n_pairs']:,} (all), {tv['matched']['n_pairs']:,} (matched).", "",
              "## Retrieval among the test vocabulary", "",
              f"Chance mean reciprocal rank {rt['chance_mrr']:.4f}.", "",
              "| input | top-1 | top-5 | mean reciprocal rank |", "|---|---:|---:|---|"]
    for name in INPUTS:
        r = rt[name]
        lines.append(f"| {name} | {r['top1']:.4f} | {r['top5']:.4f} | {fmt(r['mrr'], 4)} |")
    lines += ["", f"EEG minus shuffled EEG (reciprocal rank): {fmt(rt['eeg - shuffled_eeg (reciprocal rank)'], 4)}", "",
              "## Sentiment of the decoded text (macro-F1, chance about 0.33)", "",
              "| text given to the classifier | macro-F1 [95% CI] |", "|---|---|",
              f"| real text (same words) | {fmt(se['real_text'])} |"]
    for name in INPUTS:
        lines.append(f"| decoded from {name} | {fmt(se[name])} |")
    pairs, matched = tv["pairs"], tv["matched"]
    eeg_info = pairs["eeg - shuffled_eeg"][1] > 0
    beyond = matched["eeg+word_features - word_features"][1] > 0 if matched["n_pairs"] else False
    sentiment = se["eeg"][1] > max(se["shuffled_eeg"][2], 0.34)
    lines += ["", "## Reading", "",
              "* **EEG carries information about the read word:** "
              + ("yes, real EEG beats shuffled EEG in 2-vs-2." if eeg_info else "not detected (EEG ~ shuffled EEG)."),
              "* **Beyond length and frequency:** "
              + ("adding EEG to word features improves matched pairs." if beyond else
                 "not detected; on length- and frequency-matched pairs EEG adds nothing to word features."),
              "* **Sentiment from decoded text:** "
              + ("above shuffled EEG." if sentiment else "not above the shuffled-EEG control."),
              "", "Examples: `decoded_examples_" + s["tag"] + ".csv`."]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
