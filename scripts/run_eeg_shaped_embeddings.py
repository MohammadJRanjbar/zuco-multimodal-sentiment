"""Version A: EEG-shaped frozen word vectors -> sentiment (minutes, CPU or GPU).

Arms: text_only (no shaping), eeg (EEG subspace), shuffled_eeg (EEG targets
shuffled across training words), random_targets. Evaluated on unseen ZuCo
sentences (sentence-disjoint folds) and, optionally, transferred to SST
(no EEG) to test whether EEG-derived directions help a dataset without EEG.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.brainshaping import static  # noqa: E402
from src.brainshaping.data import (  # noqa: E402
    arm_targets, contextual_word_vectors, fit_targets, item_eeg, sentence_table,
)
from src.fusion.word_eeg import load_word_eeg  # noqa: E402
from src.neurolm.config import save_json  # noqa: E402
from src.neurolm.evaluation import (  # noqa: E402
    SENTIMENT_CLASSES, cluster_bootstrap, compute_metrics, paired_comparison,
)
from src.neurolm.splits import make_splits  # noqa: E402

ARMS = ("text_only", "eeg", "shuffled_eeg", "random_targets")
SST_TO_3 = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--word-eeg-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--encoder", default="sentence-transformers/LaBSE")
    parser.add_argument("--layer", type=int, default=-1, help="encoder hidden layer used as word vectors")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, default=32, help="EEG principal components used as targets")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--external", default="sst5", choices=["sst5", "none"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def frame(ids, y, probs, arm):
    predicted = probs.argmax(axis=1)
    out = pd.DataFrame({"sample_id": [str(i) for i in ids], "sentence_id": ids, "seed": 0, "true_id": y,
                        "predicted_id": predicted, "arm": arm,
                        "true_label": [SENTIMENT_CLASSES[i] for i in y],
                        "predicted_label": [SENTIMENT_CLASSES[i] for i in predicted]})
    for i, name in enumerate(SENTIMENT_CLASSES):
        out[f"prob_{name}"] = probs[:, i]
    return out


def summarize(predictions, comparisons=(("eeg", "shuffled_eeg"), ("eeg", "text_only"), ("eeg", "random_targets"))):
    arms = {arm: {"metrics": compute_metrics(f["true_id"], f["predicted_id"]),
                  "bootstrap": cluster_bootstrap(f, "sentence_id", n_boot=2000)}
            for arm, f in predictions.items()}
    paired = []
    for a, b in comparisons:
        if a in predictions and b in predictions:
            paired.append({"candidate": a, "baseline": b,
                           **paired_comparison(predictions[b], predictions[a], cluster="sentence_id",
                                               n_boot=2000, n_perm=5000)})
    return arms, paired


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    rng = np.random.default_rng(args.seed)
    trials = load_word_eeg(args.word_eeg_dir)
    sentences = sentence_table(trials)
    items, eeg = item_eeg(trials)
    print(f"{len(sentences)} sentences, {len(items)} word items with EEG")
    cache = os.path.join(args.out_dir, f"word_vectors_{args.encoder.replace('/', '_')}_L{args.layer}.npz")
    vectors = contextual_word_vectors(sentences["words"].tolist(), args.encoder, args.device,
                                      cache_path=cache, layer=args.layer)
    row_of = {sid: i for i, sid in enumerate(sentences["sentence_id"])}
    X_items = np.stack([vectors[row_of[s]][w] for s, w in zip(items["sentence_id"], items["word_index"])])
    y = sentences["label_id"].to_numpy()
    item_sentences = items["sentence_id"].to_numpy()
    samples = pd.DataFrame({"sample_id": sentences["sentence_id"].astype(str), "subject_id": "all",
                            "sentence_id": sentences["sentence_id"], "label_id": y})
    splits = make_splits("text", samples, args.seed, {"n_folds": 5, "val_fraction": 0.15})[:args.folds]

    collected = {arm: [] for arm in ARMS}
    fold_log = []
    for k, split in enumerate(splits):
        train_sentences = set(samples["sentence_id"].iloc[split.train])
        test_sentences = set(samples["sentence_id"].iloc[split.test])
        item_train = items["sentence_id"].isin(train_sentences).to_numpy()
        item_test = items["sentence_id"].isin(test_sentences).to_numpy()
        transform, explained = fit_targets(eeg[item_train], args.k)
        targets = transform(eeg)
        mean, std = static.standardizer(X_items[item_train])
        Xs = (X_items - mean) / std
        S = static.sentence_features(vectors, mean, std)
        entry = {"fold": k, "eeg_variance_explained_by_targets": explained}
        for arm in ARMS:
            U, ridge = None, None
            if arm != "text_only":
                arm_y = arm_targets(arm, targets, np.flatnonzero(item_train), rng)
                U, ridge = static.fit_subspace(Xs[item_train], arm_y[item_train],
                                               groups=item_sentences[item_train])
                entry[f"{arm}_encoding_r2_on_real_test_eeg"] = static.encoding_r2(ridge, Xs[item_test], targets[item_test])
                share = static.alignment_with_sentiment(U, S[split.train], y[split.train])
                entry[f"{arm}_sentiment_weight_share_in_subspace"] = share[0]
                entry["subspace_share_by_chance"] = share[1]
            probs, choice = static.select_and_predict(S[split.train], y[split.train], S[split.val], y[split.val],
                                                      S[split.test], U)
            entry[f"{arm}_chosen_beta"] = choice["beta"]
            os.makedirs(os.path.join(args.out_dir, "models"), exist_ok=True)
            static.save_model(os.path.join(args.out_dir, "models", f"zuco_{arm}_fold_{k}.npz"), U, mean, std, choice)
            collected[arm].append(frame(samples["sentence_id"].iloc[split.test].to_numpy(), y[split.test], probs, arm))
        fold_log.append(entry)
        scores = {arm: compute_metrics(collected[arm][-1]["true_id"], collected[arm][-1]["predicted_id"])["macro_f1"]
                  for arm in ARMS}
        print(f"fold {k + 1}: " + ", ".join(f"{a} {v:.3f}" for a, v in scores.items())
              + f" | EEG encoding R2 real {entry['eeg_encoding_r2_on_real_test_eeg']:.3f}, "
              f"shuffled {entry['shuffled_eeg_encoding_r2_on_real_test_eeg']:.3f}")
    zuco = {arm: pd.concat(frames, ignore_index=True) for arm, frames in collected.items()}
    for arm, f in zuco.items():
        f.to_csv(os.path.join(args.out_dir, f"zuco_predictions_{arm}.csv"), index=False)
    zuco_arms, zuco_paired = summarize(zuco)
    summary = {"encoder": args.encoder, "layer": args.layer, "k": args.k, "folds": len(splits),
               "zuco": {"arms": zuco_arms, "paired": zuco_paired, "folds": fold_log}}

    if args.external == "sst5":
        from datasets import load_dataset

        sst = load_dataset("SetFit/sst5")
        parts = {}
        for name in ("train", "validation", "test"):
            texts = [t.split() for t in sst[name]["text"]]
            labels = np.array([SST_TO_3[int(v)] for v in sst[name]["label"]])
            word_vectors = contextual_word_vectors(
                texts, args.encoder, args.device, layer=args.layer,
                cache_path=os.path.join(args.out_dir, f"sst_{name}_{args.encoder.replace('/', '_')}_L{args.layer}.npz"))
            parts[name] = (word_vectors, labels)
        transform, _ = fit_targets(eeg, args.k)
        targets = transform(eeg)
        mean, std = static.standardizer(X_items)
        Xs = (X_items - mean) / std
        features = {name: static.sentence_features(v, mean, std) for name, (v, _) in parts.items()}
        external = {}
        for arm in ARMS:
            U = None
            if arm != "text_only":
                arm_y = arm_targets(arm, targets, np.arange(len(targets)), rng)
                U, _ = static.fit_subspace(Xs, arm_y, groups=item_sentences)
            probs, choice = static.select_and_predict(features["train"], parts["train"][1], features["validation"],
                                                      parts["validation"][1], features["test"], U)
            static.save_model(os.path.join(args.out_dir, "models", f"sst3_{arm}.npz"), U, mean, std, choice)
            external[arm] = frame(np.arange(len(probs)), parts["test"][1], probs, arm)
            print(f"SST-3 {arm}: macro-F1 {compute_metrics(external[arm]['true_id'], external[arm]['predicted_id'])['macro_f1']:.4f}"
                  f" (beta {choice['beta']})")
        sst_arms, sst_paired = summarize(external)
        summary["sst3"] = {"arms": sst_arms, "paired": sst_paired}

    summary["runtime_s"] = time.time() - started
    save_json(summary, os.path.join(args.out_dir, "summary.json"))
    write_report(os.path.join(args.out_dir, "eeg_shaped_embeddings.md"), summary)
    print(open(os.path.join(args.out_dir, "eeg_shaped_embeddings.md")).read())


def write_report(path, summary):
    def block(name, part):
        lines = [f"### {name}", "", "| arm | macro-F1 | 95% CI | accuracy |", "|---|---:|---|---:|"]
        for arm, values in part["arms"].items():
            ci = values["bootstrap"]["macro_f1"]["ci95"]
            lines.append(f"| {arm} | {values['metrics']['macro_f1']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
                         f"{values['metrics']['accuracy']:.3f} |")
        lines += ["", "| comparison | Δ macro-F1 | 95% CI | p |", "|---|---:|---|---:|"]
        for c in part["paired"]:
            lines.append(f"| {c['candidate']} − {c['baseline']} | {c['delta_macro_f1']:.3f} | "
                         f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] | {c['p_value_two_sided']:.3f} |")
        return lines + [""]

    folds = pd.DataFrame(summary["zuco"]["folds"])
    lines = ["# EEG-shaped word embeddings → sentiment (version A)", "",
             f"Encoder `{summary['encoder']}` (layer {summary['layer']}), {summary['k']} EEG components, "
             f"{summary['folds']} sentence-disjoint fold(s).", "",
             "EEG helps only if `eeg` beats `shuffled_eeg` and `random_targets`.", ""]
    lines += block("ZuCo (unseen sentences)", summary["zuco"])
    lines += ["Diagnostics (mean over folds):", "",
              f"* EEG encoding R² on held-out words — real {folds['eeg_encoding_r2_on_real_test_eeg'].mean():.3f}, "
              f"shuffled {folds['shuffled_eeg_encoding_r2_on_real_test_eeg'].mean():.3f}, "
              f"random {folds['random_targets_encoding_r2_on_real_test_eeg'].mean():.3f} "
              "(positive control: word vectors should predict real EEG).",
              f"* Share of sentiment-classifier weight inside the EEG subspace: "
              f"{folds['eeg_sentiment_weight_share_in_subspace'].mean():.3f} vs {folds['subspace_share_by_chance'].mean():.3f} by chance.",
              f"* Chosen amplification β (0 = EEG directions not useful): "
              f"eeg {folds['eeg_chosen_beta'].tolist()}, shuffled {folds['shuffled_eeg_chosen_beta'].tolist()}.", ""]
    if "sst3" in summary:
        lines += block("SST-3 transfer (no EEG; EEG subspace learned from ZuCo)", summary["sst3"])
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
