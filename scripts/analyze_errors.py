"""Stage 4: where does the text model fail, and does EEG help there?

Reads the held-out predictions of a LoRA run (all arms), groups sentences by
linguistic properties, compares arms within each group with a
sentence-cluster bootstrap, reports per-reader EEG effects, and tests whether
reader-averaged word EEG can predict which sentences the text model gets wrong.
"""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)  # show progress in Colab before any crash

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import LABEL_TO_ID  # noqa: E402
from src.diagnostics import errors, signal  # noqa: E402
from src.diagnostics.representations import average_over_readers, word_eeg_sentence_means  # noqa: E402
from src.diagnostics.targets import valence_lexicon  # noqa: E402
from src.fusion.word_eeg import load_word_eeg  # noqa: E402
from src.labels import load_labels  # noqa: E402
from src.neurolm.config import save_json  # noqa: E402

COMPARISONS = [("text_eeg", "text_shuffled_eeg"), ("text_eeg", "text_only"), ("text_fixation_only", "text_only")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-run-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--word-eeg-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    arms = {}
    for arm_dir in sorted(glob.glob(os.path.join(args.lora_run_dir, "*"))):
        files = sorted(glob.glob(os.path.join(arm_dir, "fold_*.csv")))
        if files:
            arms[os.path.basename(arm_dir)] = errors.add_correctness(pd.concat(map(pd.read_csv, files), ignore_index=True))
    if "text_only" not in arms:
        raise SystemExit("the run has no text_only predictions")
    print("arms:", {k: len(v) for k, v in arms.items()})

    labels = load_labels(args.labels_csv)
    labels["label_id"] = labels["sentiment_label"].map(LABEL_TO_ID)
    tested = set(arms["text_only"]["sentence_id"])
    sentences = labels[labels["sentence_id"].isin(tested)][["sentence_id", "sentence", "label_id"]]
    properties = errors.sentence_properties(sentences, arms["text_only"], valence_lexicon())
    properties.to_csv(os.path.join(args.out_dir, "sentence_properties.csv"), index=False)

    subsets = errors.subset_effects(arms, properties, COMPARISONS, n_boot=args.n_boot)
    subsets.to_csv(os.path.join(args.out_dir, "subset_effects.csv"), index=False)
    print(subsets.round(4).to_string(index=False))
    readers = {f"{a} - {b}": errors.reader_effects(arms, a, b) for a, b in COMPARISONS if a in arms and b in arms}
    pd.concat([frame.assign(comparison=name) for name, frame in readers.items()]).to_csv(
        os.path.join(args.out_dir, "reader_effects.csv"), index=False)

    trials = load_word_eeg(args.word_eeg_dir)
    meta, X, _ = signal.long_word_table(trials)
    Z = signal.zscore_per_reader(meta, X)
    frame, sentence_means = average_over_readers(*word_eeg_sentence_means(meta, Z))
    table = frame.merge(properties[["sentence_id", "text_wrong", "text_confidence"]], on="sentence_id")
    rows = frame.reset_index().merge(table, on="sentence_id")["index"].to_numpy()
    predictability = {}
    if table["text_wrong"].nunique() == 2:
        result = signal.ridge_probe(sentence_means[rows], table["text_wrong"].astype(int).to_numpy(),
                                    groups=table["sentence_id"].to_numpy(), task="binary", n_perm=args.n_perm,
                                    perm_units=table["sentence_id"].to_numpy())
        predictability["text_wrong_from_eeg"] = result
        confidence = signal.ridge_probe(sentence_means[rows], table["text_confidence"].to_numpy(float),
                                        groups=table["sentence_id"].to_numpy(), task="regression",
                                        n_perm=args.n_perm, perm_units=table["sentence_id"].to_numpy())
        predictability["text_confidence_from_eeg"] = confidence
        if result.get("skipped"):
            print("EEG-predicts-errors probe skipped:", result["skipped"])
        else:
            print(f"EEG predicts text-model errors: AUC {result['score']:.3f} (null q95 {result['null_q95']:.3f}, "
                  f"p = {result['p_value']:.3f})")
    save_json({"arms": {k: int(len(v)) for k, v in arms.items()},
               "text_only_sentence_accuracy": float(1 - properties["text_wrong"].mean()),
               "n_text_errors": int(properties["text_wrong"].sum()),
               "eeg_predicts_text_errors": predictability, "runtime_s": time.time() - started},
              os.path.join(args.out_dir, "stage4_summary.json"))
    print(f"done in {time.time() - started:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
