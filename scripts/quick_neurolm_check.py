"""Quick answer: does frozen NeuroLM carry any sentiment information?

One model (logistic regression, fixed C, no tuning), one seed, on the cached
NeuroLM embeddings. Prints:

1. sentiment on new sentences, same readers (easiest honest test), with a
   shuffled-label check;
2. the same predictions averaged over readers per sentence;
3. sentiment on new readers AND new sentences (hardest test);
4. a positive control: can the embeddings tell readers apart at all?
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.neurolm.config import load_config, save_json, set_path_overrides  # noqa: E402
from src.neurolm.evaluation import cluster_bootstrap, predictions_frame, run_probe, sentence_aggregate  # noqa: E402
from src.neurolm.experiment import build_probe_data, feature_matrix  # noqa: E402
from src.neurolm.sanity import permute_labels  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/neurolm_probe.yaml")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--feature", default="tokenizer__mean")
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--out", default=None, help="optional JSON path for the numbers")
    args = parser.parse_args()

    config = set_path_overrides(load_config(args.config), cache_dir=args.cache_dir)
    config["paths"]["handcrafted_dir"] = None
    data = build_probe_data(config)
    samples = data["samples"]
    X = feature_matrix(data["neurolm"], args.feature)
    one_model = dict(classifier="logreg", seeds=[42], grid=[{"C": args.C}])
    started = time.time()
    print(f"NeuroLM {args.feature} ({X.shape[1]}-d): {len(samples)} trials, "
          f"{samples['subject_id'].nunique()} readers, {samples['sentence_id'].nunique()} sentences")
    print("chance macro-F1 ≈ 0.33 (3 balanced classes)\n")

    text = run_probe(X, samples, protocol="text", **one_model)
    text_f1 = text["summary"]["macro_f1"]["mean"]
    ci = cluster_bootstrap(predictions_frame(text), n_boot=1000)["macro_f1"]["ci95"]
    rng = np.random.default_rng(0)
    null = []
    for _ in range(args.shuffles):
        shuffled = permute_labels(samples, rng, "sentence", column="label_shuffled")
        result = run_probe(X, shuffled, protocol="text", target="label_shuffled", keep_predictions=False, **one_model)
        null.append(result["summary"]["macro_f1"]["mean"])
    p_value = (1 + sum(v >= text_f1 for v in null)) / (1 + len(null))
    averaged = sentence_aggregate(predictions_frame(text))["macro_f1"]["mean"]
    print(f"1. new sentences, same readers:   macro-F1 {text_f1:.3f}  (95% CI {ci[0]:.3f}–{ci[1]:.3f})")
    print(f"   shuffled labels x{args.shuffles}:          mean {np.mean(null):.3f}, max {np.max(null):.3f}  -> p = {p_value:.3f}")
    print(f"2. averaged over readers:         macro-F1 {averaged:.3f}  (per sentence)")

    joint = run_probe(X, samples, protocol="joint", **one_model)
    joint_f1 = joint["summary"]["macro_f1"]["mean"]
    print(f"3. new readers + new sentences:   macro-F1 {joint_f1:.3f}")

    readers = sorted(samples["subject_id"].unique())
    coded = samples.assign(reader=samples["subject_id"].map({s: i for i, s in enumerate(readers)}))
    reader = run_probe(X, coded, protocol="text", target="reader", class_names=readers,
                       keep_predictions=False, **one_model)
    reader_acc = reader["summary"]["accuracy"]["mean"]
    print(f"4. control - which reader is it?  accuracy {reader_acc:.3f}  (chance {1 / len(readers):.3f})")
    print(f"\n({time.time() - started:.0f}s)")

    if args.out:
        save_json({
            "feature": args.feature, "C": args.C, "n_trials": len(samples),
            "new_sentences_macro_f1": text_f1, "new_sentences_ci95": ci,
            "shuffled_null": null, "p_value": p_value,
            "averaged_over_readers_macro_f1": averaged,
            "new_readers_and_sentences_macro_f1": joint_f1,
            "reader_id_accuracy": reader_acc, "reader_chance": 1 / len(readers),
        }, args.out)


if __name__ == "__main__":
    main()
