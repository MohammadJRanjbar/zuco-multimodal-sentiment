"""One table comparing every word representation (decoding_<tag>.json + word_info_<tag>.json in --results-dir)."""

import argparse
import glob
import json
import os


def triple(value, digits=3):
    return f"{value[0]:.{digits}f} [{value[1]:.{digits}f}, {value[2]:.{digits}f}]" if value else "—"


def point(value, digits=3):
    return f"{value[0]:.{digits}f}" if value else "—"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--order", nargs="*", default=None, help="representation tags in table order")
    args = parser.parse_args()
    tags = sorted({os.path.basename(p)[len("word_info_"):-5]
                   for p in glob.glob(os.path.join(args.results_dir, "word_info_*.json"))}
                  | {os.path.basename(p)[len("decoding_"):-5]
                     for p in glob.glob(os.path.join(args.results_dir, "decoding_*.json"))})
    if args.order:
        tags = [t for t in args.order if t in tags] + [t for t in tags if t not in args.order]
    rows = []
    for tag in tags:
        dec_path = os.path.join(args.results_dir, f"decoding_{tag}.json")
        info_path = os.path.join(args.results_dir, f"word_info_{tag}.json")
        dec = json.load(open(dec_path)) if os.path.exists(dec_path) else None
        info = json.load(open(info_path)) if os.path.exists(info_path) else None
        row = {"tag": tag}
        if info:
            row["features"] = info["n_features"]
            row["cos"] = (f"{point(info['embedding']['eeg']['centred_cosine'])} / "
                          f"{point(info['embedding']['shuffled_eeg']['centred_cosine'])}")
            sid = info["sentence_identification"]
            row["sentence_id"] = (f"{sid['eeg']['accuracy']:.3f} / {sid['shuffled_eeg']['accuracy']:.3f} / "
                                  f"{sid['word_features']['accuracy']:.3f} (chance {sid['eeg']['chance']:.3f})")
        if dec:
            tv = dec["two_vs_two"]
            row["pairs"] = f"{point(tv['pairs']['eeg'])} / {point(tv['pairs']['shuffled_eeg'])}"
            row["matched"] = f"{point(tv['matched']['eeg'])} / {point(tv['matched']['noise'])}"
            row["beyond"] = triple(tv["pairs"].get("word_features+eeg - word_features"))
            rt = dec["retrieval"]
            row["mrr"] = f"{point(rt['eeg']['mrr'], 4)} / {point(rt['shuffled_eeg']['mrr'], 4)}"
            row["sentiment"] = point(dec["sentiment"]["eeg"])
        rows.append(row)
    lines = ["# Can EEG tell which word was read? All representations", "",
             "Reader-averaged words of unseen sentences (5 folds by sentence). Each cell: representation / control. "
             "2-vs-2 chance 0.5; matched pairs share length, frequency and position (control: noise); "
             "'beyond word features' = 2-vs-2 gain when the representation is added to length/frequency/position "
             "(95% CI); sentence identification among held-out sentences of the same length "
             "(EEG / shuffled / word features only).", "",
             "| representation | features | 2-vs-2 (vs shuffled) | matched 2-vs-2 (vs noise) | beyond word features | "
             "retrieval MRR (vs shuffled) | embedding centred cosine (vs shuffled) | sentence identification | "
             "decoded-text sentiment F1 |", "|---|---:|---|---|---|---|---|---|---:|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "—")) for c in ("tag", "features", "pairs", "matched", "beyond",
                                                                      "mrr", "cos", "sentence_id", "sentiment")) + " |")
    training = sorted(glob.glob(os.path.join(args.results_dir, "*_training.md")))
    for path in training:
        lines += ["", open(path).read().strip()]
    lines += ["", "**Reading.** A representation tells something about the word if it beats its shuffled control; "
              "it tells something about the word's identity (not only its form) if it also beats noise on matched "
              "pairs and adds to word features. `eye_tracking` shows what reading times alone give."]
    out = os.path.join(args.results_dir, "word_information_summary.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
